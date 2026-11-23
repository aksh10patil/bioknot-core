"""BioKnot-Core inference server.

Two engines behind one Flask app:
  POST /predict         ESMFold structure bridge -> C-alpha backbone coordinates
  POST /predict_custom  Fine-tuned ESM-2 sequence classifier -> knot topology
"""

import json
import os
import shutil
import subprocess
import time
from collections import defaultdict, deque
from threading import Lock

import numpy as np
import requests
import torch
from flask import Flask, jsonify, request, send_from_directory

from app.model import (
    CLASS_MAP,
    MAX_RESIDUES,
    MAX_TOKENS,
    load_classifier,
    load_tokenizer,
    sanitise_sequence,
)

APP_DIR = os.path.dirname(os.path.abspath(__file__))


AA1_TO_AA3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "E": "GLU", "Q": "GLN", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}


def _int_env(name, default):
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


# --- Configuration -----------------------------------------------------------
# Hard ceiling on what we will accept. /predict fans out to a third-party API,
# so an uncapped sequence turns this server into a slow proxy hammering ESMAtlas.
MAX_SEQUENCE_LENGTH = _int_env("MAX_SEQUENCE_LENGTH", 2000)
ESMFOLD_TIMEOUT = _int_env("ESMFOLD_TIMEOUT", 60)
ESMFOLD_RETRIES = _int_env("ESMFOLD_RETRIES", 2)
# Whole-request wall-clock budget for the tiling loop. Must stay comfortably
# under the gunicorn --timeout so we return a partial result instead of being
# killed mid-flight.
FOLD_DEADLINE_SECONDS = _int_env("FOLD_DEADLINE_SECONDS", 240)
RATE_LIMIT_REQUESTS = _int_env("RATE_LIMIT_REQUESTS", 10)
RATE_LIMIT_WINDOW = _int_env("RATE_LIMIT_WINDOW", 60)

ESMFOLD_URL = "https://api.esmatlas.com/foldSequence/v1/pdb/"
WINDOW_SIZE = 380
STRIDE = 200

app = Flask(__name__, static_folder=None)
# Reject oversized bodies before they reach a handler.
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024

# CORS is off by default: index.html is served from this same origin and needs
# none. Set ALLOWED_ORIGINS to a comma-separated list to opt in (e.g. for the
# separate Next.js dashboard in web_ui/).
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]
if ALLOWED_ORIGINS:
    from flask_cors import CORS

    CORS(app, resources={r"/predict*": {"origins": ALLOWED_ORIGINS}})

# --- Model bootstrap ---------------------------------------------------------
print("[Backend Setup] Initialising ESM-2 tokenizer and fine-tuned classifier...")
tokenizer = load_tokenizer()
native_model, WEIGHTS_LOADED = load_classifier()

# Without a checkpoint the classification head is random, so every prediction is
# noise -- and with multiple gunicorn workers, noise that differs per worker.
# Refuse to serve it unless explicitly allowed for local development.
ALLOW_UNTRAINED = os.environ.get("BIOKNOT_ALLOW_UNTRAINED", "").lower() in ("1", "true", "yes")

if WEIGHTS_LOADED:
    print("[Backend Setup] Fine-tuned weights loaded. Classifier is live.")
elif ALLOW_UNTRAINED:
    print(
        "[Backend Setup] WARNING: no checkpoint found and BIOKNOT_ALLOW_UNTRAINED is set. "
        "/predict_custom will return UNTRAINED, MEANINGLESS predictions."
    )
else:
    print(
        "[Backend Setup] WARNING: no checkpoint found. /predict_custom will return HTTP 503. "
        "Run `python -m training.train` to produce bioknot_native.pt."
    )

# --- Rate limiting -----------------------------------------------------------
# In-process and therefore per-worker, so the effective limit is
# RATE_LIMIT_REQUESTS * workers. Enough to stop casual abuse of the ESMFold
# proxy; put a real gateway in front if this ever needs to be authoritative.
_hits = defaultdict(deque)
_hits_lock = Lock()


def rate_limited(key):
    now = time.time()
    with _hits_lock:
        bucket = _hits[key]
        while bucket and now - bucket[0] > RATE_LIMIT_WINDOW:
            bucket.popleft()
        if len(bucket) >= RATE_LIMIT_REQUESTS:
            return True
        bucket.append(now)
        return False


def client_key():
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.remote_addr or "unknown"


def read_sequence():
    """Pull, validate and length-check the sequence from the request body."""
    data = request.get_json(silent=True) or {}
    sequence, error = sanitise_sequence(data.get("sequence", ""))
    if error:
        return None, (jsonify({"error": error}), 400)

    if len(sequence) > MAX_SEQUENCE_LENGTH:
        return None, (
            jsonify(
                {
                    "error": (
                        f"Sequence of {len(sequence)} AA exceeds the {MAX_SEQUENCE_LENGTH} AA limit "
                        "for this deployment."
                    ),
                    "max_sequence_length": MAX_SEQUENCE_LENGTH,
                }
            ),
            413,
        )

    return sequence, None


# --- ESMFold bridge ----------------------------------------------------------
def fetch_pdb_for_chunk(seq_chunk, deadline=None):
    for attempt in range(ESMFOLD_RETRIES):
        if deadline and time.time() >= deadline:
            return None
        try:
            response = requests.post(
                ESMFOLD_URL,
                data=seq_chunk,
                headers={"Content-Type": "text/plain"},
                timeout=ESMFOLD_TIMEOUT,
            )
            if response.status_code == 200:
                return response.text
            print(f"[ESMFold Bridge] Upstream returned HTTP {response.status_code}")
        except requests.RequestException as e:
            print(f"[ESMFold Bridge] Attempt {attempt + 1}/{ESMFOLD_RETRIES} failed: {e}")
        if attempt + 1 < ESMFOLD_RETRIES:
            time.sleep(2)
    return None


def extract_ca_coords(pdb_str):
    coords = []
    plddts = []
    for line in pdb_str.splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            try:
                coords.append(
                    [
                        float(line[30:38].strip()),
                        float(line[38:46].strip()),
                        float(line[46:54].strip()),
                    ]
                )
                plddts.append(float(line[60:66].strip()))
            except ValueError:
                continue
    return np.array(coords), np.array(plddts)


def kabsch_rotation(target_overlap, ref_overlap):
    """Optimal rigid rotation mapping `target_overlap` onto `ref_overlap`."""
    centroid_target = np.mean(target_overlap, axis=0)
    centroid_ref = np.mean(ref_overlap, axis=0)

    H = np.dot((target_overlap - centroid_target).T, ref_overlap - centroid_ref)
    U, _, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)

    return R, centroid_target, centroid_ref


def tile_long_sequence(sequence, deadline):
    """Fold a >400 AA sequence as overlapping windows stitched by rigid Kabsch fits.

    Returns `(coords, plddts, truncated, seam_rmsds)`. Splices at the midpoint of
    the overlap (`mid = n // 2`), keeping `full_chain` index-aligned with residue numbering.
    """
    seq_len = len(sequence)
    overlap = WINDOW_SIZE - STRIDE
    seam_rmsds = []

    pdb1 = fetch_pdb_for_chunk(sequence[0:WINDOW_SIZE], deadline)
    if pdb1 is None:
        return None, None, False, seam_rmsds

    full_chain, full_plddts = extract_ca_coords(pdb1)

    for start in range(STRIDE, seq_len, STRIDE):
        if time.time() >= deadline:
            return full_chain, full_plddts, True, seam_rmsds

        end = min(start + WINDOW_SIZE, seq_len)
        if end - start < 5:
            break

        if start >= len(full_chain):
            return full_chain, full_plddts, True, seam_rmsds

        pdb_next = fetch_pdb_for_chunk(sequence[start:end], deadline)
        if pdb_next is None:
            return full_chain, full_plddts, True, seam_rmsds

        w_coords, w_plddts = extract_ca_coords(pdb_next)
        curr_overlap_len = min(overlap, len(full_chain) - start, len(w_coords))
        if curr_overlap_len < 5:
            return full_chain, full_plddts, True, seam_rmsds

        ref_overlap = full_chain[start : start + curr_overlap_len]
        target_overlap = w_coords[:curr_overlap_len]

        R, centroid_target, centroid_ref = kabsch_rotation(target_overlap, ref_overlap)
        aligned = np.dot(w_coords - centroid_target, R.T) + centroid_ref

        seam_rmsd = float(np.sqrt(np.mean(np.sum((aligned[:curr_overlap_len] - ref_overlap) ** 2, axis=1))))
        seam_rmsds.append(seam_rmsd)

        # Splice at the midpoint of the overlap
        mid = curr_overlap_len // 2
        full_chain = np.vstack([full_chain[:start + mid], aligned[mid:]])
        full_plddts = np.concatenate([full_plddts[:start + mid], w_plddts[mid:]])

        if end == seq_len:
            break

    return full_chain, full_plddts, False, seam_rmsds


def get_cli_path():
    env_cli = os.environ.get("BIOKNOT_CLI_PATH")
    if env_cli and os.path.isfile(env_cli) and os.access(env_cli, os.X_OK):
        return env_cli
    which_cli = shutil.which("bioknot_cli")
    if which_cli:
        return which_cli
    repo_root = os.path.abspath(os.path.join(APP_DIR, ".."))
    local_release = os.path.join(repo_root, "rust", "target", "release", "bioknot_cli")
    if os.path.isfile(local_release) and os.access(local_release, os.X_OK):
        return local_release
    usr_bin = "/usr/local/bin/bioknot_cli"
    if os.path.isfile(usr_bin) and os.access(usr_bin, os.X_OK):
        return usr_bin
    return None


def verify_topology_3d(ca_coords):
    """Pipe C-alpha coordinates JSON to the Rust bioknot_cli invariant verifier."""
    cli_path = get_cli_path()
    if not cli_path:
        return {
            "error": "Rust bioknot_cli verifier binary not found.",
            "knot_type": "Unknown",
            "knot_probability": 0.0,
            "raw_node_count": len(ca_coords),
            "reduced_node_count": len(ca_coords),
            "chain_breaks": 0,
            "is_knotted": False,
        }
    try:
        input_data = json.dumps([{"x": c["x"], "y": c["y"], "z": c["z"]} for c in ca_coords])
        proc = subprocess.run(
            [cli_path],
            input=input_data,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if proc.returncode == 0:
            return json.loads(proc.stdout)
        print(f"[Rust Verifier] CLI exited with code {proc.returncode}: {proc.stderr}")
        return {
            "error": f"CLI error: {proc.stderr.strip()}",
            "knot_type": "Unknown",
            "knot_probability": 0.0,
            "raw_node_count": len(ca_coords),
            "reduced_node_count": len(ca_coords),
            "chain_breaks": 0,
            "is_knotted": False,
        }
    except Exception as e:
        print(f"[Rust Verifier] Execution failed: {e}")
        return {
            "error": str(e),
            "knot_type": "Unknown",
            "knot_probability": 0.0,
            "raw_node_count": len(ca_coords),
            "reduced_node_count": len(ca_coords),
            "chain_breaks": 0,
            "is_knotted": False,
        }


# --- Routes ------------------------------------------------------------------
@app.route("/")
def serve_index():
    return send_from_directory(APP_DIR, "index.html")


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "weights_loaded": WEIGHTS_LOADED,
            "classifier_available": WEIGHTS_LOADED or ALLOW_UNTRAINED,
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
        }
    )


@app.route("/predict", methods=["POST"])
def predict_structure():
    if rate_limited(client_key()):
        return jsonify({"error": "Rate limit exceeded. Try again shortly."}), 429

    sequence, error = read_sequence()
    if error:
        return error

    seq_len = len(sequence)
    deadline = time.time() + FOLD_DEADLINE_SECONDS
    print(f"\n[ESMFold Bridge] Request for sequence length: {seq_len} AA")

    truncated = False
    seam_rmsds = []
    if seq_len <= 400:
        pdb_str = fetch_pdb_for_chunk(sequence, deadline)
        if pdb_str is None:
            return jsonify({"error": "ESMFold API timed out or rate-limited. Retry request."}), 502
        coords_arr, plddts_arr = extract_ca_coords(pdb_str)
    else:
        print("[ESMFold Bridge] Sequence > 400 AA. Executing rigid Kabsch domain tiling...")
        coords_arr, plddts_arr, truncated, seam_rmsds = tile_long_sequence(sequence, deadline)
        if coords_arr is None:
            return jsonify({"error": "ESMFold API failed on the initial window."}), 502

    if coords_arr is None or len(coords_arr) == 0:
        return jsonify({"error": "No C-alpha atoms could be extracted from the folded structure."}), 502

    ca_coords = [
        {
            "x": float(p[0]),
            "y": float(p[1]),
            "z": float(p[2]),
            "res_name": AA1_TO_AA3.get(sequence[i].upper(), "UNK") if i < len(sequence) else "UNK",
            "plddt": float(plddts_arr[i]) if (plddts_arr is not None and i < len(plddts_arr)) else 0.0,
            "res_num": i + 1,
        }
        for i, p in enumerate(coords_arr)
    ]
    print(f"[ESMFold Bridge] Extracted {len(ca_coords)} continuous C-alpha nodes.")

    max_seam_rmsd = float(np.max(seam_rmsds)) if seam_rmsds else 0.0

    print("[Rust Verifier] Verifying 3D backbone topology with Alexander polynomial invariant...")
    topology = verify_topology_3d(ca_coords)

    return jsonify(
        {
            "engine": "ESMFold + Rigid Kabsch Tiling",
            "sequence_length": seq_len,
            "residues_resolved": len(ca_coords),
            "truncated": truncated,
            "max_seam_rmsd": max_seam_rmsd,
            "seam_rmsds": seam_rmsds,
            "topology": topology,
            "ca_coords": ca_coords,
        }
    )


@app.route("/predict_custom", methods=["POST"])
def predict_custom():
    if not WEIGHTS_LOADED and not ALLOW_UNTRAINED:
        return (
            jsonify(
                {
                    "error": (
                        "Classifier weights are not available, so no meaningful prediction can be "
                        "made. Build bioknot_native.pt via `python -m training.train`."
                    ),
                    "weights_loaded": False,
                }
            ),
            503,
        )

    sequence, error = read_sequence()
    if error:
        return error

    original_len = len(sequence)
    evaluated = sequence[:MAX_RESIDUES]
    if original_len > MAX_RESIDUES:
        print(f"[Custom AI Engine] Truncating {original_len} AA to {MAX_RESIDUES} AA model capacity.")

    print(f"\n[Custom AI Engine] Running direct sequence inference ({len(evaluated)} AA)...")

    inputs = tokenizer(evaluated, return_tensors="pt", truncation=True, max_length=MAX_TOKENS)
    with torch.no_grad():
        logits = native_model(inputs["input_ids"], inputs["attention_mask"])
        probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()
        pred_class = int(np.argmax(probs))

    payload = {
        "engine": "BioKnot Fine-Tuned Native AI",
        "sequence_length": original_len,
        "evaluated_length": len(evaluated),
        "predicted_knot": CLASS_MAP.get(pred_class, "Unknown"),
        "confidence": float(probs[pred_class]),
        "probabilities": {CLASS_MAP[i]: float(probs[i]) for i in range(len(CLASS_MAP))},
        "weights_loaded": WEIGHTS_LOADED,
    }

    if not WEIGHTS_LOADED:
        payload["warning"] = (
            "UNTRAINED MODEL: the classification head is randomly initialised. "
            "This output is noise and must not be interpreted."
        )

    return jsonify(payload)


if __name__ == "__main__":
    port = _int_env("PORT", 7860)
    print("====================================================")
    print("  BIOKNOT-CORE: HYBRID DUAL-ENGINE INFERENCE SUITE  ")
    print("====================================================")
    print("  1. ESMFold WASM Bridge : POST /predict            ")
    print("  2. Native AI Predictor : POST /predict_custom     ")
    print(f"  Open in browser: http://127.0.0.1:{port}")
    print("====================================================")
    app.run(host="0.0.0.0", port=port, debug=False)
