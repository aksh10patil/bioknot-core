"""BioKnot-Core — Gradio Space.

Two engines:
  * a fine-tuned ESM-2 head classifying backbone topology from sequence alone;
  * an ESMFold bridge that folds a sequence (tiling past the 400 AA API ceiling
    with rigid Kabsch stitching) and returns a viewable structure.

Both run on CPU: ESM-2 8M is small and the fold is an upstream API call, so the
`@spaces.GPU` function below exists only because ZeroGPU requires one. Nothing on
the user path enters it, so no GPU quota is consumed.
"""

try:
    import spaces  # noqa: F401  -- must precede torch; patches torch.cuda.*
except ImportError:
    class _MockSpaces:
        @staticmethod
        def GPU(duration=5):
            def decorator(fn):
                return fn
            return decorator
    spaces = _MockSpaces()

import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import gradio as gr
import numpy as np
import requests
import torch

try:
    from bioknot_model import (
        CLASS_MAP,
        DEFAULT_WEIGHTS_PATH,
        MAX_RESIDUES,
        MAX_TOKENS,
        load_classifier,
        load_tokenizer,
        sanitise_sequence,
    )
except ImportError:
    from space.bioknot_model import (
        CLASS_MAP,
        DEFAULT_WEIGHTS_PATH,
        MAX_RESIDUES,
        MAX_TOKENS,
        load_classifier,
        load_tokenizer,
        sanitise_sequence,
    )

MAX_SEQUENCE_LENGTH = 2000
ESMFOLD_URL = "https://api.esmatlas.com/foldSequence/v1/pdb/"
ESMFOLD_TIMEOUT = 60
ESMFOLD_RETRIES = 2
FOLD_DEADLINE_SECONDS = 240
WINDOW_SIZE = 380
STRIDE = 200

AA1_TO_AA3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "E": "GLU", "Q": "GLN", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}

# --- Model bootstrap ---------------------------------------------------------
print("[boot] loading tokenizer + classifier...", flush=True)
tokenizer = load_tokenizer()

candidate_weights = [
    DEFAULT_WEIGHTS_PATH,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "bioknot_native.pt"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bioknot_native.pt"),
    "/app/bioknot_native.pt",
]

weights_to_load = None
for w_path in candidate_weights:
    if os.path.exists(w_path):
        weights_to_load = w_path
        break

if weights_to_load:
    native_model, WEIGHTS_LOADED = load_classifier(weights_to_load)
    print(f"[boot] classifier ready from {weights_to_load} (weights_loaded={WEIGHTS_LOADED})", flush=True)
else:
    native_model, WEIGHTS_LOADED = load_classifier(None)
    print("[boot] [WARN] No pre-built bioknot_native.pt found. Please sync champion weights.", flush=True)


@spaces.GPU(duration=5)
def gpu_runtime_check() -> str:
    """Report whether a ZeroGPU device is reachable from this Space.

    Diagnostic only. The classifier and the fold bridge both run on CPU, so this
    is the sole GPU entry point and is never called on the user path.
    """
    if torch.cuda.is_available():
        return f"ZeroGPU reachable: {torch.cuda.get_device_name(0)}"
    return "No CUDA device visible (expected — this Space runs on CPU)."


# --- ESMFold bridge ----------------------------------------------------------
def fetch_pdb(seq_chunk, deadline=None):
    for attempt in range(ESMFOLD_RETRIES):
        if deadline and time.time() >= deadline:
            return None
        try:
            r = requests.post(
                ESMFOLD_URL,
                data=seq_chunk,
                headers={"Content-Type": "text/plain"},
                timeout=ESMFOLD_TIMEOUT,
            )
            if r.status_code == 200:
                return r.text
            print(f"[fold] upstream HTTP {r.status_code}", flush=True)
        except requests.RequestException as e:
            print(f"[fold] attempt {attempt + 1}/{ESMFOLD_RETRIES} failed: {e}", flush=True)
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
                    [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                )
                plddts.append(float(line[60:66].strip()))
            except ValueError:
                continue
    return np.array(coords), np.array(plddts)


def kabsch_rotation(target_overlap, ref_overlap):
    ct = np.mean(target_overlap, axis=0)
    cr = np.mean(ref_overlap, axis=0)
    H = np.dot((target_overlap - ct).T, ref_overlap - cr)
    U, _, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)
    return R, ct, cr


def tile_long_sequence(sequence, deadline):
    """Stitch overlapping windows, keeping index i aligned to residue i.

    Splices at midpoint of overlap (mid = n // 2) and records seam RMSD.
    """
    seq_len = len(sequence)
    overlap = WINDOW_SIZE - STRIDE
    seam_rmsds = []

    first = fetch_pdb(sequence[0:WINDOW_SIZE], deadline)
    if first is None:
        return None, None, False, seam_rmsds
    full_chain, full_plddts = extract_ca_coords(first)

    for start in range(STRIDE, seq_len, STRIDE):
        if time.time() >= deadline:
            return full_chain, full_plddts, True, seam_rmsds
        end = min(start + WINDOW_SIZE, seq_len)
        if end - start < 5 or start >= len(full_chain):
            return full_chain, full_plddts, True, seam_rmsds

        nxt = fetch_pdb(sequence[start:end], deadline)
        if nxt is None:
            return full_chain, full_plddts, True, seam_rmsds

        w, w_plddts = extract_ca_coords(nxt)
        n = min(overlap, len(full_chain) - start, len(w))
        if n < 5:
            return full_chain, full_plddts, True, seam_rmsds

        ref_overlap = full_chain[start : start + n]
        R, ct, cr = kabsch_rotation(w[:n], ref_overlap)
        aligned = np.dot(w - ct, R.T) + cr

        seam_rmsd = float(np.sqrt(np.mean(np.sum((aligned[:n] - ref_overlap) ** 2, axis=1))))
        seam_rmsds.append(seam_rmsd)

        mid = n // 2
        full_chain = np.vstack([full_chain[: start + mid], aligned[mid:]])
        full_plddts = np.concatenate([full_plddts[: start + mid], w_plddts[mid:]])

        if end == seq_len:
            break

    return full_chain, full_plddts, False, seam_rmsds


def ca_to_pdb(coords, plddts=None, sequence=None):
    """Write a C-alpha-only backbone as a minimal PDB with real 3-letter AA and pLDDT B-factors."""
    lines = []
    for i, p in enumerate(coords, start=1):
        res_name = "UNK"
        if sequence and i - 1 < len(sequence):
            res_name = AA1_TO_AA3.get(sequence[i - 1].upper(), "UNK")
        b_factor = float(plddts[i - 1]) if (plddts is not None and i - 1 < len(plddts)) else 0.00
        lines.append(
            f"ATOM  {i:>5}  CA  {res_name:>3} A{i:>4}    "
            f"{p[0]:>8.3f}{p[1]:>8.3f}{p[2]:>8.3f}  1.00{b_factor:>6.2f}           C"
        )
    lines.append("END")
    return "\n".join(lines)


def get_cli_path():
    env_cli = os.environ.get("BIOKNOT_CLI_PATH")
    if env_cli and os.path.isfile(env_cli) and os.access(env_cli, os.X_OK):
        return env_cli
    which_cli = shutil.which("bioknot_cli")
    if which_cli:
        return which_cli
    space_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(space_dir, "bioknot_cli"),
        os.path.join(space_dir, "..", "rust", "target", "release", "bioknot_cli"),
        os.path.join(space_dir, "..", "target", "release", "bioknot_cli"),
        "/usr/local/bin/bioknot_cli",
    ]
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def verify_topology_3d(coords):
    """Run Rust topological verification on 3D coordinates via bioknot_cli."""
    cli = get_cli_path()
    if not cli:
        return None
    try:
        data = json.dumps([{"x": float(p[0]), "y": float(p[1]), "z": float(p[2])} for p in coords])
        res = subprocess.run([cli], input=data, capture_output=True, text=True, timeout=15)
        if res.returncode == 0:
            return json.loads(res.stdout)
    except Exception as e:
        print(f"[verifier] 3D topology check error: {e}", flush=True)
    return None


def _write_pdb(text):
    fd, path = tempfile.mkstemp(suffix=".pdb")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    return path


def render_pdb_viewer(pdb_str: str) -> str:
    escaped_pdb = (
        pdb_str.replace("\\", "\\\\")
        .replace("`", "\\`")
        .replace("${", "\\${")
    )
    html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.4.2/3Dmol-min.js"></script>
<style>
  html, body {{ margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden; background-color: #0d1117; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }}
  #viewer {{ width: 100%; height: 100%; position: relative; }}
  #controls {{ position: absolute; top: 10px; right: 10px; z-index: 10; display: flex; gap: 6px; }}
  button {{ background: rgba(33, 38, 45, 0.85); color: #c9d1d9; border: 1px solid #30363d; border-radius: 6px; padding: 5px 12px; font-size: 12px; cursor: pointer; backdrop-filter: blur(4px); font-weight: 500; transition: all 0.2s ease; }}
  button:hover {{ background: #30363d; color: #58a6ff; border-color: #58a6ff; }}
</style>
</head>
<body>
<div id="viewer"></div>
<div id="controls">
  <button onclick="setStyle('cartoon')">Cartoon</button>
  <button onclick="setStyle('stick')">Sticks</button>
  <button onclick="setStyle('sphere')">Spheres</button>
  <button onclick="viewer.zoomTo()">Reset View</button>
</div>
<script>
let viewer = null;
window.onload = function() {{
  const container = document.getElementById('viewer');
  viewer = $3Dmol.createViewer(container, {{ backgroundColor: '#0d1117' }});
  const pdbData = `{escaped_pdb}`;
  viewer.addModel(pdbData, "pdb");
  setStyle('cartoon');
  viewer.zoomTo();
  viewer.render();
}};

function setStyle(style) {{
  if (!viewer) return;
  if (style === 'cartoon') {{
    viewer.setStyle({{}}, {{ cartoon: {{ color: 'spectrum' }}, stick: {{ radius: 0.15, colorscheme: 'spectrum' }} }});
  }} else if (style === 'stick') {{
    viewer.setStyle({{}}, {{ stick: {{ colorscheme: 'spectrum' }} }});
  }} else if (style === 'sphere') {{
    viewer.setStyle({{}}, {{ sphere: {{ radius: 0.8, colorscheme: 'spectrum' }} }});
  }}
  viewer.render();
}}
</script>
</body>
</html>"""
    b64 = base64.b64encode(html_content.encode("utf-8")).decode("utf-8")
    return f'<iframe src="data:text/html;base64,{b64}" style="width: 100%; height: 520px; border: 1px solid #30363d; border-radius: 8px; background: #0d1117;"></iframe>'


# --- Gradio handlers ---------------------------------------------------------
def classify(sequence: str):
    """Predict backbone knot topology from an amino-acid sequence.

    Args:
        sequence: Amino-acid sequence, raw or FASTA.

    Returns:
        Class probabilities and a summary of what was evaluated.
    """
    seq, err = sanitise_sequence(sequence or "")
    if err:
        raise gr.Error(err)
    if not WEIGHTS_LOADED:
        raise gr.Error("Classifier weights not loaded. Please provide or sync bioknot_native.pt.")
    if len(seq) > MAX_SEQUENCE_LENGTH:
        raise gr.Error(f"{len(seq)} AA exceeds the {MAX_SEQUENCE_LENGTH} AA limit.")

    evaluated = seq[:MAX_RESIDUES]
    inputs = tokenizer(evaluated, return_tensors="pt", truncation=True, max_length=MAX_TOKENS)
    with torch.no_grad():
        logits = native_model(inputs["input_ids"], inputs["attention_mask"])
        probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()

    note = f"Evaluated {len(evaluated)} of {len(seq)} AA."
    if len(seq) > MAX_RESIDUES:
        note += f" Truncated to the {MAX_RESIDUES} AA model capacity."

    return {CLASS_MAP[i]: float(probs[i]) for i in range(len(CLASS_MAP))}, note


def fold(sequence: str):
    """Fold a sequence with ESMFold and return a viewable 3D structure.

    Sequences past 400 AA are folded as overlapping windows stitched by rigid
    Kabsch alignment.

    Args:
        sequence: Amino-acid sequence, raw or FASTA.

    Returns:
        A PDB structure and a summary of the fold.
    """
    seq, err = sanitise_sequence(sequence or "")
    if err:
        raise gr.Error(err)
    if not WEIGHTS_LOADED:
        raise gr.Error("Classifier weights not loaded. Please provide or sync bioknot_native.pt.")
    if len(seq) > MAX_SEQUENCE_LENGTH:
        raise gr.Error(f"{len(seq)} AA exceeds the {MAX_SEQUENCE_LENGTH} AA limit.")

    deadline = time.time() + FOLD_DEADLINE_SECONDS

    if len(seq) <= 400:
        pdb = fetch_pdb(seq, deadline)
        if pdb is None:
            raise gr.Error("ESMFold API timed out or rate-limited. Try again shortly.")
        coords, plddts = extract_ca_coords(pdb)
        topo = verify_topology_3d(coords)
        if topo:
            topo_str = f" | Rust Topological Verifier: **{topo.get('knot_type')}** (p={topo.get('knot_probability', 1.0)*100:.1f}%)"
        else:
            topo_str = " | Rust Topological Verifier: *Standby (CLI not detected)*"
        return render_pdb_viewer(pdb), _write_pdb(pdb), f"Folded {len(seq)} AA directly — {len(coords)} residues, all-atom.{topo_str}"

    coords, plddts, truncated, seam_rmsds = tile_long_sequence(seq, deadline)
    if coords is None or len(coords) == 0:
        raise gr.Error("ESMFold API failed on the initial window.")

    max_seam = max(seam_rmsds) if seam_rmsds else 0.0
    msg = (
        f"Folded {len(seq)} AA via Kabsch tiling "
        f"({WINDOW_SIZE} AA windows, {STRIDE} AA stride) — {len(coords)} C-alpha nodes. "
        f"Max seam RMSD: {max_seam:.2f} Å."
    )
    if max_seam > 3.0:
        msg += " ⚠️ High seam RMSD detected (> 3.0 Å); domain transition may have structural strain."
    if truncated:
        msg += " PARTIAL: the fold hit its time budget or an upstream window failed, so the chain stops early."

    topo = verify_topology_3d(coords)
    if topo:
        msg += f" | Rust Topological Verifier: **{topo.get('knot_type')}** (p={topo.get('knot_probability', 1.0)*100:.1f}%)"
    else:
        msg += " | Rust Topological Verifier: *Standby (CLI not detected)*"

    pdb_text = ca_to_pdb(coords, plddts, seq)
    return render_pdb_viewer(pdb_text), _write_pdb(pdb_text), msg


def scan_domains(sequence: str, window: int = 400, stride: int = 200):
    """Slide a window along a long sequence and classify each segment.

    Args:
        sequence: Amino-acid sequence, raw or FASTA.
        window: Window length in residues.
        stride: Step between window starts.

    Returns:
        One row per window: span, predicted topology, and confidence.
    """
    seq, err = sanitise_sequence(sequence or "")
    if err:
        raise gr.Error(err)
    if not WEIGHTS_LOADED:
        raise gr.Error("Classifier weights not loaded. Please provide or sync bioknot_native.pt.")
    if len(seq) > MAX_SEQUENCE_LENGTH:
        raise gr.Error(f"{len(seq)} AA exceeds the {MAX_SEQUENCE_LENGTH} AA limit.")

    rows = []
    for start in range(0, max(len(seq) - window + 1, 1), stride):
        chunk = seq[start : start + window]
        if len(chunk) < 50:
            break
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=MAX_TOKENS)
        with torch.no_grad():
            logits = native_model(inputs["input_ids"], inputs["attention_mask"])
            probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()
        top = int(np.argmax(probs))
        rows.append(
            [f"{start + 1}–{start + len(chunk)}", CLASS_MAP[top], f"{probs[top] * 100:.2f}%"]
        )
        if start + window >= len(seq):
            break
    return rows


UBIQUITIN = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
SPOUT = (
    "MSKSKPLTGQWFIGLDIETGELRLVVPKGEEGELAGRILVGIGGGISALNAVIRLSRSGATVVAIERDRERLARAAA"
    "AAGADVVVLDAADPAAAERLAKAEGLDAVVTAMVAGAGETPERLAALRAGADVVTLDGGFRERARRVARLADVPVVA"
    "LEAGSGEEVARARRAGADVVVTGAGGE"
)

CLI_BIN = get_cli_path()
OVERVIEW = f"""
**BioKnot-Core** is a hybrid computational framework for high-throughput identification and classification of topological knots in protein structures, coupling transformer-based sequence pre-screening with an ESMFold 3D structural bridge and Alexander-polynomial topological verification.

* ⚡ **Sequence Pre-Screening (Path A)**: Fine-tuned ESM-2 (8M) transformer with masked residue pooling evaluates topological knot probabilities from sequence alone in milliseconds.
* 🧬 **3D Structural Bridge & Verification (Path B)**: *De novo* 3D atomic coordinate prediction via ESMFold with midpoint Kabsch tiling for long chains, seam RMSD diagnostics, and mathematical knot invariant verification via compiled Rust.
* 📊 **Supported Knot Topologies**: Classifies structures using Rolfsen notation into **$0_1$** (Unknot), **$3_1$** (Trefoil), **$4_1$** (Figure-Eight), and **Complex** knots.

---
### System Engine Status
| Component | Runtime Engine | Status | Details |
| :--- | :--- | :--- | :--- |
| **Path A Classifier** | ESM-2 Transformer | {'🟢 Live' if WEIGHTS_LOADED else '🔴 Unloaded'} | {'Pre-trained weights active' if WEIGHTS_LOADED else 'Missing bioknot_native.pt'} |
| **Path B Verifier** | Compiled Rust Engine | {'🟢 Active' if CLI_BIN else '🟡 Standby'} | {'`' + CLI_BIN + '`' if CLI_BIN else 'CLI not detected (Gradio SDK fallback mode)'} |
| **3D Structure Bridge** | ESMFold + Midpoint Kabsch | 🟢 Live | Window=380 AA, Stride=200 AA |
"""

with gr.Blocks(title="BioKnot-Core") as demo:
    gr.Markdown("# 🧬 BioKnot-Core\n### Edge-Computed Protein Topology Classifier & 3D Structural Bridge")
    gr.Markdown(OVERVIEW)

    with gr.Tab("Topology Classifier"):
        seq_in = gr.Textbox(label="Amino-acid sequence (raw or FASTA)", lines=5)
        classify_btn = gr.Button("Classify topology", variant="primary")
        probs_out = gr.Label(label="Predicted topology", num_top_classes=4)
        classify_note = gr.Markdown()
        gr.Examples([[UBIQUITIN], [SPOUT]], inputs=[seq_in])
        classify_btn.click(classify, inputs=[seq_in], outputs=[probs_out, classify_note])

    with gr.Tab("3D Structure (ESMFold)"):
        fold_in = gr.Textbox(label="Amino-acid sequence (raw or FASTA)", lines=5)
        fold_btn = gr.Button("Fold structure", variant="primary")
        structure_out = gr.HTML(
            value='<div style="height: 480px; display: flex; align-items: center; justify-content: center; background: #0d1117; color: #8b949e; border: 1px dashed #30363d; border-radius: 8px;">Enter a sequence and click &quot;Fold structure&quot; to render the interactive 3D model.</div>',
            label="Predicted 3D Structure",
        )
        with gr.Row():
            pdb_file_out = gr.File(label="Download PDB Structure")
            fold_note = gr.Markdown()
        gr.Examples([[UBIQUITIN]], inputs=[fold_in])
        fold_btn.click(fold, inputs=[fold_in], outputs=[structure_out, pdb_file_out, fold_note])

    with gr.Tab("Sub-Domain Scan"):
        scan_in = gr.Textbox(label="Long sequence (raw or FASTA)", lines=5)
        with gr.Row():
            window_in = gr.Slider(100, 1000, value=400, step=50, label="Window (AA)")
            stride_in = gr.Slider(50, 500, value=200, step=50, label="Stride (AA)")
        scan_btn = gr.Button("Scan sub-domains", variant="primary")
        scan_out = gr.Dataframe(
            headers=["Residue span", "Predicted topology", "Confidence"],
            label="Per-window classification",
            wrap=True,
        )
        scan_btn.click(scan_domains, inputs=[scan_in, window_in, stride_in], outputs=[scan_out])

    with gr.Tab("Runtime"):
        gr.Markdown(
            "Both engines run on CPU. This check is the Space's only GPU entry point "
            "and exists because ZeroGPU requires one — it is never hit by the tabs above, "
            "so no GPU quota is consumed by normal use."
        )
        runtime_btn = gr.Button("Check ZeroGPU runtime")
        runtime_out = gr.Textbox(label="Result")
        runtime_btn.click(gpu_runtime_check, inputs=None, outputs=[runtime_out])

if __name__ == "__main__":
    demo.launch()
