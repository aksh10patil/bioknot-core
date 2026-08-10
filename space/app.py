"""BioKnot-Core — Gradio Space.

Two engines:
  * a fine-tuned ESM-2 head classifying backbone topology from sequence alone;
  * an ESMFold bridge that folds a sequence (tiling past the 400 AA API ceiling
    with rigid Kabsch stitching) and returns a viewable structure.

Both run on CPU: ESM-2 8M is small and the fold is an upstream API call, so the
`@spaces.GPU` function below exists only because ZeroGPU requires one. Nothing on
the user path enters it, so no GPU quota is consumed.
"""

import spaces  # noqa: F401  -- must precede torch; patches torch.cuda.*

import os
import subprocess
import sys
import tempfile
import time

import gradio as gr
import numpy as np
import requests
import torch

from bioknot_model import (
    CLASS_MAP,
    DEFAULT_WEIGHTS_PATH,
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

# --- Model bootstrap ---------------------------------------------------------
print("[boot] loading tokenizer + classifier...", flush=True)
tokenizer = load_tokenizer()

if not os.path.exists(DEFAULT_WEIGHTS_PATH):
    # No checkpoint shipped with the Space: train one now. Deterministic (seeded)
    # and a few minutes on this dataset -- shipping the .pt skips all of this.
    #
    # It must run in a SUBPROCESS. This module has already imported `spaces`,
    # which patches torch to report CUDA as the active accelerator; an Adam step
    # in this process then calls into CUDA that does not exist here and raises
    # `no CUDA-capable device is detected`. A fresh interpreter sees plain CPU
    # torch and trains normally.
    print("[boot] no checkpoint found — training the head (this takes a few minutes)", flush=True)
    subprocess.run(
        [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_head.py"),
         DEFAULT_WEIGHTS_PATH],
        check=True,
    )

native_model, WEIGHTS_LOADED = load_classifier(DEFAULT_WEIGHTS_PATH)
print(f"[boot] classifier ready (weights_loaded={WEIGHTS_LOADED})", flush=True)


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
    for line in pdb_str.splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            try:
                coords.append(
                    [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                )
            except ValueError:
                continue
    return np.array(coords)


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

    If a window fails or the overlap is too short to fit, alignment can no longer
    be guaranteed, so we stop and report a truncated chain rather than emitting
    misaligned geometry.
    """
    seq_len = len(sequence)
    overlap = WINDOW_SIZE - STRIDE

    first = fetch_pdb(sequence[0:WINDOW_SIZE], deadline)
    if first is None:
        return None, False
    full_chain = extract_ca_coords(first)

    for start in range(STRIDE, seq_len, STRIDE):
        if time.time() >= deadline:
            return full_chain, True
        end = min(start + WINDOW_SIZE, seq_len)
        if end - start < 5 or start >= len(full_chain):
            return full_chain, True

        nxt = fetch_pdb(sequence[start:end], deadline)
        if nxt is None:
            return full_chain, True

        w = extract_ca_coords(nxt)
        n = min(overlap, len(full_chain) - start, len(w))
        if n < 5:
            return full_chain, True

        R, ct, cr = kabsch_rotation(w[:n], full_chain[start : start + n])
        aligned = np.dot(w - ct, R.T) + cr
        full_chain = np.vstack([full_chain[: start + n], aligned[n:]])

        if end == seq_len:
            break

    return full_chain, False


def ca_to_pdb(coords):
    """Write a C-alpha-only backbone as a minimal PDB."""
    lines = []
    for i, p in enumerate(coords, start=1):
        lines.append(
            f"ATOM  {i:>5}  CA  GLY A{i:>4}    "
            f"{p[0]:>8.3f}{p[1]:>8.3f}{p[2]:>8.3f}  1.00  0.00           C"
        )
    lines.append("END")
    return "\n".join(lines)


def _write_pdb(text):
    fd, path = tempfile.mkstemp(suffix=".pdb")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    return path


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
    if len(seq) > MAX_SEQUENCE_LENGTH:
        raise gr.Error(f"{len(seq)} AA exceeds the {MAX_SEQUENCE_LENGTH} AA limit.")

    evaluated = seq[:MAX_TOKENS]
    inputs = tokenizer(evaluated, return_tensors="pt", truncation=True, max_length=MAX_TOKENS)
    with torch.no_grad():
        logits = native_model(inputs["input_ids"], inputs["attention_mask"])
        probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()

    note = f"Evaluated {len(evaluated)} of {len(seq)} AA."
    if len(seq) > MAX_TOKENS:
        note += f" Truncated to the {MAX_TOKENS} AA model capacity."

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
    if len(seq) > MAX_SEQUENCE_LENGTH:
        raise gr.Error(f"{len(seq)} AA exceeds the {MAX_SEQUENCE_LENGTH} AA limit.")

    deadline = time.time() + FOLD_DEADLINE_SECONDS

    if len(seq) <= 400:
        pdb = fetch_pdb(seq, deadline)
        if pdb is None:
            raise gr.Error("ESMFold API timed out or rate-limited. Try again shortly.")
        n_ca = len(extract_ca_coords(pdb))
        return _write_pdb(pdb), f"Folded {len(seq)} AA directly — {n_ca} residues, all-atom."

    coords, truncated = tile_long_sequence(seq, deadline)
    if coords is None or len(coords) == 0:
        raise gr.Error("ESMFold API failed on the initial window.")

    msg = (
        f"Folded {len(seq)} AA via Kabsch tiling "
        f"({WINDOW_SIZE} AA windows, {STRIDE} AA stride) — {len(coords)} C-alpha nodes."
    )
    if truncated:
        msg += " PARTIAL: the fold hit its time budget or an upstream window failed, so the chain stops early."
    return _write_pdb(ca_to_pdb(coords)), msg


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

CAVEAT = """
> **Read this before interpreting anything.** The classifier is fine-tuned on a curated set of
> **8 sequences**. It memorises them; it does not generalise. Treat it as a demonstration of the
> architecture, not as a validated predictor. The ESMFold structure bridge is independent of it
> and does real work.
"""

with gr.Blocks(title="BioKnot-Core") as demo:
    gr.Markdown("# 🧬 BioKnot-Core\nTopological protein knot detection with an ESMFold 3D bridge.")
    gr.Markdown(CAVEAT)

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
        structure_out = gr.Model3D(label="Predicted backbone")
        fold_note = gr.Markdown()
        gr.Examples([[UBIQUITIN]], inputs=[fold_in])
        fold_btn.click(fold, inputs=[fold_in], outputs=[structure_out, fold_note])

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
    demo.launch(mcp_server=True)
