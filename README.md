---
title: BioKnot-Core Platform
emoji: 🧬
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Hybrid protein knot topology classifier with ESM-2, ESMFold, and Rust Alexander-polynomial verification
---

# BioKnot-Core

**Hybrid Protein Knot Topology Classification & 3D Structural Verification Suite**

BioKnot-Core is an end-to-end computational biology framework for identifying and verifying topological knots in protein backbones. It pairs a fast, fine-tuned protein language model (**Path A**) with a 3D structure prediction bridge and an exact Rust algebraic knot-invariant engine (**Path B**), surfacing Path A vs. Path B concordance alongside an interactive 3D structure viewer.

---

## System Architecture

```text
                        Amino Acid Sequence (Raw or Multi-Line FASTA)
                                              |
                    +-------------------------+-------------------------+
                    |                                                   |
                    v                                                   v
     [ Path A: Fast Sequence Pre-Screener ]           [ Path B: 3D Structure & Topology Verifier ]
     * ESM-2 (8M / 35M) + Last 2 Layers Unfrozen      * ESMFold API (<= 400 AA direct fold)
     * Masked Residue Pooling (MAX_RESIDUES = 1022)   * Midpoint Kabsch Tiling (> 400 AA)
     * 1D CNN Classification Head (4 Rolfsen Classes) * Seam RMSD + True AA3 + pLDDT Preservation
                    |                                                   |
                    v                                                   v
       Class Probabilities & Early-Exit Score          Rust `bioknot_cli` / WASM Invariant Engine
       (Skips ~41.6% of folds at 90% recall)           * Spherical Closure + Cyclic KMT Reduction
                    |                                  * Conservative Moller-Trumbore Piercing
                    |                                  * 50 Fibonacci-Sphere Projections
                    |                                  * Alexander Determinant |Delta(-1)|, |Delta(-2)|
                    |                                  * Chain-Break Detector (> 4.5 Angstrom)
                    |                                                   |
                    +-------------------------+-------------------------+
                                              |
                                              v
                     Concordance Report + Interactive 3D Viewer (Molstar / 3Dmol)
```

### Core Components

* **Path A — Sequence Topology Classifier (`app/model.py`, `training/train.py`):**
  * Built on `facebook/esm2_t6_8M_UR50D` (with CLI support for `facebook/esm2_t12_35M_UR50D`), unfreezing the final two transformer encoder layers and attaching a `Conv1d(hidden_dim, 256, kernel_size=5)` + `Linear` classification head.
  * Uses **residue-only masked mean pooling** (`MAX_RESIDUES = 1022`, accounting for `<cls>` and `<eos>` within ESM-2's 1024-token window), zeroing `<cls>`, `<eos>`, and `<pad>` positions both before and after convolution. Batch padding invariance is verified by unit test to a maximum logit difference of $7.82 \times 10^{-8}$.
  * Trained with `FocalLoss` ($\gamma = 2.0$), smoothed class weights, `WeightedRandomSampler`, differential learning rates ($2 \times 10^{-5}$ for ESM-2 layers, $5 \times 10^{-4}$ for the classification head), cosine annealing, validation Macro-F1 early stopping, and post-hoc temperature scaling.
* **Path B — ESMFold Bridge & Midpoint Kabsch Tiling (`app/server.py`, `space/app.py`):**
  * Folds chains $\le 400\text{ AA}$ directly via the ESMFold API (`api.esmatlas.com`).
  * For sequences $> 400\text{ AA}$ (up to `MAX_SEQUENCE_LENGTH = 2000`), tiles overlapping windows (`WINDOW_SIZE = 380`, `STRIDE = 200`, `180 AA` overlap), aligns each window via rigid-body Kabsch SVD (with determinant reflection correction), and **splices at the overlap midpoint** (`mid = curr_overlap_len // 2`) to avoid frayed window termini.
  * Reports per-seam and maximum overlap RMSD, preserves ESMFold pLDDT B-factors (`line[60:66]`), and maps residues to true 3-letter PDB codes (`AA1_TO_AA3`).
* **Rust Topological Invariant Engine (`rust/src/lib.rs`, `rust/src/bin/bioknot_cli.rs`):**
  * Compiled both to a native CLI binary (`bioknot_cli`, baked into the multi-stage Docker image) and to WebAssembly (`bioknot_wasm`, for the local Next.js studio).
  * Performs spherical open-chain closure over 50 Fibonacci-sphere directions, cyclic KMT backbone reduction with conservative Möller–Trumbore ray-triangle piercing, regular 2D projections with over/under strand depth resolution, and $(C-1) \times (C-1)$ **Alexander polynomial minor determinants** evaluated at $t = -1$ and $t = -2$ to classify $0_1$ (Unknot), $3_1$ (Trefoil), $4_1$ (Figure-Eight), $5_2$, $6_1$ (Stevedore), and Complex knots.
  * Automatically flags crystallographic or structural **chain breaks** whenever consecutive $C_\alpha$ spacing exceeds $4.5\,\text{\AA}$.

---

## Dataset & Homology-Controlled Splitting

All training and evaluation data are built reproducibly from experimentally solved structures in **KnotProt 2.0** and **RCSB PDB** (`data/build_dataset.py`), validated with `data/validate_dataset.py` (zero exact duplicates, zero cross-class substring overlaps $> 30\text{ AA}$).

| Class ID | Rolfsen Topology | Chains | Share |
| :---: | :--- | :---: | :---: |
| `0` | Unknot ($0_1$) | 660 | 61.22% |
| `1` | Trefoil ($3_1$) | 336 | 31.17% |
| `2` | Figure-Eight ($4_1$) | 58 | 5.38% |
| `3` | Complex ($5_2, 6_1$, etc.) | 24 | 2.23% |
| **Total** | **All Verified Chains** | **1,078** | **100.00%** |

### Zero-Leakage MMseqs2 Clustering (`data/cluster_splits.py`)

To prevent superfamily memorization across conserved knotted folds (e.g., SPOUT methyltransferases, UCH hydrolases, carbonic anhydrases), sequences are clustered prior to splitting using native **MMseqs2** (`20-ae60e`):

```bash
mmseqs easy-cluster <fasta> <prefix> <tmp> --min-seq-id 0.3 -c 0.8 --cov-mode 0
```

* **Total Homology Clusters:** 470 clusters from 1,078 chains ($\le 30\%$ sequence identity).
* **Cross-Validation Scheme:** 5-fold `StratifiedGroupKFold` grouped strictly by MMseqs2 cluster ID (`data/splits.json`).
* **Cluster Leakage:** **0 shared clusters** across train/validation partitions in all 5 folds.

| Fold | Train Chains (Clusters) | Validation Chains (Clusters) |
| :---: | :---: | :---: |
| Fold 0 | 845 (382) | 233 (88) |
| Fold 1 | 880 (391) | 198 (79) |
| Fold 2 | 874 (370) | 204 (100) |
| Fold 3 | 856 (369) | 222 (101) |
| Fold 4 | 857 (368) | 221 (102) |

---

## Evaluation & Benchmarks

### 1. Path A: 5-Fold Out-of-Fold Performance vs. Linear Probe Ablation (`eval/metrics.json`)

Every chain ($n = 1,078$) is evaluated strictly out-of-fold against an identical 5-fold **Frozen ESM-2 Mean-Pooled Logistic Regression** baseline on the exact same MMseqs2 cluster splits.

| Metric | Frozen ESM-2 Linear Probe | BioKnotClassifier (Fine-Tuned) | Gain ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Macro-F1** | `0.7003` | **`0.7952`** | `+0.0949` |
| **Weighted-F1** | `0.7639` | **`0.8013`** | `+0.0374` |
| **Matthews Corr. Coeff. (MCC)** | `0.5519` | **`0.6243`** | `+0.0724` |
| **Expected Calibration Error (ECE)** | — | `0.2256` | *(see Limitations)* |
| **Unknot ($0_1$) F1** ($n = 660$) | `0.8239` | **`0.8516`** (Prec `0.8155`, Rec `0.8909`) | `+0.0277` |
| **Trefoil ($3_1$) F1** ($n = 336$) | `0.6450` | **`0.6854`** (Prec `0.7724`, Rec `0.6161`) | `+0.0404` |
| **Figure-Eight ($4_1$) F1** ($n = 58$) | `0.9138` | **`0.9402`** (Prec `0.9322`, Rec `0.9483`) | `+0.0264` |
| **Complex ($5_2 / 6_1$) F1** ($n = 24$) | `0.4186` | **`0.7037`** (Prec `0.6333`, Rec `0.7917`) | **`+0.2851`** |

#### 4$\times$4 Out-of-Fold Confusion Matrix ($n = 1,078$)

| Ground Truth \ Predicted | Unknot ($0_1$) | Trefoil ($3_1$) | Figure-Eight ($4_1$) | Complex | Support |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Unknot ($0_1$)** | **588** | 59 | 4 | 9 | 660 |
| **Trefoil ($3_1$)** | 127 | **207** | 0 | 2 | 336 |
| **Figure-Eight ($4_1$)** | 3 | 0 | **55** | 0 | 58 |
| **Complex** | 3 | 2 | 0 | **19** | 24 |

#### Path A Binary Pre-Screening Efficiency (Unknot vs. Any Knot)

When used as a fast sequence triage filter before invoking 3D structure folding:

| Operating Metric | Value | Practical Impact |
| :--- | :---: | :--- |
| **Binary ROC-AUC** | `0.8757` | Strong ranking separation from sequence alone in milliseconds |
| **Binary PR-AUC** | `0.8376` | Stable precision across imbalanced knotting thresholds |
| **Binary MCC** | `0.5020` | — |
| **At 90% Knot Recall** | `61.67%` Unknot Specificity | **41.56%** of ESMFold calls skipped on this dataset (~60% projected at proteome scale where ~98% of chains are unknotted) |
| **At 95% Knot Recall** | `58.48%` Unknot Specificity | **37.66%** of ESMFold calls skipped under high-sensitivity discovery screening |

---

### 2. Path B: Rust Real-PDB Ground-Truth Benchmark (`eval/rust_benchmark.json`)

Evaluated via `cargo run --release --bin verify_knots` on **82 experimentally solved RCSB crystal structures** (`eval/pdb_fixtures/`): 50 hard unknotted controls (TIM barrels, beta-barrels, Rossmann-fold methyltransferases, multi-domain chains), 14 trefoils ($3_1$), 8 figure-eights ($4_1$), 9 five-crossing knots ($5_2$), and 1 Stevedore knot ($6_1$).

| Benchmark Metric | Result |
| :--- | :---: |
| **Total Crystal Structures Evaluated** | 82 |
| **Exact Topology Matches** | **82 / 82 (100.00%)** |
| **Unknot False Positive Rate** | **0.00% (0 / 50)** |
| **Structures Flagged with Chain Breaks ($> 4.5\,\text{\AA}$)** | 29 structures (45 total breaks) |
| **Total Benchmark Runtime** | 0.532 s |
| **Throughput** | **154.07 structures / sec** |

---

### 3. Midpoint Kabsch Stitching Geometric Round-Trip Verification (`eval/tiling_benchmark.json`)

Evaluated via `eval/tiling_benchmark.py` across **10 long experimental PDB chains** (450 to 901 AA, including hexokinases, outer-membrane porins, GroEL chaperones, and long knotted domains). Each chain is sliced into overlapping 380-AA windows (`stride = 200`, `overlap = 180`), subjected to independent rigid 3D rotations per tile, reconstructed with midpoint Kabsch SVD stitching, and verified with `bioknot_cli`.

| Verification Metric | Result |
| :--- | :---: |
| **Long Chains Tested (450 to 901 AA)** | 10 |
| **Exact Topology Preservation** | **10 / 10 (100.00%)** |
| **Overall Max Seam RMSD** | $8.34 \times 10^{-14}\,\text{\AA}$ |
| **Overall Mean Seam RMSD** | $4.68 \times 10^{-14}\,\text{\AA}$ |

> **Scope note:** This test verifies exact rigid-body round-trip fidelity (confirming the SVD alignment and midpoint splice introduce zero numerical drift or artificial crossings). On live ESMFold predictions of $> 400\text{ AA}$ proteins, independently folded overlapping windows naturally exhibit structural disagreement (~1 to $4\,\text{\AA}$ seam RMSD), which `/predict` surfaces directly via `max_seam_rmsd` with a warning above $3.0\,\text{\AA}$.

---

## Known Limitations & Scientific Transparency

1. **Probability Calibration Trade-Off ($\text{ECE} = 0.2256$):** Combining `WeightedRandomSampler` with class-weighted `FocalLoss` ($\gamma = 2.0$) resolved minority-class collapse (raising Macro-F1 from `0.40` to `0.7952`), but shifted posterior probabilities away from true empirical frequencies even after scalar temperature scaling. Vector/Dirichlet post-hoc calibration on validation logits is planned to reduce ECE below `0.08` without sacrificing minority recall.
2. **MMseqs2 Coverage Mode:** Clustering used `--min-seq-id 0.3 -c 0.8 --cov-mode 0` (bidirectional 80% coverage). Isolated single-domain constructs and longer multi-domain constructs from the same superfamily can fall into separate clusters under bidirectional coverage; evaluating under target-coverage mode (`--cov-mode 1`) is a useful next stress test, particularly for conserved $4_1$ superfamilies.
3. **Alexander Polynomial Chirality:** Evaluating $|\Delta(-1)|$ and $|\Delta(-2)|$ accurately classifies $0_1, 3_1, 4_1, 5_2, 6_1$ at $> 150$ structures/sec, but the Alexander polynomial is achiral and cannot distinguish left-handed ($-3_1$) from right-handed ($+3_1$) mirror knots (which require Jones or HOMFLY-PT invariants).
4. **Shallow Knots & Slipknots:** Current inference evaluates full-chain topology (and fixed sliding windows in sub-domain scanning); automated slipknot matrix generation and exact knot-core boundary localization are not yet implemented.

---

## Repository Layout

```text
app/            Flask API server (server.py), shared ESM-2 classifier (model.py), Molstar UI (index.html)
data/           Dataset builder (build_dataset.py), integrity validator, MMseqs2 cluster splitter, splits.json
eval/           5-fold CV evaluator (evaluate.py), RCSB fixture fetcher, tiling verifier, JSON benchmark artifacts
rust/           Pure-Rust + WASM Alexander-polynomial engine (lib.rs), bioknot_cli, verify_knots benchmark binary
space/          Hugging Face Docker Space bundle (Gradio UI, multi-stage Dockerfile, synced model & weights)
training/       MPS/CUDA/CPU fine-tuning pipeline (train.py) and Docker backbone cache warmer (preload.py)
tests/          Pytest suite (masked pooling invariance, PDB formatting, bioknot_cli, API routes)
scripts/        Local runner (run_local.sh), Space sync tool (sync_space.sh), CLI inference/scan helpers
web_ui/         Standalone Next.js + 3Dmol + Rust WASM studio (local development)
```

---

## Quick Start & Reproducing Evaluations

### 1. Local Environment Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Optional: install MMseqs2 for homology clustering reproduction
brew install mmseqs2   # macOS
```

### 2. Rebuild Dataset, Splits, Training, & Evaluations

```bash
# Build and validate verified KnotProt 2.0 + RCSB PDB dataset
python data/build_dataset.py
python data/validate_dataset.py

# Cluster at <=30% sequence identity with MMseqs2 and generate 5-fold StratifiedGroupKFold splits
python data/cluster_splits.py

# Train champion classifier (auto-selects CUDA -> MPS -> CPU)
python -m training.train --seed 42

# Run 5-fold out-of-fold evaluation + frozen ESM-2 linear probe ablation
python eval/evaluate.py

# Run Rust unit tests & 82-structure real-PDB ground-truth benchmark
cd rust && cargo test && cargo run --release --bin verify_knots && cd ..

# Run Kabsch midpoint stitching round-trip verification
python eval/tiling_benchmark.py

# Run Python test suite (including masked pooling padding-invariance check)
pytest
```

### 3. Run the Local Server

```bash
# Build native Rust verifier so /predict runs Path B verification locally
cd rust && cargo build --release --bin bioknot_cli && cd ..

# Start gunicorn (2 workers, production-like) on http://127.0.0.1:7860
./scripts/run_local.sh

# Or single-process Flask dev server
DEV=1 ./scripts/run_local.sh
```

### 4. Local Next.js + Rust WASM Studio (`web_ui/`)

```bash
cd rust && wasm-pack build --target web --out-dir ../web_ui/public/wasm
cd ../web_ui && npm install && npm run dev
```

---

## REST API Reference

| Route | Method | Description |
| :--- | :---: | :--- |
| `/` | `GET` | Molstar-based dual-engine web interface |
| `/health` | `GET` | Server readiness, checkpoint status (`weights_loaded`), and `max_sequence_length` |
| `/predict_custom` | `POST` | Path A: `{"sequence": "..."}` $\rightarrow$ predicted knot class, probabilities, evaluated length (`503` if weights missing) |
| `/predict` | `POST` | Path B: `{"sequence": "..."}` $\rightarrow$ ESMFold + midpoint Kabsch $C_\alpha$ coordinates, pLDDT, `max_seam_rmsd`, and Rust `topology` verification |

Example request:

```bash
curl -X POST http://127.0.0.1:7860/predict_custom \
  -H 'Content-Type: application/json' \
  -d '{"sequence": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"}'
```

### Environment Configuration

| Variable | Default | Purpose |
| :--- | :---: | :--- |
| `MAX_SEQUENCE_LENGTH` | `2000` | Maximum accepted sequence length in residues |
| `FOLD_DEADLINE_SECONDS` | `240` | Wall-clock budget for long-chain tiling before returning a partial structure |
| `ESMFOLD_TIMEOUT` | `60` | Per-window HTTP timeout against `api.esmatlas.com` |
| `ESMFOLD_RETRIES` | `2` | Retry attempts per window |
| `RATE_LIMIT_REQUESTS` | `10` | Per-IP request cap per window on `/predict` |
| `RATE_LIMIT_WINDOW` | `60` | Rate-limit window in seconds |
| `ALLOWED_ORIGINS` | *(unset)* | Comma-separated CORS origins (unset = same-origin only) |
| `BIOKNOT_ALLOW_UNTRAINED` | *(unset)* | Development flag allowing `/predict_custom` without a checkpoint |

---

## Docker & Hugging Face Spaces Deployment

Both the root `Dockerfile` and `space/Dockerfile` use a **multi-stage Docker build** (`rust:1.83-slim` builder $\rightarrow$ `python:3.10-slim` runtime) that compiles `bioknot_cli` into `/usr/local/bin/bioknot_cli`, caches the ESM-2 backbone, and loads the pre-trained `bioknot_native.pt` checkpoint without boot-time retraining.

To synchronize shared model code, dataset, compiled artifacts, and weights into `space/` and push to Hugging Face Spaces (`sdk: docker`):

```bash
./scripts/sync_space.sh
./scripts/sync_space.sh --upload
```

Free **CPU Basic** (`2 vCPU / 16 GB RAM`) is sufficient for production serving.