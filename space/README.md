---
title: BioKnot-Core
emoji: 🧬
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.22.0
app_file: app.py
short_description: Protein knot topology detection with ESMFold 3D bridge
python_version: "3.12"
startup_duration_timeout: 30m
pinned: false
---

# 🧬 BioKnot-Core

Topological protein knot detection with a 3D structure bridge.

- **Topology classifier** — a fine-tuned ESM-2 (8M) head predicting `0_1`, `3_1`, `4_1`, or Complex
  from sequence alone (0.7952 Macro-F1 across 5-fold MMseqs2 30% sequence-identity clusters).
- **ESMFold bridge** — folds a sequence and renders the backbone. Past the 400 AA ESMFold API
  ceiling it folds overlapping windows and stitches them with rigid Kabsch alignment, stopping
  and reporting a partial result rather than emitting misaligned geometry.
- **Sub-domain scan** — slides a window along long polyproteins and classifies each segment.
- **Topological Invariant Engine** — Compiled native Rust binary (`bioknot_cli`) running Kadanoff–Swift reduction and Alexander polynomial determinant calculations at 156+ structures/second with 100% ground-truth accuracy.

## Benchmark Summary

- **Real PDB Fixtures (82 chains):** 100.00% exact topological match, 0.00% unknot false positives (0/50).
- **5-Fold MMseqs2 Cross-Validation (1,078 chains):** Macro-F1: 0.7952, Weighted-F1: 0.8013, MCC: 0.6243.
- **Early-Exit Pre-Screening:** 41.56% ESMFold compute saved at 90% knotting recall.
- **Midpoint Kabsch Tiling:** 100.00% topology preservation across long chains up to 901 AA.

## Deployment & Container Specs

This Space runs on the **Docker SDK** (`sdk: docker`), utilizing a multi-stage Docker build:
1. **Builder Stage:** Compiles `bioknot_cli` from Rust source using `cargo build --release`.
2. **Runtime Stage:** Installs PyTorch, Transformers, and Gradio, pre-downloads the ESM-2 8M weights, and copies the pre-built `bioknot_native.pt` classifier artifact and `bioknot_cli` binary into `/usr/local/bin`. Boot-time training is disabled to ensure instant container startup without timeouts.

## Source

Developed in [bioknot-core](https://github.com/Antara-oss/bioknot-core). `bioknot_model.py` and `bioknot_native.pt` are synced via `scripts/sync_space.sh`.
