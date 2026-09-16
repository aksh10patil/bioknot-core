---
title: BioKnot-Core
emoji: 🧬
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.22.0
app_file: app.py
short_description: Topological protein knot detection with an ESMFold bridge
python_version: "3.12"
startup_duration_timeout: 30m
pinned: false
---

# 🧬 BioKnot-Core

Topological protein knot detection with a 3D structure bridge.

- **Topology classifier** — a fine-tuned ESM-2 (8M) head predicting `0_1`, `3_1`, `4_1`, or Complex
  from sequence alone.
- **ESMFold bridge** — folds a sequence and renders the backbone. Past the 400 AA ESMFold API
  ceiling it folds overlapping windows and stitches them with rigid Kabsch alignment, stopping
  and reporting a partial result rather than emitting misaligned geometry.
- **Sub-domain scan** — slides a window along long polyproteins and classifies each segment.

- **Supported Topologies** — classifies structures into $0_1$ (Unknot), $3_1$ (Trefoil), $4_1$ (Figure-Eight), and Complex knots based on Rolfsen notation.

## Hardware

Runs on CPU. ESM-2 8M is small and the fold is an upstream API call, so the `@spaces.GPU`
function in `app.py` exists only because ZeroGPU requires one — it sits behind a diagnostic
button on the Runtime tab and is never hit by normal use, so no GPU quota is consumed.

## Source

Developed in [bioknot-core](https://github.com/Antara-oss/bioknot-core). `bioknot_model.py` is
generated from `app/model.py` there by `scripts/sync_space.sh`; edit it upstream, not here.
