---
title: BioKnot-Core Platform
emoji: 🧬
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Topological protein knot detection with an ESMFold 3D bridge
---

# BioKnot-Core Platform

Hybrid topological protein knot detection with a 3D structure bridge.

- **Native transformer classifier** — fine-tuned ESM-2 head predicting $0_1$, $3_1$, $4_1$, or Complex.
- **Sub-domain window scanner** — 400 AA sliding window for long polyproteins.
- **ESMFold + Kabsch tiling** — folds sequences past the 400 AA API ceiling as overlapping
  windows stitched by rigid-body alignment, rendered as a C-alpha backbone in Molstar.

> **On the classifier's accuracy:** the curated dataset in `data/bioknot_dataset.json` contains
> **8 sequences**. The checkpoint built from it memorises that set and is a *demonstration
> artefact*, not a validated predictor. Treat `/predict_custom` output accordingly. The ESMFold
> structure bridge (`/predict`) does not depend on it.

## Layout

```
app/         Flask server, shared model definition, and the served UI
training/    Fine-tuning entrypoint and the build-time cache warmer
data/        Curated dataset and its generator
scripts/     CLI helpers (domain scan, pipeline smoke test, one-shot inference)
rust/        Rust knot-topology crate, compiled to WASM out-of-band
web_ui/      Separate Next.js dashboard — not part of the deployed Space
```

## Deploying to Hugging Face Spaces

Create a Space with **SDK: Docker**, then push this repo to it. The YAML header above supplies
the title, port, and metadata; nothing else needs configuring.

```bash
git remote add space https://huggingface.co/spaces/<user>/<space-name>
git push space main
```

The build pre-downloads the ESM-2 backbone into an image layer and trains the classifier head,
so the running container starts fast and every worker shares identical, reproducible weights.
Expect the first build to take several minutes; subsequent builds reuse the cached layers.

Free **CPU Basic** (2 vCPU / 16 GB) is sufficient — two gunicorn workers with PyTorch land at
roughly 800 MB resident.

### Configuration

All optional, set as Space variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `MAX_SEQUENCE_LENGTH` | `2000` | Hard ceiling on accepted sequence length |
| `FOLD_DEADLINE_SECONDS` | `240` | Wall-clock budget for a tiling job before it returns partial |
| `ESMFOLD_TIMEOUT` | `60` | Per-call timeout against the ESMFold API |
| `ESMFOLD_RETRIES` | `2` | Attempts per window |
| `RATE_LIMIT_REQUESTS` | `10` | Requests per IP per window on `/predict` |
| `RATE_LIMIT_WINDOW` | `60` | Rate-limit window, seconds |
| `ALLOWED_ORIGINS` | *(unset)* | Comma-separated CORS origins; unset means same-origin only |
| `BIOKNOT_ALLOW_UNTRAINED` | *(unset)* | Dev escape hatch: serve predictions with no checkpoint |

## Local development

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python -m training.train      # builds bioknot_native.pt (~2-5 min on CPU)
./scripts/run_local.sh        # http://127.0.0.1:7860
```

`DEV=1 ./scripts/run_local.sh` runs the Flask dev server instead of gunicorn.

## API

| Route | Method | Description |
| --- | --- | --- |
| `/` | GET | Molstar-based web UI |
| `/health` | GET | Status, including whether the checkpoint loaded |
| `/predict` | POST | `{"sequence": "..."}` → C-alpha coordinates |
| `/predict_custom` | POST | `{"sequence": "..."}` → knot class and probabilities |

`/predict_custom` returns **503** when no checkpoint is present rather than serving predictions
from a randomly initialised head.

```bash
curl -X POST http://127.0.0.1:7860/predict_custom \
  -H 'Content-Type: application/json' \
  -d '{"sequence": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"}'
```

## Rust WASM crate

`rust/` holds the topological analyser (`analyze_protein_knot`), which is not wired into the
deployed Space. To build it for the Next.js dashboard:

```bash
cd rust && wasm-pack build --target web --out-dir ../web_ui/public/wasm
```
