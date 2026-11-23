#!/bin/bash
# Regenerate the Space's copies of the shared model definition and dataset, then
# upload. Run from anywhere; paths resolve relative to the repo root.
#
#   ./scripts/sync_space.sh              # sync files only
#   ./scripts/sync_space.sh --upload     # sync, then push to the Space
set -euo pipefail

cd "$(dirname "$0")/.."

SPACE_ID="${SPACE_ID:-Aksh10/bioknot-core}"

python3 - <<'PY'
import pathlib

src = pathlib.Path("app/model.py").read_text()
src = src.replace(
    'REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))\n'
    'DEFAULT_WEIGHTS_PATH = os.path.join(REPO_ROOT, "bioknot_native.pt")',
    'MODULE_DIR = os.path.dirname(os.path.abspath(__file__))\n'
    'DEFAULT_WEIGHTS_PATH = os.path.join(MODULE_DIR, "bioknot_native.pt")',
)
src = src.replace(
    '"""Shared definition of the BioKnot native topological classifier.',
    '"""Shared definition of the BioKnot native topological classifier.\n\n'
    'GENERATED FILE -- do not edit here. Synced from app/model.py by scripts/sync_space.sh.',
)
src = src.replace("See `app/server.py`.", "See `app.py`.")
pathlib.Path("space/bioknot_model.py").write_text(src)
print("synced space/bioknot_model.py")
PY

cp data/bioknot_dataset.json space/data/bioknot_dataset.json
echo "synced space/data/bioknot_dataset.json"

if [ -f "bioknot_native.pt" ]; then
    cp bioknot_native.pt space/bioknot_native.pt
    echo "synced space/bioknot_native.pt (pre-built classifier artifact)"
fi

if [ -f "Dockerfile" ]; then
    cp Dockerfile space/Dockerfile
    echo "synced space/Dockerfile (multi-stage Rust + Python container spec)"
fi

CLI_BIN="rust/target/release/bioknot_cli"
if [ -f "$CLI_BIN" ]; then
    cp "$CLI_BIN" space/bioknot_cli
    chmod +x space/bioknot_cli
    echo "synced space/bioknot_cli binary"
fi

if [ "${1:-}" = "--upload" ]; then
    echo "uploading to ${SPACE_ID}..."
    # Patterns are fnmatch-based with '*' matching recursively, so "*__pycache__*"
    # catches top-level and nested caches alike; "**/__pycache__/**" misses the
    # top-level one.
    hf upload "${SPACE_ID}" space/ . --repo-type space \
        --exclude "*__pycache__*" --exclude "*.pyc"
fi
