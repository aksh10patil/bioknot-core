#!/bin/bash
# Run the BioKnot server locally the same way the container does.
#
#   ./scripts/run_local.sh            # gunicorn, 2 workers (production-like)
#   DEV=1 ./scripts/run_local.sh      # flask dev server, single process
set -euo pipefail

cd "$(dirname "$0")/.."

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export TOKENIZERS_PARALLELISM=false
PORT="${PORT:-7860}"

if [ ! -f bioknot_native.pt ]; then
    echo "No bioknot_native.pt found."
    echo "Build it with:  python -m training.train"
    echo "Continuing anyway -- /predict_custom will return HTTP 503."
    echo
fi

if [ "${DEV:-0}" = "1" ]; then
    exec python -m app.server
fi

exec gunicorn \
    --bind "127.0.0.1:${PORT}" \
    --workers 2 \
    --timeout 300 \
    --graceful-timeout 30 \
    --access-logfile - \
    app.server:app
