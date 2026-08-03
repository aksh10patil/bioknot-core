#!/bin/bash
cd ~/bioknot_wasm
source venv/bin/activate

# Free port 8000
fuser -k 8000/tcp || true

# Limit PyTorch CPU threads per worker
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

# Launch Gunicorn with 2 workers and extended timeout for heavy inference
nohup gunicorn --workers 2 \
                --bind 127.0.0.1:8000 \
                --timeout 120 \
                esmfold_full:app > gunicorn.log 2>&1 &

echo "Waiting for Gunicorn workers to bind to port 8000..."
until nc -z 127.0.0.1 8000; do
    sleep 1
done
echo "Gunicorn production server is live!"
ps aux | grep gunicorn | grep -v grep
