#!/bin/bash
cd ~/bioknot_wasm
source venv/bin/activate

# Free port 8000
fuser -k 8000/tcp || true

# Limit PyTorch CPU threads to avoid memory spike OOM kills
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

# Relaunch in background
nohup python3 esmfold_full.py > server.log 2>&1 &

echo "Waiting for process initialization..."
sleep 3
ps aux | grep esmfold_full.py | grep -v grep
