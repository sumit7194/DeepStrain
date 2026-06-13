#!/bin/bash
# Dataset build driver: independent processes, 5-way parallel.
# Optional arg: window seconds (default 256, the v1 build). e.g. ./run_build.sh 64
set -e
cd "$(dirname "$0")/.."
WIN=${1:-256}
PY=.venv/bin/python
N=$($PY scripts/build_dataset.py --init --window-sec "$WIN" | tail -1)
echo "running $N jobs (window ${WIN}s), 5-way parallel..."
seq 0 $((N-1)) | xargs -P 5 -I{} $PY scripts/build_dataset.py --job {} --window-sec "$WIN"
$PY scripts/build_dataset.py --finalize --window-sec "$WIN"
