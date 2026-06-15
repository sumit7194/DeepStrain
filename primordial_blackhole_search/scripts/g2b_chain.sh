#!/bin/bash
# G2b unattended chain: build H1+L1 shards -> train cnn_hl -> coincidence eval.
# Self-healing: build_hl.py is resumable, so a power loss mid-build just resumes;
# train/eval re-run from scratch (short). Detached (memory: nohup-long-running).
# Usage: nohup bash scripts/g2b_chain.sh > /tmp/deepstrain_g2b_chain.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
BLOG=/tmp/deepstrain_buildhl.log
BPID=/tmp/deepstrain_buildhl.pid
ts() { date +%H:%M:%S; }

# 1. ensure the build completes (resume/relaunch as needed)
for a in $(seq 1 20); do
  grep -q "HL DATASET BUILD DONE" "$BLOG" 2>/dev/null && { echo "$(ts) build done"; break; }
  if [ -f "$BPID" ] && kill -0 "$(cat "$BPID")" 2>/dev/null; then
    while kill -0 "$(cat "$BPID")" 2>/dev/null; do sleep 30; done
  else
    echo "$(ts) (re)launch resumable build (attempt $a)"
    nohup "$PY" scripts/build_hl.py >> "$BLOG" 2>&1 &
    echo $! > "$BPID"; sleep 20
  fi
done
grep -q "HL DATASET BUILD DONE" "$BLOG" 2>/dev/null || { echo "BUILD FAILED" | tee /tmp/deepstrain_g2b.done; exit 1; }

# 2. train cnn_hl (same recipe as cnn_w64: 16 ep, lr 3e-4, augment)
echo "$(ts) training cnn_hl"
"$PY" scripts/train.py --model cnn --shard-dir data/shards_w64_hl --out cnn_hl > /tmp/deepstrain_trainhl.log 2>&1

# 3. coincidence eval with the H1+L1 model
echo "$(ts) coincidence eval (cnn_hl)"
"$PY" scripts/coinc_eval.py --weights cnn_hl > /tmp/deepstrain_coinchl.log 2>&1

{ echo "G2B DONE $(ts)";
  grep "BEST VAL AUC" /tmp/deepstrain_trainhl.log | tail -1;
  grep -A6 "coincidence vs single" /tmp/deepstrain_coinchl.log | tail -8; } | tee /tmp/deepstrain_g2b.done
