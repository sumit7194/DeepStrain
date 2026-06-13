#!/bin/bash
# Self-healing overnight runner for stage-1 semi-coherent training.
# Adopts an already-running training PID; if it dies before completing,
# relaunches with --resume (detached, skips cached sweep, resumes from the last
# epoch checkpoint); on completion, runs the eval. Detached so it survives the
# session lifecycle (see memory: nohup-long-running).
#
# Usage: nohup bash scripts/overnight_train.sh <initial_pid> > /tmp/babysitter.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
TLOG=/tmp/deepstrain_train.log
ELOG=/tmp/deepstrain_eval.log
DONE=/tmp/deepstrain_overnight.done
PID="${1:-0}"
ts() { date +%H:%M:%S; }

for attempt in $(seq 1 30); do
  if [ "$PID" != "0" ]; then
    while kill -0 "$PID" 2>/dev/null; do sleep 30; done
  fi
  if grep -q "BEST VAL AUC" "$TLOG" 2>/dev/null; then
    echo "$(ts) training complete"; break
  fi
  echo "$(ts) training (pid $PID) ended incomplete -> relaunch --resume (attempt $attempt)"
  nohup "$PY" scripts/train_semicoherent.py --sweep --resume --epochs 16 >> "$TLOG" 2>&1 &
  PID=$!
  echo "$(ts) relaunched as pid $PID"
  sleep 15
done

if grep -q "BEST VAL AUC" "$TLOG" 2>/dev/null; then
  echo "$(ts) training done -> eval"
  for etry in 1 2 3; do
    nohup "$PY" scripts/eval_semicoherent.py > "$ELOG" 2>&1 &
    EPID=$!
    while kill -0 "$EPID" 2>/dev/null; do sleep 20; done
    grep -q "wrote eval_semicoherent" "$ELOG" 2>/dev/null && break
    echo "$(ts) eval ended without finishing, retry $etry"
  done
  { echo "OVERNIGHT DONE $(ts)";
    grep "BEST VAL AUC" "$TLOG" 2>/dev/null | tail -1;
    grep -A6 '"learned"' "$ELOG" 2>/dev/null | tail -8; } | tee "$DONE"
else
  echo "$(ts) GAVE UP after 30 relaunch attempts — training never completed" | tee "$DONE"
fi
