#!/bin/bash
# Launcher for the Fable (c) round only: same as run.sh (detached, job_guard.sh pause-and-alert, never kills) but at
# NORMAL priority with ONE thread, because one core is reserved for this round (brief v1.1 addendum, house rule 2).
# Usage: benchmark/run_coh2c.sh <name> <python args...>     Stop: benchmark/stop.sh <name>
cd "$(dirname "$0")/.." || exit 1
NAME=$1; shift
S=benchmark/state; mkdir -p $S
[ -f $S/$NAME.pid ] && kill -0 "$(cat $S/$NAME.pid)" 2>/dev/null && { echo "$NAME already running (pid $(cat $S/$NAME.pid))"; exit 1; }
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
nohup perl -MPOSIX -e 'setsid(); exec @ARGV' .venv/bin/python -u "$@" >> $S/$NAME.log 2>&1 &
PID=$!; echo $PID > $S/$NAME.pid
nohup benchmark/job_guard.sh "$NAME" >/dev/null 2>&1 &
echo "started $NAME pid $PID (normal priority, 1 thread, pause-and-alert guard)"
