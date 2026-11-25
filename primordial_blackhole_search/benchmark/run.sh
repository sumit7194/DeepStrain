#!/bin/bash
# Launch one benchmark step detached, under the Bridge limits (PLAN.md): nice 15, 2 threads, a pause-and-alert guard
# (job_guard.sh: never kills; memory footprint (top MEM, not RSS) > ~4 GB is inform only; disk < 5 GB or memory pressure pauses, then auto-resumes). Usage: benchmark/run.sh <name> <python args...>   e.g. benchmark/run.sh fetch benchmark/fetch.py fetch
# Writes state/<name>.pid and state/<name>.log. Stop with benchmark/stop.sh <name> (or touch state/PAUSE).
cd "$(dirname "$0")/.." || exit 1
NAME=$1; shift
S=benchmark/state; mkdir -p $S
# Memory-footprint guide value (top MEM / phys_footprint, not RSS) of the job's process group, KB (INFORM only since 2026-11-25). Default 4 GB; override with
# BENCH_RSS_CAP_KB. Validated here so a typo cannot silently disable or zero it.
CAP=${BENCH_RSS_CAP_KB:-4194304}
case "$CAP" in ''|*[!0-9]*) echo "refusing BENCH_RSS_CAP_KB='$CAP' (positive integer KB)"; exit 2;; esac
[ "$CAP" -gt 0 ] || { echo "refusing BENCH_RSS_CAP_KB=0"; exit 2; }
[ -f $S/$NAME.pid ] && kill -0 "$(cat $S/$NAME.pid)" 2>/dev/null && { echo "$NAME already running (pid $(cat $S/$NAME.pid))"; exit 1; }
export OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
nohup perl -MPOSIX -e 'setsid(); exec @ARGV' nice -n 15 .venv/bin/python -u "$@" >> $S/$NAME.log 2>&1 &
PID=$!; echo $PID > $S/$NAME.pid
# guard: PAUSE + ALERT under box pressure, auto-RESUME; RSS above the cap is INFORM only. Never kills (fleet rule
# 2026-11-25: no hard cap may end a job). See benchmark/job_guard.sh.
BENCH_RSS_CAP_KB=$CAP nohup benchmark/job_guard.sh "$NAME" >/dev/null 2>&1 &
echo "started $NAME pid $PID (footprint inform above ${CAP} KB; pause-and-alert guard)"
