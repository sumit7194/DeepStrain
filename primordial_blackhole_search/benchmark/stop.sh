#!/bin/bash
# Stop a benchmark step by its PID file only, after checking the PID really is our python job. Prefer touching
# benchmark/state/PAUSE (clean stop between units); this is for when a unit itself must be interrupted.
cd "$(dirname "$0")" || exit 1
P=state/$1.pid; [ -f "$P" ] || { echo "no pid file $P"; exit 1; }
PID=$(cat "$P"); CMD=$(ps -o command= -p "$PID")
case "$CMD" in *python*benchmark/*) PG=$(ps -o pgid= -p "$PID" | tr -d ' '); kill -TERM -- "-$PG"; echo "TERM pgid $PG ($CMD)";;
  "") echo "pid $PID not running";; *) echo "pid $PID is not our job: $CMD — refusing"; exit 1;; esac
