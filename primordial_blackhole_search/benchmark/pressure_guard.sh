#!/bin/bash
# Stop a benchmark job (its whole process group, TERM only) if the shared box comes under memory pressure:
# kernel pressure level >= warn (kern.memorystatus_vm_pressure_level: 1 normal, 2 warn, 4 critical), or swap used
# grows by more than GROW_MB over its value when the guard started.
# Usage: benchmark/pressure_guard.sh <job name> [swap growth MB, default 1536]
# 2026-11-25 (fleet hard-limit audit, approved by the Bridge): swap test changed from ABSOLUTE (> 1536 MB) to GROWTH
# over the arming baseline; this box idles at 260 MB-1 GB of swap with nothing of ours running.
# Test hooks (benchmark/guards_test.sh only): GUARD_SWAP_OVERRIDE_MB replaces the live swap reading after arming,
# GUARD_INTERVAL_S the 5-s poll.
cd "$(dirname "$0")" || exit 1
NAME=$1; GROW=${2:-1536}; INTERVAL=${GUARD_INTERVAL_S:-5}
case "$GROW" in ''|*[!0-9]*) echo "refusing non-integer growth '$GROW'" >&2; exit 2;; esac
PID=$(cat state/$NAME.pid); PG=$(ps -o pgid= -p "$PID" | tr -d ' ')
LOG=state/$NAME.log
swap_used() { sysctl -n vm.swapusage | awk '{for(i=1;i<=NF;i++) if($i=="used") {v=$(i+2); sub("M","",v); print int(v)}}'; }
BASE=$(swap_used)
echo "$(date) PRESSURE GUARD armed: swap baseline ${BASE}MB, TERM above +${GROW}MB or pressure level >= 2" >> "$LOG"
while kill -0 "$PID" 2>/dev/null; do
  LVL=$(sysctl -n kern.memorystatus_vm_pressure_level)
  SWAP=${GUARD_SWAP_OVERRIDE_MB:-$(swap_used)}
  if [ "$LVL" -ge 2 ] || [ "$SWAP" -gt $((BASE + GROW)) ]; then
    echo "$(date) PRESSURE GUARD: level=$LVL swap=${SWAP}MB (base ${BASE}) -> TERM pgid $PG" >> "$LOG"
    kill -TERM -- -"$PG"; break
  fi
  sleep "$INTERVAL"
done
