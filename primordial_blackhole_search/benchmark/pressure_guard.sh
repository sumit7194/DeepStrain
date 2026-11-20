#!/bin/bash
# Stop a benchmark job (its whole process group, TERM only) if the shared box comes under memory pressure:
# kernel pressure level >= warn (kern.memorystatus_vm_pressure_level: 1 normal, 2 warn, 4 critical) or swap used > LIMIT_MB.
# Usage: benchmark/pressure_guard.sh <job name> [swap limit MB, default 1536]
cd "$(dirname "$0")" || exit 1
NAME=$1; LIMIT=${2:-1536}
PID=$(cat state/$NAME.pid); PG=$(ps -o pgid= -p "$PID" | tr -d ' ')
LOG=state/$NAME.log
while kill -0 "$PID" 2>/dev/null; do
  LVL=$(sysctl -n kern.memorystatus_vm_pressure_level)
  SWAP=$(sysctl -n vm.swapusage | awk '{for(i=1;i<=NF;i++) if($i=="used") {v=$(i+2); sub("M","",v); print int(v)}}')
  if [ "$LVL" -ge 2 ] || [ "$SWAP" -gt "$LIMIT" ]; then
    echo "$(date) PRESSURE GUARD: level=$LVL swap=${SWAP}MB -> TERM pgid $PG" >> "$LOG"
    kill -TERM -- -"$PG"; break
  fi
  sleep 5
done
