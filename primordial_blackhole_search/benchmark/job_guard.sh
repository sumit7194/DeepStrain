#!/bin/bash
# Guard one benchmark job: PAUSE it (SIGSTOP its process group) and ALERT the Bridge under box pressure, then
# RESUME it (SIGCONT) automatically when the pressure clears. It NEVER kills.
#
# Fleet rule (Sumit, 2026-11-25, via the Bridge): no hard cap may end a job. Disk and memory pressure pause and
# alert, then resume on recovery. SIGSTOP frees no memory on macOS (09-24 lesson), but it stops growth, and a
# human or sister session frees the memory.
#
# Usage: benchmark/job_guard.sh <job name>   (run.sh starts it; reads state/<name>.pid)
#
# Conditions:
#   PAUSE    kernel pressure level >= 2, OR swap used > arming baseline + GUARD_SWAP_GROW_MB (1536),
#            OR free disk < GUARD_DISK_PAUSE_GB (5)
#   RESUME   level 1 AND swap <= baseline + grow AND free disk >= GUARD_DISK_RESUME_GB (6)
#   CRITICAL level >= 4 OR free disk < 2 GB: pause (as above) plus a CRITICAL alert
#   INFORM   process-group RSS > BENCH_RSS_CAP_KB (4 GB): log and alert once, NO action
# Alerts go to GUARD_INBOX (default: the Bridge's inbox directory).
# Test hooks: GUARD_SWAP_OVERRIDE_FILE / GUARD_FREE_OVERRIDE_FILE (read every poll), GUARD_INTERVAL_S.
cd "$(dirname "$0")" || exit 1
NAME=$1
INTERVAL=${GUARD_INTERVAL_S:-5}; GROW=${GUARD_SWAP_GROW_MB:-1536}; CAP=${BENCH_RSS_CAP_KB:-4194304}
DPAUSE=${GUARD_DISK_PAUSE_GB:-5}; DRESUME=${GUARD_DISK_RESUME_GB:-6}
INBOX=${GUARD_INBOX:-/Users/sumit/Github/.claude-coordination/inbox/bridge}
for v in "$INTERVAL" "$GROW" "$CAP" "$DPAUSE" "$DRESUME"; do
  case "$v" in ''|*[!0-9]*) echo "job_guard: refusing non-integer setting '$v'" >&2; exit 2;; esac
done
LOG=state/$NAME.log
PID=$(cat "state/$NAME.pid" 2>/dev/null) || { echo "job_guard: no pid file for $NAME" >&2; exit 2; }
sleep 2
CMD=$(ps -o command= -p "$PID")
case "$CMD" in *python*benchmark/*) : ;; *) echo "job_guard: pid $PID is not our job ('$CMD'), refusing" >&2; exit 2;; esac
PG=$(ps -o pgid= -p "$PID" | tr -d ' ')

say() { echo "$(date) GUARD[$NAME]: $*" >> "$LOG"; }
alert() {  # $1 event tag, $2 text
  mkdir -p "$INBOX"
  printf '# deepstrain -> Bridge: job %s %s at %s\n- %s\n- pid %s, pgid %s, command: %s\n' \
    "$NAME" "$1" "$(date '+%F %T %Z')" "$2" "$PID" "$PG" "$CMD" > "$INBOX/$(date +%F_%H%M%S)_deepstrain_${NAME}_$1.md"
}
swap_used() {
  if [ -n "${GUARD_SWAP_OVERRIDE_FILE:-}" ] && [ -f "$GUARD_SWAP_OVERRIDE_FILE" ]; then cat "$GUARD_SWAP_OVERRIDE_FILE"; return; fi
  sysctl -n vm.swapusage | awk '{for(i=1;i<=NF;i++) if($i=="used") {v=$(i+2); sub("M","",v); print int(v)}}'
}
free_gb() {
  if [ -n "${GUARD_FREE_OVERRIDE_FILE:-}" ] && [ -f "$GUARD_FREE_OVERRIDE_FILE" ]; then cat "$GUARD_FREE_OVERRIDE_FILE"; return; fi
  df -g / | awk 'NR==2{print $4}'
}

BASE=$(sysctl -n vm.swapusage | awk '{for(i=1;i<=NF;i++) if($i=="used") {v=$(i+2); sub("M","",v); print int(v)}}')
say "armed: swap baseline ${BASE}MB (pause above +${GROW}), disk pause < ${DPAUSE} GB / resume >= ${DRESUME} GB, RSS inform > ${CAP} KB"
PAUSED=0; INFORMED=0; PEAK=0
while kill -0 "$PID" 2>/dev/null; do
  LVL=$(sysctl -n kern.memorystatus_vm_pressure_level); SWAP=$(swap_used); FREE=$(free_gb)
  RSS=$(ps -axo pgid=,rss= | awk -v g="$PG" '$1==g{s+=$2}END{print s+0}')
  [ "$RSS" -gt "$PEAK" ] && PEAK=$RSS
  if [ "$RSS" -gt "$CAP" ] && [ $INFORMED -eq 0 ]; then
    INFORMED=1; say "INFORM: group RSS ${RSS} KB > ${CAP} KB (no action taken)"
    alert inform "RSS ${RSS} KB passed the ${CAP} KB guide value; NO action taken (inform only)"
  fi
  if [ "$LVL" -ge 2 ] || [ "$SWAP" -gt $((BASE + GROW)) ] || [ "$FREE" -lt "$DPAUSE" ]; then
    if [ $PAUSED -eq 0 ]; then
      kill -STOP -- -"$PG"; PAUSED=1
      TAG=paused; [ "$LVL" -ge 4 ] || [ "$FREE" -lt 2 ] && TAG=CRITICAL_paused
      say "PAUSED (SIGSTOP pgid $PG): level=$LVL swap=${SWAP}MB (base ${BASE}) free=${FREE}GB"
      alert "$TAG" "paused (SIGSTOP, not killed): pressure level $LVL, swap ${SWAP} MB (baseline ${BASE}, bar +${GROW}), free disk ${FREE} GB. Resumes automatically when level=1, swap <= $((BASE + GROW)) MB and disk >= ${DRESUME} GB."
    fi
  elif [ $PAUSED -eq 1 ] && [ "$LVL" -le 1 ] && [ "$FREE" -ge "$DRESUME" ]; then
    kill -CONT -- -"$PG"; PAUSED=0
    say "RESUMED (SIGCONT pgid $PG): level=$LVL swap=${SWAP}MB free=${FREE}GB"
    alert resumed "resumed automatically: pressure level $LVL, swap ${SWAP} MB, free disk ${FREE} GB"
  fi
  sleep "$INTERVAL"
done
say "job exited; process-group RSS peak $((PEAK/1024)) MB (sampled every ${INTERVAL} s)"
