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
# Conditions (amended 2026-11-27 13:28 on the Bridge's decision: kernel level 2 became this box's steady state while
# a stopped job keeps all its memory, so 'resume only at level 1' paused for ~a day with no safety gain; swap clause
# amended 2026-11-27 19:14 on the Bridge's decision: an absolute swap level paused L128 at 18:33 with 75% memory free
# and level 1 and could not clear, because macOS swap only drains as its pages are touched again):
#   PAUSE    kernel pressure level >= 4 (critical), OR memory free % < GUARD_MEMFREE_PAUSE_PCT (25),
#            OR swap used > swap baseline + GUARD_SWAP_GROWTH_MB (512), OR free disk < GUARD_DISK_PAUSE_GB (5)
#            swap baseline = swap used at arming, reset to swap used at every resume (swap is a GROWTH signal)
#   RESUME   level <= 2 AND memory free % >= GUARD_MEMFREE_RESUME_PCT (35) AND free disk >= GUARD_DISK_RESUME_GB (6)
#            AND swap has not risen (by more than GUARD_SWAP_RISE_TOL_MB, 32) for GUARD_SWAP_STILL_S (600 s)
#   (13:28-19:14 2026-11-27: PAUSE at swap > 3072 MB, RESUME at swap <= 2048 MB; before 13:28: PAUSE at level >= 2
#    or swap > arming baseline + 1536 MB, RESUME only at level 1)
#   CRITICAL level >= 4 OR free disk < 2 GB: pause (as above) plus a CRITICAL alert
#   memory free % = 'System-wide memory free percentage' from memory_pressure -Q
#   INFORM   process-group MEMORY FOOTPRINT (top MEM = phys_footprint; NOT RSS, which misses Metal/MPS unified memory:
#            a 64x2^21 MPS batch showed ~5.8 GB footprint at tens of MB RSS, 2026-11-26) > BENCH_RSS_CAP_KB (4 GB):
#            log and alert once, NO action
#   INFORM   total swap used > GUARD_SWAP_INFORM_MB (6144) or free disk < GUARD_DISK_INFORM_GB (20): log and alert
#            once per crossing, NO action
# Alerts go to GUARD_INBOX (default: the Bridge's inbox directory).
# Test hooks: GUARD_SWAP_OVERRIDE_FILE / GUARD_FREE_OVERRIDE_FILE / GUARD_LEVEL_OVERRIDE_FILE /
# GUARD_MEMFREE_OVERRIDE_FILE (read every poll), GUARD_INTERVAL_S.
cd "$(dirname "$0")" || exit 1
NAME=$1
INTERVAL=${GUARD_INTERVAL_S:-5}; CAP=${BENCH_RSS_CAP_KB:-4194304}
MPAUSE=${GUARD_MEMFREE_PAUSE_PCT:-25}; MRESUME=${GUARD_MEMFREE_RESUME_PCT:-35}
DPAUSE=${GUARD_DISK_PAUSE_GB:-5}; DRESUME=${GUARD_DISK_RESUME_GB:-6}
SGROW=${GUARD_SWAP_GROWTH_MB:-512}; STILL=${GUARD_SWAP_STILL_S:-600}; STOL=${GUARD_SWAP_RISE_TOL_MB:-32}
SINF=${GUARD_SWAP_INFORM_MB:-6144}; DINF=${GUARD_DISK_INFORM_GB:-20}
INBOX=${GUARD_INBOX:-/Users/sumit/Github/.claude-coordination/inbox/bridge}
for v in "$INTERVAL" "$CAP" "$MPAUSE" "$MRESUME" "$DPAUSE" "$DRESUME" "$SGROW" "$STILL" "$STOL" "$SINF" "$DINF"; do
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
level() {
  if [ -n "${GUARD_LEVEL_OVERRIDE_FILE:-}" ] && [ -f "$GUARD_LEVEL_OVERRIDE_FILE" ]; then cat "$GUARD_LEVEL_OVERRIDE_FILE"; return; fi
  sysctl -n kern.memorystatus_vm_pressure_level
}
memfree_pct() {
  if [ -n "${GUARD_MEMFREE_OVERRIDE_FILE:-}" ] && [ -f "$GUARD_MEMFREE_OVERRIDE_FILE" ]; then cat "$GUARD_MEMFREE_OVERRIDE_FILE"; return; fi
  memory_pressure -Q | awk -F': ' '/free percentage/{v=$2; sub("%","",v); print int(v)}'
}

group_mem_kb() {  # sum of top's MEM (= phys_footprint, includes Metal/MPS unified memory) over the job's process group
  top -l 1 -stats pgrp,mem 2>/dev/null | awk -v g="$PG" '$1==g{v=$2; sub(/[+-]$/,"",v); u=substr(v,length(v)); n=v+0;
    if(u=="G")n*=1048576; else if(u=="M")n*=1024; else if(u=="B")n/=1024; s+=n} END{printf "%d", s}'
}
SBASE=$(swap_used); case "$SBASE" in ''|*[!0-9]*) SBASE=0;; esac
REF_SWAP=$SBASE; REF_T=$(date +%s)
say "armed (rule of 2026-11-27 19:14): pause at level>=4 | mem free<${MPAUSE}% | swap>baseline+${SGROW}MB | disk<${DPAUSE}GB; resume at level<=2 & mem free>=${MRESUME}% & disk>=${DRESUME}GB & swap not risen >${STOL}MB for ${STILL}s (baseline reset at each resume); swap baseline ${SBASE}MB; inform: footprint (top MEM) > ${CAP} KB, swap > ${SINF}MB, disk < ${DINF}GB"
PAUSED=0; INFORMED=0; PEAK=0; SINFORMED=0; DINFORMED=0
while kill -0 "$PID" 2>/dev/null; do
  LVL=$(level); SWAP=$(swap_used); FREE=$(free_gb); MEMF=$(memfree_pct); NOW=$(date +%s)
  for v in "$LVL" "$MEMF" "$SWAP" "$FREE"; do  # an unreadable metric counts as pressure (pause, never resume)
    case "$v" in ''|*[!0-9]*) LVL=4; MEMF=0; SWAP=${SWAP:-0}; FREE=${FREE:-0}; case "$SWAP$FREE" in *[!0-9]*) SWAP=0; FREE=0;; esac;; esac
  done
  if [ "$SWAP" -gt $((REF_SWAP + STOL)) ]; then REF_SWAP=$SWAP; REF_T=$NOW; fi  # a rise restarts the stillness clock
  RSS=$(group_mem_kb)
  [ "$RSS" -gt "$PEAK" ] && PEAK=$RSS
  if [ "$RSS" -gt "$CAP" ] && [ $INFORMED -eq 0 ]; then
    INFORMED=1; say "INFORM: group footprint ${RSS} KB > ${CAP} KB (no action taken)"
    alert inform "memory footprint ${RSS} KB passed the ${CAP} KB guide value; NO action taken (inform only)"
  fi
  if [ "$SWAP" -gt "$SINF" ] && [ $SINFORMED -eq 0 ]; then
    SINFORMED=1; say "INFORM: total swap ${SWAP} MB > ${SINF} MB (no action taken)"
    alert inform_swap "total swap used ${SWAP} MB passed ${SINF} MB; NO action taken (inform only)"
  elif [ "$SWAP" -le "$SINF" ]; then SINFORMED=0; fi
  if [ "$FREE" -lt "$DINF" ] && [ $DINFORMED -eq 0 ]; then
    DINFORMED=1; say "INFORM: free disk ${FREE} GB < ${DINF} GB (no action taken)"
    alert inform_disk "free disk ${FREE} GB fell below ${DINF} GB; NO action taken (inform only)"
  elif [ "$FREE" -ge "$DINF" ]; then DINFORMED=0; fi
  if [ $PAUSED -eq 0 ]; then
    if [ "$LVL" -ge 4 ] || [ "$MEMF" -lt "$MPAUSE" ] || [ "$SWAP" -gt $((SBASE + SGROW)) ] || [ "$FREE" -lt "$DPAUSE" ]; then
      kill -STOP -- -"$PG"; PAUSED=1
      TAG=paused; [ "$LVL" -ge 4 ] || [ "$FREE" -lt 2 ] && TAG=CRITICAL_paused
      say "PAUSED (SIGSTOP pgid $PG): level=$LVL memfree=${MEMF}% swap=${SWAP}MB (baseline ${SBASE}MB) free=${FREE}GB"
      alert "$TAG" "paused (SIGSTOP, not killed): pressure level $LVL, memory free ${MEMF}%, swap ${SWAP} MB (baseline ${SBASE} MB, pause above +${SGROW}), free disk ${FREE} GB. Resumes automatically when level <= 2, memory free >= ${MRESUME}%, disk >= ${DRESUME} GB and swap has not risen for ${STILL} s."
    fi
  elif [ "$LVL" -le 2 ] && [ "$MEMF" -ge "$MRESUME" ] && [ "$FREE" -ge "$DRESUME" ] && [ $((NOW - REF_T)) -ge "$STILL" ]; then
    kill -CONT -- -"$PG"; PAUSED=0; SBASE=$SWAP
    say "RESUMED (SIGCONT pgid $PG): level=$LVL memfree=${MEMF}% swap=${SWAP}MB (new baseline; still $((NOW - REF_T)) s) free=${FREE}GB"
    alert resumed "resumed automatically: pressure level $LVL, memory free ${MEMF}%, swap ${SWAP} MB (new baseline), free disk ${FREE} GB"
  fi
  sleep "$INTERVAL"
done
say "job exited; process-group footprint peak $((PEAK/1024)) MB (sampled every ${INTERVAL} s)"
