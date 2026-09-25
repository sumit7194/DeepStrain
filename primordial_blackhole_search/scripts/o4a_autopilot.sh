#!/bin/bash
# M11 autopilot (user's option (b), relayed by The Bridge 2026-09-26). Runs detached; survives an app close.
#   1. Every hour: scripts/o4a_gwosc_check.py (PASS = H1 AND L1 each deliver 64 s within 300 s).
#   2. On PASS: launch `o4a_ssm_rescore.py --stage background` exactly as registered (840e336), niced.
#   3. While it runs, every 60 s: terminate it (SIGTERM, our own PID only) if free disk < 5 GB, free swap < 512 MB,
#      or no background segment has completed for 90 min (the stall rule -- the stalled segment counts as FAILED).
#   4. Inbox notes for The Bridge on launch and on end. Give up after 48 h of failed checks.
# Freeze and scoring are NOT automated: the pre-registration requires frozen.json to be committed before scoring.
cd "$(dirname "$0")/.." || exit 1
OUT=results/o4a_ssm; LOG=$OUT/autopilot.log; BGLOG=$OUT/background.log
INBOX=/Users/sumit/Github/.claude-coordination/inbox/bridge; mkdir -p "$INBOX"
note() { printf "%s\n" "$2" > "$INBOX/$(date +%F)_deepstrain_m11_$1.md"; echo "$(date) NOTE $1" >> "$LOG"; }
ncache() { ls $OUT/segs/seg_*.npz 2>/dev/null | wc -l | tr -d ' '; }
START=$(date +%s); DEADLINE=$((START + 48*3600))
echo "$(date) autopilot start (pid $$), cached $(ncache)/30" >> "$LOG"
while :; do
  if .venv/bin/python scripts/o4a_gwosc_check.py >> "$LOG" 2>&1; then
    echo "$(date) CHECK PASS -> launching" >> "$LOG"; break
  fi
  echo "$(date) CHECK FAIL" >> "$LOG"
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    note gaveup "DeepStrain M11 autopilot GAVE UP at $(date): 48 h of hourly GWOSC checks, none passed (H1+L1 each within 300 s). M11 not launched; $(ncache)/30 background segments cached, nothing scored. Log: primordial_blackhole_search/$LOG"
    exit 0
  fi
  sleep 3600
done
nohup nice -n 10 .venv/bin/python scripts/o4a_ssm_rescore.py --stage background >> "$BGLOG" 2>&1 &
PID=$!
note launched "DeepStrain M11 LAUNCHED at $(date) (pid $PID) after a clean GWOSC check. Cached $(ncache)/30 at launch. Guard: disk<5GB / swap<512MB / 90-min stall -> SIGTERM of pid $PID only. Log: primordial_blackhole_search/$BGLOG"
LAST=$(date +%s); SEEN=$(ncache); WHY="completed its pass"
while ps -p $PID >/dev/null 2>&1; do
  N=$(ncache); [ "$N" != "$SEEN" ] && { SEEN=$N; LAST=$(date +%s); }
  AV=$(df -k /System/Volumes/Data | awk 'NR==2{print int($4/1048576)}')
  SF=$(sysctl -n vm.swapusage | awk '{gsub(/M/,"",$9); print int($9)}')
  if [ "$AV" -lt 5 ] || [ "$SF" -lt 512 ]; then WHY="GUARD-STOPPED (disk ${AV} GB, swap free ${SF} MB)"; kill $PID; break; fi
  if [ $(( $(date +%s) - LAST )) -ge 5400 ]; then WHY="STALL-STOPPED (no segment completed for 90 min; that segment counts as FAILED)"; kill $PID; break; fi
  sleep 60
done
sleep 5
N=$(ncache)
FLOOR=$([ "$N" -ge 20 ] && echo "at/above the floor of 20 -> ready for freeze (manual, then commit, then score)" || echo "BELOW the floor of 20 -> freeze refuses as registered")
note ended "DeepStrain M11 background ENDED at $(date): $WHY. Cached $N/30 -- $FLOOR. Nothing scored yet. Log: primordial_blackhole_search/$BGLOG"
