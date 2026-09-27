#!/bin/bash
# M11 prefetch loop (The Bridge / user, 2026-09-27). Detached; survives an app close. Every hour runs
# scripts/o4a_prefetch.py (fetch + completeness-check the files freeze/score need; NO freeze, NO scoring).
# Inbox notes: when the ESSENTIAL files are all cached, when ALL are cached (then exits), or after 72 h (gives up).
cd "$(dirname "$0")/.." || exit 1
OUT=results/o4a_ssm; LOG=$OUT/prefetch.log
INBOX=/Users/sumit/Github/.claude-coordination/inbox/bridge; mkdir -p "$INBOX"
note() { printf "%s\n" "$2" > "$INBOX/$(date +%F)_deepstrain_m11_$1.md"; echo "$(date) NOTE $1" >> "$LOG"; }
DEADLINE=$(( $(date +%s) + 72*3600 )); ESS_SENT=0
echo "$(date) prefetch loop start (pid $$)" >> "$LOG"
while :; do
  echo "$(date) pass" >> "$LOG"
  .venv/bin/python scripts/o4a_prefetch.py >> "$LOG" 2>&1; RC=$?
  if [ $RC -eq 0 ]; then
    note prefetch_complete "DeepStrain M11: ALL strain files for freeze + score are cached and complete ($(date)). Ready for freeze -> commit -> score by hand, as registered. Status: primordial_blackhole_search/$OUT/prefetch.json"
    exit 0
  fi
  if [ $RC -eq 3 ] && [ $ESS_SENT -eq 0 ]; then
    note prefetch_essential "DeepStrain M11: the ESSENTIAL files (freeze PSD + both primary triggers, H1/L1) are cached and complete ($(date)); out-of-domain reporting files still pending. Primary verdicts can run now. Status: primordial_blackhole_search/$OUT/prefetch.json"
    ESS_SENT=1
  fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    note prefetch_gaveup "DeepStrain M11 prefetch GAVE UP after 72 h ($(date)); nothing frozen or scored. Status: primordial_blackhole_search/$OUT/prefetch.json"
    exit 0
  fi
  sleep 3600
done
