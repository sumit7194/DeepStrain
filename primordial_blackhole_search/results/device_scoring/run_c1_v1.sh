#!/bin/bash
# Deferred device scoring (Bridge slot 2026-10-17): C1 (new) then V1 re-score with the sha-verified final.pt.
cd "$(dirname "$0")/../.." || exit 1
for w in dev_c1_original_s1 dev_v1_stream_s1; do
  key=$w; [ "$w" = dev_v1_stream_s1 ] && key=dev_v1_stream_s1_recheck
  echo "$(date) start $w (cache-key $key)"
  nice -n 10 .venv/bin/python scripts/track_eval.py --window-sec 64 --weights "$w" --cache-key "$key" || echo "$(date) FAILED $w"
  echo "$(date) end $w"
done
echo "$(date) ALL DONE"
