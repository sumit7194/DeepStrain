#!/bin/bash
# Final device scoring (2026-10-23, user-approved, Bridge notified): C2, C3, C4, one at a time, nice 10; then the verdict.
cd "$(dirname "$0")/../.." || exit 1
for s in 2 3 4; do
  w=dev_c${s}_original_s$s
  echo "$(date) start $w"
  nice -n 10 .venv/bin/python scripts/track_eval.py --window-sec 64 --weights "$w" --cache-key "$w" || echo "$(date) FAILED $w"
  echo "$(date) end $w"
done
.venv/bin/python device_stream/verdict.py
echo "$(date) ALL DONE"
