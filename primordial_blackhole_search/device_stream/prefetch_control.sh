#!/bin/bash
# Runs INSIDE the Debian chroot, once, in the background: put the 16 control (Arm C) segments on the phone so
# control work can run with no internet (offline mode in run_queue.sh). Same fetch_segment as the trainer.
cd /root/pbhcode || exit 1
export PYTHONPATH=/root/pbhcode
until nice -n 10 /root/pbh/.venv/bin/python -W ignore -c '
import json, sys
from pbh import config as C
from pbh.data import fetch_segment, segment_path
todo = [g for g in json.load(open(C.DATA_DIR / "manifest.json"))["H1"]["train"] if not segment_path("H1", g).exists()]
print(f"control segments missing: {len(todo)}", flush=True)
for g in todo:
    fetch_segment("H1", g)
    print(f"fetched {g}", flush=True)
'; do
  echo "$(date -u '+%F %T') prefetch failed (offline?); retrying in 10 min"; sleep 600
done
echo "$(date -u '+%F %T') PREFETCH_CONTROL_DONE"
