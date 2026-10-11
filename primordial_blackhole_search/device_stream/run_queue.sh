#!/bin/bash
# Runs INSIDE the Debian chroot on the phone: the pre-registered job order, each job resumable.
# A job that exits non-zero is retried after 5 min from its last checkpoint.
# OFFLINE MODE (amendment 2): if a fresh-noise job fails while the internet is down and the 16 control segments are
# on disk, work on the next unfinished control run for up to 30 min, then re-check. Scheduling only: every run is
# deterministic per (seed, segment) and checkpointed, so results are identical to the pre-registered order.
# STALL WATCHDOG (amendment 3): a job is killed (then retried from its checkpoint like any failure) when it has logged
# no new segment for STALL_S AND the phone has received < NET_MIN_B over the last NET_QUIET_S -- a dead connection or a
# hang, never a slow-but-moving download. Scheduling only; the trainer and every random draw are unchanged.
# Started at boot by the init via /mnt/userdata/autostart.sh; the lock makes a second start a no-op.
cd /root/pbhcode || exit 1
exec 9>/root/pbhcode/.queue.lock
flock -n 9 || { echo "$(date -u '+%F %T') queue already running; not starting a second one"; exit 0; }
export PYTHONPATH=/root/pbhcode
log() { echo "$(date -u '+%F %T') $*"; }
sha256sum -c --quiet FROZEN.sha256 || { log "FROZEN hashes do not match: refusing to train"; exit 1; }
PY=/root/pbh/.venv/bin/python
online() { curl -s -o /dev/null -m 20 -I https://gwosc.org/; }
controls_local() { $PY -c '
import json, sys
from pbh import config as C
from pbh.data import segment_path
sys.exit(0 if all(segment_path("H1", g).exists() for g in json.load(open(C.DATA_DIR / "manifest.json"))["H1"]["train"]) else 1)'; }
STALL_S=${STALL_S:-2700}; NET_QUIET_S=${NET_QUIET_S:-900}; NET_MIN_B=${NET_MIN_B:-1000000}; POLL_S=${POLL_S:-60}
rx_bytes() { awk '/wlan0:|usb0:/ {sub(/.*:/, ""); s += $1} END {print s+0}' /proc/net/dev; }
# run_watched SOURCE SEED [extra trainer args]: the trainer under the stall watchdog; returns its exit status
run_watched() {
  local src=$1 seed=$2; shift 2
  local seg="runs/${src}_s${seed}/segments.jsonl" t0 now seg_t rx rx_mark net_t
  setsid ${TRAINER:-$PY -W ignore stream_train.py} --source "$src" --seed "$seed" "$@" &   # own process group
  local pid=$!
  t0=$(date +%s); rx_mark=$(rx_bytes); net_t=$t0
  while kill -0 "$pid" 2>/dev/null; do
    sleep "$POLL_S"
    now=$(date +%s); rx=$(rx_bytes)
    [ $((rx - rx_mark)) -ge "$NET_MIN_B" ] && { rx_mark=$rx; net_t=$now; }
    seg_t=$t0; [ -f "$seg" ] && { m=$(stat -c %Y "$seg"); [ "$m" -gt "$seg_t" ] && seg_t=$m; }
    if [ $((now - seg_t)) -gt "$STALL_S" ] && [ $((now - net_t)) -gt "$NET_QUIET_S" ]; then
      log "STALL: $src s$seed no segment for $((now - seg_t)) s and < $NET_MIN_B B received for $((now - net_t)) s; killing pid $pid"
      kill -TERM -- "-$pid" 2>/dev/null; sleep 20; kill -KILL -- "-$pid" 2>/dev/null
    fi
  done
  wait "$pid"
}
next_control() { for s in 0 1 2 3 4; do [ -f "runs/original_s$s/final.pt" ] || { echo "$s"; return 0; }; done; return 1; }

for job in "stream 0" "original 0" "stream 1" "original 1" "original 2" "original 3" "original 4"; do
  set -- $job
  [ -f "runs/$1_s$2/final.pt" ] && { log "skip $1 s$2 (done)"; continue; }
  until run_watched "$1" "$2"; do
    if [ "$1" = stream ] && ! online && controls_local && c=$(next_control); then
      log "job $1 s$2 failed and internet is down: offline work on control C$c for up to 30 min"
      timeout 1800 $PY -W ignore stream_train.py --source original --seed "$c"
      log "offline window over (rc=$?); re-checking internet"
    else
      log "job $1 s$2 exited non-zero; retrying in 300 s"
      sleep 300
    fi
  done
done
log "QUEUE_DONE"
