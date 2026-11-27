#!/bin/bash
# Regression tests for the guards under the fleet rule of 2026-11-25: NO hard cap may end a job. Pressure PAUSES
# (SIGSTOP) and ALERTS; recovery RESUMES (SIGCONT); RSS over the guide value is INFORM only; nothing ever kills.
#   1. ../../.claude-coordination/blackhole_watchdog_9f21.sh  pause/resume on swap or compressor GROWTH
#   2. benchmark/job_guard.sh (via run.sh)                    pause/resume by the 2026-11-27 19:14 rule (level,
#                                                              memory free %, swap GROWTH, disk); footprint, swap and
#                                                              disk inform (the swap amendment's own pass/fail
#                                                              tests: benchmark/guard_swap_test.sh)
#   3. benchmark/run.sh                                       BENCH_RSS_CAP_KB validation
# Every case checks the process state, not the log: 'T' in ps stat = stopped (paused). Alerts go to a test inbox
# under state/, never to the Bridge's real one. Uses only a tiny dummy process. ~1.5 min.
set -u
B="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(cd "$B/.." && pwd)"
W=/Users/sumit/Github/.claude-coordination/blackhole_watchdog_9f21.sh
S="$B/state"; mkdir -p "$S"
TI="$S/guard_test_inbox"; rm -rf "$TI"; mkdir -p "$TI"
export GUARD_INBOX="$TI" WD_INBOX="$TI"
FAIL=0
ok()    { printf '  PASS  %s\n' "$1"; }
bad()   { printf '  FAIL  %s\n' "$1"; FAIL=1; }
check() { if [ "$2" = "$3" ]; then ok "$1 ($2)"; else bad "$1: got '$2', want '$3'"; fi; }
state() { local s; s=$(ps -o stat= -p "$1" 2>/dev/null) || { echo dead; return; }; case "$s" in *T*) echo paused;; *) echo running;; esac; }
swap_live() { sysctl -n vm.swapusage | awk '{for(i=1;i<=NF;i++) if($i=="used") {v=$(i+2); sub("M","",v); print int(v)}}'; }
printf 'import time\nwhile True: time.sleep(1)\n' > "$S/guard_dummy.py"
DF=$(df -g / | awk 'NR==2{print $4}')
[ "$DF" -lt 6 ] && { echo "free disk ${DF} GB < 6: the resume cases cannot be tested now"; exit 3; }

echo "1. watchdog decision rule (no process touched)"
check "swap 1000->1100: old absolute rule acted on this"  "$("$W" --decide 1000 1100 3000 3100)" OK
check "swap 300->900 (+600 > +512)"                      "$("$W" --decide 300 900 3000 3000)" PAUSE
check "compressor 3000->4100 (+1100 > +1024)"            "$("$W" --decide 300 300 3000 4100)" PAUSE
echo "1b. watchdog live: pause, resume, never kill"
OV="$S/guard_wd_swap"; rm -f "$OV"
( cd "$ROOT" && exec python3 "$S/guard_dummy.py" ) & D=$!; sleep 1
WD_SWAP_OVERRIDE_FILE="$OV" WD_INTERVAL_S=1 WD_LOG="$S/guard_wd.log" "$W" "$D" & WP=$!; sleep 2
check "no pressure: target keeps running" "$(state $D)" running
echo $(( $(swap_live) + 5000 )) > "$OV"; sleep 3
check "swap +5000 MB: target PAUSED (not killed)" "$(state $D)" paused
echo $(( $(swap_live) )) > "$OV"; sleep 3
check "pressure cleared: target RESUMED" "$(state $D)" running
check "alerts written (paused + resumed)" "$(ls "$TI" | grep -c watchdog)" 2
kill "$D"; wait "$WP" 2>/dev/null
echo "1c. watchdog target validation (existing test)"
bash /Users/sumit/Github/.claude-coordination/blackhole_watchdog_test.sh | sed 's/^/    /' || FAIL=1

echo "2. job_guard via run.sh: pause, resume, inform, never kill"
cd "$ROOT" || exit 1
SO="$S/guard_swap"; FO="$S/guard_free"; LO="$S/guard_level"; MO="$S/guard_memfree"
rm -f "$SO" "$FO" "$LO" "$MO" "$S/guardrun.pid" "$S/guardrun.log"
echo 6000 > "$SO"; echo 9 > "$FO"; echo 1 > "$LO"; echo 50 > "$MO"
GUARD_SWAP_OVERRIDE_FILE="$SO" GUARD_FREE_OVERRIDE_FILE="$FO" GUARD_LEVEL_OVERRIDE_FILE="$LO" \
GUARD_MEMFREE_OVERRIDE_FILE="$MO" GUARD_INTERVAL_S=1 GUARD_SWAP_STILL_S=3 BENCH_RSS_CAP_KB=1000 \
  "$B/run.sh" guardrun "benchmark/state/guard_dummy.py" >/dev/null
P=$(cat "$S/guardrun.pid"); sleep 5
check "footprint over a 1000-KB guide value: INFORM only, still running" "$(state $P)" running
check "inform line logged" "$(grep -c 'INFORM: group footprint' "$S/guardrun.log")" 1
check "free disk 9 GB (< 20): disk INFORM logged, still running" "$(grep -c 'INFORM: free disk' "$S/guardrun.log")/$(state $P)" 1/running
echo 2 > "$LO"; sleep 3
check "level 2 (warn; the old rule paused on this): still running" "$(state $P)" running
echo 4 > "$LO"; sleep 3
check "level 4 (critical): PAUSED" "$(state $P)" paused
check "critical level raised a CRITICAL alert" "$(ls "$TI" | grep -c 'guardrun_CRITICAL_paused')" 1
echo 2 > "$LO"; sleep 3
check "level back to 2 with memory free 50%: RESUMED" "$(state $P)" running
echo 30 > "$MO"; sleep 3
check "memory free 30% (>= 25): still running" "$(state $P)" running
echo 20 > "$MO"; sleep 3
check "memory free 20% (< 25): PAUSED" "$(state $P)" paused
echo 30 > "$MO"; sleep 3
check "memory free 30% (hysteresis, < 35): still paused" "$(state $P)" paused
echo 40 > "$MO"; sleep 3
check "memory free 40% (>= 35): RESUMED" "$(state $P)" running
echo 6400 > "$SO"; sleep 3
check "swap 6000->6400 MB (+400 <= 512): still running" "$(state $P)" running
check "swap 6400 MB (> 6144): swap INFORM logged, no action" "$(grep -c 'INFORM: total swap' "$S/guardrun.log")" 1
echo 6600 > "$SO"; sleep 3
check "swap 6600 MB (+600 > 512 over baseline): PAUSED" "$(state $P)" paused
echo 6700 > "$SO"; sleep 2
check "swap still rising: still paused" "$(state $P)" paused
sleep 4
check "swap still for > 3 s (stale, high): RESUMED" "$(state $P)" running
echo x > "$MO"; sleep 3
check "unreadable memory metric: PAUSED (fail safe)" "$(state $P)" paused
echo 40 > "$MO"; sleep 3
check "metric readable again: RESUMED" "$(state $P)" running
echo 4 > "$FO"; sleep 3
check "free disk 4 GB (< 5): PAUSED" "$(state $P)" paused
echo 5 > "$FO"; sleep 3
check "free disk 5 GB (hysteresis, < 6): still paused" "$(state $P)" paused
echo 7 > "$FO"; sleep 3
check "free disk 7 GB: RESUMED" "$(state $P)" running
echo 1 > "$FO"; sleep 3
check "free disk 1 GB: paused" "$(state $P)" paused
check "CRITICAL alerts: level 4, unreadable metric, disk 1 GB" "$(ls "$TI" | grep -c 'guardrun_CRITICAL_paused')" 3
echo 9 > "$FO"; sleep 3
check "never killed: alive and resumed after every event" "$(state $P)" running
"$B/stop.sh" guardrun >/dev/null; sleep 2
check "stop.sh still stops a job by hand" "$(state $P)" dead
echo "3. run.sh validation"
out=$(BENCH_RSS_CAP_KB=abc "$B/run.sh" guardrun2 "benchmark/state/guard_dummy.py"); check "non-numeric guide value refused" "$?" 2
out=$(BENCH_RSS_CAP_KB=0 "$B/run.sh" guardrun2 "benchmark/state/guard_dummy.py"); check "zero guide value refused" "$?" 2
rm -f "$S/guardrun.pid" "$S/guardrun2.pid" "$S/guard_dummy.py" "$SO" "$FO" "$LO" "$MO" "$OV"

[ $FAIL -eq 0 ] && echo "GUARDS: ALL GREEN" || echo "GUARDS: FAILURES"
exit $FAIL
