#!/bin/bash
# Regression tests for the guards under the fleet rule of 2026-11-25: NO hard cap may end a job. Pressure PAUSES
# (SIGSTOP) and ALERTS; recovery RESUMES (SIGCONT); RSS over the guide value is INFORM only; nothing ever kills.
#   1. ../../.claude-coordination/blackhole_watchdog_9f21.sh  pause/resume on swap or compressor GROWTH
#   2. benchmark/job_guard.sh (via run.sh)                    pause/resume on swap growth and on low disk; RSS inform
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
LVL=$(sysctl -n kern.memorystatus_vm_pressure_level)
[ "$LVL" -ge 2 ] && { echo "box under memory pressure (level $LVL): the resume cases cannot be tested now"; exit 3; }
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
SO="$S/guard_swap"; FO="$S/guard_free"; rm -f "$SO" "$FO" "$S/guardrun.pid" "$S/guardrun.log"
GUARD_SWAP_OVERRIDE_FILE="$SO" GUARD_FREE_OVERRIDE_FILE="$FO" GUARD_INTERVAL_S=1 BENCH_RSS_CAP_KB=1000 \
  "$B/run.sh" guardrun "benchmark/state/guard_dummy.py" >/dev/null
P=$(cat "$S/guardrun.pid"); sleep 5
BASE=$(sed -n 's/.*swap baseline \([0-9]*\)MB.*/\1/p' "$S/guardrun.log" | head -1)
check "footprint over a 1000-KB guide value: INFORM only, still running" "$(state $P)" running
check "inform line logged" "$(grep -c 'INFORM' "$S/guardrun.log")" 1
echo $((BASE + 1500)) > "$SO"; sleep 3
check "swap baseline+1500 (< +1536; the old absolute rule acted on this)" "$(state $P)" running
echo $((BASE + 2000)) > "$SO"; sleep 3
check "swap baseline+2000 (> +1536): PAUSED" "$(state $P)" paused
echo "$BASE" > "$SO"; sleep 3
check "swap back to baseline: RESUMED" "$(state $P)" running
echo 4 > "$FO"; sleep 3
check "free disk 4 GB (< 5): PAUSED" "$(state $P)" paused
echo 5 > "$FO"; sleep 3
check "free disk 5 GB (hysteresis, < 6): still paused" "$(state $P)" paused
echo 7 > "$FO"; sleep 3
check "free disk 7 GB: RESUMED" "$(state $P)" running
echo 1 > "$FO"; sleep 3
check "free disk 1 GB: paused" "$(state $P)" paused
check "critical disk raised a CRITICAL alert" "$(ls "$TI" | grep -c 'guardrun_CRITICAL_paused')" 1
echo 9 > "$FO"; sleep 3
check "never killed: alive and resumed after every event" "$(state $P)" running
"$B/stop.sh" guardrun >/dev/null; sleep 2
check "stop.sh still stops a job by hand" "$(state $P)" dead
echo "3. run.sh validation"
out=$(BENCH_RSS_CAP_KB=abc "$B/run.sh" guardrun2 "benchmark/state/guard_dummy.py"); check "non-numeric guide value refused" "$?" 2
out=$(BENCH_RSS_CAP_KB=0 "$B/run.sh" guardrun2 "benchmark/state/guard_dummy.py"); check "zero guide value refused" "$?" 2
rm -f "$S/guardrun.pid" "$S/guardrun2.pid" "$S/guard_dummy.py" "$SO" "$FO" "$OV"

[ $FAIL -eq 0 ] && echo "GUARDS: ALL GREEN" || echo "GUARDS: FAILURES"
exit $FAIL
