#!/bin/bash
# Regression tests for the three hard limits changed in the 2026-11-25 fleet audit (each: fires when it should,
# stays quiet when it should, and a negative case that the old ABSOLUTE swap rule would have failed).
#   1. ../../.claude-coordination/blackhole_watchdog_9f21.sh  swap/compressor GROWTH over the arming baseline
#   2. benchmark/pressure_guard.sh                           swap GROWTH over baseline (+1536 MB default)
#   3. benchmark/run.sh                                      BENCH_RSS_CAP_KB override, default 4 GB, validated
# Uses only a tiny dummy process (a sleep loop under benchmark/state/), never a real job. ~1 min.
set -u
B="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(cd "$B/.." && pwd)"
W=/Users/sumit/Github/.claude-coordination/blackhole_watchdog_9f21.sh
S="$B/state"; mkdir -p "$S"
FAIL=0
ok()   { printf '  PASS  %s\n' "$1"; }
bad()  { printf '  FAIL  %s\n' "$1"; FAIL=1; }
check(){ if [ "$2" = "$3" ]; then ok "$1 ($2)"; else bad "$1: got '$2', want '$3'"; fi; }
swap_used() { sysctl -n vm.swapusage | awk '{for(i=1;i<=NF;i++) if($i=="used") {v=$(i+2); sub("M","",v); print int(v)}}'; }
printf 'import time\nwhile True: time.sleep(1)\n' > "$S/guard_dummy.py"
LVL=$(sysctl -n kern.memorystatus_vm_pressure_level)
[ "$LVL" -ge 2 ] && { echo "box is under memory pressure (level $LVL): the 'stays quiet' cases cannot be tested now"; exit 3; }

echo "1. watchdog decision rule (no process touched)"
check "swap 1000->1100, comp +100: old absolute rule KILLED this"   "$("$W" --decide 1000 1100 3000 3100)" OK
check "swap 300->900 (+600 > +512)"                                  "$("$W" --decide 300 900 3000 3000)" KILL
check "compressor 3000->4100 (+1100 > +1024)"                        "$("$W" --decide 300 300 3000 4100)" KILL
check "exactly +512 swap is not above the bar"                       "$("$W" --decide 300 812 3000 3000)" OK
echo "1b. watchdog live, on a dummy in this repo"
( cd "$ROOT" && exec python3 "$S/guard_dummy.py" ) & D=$!; sleep 1
WD_SWAP_GROW_MB=-100000 WD_INTERVAL_S=1 WD_LOG="$S/guard_wd.log" "$W" "$D" & WP=$!; sleep 3
if kill -0 "$D" 2>/dev/null; then bad "watchdog with a negative growth bar did not kill the dummy"; kill "$D"; else ok "watchdog fires and kills its own target"; fi
wait "$WP" 2>/dev/null
( cd "$ROOT" && exec python3 "$S/guard_dummy.py" ) & D=$!; sleep 1
WD_INTERVAL_S=1 WD_LOG="$S/guard_wd.log" "$W" "$D" & WP=$!; sleep 3
if kill -0 "$D" 2>/dev/null; then ok "watchdog at default bars leaves a healthy target alone (box swap $(swap_used) MB)"; else bad "watchdog killed a healthy target at default bars"; fi
kill "$D"; wait "$WP" 2>/dev/null
echo "1c. watchdog target validation (existing test)"
bash /Users/sumit/Github/.claude-coordination/blackhole_watchdog_test.sh | sed 's/^/    /' || FAIL=1

echo "2. pressure_guard"
run_guard() {  # $1 override MB or empty -> prints alive/dead after 3 s
  ( cd "$ROOT" && exec perl -MPOSIX -e 'setsid(); exec @ARGV' python3 "$S/guard_dummy.py" ) & local d=$!
  sleep 1; echo $d > "$S/guardtest.pid"
  if [ -n "$1" ]; then GUARD_SWAP_OVERRIDE_MB=$1 GUARD_INTERVAL_S=1 "$B/pressure_guard.sh" guardtest & else GUARD_INTERVAL_S=1 "$B/pressure_guard.sh" guardtest & fi
  local g=$!; sleep 3
  if kill -0 $d 2>/dev/null; then echo alive; kill $d; else echo dead; fi
  wait $g 2>/dev/null
}
BASE=$(swap_used)
check "swap baseline+2000 MB (> +1536)"                         "$(run_guard $((BASE + 2000)))" dead
check "swap baseline+1500 MB: old absolute rule TERMed this"    "$(run_guard $((BASE + 1500)))" alive
check "live swap reading, default bar"                          "$(run_guard "")" alive

echo "3. run.sh RSS cap"
cd "$ROOT" || exit 1
out=$(BENCH_RSS_CAP_KB=abc "$B/run.sh" guardrun "benchmark/state/guard_dummy.py"); rc=$?
check "non-numeric cap refused" "$rc" 2
out=$(BENCH_RSS_CAP_KB=0 "$B/run.sh" guardrun "benchmark/state/guard_dummy.py"); rc=$?
check "zero cap refused" "$rc" 2
rm -f "$S/guardrun.pid"
BENCH_RSS_CAP_KB=1000 "$B/run.sh" guardrun "benchmark/state/guard_dummy.py" >/dev/null
P=$(cat "$S/guardrun.pid"); sleep 12
if kill -0 "$P" 2>/dev/null; then bad "1000-KB cap did not TERM the dummy"; "$B/stop.sh" guardrun >/dev/null; else ok "1000-KB cap TERMs the dummy ($(grep -c 'WATCHDOG: rss' "$S/guardrun.log") watchdog line)"; fi
rm -f "$S/guardrun.pid"
"$B/run.sh" guardrun "benchmark/state/guard_dummy.py" | grep -q "RSS cap 4194304 KB" && ok "default cap is 4194304 KB" || bad "default cap not 4194304 KB"
P=$(cat "$S/guardrun.pid"); sleep 12
if kill -0 "$P" 2>/dev/null; then ok "default cap leaves a 10-MB dummy alone"; "$B/stop.sh" guardrun >/dev/null; else bad "default cap killed a small dummy"; fi
rm -f "$S/guardrun.pid" "$S/guardtest.pid" "$S/guard_dummy.py"

[ $FAIL -eq 0 ] && echo "GUARDS: ALL GREEN" || echo "GUARDS: FAILURES"
exit $FAIL
