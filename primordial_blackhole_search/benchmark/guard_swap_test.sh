#!/bin/bash
# The Bridge's three tests for the 2026-11-27 19:14 swap amendment of job_guard.sh. Usage: guard_swap_test.sh [guard]
# (default job_guard.sh; pass another copy to show a test CAN fail, e.g. the 13:28 guard from git or a mutant).
#   (i)   stale-high swap (3,360 MB, the L128 pause of 18:33) with good level/memory: runs, and resumes after a pause
#   (ii)  swap growth > 512 MB above the baseline pauses; while swap keeps rising it stays paused; once still it
#         resumes and the baseline resets
#   (iii) the level / memory-free / disk rule of 13:28 is unchanged (own guard instance at swap 1,000 MB)
# Checks process state ('T' = stopped), never the log. Tiny dummy process in its own session; test inbox only. ~1 min.
set -u
B="$(cd "$(dirname "$0")" && pwd)"; S="$B/state"; G="${1:-$B/job_guard.sh}"
T="$B/.guard_under_test.sh"; cp "$G" "$T"; chmod +x "$T"
TI="$S/guard_swap_test_inbox"; rm -rf "$TI"; mkdir -p "$TI"
SO="$S/gst_swap"; FO="$S/gst_free"; LO="$S/gst_level"; MO="$S/gst_memfree"
export GUARD_INBOX="$TI" GUARD_SWAP_OVERRIDE_FILE="$SO" GUARD_FREE_OVERRIDE_FILE="$FO" GUARD_LEVEL_OVERRIDE_FILE="$LO" \
  GUARD_MEMFREE_OVERRIDE_FILE="$MO" GUARD_INTERVAL_S=1 GUARD_SWAP_STILL_S=4 BENCH_RSS_CAP_KB=999999999
FAIL=0
check() { if [ "$2" = "$3" ]; then printf '  PASS  %s (%s)\n' "$1" "$2"; else printf '  FAIL  %s: got %s, want %s\n' "$1" "$2" "$3"; FAIL=1; fi; }
state() { local s; s=$(ps -o stat= -p "$1" 2>/dev/null) || { echo dead; return; }; case "$s" in *T*) echo paused;; *) echo running;; esac; }
printf 'import time\nwhile True: time.sleep(1)\n' > "$S/gst_dummy.py"
start() {  # $1 name; dummy in its own session (as run.sh does) so SIGSTOP of its group never touches this shell
  perl -MPOSIX -e 'setsid(); exec @ARGV' python3 "$S/gst_dummy.py" & echo $! > "$S/$1.pid"
  "$T" "$1" & GP=$!; sleep 4; P=$(cat "$S/$1.pid")
}
finish() { kill -TERM "$P" 2>/dev/null; wait "$GP" 2>/dev/null; rm -f "$S/$1.pid"; }

echo "(i) stale-high swap with good level and memory"
echo 3360 > "$SO"; echo 1 > "$LO"; echo 75 > "$MO"; echo 34 > "$FO"
start gst1
check "swap 3360 MB, level 1, memory free 75%: keeps running" "$(state $P)" running
echo 4 > "$LO"; sleep 3
check "level 4: PAUSED" "$(state $P)" paused
echo 2 > "$LO"; sleep 3
check "level 2, memory 75%, swap still 3360 MB (stale, not rising): RESUMED" "$(state $P)" running
echo "(ii) swap growth above the baseline (baseline reset to 3360 at that resume)"
echo 3800 > "$SO"; sleep 3
check "swap 3800 MB (+440 <= 512): still running" "$(state $P)" running
echo 3900 > "$SO"; sleep 3
check "swap 3900 MB (+540 > 512): PAUSED" "$(state $P)" paused
echo 4000 > "$SO"; sleep 2
check "swap still rising (4000 MB), level/memory good: stays PAUSED" "$(state $P)" paused
sleep 5
check "swap still at 4000 MB for > 4 s: RESUMED" "$(state $P)" running
echo 4400 > "$SO"; sleep 3
check "swap 4400 MB (+400 over the new 4000 baseline): still running" "$(state $P)" running
echo 4600 > "$SO"; sleep 3
check "swap 4600 MB (+600 over the new baseline): PAUSED" "$(state $P)" paused
finish gst1

echo "(iii) level / memory-free / disk rule unchanged (13:28 rule; swap held at 1000 MB)"
echo 1000 > "$SO"; echo 1 > "$LO"; echo 75 > "$MO"; echo 34 > "$FO"
start gst3
echo 2 > "$LO"; echo 34 > "$MO"; sleep 3
check "level 2 with memory free 34% (the 18:33 reading): still running" "$(state $P)" running
echo 3 > "$LO"; sleep 3
check "level 3: still running" "$(state $P)" running
echo 4 > "$LO"; sleep 3
check "level 4: PAUSED" "$(state $P)" paused
echo 3 > "$LO"; echo 75 > "$MO"; sleep 3
check "level 3 (hysteresis, > 2): still paused" "$(state $P)" paused
echo 2 > "$LO"; sleep 3
check "level 2, memory free 75%: RESUMED" "$(state $P)" running
echo 24 > "$MO"; sleep 3
check "memory free 24% (< 25): PAUSED" "$(state $P)" paused
echo 30 > "$MO"; sleep 3
check "memory free 30% (hysteresis, < 35): still paused" "$(state $P)" paused
echo 36 > "$MO"; sleep 3
check "memory free 36%: RESUMED" "$(state $P)" running
echo 4 > "$FO"; sleep 3
check "free disk 4 GB (< 5): PAUSED" "$(state $P)" paused
echo 5 > "$FO"; sleep 3
check "free disk 5 GB (hysteresis, < 6): still paused" "$(state $P)" paused
echo 7 > "$FO"; sleep 3
check "free disk 7 GB: RESUMED" "$(state $P)" running
finish gst3
rm -f "$T" "$S/gst_dummy.py" "$SO" "$FO" "$LO" "$MO"
[ $FAIL -eq 0 ] && echo "SWAP AMENDMENT TESTS: ALL GREEN" || echo "SWAP AMENDMENT TESTS: FAILURES"
exit $FAIL
