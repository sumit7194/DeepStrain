#!/bin/bash
# In-session watcher: prints only NEW lines of a log (tail -n 0 -F) and one line when the PID is gone. Reports only;
# never signals the job.  Usage: coh2t_watch.sh <pid> <log>
PID=$1; LOG=$2
tail -n 0 -F "$LOG" 2>/dev/null &
T=$!
while kill -0 "$PID" 2>/dev/null; do sleep 5; done
sleep 1; kill "$T" 2>/dev/null
echo "WATCH: pid $PID is no longer running"
