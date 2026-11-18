#!/bin/bash
# Launch one benchmark step detached, under the Bridge limits (PLAN.md): nice 15, 2 threads, process-tree RSS cap
# ~4 GB, free-disk floor 5 GB. Usage: benchmark/run.sh <name> <python args...>   e.g. benchmark/run.sh fetch benchmark/fetch.py fetch
# Writes state/<name>.pid and state/<name>.log. Stop with benchmark/stop.sh <name> (or touch state/PAUSE).
cd "$(dirname "$0")/.." || exit 1
NAME=$1; shift
S=benchmark/state; mkdir -p $S
[ -f $S/$NAME.pid ] && kill -0 "$(cat $S/$NAME.pid)" 2>/dev/null && { echo "$NAME already running (pid $(cat $S/$NAME.pid))"; exit 1; }
export OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
nohup perl -MPOSIX -e 'setsid(); exec @ARGV' nice -n 15 .venv/bin/python -u "$@" >> $S/$NAME.log 2>&1 &
PID=$!; echo $PID > $S/$NAME.pid
# watchdog: sum RSS over the job's process group; TERM the group (never STOP) above the cap or below the disk floor
nohup bash -c "
  sleep 5; PG=\$(ps -o pgid= -p $PID | tr -d ' '); PEAK=0
  while kill -0 $PID 2>/dev/null; do
    RSS=\$(ps -axo pgid=,rss= | awk -v g=\$PG '\$1==g{s+=\$2}END{print s+0}')
    FREE=\$(df -g / | awk 'NR==2{print \$4}')
    [ \"\$RSS\" -gt \"\$PEAK\" ] && PEAK=\$RSS
    if [ \"\$RSS\" -gt 4194304 ] || [ \"\$FREE\" -lt 5 ]; then
      echo \"\$(date) WATCHDOG: rss=\${RSS}KB free=\${FREE}GB -> TERM pgid \$PG\" >> $S/$NAME.log
      kill -TERM -- -\$PG; break
    fi
    sleep 5
  done
  echo \"\$(date) WATCHDOG: process-group RSS peak \$((PEAK/1024)) MB (sampled every 5 s)\" >> $S/$NAME.log" >/dev/null 2>&1 &
echo "started $NAME pid $PID"
