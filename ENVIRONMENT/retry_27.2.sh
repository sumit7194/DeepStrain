#!/bin/bash
# macOS 27.2 regression retry (Bridge rules 2026-11-17): start >=2 h from launch, at most 3 attempts >=1 h apart,
# one file at a time (fetch_test_segments.py is sequential), nice 10, stop if free disk < 5 GB. On success run the
# torch MPS/CPU spot check. Logs to ENVIRONMENT/retry_27.2.log.
cd "$(dirname "$0")/../primordial_blackhole_search" || exit 1
L=../ENVIRONMENT/retry_27.2.log
free_gb() { df -g / | awk 'NR==2{print $4}'; }
echo "$(date) launched; first attempt in 2 h" >> $L
sleep 7200
for a in 1 2 3; do
  [ "$(free_gb)" -lt 5 ] && { echo "$(date) STOP: free disk < 5 GB" >> $L; exit 1; }
  echo "$(date) attempt $a" >> $L
  nice -n 10 .venv/bin/python -u ../ENVIRONMENT/fetch_test_segments.py >> $L 2>&1
  n=$(ls data/noise/H1_*_4096.hdf5 2>/dev/null | wc -l | tr -d ' ')
  echo "$(date) attempt $a: $n/6 segments on disk" >> $L
  if [ "$n" -ge 6 ]; then
    OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 nice -n 10 .venv/bin/python -u ../ENVIRONMENT/regress_check.py ../ENVIRONMENT/regression_macos27.2_26B123.json >> $L 2>&1
    echo "$(date) REGRESSION DONE" >> $L; exit 0
  fi
  [ $a -lt 3 ] && sleep 3600
done
echo "$(date) ALL 3 ATTEMPTS FAILED" >> $L; exit 2
