#!/bin/bash
# Live monitor for the Build-C VM run on alphaludo-l4.
# Usage:  watch -n 20 ./monitor_vm.sh      (refresh every 20s)
#    or:  ./monitor_vm.sh                   (one-shot)
# LOG defaults to the learned-coincidence run; override: LOG=coinc_far ./monitor_vm.sh
LOG="${LOG:-learned_full}"
gcloud compute ssh alphaludo-l4 --zone=us-east1-d --command="
f=~/deepstrain/\$(echo $LOG).log
echo '──────── DeepStrain VM ('$LOG') ────────'
if grep -qE '_DONE' \$f 2>/dev/null; then
  echo 'STATUS: ✅ DONE'
elif pgrep -f coinc_ >/dev/null; then
  echo \"STATUS: 🔄 running (PID \$(pgrep -f coinc_ | head -1))\"
else
  echo 'STATUS: ⏹  not running'
fi
last=\$(grep -oE 'seg [0-9]+/[0-9]+' \$f 2>/dev/null | tail -1)
phase=\$(grep -E 'pooled|training|FAR sweep|coincident sensitive' \$f 2>/dev/null | tail -1)
[ -n \"\$last\" ] && echo \"progress: \$last   \$phase\"
[ -n \"\$phase\" ] && echo \"phase: \$phase\"
echo '── resources ──'
printf 'cpu load (of 8): '; uptime | grep -oP 'load average: \K[0-9.]+'
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | sed 's/^/gpu: /'
free -g | awk '/Mem/{print \"ram: \"\$3\"/\"\$2\" GB\"}'
echo '── latest log ──'
grep -vE 'pkg-config|Warning|warn' \$f 2>/dev/null | tail -3
" 2>&1 | grep -vE "Warning: Permanently added|Updating project ssh|Waiting for|Writing|WARNING: The private"
