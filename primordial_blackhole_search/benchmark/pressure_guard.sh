#!/bin/bash
# Superseded 2026-11-25 by job_guard.sh (fleet rule: no hard cap may end a job; pressure PAUSES + ALERTS, auto-resumes).
# Kept as a thin alias so older launch notes still work. Usage: benchmark/pressure_guard.sh <job name> [swap growth MB]
[ -n "${2:-}" ] && export GUARD_SWAP_GROW_MB=$2
exec "$(dirname "$0")/job_guard.sh" "$1"
