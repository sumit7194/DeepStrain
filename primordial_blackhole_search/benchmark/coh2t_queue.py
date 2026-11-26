"""Step T evening queue (the Bridge's go, 2026-11-26): the build runs, serially, one process at a time.  [Claude Opus 5.5]
plan -> prepseries -> vseries -> k1precheck -> 2-box MPS pilot (+ the engine re-time, verification item 5).
Launched by run.sh, so job_guard.sh (pause-only, phys_footprint) watches the whole process group. A step that fails
stops the queue at that step boundary (a correctness gate, not a cap) and is reported; nothing is killed.
  benchmark/run.sh coh2t_queue benchmark/coh2t_queue.py
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

B = Path(__file__).resolve().parent
PY = sys.executable
DRV = str(B / "coh2t_drive.py")
ENV = dict(os.environ, OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] QUEUE {msg}", flush=True)


def step(name, args, env=None):
    t0 = time.time()
    log(f"START {name}: {' '.join(args)}")
    rc = subprocess.run([PY, "-u", *args], env=env or ENV).returncode
    log(f"{'DONE' if rc == 0 else 'FAILED'} {name} rc={rc} in {time.time() - t0:.0f} s")
    if rc != 0:
        log(f"STOPPED at {name}; the steps after it were not run")
        sys.exit(rc)


def pilot_boxes():
    """Two boxes for the pilot: the Mc nearest 0.35 (the largest grids) and nearest 0.50, among boxes with bg+twin."""
    pl = json.loads((B / "state" / "coh2t" / "plan.json").read_text())
    has = {c["tid"] for c in pl["candidates"] if c["kind"] == "bg" and c["valid"]}
    mcs = pl["mcs"]
    pick = []
    for target in (0.35, 0.50):
        pick.append(min(has, key=lambda t: abs(mcs[t] - target)))
    return pick


if __name__ == "__main__":
    step("plan", [DRV, "plan"])
    step("prepseries", [DRV, "prepseries"])
    step("vseries", [DRV, "vseries"])
    step("k1precheck", [DRV, "k1precheck"])
    boxes = pilot_boxes()
    log(f"pilot boxes {boxes}")
    step("retime", [str(B / "coh2c_local.py"), "tlocal", "4.0", "1", str(128 * 32768)], dict(ENV, COH2C_DEV="mps"))
    step("pilot", [DRV, "run", "128", "bg,twin,fg"], dict(ENV, COH2C_DEV="mps", COH2T_BOXES=",".join(map(str, boxes))))
    log("ALL DONE")
