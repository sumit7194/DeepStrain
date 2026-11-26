"""Step T, the L_max = 128 stage (Sumit's GO, staged per S2, relayed by the Bridge 2026-11-27).  [Claude Opus 5.5]

Serial, one step at a time, under run.sh / job_guard (true-memory footprint, pause-only; declared peak <= 4.3 GB).
A step that fails (a correctness gate or any error) stops the queue at that step boundary; nothing is killed.
Each `run` step is done in sub-runs of RECYCLE boxes, a fresh driver process each (2026-11-27, 41 boxes in: the
driver's footprint grew ~14 MB/box after every cache had saturated, in both the MPS pool and the CPU heap; a new
process per sub-run resets it. Boxes are independent and checkpointed, so outputs are unchanged.)
Order (registration §8; C14; C22):
  1 run 128 bg,twin,fg,null,wrong          main pass (box-major, checkpointed per box)
  2 calib 128                              C6 decoy target (write-once)
  3 run 128 dec_sg,dec_train,dec_drift     F2 decoys
  4 run 128 allen_off,allen_on,allen_fail  K1's second clause (C11)
  5 k2select ; k2run 128                   K2 (needs the real noise series, so before 6)
  6 dropseries                             the real noise series off the disk (hashes recorded and committed first)
  7 run 128 k1mc - r, r = 0..7             K1-(A), FIR generator (C22), one realisation at a time
  8 reduce 128 ; evaluate 128 ; gridstats  R_tune, bootstrap, every gate; the S2 stop rule is applied by humans
  python benchmark/coh2t_queue_L128.py     (via benchmark/run.sh coh2t_L128 benchmark/coh2t_queue_L128.py)
"""
import os
import subprocess
import sys
import time
from pathlib import Path

B = Path(__file__).resolve().parent
PY = sys.executable
DRV, RED = str(B / "coh2t_drive.py"), str(B / "coh2t_reduce.py")
ENV = dict(os.environ, OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
           COH2C_DEV="mps")


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] QUEUE {msg}", flush=True)


def step(name, args):
    t0 = time.time()
    log(f"START {name}: {' '.join(args)}")
    rc = subprocess.run([PY, "-u", *args], env=ENV).returncode
    log(f"{'DONE' if rc == 0 else 'FAILED'} {name} rc={rc} in {(time.time() - t0) / 3600:.2f} h")
    if rc != 0:
        log(f"STOPPED at {name}; later steps not run")
        sys.exit(rc)


RECYCLE = 50


def left(kinds, real="-"):
    return int(subprocess.run([PY, DRV, "remaining", "128", kinds] + ([] if real == "-" else [real]), env=ENV,
                              capture_output=True, text=True, check=True).stdout.strip())


def run_step(name, kinds, real="-"):
    n0, k = left(kinds, real), 0
    while (n := left(kinds, real)) > 0:
        step(f"{name}[{k}] ({n} boxes left)", [DRV, "run", "128", kinds, str(RECYCLE), real])
        assert left(kinds, real) < n, f"{name}: a sub-run made no progress"
        k += 1
    log(f"{name}: all {n0} boxes done in {k} sub-runs")


if __name__ == "__main__":
    run_step("main", "bg,twin,fg,null,wrong")
    step("calib", [DRV, "calib", "128"])
    run_step("decoys", "dec_sg,dec_train,dec_drift")
    run_step("allen", "allen_off,allen_on,allen_fail")
    step("k2select", [DRV, "k2select"])
    step("k2run", [DRV, "k2run", "128"])
    step("dropseries", [DRV, "dropseries"])
    for r in range(8):
        run_step(f"k1mc_r{r}", "k1mc", str(r))
    step("reduce", [RED, "reduce", "128"])
    step("evaluate", [RED, "evaluate", "128"])
    step("gridstats", [DRV, "gridstats"])
    log("L128 STAGE DONE")
