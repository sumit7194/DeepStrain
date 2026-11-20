"""Light driver for realbg_cascade.py: imports nothing heavy, runs each step in its own short-lived process (pycbc
retains waveform memory; MPS/torch memory is returned between steps). Resumable: every step skips finished outputs.

  benchmark/run.sh realbg benchmark/realbg_drive.py
"""
import json
import subprocess
import sys
from pathlib import Path

B = Path(__file__).resolve().parent
R = str(B / "realbg_cascade.py")
STATE = B / "state" / "realbg"
SEGS = json.loads((B / "manifest.json").read_text())["test"]


def step(*args):
    subprocess.run([sys.executable, "-u", R, *map(str, args)], check=True)


step("templates") if not (STATE / "templates.npz").exists() else None
for g in SEGS:
    for det in ("H1", "L1"):
        if not (STATE / f"gates_{det}_{g}.json").exists():
            step("gates", det, g)
for g in SEGS:
    for det in ("H1", "L1"):
        todo = [m for m in ("gated", "ungated", "twin") if not (STATE / f"bins_{m}_{det}_{g}.npz").exists()]
        if todo:
            step("series", det, g, ",".join(todo))
if not (STATE / "caught.json").exists():
    step("caught")
for g in SEGS:
    if not (STATE / f"fg_{g}.json").exists():
        step("fg", g)
if not (STATE / "g3.json").exists():
    step("g3")
step("combine")
print("REALBG CASCADE COMPLETE", flush=True)
