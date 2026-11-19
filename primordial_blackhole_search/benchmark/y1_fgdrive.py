"""Light driver for Y1's foreground (imports nothing heavy, so it adds ~nothing to the watchdog's process-group RSS).

pycbc's waveform generation leaks memory with every distinct-length waveform (measured 2026-11-20: 30 random-mass
TaylorF2 waveforms take a process from 0.2 to 2.0 GB, numpy and default FFT backends alike), so injection windows are
built in short-lived children of 4 segments each, one at a time; then each realization is filtered in its own process,
which only reads the saved windows. Usage: benchmark/run.sh y1fg benchmark/y1_fgdrive.py
"""
import subprocess
import sys
from pathlib import Path

B = Path(__file__).resolve().parent
Y1 = str(B / "y1_bank.py")
STATE = B / "state" / "y1"
K, PARTS = 8, 6

for r in range(K):
    if (STATE / f"fg_{r}.json").exists():
        continue
    for p in range(PARTS):
        if not (STATE / f"win_{r}_p{p}.npz").exists():
            subprocess.run([sys.executable, "-u", Y1, "fgwin", str(r), str(p)], check=True)
    subprocess.run([sys.executable, "-u", Y1, "fg", "1", "one"], check=True)
print("Y1 FOREGROUND COMPLETE (driver)", flush=True)
