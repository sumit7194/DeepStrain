#!/usr/bin/env python
"""Milestone 20 (R2 v2, stage A): extract raw strain for the `ringdown`-package cross-check.

The proper FD coherent pipeline (`ringdown`, Isi/Farr) lives in .venv311 (Python 3.11) which has no
gwpy; this script runs in the main .venv (3.12, gwpy + GWOSC cache) and dumps raw 4 kHz strain around
each event to npz for 21_ringdown_crosscheck.py: 128 s per detector (long enough for ACF estimation
off-source of the ~0.1 s analysis segment).
"""
import sys
from pathlib import Path

import numpy as np
from gwpy.timeseries import TimeSeries

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rdlib

DATA = Path(__file__).resolve().parent.parent / "data"
SPAN = 128.0  # s of raw strain per event/detector (ACF needs plenty of off-source)
EVENTS = ["GW150914", "GW250114_082203"]


def main() -> None:
    for ev in EVENTS:
        gps = rdlib.event_gps(ev)
        out = DATA / f"20_strain_{ev}.npz"
        if out.exists():
            print(f"{ev}: cached ({out.name})")
            continue
        arrs = {}
        for det in ("H1", "L1"):
            for attempt in range(5):
                try:
                    ts = TimeSeries.fetch_open_data(det, gps - SPAN / 2, gps + SPAN / 2, cache=True)
                    break
                except Exception as e:
                    print(f"  {ev} {det} attempt {attempt+1}: {type(e).__name__}", flush=True)
            else:
                raise RuntimeError(f"could not fetch {ev} {det}")
            arrs[f"{det}_t0"] = float(ts.t0.value)
            arrs[f"{det}_fs"] = float(ts.sample_rate.value)
            arrs[f"{det}_h"] = ts.value.astype(np.float64)
            n_nan = int(np.isnan(ts.value).sum())
            print(f"  {ev} {det}: {float(ts.duration.value):.0f}s @ {arrs[f'{det}_fs']:.0f}Hz, NaNs={n_nan}", flush=True)
        np.savez(out, gps=gps, **arrs)
        print(f"{ev}: wrote {out.name}")


if __name__ == "__main__":
    main()
