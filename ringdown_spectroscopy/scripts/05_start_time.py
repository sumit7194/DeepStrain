#!/usr/bin/env python
"""Milestone 5: the poisoned choice, reproduced.

Scan the ringdown start time t0 from peak+0 to peak+10 ms and watch the
one-tone (M, chi) answer drift — the start-time sensitivity at the heart of
the Isi/Farr vs Cotesta controversy.

Early t0: the short-lived overtone (and merger junk) contaminate a one-tone
model -> biased answer. Late t0: the tone has decayed below the noise ->
unstable answer. There is no safe single choice; that's the methodological
opening this project aims at.

Usage:
    python 05_start_time.py GW150914
    python 05_start_time.py GW250114_082203
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import rdlib

EVENT = sys.argv[1] if len(sys.argv) > 1 else "GW150914"
PLOTS = Path(__file__).resolve().parent.parent / "plots"
RESULTS = Path(__file__).resolve().parent.parent / "results"
PLOTS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

gps = rdlib.event_gps(EVENT)
series, peaks = {}, {}
for det in ("H1", "L1"):
    white = rdlib.fetch_whitened(det, gps, bandpass=False)
    peaks[det] = rdlib.find_peak(white.bandpass(*rdlib.BAND), gps)
    seg = white.crop(gps - 1, gps + 1)
    series[det] = (seg.times.value, seg.value)

peak = max(peaks.values())
kerr = rdlib.KerrMap(2, 2, 0)
offsets_ms = np.arange(0.0, 10.5, 1.0)
rows = []
for off in offsets_ms:
    t0 = peak + off / 1000.0
    fit = rdlib.fit_modes(series, t0, n_modes=1, seed=int(off), window=0.03)
    f_hz, tau_s = fit["modes"][0]["f"], fit["modes"][0]["tau"]
    try:
        mass, chi = kerr.mass_chi(f_hz, tau_s)
    except ValueError:
        mass = chi = np.nan
    rows.append(dict(offset_ms=float(off), f=f_hz, tau_ms=tau_s * 1e3, mass=mass, chi=chi))
    print(f"  t0=peak+{off:4.1f} ms: f={f_hz:6.1f} Hz tau={tau_s*1e3:5.2f} ms "
          f"-> M={mass:5.1f} chi={chi:4.2f}" if not np.isnan(mass) else
          f"  t0=peak+{off:4.1f} ms: f={f_hz:6.1f} Hz tau={tau_s*1e3:5.2f} ms -> not Kerr-invertible")

(RESULTS / f"05_start_time_{EVENT}.json").write_text(json.dumps(rows, indent=2))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
m = np.array([r["mass"] for r in rows])
c = np.array([r["chi"] for r in rows])
axes[0].plot(offsets_ms, m, "o-", color="steelblue")
axes[0].set_xlabel("ringdown start time t0 [ms after peak]")
axes[0].set_ylabel("inferred remnant mass [M_sun]")
axes[1].plot(offsets_ms, c, "o-", color="darkorange")
axes[1].set_xlabel("ringdown start time t0 [ms after peak]")
axes[1].set_ylabel("inferred remnant spin chi")
fig.suptitle(f"{EVENT}: the answer depends on WHEN you say ringing starts (one-tone fit)")
fig.tight_layout()
fig.savefig(PLOTS / f"05_start_time_{EVENT}.png", dpi=140)
print(f"  wrote {PLOTS / f'05_start_time_{EVENT}.png'}")
