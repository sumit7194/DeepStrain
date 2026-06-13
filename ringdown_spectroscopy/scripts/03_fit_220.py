#!/usr/bin/env python
"""Milestone 3: the classical baseline.

Fit ONE damped sinusoid (the l=m=2, n=0 fundamental tone) to the post-peak
whitened strain, jointly in H1+L1 (shared f, tau; per-detector amp/phase),
then invert (f, tau) -> (mass, spin) with the Kerr 220 map.

Start time: peak + 3 ms (late enough that the short-lived overtone has decayed
by ~9x, so a one-tone model is a fair description; the t0-dependence itself is
studied in 05_start_time.py).

Usage:
    python 03_fit_220.py GW150914
    python 03_fit_220.py GW250114_082203
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import rdlib

EVENT = sys.argv[1] if len(sys.argv) > 1 else "GW150914"
T0_OFFSET_MS = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
PLOTS = Path(__file__).resolve().parent.parent / "plots"
PLOTS.mkdir(exist_ok=True)

gps = rdlib.event_gps(EVENT)
print(f"{EVENT}: catalog GPS {gps}")

series, peaks = {}, {}
for det in ("H1", "L1"):
    white = rdlib.fetch_whitened(det, gps)
    peaks[det] = rdlib.find_peak(white, gps)
    seg = white.crop(gps - 1, gps + 1)
    series[det] = (seg.times.value, seg.value)
    print(f"  {det}: peak GPS {peaks[det]:.4f}")

# fit from the H1 peak epoch (per-detector phases absorb the light-travel offset)
t0 = max(peaks.values()) + T0_OFFSET_MS / 1000.0
fit = rdlib.fit_modes(series, t0, n_modes=1)
f_hz, tau_s = fit["modes"][0]["f"], fit["modes"][0]["tau"]
print(f"\n  fitted 220 tone: f = {f_hz:.1f} Hz, tau = {tau_s * 1e3:.2f} ms")

kerr = rdlib.KerrMap(2, 2, 0)
try:
    mass, chi = kerr.mass_chi(f_hz, tau_s)
    print(f"  => Kerr 220 inversion: M = {mass:.1f} M_sun (detector frame), chi = {chi:.2f}")
except ValueError as e:
    mass = chi = None
    print(f"  => not invertible as a Kerr 220 tone: {e}")

# ----- plot: data vs model in each detector
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for ax, det in zip(axes, ("H1", "L1")):
    t, v = series[det]
    m = (t >= t0 - 0.005) & (t <= t0 + 0.05)
    ax.plot((t[m] - t0) * 1e3, v[m], lw=0.9, label="whitened data", color="gray")
    p = fit["per_det"][det][0] | fit["modes"][0]
    model = rdlib.damped_sinusoids(t[m], t0, [p])
    ax.plot((t[m] - t0) * 1e3, model, lw=1.4, label="220 fit", color="crimson")
    ax.set_title(f"{EVENT} {det}: one-tone fit from peak+{T0_OFFSET_MS:.0f} ms")
    ax.set_xlabel("time from t0 [ms]")
    ax.set_ylabel("whitened strain")
    ax.legend(loc="upper right", fontsize=8)
fig.suptitle(
    f"f={f_hz:.1f} Hz, tau={tau_s*1e3:.2f} ms"
    + (f"  ->  M={mass:.1f} Msun, chi={chi:.2f}" if mass else "")
)
fig.tight_layout()
out = PLOTS / f"03_{EVENT}_fit220.png"
fig.savefig(out, dpi=140)
print(f"  wrote {out}")
