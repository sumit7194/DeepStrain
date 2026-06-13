#!/usr/bin/env python
"""Milestone 1: fetch public strain for an event, whiten it, and plot
(a) the full chirp and (b) the post-peak ringdown window.

Usage:
    python 01_fetch_data.py GW150914
    python 01_fetch_data.py GW250114_082203
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gwosc import datasets
from gwpy.timeseries import TimeSeries

EVENT = sys.argv[1] if len(sys.argv) > 1 else "GW150914"
DETECTORS = ("H1", "L1")
PLOTS = Path(__file__).resolve().parent.parent / "plots"
PLOTS.mkdir(exist_ok=True)

gps = datasets.event_gps(EVENT)
print(f"{EVENT}: GPS peak time = {gps}")

fig, axes = plt.subplots(2, len(DETECTORS), figsize=(7 * len(DETECTORS), 7), squeeze=False)

for col, det in enumerate(DETECTORS):
    print(f"  fetching {det} (64 s of strain around the event) ...")
    # 64 s so the whitening filter has enough data to estimate the noise spectrum
    strain = TimeSeries.fetch_open_data(det, gps - 32, gps + 32, cache=True)
    # 30-350 Hz: the band where the chirp and the ~250 Hz ringdown tones live
    white = strain.whiten(4, 2).bandpass(30, 350)

    # catalog GPS is rounded; find this detector's peak within +-0.1 s of it
    search = white.crop(gps - 0.1, gps + 0.1)
    peak = search.times.value[abs(search.value).argmax()]
    print(f"    {det} peak at GPS {peak:.4f} (catalog {gps})")

    # (a) the chirp: 0.25 s before peak to 0.05 s after
    chirp = white.crop(peak - 0.25, peak + 0.05)
    ax = axes[0][col]
    ax.plot(chirp.times.value - peak, chirp.value, lw=0.8)
    ax.set_title(f"{EVENT} — {det}: merger")
    ax.set_xlabel("time from peak [s]")
    ax.set_ylabel("whitened strain")
    ax.axvline(0, color="r", ls="--", lw=0.8, alpha=0.6)

    # (b) the ringdown window: the ~0.1 s right after the peak
    ring = white.crop(peak - 0.01, peak + 0.1)
    ax = axes[1][col]
    ax.plot(ring.times.value - peak, ring.value, lw=0.8, color="darkorange")
    ax.set_title(f"{det}: post-peak ringdown window")
    ax.set_xlabel("time from peak [s]")
    ax.set_ylabel("whitened strain")
    ax.axvline(0, color="r", ls="--", lw=0.8, alpha=0.6)

fig.tight_layout()
out = PLOTS / f"01_{EVENT}_ringdown.png"
fig.savefig(out, dpi=140)
print(f"  wrote {out}")
