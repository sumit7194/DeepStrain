"""Step 01 — fetch strain around a cataloged merger and sanity-check the toolchain.

Downloads public strain from GWOSC for both LIGO detectors around an event,
whitens + bandpasses it, and plots (a) the famous chirp and (b) the post-ringdown
segment that the echo search actually cares about.

Usage:
    python scripts/01_fetch_data.py              # defaults to GW150914
    python scripts/01_fetch_data.py GW170814     # any cataloged event name
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gwosc.datasets import event_gps
from gwpy.timeseries import TimeSeries

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

# How much data to pull around the event. 32 s gives the whitener enough context
# and contains every plausible echo train (predicted spacing is ~0.1-0.3 s).
WINDOW = 16  # seconds on each side of merger

DETECTORS = ("H1", "L1")


def fetch(event: str) -> dict[str, TimeSeries]:
    """Download (or load cached) strain for each detector around `event`."""
    gps = event_gps(event)
    print(f"{event}: GPS {gps}")
    series = {}
    for det in DETECTORS:
        cache = DATA / f"{event}_{det}.hdf5"
        if cache.exists():
            ts = TimeSeries.read(cache)
            print(f"  {det}: loaded cache {cache.name}")
        else:
            print(f"  {det}: downloading {2 * WINDOW}s from GWOSC ...")
            ts = TimeSeries.fetch_open_data(det, gps - WINDOW, gps + WINDOW, cache=False)
            ts.write(cache, overwrite=True)
            print(f"  {det}: cached -> {cache.name}")
        series[det] = ts
    return series


def whiten(ts: TimeSeries) -> TimeSeries:
    """Standard hygiene: whiten against the segment's own PSD, then bandpass.

    Whitening flattens the detector's coloured noise so every frequency counts
    equally; the bandpass keeps the audio band where BBH signals (and any echoes
    of them) live.
    """
    white = ts.whiten(4, 2)  # 4 s FFT, 2 s overlap
    return white.bandpass(30, 350)


def main() -> None:
    event = sys.argv[1] if len(sys.argv) > 1 else "GW150914"
    DATA.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)

    gps = event_gps(event)
    series = fetch(event)

    fig, axes = plt.subplots(len(DETECTORS), 2, figsize=(14, 6), sharex="col")
    for row, det in enumerate(DETECTORS):
        w = whiten(series[det])

        # Left: the merger itself (+-0.2 s) — the sanity check that the
        # pipeline reproduces the most famous waveform on Earth.
        chirp = w.crop(gps - 0.2, gps + 0.2)
        axes[row][0].plot(chirp.times.value - gps, chirp.value, lw=0.8)
        axes[row][0].set_ylabel(f"{det}\nwhitened strain")

        # Right: the post-ringdown window (0.05 s to 3 s after merger) — where
        # an echo train at ~0.1-0.3 s spacing would live. By eye this should be
        # pure noise; the whole project is about testing that statistically.
        post = w.crop(gps + 0.05, gps + 3.0)
        axes[row][1].plot(post.times.value - gps, post.value, lw=0.5)

    axes[0][0].set_title(f"{event} merger (the sanity check)")
    axes[0][1].set_title("post-ringdown segment (the echo hunting ground)")
    for ax in axes[-1]:
        ax.set_xlabel("time since merger [s]")
    fig.tight_layout()
    out = RESULTS / f"{event}_overview.png"
    fig.savefig(out, dpi=140)
    print(f"plot -> {out}")


if __name__ == "__main__":
    main()
