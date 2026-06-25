"""Step 13 — E2 (PLAN.md): harden the on-source null with an INDEPENDENT, different-time background.

05_background draws its noise-only segments from the SAME 512 s block as the event (independent
slices, but one PSD, adjacent in time -> N~160 limits the smallest p to ~1/160, and stationarity is
assumed). This rebuilds the background from blocks at several time OFFSETS from the event (hours away,
each whitened against its own PSD), pools them, and recomputes the on-source p-values. If the nulls
hold against this harder, genuinely-independent background, the non-detection is robust.

Run:  .venv/bin/python scripts/13_independent_bg.py [--event GW150914]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from gwosc.datasets import event_gps
from gwpy.timeseries import TimeSeries

from echolib import (
    BAND,
    DATA,
    DETECTORS,
    GW150914_DT_PRED,
    RESULTS,
    _longest_finite,
    comb_score,
    load_segments,
    progress,
    whiten_bp,
)

# offsets (s) from the event GPS for the off-source blocks (both signs; hours away -> independent,
# and clear of the other O1 events: LVT151012 +28d, GW151226 +103d from GW150914)
OFFSETS = [o * s for o in (1800, 3600, 7200, 14400, 28800, 57600) for s in (-1, 1)]
SEG_DUR, SPAN, EDGE = 3.0, 512.0, 8.0


def offset_segments(det: str, event: str, offset: float) -> list[np.ndarray]:
    """Whitened 3-s off-source segments from a 512-s block `offset` s from the event (cached)."""
    gps = event_gps(event) + offset
    cache = DATA / f"{event}_off{int(offset)}_{det}_{int(SPAN)}s.hdf5"
    try:
        if cache.exists():
            raw = TimeSeries.read(cache)
        else:
            raw = TimeSeries.fetch_open_data(det, gps - SPAN / 2, gps + SPAN / 2, cache=False)
            raw.write(cache, overwrite=True)
        finite = _longest_finite(raw)
        if float(finite.duration.value) < 64.0:     # too short after NaN-cropping for a stable whiten
            print(f"  [skip] {det} offset {offset:+.0f}s: only {float(finite.duration.value):.0f}s finite", flush=True)
            return []
        block = whiten_bp(finite, BAND)
    except Exception as e:
        print(f"  [skip] {det} offset {offset:+.0f}s: {type(e).__name__}", flush=True)
        return []
    fs = float(block.sample_rate.value); t0 = float(block.t0.value); x = block.value
    n = int(round(SEG_DUR * fs))
    segs, t = [], t0 + EDGE
    while t + SEG_DUR < t0 + (len(x) / fs) - EDGE:
        i = int(round((t - t0) * fs)); seg = x[i:i + n]
        if len(seg) == n:
            segs.append(seg)
        t += SEG_DUR
    return segs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", default="GW150914")
    args = ap.parse_args()
    event = args.event
    dt_pred = GW150914_DT_PRED if event == "GW150914" else 0.0579
    dt_grid = np.arange(max(0.03, dt_pred / 2), 0.5, 0.002)
    j = int(np.argmin(np.abs(dt_grid - dt_pred)))

    segs = {det: load_segments(event, det) for det in DETECTORS}
    fs = segs["H1"].fs

    def net(pair):
        return sum(comb_score(pair[d], fs, dt_grid) for d in DETECTORS)

    # --- on-source score (same as 05) ---
    on = net({d: segs[d].on for d in DETECTORS})
    on_max, on_pred = float(on.max()), float(on[j])

    # --- INDEPENDENT background: pool off-source segments from offset blocks ---
    off = {d: [] for d in DETECTORS}
    for k, offset in enumerate(OFFSETS):
        for d in DETECTORS:
            off[d].extend(offset_segments(d, event, offset))
        progress("13_independent_bg", k + 1, len(OFFSETS))
    n_bg = min(len(off[d]) for d in DETECTORS)
    print(f"{event}: independent background = {n_bg} pairs from {len(OFFSETS)} offset blocks "
          f"(vs {len(segs['H1'].off)} shared-block)", flush=True)

    bg_max, bg_pred = [], []
    for i in range(n_bg):
        s = net({d: off[d][i] for d in DETECTORS})
        bg_max.append(float(s.max())); bg_pred.append(float(s[j]))
    bg_max, bg_pred = np.array(bg_max), np.array(bg_pred)

    def pval(on_s, bg):
        return float((np.sum(bg >= on_s) + 1) / (len(bg) + 1))

    p_max, p_pred = pval(on_max, bg_max), pval(on_pred, bg_pred)

    # --- compare to the shared-block background (05) ---
    prev = {}
    f05 = RESULTS / "05_background.npy"
    if f05.exists():
        d = np.load(f05, allow_pickle=True).item()
        prev = {"p_max": float(d["p_max"]), "p_pred": float(d["p_pred"])}

    print(f"\n=== {event} ON-SOURCE vs INDEPENDENT background (n={n_bg}) ===")
    print(f"A (max over grid):    on={on_max:.3f}, p = {p_max:.3f}"
          + (f"   (shared-block 05: {prev['p_max']:.3f})" if prev else ""))
    print(f"B (at predicted Δt):  on={on_pred:.3f}, p = {p_pred:.3f}"
          + (f"   (shared-block 05: {prev['p_pred']:.3f})" if prev else ""))
    verdict = "NULL HOLDS (still consistent with noise)" if min(p_max, p_pred) > 0.05 else "NOT null -- investigate"
    print(f"verdict: {verdict}")

    import json
    (RESULTS / f"13_independent_bg_{event}.json").write_text(json.dumps(
        {"event": event, "n_bg": n_bg, "n_offset_blocks": len(OFFSETS), "on_max": on_max, "on_pred": on_pred,
         "p_max": p_max, "p_pred": p_pred, "shared_block_05": prev,
         "dt_pred": dt_pred, "offsets_s": OFFSETS}, indent=1))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, bg, on_s, p, lab in ((axes[0], bg_max, on_max, p_max, "A: max over Δt grid"),
                                 (axes[1], bg_pred, on_pred, p_pred, f"B: at predicted Δt={dt_pred:.3f}s")):
        ax.hist(bg, bins=30, color="steelblue", alpha=0.7, label=f"independent bg (n={n_bg})")
        ax.axvline(on_s, color="crimson", lw=2, label=f"on-source\np={p:.3f}")
        ax.set_title(lab); ax.set_xlabel("network comb score"); ax.legend()
    fig.suptitle(f"{event}: on-source vs INDEPENDENT (different-time) background — {verdict}")
    fig.tight_layout()
    out = RESULTS / f"13_independent_bg_{event}.png"
    fig.savefig(out, dpi=140)
    print(f"plot -> {out}")


if __name__ == "__main__":
    main()
