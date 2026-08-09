"""Does a consistency statistic beat the sum at deep FAR? Both sides of the trade, measured.

MOTIVATION. far_background_validation.py found the sum-statistic deep tail is glitch-dominated: the 8 events
that set 1/decade trace back to just 2 distinct H1 windows, and first-half vs second-half thresholds disagree
by 32-46%. far_glitch_anatomy.py found the loudest zero-lag "coincidence" is simply the loudest H1 glitch with
Livingston seeing nothing. A statistic no single detector can inflate should fix the background. The open
question was always whether it also throws away signal.

WHY THIS IS TESTABLE NOW. It was previously recorded as untestable, because measuring the signal side seemed
to need injections into O4b strain that far_deep.py purged. It does not: o4_sensitive_distance_rows_matched
and coinc_triple_rows_o4b already store PER-DETECTOR scores (sH1, sL1) for 4800 O4b injections. Background
comes from the far_scores cache, signal from those tables — no strain required.

  NOTE ON G2a. An earlier rung tested min/prod/max+min and found sum best. That was at a 4.6-yr background,
  i.e. entirely in the shallow regime, and it compared discrimination, not glitch robustness at depth. This
  is a different question: the deep tail, where the sum's problem actually lives.

THREE STATISTICS
  sum   sH + sL                     the incumbent
  min   min(sH, sL)                 consistency; a single glitch cannot inflate it
  veto  sum, but rejected when one detector contributes < 15% of the other (the anatomy finding, applied)

Both sides are reported at MATCHED false-alarm rate, which is the only fair comparison: a statistic with a
cleaner background gets a lower threshold, and that is exactly the advantage being measured.

CAVEAT. The injection segments overlap the background segments only partially (3/5 and 5/8), same era and
detectors. Stated, not hidden.

Run:  .venv/bin/python scripts/far_min_vs_sum.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pbh import config as C

CACHE = C.RESULTS_DIR / "far_scores"
YEAR_S = 3.156e7
KEEP = 20_000
ONE_SIDED = 0.15
FARS = (("1/month", 12.0), ("1/year", 1.0), ("1/decade", 0.1))
INJ = ["o4_sensitive_distance_rows_matched.parquet", "coinc_triple_rows_o4b.parquet"]


def stats_of(a, b):
    """The three candidate statistics, computed on aligned score pairs."""
    mn, mx = np.minimum(a, b), np.maximum(a, b)
    veto = np.where((mx <= 0) | (mn >= ONE_SIDED * mx), a + b, -np.inf)
    return {"sum": a + b, "min": mn, "veto": veto}


def background(H, L, keep=KEEP):
    """Top-`keep` of each statistic over N-1 distinct circular lags, plus the H1 windows behind the deep tail."""
    N = len(H)
    tops = {k: np.full(keep, -np.inf, dtype=np.float32) for k in ("sum", "min", "veto")}
    for k in range(1, N):
        for name, v in stats_of(H, np.roll(L, k)).items():
            t = tops[name]
            if v.max() > t[0]:
                t = np.partition(np.concatenate([t, v.astype(np.float32)]), -keep)[-keep:]
                t.sort()
                tops[name] = t
    return tops, N - 1


def ladder(top, n_lags, live_s):
    bg_yr = n_lags * live_s / YEAR_S
    out = {}
    for label, per_year in FARS:
        k = int(round(per_year * bg_yr))
        if 1 <= k <= len(top) and np.isfinite(top[-k]):
            out[label] = {"threshold": float(top[-k]), "k": k}
    return out, bg_yr


def snr50(snr, det):
    """SNR at 50% detection by linear interpolation on a binned efficiency curve; sensitive distance ~ 1/SNR50."""
    edges = np.arange(4, 41, 3.0)
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (snr >= lo) & (snr < hi)
        if m.sum() >= 15:
            xs.append((lo + hi) / 2); ys.append(det[m].mean())
    xs, ys = np.array(xs), np.array(ys)
    if len(xs) < 2 or ys.max() < 0.5 or ys.min() > 0.5:
        return None
    i = np.argmax(ys >= 0.5)
    if i == 0:
        return float(xs[0])
    x0, x1, y0, y1 = xs[i-1], xs[i], ys[i-1], ys[i]
    return float(x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0))


def main() -> None:
    segs = sorted(int(p.stem.split("_")[1]) for p in CACHE.glob("seg_*.npz"))
    per = [np.load(CACHE / f"seg_{g}.npz") for g in segs]
    Hs = [d["h"].astype(np.float32) for d in per]; Ls = [d["l"].astype(np.float32) for d in per]
    H, L = np.concatenate(Hs), np.concatenate(Ls)
    live = len(segs) * C.SEGMENT_LEN
    res = {"n_segments": len(segs), "n_windows": int(len(H))}

    print(f"background from {len(segs)} O4b segments, {len(H)} windows", flush=True)
    tops, n_lags = background(H, L)
    lads = {}
    for name, t in tops.items():
        lads[name], bg_yr = ladder(t, n_lags, live)
    res["background_years"] = bg_yr
    res["ladders"] = lads
    print(f"background = {n_lags} lags x {live/3600:.1f} h = {bg_yr:.1f} yr\n")
    print(f"{'FAR':>10} " + "".join(f"{n:>12}" for n in tops))
    for label, _ in FARS:
        row = "".join(f"{lads[n][label]['threshold']:>12.3f}" if label in lads[n] else f"{'--':>12}" for n in tops)
        print(f"{label:>10} " + row)

    # ---- background quality: stationarity (the sum's failure mode) --------------------------------------
    print("\nBACKGROUND STABILITY (first half vs second half of segments, same FAR)")
    half = len(segs) // 2
    res["stationarity"] = {}
    halves = {}
    for nm, sl in (("first", slice(0, half)), ("second", slice(half, len(segs)))):
        h = np.concatenate(Hs[sl]); l = np.concatenate(Ls[sl])
        t, nl_ = background(h, l, keep=5000)
        halves[nm] = {k: ladder(v, nl_, (sl.stop - sl.start) * C.SEGMENT_LEN)[0] for k, v in t.items()}
    for name in tops:
        spreads = []
        for label, _ in FARS:
            a = halves["first"][name].get(label); b = halves["second"][name].get(label)
            if a and b:
                spreads.append(100 * abs(a["threshold"] - b["threshold"]) / max(abs(a["threshold"]), abs(b["threshold"])))
        res["stationarity"][name] = {"max_half_to_half_pct": max(spreads) if spreads else None,
                                     "per_far_pct": spreads}
        print(f"  {name:>5}: half-to-half spread " + ", ".join(f"{s:.0f}%" for s in spreads)
              + f"   (worst {max(spreads):.0f}%)" if spreads else "")

    # ---- signal side -----------------------------------------------------------------------------------
    frames = []
    for f in INJ:
        p = C.RESULTS_DIR / f
        if not p.exists():
            continue
        d = pd.read_parquet(p)
        if "era" in d:
            d = d[d.era == "O4b"]
        frames.append(d[["chirp_mass", "target_snr", "sH1", "sL1"]])
    inj = pd.concat(frames, ignore_index=True)
    a, b = inj.sH1.to_numpy(), inj.sL1.to_numpy()
    sig = stats_of(a, b)
    res["n_injections"] = int(len(inj))
    print(f"\nSIGNAL RECOVERY on {len(inj)} O4b injections (matched FAR)")
    print(f"{'FAR':>10} " + "".join(f"{n+' det%':>12}" for n in tops) + "   |  SNR50 (lower = more reach)")
    res["signal"] = {}
    for label, _ in FARS:
        cells, s50 = [], []
        for name in tops:
            if label not in lads[name]:
                cells.append(f"{'--':>12}"); s50.append(None); continue
            det = sig[name] >= lads[name][label]["threshold"]
            cells.append(f"{100*det.mean():>11.1f}%")
            s50.append(snr50(inj.target_snr.to_numpy(), det))
            res["signal"].setdefault(label, {})[name] = {"det_frac": float(det.mean()),
                                                         "snr50": s50[-1]}
        print(f"{label:>10} " + "".join(cells) + "   |  "
              + ", ".join(f"{n} {('%.1f' % v) if v else 'n/a'}" for n, v in zip(tops, s50)))

    # ---- verdict ---------------------------------------------------------------------------------------
    print("\nVERDICT (sensitive distance ~ 1/SNR50, relative to sum at the same FAR)")
    res["verdict"] = {}
    for label in res["signal"]:
        base = res["signal"][label].get("sum", {}).get("snr50")
        if not base:
            continue
        row = {}
        for name in tops:
            v = res["signal"][label].get(name, {}).get("snr50")
            row[name] = round(base / v, 3) if v else None
        res["verdict"][label] = row
        print(f"  {label:>9}: " + "  ".join(f"{n} {('%.2fx' % row[n]) if row[n] else 'n/a'}" for n in tops))

    (C.RESULTS_DIR / "far_min_vs_sum.json").write_text(json.dumps(res, indent=2))
    print("\nwrote far_min_vs_sum.json")


if __name__ == "__main__":
    main()
