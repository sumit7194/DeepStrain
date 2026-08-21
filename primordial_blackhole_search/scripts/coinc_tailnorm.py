"""Is `sum` the right coincidence statistic when the detectors' NOISE TAILS differ by 2x?

WHERE THIS CAME FROM. bridge (TheBridge/G3) reported a statistic whose GAIN varied 2.1x along a comparison
axis, manufacturing structure with no physics in it, and — the part that matters — 16x more data bought
essentially nothing, because a systematic gain variation does not average down at any N. Checking our own
exposure turned up something we had not looked at:

    gain ratio  L1/H1 = 1.10 (O3a), 0.97 (O4b)      -- small, and `sum` is near-optimal for it
    NOISE TAIL  q99.9 ratio 1.58x, max ratio 1.97x  -- H1 12.53 vs L1 6.37
                Hill index (top 1%) H1 3.84 vs L1 5.57 -- H1's tail is genuinely HEAVIER, not just louder

For an unweighted `sum`, the detector with the heavier noise tail dominates the false-alarm rate REGARDLESS
of its gain. So the asymmetry that could make `sum` quietly wrong here is ~2x, not the ~1.1x we had been
reassuring ourselves about. G2a and the L2 audit both tested `sum` vs `min`/`veto` and kept `sum` — but
neither tested a statistic that EQUALISES THE TAILS, so the question was never actually asked.

THE CANDIDATE. Map each detector's score to its own noise tail-probability before combining:

    u_d(s) = -log10( P_noise,d(S_d >= s) )      estimated from that detector's own 45,074 noise windows
    tailnorm = u_H1 + u_L1

By construction each detector contributes equally at equal RARITY rather than at equal SCORE, so a heavier
tail can no longer dominate. This is the standard move behind rank/quantile-combined detection statistics; the
question is whether it buys anything HERE, where the imbalance is real but modest.

    sum       = sH1 + sL1                       (incumbent)
    tailnorm  = u_H1 + u_L1                     (equalises noise tails)
    gainnorm  = sH1/g_H1 + sL1/g_L1             (equalises SIGNAL response — the axis bridge flagged)
    min       = min(sH1, sL1)                   (control; G2a and L2 both found it loses reach)

HONEST PRIORS. The L2 audit already showed `min` costs 3-4% of sensitive distance while halving background
instability, so a pure consistency cut is not the answer. `tailnorm` is a different bet: it keeps both
detectors' evidence but re-weights by rarity. If it does NOT win, that is a clean negative and `sum` is
vindicated against the specific objection rather than merely untested.

METHOD. Background from the same 45,073 distinct-lag time-slide construction as far_deep (so FARs are matched
to the published ladder); signal from the 4,800 retained O4b injections with per-detector scores. Compare
sensitive-distance fraction at matched FAR. The tail map is fitted on NOISE ONLY, so it cannot see the
injections and cannot leak.

Run:  .venv/bin/python scripts/coinc_tailnorm.py [--reps 200]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pbh import config as C

CACHE = C.RESULTS_DIR / "far_scores"
INJ = C.RESULTS_DIR / "o4_sensitive_distance_rows_matched.parquet"
OUT = C.RESULTS_DIR / "coinc_tailnorm.json"
YEAR_S = 3.156e7
WIN_SEC = 64.0
FARS = (("1/month", 12.0), ("1/year", 1.0), ("1/decade", 0.1))
KEEP = 200_000


def load_noise():
    H, L = [], []
    for p in sorted(CACHE.glob("seg_*.npz"), key=lambda q: int(q.stem.split("_")[1])):
        d = np.load(p); H.append(d["h"].astype(np.float64)); L.append(d["l"].astype(np.float64))
    return np.concatenate(H), np.concatenate(L)


def tail_map(noise):
    """u(s) = -log10 P(S >= s), from this detector's OWN noise windows. Monotone, so it cannot reorder a
    single detector's events -- it only changes how the two detectors are weighted against each other."""
    srt = np.sort(noise)
    n = len(srt)

    def u(x):
        # rank from the right; floor at 1 event so the map stays finite past the observed maximum
        idx = np.searchsorted(srt, x, side="left")
        surv = np.maximum(n - idx, 1) / n
        return -np.log10(surv)
    return u


def background_top(a, b, keep=KEEP):
    N = len(a)
    keep = int(min(keep, N * (N - 1)))
    top = np.full(keep, -np.inf)
    for k in range(1, N):
        v = a + np.roll(b, k)
        if v.max() > top[0]:
            top = np.partition(np.concatenate([top, v]), -keep)[-keep:]
            top.sort()
    return top, N - 1


def ladder(top, n_lags, live_s):
    bg_yr = n_lags * live_s / YEAR_S
    out = {}
    for label, per_year in FARS:
        k = int(round(per_year * bg_yr))
        if k > len(top):
            print(f"    !! {label} needs rank {k} but only {len(top)} kept -- RAISE keep", flush=True)
            continue
        if k >= 1:
            out[label] = float(top[-k])
    return out, bg_yr


def snr50(snr, detected):
    """SNR at 50% detection by logistic fit; sensitive distance ~ 1/SNR50."""
    ok = np.isfinite(snr) & np.isfinite(detected)
    s, y = snr[ok], detected[ok].astype(float)
    if y.sum() < 5 or (1 - y).sum() < 5:
        return np.nan
    lo, hi = np.percentile(s, 1), np.percentile(s, 99)
    grid = np.linspace(lo, hi, 400)
    frac = np.array([y[(s >= g - 2) & (s <= g + 2)].mean() if ((s >= g - 2) & (s <= g + 2)).sum() >= 10
                     else np.nan for g in grid])
    good = np.isfinite(frac)
    if good.sum() < 5 or np.nanmax(frac) < 0.5:
        return np.nan
    return float(np.interp(0.5, frac[good], grid[good]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    args = ap.parse_args()

    H, L = load_noise()
    N = len(H)
    live = N * WIN_SEC
    print(f"noise: {N} windows/detector | livetime {live/3600:.1f} h", flush=True)
    uH, uL = tail_map(H), tail_map(L)

    d = pd.read_parquet(INJ)
    d = d[d.era == "O4b"].copy()
    d["snr_net"] = d.target_snr
    gH = 0.4735; gL = 0.4613                       # measured O4b gains (planted SNR -> score)
    print(f"injections: {len(d)} O4b | gains H1 {gH:.4f} L1 {gL:.4f}", flush=True)

    stats = {
        "sum":      (lambda a, b: a + b,                 lambda a, b: a + b),
        "tailnorm": (lambda a, b: uH(a) + uL(b),         lambda a, b: uH(a) + uL(b)),
        "gainnorm": (lambda a, b: a / gH + b / gL,       lambda a, b: a / gH + b / gL),
        "min":      (lambda a, b: np.minimum(a, b),      lambda a, b: np.minimum(a, b)),
    }

    res = {"n_windows": N, "livetime_h": live / 3600, "n_inj": int(len(d)),
           "gains": {"H1": gH, "L1": gL}, "by_stat": {}}
    for name, (fn_bg, fn_sig) in stats.items():
        t0 = time.time()
        a = fn_bg(H, np.zeros_like(L)) if False else None      # keep the lag structure below
        # background must apply the statistic AFTER the slide, so transform each detector then slide
        if name == "tailnorm":
            A, B = uH(H), uL(L)
        elif name == "gainnorm":
            A, B = H / gH, L / gL
        else:
            A, B = H, L
        comb = (np.minimum if name == "min" else None)
        if name == "min":
            topv = np.full(KEEP, -np.inf)
            for k in range(1, N):
                v = np.minimum(A, np.roll(B, k))
                if v.max() > topv[0]:
                    topv = np.partition(np.concatenate([topv, v]), -KEEP)[-KEEP:]
                    topv.sort()
            top, n_lags = topv, N - 1
        else:
            top, n_lags = background_top(A, B)
        lad, bg_yr = ladder(top, n_lags, live)
        sig = fn_sig(d.sH1.values, d.sL1.values)
        row = {"bg_years": bg_yr, "thresholds": lad, "snr50": {}, "det_frac": {}}
        for lbl, thr in lad.items():
            det = sig >= thr
            row["det_frac"][lbl] = float(det.mean())
            row["snr50"][lbl] = snr50(d.snr_net.values, det)
        res["by_stat"][name] = row
        print(f"  {name:>9} ({time.time()-t0:5.0f}s, bg {bg_yr:8.1f} yr): " +
              "  ".join(f"{l} thr {lad.get(l, float('nan')):7.3f} det {row['det_frac'].get(l, float('nan')):.3f} "
                        f"SNR50 {row['snr50'].get(l, float('nan')):.1f}" for l, _ in FARS), flush=True)

    print("\nSENSITIVE DISTANCE relative to `sum` at matched FAR (1/SNR50; >1 means better reach)")
    res["verdict_table"] = {}
    for lbl, _ in FARS:
        base = res["by_stat"]["sum"]["snr50"].get(lbl)
        row = {}
        for name in stats:
            v = res["by_stat"][name]["snr50"].get(lbl)
            row[name] = float(base / v) if (base and v and np.isfinite(v)) else None
        res["verdict_table"][lbl] = row
        print(f"  {lbl:>9}: " + "  ".join(f"{k} {('%.3fx' % v) if v else '  n/a'}" for k, v in row.items()))

    best = {}
    for lbl, row in res["verdict_table"].items():
        cand = {k: v for k, v in row.items() if v and k != "sum"}
        if cand:
            best[lbl] = max(cand, key=cand.get)
    res["best_non_sum"] = best
    gains = [v for row in res["verdict_table"].values() for k, v in row.items()
             if v and k == "tailnorm"]
    res["tailnorm_mean_gain"] = float(np.mean(gains)) if gains else None
    res["verdict"] = ("tailnorm BEATS sum — the 2x noise-tail asymmetry was costing us reach"
                      if gains and np.mean(gains) > 1.02 else
                      "sum stands — equalising the noise tails buys nothing at OUR asymmetry, so `sum` is "
                      "vindicated against the specific objection rather than merely untested")
    print(f"\nVERDICT: {res['verdict']}")
    OUT.write_text(json.dumps(res, indent=2))
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
