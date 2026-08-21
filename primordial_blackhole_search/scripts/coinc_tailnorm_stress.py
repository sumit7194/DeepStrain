"""Stress-test the tailnorm +5%: is it real, or is it fitted to the noise realisation it flattens?

THE CLAIM UNDER TEST. coinc_tailnorm found that mapping each detector's score to its own noise
tail-probability before summing gains +5.2% / +5.1% sensitive distance at 1/month / 1/year over the
incumbent `sum` -- and exactly 0.0% at 1/decade. Two things make that suspicious enough to withhold:

  (A) LEAKAGE. The tail map u_d(s) = -log10 P(S_d >= s) is estimated from the SAME 45,074 noise windows whose
      time-slides then form the background. A map fitted to one noise realisation can flatten that
      realisation's tail specifically, lowering the matched-FAR threshold without any real gain in
      discrimination. Held-out validation is the only way to tell -- and note this is the mechanism
      direction, not the overfitting direction, so a clean in-sample result proves nothing.

  (B) THE 1/decade ZERO IS THE WRONG SHAPE. If equalising tails helps, it should help MOST where tail
      asymmetry dominates -- the deep rungs -- not vanish there. A concrete alternative explanation:
      the empirical map is CENSORED. Past the loudest observed noise window it floors at
      u_max = log10(N) = 4.65, so every event beyond that maps to one value and the statistic degenerates
      to a constant. Our deepest rungs live in exactly that region (the zero-lag population test showed
      1/year and beyond sit past ALL zero-lag data). If so, tailnorm works only where the noise is
      genuinely sampled -- the same structural wall as everything else this week, arrived at from a new
      direction.

TESTS
  1. HELD-OUT NOISE: fit the tail map on a random half of SEGMENTS, build background + threshold on the
     other half. Segment-level split, not window-level, so within-segment correlation cannot leak.
  2. CENSORING: what fraction of background events at each rung sit at or above u_max, where the map carries
     no information? Directly tests explanation (B).
  3. SIGNIFICANCE: bootstrap the 2,400 injections; report the 90% CI on the tailnorm/sum ratio. +5% is
     meaningless without it.

PRE-REGISTERED: the gain survives only if the held-out ratio stays >1.02 AND its bootstrap CI excludes 1.0.
Otherwise this is recorded as an in-sample artefact and `sum` stands.

Run:  .venv/bin/python scripts/coinc_tailnorm_stress.py [--boot 500]
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
OUT = C.RESULTS_DIR / "coinc_tailnorm_stress.json"
YEAR_S = 3.156e7
WIN_SEC = 64.0
FARS = (("1/month", 12.0), ("1/year", 1.0), ("1/decade", 0.1))
KEEP = 200_000


def load_by_segment():
    segs = []
    for p in sorted(CACHE.glob("seg_*.npz"), key=lambda q: int(q.stem.split("_")[1])):
        d = np.load(p)
        segs.append((d["h"].astype(np.float64), d["l"].astype(np.float64)))
    return segs


def tail_map(noise):
    srt = np.sort(noise); n = len(srt)
    u_max = np.log10(n)

    def u(x):
        idx = np.searchsorted(srt, x, side="left")
        return -np.log10(np.maximum(n - idx, 1) / n)
    return u, float(u_max)


def bg_top(a, b, keep=KEEP):
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
        if 1 <= k <= len(top):
            out[label] = (float(top[-k]), k)
    return out, bg_yr


def snr50(snr, det):
    y = det.astype(float)
    if y.sum() < 5 or (1 - y).sum() < 5:
        return np.nan
    grid = np.linspace(np.percentile(snr, 1), np.percentile(snr, 99), 400)
    frac, keep = [], []
    for g in grid:
        m = (snr >= g - 2) & (snr <= g + 2)
        if m.sum() >= 10:
            frac.append(y[m].mean()); keep.append(g)
    if len(frac) < 5 or max(frac) < 0.5:
        return np.nan
    return float(np.interp(0.5, np.array(frac), np.array(keep)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=500)
    args = ap.parse_args()
    rng = np.random.default_rng(5)

    segs = load_by_segment()
    ns = len(segs)
    idx = rng.permutation(ns)
    fit_i, ev_i = idx[: ns // 2], idx[ns // 2:]
    Hf = np.concatenate([segs[i][0] for i in fit_i]); Lf = np.concatenate([segs[i][1] for i in fit_i])
    He = np.concatenate([segs[i][0] for i in ev_i]);  Le = np.concatenate([segs[i][1] for i in ev_i])
    print(f"segments {ns} -> fit {len(fit_i)} ({len(Hf)} windows) | eval {len(ev_i)} ({len(He)} windows)",
          flush=True)

    uH, uHmax = tail_map(Hf)                 # map fitted on the FIT half only
    uL, uLmax = tail_map(Lf)
    live_e = len(He) * WIN_SEC

    d = pd.read_parquet(INJ); d = d[d.era == "O4b"].copy()
    snr = d.target_snr.values
    sig = {"sum": d.sH1.values + d.sL1.values,
           "tailnorm": uH(d.sH1.values) + uL(d.sL1.values)}

    res = {"n_segments": ns, "fit_windows": len(Hf), "eval_windows": len(He),
           "u_max": {"H1": uHmax, "L1": uLmax}, "by_stat": {}, "censoring": {}}
    for name in ("sum", "tailnorm"):
        t0 = time.time()
        A, B = (He, Le) if name == "sum" else (uH(He), uL(Le))
        top, n_lags = bg_top(A, B)
        lad, bg_yr = ladder(top, n_lags, live_e)
        row = {"bg_years": bg_yr, "thresholds": {k: v[0] for k, v in lad.items()}, "snr50": {}}
        for lbl, (thr, k) in lad.items():
            row["snr50"][lbl] = snr50(snr, sig[name] >= thr)
        res["by_stat"][name] = row
        print(f"  {name:>9} ({time.time()-t0:4.0f}s, bg {bg_yr:7.1f} yr): " +
              "  ".join(f"{l} thr {lad[l][0]:7.3f} SNR50 {row['snr50'][l]:.1f}"
                        for l, _ in FARS if l in lad), flush=True)
        if name == "tailnorm":
            # (B) censoring: how much of the background at each rung sits where the map is saturated?
            for lbl, (thr, k) in lad.items():
                sel = top[-k:]
                res["censoring"][lbl] = {"threshold": thr, "u_ceiling_sum": uHmax + uLmax,
                                         "frac_at_ceiling": float(np.mean(sel >= uHmax + uLmax - 1e-9))}

    print("\nHELD-OUT sensitive distance, tailnorm / sum (>1 = tailnorm better)")
    res["ratio"] = {}
    for lbl, _ in FARS:
        a = res["by_stat"]["sum"]["snr50"].get(lbl); b = res["by_stat"]["tailnorm"]["snr50"].get(lbl)
        r = float(a / b) if (a and b and np.isfinite(a) and np.isfinite(b)) else None
        res["ratio"][lbl] = r
        print(f"  {lbl:>9}: {('%.3fx' % r) if r else 'n/a'}")

    print("\nCENSORING — fraction of the background at each rung sitting at the map's ceiling")
    for lbl, c in res["censoring"].items():
        print(f"  {lbl:>9}: thr {c['threshold']:.3f} vs ceiling {c['u_ceiling_sum']:.3f} -> "
              f"{100*c['frac_at_ceiling']:.1f}% censored")

    # (3) significance on the held-out thresholds
    print(f"\nBOOTSTRAP ({args.boot} resamples of the {len(snr)} injections)")
    res["boot"] = {}
    for lbl, _ in FARS:
        ts = res["by_stat"]["sum"]["thresholds"].get(lbl)
        tt = res["by_stat"]["tailnorm"]["thresholds"].get(lbl)
        if ts is None or tt is None:
            continue
        vals = []
        for _ in range(args.boot):
            j = rng.integers(0, len(snr), len(snr))
            a = snr50(snr[j], sig["sum"][j] >= ts)
            b = snr50(snr[j], sig["tailnorm"][j] >= tt)
            if a and b and np.isfinite(a) and np.isfinite(b):
                vals.append(a / b)
        if len(vals) < 50:
            continue
        v = np.array(vals)
        res["boot"][lbl] = {"median": float(np.median(v)),
                            "ci90": [float(np.percentile(v, 5)), float(np.percentile(v, 95))],
                            "p_gt_1": float((v > 1).mean())}
        b = res["boot"][lbl]
        print(f"  {lbl:>9}: {b['median']:.3f}x  90% CI [{b['ci90'][0]:.3f}, {b['ci90'][1]:.3f}]  "
              f"P(>1) = {b['p_gt_1']:.2f}")

    surv = [lbl for lbl, b in res["boot"].items()
            if (res["ratio"].get(lbl) or 0) > 1.02 and b["ci90"][0] > 1.0]
    res["survives"] = surv
    res["verdict"] = (f"HOLDS at {surv} — held-out gain >1.02 with CI excluding 1" if surv else
                      "IN-SAMPLE ARTEFACT — the gain does not survive held-out noise or is not significant; "
                      "`sum` stands")
    print(f"\nVERDICT: {res['verdict']}")
    OUT.write_text(json.dumps(res, indent=2))
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
