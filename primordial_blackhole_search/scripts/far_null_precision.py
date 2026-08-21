"""Have we been quoting the wrong number? Threshold-at-fixed-FAR vs FAR-of-the-observed-event.

THE ARGUMENT. Two measurements this week say the deep-FAR ladder saturates:
  * far_effective_n: the threshold at a fixed FAR barely improves with data -- sigma ~ n^-0.10, consistent
    with ZERO at two of three rungs. Four times the data does not halve it.
  * far_zerolag_population: 3 of 4 rungs sit BEYOND every zero-lag sample we have, so the assumption they
    rest on cannot be verified where they live, and no amount of sliding fixes that.

Both are properties of the SAME quantity -- an extreme quantile of the coincidence distribution, estimated by
extrapolating into a sparse, glitch-dominated tail. But that quantity is not what a null result actually
needs. **A null is a statement about the event we OBSERVED**, and the natural statistic is the FAR of the
loudest zero-lag event: "the loudest thing in our data happens by chance 11.5 times a year".

WHY THAT SHOULD BE BETTER DETERMINED, and it is not merely a re-parameterisation. The threshold at 1/decade
asks "what value is exceeded 0.1 times per year?" -- a rank-412 order statistic out of 2e9 pairs, set by a
handful of glitchy segments. The FAR of a FIXED value asks "how often is 11.295 exceeded?" -- a COUNT in a
region where we measured 47,500 background events. One extrapolates to where the data is thinnest; the other
interpolates where it is thickest. Both scale as N^2, so the ratio that forms the FAR should converge even
though the quantile does not.

    threshold(FAR)   : invert the tail at a fixed rate      -> sparse, extrapolated, sigma ~ n^-0.10
    FAR(statistic)   : count exceedances of a fixed value   -> dense, interpolated, sigma ~ ?

PRE-REGISTERED (fixed before any number was produced):
  * primary: alpha_FAR in sigma(log10 FAR) ~ n^-alpha, measured on the SAME subsets that gave the threshold
    its alpha, so the comparison is like-for-like and no sample-size or seed difference can explain a gap
  * alpha_FAR > 0.35 while alpha_thr < 0.20  => the FAR of the observed event genuinely converges where the
    threshold does not, and our null should be re-quoted
  * both flat                                => the saturation is a property of the deep tail itself, not of
    how we choose to summarise it. That is the outcome that would REFUTE this idea, and it is a real result:
    it would mean no summary statistic escapes the glitch limit.
  * a caveat that must survive to the write-up: FAR(S0) is evaluated at S0 = 11.295, our observed loudest
    zero-lag. That value is IN the data, so the comparison is honest for THIS null but does not license
    quoting a well-determined FAR at an arbitrary deeper value -- the sparse-tail problem returns as soon as
    S0 moves past where background events are plentiful. Reported as FAR(S0) vs S0 to make the limit visible.

Run:  .venv/bin/python scripts/far_null_precision.py [--reps 40]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pbh import config as C

CACHE = C.RESULTS_DIR / "far_scores"
OUT = C.RESULTS_DIR / "far_null_precision.json"
YEAR_S = 3.156e7
WIN_SEC = 64.0
SIZES = (40, 80, 160, 240, 320)
FIT_MAX_N = 240
FARS = (("1/month", 12.0), ("1/year", 1.0), ("1/decade", 0.1))
ZERO_LAG_MAX = 11.295368194580078          # our observed loudest zero-lag coincidence
PROBE_S0 = (7.5, 9.5, ZERO_LAG_MAX, 13.0)  # sweep S0 to expose where interpolation turns into extrapolation


def load_segments():
    Hs, Ls = [], []
    for p in sorted(CACHE.glob("seg_*.npz"), key=lambda q: int(q.stem.split("_")[1])):
        d = np.load(p); Hs.append(d["h"].astype(np.float32)); Ls.append(d["l"].astype(np.float32))
    return Hs, Ls


def measure(H, L):
    """One sweep gives BOTH summaries: thresholds at fixed FAR, and FAR at fixed statistic values."""
    N = len(H)
    bg_yr = (N - 1) * N * WIN_SEC / YEAR_S
    need = max(int(round(pyr * bg_yr)) for _, pyr in FARS)
    keep = int(min(max(4 * need, 5_000), N * (N - 1)))
    top = np.full(keep, -np.inf, dtype=np.float32)
    counts = np.zeros(len(PROBE_S0), dtype=np.int64)
    s0 = np.array(PROBE_S0, dtype=np.float32)
    for k in range(1, N):
        v = H + np.roll(L, k)
        counts += (v[:, None] >= s0[None, :]).sum(axis=0)
        if v.max() > top[0]:
            top = np.partition(np.concatenate([top, v]), -keep)[-keep:]
            top.sort()
    thr = {}
    for label, per_year in FARS:
        k = int(round(per_year * bg_yr))
        if 1 <= k <= len(top):
            thr[label] = float(top[-k])
    far = {f"{s:.3f}": (float(c) / bg_yr if bg_yr > 0 else float("nan"))
           for s, c in zip(PROBE_S0, counts)}
    return thr, far, bg_yr, {f"{s:.3f}": int(c) for s, c in zip(PROBE_S0, counts)}


def fit(xs, ys):
    x, y = np.log(np.array(xs, float)), np.log(np.array(ys, float))
    A = np.vstack([x, np.ones_like(x)]).T
    m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(m)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=40)
    args = ap.parse_args()

    Hs, Ls = load_segments()
    pool = len(Hs)
    rng = np.random.default_rng(101)
    print(f"pool {pool} segments | sizes {SIZES} | {args.reps} subsets each", flush=True)
    print(f"probing FAR at S0 = {PROBE_S0}  (zero-lag max = {ZERO_LAG_MAX:.3f})\n", flush=True)

    rows = []
    for n in SIZES:
        t0 = time.time()
        T = {lbl: [] for lbl, _ in FARS}
        F = {f"{s:.3f}": [] for s in PROBE_S0}
        Cnt = {f"{s:.3f}": [] for s in PROBE_S0}
        for _ in range(args.reps):
            sel = rng.choice(pool, n, replace=False)
            thr, far, _, cnt = measure(np.concatenate([Hs[i] for i in sel]),
                                       np.concatenate([Ls[i] for i in sel]))
            for k, v in thr.items():
                T[k].append(v)
            for k, v in far.items():
                F[k].append(v); Cnt[k].append(cnt[k])
        fpc = np.sqrt(max(1e-9, 1.0 - n / pool))
        for lbl, _ in FARS:
            if len(T[lbl]) >= 3:
                a = np.array(T[lbl])
                rows.append({"n": n, "kind": "threshold", "name": lbl, "mean": float(a.mean()),
                             "sd": float(a.std(ddof=1)) / fpc,
                             "rel_sd": float(a.std(ddof=1) / fpc / abs(a.mean()))})
        for k, v in F.items():
            a = np.array(v)
            if (a > 0).sum() < 3:
                continue
            lg = np.log10(a[a > 0])
            # relative spread of a RATE is naturally measured in dex; convert to a fractional equivalent
            rows.append({"n": n, "kind": "far_at_S0", "name": k, "mean": float(a.mean()),
                         "sd_dex": float(lg.std(ddof=1)) / fpc,
                         "rel_sd": float(10 ** (lg.std(ddof=1) / fpc) - 1),
                         "mean_count": float(np.mean(Cnt[k]))})
        print(f"  n={n:4d} ({time.time()-t0:5.0f}s, fpc {fpc:.3f})", flush=True)
        for r in [r for r in rows if r["n"] == n]:
            extra = f" (mean count {r['mean_count']:.0f})" if r["kind"] == "far_at_S0" else ""
            print(f"      {r['kind']:>10} {r['name']:>9}  mean {r['mean']:9.4g}  rel sd {100*r['rel_sd']:6.1f}%{extra}",
                  flush=True)

    res = {"pool": pool, "reps": args.reps, "rows": rows, "zero_lag_max": ZERO_LAG_MAX,
           "fit_max_n": FIT_MAX_N}
    print(f"\nSCALING (fit on n <= {FIT_MAX_N}, FPC-corrected)")
    res["alpha"] = {}
    for kind in ("threshold", "far_at_S0"):
        for name in sorted({r["name"] for r in rows if r["kind"] == kind}):
            sub = [r for r in rows if r["kind"] == kind and r["name"] == name and r["n"] <= FIT_MAX_N]
            if len(sub) < 3:
                continue
            a = -fit([r["n"] for r in sub], [r["rel_sd"] for r in sub])
            res["alpha"][f"{kind}:{name}"] = a
            print(f"  {kind:>10} {name:>9}: rel sd ~ n^-{a:.3f}")

    thr_a = [v for k, v in res["alpha"].items() if k.startswith("threshold")]
    far_a = {k.split(":")[1]: v for k, v in res["alpha"].items() if k.startswith("far_at_S0")}
    a_obs = far_a.get(f"{ZERO_LAG_MAX:.3f}")
    res["alpha_threshold_mean"] = float(np.mean(thr_a)) if thr_a else None
    res["alpha_far_at_observed"] = a_obs
    print(f"\nmean alpha(threshold) = {res['alpha_threshold_mean']:.3f}   "
          f"alpha(FAR at observed {ZERO_LAG_MAX:.2f}) = {a_obs if a_obs is None else round(a_obs,3)}")

    ok = (a_obs is not None and a_obs > 0.35 and (res["alpha_threshold_mean"] or 1) < 0.20)
    res["verdict"] = ("FAR-of-observed CONVERGES where the threshold does not — re-quote the null as the "
                      "FAR of the loudest event" if ok else
                      "no escape: both summaries saturate — the limit is the deep tail itself, not the choice "
                      "of statistic")
    print(f"VERDICT: {res['verdict']}")
    OUT.write_text(json.dumps(res, indent=2))
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
