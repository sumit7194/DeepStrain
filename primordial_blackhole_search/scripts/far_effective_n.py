"""What actually limits a time-slide threshold? Measure the scaling law, don't assume it.

WHERE THIS SITS. Two limits are now measured. L2: 51x more background left 1/decade resting on the SAME 8
distinct H1 windows -- livetime does not buy independent samples. The zero-lag population test: the deep
rungs sit beyond all zero-lag data, so sliding does not buy verification reach either. Both are statements
that time-slides saturate. Neither says HOW FAST, and that is the number a person planning a search needs.

THE CLAIM TO TEST, stated as a mechanism rather than a curve. Background-years grow as n^2 (both the lag
count and the livetime grow with n), while the number of distinct loud windows feeding the tail should grow
only as n^1. If the tail's loud windows are the effective independent draws, then the threshold's sampling
error must be governed by N_eff and NOTHING ELSE:

    sigma_T * sqrt(N_eff) = const,  across every subsample size AND every FAR rung

That is a strong, falsifiable statement -- far stronger than fitting sigma_T ~ n^-alpha, because it predicts
that points from DIFFERENT rungs (which have wildly different event counts) must COLLAPSE onto one curve when
plotted against N_eff. A power-law fit alone cannot distinguish "N_eff controls it" from "n controls it";
the collapse can. If the collapse fails, N_eff is not the controlling variable and the L2 story needs
revising -- which is the outcome worth hunting.

If it holds, the practical consequence is a planning number: sigma_T ~ n^-1/2 ~ bg_years^-1/4, i.e.
**halving your threshold uncertainty costs 4x the segments and 16x the background-years.**

METHOD. For each subsample size n, draw M independent random subsets of n segments from the 727 on disk,
recompute the ladder in each, and take the spread ACROSS subsets -- a direct measurement of the sampling
distribution, not a jackknife proxy for it.

THE BIAS THAT WOULD HAVE FAKED THE RESULT. Subsets are drawn without replacement from a FINITE pool, so as n
approaches 727 they share more and more segments and the measured spread is suppressed -- at n=727 it is
exactly zero by construction. Uncorrected, that suppression at large n masquerades as faster-than-true
improvement and STEEPENS the fitted exponent. Every spread is therefore divided by the finite-population
correction sqrt(1 - n/727), the primary fit is restricted to n <= 240 (pool fraction <= 1/3), and both raw
and corrected values are reported so the correction can be audited rather than trusted.

PRE-REGISTERED:
  * primary   : does sigma_T * sqrt(N_eff) collapse across rungs? (scatter of that product <= 25% => yes)
  * secondary : exponents alpha in sigma_T ~ n^-alpha and beta in N_eff ~ n^beta
  * prediction: beta ~ 1.0, alpha ~ 0.5. Our two existing points (100 segs -> 33-44%, 727 -> 10-12%) imply
    alpha ~ 0.62, i.e. slightly STEEPER than the naive 0.5 -- if that survives with error bars it means the
    loud-window count grows a little faster than linearly, and beta should show it.
  * a rung is skipped where it needs <1 background event; those are absent, not zero.

Run:  .venv/bin/python scripts/far_effective_n.py [--reps 20]
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
OUT = C.RESULTS_DIR / "far_effective_n.json"
YEAR_S = 3.156e7
WIN_SEC = 64.0
SIZES = (20, 40, 80, 160, 240, 320)
FIT_MAX_N = 240                      # pool fraction <= 1/3: beyond this the FPC is doing too much work
FARS = (("1/month", 12.0), ("1/year", 1.0), ("1/decade", 0.1))
KEEP = 200_000


def load_segments():
    Hs, Ls = [], []
    for p in sorted(CACHE.glob("seg_*.npz"), key=lambda q: int(q.stem.split("_")[1])):
        d = np.load(p); Hs.append(d["h"].astype(np.float32)); Ls.append(d["l"].astype(np.float32))
    return Hs, Ls


def ladder_and_neff(H, L):
    """Threshold at each FAR, and the number of DISTINCT H1 windows that produce events above it."""
    N = len(H)
    bg_yr = (N - 1) * N * WIN_SEC / YEAR_S
    # sized to the shallowest rung this subset supports (see far_estimator_bias): a fixed 200k cap makes
    # small subsets ~20x slower for events they will never read.
    need = max(int(round(pyr * bg_yr)) for _, pyr in FARS)
    keep = int(min(max(4 * need, 5_000), N * (N - 1)))
    top = np.full(keep, -np.inf, dtype=np.float32)
    for k in range(1, N):
        v = H + np.roll(L, k)
        if v.max() > top[0]:
            top = np.partition(np.concatenate([top, v]), -keep)[-keep:]
            top.sort()
    # Thresholds first, then ONE more sweep that scores every rung at once. A sweep per rung costs 4x for
    # nothing: the loop over lags is identical, only the comparison threshold differs.
    thr = {}
    for label, per_year in FARS:
        k = int(round(per_year * bg_yr))
        if 1 <= k <= len(top):
            thr[label] = (float(top[-k]), k)                # absent (not zero) when the rung needs <1 event
    hits = {lbl: set() for lbl in thr}
    if thr:
        lo = min(t for t, _ in thr.values())
        for kk in range(1, N):
            v = H + np.roll(L, kk)
            idx = np.where(v >= lo)[0]
            if not len(idx):
                continue
            vi = v[idx]
            for lbl, (t, _) in thr.items():
                sel = idx[vi >= t]
                if len(sel):
                    hits[lbl].update(sel.tolist())
    out = {lbl: {"threshold": t, "n_bg_events": k, "n_eff": len(hits[lbl])}
           for lbl, (t, k) in thr.items()}
    return out, bg_yr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args()

    Hs, Ls = load_segments()
    pool = len(Hs)
    rng = np.random.default_rng(11)
    print(f"pool: {pool} segments | sizes {SIZES} | {args.reps} random subsets each\n", flush=True)

    rows = []
    for n in SIZES:
        t0 = time.time()
        per = {lbl: {"thr": [], "neff": []} for lbl, _ in FARS}
        bgs = []
        for r in range(args.reps):
            sel = rng.choice(pool, n, replace=False)
            H = np.concatenate([Hs[i] for i in sel]); L = np.concatenate([Ls[i] for i in sel])
            lad, bg = ladder_and_neff(H, L)
            bgs.append(bg)
            for lbl, v in lad.items():
                per[lbl]["thr"].append(v["threshold"]); per[lbl]["neff"].append(v["n_eff"])
        fpc = np.sqrt(max(1e-9, 1.0 - n / pool))
        for lbl, _ in FARS:
            t = np.array(per[lbl]["thr"])
            if len(t) < 3:
                continue
            ne = float(np.mean(per[lbl]["neff"]))
            sd_raw = float(t.std(ddof=1))
            bs = np.array([np.std(rng.choice(t, len(t), replace=True), ddof=1) for _ in range(2000)])
            sd_lo, sd_hi = float(np.percentile(bs, 5)), float(np.percentile(bs, 95))
            rows.append({"n_segments": n, "far": lbl, "bg_years": float(np.mean(bgs)),
                         "threshold_mean": float(t.mean()), "sd_raw": sd_raw,
                         "sd": sd_raw / fpc, "sd_ci90": [sd_lo / fpc, sd_hi / fpc],
                         "fpc": float(fpc), "n_eff": ne, "reps": len(t)})
        here = [r for r in rows if r["n_segments"] == n]
        desc = "  ".join(f"{r['far']} sd {r['sd']:.3f}/Neff {r['n_eff']:.0f}" for r in here) or "no rung reachable"
        print(f"  n={n:4d} ({time.time()-t0:5.0f}s, bg {np.mean(bgs):8.1f} yr, fpc {fpc:.3f}): {desc}", flush=True)

    res = {"pool": pool, "reps": args.reps, "rows": rows, "fit_max_n": FIT_MAX_N}

    def fit(xs, ys):
        x, y = np.log(np.array(xs, float)), np.log(np.array(ys, float))
        A = np.vstack([x, np.ones_like(x)]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        return float(m), float(np.exp(c))

    print("\nEXPONENTS (primary fit restricted to n <= %d, FPC-corrected)" % FIT_MAX_N)
    res["exponents"] = {}
    for lbl, _ in FARS:
        sub = [r for r in rows if r["far"] == lbl and r["n_segments"] <= FIT_MAX_N]
        if len(sub) < 3:
            continue
        a, _ = fit([r["n_segments"] for r in sub], [r["sd"] for r in sub])
        # propagate each sigma's CI into alpha by refitting on draws from within those CIs
        aa = []
        for _ in range(2000):
            ys = [float(rng.uniform(r["sd_ci90"][0], r["sd_ci90"][1])) for r in sub]
            aa.append(fit([r["n_segments"] for r in sub], ys)[0])
        a_lo, a_hi = float(np.percentile(aa, 5)), float(np.percentile(aa, 95))
        b, _ = fit([r["n_segments"] for r in sub], [r["n_eff"] for r in sub])
        res["exponents"][lbl] = {"alpha_sigma_vs_n": -a, "alpha_ci90": [-a_hi, -a_lo],
                                 "beta_neff_vs_n": b}
        print(f"  {lbl:>9}: sigma ~ n^-{-a:.3f} [{-a_hi:+.3f},{-a_lo:+.3f}]   N_eff ~ n^{b:.3f}")

    # ---- PRIMARY: does N_eff control the error? sigma*sqrt(N_eff) must be constant across n AND rungs ----
    print("\nPRIMARY — collapse test: sigma * sqrt(N_eff) should be CONSTANT if N_eff is the controlling variable")
    prod = []
    for r in rows:
        if r["n_segments"] <= FIT_MAX_N and r["n_eff"] > 0:
            r["sigma_sqrt_neff"] = r["sd"] * np.sqrt(r["n_eff"])
            prod.append(r["sigma_sqrt_neff"])
    prod = np.array(prod)
    scat = float(prod.std(ddof=1) / prod.mean())
    res["collapse"] = {"mean": float(prod.mean()), "sd": float(prod.std(ddof=1)),
                       "rel_scatter": scat, "n_points": int(len(prod)), "bar": 0.25}
    for lbl, _ in FARS:
        vals = [r["sigma_sqrt_neff"] for r in rows
                if r["far"] == lbl and r["n_segments"] <= FIT_MAX_N and "sigma_sqrt_neff" in r]
        if vals:
            print(f"  {lbl:>9}: " + " ".join(f"{v:5.2f}" for v in vals))
    print(f"  => relative scatter {100*scat:.0f}% over {len(prod)} points (bar: <=25% => N_eff controls it)")

    res["collapse_holds"] = bool(scat <= 0.25)
    a_mean = float(np.mean([e["alpha_sigma_vs_n"] for e in res["exponents"].values()]))
    b_mean = float(np.mean([e["beta_neff_vs_n"] for e in res["exponents"].values()]))
    res["alpha_mean"], res["beta_mean"] = a_mean, b_mean
    # A planning number is only meaningful if sigma actually falls. At alpha ~ 0, 2**(1/alpha) explodes into
    # a meaningless 1e55 -- an artifact of dividing by ~zero, not a finding. Report it only when the exponent
    # is resolvably positive, and otherwise say plainly that no amount of data buys the improvement.
    res["cost_to_halve_sigma"] = ({"segments_factor": float(2 ** (1 / a_mean)),
                                   "bg_years_factor": float(4 ** (1 / a_mean))} if a_mean > 0.15 else
                                  {"segments_factor": None, "bg_years_factor": None,
                                   "reason": f"alpha={a_mean:.3f} is not resolvably >0: sigma does not fall "
                                             "with n over the measured range, so no data volume halves it"})
    print(f"\nMEAN alpha {a_mean:.3f} (sigma ~ n^-alpha) | beta {b_mean:.3f} (N_eff ~ n^beta)")
    c = res["cost_to_halve_sigma"]
    print(f"PLANNING NUMBER: halving sigma costs {c['segments_factor']:.1f}x segments = "
          f"{c['bg_years_factor']:.1f}x background-years" if c["segments_factor"]
          else f"PLANNING NUMBER: none — {c['reason']}")
    res["verdict"] = ("N_eff is the controlling variable — the collapse holds" if res["collapse_holds"]
                      else "collapse FAILS — N_eff alone does not govern the error, revisit the L2 story")
    print(f"VERDICT: {res['verdict']}")
    OUT.write_text(json.dumps(res, indent=2))
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
