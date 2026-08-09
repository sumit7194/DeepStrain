"""Stress-test the 80.5-year deep-FAR background itself: are its assumptions actually met?

far_deep.py quotes a 1/decade threshold from a background built by global time-slides. That number rests on
assumptions we asserted rather than measured. This audits them. Each test can only make the headline WEAKER
or better-caveated; none can make it stronger, which is the point.

V1  INDEPENDENCE. Time-slides assume H1 and L1 noise are independent, so that a shifted pair is a fair model
    of an accidental coincidence. If the detectors share environmental noise, real (zero-lag) pairs are
    correlated, shifted pairs are not, and the background UNDERSTATES the true accidental rate -> thresholds
    too low -> a false alarm looks significant. Test: zero-lag corr(sH, sL) against the null distribution of
    corr over all N-1 lags.

V2  EFFECTIVE SAMPLE SIZE. "80.5 years" is livetime, NOT 80.5 years of independent data. All N(N-1) pairs are
    built from only 2N underlying window scores, and the extreme tail is dominated by the few loudest windows
    in each detector. If the 8 events that set 1/decade trace back to 2-3 distinct H1 windows, the threshold
    rests on a handful of glitches, not on 80 years of statistics.

V3  SMALL-NUMBER UNCERTAINTY. 1/decade is the 8th-loudest background event. Poisson on 8 counts is +-35% in
    rate. Propagate that into a threshold band so the ladder is quoted with an error bar, not as an exact
    number.

V4  CONVERGENCE. Recompute the ladder from segment prefixes (10, 20, ... 100). A threshold still drifting at
    n=100 has not converged and the deepest rung is premature.

V5  STATIONARITY. First half vs second half of the segments at a common FAR. O4b spans months; if detector
    behaviour drifts, one pooled background is the wrong model.

V6  SHARED DATA QUALITY. Per-segment mean scores in H1 vs L1. Correlated data quality is the physical
    mechanism that would break V1, so measure it directly rather than inferring it.

Run:  .venv/bin/python scripts/far_background_validation.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pbh import config as C

CACHE = C.RESULTS_DIR / "far_scores"
YEAR_S = 3.156e7
KEEP = 20_000                 # >> the ~966 events that set the shallowest rung (1/month)
FARS = (("1/month", 12.0), ("1/year", 1.0), ("1/decade", 0.1))


def bg_top(sH, sL, keep=KEEP):
    """Global time-slide background, N-1 distinct lags, loudest `keep` values (ascending)."""
    N = len(sH)
    top = np.full(min(keep, N * (N - 1)), -np.inf, dtype=np.float32)
    for k in range(1, N):
        vals = sH + np.roll(sL, k)
        if vals.max() > top[0]:
            top = np.partition(np.concatenate([top, vals]), -len(top))[-len(top):]
            top.sort()
    return top, N - 1


def ladder(top, n_lags, live_s):
    bg_yr = n_lags * live_s / YEAR_S
    out = {}
    for label, per_year in FARS:
        k = int(round(per_year * bg_yr))
        if 1 <= k <= len(top):
            out[label] = {"threshold": float(top[-k]), "n_bg_events": k}
    return out, bg_yr


def main() -> None:
    segs = sorted(int(p.stem.split("_")[1]) for p in CACHE.glob("seg_*.npz"))
    per = [np.load(CACHE / f"seg_{g}.npz") for g in segs]
    Hs = [d["h"].astype(np.float32) for d in per]
    Ls = [d["l"].astype(np.float32) for d in per]
    H, L = np.concatenate(Hs), np.concatenate(Ls)
    N = len(H)
    live_1 = C.SEGMENT_LEN
    res = {"n_segments": len(segs), "n_windows": N}
    print(f"{N} windows, {len(segs)} segments\n")

    # ---- V1 independence -------------------------------------------------------------------------------
    Hc, Lc = H - H.mean(), L - L.mean()
    denom = np.sqrt((Hc**2).sum() * (Lc**2).sum())
    r0 = float((Hc * Lc).sum() / denom)
    rs = np.array([float((Hc * np.roll(Lc, k)).sum() / denom) for k in range(1, N)])
    z = (r0 - rs.mean()) / rs.std()
    p_emp = float((np.abs(rs) >= abs(r0)).mean())
    res["V1_independence"] = {"zero_lag_r": r0, "lag_null_mean": float(rs.mean()),
                              "lag_null_sd": float(rs.std()), "z": float(z), "p_two_sided": p_emp}
    print("V1 INDEPENDENCE (time-slides assume H1 _|_ L1)")
    print(f"  zero-lag corr(H1,L1) = {r0:+.4f}   null over {N-1} lags: {rs.mean():+.4f} +- {rs.std():.4f}")
    print(f"  z = {z:+.2f}, empirical two-sided p = {p_emp:.3f} -> "
          f"{'NO detectable shared noise: assumption holds' if p_emp > 0.05 else 'CORRELATED — background may be biased LOW'}")

    # ---- V6 shared data quality (the mechanism behind V1) ----------------------------------------------
    mh = np.array([h.mean() for h in Hs]); ml = np.array([l.mean() for l in Ls])
    r_seg = float(np.corrcoef(mh, ml)[0, 1])
    # permutation p: how often does a random pairing of segments beat it?
    rng = np.random.default_rng(0)
    perm = np.array([np.corrcoef(mh, rng.permutation(ml))[0, 1] for _ in range(2000)])
    p_seg = float((np.abs(perm) >= abs(r_seg)).mean())
    res["V6_segment_quality"] = {"r_segment_mean": r_seg, "p_perm": p_seg}
    print(f"\nV6 SHARED DATA QUALITY: per-segment mean score corr(H1,L1) = {r_seg:+.3f}, perm p = {p_seg:.3f}"
          f" -> {'independent' if p_seg > 0.05 else 'segments co-vary'}")

    # ---- main background + V2 effective sample size ----------------------------------------------------
    top, n_lags = bg_top(H, L)
    lad, bg_yr = ladder(top, n_lags, len(segs) * live_1)
    res["background_years"] = bg_yr
    res["ladder"] = lad
    print(f"\nbackground: {n_lags} lags x {len(segs)*live_1/3600:.1f} h = {bg_yr:.1f} yr")

    print("\nV2 EFFECTIVE SAMPLE SIZE (how many DISTINCT windows underpin each rung?)")
    deep_thr = min(v["threshold"] for v in lad.values())
    hits = []
    for k in range(1, N):
        s = H + np.roll(L, k)
        idx = np.flatnonzero(s >= deep_thr)
        for i in idx:
            hits.append((float(s[i]), int(i), int((i - k) % N)))
    hits.sort(reverse=True)
    res["V2_effective_n"] = {}
    for label, _ in FARS:
        if label not in lad:
            continue
        kk = lad[label]["n_bg_events"]
        sub = hits[:kk]
        nh, nl = len({h[1] for h in sub}), len({h[2] for h in sub})
        res["V2_effective_n"][label] = {"n_bg_events": kk, "distinct_H1_windows": nh, "distinct_L1_windows": nl,
                                        "distinct_frac": (nh + nl) / (2 * kk)}
        print(f"  {label:>9}: {kk:>4} bg events <- {nh:>4} distinct H1 windows, {nl:>4} distinct L1 windows"
              f"  ({100*(nh+nl)/(2*kk):.0f}% distinct)")

    # ---- V3 Poisson band -------------------------------------------------------------------------------
    print("\nV3 SMALL-NUMBER UNCERTAINTY (Poisson on the count that sets each rung)")
    res["V3_poisson"] = {}
    for label, _ in FARS:
        if label not in lad:
            continue
        k = lad[label]["n_bg_events"]
        lo_k, hi_k = max(1, int(round(k - np.sqrt(k)))), min(len(top), int(round(k + np.sqrt(k))))
        band = (float(top[-hi_k]), float(top[-lo_k]))     # more events -> lower threshold
        res["V3_poisson"][label] = {"threshold": lad[label]["threshold"], "band": band, "k": k}
        print(f"  {label:>9}: {lad[label]['threshold']:6.3f}  [{band[0]:.3f}, {band[1]:.3f}]  (k={k}, +-sqrt(k))")

    # ---- V4 convergence --------------------------------------------------------------------------------
    print("\nV4 CONVERGENCE (ladder from segment prefixes)")
    res["V4_convergence"] = []
    for n in [10, 20, 40, 60, 80, len(segs)]:
        if n > len(segs):
            continue
        h = np.concatenate(Hs[:n]); l = np.concatenate(Ls[:n])
        t, nl_ = bg_top(h, l, keep=min(KEEP, 5000))
        la, yr = ladder(t, nl_, n * live_1)
        row = {"n_segments": n, "bg_years": yr, **{k: v["threshold"] for k, v in la.items()}}
        res["V4_convergence"].append(row)
        print(f"  n={n:>3} ({yr:6.1f} yr): " + "  ".join(f"{k} {v['threshold']:6.3f}" for k, v in la.items()))
    print("  => compare the SAME rung down the column; drift at n=100 would mean not converged.")

    # ---- V5 stationarity -------------------------------------------------------------------------------
    print("\nV5 STATIONARITY (first half vs second half, matched FAR)")
    half = len(segs) // 2
    res["V5_stationarity"] = {}
    for name, sl in (("first_half", slice(0, half)), ("second_half", slice(half, len(segs)))):
        h = np.concatenate(Hs[sl]); l = np.concatenate(Ls[sl])
        t, nl_ = bg_top(h, l, keep=min(KEEP, 5000))
        la, yr = ladder(t, nl_, (sl.stop - sl.start) * live_1)
        res["V5_stationarity"][name] = {"bg_years": yr, **{k: v["threshold"] for k, v in la.items()}}
        print(f"  {name:>12} ({yr:5.1f} yr): " + "  ".join(f"{k} {v['threshold']:6.3f}" for k, v in la.items()))
    common = set(res["V5_stationarity"]["first_half"]) & set(res["V5_stationarity"]["second_half"]) - {"bg_years"}
    for k in sorted(common):
        a, b = res["V5_stationarity"]["first_half"][k], res["V5_stationarity"]["second_half"][k]
        print(f"  {k}: {a:.3f} vs {b:.3f}  ({100*abs(a-b)/max(a,b):.1f}% apart)")

    # ---- V7 how much does any single chunk of data matter? --------------------------------------------
    # V5's half-to-half gap is not proof of drift: bulk noise is identical between halves (median -0.81 vs
    # -0.77), and 6 of the 8 loudest H1 windows sit in ONE segment. So the honest error bar on each rung is
    # a leave-block-out jackknife, which asks exactly "would a different 10% of the data have moved this?".
    print("\nV7 JACKKNIFE (drop each 10% block of segments in turn)")
    res["V7_jackknife"] = {}
    nb, jk = 10, {lbl: [] for lbl, _ in FARS}
    bs = len(segs) // nb
    for b in range(nb):
        keepi = [i for i in range(len(segs)) if not (b * bs <= i < (b + 1) * bs)]
        h = np.concatenate([Hs[i] for i in keepi]); l = np.concatenate([Ls[i] for i in keepi])
        t, nl_ = bg_top(h, l, keep=min(KEEP, 5000))
        la, _ = ladder(t, nl_, len(keepi) * live_1)
        for lbl, v in la.items():
            jk[lbl].append(v["threshold"])
    for lbl, _ in FARS:
        v = np.array(jk[lbl])
        if not len(v):
            continue
        full = lad[lbl]["threshold"]
        res["V7_jackknife"][lbl] = {"full": full, "min": float(v.min()), "max": float(v.max()),
                                    "sd": float(v.std()), "spread_pct": float(100 * (v.max() - v.min()) / full)}
        print(f"  {lbl:>9}: full {full:6.3f}   jackknife [{v.min():6.3f}, {v.max():6.3f}]  "
              f"sd {v.std():.3f}  spread {100*(v.max()-v.min())/full:.0f}% of the quoted value")

    # labelled diagnostic, NOT a tuned result: the single segment holding the glitch cluster
    worst = int(np.argmax([h.max() for h in Hs]))
    keepi = [i for i in range(len(segs)) if i != worst]
    h = np.concatenate([Hs[i] for i in keepi]); l = np.concatenate([Ls[i] for i in keepi])
    t, nl_ = bg_top(h, l, keep=min(KEEP, 5000))
    la, _ = ladder(t, nl_, len(keepi) * live_1)
    res["V7_drop_worst_segment"] = {"segment_index": worst, "gps": segs[worst],
                                    **{k: v["threshold"] for k, v in la.items()}}
    print(f"  diagnostic — drop the single glitchiest segment (idx {worst}, gps {segs[worst]}): "
          + "  ".join(f"{k} {v['threshold']:.3f}" for k, v in la.items()))
    print("  (reported to show the dependence, NOT adopted — post-hoc removal of the loudest segment would be tuning.)")

    # ---- V8 the question all of the above exists to answer ---------------------------------------------
    # Every test so far attacks the THRESHOLD. What survives is the SEARCH RESULT: for each configuration,
    # compare the zero-lag maximum against the threshold derived from that same configuration. A null that
    # holds only for one choice of data and statistic is not a null.
    print("\nV8 IS THE SEARCH NULL IN EVERY CONFIGURATION? (zero-lag vs its OWN matched threshold)")
    res["V8_null_robustness"] = []
    worst_i = int(np.argmax([h.max() for h in Hs]))
    configs = [("all segments", list(range(len(segs)))),
               (f"drop glitchiest seg {worst_i}", [i for i in range(len(segs)) if i != worst_i])]
    for cname, keepi in configs:
        h = np.concatenate([Hs[i] for i in keepi]); l = np.concatenate([Ls[i] for i in keepi])
        n = len(h); bg_yr_c = (n - 1) * len(keepi) * live_1 / YEAR_S
        for sname, fn in (("sum", lambda a, b: a + b), ("min", np.minimum)):
            top = np.full(5000, -np.inf, dtype=np.float32)
            for k in range(1, n):
                v = fn(h, np.roll(l, k))
                if v.max() > top[0]:
                    top = np.partition(np.concatenate([top, v]), -5000)[-5000:]; top.sort()
            zl = float(fn(h, l).max())
            kk = int(round(0.1 * bg_yr_c))
            thr = float(top[-kk]) if 1 <= kk <= len(top) else None
            row = {"config": cname, "statistic": sname, "far": "1/decade", "threshold": thr,
                   "zero_lag_max": zl, "null": bool(thr is not None and zl < thr),
                   "margin_x": round(thr / zl, 2) if thr and zl > 0 else None}
            res["V8_null_robustness"].append(row)
            print(f"  {cname:>26} | {sname:>3} | thr {thr:6.3f} | zero-lag {zl:6.3f} | "
                  f"{'NULL' if row['null'] else 'DETECTION'}")
    res["all_configs_null"] = all(r["null"] for r in res["V8_null_robustness"])
    print(f"  => null holds in {sum(r['null'] for r in res['V8_null_robustness'])}/"
          f"{len(res['V8_null_robustness'])} configurations")

    (C.RESULTS_DIR / "far_background_validation.json").write_text(json.dumps(res, indent=2))
    print("\nwrote far_background_validation.json")


if __name__ == "__main__":
    main()
