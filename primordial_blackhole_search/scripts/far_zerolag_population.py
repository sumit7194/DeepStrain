"""The 45,073 zero-lag measurements we have always thrown away.

WHERE THIS SITS. Every deep-FAR result so far compares ONE number -- the loudest zero-lag coincidence --
against a threshold. But zero-lag gives us **45,074 coincidence measurements**, one per window, and we
discard 45,073 of them to look at a maximum. Two questions live in the part we discard, and neither can be
asked of a single event:

  (Q1) IS THE INDEPENDENCE ASSUMPTION TRUE WHERE IT MATTERS? The 4,120-yr ladder rests entirely on H1 _|_ L1,
       which we verified via the BULK correlation (r = -0.0022, p = 0.65). Bulk independence does not imply
       TAIL independence: two detectors can be uncorrelated in ordinary noise and still both respond to the
       same environmental transient. We have tested the assumption exactly where it is easiest to satisfy,
       and never where the ladder actually uses it.

  (Q2) IS THERE A SUB-THRESHOLD POPULATION? A real subsolar population too faint for any single event to
       cross threshold would not show up as a loud candidate at all -- it would show as a small STATISTICAL
       EXCESS spread across the whole zero-lag distribution. That is a genuinely different search from
       "is there a loud event", run on data already on disk.

THE NULL, and why time-slides are exactly the right one. Under H1 _|_ L1, the zero-lag sum H_i + L_i and the
slid sum H_i + L_j (i != j) are draws from the SAME distribution. So the time-slide background IS the
independence null, built from the same windows and the same noise -- no model, no assumption about the shape
of either detector's score distribution. Any systematic difference between the two distributions is either
correlated noise (Q1) or signal (Q2).

DISCRIMINATING Q1 FROM Q2 -- an excess alone cannot tell them apart, so we also ask whether it is ONE-SIDED.
Correlated glitches drive one detector far harder than the other (the L2 audit established the signature:
the loudest zero-lag event is H1 +12.53 with L1 at -1.24). A real signal population is TWO-SIDED, since both
detectors see the same astrophysical strain. Any excess is therefore reported with its one-sidedness.

METHOD. One pass over all 45,073 distinct lags accumulates the background into a fine histogram (quantised
bincount, so the full distribution is retained rather than only a top-k tail). Everything else -- the CDF at
any threshold, the excess ladder, the KS comparison -- is then read off that histogram at no extra cost.

PRE-REGISTERED (fixed before any number was produced):
  * excess ladder at expected counts 10000/1000/100/10/1: |z| > 3 after a Bonferroni factor for the number
    of thresholds tested -> investigate; a MONOTONIC excess growing toward the tail is the signature of a
    population, a single isolated bin is not
  * tail dependence at the 90/99/99.9 percentiles: joint exceedances vs the independence expectation
  * the honest expectation is AGREEMENT: this most likely returns "zero-lag matches background everywhere",
    which is the first validation of our independence assumption in the TAIL rather than the bulk. That is a
    result, not a failure -- it is the assumption the committed 4,120-yr ladder depends on.

CAVEAT. The background's effective sample size is ~2N distinct scores, not N^2 pairs (this is the L2
finding), so the background CDF is well-determined in the BULK and progressively less so in the deep tail.
The excess ladder is therefore trustworthy where expected counts are large and must be read with the
effective-N caveat where they are small -- the same caveat that governs the ladder itself.

Run:  .venv/bin/python scripts/far_zerolag_population.py [--bins 40000]
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
OUT = C.RESULTS_DIR / "far_zerolag_population.json"
YEAR_S = 3.156e7
WIN_SEC = 64.0
EXPECTED_COUNTS = (10000, 1000, 100, 10, 1)      # ladder rungs, in EXPECTED zero-lag events
QUANTILES = (90.0, 99.0, 99.9)


def load():
    H, L = [], []
    for p in sorted(CACHE.glob("seg_*.npz"), key=lambda q: int(q.stem.split("_")[1])):
        d = np.load(p); H.append(d["h"]); L.append(d["l"])
    return np.concatenate(H).astype(np.float64), np.concatenate(L).astype(np.float64)


def background_hist(H, L, lo, hi, bins):
    """Full background distribution over all N-1 distinct lags, as a histogram.

    A top-k tail would answer the tail questions but not the bulk ones; a histogram costs the same single
    pass and retains the whole distribution, which is what Q2 (a diffuse sub-threshold excess) needs."""
    N = len(H)
    scale = bins / (hi - lo)
    counts = np.zeros(bins + 2, dtype=np.int64)
    t0 = time.time()
    for k in range(1, N):
        v = H + np.roll(L, k)
        idx = np.clip(((v - lo) * scale).astype(np.int64) + 1, 0, bins + 1)
        counts += np.bincount(idx, minlength=bins + 2)
        if k % 5000 == 0:
            print(f"    lag {k}/{N-1} ({time.time()-t0:.0f}s)", flush=True)
    return counts


def surv(counts, edges_lo, scale, t):
    """Survival count P(X >= t) * n_total, read off the histogram (conservative: whole bins)."""
    i = int(np.clip((t - edges_lo) * scale, 0, len(counts) - 2)) + 1
    return int(counts[i:].sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bins", type=int, default=40000)
    args = ap.parse_args()

    H, L = load()
    N = len(H)
    Z = H + L                                     # THE 45,074 measurements we normally discard
    lo = float(H.min() + L.min()) - 1e-6
    hi = float(H.max() + L.max()) + 1e-6
    scale = args.bins / (hi - lo)
    live_h = N * WIN_SEC / 3600
    print(f"{N} windows | zero-lag pairs {N} | background lags {N-1} | livetime {live_h:.1f} h", flush=True)
    print(f"score-sum range [{lo:.2f}, {hi:.2f}] over {args.bins} bins\n", flush=True)

    counts = background_hist(H, L, lo, hi, args.bins)
    n_bg = int(counts.sum())
    res = {"n_windows": N, "n_bg_pairs": n_bg, "livetime_h": live_h,
           "background_years": (N - 1) * N * WIN_SEC / YEAR_S}

    # ---- Q1/Q2 EXCESS LADDER: observed zero-lag counts vs the independence null -------------------------
    print(f"\nEXCESS LADDER (zero-lag vs background null, {n_bg} background pairs)")
    print(f"{'expected':>9} {'threshold':>10} {'observed':>9} {'ratio':>7} {'z':>7}  one-sided%")
    ladder = []
    for want in EXPECTED_COUNTS:
        p_target = want / N
        # threshold whose background survival fraction gives `want` expected zero-lag events
        cum = np.cumsum(counts[::-1])[::-1]
        frac = cum / n_bg
        j = int(np.argmax(frac <= p_target))
        thr = lo + (j - 1) / scale
        exp = float(frac[j] * N)
        obs = int((Z >= thr).sum())
        z = (obs - exp) / np.sqrt(exp) if exp > 0 else float("nan")
        sel = Z >= thr
        # one-sidedness is only defined for POSITIVE scores: `min < 0.15*max` inverts its meaning once the
        # louder detector is itself negative, which it is at the shallow rungs. Restrict to max>0 and report
        # the sub-sample it was computed on, rather than emit a number that silently changes meaning.
        mx, mn = np.maximum(H[sel], L[sel]), np.minimum(H[sel], L[sel])
        ok = mx > 0
        one = float(np.mean(mn[ok] < 0.15 * mx[ok])) if ok.sum() else float("nan")
        n_one = int(ok.sum())
        ladder.append({"expected": exp, "threshold": float(thr), "observed": obs,
                       "ratio": obs / exp if exp else None, "z": float(z), "one_sided_frac": one,
                       "one_sided_n": n_one})
        print(f"{exp:9.1f} {thr:10.3f} {obs:9d} {obs/exp if exp else 0:7.2f} {z:7.2f}  "
              f"{100*one if obs else float('nan'):9.0f}%")
    res["excess_ladder"] = ladder
    nz = [r for r in ladder if np.isfinite(r["z"])]
    worst = max(nz, key=lambda r: abs(r["z"])) if nz else None
    bonf = len(EXPECTED_COUNTS)
    res["max_abs_z"] = abs(worst["z"]) if worst else None
    res["bonferroni_bar"] = 3.0
    print(f"  => largest |z| = {abs(worst['z']):.2f} at expected {worst['expected']:.0f} "
          f"(bar |z|>3 after x{bonf} trials)")

    # ---- Q1 TAIL DEPENDENCE: joint exceedances vs the independence expectation -------------------------
    print("\nTAIL DEPENDENCE (joint H1/L1 exceedances at zero lag vs independence)")
    print(f"{'pctile':>7} {'observed':>9} {'expected':>9} {'ratio':>7} {'perm p':>8}")
    dep = []
    rng = np.random.default_rng(0)
    for q in QUANTILES:
        qh, ql = np.percentile(H, q), np.percentile(L, q)
        obs = int(((H >= qh) & (L >= ql)).sum())
        exp = N * (1 - q / 100) ** 2
        perm = np.array([int(((H >= qh) & (rng.permutation(L) >= ql)).sum()) for _ in range(2000)])
        p = float((np.abs(perm - exp) >= abs(obs - exp)).mean())
        dep.append({"pctile": q, "observed": obs, "expected": exp,
                    "ratio": obs / exp if exp else None, "p_perm": p})
        print(f"{q:7.1f} {obs:9d} {exp:9.1f} {obs/exp if exp else 0:7.2f} {p:8.3f}")
    res["tail_dependence"] = dep

    # ---- Q2 BULK SHIFT: would a diffuse population lift the whole distribution? ------------------------
    # Compare the zero-lag CDF to the background CDF everywhere, not just above a threshold.
    edges = lo + (np.arange(args.bins + 1)) / scale
    bg_cdf = np.cumsum(counts[1:-1]) / max(1, counts[1:-1].sum())
    zi = np.clip(((Z - lo) * scale).astype(np.int64), 0, args.bins - 1)
    z_cdf = np.cumsum(np.bincount(zi, minlength=args.bins)) / N
    ks = float(np.max(np.abs(z_cdf - bg_cdf)))
    ks_crit = 1.358 / np.sqrt(N)                 # 95% two-sided, n_eff = zero-lag sample size
    res["ks"] = {"stat": ks, "crit95": float(ks_crit), "exceeds": bool(ks > ks_crit)}
    res["median_shift"] = float(np.median(Z) - edges[int(np.argmax(bg_cdf >= 0.5))])
    print(f"\nBULK: KS(zero-lag, background) = {ks:.5f} vs 95% crit {ks_crit:.5f} -> "
          f"{'DIFFERENT' if ks > ks_crit else 'consistent'}")
    print(f"      median shift = {res['median_shift']:+.4f}")

    sig = (res["max_abs_z"] or 0) > 3.0 or any(d["p_perm"] < 0.05 / len(QUANTILES) for d in dep)
    res["verdict"] = ("SOMETHING TO INVESTIGATE — see ladder/dependence" if sig else
                      "CLEAN — zero-lag matches the independence null in bulk and tail")
    print(f"\nVERDICT: {res['verdict']}")
    OUT.write_text(json.dumps(res, indent=2))
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
