"""Is the jackknife we published actually measuring the uncertainty it claims?

WHY THIS EXISTS. L2 (2026-08-19) reported the deep-FAR threshold precision improving from +-33-44% at 100
segments to +-10-12% at 727, and I wrote that up as "the audit's central complaint answered by data". Both
numbers came from the SAME estimator: a leave-10%-block-out jackknife. `far_effective_n.py` then measured
the sampling spread a different way -- the SD of thresholds across INDEPENDENT random subsets -- and found
something the jackknife never showed:

    * sigma is FLAT in n (alpha = 0.005 over 20..320 segments, a 145x range in background-years)
    * between-subset SD at n=240 is 1.7-1.9, while the published jackknife at n=727 gives 0.41

Those cannot both be right unless sigma falls steeply between 240 and 727 -- and we measured that it does
not fall at all. So one of two things is true, and they have opposite consequences:

    (A) the jackknife UNDERSTATES the true sampling error (bias), or
    (B) the two estimators legitimately differ because they answer different questions at different n.

THE EXPERIMENT THAT SEPARATES THEM. Run both estimators at the SAME n, on the SAME subsets. Then no
sample-size difference can explain a gap, and the ratio is a pure estimator-bias factor.

THE SHARPER QUESTION -- does the bias GROW with n? The jackknife drops 10% of segments. At n=100 that is 10
segments; at n=727 it is 72. But the deep tail is set by a handful of glitchy segments, and with 727 segments
in play, dropping 72 of them almost never removes THE dominant glitch -- so the jackknife should look
*calmer* the more data you have, regardless of the true uncertainty. If the bias factor grows with n, then
the published "33-44% -> 10-12% improvement" is at least partly an artifact of the estimator rather than a
real gain in precision, and the L2 write-up needs correcting.

PRE-REGISTERED:
  * bias(n) = between-subset SD / mean within-subset jackknife SD, measured at n = 100 and n = 320
  * bias ratio > 1.5 at either n            -> the jackknife understates the true error
  * bias(320) / bias(100) > 1.3             -> the bias GROWS with n => the published improvement is inflated
  * the jackknife is reproduced EXACTLY as L2 computed it (contiguous 10% blocks, np.std with ddof=0), so
    this measures our published estimator and not a lookalike

Run:  .venv/bin/python scripts/far_estimator_bias.py [--reps 10] [--sizes 100 320]
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
OUT = C.RESULTS_DIR / "far_estimator_bias.json"
YEAR_S = 3.156e7
WIN_SEC = 64.0
FARS = (("1/month", 12.0), ("1/year", 1.0), ("1/decade", 0.1))
KEEP = 200_000
N_BLOCKS = 10


def load_segments():
    Hs, Ls = [], []
    for p in sorted(CACHE.glob("seg_*.npz"), key=lambda q: int(q.stem.split("_")[1])):
        d = np.load(p); Hs.append(d["h"].astype(np.float32)); Ls.append(d["l"].astype(np.float32))
    return Hs, Ls


def ladder(H, L):
    """Thresholds only -- the n_eff sweep is not needed here and doubles the cost."""
    N = len(H)
    bg_yr = (N - 1) * N * WIN_SEC / YEAR_S
    # `keep` sized to the SHALLOWEST rung this subset can support, not to a fixed constant: the partition
    # cost scales with keep, and a cap set for the 727-segment ladder makes every small subset ~20x slower
    # for events it will never read. 4x headroom over the deepest rank actually needed.
    need = max(int(round(pyr * bg_yr)) for _, pyr in FARS)
    keep = int(min(max(4 * need, 5_000), N * (N - 1)))
    top = np.full(keep, -np.inf, dtype=np.float32)
    for k in range(1, N):
        v = H + np.roll(L, k)
        if v.max() > top[0]:
            top = np.partition(np.concatenate([top, v]), -keep)[-keep:]
            top.sort()
    out = {}
    for label, per_year in FARS:
        k = int(round(per_year * bg_yr))
        if k > len(top):
            print(f"    !! {label} needs rank {k} but only {len(top)} kept -- RAISE keep, not a data limit",
                  flush=True)
            continue
        if k >= 1:
            out[label] = float(top[-k])
    return out


def jackknife_sd(Hs, Ls, idx):
    """EXACTLY L2's estimator: contiguous 10% blocks, np.std (ddof=0) over the fold thresholds."""
    per = {lbl: [] for lbl, _ in FARS}
    bs = max(1, len(idx) // N_BLOCKS)
    for b in range(N_BLOCKS):
        keep = [idx[i] for i in range(len(idx)) if not (b * bs <= i < (b + 1) * bs)]
        if not keep:
            continue
        lad = ladder(np.concatenate([Hs[i] for i in keep]), np.concatenate([Ls[i] for i in keep]))
        for lbl, v in lad.items():
            per[lbl].append(v)
    return {lbl: float(np.std(v)) for lbl, v in per.items() if len(v) >= 3}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--sizes", type=int, nargs="+", default=[100, 320])
    args = ap.parse_args()

    Hs, Ls = load_segments()
    pool = len(Hs)
    rng = np.random.default_rng(23)
    print(f"pool {pool} segments | sizes {args.sizes} | {args.reps} subsets each "
          f"| jackknife = {N_BLOCKS} contiguous blocks (L2's estimator)\n", flush=True)

    res = {"pool": pool, "reps": args.reps, "n_blocks": N_BLOCKS, "by_size": {}}
    for n in args.sizes:
        t0 = time.time()
        thr = {lbl: [] for lbl, _ in FARS}
        jack = {lbl: [] for lbl, _ in FARS}
        for r in range(args.reps):
            idx = rng.choice(pool, n, replace=False).tolist()
            lad = ladder(np.concatenate([Hs[i] for i in idx]), np.concatenate([Ls[i] for i in idx]))
            for lbl, v in lad.items():
                thr[lbl].append(v)
            for lbl, v in jackknife_sd(Hs, Ls, idx).items():
                jack[lbl].append(v)
            print(f"    n={n} rep {r+1}/{args.reps} ({time.time()-t0:.0f}s)", flush=True)
        fpc = np.sqrt(max(1e-9, 1.0 - n / pool))
        row = {}
        for lbl, _ in FARS:
            if len(thr[lbl]) < 3 or not jack[lbl]:
                continue
            between = float(np.std(thr[lbl], ddof=1)) / fpc
            jk = float(np.mean(jack[lbl]))
            row[lbl] = {"between_subset_sd": between, "jackknife_sd": jk,
                        "bias_ratio": between / jk if jk > 0 else None,
                        "threshold_mean": float(np.mean(thr[lbl])), "fpc": float(fpc)}
        res["by_size"][str(n)] = row
        print(f"\n  n={n} ({time.time()-t0:.0f}s, fpc {fpc:.3f})")
        print(f"    {'FAR':>9} {'between':>9} {'jackknife':>10} {'bias':>6}")
        for lbl, v in row.items():
            print(f"    {lbl:>9} {v['between_subset_sd']:9.3f} {v['jackknife_sd']:10.3f} "
                  f"{v['bias_ratio']:6.2f}x")

    sizes = [s for s in map(str, args.sizes) if s in res["by_size"]]
    if len(sizes) >= 2:
        lo, hi = sizes[0], sizes[-1]
        common = set(res["by_size"][lo]) & set(res["by_size"][hi])
        growth = {k: res["by_size"][hi][k]["bias_ratio"] / res["by_size"][lo][k]["bias_ratio"]
                  for k in common}
        res["bias_growth"] = growth
        mean_growth = float(np.mean(list(growth.values())))
        mean_bias = float(np.mean([v["bias_ratio"] for v in res["by_size"][hi].values()]))
        res["mean_bias_at_largest_n"] = mean_bias
        res["mean_bias_growth"] = mean_growth
        print(f"\nBIAS GROWTH n={lo} -> n={hi}: " + "  ".join(f"{k} {g:.2f}x" for k, g in growth.items()))
        print(f"  mean bias at n={hi}: {mean_bias:.2f}x | mean growth {mean_growth:.2f}x")
        res["verdict"] = (
            "jackknife UNDERSTATES the error AND the bias grows with n => the published "
            "33-44% -> 10-12% improvement is inflated by the estimator" if mean_bias > 1.5 and mean_growth > 1.3
            else "jackknife understates the error, but the bias is stable in n => the published RATIO stands"
            if mean_bias > 1.5
            else "jackknife is unbiased at this scale => the published numbers stand as-is")
        print(f"\nVERDICT: {res['verdict']}")
    OUT.write_text(json.dumps(res, indent=2))
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
