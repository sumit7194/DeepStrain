"""Is our KS = 0.00214 a measurement, or the floor of our own binning?

THE TECHNIQUE, from `bridge` via the coordination channel: before quoting a small residual, sweep the
numerical floor that could be producing it and check the residual does not track it. They swept a clip
floor five decades and got a band 2254x smaller than the spread they were reporting, which is what licensed
calling it physical rather than a precision limit. Cheap, and the difference between a finding and a floor.

WHAT WE QUOTE. far_zerolag_population reports KS(zero-lag, background) = 0.00214 against a 95% critical
value of 0.00640, and RESULTS.md says the two CDFs "agree to ~0.2% EVERYWHERE". That number is load-bearing:
it is the evidence that H1 _|_ L1 holds across the whole distribution, and the 4,120-yr ladder rests on that
independence. But both CDFs were built by QUANTISING scores into 40,000 bins, and a KS statistic computed
between two binned CDFs cannot resolve differences finer than the bin grid. So 0.00214 could be a real
agreement or an artefact of the grid, and the published claim does not distinguish them.

THE TEST. Accumulate the background ONCE at the finest binning (the expensive part is the 45,073-lag sweep,
not the histogram), then coarsen by summing adjacent bins to obtain every coarser grid for free. Recompute
KS at each. If KS is flat across a wide sweep of bin counts, it is physical. If it tracks the bin width, we
have been quoting our own resolution.

PRE-REGISTERED: KS varying by <20% across a >=32x range of bin counts => physical, the claim stands as
written. A systematic trend with bin width => the number is a floor and RESULTS.md must be re-quoted as an
upper bound rather than a measurement.

Run:  .venv/bin/python scripts/far_ks_clipband.py [--bins 640000]
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
OUT = C.RESULTS_DIR / "far_ks_clipband.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bins", type=int, default=640_000)
    args = ap.parse_args()

    H, L = [], []
    for p in sorted(CACHE.glob("seg_*.npz"), key=lambda q: int(q.stem.split("_")[1])):
        d = np.load(p); H.append(d["h"].astype(np.float64)); L.append(d["l"].astype(np.float64))
    H, L = np.concatenate(H), np.concatenate(L)
    N = len(H)
    Z = H + L
    lo = float(H.min() + L.min()) - 1e-6
    hi = float(H.max() + L.max()) + 1e-6
    fine = args.bins
    scale = fine / (hi - lo)
    print(f"{N} windows | range [{lo:.3f}, {hi:.3f}] | finest grid {fine} bins "
          f"(width {1/scale:.2e})", flush=True)

    counts = np.zeros(fine + 2, dtype=np.int64)
    t0 = time.time()
    for k in range(1, N):
        v = H + np.roll(L, k)
        idx = np.clip(((v - lo) * scale).astype(np.int64) + 1, 0, fine + 1)
        counts += np.bincount(idx, minlength=fine + 2)
        if k % 10000 == 0:
            print(f"    lag {k}/{N-1} ({time.time()-t0:.0f}s)", flush=True)
    zi = np.clip(((Z - lo) * scale).astype(np.int64), 0, fine - 1)
    zcounts = np.bincount(zi, minlength=fine)

    print(f"\n{'bins':>9} {'bin width':>11} {'KS':>10} {'crit95':>9}  verdict")
    rows = []
    crit = 1.358 / np.sqrt(N)
    f = 1
    while fine % f == 0 and fine // f >= 2000:
        nb = fine // f
        bgc = counts[1:-1].reshape(nb, f).sum(axis=1)
        zc = zcounts.reshape(nb, f).sum(axis=1)
        bg_cdf = np.cumsum(bgc) / bgc.sum()
        z_cdf = np.cumsum(zc) / zc.sum()
        ks = float(np.max(np.abs(z_cdf - bg_cdf)))
        rows.append({"bins": int(nb), "bin_width": float((hi - lo) / nb), "ks": ks})
        print(f"{nb:9d} {(hi-lo)/nb:11.2e} {ks:10.5f} {crit:9.5f}  "
              f"{'consistent' if ks < crit else 'DIFFERENT'}")
        f *= 2

    ks_vals = np.array([r["ks"] for r in rows])
    span = float(ks_vals.max() / ks_vals.min())
    nb_span = rows[0]["bins"] / rows[-1]["bins"]
    # does KS track bin width? correlation of log(KS) with log(width) -- a floor would trend, physics wouldn't
    lw = np.log([r["bin_width"] for r in rows]); lk = np.log(ks_vals)
    trend = float(np.polyfit(lw, lk, 1)[0])
    res = {"n_windows": N, "finest_bins": fine, "crit95": float(crit), "rows": rows,
           "ks_span_ratio": span, "bin_count_span": float(nb_span),
           "d_logKS_d_logWidth": trend,
           "published_ks": 0.00214, "published_bins": 40000}
    print(f"\nKS varies {span:.3f}x across a {nb_span:.0f}x range of bin counts")
    print(f"d(log KS)/d(log bin width) = {trend:+.4f}   (0 => independent of the grid; ~1 => a floor)")
    physical = span < 1.20 and abs(trend) < 0.10
    res["physical"] = bool(physical)
    res["verdict"] = ("PHYSICAL — KS is flat across the grid sweep, so 0.00214 is a measurement and the "
                      "'CDFs agree to ~0.2% everywhere' claim stands as written" if physical else
                      "FLOOR — KS tracks the binning; re-quote it as an upper bound, not a measurement")
    print(f"VERDICT: {res['verdict']}")
    OUT.write_text(json.dumps(res, indent=2))
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
