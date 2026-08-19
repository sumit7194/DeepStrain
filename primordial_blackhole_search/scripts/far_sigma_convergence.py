"""Does the threshold's error bar even EXIST? Testing for a heavy tail before quoting a sigma.

THE OBSERVATION THAT FORCED THIS. Two independent runs of far_effective_n disagreed on sigma at n=320 by
1.5x (1.52 vs 2.27), and raising the rep count 20 -> 40 moved sigma SYSTEMATICALLY UP at every size
(1/decade at n=40: 2.13 -> 3.14; n=80: 1.99 -> 2.46), not scattered either way. A sample standard deviation
that grows with sample size is the classic signature of a distribution whose variance is infinite or nearly
so -- in which case sigma is not a noisy estimate of something real, it is an artifact of how many samples
were drawn, and NO amount of data makes it converge.

WHY THIS DECIDES A WRITE-UP AND NOT JUST A NUMBER. Every precision claim in this arc is a sigma or a range:
the original audit's +-33-44%, L2's +-10-12%, and the +-1.7 I was about to publish as the correction. If the
sampling distribution of the threshold is heavy-tailed with index < 2, all of them are ill-defined -- the
right summary is a QUANTILE of the sampling distribution, which exists regardless.

THREE TESTS, and the third is the one that decides practice:
  1. sigma vs rep count m: computed on the first m of M draws. GROWS => heavy tail; FLATTENS => sigma exists.
  2. Hill tail index on the upper tail of the threshold draws. alpha_tail < 2 => infinite variance;
     < 1 => the mean does not exist either.
  3. sigma vs the 5-95% INTERQUANTILE RANGE, both as functions of m. A quantile spread is well-defined for
     any distribution, so if the IQ range flattens while sigma keeps climbing, that is direct evidence to
     quote quantiles and stop quoting sigma.

PRE-REGISTERED:
  * Hill index < 2 (90% CI upper bound below 2) => infinite variance. THIS IS THE PRIMARY DISCRIMINATOR;
    it is a direct estimate of the tail index rather than an inference from estimator behaviour.
  * sigma's LATE-HALF slope materially exceeding the IQ range's LATE-HALF slope => sigma still climbing where
    a well-defined spread has settled.

  A RULE THAT FIRED FALSELY, AND WHY (kept as a warning). The first version compared sigma's growth over the
  WHOLE range m=10..200 against the IQ range's drift over the LATE HALF only, and declared "heavy-tailed" for
  all three rungs. That is apples to oranges: a sample SD is downward-biased at small m and climbs toward its
  true value for ANY distribution, so growth from m=10 is expected and carries no tail information. Compared
  like-for-like, sigma's late slope (+0.03) was actually FLATTER than the IQ range's (+0.06) -- the opposite
  of the signature -- and the Hill index came back 20-195, i.e. a very LIGHT tail. The verdict was an artifact
  of the comparison window, not a property of the data.

Run:  .venv/bin/python scripts/far_sigma_convergence.py [--n 160] [--draws 200]
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
OUT = C.RESULTS_DIR / "far_sigma_convergence.json"
YEAR_S = 3.156e7
WIN_SEC = 64.0
FARS = (("1/month", 12.0), ("1/year", 1.0), ("1/decade", 0.1))


def load_segments():
    Hs, Ls = [], []
    for p in sorted(CACHE.glob("seg_*.npz"), key=lambda q: int(q.stem.split("_")[1])):
        d = np.load(p); Hs.append(d["h"].astype(np.float32)); Ls.append(d["l"].astype(np.float32))
    return Hs, Ls


def ladder(H, L):
    N = len(H)
    bg_yr = (N - 1) * N * WIN_SEC / YEAR_S
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
        if 1 <= k <= len(top):
            out[label] = float(top[-k])
    return out


def hill(x, k=None):
    """Hill estimator of the tail index on the k largest order statistics (+ its standard error)."""
    x = np.sort(np.asarray(x, float))
    x = x[x > 0]
    if len(x) < 20:
        return None, None
    k = k or max(10, len(x) // 5)
    k = min(k, len(x) - 1)
    tail = x[-(k + 1):]
    g = float(np.mean(np.log(tail[1:]) - np.log(tail[0])))
    return (1.0 / g if g > 0 else None), (1.0 / g / np.sqrt(k) if g > 0 else None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=160)
    ap.add_argument("--draws", type=int, default=200)
    args = ap.parse_args()

    Hs, Ls = load_segments()
    pool = len(Hs)
    rng = np.random.default_rng(31)
    print(f"pool {pool} | subset size n={args.n} | {args.draws} independent draws\n", flush=True)

    thr = {lbl: [] for lbl, _ in FARS}
    t0 = time.time()
    for r in range(args.draws):
        idx = rng.choice(pool, args.n, replace=False)
        for lbl, v in ladder(np.concatenate([Hs[i] for i in idx]),
                             np.concatenate([Ls[i] for i in idx])).items():
            thr[lbl].append(v)
        if (r + 1) % 25 == 0:
            print(f"  {r+1}/{args.draws} draws ({time.time()-t0:.0f}s)", flush=True)

    res = {"n": args.n, "draws": args.draws, "pool": pool, "by_far": {}}
    ms = [m for m in (10, 20, 40, 80, 120, 160, 200) if m <= args.draws]
    for lbl, _ in FARS:
        t = np.array(thr[lbl])
        if len(t) < 40:
            continue
        sds = [float(np.std(t[:m], ddof=1)) for m in ms]
        iqs = [float(np.percentile(t[:m], 95) - np.percentile(t[:m], 5)) for m in ms]
        hi, hse = hill(t - t.min() + 1e-9)
        growth = sds[-1] / sds[0]
        late = (sds[-1] - sds[len(sds) // 2]) / sds[len(sds) // 2]
        iq_drift = abs(iqs[-1] - iqs[len(iqs) // 2]) / iqs[len(iqs) // 2]
        res["by_far"][lbl] = {"m": ms, "sigma": sds, "iq90": iqs, "hill": hi, "hill_se": hse,
                              "sigma_growth": growth, "late_slope": late, "iq_drift": iq_drift,
                              "draws": [float(x) for x in t],      # retained: re-analysis must not need a re-run
                              "median": float(np.median(t)),
                              "q05_q95": [float(np.percentile(t, 5)), float(np.percentile(t, 95))]}
        print(f"\n{lbl}   (median {np.median(t):.3f})")
        print("  m      :" + "".join(f"{m:8d}" for m in ms))
        print("  sigma  :" + "".join(f"{v:8.3f}" for v in sds))
        print("  IQ90   :" + "".join(f"{v:8.3f}" for v in iqs))
        print(f"  sigma growth over m: {growth:.2f}x | late-half slope {late:+.2f} | "
              f"IQ90 drift {iq_drift:+.2f}")
        if hi:
            print(f"  Hill tail index {hi:.2f} +- {hse:.2f}  "
                  f"({'INFINITE variance (<2)' if hi + 1.64*hse < 2 else 'variance may exist'})")

    # like-for-like (both late-half), with the Hill index as the primary and independent discriminator
    heavy = [k for k, v in res["by_far"].items()
             if (v["hill"] is not None and v["hill"] + 1.64 * (v["hill_se"] or 0) < 2.0)
             or (v["late_slope"] > v["iq_drift"] + 0.10)]
    res["heavy_tailed_rungs"] = heavy
    res["verdict"] = (f"HEAVY-TAILED at {heavy}: sigma does not converge while the quantile spread does — "
                      "quote the 5-95% range, not a +-sigma" if heavy else
                      "sigma CONVERGES (light tail, Hill >> 2; late-half slope flat) — a +-sigma is a "
                      "legitimate summary, and the small-m growth is ordinary SD bias, not a heavy tail")
    print(f"\nVERDICT: {res['verdict']}")
    OUT.write_text(json.dumps(res, indent=2))
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
