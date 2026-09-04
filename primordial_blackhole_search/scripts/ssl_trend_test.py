"""The pre-registered test of the SSL data-wall trend -- written BEFORE the n=20 result exists.

WHY IT IS A SEPARATE SCRIPT, AND WHY IT IS COMMITTED EARLY. RESULTS.md declares the statistic, the bar and
the decomposition for the n=20 run. Declaring an analysis and then writing the code that performs it after
seeing the numbers leaves every small choice -- resample unit, tie handling, which cells enter the
decomposition -- open at exactly the moment they stop being innocent. So the code goes in first, against the
n=5 pilot, and the n=20 artifact is passed to the same unmodified script when it lands.

THE ORDERING PROBLEM THIS DOES NOT SOLVE, stated because it is real. The bootstrap became the primary test
AFTER the normal-theory number came in at 2.94 against a bar of 3. A test adopted after a near miss, which
then reports a friendlier p-value on the same data, is the shape of a test chosen for its answer. The defence
is that the FIRST write-up named "an error model that respects the censoring" as the thing that would settle
it, before any bootstrap had been run -- but the defence had to exist, which is why the primary analysis is
restricted to seeds the pilot never drew.

WHAT IS PRE-REGISTERED (RESULTS.md, "PRE-REGISTRATION II"):
  primary        fresh seeds only; bootstrap over seeds within budget, 10k resamples,
                 p = fraction of draws with gap <= 0
  bar            p < 0.0027 resolved | 0.0027-0.05 suggestive | >= 0.05 not resolved
  secondary      pooled with the pilot, labelled as such; normal-theory sigma for comparability
  decomposition  P(clear the 1%-FAR floor) and mean-given-cleared, per budget, reported regardless --
                 if the trend is carried by floor-clearing alone, "SSL buys sensitive distance" is the
                 wrong description of it

Run:  .venv/bin/python scripts/ssl_trend_test.py results/ssl_sensdist_seeds20.json
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results"
# The 5-seed run, not the original 2-seed one: the June artifact predates the retention fix and stores only
# means, so it cannot be bootstrapped at all. That absence is what prompted this whole line of work.
PILOT = RESULTS / "ssl_sensdist_seeds5.json"
N_BOOT = 10_000
BAR_RESOLVED, BAR_SUGGESTIVE = 0.0027, 0.05
LO, HI = "2000", "8000"


def cell(d, budget):
    f = d["results"][budget]["FAR1pct"]
    return np.asarray(f["scratch_per_seed"], float), np.asarray(f["ssl_per_seed"], float)


def gap_of(lo_s, lo_l, hi_s, hi_l):
    return (lo_l.mean() - lo_s.mean()) - (hi_l.mean() - hi_s.mean())


def bootstrap(d, rng):
    """Resample SEEDS within each cell. Zeros resample as zeros, so the censoring is carried, not modelled."""
    cells = [*cell(d, LO), *cell(d, HI)]
    draws = np.empty(N_BOOT)
    for b in range(N_BOOT):
        draws[b] = gap_of(*[c[rng.integers(0, len(c), len(c))] for c in cells])
    return gap_of(*cells), draws


def normal_sigma(d):
    out = []
    for budget in (LO, HI):
        s, l = cell(d, budget)
        out.append((l.mean() - s.mean(), math.hypot(s.std(ddof=1), l.std(ddof=1)) / math.sqrt(len(s))))
    gap = out[0][0] - out[1][0]
    se = math.hypot(out[0][1], out[1][1])
    return gap, se, gap / se


def decompose(d):
    """Split the censored metric: how OFTEN a model clears the floor, vs how far it gets once it does."""
    rows = {}
    for budget in sorted(d["results"], key=int):
        r = {}
        for tag, arr in zip(("scratch", "ssl"), cell(d, budget)):
            cleared = arr > 0
            r[tag] = {"p_clear": float(cleared.mean()),
                      "mean_given_cleared": float(arr[cleared].mean()) if cleared.any() else None,
                      "n": int(arr.size)}
        rows[budget] = r
    return rows


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else PILOT
    d = json.loads(path.read_text())
    fresh = d.get("seed_offset", 0) > 0
    rng = np.random.default_rng(20260904)

    print(f"artifact {path.name}  seeds={d['seeds']} values={d.get('seed_values', 'unrecorded')}")
    print(f"role: {'PRIMARY (fresh seeds)' if fresh else 'PILOT (seeds the follow-up does not reuse)'}\n")

    gap, draws = bootstrap(d, rng)
    p = float((draws <= 0).mean())
    ci = [float(np.percentile(draws, 0.5)), float(np.percentile(draws, 99.5))]
    n_gap, n_se, n_sigma = normal_sigma(d)
    verdict = ("RESOLVED" if p < BAR_RESOLVED else
               "SUGGESTIVE" if p < BAR_SUGGESTIVE else "NOT RESOLVED")

    print(f"gap (delta@{LO} - delta@{HI})   {gap:+.4f}")
    print(f"bootstrap p(gap <= 0)          {p:.5f}   99% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]  ({N_BOOT} draws)")
    print(f"normal-theory (comparability)  {n_gap:+.4f} +- {n_se:.4f} = {n_sigma:.2f} sigma")
    print(f"\nPRE-REGISTERED VERDICT: {verdict}   "
          f"(bar: p<{BAR_RESOLVED} resolved | <{BAR_SUGGESTIVE} suggestive)")

    print("\ndecomposition -- does the trend come from CLEARING the floor or from distance once cleared?")
    print(f"{'budget':>7} {'model':>8} {'P(clear)':>9} {'mean|cleared':>13} {'n':>4}")
    dec = decompose(d)
    for budget, r in dec.items():
        for tag in ("scratch", "ssl"):
            m = r[tag]["mean_given_cleared"]
            print(f"{budget:>7} {tag:>8} {r[tag]['p_clear']:>9.2f} "
                  f"{('n/a' if m is None else f'{m:.3f}'):>13} {r[tag]['n']:>4}")

    lo = dec[LO]
    driver = ("floor-clearing" if lo["scratch"]["p_clear"] == 0.0 and lo["ssl"]["p_clear"] > 0.5
              else "mixed")
    print(f"\nat {LO} labels the gain is driven by: {driver}"
          + (" -- 'SSL buys sensitive distance' overstates it; SSL makes the difference between a model that "
             "reaches the 1%-FAR floor at all and one that does not" if driver == "floor-clearing" else ""))

    out = RESULTS / f"ssl_trend_test{'_seeds' + str(d['seeds']) if fresh else '_pilot'}.json"
    out.write_text(json.dumps({"artifact": path.name, "seeds": d["seeds"],
                               "seed_values": d.get("seed_values"), "role": "primary" if fresh else "pilot",
                               "gap": gap, "bootstrap_p": p, "ci99": ci, "n_boot": N_BOOT,
                               "normal_gap": n_gap, "normal_se": n_se, "normal_sigma": n_sigma,
                               "verdict": verdict, "decomposition": dec, "driver_at_low_budget": driver},
                              indent=2))
    print(f"\nwrote {out.name}")


if __name__ == "__main__":
    main()
