"""Does the stacking validation test the METHOD, or the arithmetic of inverse-variance weighting?

WHAT 12_stacking CLAIMS. "sigma(delta) tightens as sqrt(N) on informative injections (N=8 -> 0.095 vs ideal
0.097, unbiased, calibrated)", gated, and recorded in CLAUDE.md as the solid half of v5. Three structural
problems, none of which the existing checks can detect:

  (S2 IS NEAR-TAUTOLOGICAL). stack() returns sig = 1/sqrt(sum 1/sigma_i^2), and S2 compares that against
  sig_single/sqrt(N). When the per-event sigma_i are similar those are THE SAME FORMULA -- inverse-variance
  weighting gives sigma/sqrt(N) by construction. S2 therefore verifies arithmetic, not that stacking works.
  Measured agreement is ~1% at every N, which is what an identity looks like.

  (S3 CANNOT FAIL AT delta_true = 0). The delta prior is BoxUniform(-0.5, +0.5), centred EXACTLY on zero,
  and every coverage test injects delta_true = 0.0. A posterior that learned nothing and returned the prior
  scores mu ~ 0 and covers every time. Observed coverage is 1.00 at all five N -- and the gate bar is
  0.80 <= coverage <= 1.0, which ADMITS the maximum possible over-coverage. Under a true 90% interval,
  40/40 coverage has probability 0.9^40 = 1.5%; at five values of N it is decisive.

  (THE INJECTIONS MAY NOT BE INFORMATIVE). The prior SD of U(-0.5, 0.5) is 1/sqrt(12) = 0.2887. 12_stacking
  measures sig_single = 0.2754 -- only 4.6% narrower than the prior -- while the code calls this the
  "informative-loudness range" and 09/R2a reports sigma(delta) ~ 0.14 elsewhere. If these posteriors are
  prior-dominated, the sqrt(N) tightening is real arithmetic performed on near-zero information.

  This is the same artefact the v5 stress-test already found for REAL events ("all 7 fainter events return
  the prior"), sitting undetected inside the INJECTION validation that was supposed to certify the method --
  invisible precisely because the test was run where returning the prior is indistinguishable from success.

THE TESTS, each chosen so it CAN fail:
  A. INFORMATION. sigma_single vs the prior SD. Ratio ~1 => the posterior is the prior.
  B. COVERAGE OFF-CENTRE. Repeat the coverage check at delta_true != 0. A prior-dominated posterior is pulled
     toward 0, so coverage must COLLAPSE as |delta_true| grows. At delta_true = 0 it cannot.
  C. EMPIRICAL SCATTER vs CLAIMED SIGMA. 12_ computes mu_s over R realizations and uses only its MEAN,
     discarding the spread -- which is the one quantity that tests whether the claimed sigma is right.
     std(mu_s) / mean(sig_s) ~ 1 => honest; << 1 => the intervals are too wide.

PRE-REGISTERED: ratio_to_prior > 0.85 (A) OR coverage collapsing below 0.5 by |delta|=0.3 (B) means the
stacking validation is certifying prior-return, and the v5 "method validated" claim must be re-scoped to
"the arithmetic is correct". Scatter ratio < 0.7 (C) means the intervals are conservative and the quoted
sigma is not the achieved precision.

Run:  .venv/bin/python scripts/30_stacking_audit.py [--reps 40]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rdlib
import sbilib
from sbilib import Embed  # noqa: F401 -- module-level import is
# LOAD-BEARING: the pickled posterior resolves __main__.Embed at load time (documented gotcha)

RESULTS = Path(__file__).resolve().parent.parent / "results"

OUT = RESULTS / "30_stacking_audit.json"
PRIOR_LO, PRIOR_HI = -0.5, 0.5
PRIOR_SD = (PRIOR_HI - PRIOR_LO) / np.sqrt(12)
DELTAS = (0.0, 0.15, 0.30)
N_GRID = (1, 8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--n-post", type=int, default=300)
    args = ap.parse_args()
    rng = np.random.default_rng(30)

    posterior = torch.load(RESULTS / "09_posterior_150k.pt", weights_only=False)
    T = json.loads((RESULTS / "10_recalibration.json").read_text())["T"]
    print(f"NPE + recalibration T={T} | prior U({PRIOR_LO}, {PRIOR_HI}), SD {PRIOR_SD:.4f}\n", flush=True)

    def delta_post(x_obs):
        s = posterior.sample((args.n_post,), x=x_obs, show_progress_bars=False).numpy()[:, 2]
        s = np.median(s) + T * (s - np.median(s))
        return float(np.mean(s)), float(np.std(s))

    def inject(delta_true):
        m, c = rng.uniform(55, 105), rng.uniform(0.2, 0.9)
        x = torch.tensor(sbilib.simulate(m, c, delta_true, rng).reshape(1, -1))
        return delta_post(x)

    def stack(mus, sigs):
        w = 1.0 / np.asarray(sigs) ** 2
        return float((np.asarray(mus) * w).sum() / w.sum()), float(1.0 / np.sqrt(w.sum()))

    res = {"prior_sd": float(PRIOR_SD), "T": T, "reps": args.reps, "by_delta": {}}

    # ---- A: is a single-event posterior informative at all? ----------------------------------------
    singles = [inject(0.0) for _ in range(30)]
    sig_single = float(np.mean([s for _, s in singles]))
    ratio = sig_single / PRIOR_SD
    res["A_information"] = {"sigma_single": sig_single, "prior_sd": float(PRIOR_SD),
                            "ratio_to_prior": ratio}
    print(f"A. sigma_single {sig_single:.4f} vs prior SD {PRIOR_SD:.4f} -> ratio {ratio:.3f}"
          f"  ({'PRIOR-DOMINATED' if ratio > 0.85 else 'informative'})\n", flush=True)

    # ---- B/C: coverage off-centre, and empirical scatter vs claimed sigma --------------------------
    print(f"{'delta_true':>10} {'N':>3} {'coverage':>9} {'mean(mu)':>9} {'std(mu)':>8} "
          f"{'mean(sig)':>10} {'scatter/claim':>14}")
    for dt in DELTAS:
        res["by_delta"][str(dt)] = {}
        for n in N_GRID:
            mus, sigs, cov = [], [], []
            for r in range(args.reps):
                pair = [inject(dt) for _ in range(n)]
                mu, sig = stack([m for m, _ in pair], [s for _, s in pair])
                mus.append(mu); sigs.append(sig)
                cov.append(abs(mu - dt) < 1.645 * sig)
                rdlib.progress(f"30_d{dt}_N{n}", r + 1, args.reps)
            mus, sigs = np.array(mus), np.array(sigs)
            row = {"coverage": float(np.mean(cov)), "mean_mu": float(mus.mean()),
                   "std_mu": float(mus.std(ddof=1)), "mean_sigma": float(sigs.mean()),
                   "scatter_over_claim": float(mus.std(ddof=1) / sigs.mean()),
                   "bias": float(mus.mean() - dt)}
            res["by_delta"][str(dt)][str(n)] = row
            print(f"{dt:>10.2f} {n:>3} {row['coverage']:>9.2f} {row['mean_mu']:>+9.3f} "
                  f"{row['std_mu']:>8.3f} {row['mean_sigma']:>10.3f} {row['scatter_over_claim']:>14.2f}",
                  flush=True)

    # verdicts
    cov_far = res["by_delta"][str(DELTAS[-1])][str(N_GRID[-1])]["coverage"]
    shrink = [res["by_delta"][str(d)][str(N_GRID[-1])]["bias"] / d for d in DELTAS if d > 0]
    scat = np.mean([res["by_delta"][str(d)][str(n)]["scatter_over_claim"]
                    for d in DELTAS for n in N_GRID])
    res["B_coverage_at_max_delta"] = cov_far
    res["B_mean_relative_bias"] = float(np.mean(shrink))
    res["C_mean_scatter_over_claim"] = float(scat)
    prior_dominated = ratio > 0.85 or cov_far < 0.5
    res["prior_dominated"] = bool(prior_dominated)
    res["intervals_conservative"] = bool(scat < 0.7)
    res["verdict"] = (
        "PRIOR-DOMINATED — the stacking validation certifies prior-return, not method performance; "
        "re-scope v5's claim to 'the inverse-variance arithmetic is correct'" if prior_dominated else
        "the posteriors carry real information off-centre; the stacking validation stands")
    print(f"\nB. coverage at delta={DELTAS[-1]}, N={N_GRID[-1]}: {cov_far:.2f}   "
          f"mean shrinkage of the recovered delta: {np.mean(shrink):+.2f} of truth")
    print(f"C. mean std(mu)/mean(sigma) = {scat:.2f}   "
          f"({'intervals too WIDE' if scat < 0.7 else 'consistent'})")
    print(f"\nVERDICT: {res['verdict']}")
    OUT.write_text(json.dumps(res, indent=2))
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
