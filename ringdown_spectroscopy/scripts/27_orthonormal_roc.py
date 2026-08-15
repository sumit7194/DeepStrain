"""Is the orthonormal-QNM significance boost INFORMATION, or a reparameterization?

MOTIVATION. arXiv:2605.03576 (Ringdown Analysis of GW250114 with Orthonormal Modes) reports that because QNMs
are not orthogonal, "the inclusion of multiple QNMs induces correlations among them, which can hinder the
robust identification of subdominant QNMs" — and that orthonormalizing lifts the GW250114 first-overtone
significance from 82.5% to 99.9%. That is a large claim about a basis change, and we have a parked negative it
might explain (v4 tone-count, AUC ~0.61). Before adopting it across our arc (L3), settle what it buys.

THE CONCERN. P(A221 != 0) is NOT basis-invariant, whereas a nested-model comparison is; and orthonormalizing
silently changes the implicit prior (flat over (A220, A221) is not flat over the rotated coordinates). So a
significance jump can occur with no gain in information. ROC/AUC against KNOWN injected truth cannot be gamed
this way: it measures detection, not the size of a reported number.

PRE-REGISTERED PREDICTION (stated before running; this is the falsifiable part).
With QNM frequencies/damping FIXED and whitened (white) noise, the model is LINEAR in the amplitudes: each mode
spans a 2-D real quadrature subspace. Overtone detection is then the nested comparison V1 = span{220} vs
V2 = span{220, 221}, whose GLRT is the power in V2 (-) V1 — exactly what Gram-Schmidt returns as the last two
orthonormal vectors. The same quantity appears in the non-orthogonal fit as A221 weighted by the FULL 2x2
covariance block, because the Schur complement of the 220 block of H^T H is the Gram matrix of the
orthogonalized 221 directions. Therefore:

    ==> `orth` and `nonorth_proper` must have IDENTICAL ROC (to floating point).
    ==> If instead the non-orthogonal analysis uses only the DIAGONAL of the covariance (i.e. ignores the
        correlation it is being blamed for), it must be STRICTLY WORSE.

If that holds, the honest reading is: orthonormalization helps only where the non-orthogonal analysis was
mishandling correlations; handled properly, the basis carries no information. `nonorth_naive` is our stand-in
for "hindered by correlations", so the gap orth - naive is the size of the effect that IS real.

THREE STATISTICS (all on identical data, same noise realizations — paired, so differences are not sampling luck)
  orth            power in the Gram-Schmidt complement of the 220 subspace       (the paper's spirit)
  nonorth_proper  A221 whitened by the full 2x2 covariance block                 (correlations handled)
  nonorth_naive   A221 whitened by the diagonal only                             (correlations ignored)

STAGE A (here) is the exact linear case, where the algebra above applies and the prediction is sharp.
STAGE B (28_...) breaks the assumptions — start-time jitter, (M, chi) mismatch, real noise — which is where a
genuine difference could live, and would mean the boost is about ROBUSTNESS TO MISMATCH, not decorrelation.

Run:  .venv/bin/python scripts/27_orthonormal_roc.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sbilib  # noqa: E402  (FS, N_SAMP, CHI_GRID, W220, W221)

MASS, CHI = 68.0, 0.69          # GW250114-like remnant
N_TRIAL = 40_000                # per amplitude, per class
AMP_FRACS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8]
A220 = 8.0                      # whitened amplitude of the fundamental
RESULTS = Path(__file__).resolve().parent.parent / "results"


def qnm_basis(mass, chi, t):
    """The 4 real quadrature templates: [220cos, 220sin, 221cos, 221sin]. Columns of the design matrix."""
    i = min(max(np.searchsorted(sbilib.CHI_GRID, chi), 0), len(sbilib.CHI_GRID) - 1)
    f1, tau1 = sbilib.W220[i][0] / mass, sbilib.W220[i][1] * mass
    f2, tau2 = sbilib.W221[i][0] / mass, sbilib.W221[i][1] * mass
    cols = []
    for f, tau in ((f1, tau1), (f2, tau2)):
        env = np.exp(-t / tau)
        cols += [env * np.cos(2 * np.pi * f * t), env * np.sin(2 * np.pi * f * t)]
    return np.column_stack(cols), (f1, tau1, f2, tau2)


def statistics(H):
    """Precompute the three detection statistics as linear maps on the data (H is fixed in stage A)."""
    G = H.T @ H
    Ginv = np.linalg.inv(G)
    pinv = Ginv @ H.T                       # least-squares amplitude estimator
    C22 = Ginv[2:, 2:]                      # covariance block of the 221 amplitudes
    C22inv = np.linalg.inv(C22)
    Q, _ = np.linalg.qr(H)                  # Gram-Schmidt: Q[:, :2] spans V1, Q[:, 2:] spans V2 (-) V1
    return pinv, C22inv, np.diag(1.0 / np.diag(C22)), Q[:, 2:]


def auc(neg, pos):
    """Rank-based AUC (Mann-Whitney), i.e. P(a signal trial outranks a noise trial)."""
    allv = np.concatenate([neg, pos])
    r = np.empty(len(allv))
    r[np.argsort(allv, kind="mergesort")] = np.arange(1, len(allv) + 1)
    rp = r[len(neg):].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(neg) * len(pos)))


def main() -> None:
    t = np.arange(sbilib.N_SAMP) / sbilib.FS
    H, (f1, tau1, f2, tau2) = qnm_basis(MASS, CHI, t)
    pinv, C22inv, C22diaginv, Qperp = statistics(H)

    # how non-orthogonal ARE they? (the quantity the paper's argument rests on)
    n0, n1 = H[:, 0] / np.linalg.norm(H[:, 0]), H[:, 2] / np.linalg.norm(H[:, 2])
    overlap = float(abs(n0 @ n1))
    print(f"M={MASS} chi={CHI}: 220 = {f1:.1f} Hz / {1e3*tau1:.2f} ms, 221 = {f2:.1f} Hz / {1e3*tau2:.2f} ms")
    print(f"  |<220|221>| = {overlap:.3f}  (nonorthogonality: 0 = independent, 1 = degenerate)")
    print(f"  df = {abs(f1-f2):.1f} Hz over a {1e3*tau2:.2f} ms overtone -> the whole difficulty in one line\n")

    rng = np.random.default_rng(0)
    out = {"mass": MASS, "chi": CHI, "overlap_220_221": overlap, "n_trial": N_TRIAL,
           "a220": A220, "rows": []}

    def stats_for(d):
        """d: (n, N_SAMP). Returns the three statistics, all computed on the SAME data."""
        th = d @ pinv.T                                     # (n, 4) amplitude estimates
        a2 = th[:, 2:]                                      # the 221 quadratures
        return {"orth": np.sum((d @ Qperp) ** 2, axis=1),
                "nonorth_proper": np.einsum("ni,ij,nj->n", a2, C22inv, a2),
                "nonorth_naive": np.einsum("ni,ij,nj->n", a2, C22diaginv, a2)}

    # null class: fundamental only, no overtone. Reused for every amplitude (paired comparison).
    ph = rng.uniform(-np.pi, np.pi, N_TRIAL)
    s220 = A220 * (np.cos(ph)[:, None] * H[:, 0] + np.sin(ph)[:, None] * H[:, 1]) / np.linalg.norm(H[:, 0])
    noise0 = rng.standard_normal((N_TRIAL, sbilib.N_SAMP))
    null = stats_for(s220 + noise0)

    print(f"{'amp_frac':>9} {'overtone SNR':>13} {'|':>2} " + "".join(f"{k:>17}" for k in null))
    for af in AMP_FRACS:
        ph2 = rng.uniform(-np.pi, np.pi, N_TRIAL)
        s221 = A220 * af * (np.cos(ph2)[:, None] * H[:, 2] + np.sin(ph2)[:, None] * H[:, 3]) / np.linalg.norm(H[:, 2])
        # the DETECTABLE overtone SNR is the part orthogonal to the 220 subspace -- what any method can see
        det_snr = float(np.sqrt(np.mean(np.sum((s221 @ Qperp) ** 2, axis=1))))
        d = s220 + s221 + rng.standard_normal((N_TRIAL, sbilib.N_SAMP))
        sig = stats_for(d)
        row = {"amp_frac": af, "overtone_snr_detectable": det_snr,
               "auc": {k: auc(null[k], sig[k]) for k in sig}}
        out["rows"].append(row)
        print(f"{af:>9.2f} {det_snr:>13.2f} {'|':>2} " + "".join(f"{row['auc'][k]:>17.4f}" for k in sig))

    # ---- verdict against the pre-registered prediction --------------------------------------------------
    dmax = max(abs(r["auc"]["orth"] - r["auc"]["nonorth_proper"]) for r in out["rows"])
    naive_gap = [r["auc"]["orth"] - r["auc"]["nonorth_naive"] for r in out["rows"]]
    out["max_abs_auc_diff_orth_vs_proper"] = dmax
    out["max_auc_gain_over_naive"] = max(naive_gap)
    out["prediction_held"] = bool(dmax < 0.005)
    print(f"\nPREDICTION 1 — orth == nonorth_proper: max |dAUC| = {dmax:.5f} -> "
          f"{'HELD: the basis carries NO information' if out['prediction_held'] else 'BROKEN — investigate'}")
    print(f"PREDICTION 2 — orth  >  nonorth_naive: max gain = {max(naive_gap):+.4f} -> "
          f"{'HELD: the real effect is mishandled correlations, not the basis' if max(naive_gap) > 0.005 else 'no gap'}")
    out["conclusion"] = (
        "In the exact linear case an orthonormal QNM basis and a properly-handled non-orthogonal fit are the "
        "SAME detector (identical ROC). Orthonormalization only helps relative to an analysis that ignores the "
        "220/221 covariance. A reported significance can therefore rise with no gain in information, so the "
        "82.5%->99.9% boost cannot be attributed to decorrelation per se without a mismatch mechanism — "
        "which stage B tests.")
    (RESULTS / "27_orthonormal_roc.json").write_text(json.dumps(out, indent=2))
    print("\nwrote 27_orthonormal_roc.json")


if __name__ == "__main__":
    main()
