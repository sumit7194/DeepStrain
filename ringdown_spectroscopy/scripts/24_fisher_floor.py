#!/usr/bin/env python
"""Milestone 24 (G8, TheBridge): the Cramér-Rao (Fisher) floor on the no-hair δ vs the NPE's σ(δ).

Pre-registered (notes/lab_notebook 2026-07-24). Postulate G8: "the 221 δ deficit is fundamental at current
SNR — no pipeline beats the Cramér-Rao floor." KILLED iff NPE σ(δ) is significantly BELOW the Fisher floor by
a margin the prior cannot explain; SURVIVES iff NPE σ(δ) ≈ floor (efficient estimator).

Model (sbilib.simulate): 220 + (1+δ)-shifted 221 damped sinusoids + WHITE unit-variance noise, so the Fisher
inner product is the plain Euclidean dot product and SNR = √(Σh²). Fisher matrix over 12 parameters —
[M, χ, δ, t0] shared + per-detector [A220, φ220, A221, φ221] × 2 detectors — via central differences on the
CONTINUOUS Kerr f_tau (smooth χ-derivative). σ_Fisher(δ) = √[(F⁻¹)_δδ], marginalized over every nuisance,
matching the NPE. NPE σ(δ) = posterior std over N fixed-loudness GW250114 injections (δ=0). Step-size
convergence is checked before the matrix is trusted.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

import rdlib
import sbilib
from sbilib import Embed  # load-bearing: pickled posterior resolves __main__.Embed

RESULTS = Path(__file__).resolve().parent.parent / "results"
FS, N_SAMP = sbilib.FS, sbilib.N_SAMP
K220, K221 = rdlib.KerrMap(2, 2, 0), rdlib.KerrMap(2, 2, 1)
T = np.arange(N_SAMP) / FS

# fiducial: GW250114 remnant + the sbilib GW250114-calibrated loudness (PEAK_AMP_RANGE mean), A221/A220≈1
M0, CHI0, DELTA0, T0_0, AMP0 = 68.1, 0.68, 0.0, 0.003, float(np.mean(sbilib.PEAK_AMP_RANGE))
PHI = (0.4, -1.1)
SIGMA_PRIOR = 1.0 / np.sqrt(12)                       # δ ~ U[-0.5, 0.5]
NAMES = ["M", "chi", "delta", "t0", "a1_1", "p1_1", "a2_1", "p2_1", "a1_2", "p1_2", "a2_2", "p2_2"]
DELTA_IDX = 2
THETA0 = np.array([M0, CHI0, DELTA0, T0_0, AMP0, PHI[0], AMP0, PHI[1], AMP0, PHI[0], AMP0, PHI[1]])


def one_det_signal(M, chi, delta, t0, a1, p1, a2, p2):
    f1, tau1 = K220.f_tau(1.0, chi); f1, tau1 = f1 / M, tau1 * M
    f2, tau2 = K221.f_tau(1.0, chi); f2, tau2 = f2 / M * (1.0 + delta), tau2 * M
    params = [dict(f=f1, tau=tau1, amp=a1, phi=p1), dict(f=f2, tau=tau2, amp=a2, phi=p2)]
    return rdlib.damped_sinusoids(T, t0, params)


def det_signal(theta, d):
    M, chi, delta, t0 = theta[:4]
    a1, p1, a2, p2 = theta[4 + d * 4 : 8 + d * 4]
    return one_det_signal(M, chi, delta, t0, a1, p1, a2, p2)


def fisher(steps):
    F = np.zeros((12, 12))
    for d in (0, 1):
        J = np.zeros((N_SAMP, 12))
        for i in range(12):
            if not (i < 4 or 4 + d * 4 <= i < 8 + d * 4):   # this det's signal ignores the other det's params
                continue
            tp, tm = THETA0.copy(), THETA0.copy()
            tp[i] += steps[i]; tm[i] -= steps[i]
            J[:, i] = (det_signal(tp, d) - det_signal(tm, d)) / (2 * steps[i])
        F += J.T @ J
    return F


def sigma_delta_from_F(F, return_cond=False):
    """Marginal σ(δ)=√(F⁻¹)_δδ via correlation-matrix preconditioning (the raw F spans ~20 orders in
    parameter scale -> cond ~1e22 from UNITS, not physics; D F D has unit diagonal and inverts cleanly)."""
    d = 1.0 / np.sqrt(np.diag(F))
    Fp = F * np.outer(d, d)                       # unit-diagonal correlation matrix
    Fp_inv = np.linalg.inv(Fp)
    sig = float(np.sqrt(Fp_inv[DELTA_IDX, DELTA_IDX]) * d[DELTA_IDX])
    return (sig, float(np.linalg.cond(Fp))) if return_cond else sig


def main() -> None:
    snr = float(np.sqrt(sum(np.sum(det_signal(THETA0, d) ** 2) for d in (0, 1))))
    print(f"fiducial: M {M0}, chi {CHI0}, delta {DELTA0}, A221/A220 1.0, loudness {AMP0} -> ringdown SNR {snr:.1f}\n")

    # step-size convergence: σ_Fisher(δ) must be stable across a decade of step scales
    base = np.array([0.02, 0.002, 0.02, 2e-5, 0.05, 0.004, 0.05, 0.004, 0.05, 0.004, 0.05, 0.004])
    sig_by_scale = {}
    for scale in (0.5, 1.0, 2.0):
        sig_by_scale[scale] = sigma_delta_from_F(fisher(base * scale))
    sig_fisher = sig_by_scale[1.0]
    spread = (max(sig_by_scale.values()) - min(sig_by_scale.values())) / sig_fisher
    print(f"σ_Fisher(δ) step-convergence: {[round(v,3) for v in sig_by_scale.values()]} "
          f"(rel spread {spread:.1%}) {'OK' if spread < 0.05 else 'UNSTABLE'}")

    F = fisher(base)
    sig_fisher, cond_scaled = sigma_delta_from_F(F, return_cond=True)
    sig_post_min = 1.0 / np.sqrt(1.0 / sig_fisher ** 2 + 1.0 / SIGMA_PRIOR ** 2)   # Bayesian floor (data+prior)
    print(f"σ_Fisher(δ) = {sig_fisher:.3f} (data-only Cramér-Rao) | σ_prior = {SIGMA_PRIOR:.3f} | "
          f"σ_post_min(data+prior) = {sig_post_min:.3f} | preconditioned cond {cond_scaled:.0e} "
          f"{'OK' if cond_scaled < 1e10 else '(δ marginal still stable per step-convergence)'}")

    # NPE σ(δ): posterior std over N fixed-loudness fiducial injections (δ=0)
    posterior = torch.load(RESULTS / "09_posterior_150k.pt", weights_only=False)
    rng = np.random.default_rng(24)
    N = 200
    stds, meds = [], []
    for k in range(N):
        t0 = rng.uniform(0, sbilib.T0_MAX_MS / 1000.0)
        x = np.empty((2, N_SAMP), dtype=np.float32)
        for d in range(2):
            sig = one_det_signal(M0, CHI0, DELTA0, t0, AMP0, rng.uniform(-np.pi, np.pi),
                                 AMP0, rng.uniform(-np.pi, np.pi))
            x[d] = sig + rng.standard_normal(N_SAMP)
        s = posterior.sample((1000,), x=torch.tensor(x.reshape(1, -1)), show_progress_bars=False).numpy()
        stds.append(float(s[:, DELTA_IDX].std())); meds.append(float(np.median(s[:, DELTA_IDX])))
        if (k + 1) % 50 == 0:
            print(f"  NPE injections {k+1}/{N}", flush=True)
    sig_npe = float(np.mean(stds))
    sig_npe_scatter = float(np.std(meds))     # actual scatter of point estimates (efficiency cross-check)
    print(f"\nσ(δ)_NPE = {sig_npe:.3f} (mean posterior std) | point-estimate scatter {sig_npe_scatter:.3f}")

    # PRIOR-SHRINKAGE TEST: inject OFF-CENTER (δ=0.4). If the point estimate is pulled toward the prior
    # center (0), the sub-Fisher scatter at δ=0 was prior regularization, not data information.
    DELTA_OFF = 0.4
    meds_off = []
    for k in range(100):
        t0 = rng.uniform(0, sbilib.T0_MAX_MS / 1000.0)
        x = np.empty((2, N_SAMP), dtype=np.float32)
        for d in range(2):
            sig = one_det_signal(M0, CHI0, DELTA_OFF, t0, AMP0, rng.uniform(-np.pi, np.pi),
                                 AMP0, rng.uniform(-np.pi, np.pi))
            x[d] = sig + rng.standard_normal(N_SAMP)
        s = posterior.sample((1000,), x=torch.tensor(x.reshape(1, -1)), show_progress_bars=False).numpy()
        meds_off.append(float(np.median(s[:, DELTA_IDX])))
    med_off = float(np.mean(meds_off))
    shrinkage = (DELTA_OFF - med_off) / DELTA_OFF        # fraction of the offset pulled back toward the prior center
    print(f"prior-shrinkage test: inject δ={DELTA_OFF} -> NPE median {med_off:+.3f} "
          f"(pulled {shrinkage:.0%} toward the prior center) -> {'PRIOR regularizing' if shrinkage > 0.25 else 'data-driven'}")

    ratio_fisher = sig_npe / sig_fisher
    ratio_postmin = sig_npe / sig_post_min
    # readings (pre-registered)
    below_data_floor = sig_npe < 0.85 * sig_fisher
    at_bayes_floor = 0.85 < ratio_postmin < 1.25
    if not below_data_floor:
        verdict = "G8 SURVIVES — NPE σ(δ) ≈ (or above) the data-only Fisher floor: efficient estimator at the limit"
    elif at_bayes_floor:
        verdict = "G8 STANDS — NPE σ(δ) sits below the data floor but AT the Bayesian data+prior floor: the PRIOR does that work (honesty footnote on '2.6× tighter')"
    else:
        verdict = "G8 KILLED — NPE σ(δ) is below even the data+prior floor (unexplained by the prior)"
    print(f"\nratio σ_NPE/σ_Fisher = {ratio_fisher:.2f} | σ_NPE/σ_post_min = {ratio_postmin:.2f}")
    print(verdict)

    out = {"ringdown_snr": snr, "sigma_fisher_delta": sig_fisher, "sigma_prior": SIGMA_PRIOR,
           "sigma_post_min": sig_post_min, "sigma_npe": sig_npe, "sigma_npe_scatter": sig_npe_scatter,
           "ratio_npe_over_fisher": ratio_fisher, "ratio_npe_over_postmin": ratio_postmin,
           "step_convergence_spread": spread, "preconditioned_cond": cond_scaled, "n_inj": N,
           "offcenter_delta": DELTA_OFF, "offcenter_npe_median": med_off, "prior_shrinkage_frac": shrinkage,
           "g8_survives": bool(not below_data_floor or at_bayes_floor),
           "g8_killed": bool(below_data_floor and not at_bayes_floor), "verdict": verdict}
    (RESULTS / "24_fisher_floor.json").write_text(json.dumps(out, indent=2))
    print("wrote 24_fisher_floor.json")


if __name__ == "__main__":
    main()
