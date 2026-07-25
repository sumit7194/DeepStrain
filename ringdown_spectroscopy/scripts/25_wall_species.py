#!/usr/bin/env python
"""Milestone 25 (TheBridge wall-taxonomy): is the no-hair δ wall species-1 (precision) or species-2 (information)?

The bridge classified our δ wall CROSSABLE (species-1: "more SNR moves the Fisher floor") and asked us to try to
falsify that by injecting at escalating SNR. Three legs, PREDICTIONS STATED BEFORE RUNNING (lab notebook):

  LEG 1 (statistical, analytically determined): for h = A·s(shape) with Gaussian noise, block-inversion gives
        (F^-1)_δδ ∝ A^-2 ⇒ σ_Fisher(δ) ∝ 1/SNR EXACTLY. No finite saturation is possible; the only alternative
        is a singular sub-block (σ=∞ at every SNR, a perfect degeneracy — not a floor). PREDICTION: slope of
        log σ vs log SNR = -1.000. This leg VALIDATES the implementation; it cannot discover a species-2 wall.

  LEG 2 (systematic, the interesting one): a real pipeline also carries waveform-model bias. Cutler-Vallisneri:
        Δθ_i = (F^-1)_ij ⟨∂_j h | Δh⟩ with Δh the un-modeled residual. Both factors scale (F^-1 ∝ A^-2,
        ⟨∂h|Δh⟩ ∝ A^2) ⇒ **Δθ is SNR-INDEPENDENT**. PREDICTION: the δ bias from an un-modeled 222 second
        overtone (the published GW250114 analyses fit n=2; our 220+221 model omits it) is flat in SNR.

  LEG 3 (total): total error = sqrt(σ_stat² + Δ_sys²) → Δ_sys as SNR → ∞. If Δ_sys > 0 the TOTAL error
        SATURATES even though the Fisher floor does not. Report the crossover SNR where systematics take over.

Verdict logic: species-1 in the statistical channel (bridge stands) BUT a saturating total-error floor whose
mechanism is model fidelity, not in-channel degeneracy — a distinct third thing the two-option framing misses.
"""
import json
import sys
from pathlib import Path

import numpy as np

import rdlib
import sbilib

RESULTS = Path(__file__).resolve().parent.parent / "results"
FS, N_SAMP = sbilib.FS, sbilib.N_SAMP
K220, K221, K222 = rdlib.KerrMap(2, 2, 0), rdlib.KerrMap(2, 2, 1), rdlib.KerrMap(2, 2, 2)
T = np.arange(N_SAMP) / FS
M0, CHI0, DELTA0, T0_0 = 68.1, 0.68, 0.0, 0.003
PHI = (0.4, -1.1)
DELTA_IDX = 2
NAMES = ["M", "chi", "delta", "t0", "a1_1", "p1_1", "a2_1", "p2_1", "a1_2", "p1_2", "a2_2", "p2_2"]
A222_FRAC = 0.3      # un-modeled 222 amplitude as a fraction of A220 (conservative; NR gives ~0.3-1)


def model_signal(theta, d, amp_scale=1.0):
    """The 220+(1+δ)221 model the pipeline fits (what our NPE assumes)."""
    M, chi, delta, t0 = theta[:4]
    a1, p1, a2, p2 = theta[4 + d * 4 : 8 + d * 4]
    f1, tau1 = K220.f_tau(1.0, chi); f1, tau1 = f1 / M, tau1 * M
    f2, tau2 = K221.f_tau(1.0, chi); f2, tau2 = f2 / M * (1.0 + delta), tau2 * M
    params = [dict(f=f1, tau=tau1, amp=a1 * amp_scale, phi=p1),
              dict(f=f2, tau=tau2, amp=a2 * amp_scale, phi=p2)]
    return rdlib.damped_sinusoids(T, t0, params)


def unmodelled_222(theta, d, amp_scale=1.0):
    """The residual Δh the model CANNOT represent: a real 222 second overtone."""
    M, chi, _, t0 = theta[:4]
    a1 = theta[4 + d * 4]
    f3, tau3 = K222.f_tau(1.0, chi); f3, tau3 = f3 / M, tau3 * M
    return rdlib.damped_sinusoids(T, t0, [dict(f=f3, tau=tau3, amp=a1 * A222_FRAC * amp_scale, phi=0.7)])


def theta_at(amp):
    return np.array([M0, CHI0, DELTA0, T0_0, amp, PHI[0], amp, PHI[1], amp, PHI[0], amp, PHI[1]])


def jac(theta, d, steps, amp_scale=1.0):
    J = np.zeros((N_SAMP, 12))
    for i in range(12):
        if not (i < 4 or 4 + d * 4 <= i < 8 + d * 4):
            continue
        tp, tm = theta.copy(), theta.copy()
        tp[i] += steps[i]; tm[i] -= steps[i]
        J[:, i] = (model_signal(tp, d, amp_scale) - model_signal(tm, d, amp_scale)) / (2 * steps[i])
    return J


def fisher_and_bias(amp, steps):
    """Return (σ_Fisher(δ), Cutler-Vallisneri δ bias from the un-modeled 222, ringdown SNR)."""
    theta = theta_at(amp)
    F = np.zeros((12, 12)); b = np.zeros(12); snr2 = 0.0
    for d in (0, 1):
        J = jac(theta, d, steps)
        F += J.T @ J
        dh = unmodelled_222(theta, d)          # un-modeled residual present in the DATA
        b += J.T @ dh                          # ⟨∂h | Δh⟩
        snr2 += np.sum(model_signal(theta, d) ** 2)
    dsc = 1.0 / np.sqrt(np.diag(F))            # correlation-matrix preconditioning (scale, not physics)
    Finv = (np.outer(dsc, dsc) * np.linalg.inv(F * np.outer(dsc, dsc)))
    sig = float(np.sqrt(Finv[DELTA_IDX, DELTA_IDX]))
    bias = float((Finv @ b)[DELTA_IDX])
    return sig, bias, float(np.sqrt(snr2))


def main() -> None:
    base = np.array([0.02, 0.002, 0.02, 2e-5, 0.05, 0.004, 0.05, 0.004, 0.05, 0.004, 0.05, 0.004])
    amps = [7.0, 14.0, 28.0, 70.0, 140.0, 700.0, 7000.0]   # ~SNR 25 -> 25,000
    rows = []
    print(f"un-modeled residual: 222 second overtone at {A222_FRAC:.0%} of A220\n")
    print(f"{'SNR':>9} | {'sigma_stat(d)':>13} {'sigma*SNR':>10} | {'bias_sys(d)':>11} | {'total err':>9}")
    for a in amps:
        # step sizes for amplitude params scale with the amplitude (relative step held fixed)
        steps = base.copy(); steps[4::2] = 0.05 * (a / 7.0)
        sig, bias, snr = fisher_and_bias(a, steps)
        tot = float(np.hypot(sig, bias))
        rows.append(dict(snr=snr, sigma_stat=sig, sigma_times_snr=sig * snr, bias_sys=bias, total=tot))
        print(f"{snr:>9.0f} | {sig:>13.4f} {sig*snr:>10.2f} | {bias:>11.4f} | {tot:>9.4f}", flush=True)

    # Is any SNR-variation in the bias PHYSICAL or numerical? The CV bias needs a near-degenerate 12x12
    # inversion, so quantify the numerical noise floor: vary the step size at FIXED SNR. If the spread across
    # step sizes is comparable to the spread across SNR, the latter is numerics, not physics.
    ref_biases = []
    for scale in (0.5, 0.7, 1.0, 1.4, 2.0):
        steps = base.copy() * scale; steps[4::2] = 0.05 * scale
        ref_biases.append(fisher_and_bias(7.0, steps)[1])
    num_spread = float((max(ref_biases) - min(ref_biases)) / abs(np.mean(ref_biases)))
    print(f"\n[numerics] δ-bias across a 4x step-size range at FIXED SNR: "
          f"{[round(b,4) for b in ref_biases]} -> spread {num_spread:.1%} (the numerical noise floor)")

    snrs = np.array([r["snr"] for r in rows]); sigs = np.array([r["sigma_stat"] for r in rows])
    biases = np.array([r["bias_sys"] for r in rows])
    slope = float(np.polyfit(np.log(snrs), np.log(sigs), 1)[0])
    bias_spread = float((biases.max() - biases.min()) / np.abs(np.mean(biases)))
    sig_snr_spread = float((np.max(sigs * snrs) - np.min(sigs * snrs)) / np.mean(sigs * snrs))
    sys_floor = float(np.abs(np.mean(biases)))
    crossover_snr = float(np.interp(0, -(sigs - np.abs(biases)), snrs)) if (sigs[-1] < abs(biases[-1])) else float("nan")

    species1 = abs(slope + 1.0) < 0.02
    # the bias counts as SNR-independent if its variation across a 1000x SNR range is no larger than the
    # measured numerical noise floor (step-size spread at fixed SNR) -- i.e. consistent with constant
    bias_is_flat = bias_spread <= max(1.5 * num_spread, 0.05)
    saturates = bias_is_flat and sys_floor > 0.01

    print(f"\nLEG 1 (statistical): d log σ / d log SNR = {slope:+.4f} "
          f"(predicted exactly -1); σ·SNR constant to {sig_snr_spread:.1%}")
    print(f"LEG 2 (systematic): δ bias from un-modeled 222 = {sys_floor:.4f}, "
          f"SNR-variation {bias_spread:.1%} vs numerical floor {num_spread:.1%} -> "
          f"{'SNR-INDEPENDENT (as predicted; variation is numerics)' if bias_is_flat else 'genuinely varies'}")
    print(f"LEG 3 (total): total error -> {sys_floor:.4f} as SNR -> inf; "
          f"statistics stop dominating at SNR ~ {crossover_snr:.0f}")

    verdict = (
        "STATISTICAL CHANNEL = SPECIES-1 (bridge's classification STANDS: sigma_Fisher ∝ 1/SNR exactly, "
        "no in-channel degeneracy). BUT the TOTAL error SATURATES at an SNR-independent waveform-systematic "
        "floor -> more SNR alone does not cross the no-hair wall past that floor; the mechanism is MODEL "
        "FIDELITY (crossable by better waveforms), not in-channel identity (species-2)."
        if species1 and saturates else
        ("species-1, no systematic saturation found" if species1 else "NOT species-1 -- Fisher scaling broke")
    )
    print(f"\nVERDICT: {verdict}")
    (RESULTS / "25_wall_species.json").write_text(json.dumps(
        {"a222_frac": A222_FRAC, "rows": rows, "loglog_slope": slope, "sigma_snr_spread": sig_snr_spread,
         "bias_snr_spread": bias_spread, "bias_numerical_spread": num_spread,
         "bias_is_flat": bool(bias_is_flat), "systematic_floor_delta": sys_floor, "crossover_snr": crossover_snr,
         "statistical_species1": bool(species1), "total_error_saturates": bool(saturates),
         "verdict": verdict}, indent=2))
    print("wrote 25_wall_species.json")


if __name__ == "__main__":
    main()
