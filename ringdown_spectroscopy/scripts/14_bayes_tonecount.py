"""R2 (PLAN.md): explicit BAYESIAN tone-count model selection — 1-tone vs 2-tone, the field's method.

Our v4 ML tone-count classifier was a parked NEGATIVE (information-limited; P(2-tone|GW250114)=0.32, an
overfitting mirage once fixed → calibrated-but-weak AUC~0.61). The guardrail sanctions the non-ML route:
EXPLICIT Bayesian model selection. The ringdown amplitudes are LINEAR given (M, χ, t0), so the evidence
marginalizes them ANALYTICALLY (linear-Gaussian); the 3 nonlinear params (M, χ, and the start time t0 —
the latter MARGINALIZED, which is the crux of the overtone controversy) are integrated on a grid.

logB_21 = log[Z(2-tone)/Z(1-tone)] is CALIBRATED on injections (1-tone vs 2-tone at a sweep of overtone
SNR) before being applied to the real event — we trust the calibrated separation, not the absolute Bayes
factor (whose Occam term depends on the amplitude prior σ_a; σ_a-sensitivity is reported).

*** OUTCOME (2026-06-25): this SIMPLIFIED version is NOT a fair test — PARKED, see notes/lab_notebook.md. ***
The oracle diagnostic (Bayes factor at the TRUE M,χ,t0) shows NO 1-tone/2-tone separation at any σ_a — the
220/221 modes are near-degenerate over the 0.04 s segment (4 Hz apart; only the damping differs), so the
2-tone model just flexibly fits the loud 220 regardless of a real overtone. BUT the PUBLISHED GW250114
analysis (arXiv:2509.08099) DOES detect the overtone, so the failure here is an IMPLEMENTATION limit of the
simplified machinery (time-domain, white-noise likelihood, independent per-detector amplitudes, flat σ_a),
NOT the information limit. A fair R2 needs the proper frequency-domain coherent pipeline with physical
priors (the `ringdown` package — Python 3.11, deferred). Kept as the reproducible diagnostic; NOT gated.

Run:  .venv/bin/python scripts/14_bayes_tonecount.py [--n-cal 60] [--smoke]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp

import rdlib
import sbilib
from sbilib import CHI_GRID, FS, N_SAMP, W220, W221

RESULTS = Path(__file__).resolve().parent.parent / "results"
PLOTS = Path(__file__).resolve().parent.parent / "plots"
EVENT = "GW250114_082203"
SIGMA_A = 8.0                         # amplitude prior scale [whitened-σ]; PEAK_AMP_RANGE is (2,12)
T = np.arange(N_SAMP) / FS


def design(mass, chi, t0, two_tone):
    """N×k basis of damped cos/sin at the Kerr (f,τ) — same W220/W221 convention as the injection."""
    i = min(max(np.searchsorted(CHI_GRID, chi), 0), len(CHI_GRID) - 1)
    f1, tau1 = W220[i][0] / mass, W220[i][1] * mass
    d = T - t0
    m = (d >= 0).astype(float)
    cols = [np.exp(-np.clip(d, 0, None) / tau1) * np.cos(2 * np.pi * f1 * d) * m,
            np.exp(-np.clip(d, 0, None) / tau1) * np.sin(2 * np.pi * f1 * d) * m]
    if two_tone:
        f2, tau2 = W221[i][0] / mass, W221[i][1] * mass
        cols += [np.exp(-np.clip(d, 0, None) / tau2) * np.cos(2 * np.pi * f2 * d) * m,
                 np.exp(-np.clip(d, 0, None) / tau2) * np.sin(2 * np.pi * f2 * d) * m]
    return np.stack(cols, axis=1)


def _ll_eta(y, G, sigma_a):
    """η-dependent part of the linear-Gaussian log-evidence (constants N·log2π, yᵀy cancel in logB)."""
    A = G.T @ G
    b = G.T @ y
    S = np.eye(G.shape[1]) + sigma_a ** 2 * A
    _, logdet = np.linalg.slogdet(S)
    quad = sigma_a ** 2 * (b @ np.linalg.solve(S, b))
    return -0.5 * (logdet - quad)


def log_bayes(segs, m_grid, chi_grid, t0_grid, sigma_a=SIGMA_A):
    """logB_21 = log Z(2-tone) - log Z(1-tone), each marginalized over (M, χ, t0) on the grid.
    segs: list of whitened detector segments (independent amplitudes per detector → sum log-evidence)."""
    z = {True: [], False: []}
    for mass in m_grid:
        for chi in chi_grid:
            for t0 in t0_grid:
                for two in (False, True):
                    G = design(mass, chi, t0, two)
                    z[two].append(sum(_ll_eta(y, G, sigma_a) for y in segs))
    return float(logsumexp(z[True]) - logsumexp(z[False]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cal", type=int, default=60)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    n_cal = 8 if args.smoke else args.n_cal
    rng = np.random.default_rng(0)

    # grids (coarser in smoke); start time marginalized over the first few ms
    if args.smoke:
        m_grid, chi_grid, t0_grid = np.linspace(60, 90, 12), np.linspace(0.5, 0.85, 8), np.linspace(0, 0.0015, 4)
    else:
        m_grid, chi_grid, t0_grid = np.linspace(55, 100, 34), np.linspace(0.4, 0.9, 16), np.linspace(0, 0.002, 7)
    M_T, CHI_T = 76.0, 0.76                 # GW250114-class remnant
    print(f"grid {len(m_grid)}×{len(chi_grid)}×{len(t0_grid)}, σ_a={SIGMA_A}, n_cal={n_cal}", flush=True)

    # --- CALIBRATION: logB for 1-tone and 2-tone injections across a sweep of overtone SNR ---
    AMP_FRACS = [0.0, 0.3, 0.5, 0.8]        # 221/220 ratio; 0.0 == 1-tone
    cal = {}
    for af in AMP_FRACS:
        n_tones = 1 if af == 0.0 else 2
        logBs, snrs = [], []
        for k in range(n_cal):
            x, osnr = sbilib.simulate_tonecount(M_T, CHI_T, n_tones, af, rng)
            logBs.append(log_bayes([x[0], x[1]], m_grid, chi_grid, t0_grid))
            snrs.append(osnr)
            rdlib.progress("14_bayes_tonecount", len(cal) * n_cal + k + 1, len(AMP_FRACS) * n_cal + 1)
        cal[af] = {"logB": logBs, "overtone_snr": float(np.median(snrs))}
        print(f"  amp_frac={af} (overtone SNR≈{np.median(snrs):4.1f}, {n_tones}-tone): "
              f"logB median={np.median(logBs):+6.2f}  [{np.percentile(logBs,10):+.1f},{np.percentile(logBs,90):+.1f}]",
              flush=True)

    # operating threshold = midpoint that best separates 1-tone (af=0) from the loudest 2-tone
    b1 = np.array(cal[0.0]["logB"]); b2 = np.array(cal[AMP_FRACS[-1]]["logB"])
    thr = float(0.5 * (np.median(b1) + np.median(b2)))
    # detection efficiency vs overtone SNR at this threshold
    eff = {af: float(np.mean(np.array(cal[af]["logB"]) > thr)) for af in AMP_FRACS}
    fpr = eff[0.0]                                                  # 1-tone called 2-tone (false positive)
    print(f"\nthreshold logB>{thr:+.2f}: false-positive (1-tone) {fpr:.2f}; "
          f"2-tone efficiency " + " ".join(f"af{af}:{eff[af]:.2f}" for af in AMP_FRACS[1:]))

    # --- REAL GW250114 ---
    gps = rdlib.event_gps(EVENT)
    segs = []
    for det in ("H1", "L1"):
        white = rdlib.fetch_whitened(det, gps, bandpass=False)
        pk = rdlib.find_peak(white.bandpass(*rdlib.BAND), gps)
        segs.append(white.crop(pk, pk + N_SAMP / FS + 0.01).value[:N_SAMP].astype(float))
    logB_real = log_bayes(segs, m_grid, chi_grid, t0_grid)
    # σ_a sensitivity on the real event (the Occam term depends on it)
    sens = {sa: log_bayes(segs, m_grid, chi_grid, t0_grid, sigma_a=sa) for sa in (4.0, 8.0, 16.0)}
    verdict = "favors 2-tone (overtone)" if logB_real > thr else "does NOT clear the 2-tone threshold"
    print(f"\n=== {EVENT} REAL: logB = {logB_real:+.2f}  (threshold {thr:+.2f}) -> {verdict} ===")
    print(f"σ_a sensitivity: " + " ".join(f"σ{sa}:{v:+.1f}" for sa, v in sens.items()))

    (RESULTS / "14_bayes_tonecount.json").write_text(json.dumps(
        {"event": EVENT, "sigma_a": SIGMA_A, "threshold": thr, "false_positive_1tone": fpr,
         "efficiency": {str(af): eff[af] for af in AMP_FRACS},
         "calibration": {str(af): {"logB_median": float(np.median(cal[af]["logB"])),
                                   "overtone_snr": cal[af]["overtone_snr"]} for af in AMP_FRACS},
         "logB_real": logB_real, "sigma_a_sensitivity": {str(k): v for k, v in sens.items()},
         "verdict": verdict}, indent=2))

    fig, ax = plt.subplots(figsize=(8, 5))
    for af in AMP_FRACS:
        ax.hist(cal[af]["logB"], bins=15, alpha=0.55,
                label=f"af={af} (SNR≈{cal[af]['overtone_snr']:.0f}, {'1' if af==0 else '2'}-tone)")
    ax.axvline(thr, color="k", ls="--", label=f"threshold {thr:+.1f}")
    ax.axvline(logB_real, color="crimson", lw=2.5, label=f"GW250114 real = {logB_real:+.1f}")
    ax.set_xlabel("log Bayes factor  B(2-tone / 1-tone)"); ax.set_ylabel("count")
    ax.set_title(f"{EVENT}: explicit Bayesian tone-count — calibrated on injections")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(PLOTS / "14_bayes_tonecount.png", dpi=140)
    print(f"wrote 14_bayes_tonecount.json + .png")


if __name__ == "__main__":
    main()
