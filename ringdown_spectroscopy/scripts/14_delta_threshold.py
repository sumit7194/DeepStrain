#!/usr/bin/env python
"""Milestone 14: the δ-MEASURABILITY THRESHOLD — at what ringdown loudness does the
no-hair test stop returning the prior?

13_more_events.py found only GW250114 (the loud one) measures δ; fainter events
return ≈ the prior. This maps WHY: inject Kerr (δ=0) ringdowns across a sweep of
loudness, run the v2/v3 NPE, and measure σ(δ) vs the injected ringdown SNR. The
loudness where σ(δ) drops below the prior (0.289) is the informative threshold —
a clean, rigorous completion of the stacking story (no real-data transfer needed).

Run:  python 14_delta_threshold.py [--n 60]
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import rdlib
import sbilib
from sbilib import (CHI_GRID, FS, N_SAMP, T0_MAX_MS, W220, W221, Embed)

RESULTS = Path(__file__).resolve().parent.parent / "results"
PLOTS = Path(__file__).resolve().parent.parent / "plots"
PRIOR_SIG = 1.0 / np.sqrt(12)            # δ ~ Uniform[-0.5,0.5] -> σ 0.289
A220_GRID = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0)
rng = None                               # set in main() from --seed


def inject(mass, chi, a220):
    """Kerr (δ=0) ringdown at a FIXED 220 loudness; returns (segment, whitened SNR)."""
    i = min(max(np.searchsorted(CHI_GRID, chi), 0), len(CHI_GRID) - 1)
    f1, tau1 = W220[i][0] / mass, W220[i][1] * mass
    f2, tau2 = W221[i][0] / mass, W221[i][1] * mass
    t = np.arange(N_SAMP) / FS
    t0 = rng.uniform(0, T0_MAX_MS / 1000.0)
    x = np.empty((2, N_SAMP), dtype=np.float32)
    snr2 = 0.0
    for d in range(2):
        amp1 = a220 * rng.uniform(0.7, 1.3)
        amp2 = amp1 * rng.uniform(0.5, 1.5)
        sig = (rdlib.damped_sinusoids(t, t0, [dict(f=f1, tau=tau1, amp=amp1, phi=rng.uniform(-np.pi, np.pi))])
               + rdlib.damped_sinusoids(t, t0, [dict(f=f2, tau=tau2, amp=amp2, phi=rng.uniform(-np.pi, np.pi))]))
        snr2 += float(np.sum(sig ** 2))          # whitened -> MF SNR^2 = sum(sig^2)
        x[d] = sig + rng.standard_normal(N_SAMP)
    return x, float(np.sqrt(snr2))


def main():
    global rng
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed); torch.manual_seed(args.seed)
    posterior = torch.load(RESULTS / "09_posterior_150k.pt", weights_only=False)
    T = json.loads((RESULTS / "10_recalibration.json").read_text())["T"]
    print(f"loaded NPE + T={T}; prior σ(δ)={PRIOR_SIG:.3f}\n")

    rows = []
    print(f"{'a220':>5} {'SNR':>6} {'sigma(delta)':>12} {'/prior':>7} verdict")
    for a in A220_GRID:
        sigs, snrs = [], []
        for _ in range(args.n):
            m, c = rng.uniform(55, 105), rng.uniform(0.2, 0.9)
            x, snr = inject(m, c, a)
            s = posterior.sample((300,), x=torch.tensor(x.reshape(1, -1)),
                                 show_progress_bars=False).numpy()[:, 2]
            s = np.median(s) + T * (s - np.median(s))      # v3 recalibration
            sigs.append(float(np.std(s))); snrs.append(snr)
            rdlib.progress("14_delta_threshold", len(rows) * args.n + len(sigs),
                           len(A220_GRID) * args.n)
        sig, snr = float(np.median(sigs)), float(np.median(snrs))
        ratio = sig / PRIOR_SIG
        rows.append(dict(a220=a, snr=snr, sigma=sig, ratio=ratio))
        print(f"{a:>5.0f} {snr:>6.1f} {sig:>12.3f} {ratio:>7.2f} "
              f"{'INFORMATIVE' if ratio < 0.85 else '~prior'}")

    # onset of informativeness: SNR where σ/prior first drops below 0.90 (a 10% tightening).
    # GW250114 (real, 13_more_events) reaches ratio ~0.83 -> it sits at/just past the trained edge.
    snrs = np.array([r["snr"] for r in rows]); ratios = np.array([r["ratio"] for r in rows])
    o = np.argsort(snrs); sr, rr = snrs[o], ratios[o]
    onset = float(np.interp(0.90, rr[::-1], sr[::-1])) if rr.min() < 0.90 else float("nan")
    imin = int(np.argmin(rr)); gw250114_ratio = 0.83
    print(f"\nonset of informativeness (σ/prior < 0.90): ringdown SNR ≈ {onset:.0f}")
    print(f"best within trained loudness: σ/prior = {rr[imin]:.2f} at ringdown SNR {sr[imin]:.0f} "
          f"(only ~{(1-rr[imin])*100:.0f}% tighter than prior)")
    print(f"GW250114 (real) reaches {gw250114_ratio:.2f} -> it sits at/just past this edge")
    print("⇒ explains 13_more_events: δ is informative ONLY at GW250114-class loudness, and even "
          "there only marginally; fainter events return ≈ the prior. The SNR information wall, mapped.")

    if args.seed != 0:
        print(f"[seed {args.seed}] robustness check — canonical artifacts not overwritten")
        return
    (RESULTS / "14_delta_threshold.json").write_text(json.dumps(
        {"prior_sigma": PRIOR_SIG, "onset_snr": onset, "best_ratio": float(rr[imin]),
         "best_ratio_snr": float(sr[imin]), "gw250114_ratio": gw250114_ratio, "curve": rows}, indent=2))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(snrs[o], [rows[i]["sigma"] for i in o], "o-", label="σ(δ) (recovered)")
    ax.axhline(PRIOR_SIG, color="gray", ls="--", label=f"prior σ {PRIOR_SIG:.3f}")
    ax.axhline(0.90 * PRIOR_SIG, color="crimson", ls=":", label="onset (0.90·prior)")
    ax.axhline(gw250114_ratio * PRIOR_SIG, color="darkgreen", ls=":", label="GW250114 (real) 0.83·prior")
    if np.isfinite(onset):
        ax.axvline(onset, color="crimson", alpha=0.4)
    ax.set_xlabel("injected ringdown SNR (whitened)"); ax.set_ylabel("σ(δ)")
    ax.set_title("No-hair δ measurability vs ringdown loudness")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(PLOTS / "14_delta_threshold.png", dpi=140)
    print(f"wrote 14_delta_threshold.json + .png")


if __name__ == "__main__":
    main()
