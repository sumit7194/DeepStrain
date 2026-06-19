#!/usr/bin/env python
"""Milestone 12 (roadmap P1): multi-event no-hair δ STACKING.

Single-event σ(δ) ≈ 0.24 (GW250114). If the Kerr deviation δ is universal, many
events combine: under a common-δ assumption with (recalibrated) Gaussian per-event
posteriors, the stacked precision adds — σ_stack = (Σ 1/σ_i²)^(-1/2) — so N events
tighten σ(δ) by ~√N. This is the one place more data directly sharpens a real GR test.

Reuses the v2/v3 amortized, start-time-marginalized no-hair NPE (09_posterior_150k.pt)
+ the v3 temperature recalibration (T from 10_recalibration.json).

Pre-registered gates:
  S1 (unbiased): for δ=0 (Kerr) injections, the stacked δ median stays centred on 0.
  S2 (tightening): σ_stack(N) ≈ σ_single/√N  (ratio √N within ~15%).
  S3 (coverage): the stacked 90% CI covers the true δ ~90% of the time.
  S4 (application): GW250114+GW150914 stacked δ — report the combined CI vs single-event.

Usage:  python 12_stacking.py [--smoke]
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
from sbilib import Embed, N_SAMP, SEG  # Embed at module level: pickled posterior resolves it

torch.manual_seed(0)
rng = np.random.default_rng(0)
RESULTS = Path(__file__).resolve().parent.parent / "results"
PLOTS = Path(__file__).resolve().parent.parent / "plots"
N_GRID = (1, 2, 3, 5, 8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    R = 8 if args.smoke else 40           # realizations per N
    n_post = 300                          # posterior samples per event

    posterior = torch.load(RESULTS / "09_posterior_150k.pt", weights_only=False)
    T = json.loads((RESULTS / "10_recalibration.json").read_text())["T"]
    print(f"loaded NPE + recalibration T={T}")

    def delta_post(x_obs):
        """(mean, std) of the RECALIBRATED δ posterior for one observation."""
        s = posterior.sample((n_post,), x=x_obs, show_progress_bars=False).numpy()[:, 2]
        s = np.median(s) + T * (s - np.median(s))     # v3 temperature widening
        return float(np.mean(s)), float(np.std(s))

    def stack(mus, sigs):
        """Common-δ precision-weighted combination of Gaussian posteriors."""
        w = 1.0 / np.asarray(sigs) ** 2
        sig = float(1.0 / np.sqrt(w.sum()))
        return float((np.asarray(mus) * w).sum() / w.sum()), sig

    def inject(delta_true):
        m, c = rng.uniform(55, 105), rng.uniform(0.2, 0.9)   # informative-loudness range
        x = torch.tensor(sbilib.simulate(m, c, delta_true, rng).reshape(1, -1))
        return delta_post(x)

    # ---- S1/S2/S3: injection validation (δ=0 truth) ------------------------
    sig_single = np.mean([inject(0.0)[1] for _ in range(20)])  # reference single-event σ
    print(f"\nsingle-event σ(δ) ≈ {sig_single:.3f}")
    print(f"{'N':>3} {'σ_stack':>8} {'σ_single/√N':>12} {'median δ':>9} {'90% cover':>10}")
    rows = []
    for ni, n in enumerate(N_GRID):
        sig_s, mu_s, cov = [], [], []
        for r in range(R):
            mus, sigs = zip(*[inject(0.0) for _ in range(n)])
            mu, sig = stack(mus, sigs)
            mu_s.append(mu); sig_s.append(sig)
            cov.append(abs(mu - 0.0) < 1.645 * sig)        # 90% CI covers true δ=0
            rdlib.progress("12_stacking_inj", ni * R + r + 1, len(N_GRID) * R)
        expect = sig_single / np.sqrt(n)
        rows.append(dict(N=n, sigma_stack=float(np.mean(sig_s)), expect=float(expect),
                         median=float(np.mean(mu_s)), coverage=float(np.mean(cov))))
        print(f"{n:>3} {np.mean(sig_s):>8.3f} {expect:>12.3f} {np.mean(mu_s):>+9.3f} {np.mean(cov):>10.2f}")

    # gate checks
    big = rows[-1]
    s2 = abs(big["sigma_stack"] - big["expect"]) / big["expect"] < 0.15
    s1 = abs(big["median"]) < 0.10
    s3 = all(0.80 <= r["coverage"] <= 1.0 for r in rows)
    print(f"\nS1 unbiased (|median|<0.10 at N=8): {s1}  ({big['median']:+.3f})")
    print(f"S2 tightening (σ_stack≈σ/√N at N=8): {s2}  ({big['sigma_stack']:.3f} vs {big['expect']:.3f})")
    print(f"S3 coverage (all N in [0.80,1.0]): {s3}")

    # ---- S4: real-event application ---------------------------------------
    print("\nS4: real events")
    EVENTS = {"GW250114_082203": "GW250114", "GW150914": "GW150914"}
    evpost = {}
    for ev, lab in EVENTS.items():
        try:
            gps = rdlib.event_gps(ev)
            segs = []
            for det in ("H1", "L1"):
                white = rdlib.fetch_whitened(det, gps, bandpass=False)
                pk = rdlib.find_peak(white.bandpass(*rdlib.BAND), gps)
                seg = white.crop(pk, pk + SEG + 0.01).value[:N_SAMP]
                assert len(seg) == N_SAMP
                segs.append(seg)
            x = torch.tensor(np.stack(segs).reshape(1, -1).astype(np.float32))
            mu, sig = delta_post(x)
            evpost[lab] = (mu, sig)
            print(f"  {lab}: δ = {mu:+.3f} ± {sig:.3f}")
        except Exception as e:
            print(f"  {lab}: skipped ({e})")
    stacked = None
    if len(evpost) >= 2:
        mu_s, sig_s = stack([v[0] for v in evpost.values()], [v[1] for v in evpost.values()])
        best = min(evpost.values(), key=lambda v: v[1])
        stacked = dict(mu=mu_s, sigma=sig_s, ci90=[mu_s - 1.645 * sig_s, mu_s + 1.645 * sig_s])
        print(f"  STACKED ({'+'.join(evpost)}): δ = {mu_s:+.3f} ± {sig_s:.3f} "
              f"[{stacked['ci90'][0]:+.3f}, {stacked['ci90'][1]:+.3f}] 90% "
              f"(tightest single σ {best[1]:.3f} → stacked {sig_s:.3f})")

    out = dict(sigma_single=float(sig_single), injection=rows,
               gates=dict(S1_unbiased=bool(s1), S2_tightening=bool(s2), S3_coverage=bool(s3)),
               events={k: dict(mu=v[0], sigma=v[1]) for k, v in evpost.items()}, stacked=stacked, T=T)
    (RESULTS / "12_stacking.json").write_text(json.dumps(out, indent=2))

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    Ns = np.array([r["N"] for r in rows])
    ax[0].plot(Ns, [r["sigma_stack"] for r in rows], "o-", label="σ_stack (measured)")
    ax[0].plot(Ns, sig_single / np.sqrt(Ns), "k--", label="σ_single/√N (ideal)")
    ax[0].set_xlabel("events stacked N"); ax[0].set_ylabel("σ(δ)")
    ax[0].set_title("S2: stacking tightens σ(δ) as √N"); ax[0].legend(); ax[0].grid(alpha=0.3)
    if stacked:
        labs = list(evpost) + ["STACKED"]
        mus = [evpost[k][0] for k in evpost] + [stacked["mu"]]
        sgs = [evpost[k][1] for k in evpost] + [stacked["sigma"]]
        ax[1].errorbar(range(len(labs)), mus, yerr=[1.645 * s for s in sgs], fmt="o", capsize=5)
        ax[1].axhline(0, color="gray", ls=":", label="Kerr (δ=0)")
        ax[1].set_xticks(range(len(labs)), labs); ax[1].set_ylabel("no-hair δ (90% CI)")
        ax[1].set_title("S4: real-event stacked no-hair δ"); ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(PLOTS / "12_stacking.png", dpi=140)
    print(f"\nwrote {RESULTS/'12_stacking.json'} + {PLOTS/'12_stacking.png'}")


if __name__ == "__main__":
    main()
