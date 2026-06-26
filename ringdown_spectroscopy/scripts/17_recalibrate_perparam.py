#!/usr/bin/env python
"""Milestone 17 (R1, PLAN.md): PER-PARAMETER temperature recalibration of the no-hair NPE.

v3 (10_recalibrate) fit ONE global temperature T=1.05 (held-out coverage 0.91/0.92/0.90). R1 fits a
SEPARATE temperature T_M, T_chi, T_delta for each parameter. Because widen() rescales each column about
its own median independently, parameter j's coverage depends only on T_j — so one T-sweep yields all three
per-parameter optima. Validate held-out per-param coverage (gate: each in [0.85,0.95]); re-report GW250114's
delta under its OWN temperature T_delta. Low-value by design (v3's global T already lands each param at
0.90-0.92) — this just confirms per-param tuning doesn't beat the global fit, and reports honestly if it does.
"""
import json
from pathlib import Path

import numpy as np
import torch

import rdlib

torch.manual_seed(3)
rng = np.random.default_rng(3)

RESULTS = Path(__file__).resolve().parent.parent / "results"
FS, SEG = 4096.0, 0.04
N_SAMP = int(SEG * FS) + int(SEG * FS) % 2
PARAMS = ("M", "chi", "delta")

import sbilib
from sbilib import Embed  # load-bearing: pickled posteriors resolve __main__.Embed at load time


def simulate(mass, chi, delta, n_det=2):
    return sbilib.simulate(mass, chi, delta, rng, n_det)


def widen(s, T):
    """T may be scalar or per-parameter vector (3,) — broadcasts per column."""
    med = np.median(s, axis=0)
    return med + np.asarray(T) * (s - med)


def coverage(sims, T):
    hits = np.zeros(3)
    for truth, s in sims:
        sw = widen(s, T)
        for j in range(3):
            lo, hi = np.percentile(sw[:, j], [5, 95])
            hits[j] += lo <= truth[j] <= hi
    return hits / len(sims)


def main() -> None:
    posterior = torch.load(RESULTS / "09_posterior_150k.pt", weights_only=False)
    print("drawing 1000 calibration sims ...")
    sims = []
    for k in range(1000):
        truth = (rng.uniform(40, 120), rng.uniform(0.05, 0.95), rng.uniform(-0.5, 0.5))
        x = torch.tensor(simulate(*truth).reshape(1, -1))
        s = posterior.sample((300,), x=x, show_progress_bars=False).numpy()
        sims.append((truth, s))
        if k % 20 == 0:
            rdlib.progress("17_recal_perparam_sims", k, 1000)

    fit, held = sims[:600], sims[600:]
    # one T-sweep: column j's coverage at widening T depends only on T_j, so covs[:,j] vs T = param j's curve
    Ts = np.round(np.arange(1.0, 2.0, 0.05), 2)
    covs_fit = np.array([coverage(fit, T) for T in Ts])             # (n_T, 3)
    best_T = np.array([float(Ts[np.argmin(np.abs(covs_fit[:, j] - 0.90))]) for j in range(3)])

    cov_h = coverage(held, best_T)                                  # per-param held-out coverage at per-param T
    cov_raw = coverage(held, 1.0)
    # global-T reference (v3): the single T minimizing mean |coverage-0.90| on fit
    glob_T = float(Ts[np.argmin([abs(coverage(fit, T).mean() - 0.90) for T in Ts])])
    cov_glob = coverage(held, glob_T)
    print(f"R1 per-param temperatures: T_M={best_T[0]:.2f} T_chi={best_T[1]:.2f} T_delta={best_T[2]:.2f}")
    print(f"   held-out coverage @per-param T: M {cov_h[0]:.2f}, chi {cov_h[1]:.2f}, delta {cov_h[2]:.2f}")
    print(f"   vs global T={glob_T:.2f}:        M {cov_glob[0]:.2f}, chi {cov_glob[1]:.2f}, delta {cov_glob[2]:.2f}")
    print(f"   vs raw   T=1.00:                 M {cov_raw[0]:.2f}, chi {cov_raw[1]:.2f}, delta {cov_raw[2]:.2f}")
    in_band = bool(np.all((cov_h >= 0.85) & (cov_h <= 0.95)))
    # improvement over global: mean abs deviation from 0.90, per-param vs global
    mad_pp = float(np.abs(cov_h - 0.90).mean()); mad_gl = float(np.abs(cov_glob - 0.90).mean())
    print(f"   per-param each in [0.85,0.95]: {in_band} | mean|cov-0.90| per-param {mad_pp:.3f} vs global {mad_gl:.3f}")

    # GW250114 delta under its OWN temperature T_delta
    gps = rdlib.event_gps("GW250114_082203")
    segs = []
    for det in ("H1", "L1"):
        white = rdlib.fetch_whitened(det, gps, bandpass=False)
        pk = rdlib.find_peak(white.bandpass(*rdlib.BAND), gps)
        segs.append(white.crop(pk, pk + SEG + 0.01).value[:N_SAMP])
    x_obs = torch.tensor(np.stack(segs).reshape(1, -1).astype(np.float32))
    s = posterior.sample((3000,), x=x_obs, show_progress_bars=False).numpy()
    sw = widen(s, best_T)
    d_med, d_lo, d_hi = np.percentile(sw[:, 2], [50, 5, 95])
    kerr = bool(d_lo <= 0.0 <= d_hi)
    print(f"GW250114 delta @T_delta={best_T[2]:.2f}: {d_med:+.2f} [{d_lo:+.2f}, {d_hi:+.2f}] 90% (Kerr inside: {kerr})")

    (RESULTS / "17_recalibrate_perparam.json").write_text(json.dumps(
        {"T_perparam": dict(zip(PARAMS, best_T.tolist())), "T_global": glob_T,
         "coverage_heldout_perparam": cov_h.tolist(), "coverage_heldout_global": cov_glob.tolist(),
         "coverage_raw": cov_raw.tolist(), "each_in_band": in_band,
         "mad_perparam": mad_pp, "mad_global": mad_gl,
         "gw250114_delta": [float(d_med), float(d_lo), float(d_hi)], "kerr_inside_90": kerr}, indent=2))
    print(f"wrote {RESULTS / '17_recalibrate_perparam.json'}")


if __name__ == "__main__":
    main()
