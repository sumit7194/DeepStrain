#!/usr/bin/env python
"""Milestone 10 (v3): post-hoc temperature recalibration of the no-hair NPE.

The 150k posterior is mildly overconfident (0.84-0.88 vs 0.90, architectural).
Fit one global temperature T (widen samples about the median) on 150 fresh
sims; validate coverage on 150 held-out; re-report GW250114 honestly.
Pre-registration T1-T3 in notes/lab_notebook.md (2026-06-13).
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
T0_MAX_MS, PEAK_AMP_RANGE = 6.0, (2.0, 12.0)

import sbilib
from sbilib import Embed  # module-level name: pickled posteriors resolve
# __main__.Embed at load time, so this alias is load-bearing


def simulate(mass, chi, delta, n_det=2):
    return sbilib.simulate(mass, chi, delta, rng, n_det)


def widen(s, T):
    med = np.median(s, axis=0)
    return med + T * (s - med)


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
    print("drawing 1000 calibration sims (fix round: 300 was noise-limited) ...")
    sims = []
    for k in range(1000):
        truth = (rng.uniform(40, 120), rng.uniform(0.05, 0.95), rng.uniform(-0.5, 0.5))
        x = torch.tensor(simulate(*truth).reshape(1, -1))
        s = posterior.sample((300,), x=x, show_progress_bars=False).numpy()
        sims.append((truth, s))
        if k % 20 == 0:
            rdlib.progress("10_recal_sims", k, 1000)

    fit, held = sims[:600], sims[600:]
    Ts = np.arange(1.0, 1.65, 0.05)
    best_T, best_gap = 1.0, 9
    for T in Ts:
        gap = abs(float(coverage(fit, T).mean()) - 0.90)
        if gap < best_gap:
            best_T, best_gap = float(T), gap
    cov_h = coverage(held, best_T)
    cov_raw = coverage(held, 1.0)
    print(f"T1 fitted temperature: {best_T:.2f}")
    print(f"T2 held-out coverage @T: M {cov_h[0]:.2f}, chi {cov_h[1]:.2f}, "
          f"delta {cov_h[2]:.2f} (raw was {cov_raw[0]:.2f}/{cov_raw[1]:.2f}/{cov_raw[2]:.2f}; "
          f"gate each in [0.85,0.95], mean in [0.88,0.92] -> mean {cov_h.mean():.3f})")

    # T3: GW250114 with honest calibration
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
    print(f"T3 GW250114 recalibrated: delta = {d_med:+.2f} [{d_lo:+.2f}, {d_hi:+.2f}] 90% "
          f"(Kerr inside: {kerr})")

    (RESULTS / "10_recalibration.json").write_text(json.dumps(
        {"T": best_T, "coverage_heldout": cov_h.tolist(), "coverage_raw": cov_raw.tolist(),
         "gw250114_delta": [float(d_med), float(d_lo), float(d_hi)],
         "kerr_inside_90": kerr}, indent=2))
    print(f"wrote {RESULTS / '10_recalibration.json'}")


if __name__ == "__main__":
    main()
