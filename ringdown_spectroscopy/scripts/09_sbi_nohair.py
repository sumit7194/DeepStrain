#!/usr/bin/env python
"""Milestone 9 (v2): the network IS the no-hair test.

Extends 08's start-time-marginalized NPE from (M, chi) to (M, chi, delta),
where delta is 07's deviation parameter: the overtone frequency is scaled by
(1 + delta) while everything else stays Kerr-locked. The trained network
outputs the no-hair posterior directly — the poisoned start-time choice
marginalized by construction, the Kerr test amortized into the weights.

Pre-registered gates R1-R3 in notes/lab_notebook.md (2026-06-12).

Usage:
    python 09_sbi_nohair.py        # train + referee + apply (CPU)
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from gwpy.timeseries import TimeSeries

import rdlib

torch.manual_seed(0)
rng = np.random.default_rng(0)

PLOTS = Path(__file__).resolve().parent.parent / "plots"
RESULTS = Path(__file__).resolve().parent.parent / "results"
EVENT = "GW250114_082203"
FS = 4096.0
SEG = 0.04
N_SAMP = int(SEG * FS)
N_SAMP += N_SAMP % 2
T0_MAX_MS = 6.0
N_TRAIN = 150_000  # fix round: 90k gave M-coverage 0.83 (gate 0.85) — the one
PEAK_AMP_RANGE = (2.0, 12.0)  # pre-registered fix is more simulations

import sbilib
from sbilib import Embed, kerr220, kerr221  # Embed at module level is
# load-bearing: pickled posteriors resolve __main__.Embed at load time


def simulate(mass, chi, delta, n_det=2):
    return sbilib.simulate(mass, chi, delta, rng, n_det)


POST_CACHE = RESULTS / "09_posterior_150k.pt"

from sbi.inference import NPE
from sbi.neural_nets import posterior_nn
from sbi.utils import BoxUniform


prior = BoxUniform(torch.tensor([40.0, 0.05, -0.5]), torch.tensor([120.0, 0.95, 0.5]))

if POST_CACHE.exists():
    print(f"loading cached posterior from {POST_CACHE}")
    posterior = torch.load(POST_CACHE, weights_only=False)
else:
    print(f"simulating {N_TRAIN} ringdowns (theta = M, chi, delta) ...")
    thetas = np.column_stack([
        rng.uniform(40, 120, N_TRAIN),
        rng.uniform(0.05, 0.95, N_TRAIN),
        rng.uniform(-0.5, 0.5, N_TRAIN),
    ])
    xs = np.empty((N_TRAIN, 2 * N_SAMP), dtype=np.float32)
    for k, (m, c, d) in enumerate(thetas):
        xs[k] = simulate(m, c, d).reshape(-1)
        if k % 2000 == 0:
            rdlib.progress("09_sbi_nohair_sims", k, N_TRAIN)
    density = posterior_nn(model="maf", embedding_net=Embed(), hidden_features=80,
                           num_transforms=5)
    npe = NPE(prior=prior, density_estimator=density)
    npe.append_simulations(torch.tensor(thetas, dtype=torch.float32), torch.tensor(xs))
    print("training NPE ...")
    with rdlib.heartbeat("09_sbi_nohair_train"):
        estimator = npe.train(training_batch_size=256, max_num_epochs=80,
                              stop_after_epochs=10, show_train_summary=True)
    posterior = npe.build_posterior(estimator)
    torch.save(posterior, POST_CACHE)

# ------------------------------------------------------------- R1: calibration
print("\nR1: per-parameter 90% coverage on 200 held-out simulations")
hits = np.zeros(3)
for k in range(200):
    m, c, d = rng.uniform(40, 120), rng.uniform(0.05, 0.95), rng.uniform(-0.5, 0.5)
    x_obs = torch.tensor(simulate(m, c, d).reshape(1, -1))
    s = posterior.sample((300,), x=x_obs, show_progress_bars=False).numpy()
    for j, truth in enumerate((m, c, d)):
        lo, hi = np.percentile(s[:, j], [5, 95])
        hits[j] += lo <= truth <= hi
    if k % 20 == 0:
        rdlib.progress("09_R1_coverage", k, 200)
cov = hits / 200
print(f"  coverage: M {cov[0]:.2f}, chi {cov[1]:.2f}, delta {cov[2]:.2f} "
      f"(gate: each in [0.85, 0.95])")

# ----------------------------------------------- R2: real-O4-noise injections
print("R2: injections into REAL O4 noise (M=68, chi=0.69)")
F1_T, TAU1_T = kerr220.f_tau(68.0, 0.69)
F2_T, TAU2_T = kerr221.f_tau(68.0, 0.69)
base = rdlib.event_gps(EVENT)
inj_results = {0.0: [], 0.3: []}
k_noise = 0
for d_true in (0.0, 0.3):
    got = 0
    while got < 6 and k_noise < 24:
        center = base - 300 - 128 * k_noise
        k_noise += 1
        try:
            segs = []
            for det in ("H1", "L1"):
                raw = TimeSeries.fetch_open_data(det, center - 32, center + 32, cache=True)
                if not np.isfinite(raw.value).all():
                    raise ValueError("NaNs")
                params = [
                    dict(f=F1_T, tau=TAU1_T, amp=2.0e-21, phi=rng.uniform(-np.pi, np.pi)),
                    dict(f=F2_T * (1 + d_true), tau=TAU2_T, amp=2.0e-21,
                         phi=rng.uniform(-np.pi, np.pi)),
                ]
                inj = rdlib.inject_ringdown(raw, center, params).whiten(4, 2)
                seg = inj.crop(center - 0.002, center - 0.002 + SEG + 0.01).value[:N_SAMP]
                assert len(seg) == N_SAMP
                segs.append(seg)
            x_obs = torch.tensor(np.stack(segs).reshape(1, -1).astype(np.float32))
            s = posterior.sample((400,), x=x_obs, show_progress_bars=False).numpy()
            med = np.median(s, axis=0)
            lo_d, hi_d = np.percentile(s[:, 2], [5, 95])
            inj_results[d_true].append(dict(M=float(med[0]), chi=float(med[1]),
                                            delta=float(med[2]),
                                            d90=[float(lo_d), float(hi_d)]))
            got += 1
            print(f"  [d_true={d_true:+.1f}] M={med[0]:5.1f} chi={med[1]:.2f} "
                  f"delta={med[2]:+.2f} [{lo_d:+.2f}, {hi_d:+.2f}]")
        except Exception as e:
            print(f"  [noise {k_noise}] skipped: {e}")

for d_true, rows in inj_results.items():
    if rows:
        ds = np.array([r["delta"] for r in rows])
        print(f"  d_true={d_true:+.1f}: delta_hat = {ds.mean():+.3f} ± {ds.std():.3f} (n={len(ds)})")

# --------------------------------------------------------------- R3: GW250114
gps = rdlib.event_gps(EVENT)
segs = []
for det in ("H1", "L1"):
    white = rdlib.fetch_whitened(det, gps, bandpass=False)
    pk = rdlib.find_peak(white.bandpass(*rdlib.BAND), gps)
    seg = white.crop(pk, pk + SEG + 0.01).value[:N_SAMP]
    assert len(seg) == N_SAMP
    segs.append(seg)
x_obs = torch.tensor(np.stack(segs).reshape(1, -1).astype(np.float32))
s = posterior.sample((3000,), x=x_obs, show_progress_bars=False).numpy()
q = lambda j: np.percentile(s[:, j], [50, 5, 95])
m_q, c_q, d_q = q(0), q(1), q(2)
kerr_in_90 = bool(d_q[1] <= 0.0 <= d_q[2])
print(f"\nR3 {EVENT} no-hair posterior (start-time marginalized, amortized):")
print(f"  M     = {m_q[0]:.1f}  [{m_q[1]:.1f}, {m_q[2]:.1f}] 90%")
print(f"  chi   = {c_q[0]:.2f}  [{c_q[1]:.2f}, {c_q[2]:.2f}] 90%")
print(f"  delta = {d_q[0]:+.2f}  [{d_q[1]:+.2f}, {d_q[2]:+.2f}] 90%  "
      f"(07 classical: -0.16, 2sigma 0.72; Kerr inside 90%: {kerr_in_90})")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
axes[0].scatter(s[:, 0], s[:, 1], s=4, alpha=0.15, c=s[:, 2], cmap="coolwarm",
                vmin=-0.5, vmax=0.5)
axes[0].scatter([68], [0.69], s=160, marker="*", color="k", zorder=5,
                label="published (~68, 0.69)")
axes[0].set_xlabel("remnant mass [M_sun, detector frame]")
axes[0].set_ylabel("remnant spin chi")
axes[0].set_title(f"{EVENT}: (M, chi) posterior, colored by delta")
axes[0].legend()

axes[1].hist(s[:, 2], bins=50, color="lightsteelblue", edgecolor="white", density=True)
axes[1].axvline(0, color="k", lw=1.5, label="Kerr (delta = 0)")
axes[1].axvline(d_q[0], color="crimson", lw=2, label=f"median {d_q[0]:+.2f}")
axes[1].axvspan(d_q[1], d_q[2], color="crimson", alpha=0.12, label="90% CI")
axes[1].axvline(-0.16, color="gray", ls="--", lw=1.2, label="07 classical best (-0.16)")
axes[1].set_xlabel("no-hair deviation delta")
axes[1].set_title("the no-hair posterior — the network IS the test")
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(PLOTS / "09_nohair_GW250114.png", dpi=140)
print(f"wrote {PLOTS / '09_nohair_GW250114.png'}")

out = dict(event=EVENT, coverage=dict(M=cov[0], chi=cov[1], delta=cov[2]),
           injections=inj_results,
           posterior=dict(mass=list(m_q), chi=list(c_q), delta=list(d_q)),
           kerr_inside_90=kerr_in_90)
(RESULTS / "09_nohair_GW250114.json").write_text(json.dumps(out, indent=2))
print(f"wrote {RESULTS / '09_nohair_GW250114.json'}")
