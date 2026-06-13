#!/usr/bin/env python
"""Milestone 8: simulation-based inference with the start time MARGINALIZED.

The novelty angle (v1 prototype): the classical analyses must CHOOSE when
ringing starts (the poisoned choice, 05). Here the simulator randomizes the
start time inside the analysis segment, so the neural posterior estimator
learns p(M, chi | data) with the start time integrated out — no choice made,
ever.

Design:
  simulator  theta = (M, chi); nuisances drawn fresh each sim and NEVER given
             to the network: per-detector amplitudes, phases, the 221 overtone
             (locked to Kerr position), and the start-time offset t0 ~ U(0,6) ms.
             Signal = whitened-domain two-tone Kerr ringdown + unit white noise.
  network    1D CNN embedding over (2 det x 164 samples) -> masked autoregressive
             flow (sbi's NPE).
  referee    (i) coverage on held-out sims; (ii) injections into REAL O4 noise
             (the Gaussian-noise assumption gets tested here); (iii) GW250114.

Usage:
    python 08_sbi.py            # train + validate + apply (CPU, ~10 min)
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
SEG = 0.04            # 40 ms analysis segment
N_SAMP = int(SEG * FS)  # 163 -> use 164
N_SAMP += N_SAMP % 2
T0_MAX_MS = 6.0
N_TRAIN = 60_000
PEAK_AMP_RANGE = (2.0, 12.0)   # whitened peak amplitude of the 220 (GW250114 ~ 7)

kerr220, kerr221 = rdlib.KerrMap(2, 2, 0), rdlib.KerrMap(2, 2, 1)

# precompute Kerr maps on a grid for fast simulation
CHI_GRID = np.linspace(0.01, 0.97, 200)
W220 = np.array([kerr220.f_tau(1.0, c) for c in CHI_GRID])  # (f*Msun, tau/Msun) at M=1
W221 = np.array([kerr221.f_tau(1.0, c) for c in CHI_GRID])


def simulate(mass, chi, n_det=2):
    """Whitened-domain segment: two Kerr tones + unit white noise."""
    i = np.searchsorted(CHI_GRID, chi)
    i = min(max(i, 0), len(CHI_GRID) - 1)
    f1, tau1 = W220[i][0] / mass, W220[i][1] * mass
    f2, tau2 = W221[i][0] / mass, W221[i][1] * mass
    t = np.arange(N_SAMP) / FS
    t0 = rng.uniform(0, T0_MAX_MS / 1000.0)
    x = np.empty((n_det, N_SAMP), dtype=np.float32)
    a220 = rng.uniform(*PEAK_AMP_RANGE)
    for d in range(n_det):
        amp1 = a220 * rng.uniform(0.7, 1.3)          # det-to-det antenna variation
        amp2 = amp1 * rng.uniform(0.5, 1.5)          # overtone amplitude ratio
        params = [
            dict(f=f1, tau=tau1, amp=amp1, phi=rng.uniform(-np.pi, np.pi)),
            dict(f=f2, tau=tau2, amp=amp2, phi=rng.uniform(-np.pi, np.pi)),
        ]
        x[d] = rdlib.damped_sinusoids(t, t0, params) + rng.standard_normal(N_SAMP)
    return x


POST_CACHE = RESULTS / "08_posterior.pt"

# --------------------------------------------------------------- training set
print(f"simulating {N_TRAIN} ringdowns ...")
if not POST_CACHE.exists():
    thetas = np.column_stack([rng.uniform(40, 120, N_TRAIN), rng.uniform(0.05, 0.95, N_TRAIN)])
    xs = np.stack([simulate(m, c) for m, c in thetas]).astype(np.float32)
    theta_t = torch.tensor(thetas, dtype=torch.float32)
    x_t = torch.tensor(xs.reshape(N_TRAIN, -1))

# --------------------------------------------------------------- embedding + NPE
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn
from sbi.utils import BoxUniform


class Embed(torch.nn.Module):
    def __init__(self, n_out=48):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Unflatten(1, (2, N_SAMP)),
            torch.nn.Conv1d(2, 16, 9, padding=4), torch.nn.ReLU(),
            torch.nn.Conv1d(16, 32, 9, padding=4, stride=2), torch.nn.ReLU(),
            torch.nn.Conv1d(32, 32, 9, padding=4, stride=2), torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.LazyLinear(128), torch.nn.ReLU(),
            torch.nn.Linear(128, n_out),
        )

    def forward(self, x):
        return self.net(x)


prior = BoxUniform(torch.tensor([40.0, 0.05]), torch.tensor([120.0, 0.95]))
if POST_CACHE.exists():
    print(f"loading cached posterior from {POST_CACHE}")
    posterior = torch.load(POST_CACHE, weights_only=False)
else:
    density = posterior_nn(model="maf", embedding_net=Embed(), hidden_features=64,
                           num_transforms=4)
    npe = NPE(prior=prior, density_estimator=density)
    npe.append_simulations(theta_t, x_t)
    print("training NPE ...")
    estimator = npe.train(training_batch_size=256, max_num_epochs=60,
                          stop_after_epochs=8, show_train_summary=True)
    posterior = npe.build_posterior(estimator)
    torch.save(posterior, POST_CACHE)

# --------------------------------------------------------------- referee i: sims
print("\nreferee (i): coverage on 200 held-out simulations")
hits90 = 0
for _ in range(200):
    m, c = rng.uniform(40, 120), rng.uniform(0.05, 0.95)
    x_obs = torch.tensor(simulate(m, c).reshape(1, -1))
    s = posterior.sample((300,), x=x_obs, show_progress_bars=False).numpy()
    lo_m, hi_m = np.percentile(s[:, 0], [5, 95])
    lo_c, hi_c = np.percentile(s[:, 1], [5, 95])
    if lo_m <= m <= hi_m and lo_c <= c <= hi_c:
        hits90 += 1
print(f"  joint 90%x90% interval coverage: {hits90/200:.2f} (expect ~0.81 if calibrated)")

# --------------------------------------------------------------- referee ii: real noise
print("referee (ii): injections into REAL O4 noise (truth M=68, chi=0.69)")
F1_T, TAU1_T = kerr220.f_tau(68.0, 0.69)
F2_T, TAU2_T = kerr221.f_tau(68.0, 0.69)
base = rdlib.event_gps(EVENT)
recov = []
for k in range(8):
    center = base - 300 - 128 * k
    try:
        segs = []
        for det in ("H1", "L1"):
            raw = TimeSeries.fetch_open_data(det, center - 32, center + 32, cache=True)
            if not np.isfinite(raw.value).all():
                raise ValueError("NaNs")
            t_inj = center
            params = [
                dict(f=F1_T, tau=TAU1_T, amp=2.0e-21, phi=rng.uniform(-np.pi, np.pi)),
                dict(f=F2_T, tau=TAU2_T, amp=2.0e-21, phi=rng.uniform(-np.pi, np.pi)),
            ]
            inj = rdlib.inject_ringdown(raw, t_inj, params).whiten(4, 2)
            # segment starts 2 ms BEFORE injection: t0 offset is unknown to the net
            seg = inj.crop(t_inj - 0.002, t_inj - 0.002 + SEG + 0.01).value[:N_SAMP]
            assert len(seg) == N_SAMP, f"got {len(seg)}"
            segs.append(seg)
        x_obs = torch.tensor(np.stack(segs).reshape(1, -1).astype(np.float32))
        s = posterior.sample((400,), x=x_obs, show_progress_bars=False).numpy()
        recov.append((np.median(s[:, 0]), np.median(s[:, 1])))
        print(f"  [inj {k}] M={np.median(s[:,0]):5.1f} (68), chi={np.median(s[:,1]):.2f} (0.69)")
    except Exception as e:
        print(f"  [inj {k}] skipped: {e}")

recov = np.array(recov)
if len(recov):
    print(f"  medians over {len(recov)} injections: M={recov[:,0].mean():.1f}+-{recov[:,0].std():.1f}, "
          f"chi={recov[:,1].mean():.2f}+-{recov[:,1].std():.2f}")
else:
    print("  WARNING: no injections recovered")

# --------------------------------------------------------------- apply: GW250114
gps = rdlib.event_gps(EVENT)
segs = []
for det in ("H1", "L1"):
    white = rdlib.fetch_whitened(det, gps, bandpass=False)
    pk = rdlib.find_peak(white.bandpass(*rdlib.BAND), gps)
    # start AT the peak: the simulator generates pure post-start ringdown tones,
    # so the analysis segment must not contain pre-peak (merger) signal
    seg = white.crop(pk, pk + SEG + 0.01).value[:N_SAMP]
    assert len(seg) == N_SAMP, f"got {len(seg)}"
    segs.append(seg)
x_obs = torch.tensor(np.stack(segs).reshape(1, -1).astype(np.float32))
s = posterior.sample((2000,), x=x_obs, show_progress_bars=False).numpy()
m_med, m_lo, m_hi = np.percentile(s[:, 0], [50, 5, 95])
c_med, c_lo, c_hi = np.percentile(s[:, 1], [50, 5, 95])
print(f"\n{EVENT} NPE posterior (start-time marginalized):")
print(f"  M   = {m_med:.1f}  [{m_lo:.1f}, {m_hi:.1f}] 90% (detector frame)")
print(f"  chi = {c_med:.2f}  [{c_lo:.2f}, {c_hi:.2f}] 90%")

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(s[:, 0], s[:, 1], s=4, alpha=0.18, color="purple", label="NPE posterior samples")
if len(recov):
    ax.scatter(recov[:, 0], recov[:, 1], s=60, marker="x", color="k",
               label="real-noise injection medians (truth 68, 0.69)")
ax.scatter([68], [0.69], s=160, marker="*", color="red", zorder=5, label="injection truth")
ax.set_xlabel("remnant mass [M_sun, detector frame]")
ax.set_ylabel("remnant spin chi")
ax.set_title(f"{EVENT}: neural posterior, start time marginalized\n"
             "(network never told when ringing starts)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(PLOTS / "08_sbi_GW250114.png", dpi=140)
print(f"wrote {PLOTS / '08_sbi_GW250114.png'}")

out = dict(event=EVENT, coverage90x90=hits90 / 200,
           injection_medians=recov.tolist(),
           posterior=dict(mass=[m_med, m_lo, m_hi], chi=[c_med, c_lo, c_hi]))
(RESULTS / "08_sbi_GW250114.json").write_text(json.dumps(out, indent=2))
print(f"wrote {RESULTS / '08_sbi_GW250114.json'}")
