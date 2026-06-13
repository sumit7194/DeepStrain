"""Step 07 — echoes v2: the noise-trained ML scorer, judged by the v1 harness.

A small 1-D conv autoencoder learns the detector's whitened noise from
off-source segments (training pool only — never the segments that define
significance, never on-source data). The v2 statistic is the SAME
network-coherent comb as v1, applied to the autoencoder's
reconstruction-error envelope instead of the raw strain envelope.
The verdict comes exclusively from the v1 machinery: background distribution
on the held-out pool, injection-recovery head-to-head at p < 0.01, on-source
p-values. Pre-registration (V1-V3) in notes/lab_notebook.md, 2026-06-12.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from echolib import (
    DETECTORS,
    GW150914_DT_PRED,
    RESULTS,
    comb_on_env,
    echo_train,
    load_segments,
    progress,
)
from torch import nn

EVENT = "GW150914"
WIN = 2048  # 0.5 s windows
STRIDE_TRAIN = 1024
STRIDE_SCORE = 512  # finer stride when scoring, for time resolution of e(t)
N_TRAIN_PAIRS = 100  # contamination split: first 100 pairs train the scorer,
N_STEPS = 4000  # the remaining pairs define background + injections
AMPS = np.array([0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0])
N_TRIALS = 50
CONTROL_AMPS = (0.5, 1.0)  # irregular-spacing control: comb must NOT fire


class ConvAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(1, 16, 9, stride=4, padding=4), nn.Tanh(),
            nn.Conv1d(16, 32, 9, stride=4, padding=4), nn.Tanh(),
            nn.Conv1d(32, 64, 9, stride=4, padding=4), nn.Tanh(),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(64, 32, 8, stride=4, padding=2), nn.Tanh(),
            nn.ConvTranspose1d(32, 16, 8, stride=4, padding=2), nn.Tanh(),
            nn.ConvTranspose1d(16, 1, 8, stride=4, padding=2),
        )

    def forward(self, x):
        return self.dec(self.enc(x))


def windows(seg: np.ndarray, stride: int) -> np.ndarray:
    n = (len(seg) - WIN) // stride + 1
    return np.stack([seg[i * stride : i * stride + WIN] for i in range(n)])


def train_scorer(train_segs: list[np.ndarray], seed: int = 0) -> ConvAE:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X = np.concatenate([windows(s, STRIDE_TRAIN) for s in train_segs]).astype(np.float32)
    model = ConvAE()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(N_STEPS):
        idx = rng.integers(0, len(X), 64)
        batch = torch.from_numpy(X[idx])[:, None, :]
        rec = model(batch)
        loss = nn.functional.mse_loss(rec, batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0:
            progress("echoes_v2_scorer", step, N_STEPS, loss=loss.item())
            if step % 1000 == 0:
                print(f"  step {step:5d}  mse {loss.item():.4f}")
    model.eval()
    return model


def error_envelope(model: ConvAE, seg: np.ndarray, fs: float) -> np.ndarray:
    """Per-sample reconstruction-error envelope via overlapped windows."""
    W = windows(seg, STRIDE_SCORE).astype(np.float32)
    with torch.no_grad():
        rec = model(torch.from_numpy(W)[:, None, :])[:, 0, :].numpy()
    err = (W - rec) ** 2
    env = np.zeros(len(seg))
    cnt = np.zeros(len(seg))
    for i in range(len(W)):
        s = i * STRIDE_SCORE
        env[s : s + WIN] += err[i]
        cnt[s : s + WIN] += 1
    env = env / np.maximum(cnt, 1)
    # light smoothing, mean removal — mirrors echolib.envelope conventions
    w = max(3, int(round(10e-3 * fs)))
    env = np.convolve(env, np.ones(w) / w, mode="same")
    return env - env.mean()


def v2_network_scores(models, pair, fs, dt_grid, j_pred):
    total = np.zeros(len(dt_grid))
    for det, x in pair.items():
        total += comb_on_env(error_envelope(models[det], x, fs), fs, dt_grid)
    return float(total.max()), float(total[j_pred])


def main() -> None:
    dt_grid = np.arange(0.05, 0.5, 0.005)
    j_pred = int(np.argmin(np.abs(dt_grid - GW150914_DT_PRED)))
    rng = np.random.default_rng(2026)

    all_segs = {det: load_segments(EVENT, det) for det in DETECTORS}
    fs = all_segs["H1"].fs
    n_off = min(len(all_segs[d].off) for d in DETECTORS)
    train_idx = range(N_TRAIN_PAIRS)
    eval_idx = range(N_TRAIN_PAIRS, n_off)
    print(f"split: {N_TRAIN_PAIRS} train pairs / {n_off - N_TRAIN_PAIRS} eval pairs")

    models = {}
    for det in DETECTORS:
        print(f"training scorer for {det}:")
        models[det] = train_scorer([all_segs[det].off[i] for i in train_idx])

    # V1: background distribution of the v2 statistic on the held-out pool
    bg_max, bg_pred = [], []
    for i in eval_idx:
        pair = {det: all_segs[det].off[i] for det in DETECTORS}
        m, p = v2_network_scores(models, pair, fs, dt_grid, j_pred)
        bg_max.append(m)
        bg_pred.append(p)
    bg_max, bg_pred = np.array(bg_max), np.array(bg_pred)
    thresh = float(np.quantile(bg_pred, 0.99))
    print(f"V1 background: N={len(bg_pred)}, stat-B 99th pct = {thresh:.3f}")

    # V2: head-to-head injection recovery (injections into EVAL segments only)
    n_samp = len(all_segs["H1"].off[0])
    eval_list = list(eval_idx)
    eff = []
    for amp in AMPS:
        hits = 0
        for _ in range(N_TRIALS):
            i = eval_list[int(rng.integers(0, len(eval_list)))]
            inj = echo_train(n_samp, fs, GW150914_DT_PRED, amp=amp, rng=rng)
            total = np.zeros(len(dt_grid))
            for det in DETECTORS:
                total += comb_on_env(
                    error_envelope(models[det], all_segs[det].off[i] + inj, fs),
                    fs, dt_grid,
                )
            hits += float(total[j_pred]) > thresh
        eff.append(hits / N_TRIALS)
        print(f"V2 amp {amp:4.1f}: {hits}/{N_TRIALS} recovered ({100 * hits / N_TRIALS:.0f}%)")

    # C2 specificity control: same pulse count and energies, IRREGULAR spacing.
    # Statistic B is the comb at the predicted spacing, so it must stay quiet:
    # if it fires on aperiodic energy, the detector is not an echo detector.
    print("C2 irregular-spacing control:")
    ctrl = {}
    for amp in CONTROL_AMPS:
        hits = 0
        for _ in range(N_TRIALS):
            i = eval_list[int(rng.integers(0, len(eval_list)))]
            inj = np.zeros(n_samp)
            t = np.arange(n_samp) / fs
            phase = rng.uniform(0, 2 * np.pi)
            for k in range(6):
                t_k = rng.uniform(0.1, 2.8)
                dt_k = t - t_k
                inj += ((-1) ** k * amp * (0.7**k)
                        * np.where(dt_k >= 0,
                                   np.exp(-dt_k / 0.02) * np.sin(2 * np.pi * 250 * dt_k + phase),
                                   0.0))
            total = np.zeros(len(dt_grid))
            for det in DETECTORS:
                total += comb_on_env(
                    error_envelope(models[det], all_segs[det].off[i] + inj, fs),
                    fs, dt_grid,
                )
            hits += float(total[j_pred]) > thresh
        ctrl[amp] = hits / N_TRIALS
        print(f"   amp {amp:4.1f} irregular: {hits}/{N_TRIALS} fired "
              f"({100 * hits / N_TRIALS:.0f}%)  (gate: ~1%)")

    # V3: on-source
    on = {det: all_segs[det].on for det in DETECTORS}
    on_max, on_pred = v2_network_scores(models, on, fs, dt_grid, j_pred)
    p_pred = (np.sum(bg_pred >= on_pred) + 1) / (len(bg_pred) + 1)
    p_max = (np.sum(bg_max >= on_max) + 1) / (len(bg_max) + 1)
    print(f"V3 on-source {EVENT}: A p = {p_max:.3f}, B p = {p_pred:.3f}")

    v1 = np.load(RESULTS / "06_sensitivity.npy", allow_pickle=True).item()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(AMPS, eff, "o-", color="crimson", label="v2: ML scorer (comb on AE residuals)")
    ax.plot(v1["amps"], v1["eff"], "s--", color="darkslategray",
            label="v1: comb on raw envelope")
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_xlabel("injected first-pulse amplitude [whitened-noise σ]")
    ax.set_ylabel("recovery fraction (p < 0.01 vs own background)")
    ax.set_title(f"{EVENT}: the head-to-head — noise-trained ML scorer vs the comb baseline")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    fig.tight_layout()
    out = RESULTS / "07_ml_vs_comb.png"
    fig.savefig(out, dpi=140)
    print(f"plot -> {out}")

    np.save(
        RESULTS / "07_ml_scorer.npy",
        {"amps": AMPS, "eff_v2": np.array(eff), "thresh": thresh,
         "bg_pred": bg_pred, "bg_max": bg_max,
         "on_p_pred": p_pred, "on_p_max": p_max, "control_irregular": ctrl},
        allow_pickle=True,
    )
    for det in DETECTORS:
        torch.save(models[det].state_dict(), RESULTS / f"07_scorer_{det}.pt")


if __name__ == "__main__":
    main()
