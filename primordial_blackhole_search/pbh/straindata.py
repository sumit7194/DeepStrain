"""On-the-fly strain-injection dataset for the stage-1 semi-coherent model.

Yields 64-s WHITENED STRAIN windows: a random noise slice, with probability
p_inject a pooled whitened waveform added at a random merger position and SNR.
Because both noise and waveforms are pre-whitened (same convention), injection
is a cheap array-add — no waveform generation or FFT in the loop, so
num_workers=0 keeps up with the model easily.

target_snr is the FULL-signal optimal SNR (pool snr_ref), matching the oracle /
cnn_w64 convention; the waveform's last 64 s carry the in-window portion.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from . import config as C

WIN = 64 * C.SAMPLE_RATE
CROP = C.WHITEN_CROP_SEC * C.SAMPLE_RATE


class StrainInjectionDataset(Dataset):
    def __init__(self, noise, pool_x, pool_snr, n_samples,
                 snr_range=(6.0, 40.0), p_inject=0.5, deterministic=False):
        self.noise = noise            # list of 1-D whitened segment arrays
        self.pool_x = pool_x          # (Npool, WIN) float16, merger at last sample
        self.pool_snr = pool_snr      # (Npool,) full optimal SNR
        self.n = n_samples
        self.snr_range = snr_range
        self.p = p_inject
        self.det = deterministic

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        rng = np.random.default_rng(i) if self.det else np.random.default_rng()
        seg = self.noise[rng.integers(len(self.noise))]
        s = int(rng.integers(CROP, len(seg) - CROP - WIN))
        win = seg[s : s + WIN].astype(np.float32).copy()
        label = 0.0
        if rng.random() < self.p:
            k = int(rng.integers(len(self.pool_x)))
            wf = self.pool_x[k].astype(np.float32)   # last 64 s, merger at end
            target = float(rng.uniform(*self.snr_range))
            scale = target / float(self.pool_snr[k])
            m = int(rng.integers(WIN // 2, WIN))     # merger position in-window
            win[:m] += scale * wf[WIN - m :]         # align waveform end to m
            label = 1.0
        return torch.from_numpy(win).unsqueeze(0), np.float32(label)


def load_noise(gps_list) -> list:
    """Load + whiten the given segments once (held in RAM for the dataset)."""
    from .data import whiten_segment

    out = []
    for gps in gps_list:
        w, _, _ = whiten_segment("H1", gps)
        out.append(w.astype(np.float32))
    return out
