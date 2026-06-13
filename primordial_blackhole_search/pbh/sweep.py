"""Model sweep over whitened segments — the single source of the window grid.

A SweepGrid fixes how a segment is diced into scored windows. The default is
v1 (256 s windows, 8 s hop, 256 time bins); SweepGrid.short(W) builds a shorter,
non-overlapping grid (rung 2) whose windows each see an independent arc of the
chirp. Spectrogram frames are pure strides (no padding), so window i covers
samples [i*hop_frames*HOP_SAMP, i*hop_frames*HOP_SAMP + win_samp) relative to
the WHITEN_CROP_SEC crop — the same map for every grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from . import config as C
from .spectrogram import spectrogram

HOP_SAMP = int(C.STFT_HOP_SEC * C.SAMPLE_RATE)  # samples per frame hop
NPERSEG = int(C.STFT_SEC * C.SAMPLE_RATE)  # samples per FFT frame
CROP = C.WHITEN_CROP_SEC * C.SAMPLE_RATE


def _frames_for(window_sec: float) -> int:
    """Spectrogram frames spanning a window of window_sec seconds."""
    return 1 + (int(window_sec * C.SAMPLE_RATE) - NPERSEG) // HOP_SAMP


@dataclass(frozen=True)
class SweepGrid:
    frames_per_win: int = 1 + (C.WINDOW_SEC * C.SAMPLE_RATE - NPERSEG) // HOP_SAMP
    hop_frames: int = 16  # v1: 8 s hop between scored windows
    n_time_bins: int = C.N_TIME_BINS

    @classmethod
    def short(cls, window_sec: float) -> "SweepGrid":
        """Shorter, non-overlapping grid (hop == window) for rung 2."""
        f = _frames_for(window_sec)
        return cls(frames_per_win=f, hop_frames=f, n_time_bins=f // 2)

    @property
    def win_samp(self) -> int:
        return (self.frames_per_win - 1) * HOP_SAMP + NPERSEG

    @property
    def win_hop_samp(self) -> int:
        return self.hop_frames * HOP_SAMP

    def n_windows(self, n_frames: int) -> int:
        return len(np.arange(0, n_frames - self.frames_per_win, self.hop_frames))


V1_GRID = SweepGrid()


def pool_and_log(spec_win: np.ndarray, n_time_bins: int = C.N_TIME_BINS) -> np.ndarray:
    n = (spec_win.shape[1] // 2) * 2
    pooled = spec_win[:, :n].reshape(C.N_FREQ_BINS, n // 2, 2).max(axis=2)
    if pooled.shape[1] < n_time_bins:
        pooled = np.pad(
            pooled, ((0, 0), (0, n_time_bins - pooled.shape[1])), mode="edge"
        )
    return np.log1p(pooled[:, :n_time_bins]).astype(np.float32)


@torch.no_grad()
def score_windows(model, device, wins: np.ndarray) -> np.ndarray:
    out = []
    for i in range(0, len(wins), 256):
        batch = torch.from_numpy(wins[i : i + 256]).unsqueeze(1).to(device)
        out.append(model(batch).float().cpu().numpy())
    return np.concatenate(out)


def segment_window_scores(model, device, w: np.ndarray, grid: SweepGrid = V1_GRID):
    """Slide grid windows over a whitened segment, score each."""
    spec = spectrogram(w[CROP:-CROP])
    starts = np.arange(0, spec.shape[1] - grid.frames_per_win, grid.hop_frames)
    wins = np.stack(
        [
            pool_and_log(spec[:, s : s + grid.frames_per_win], grid.n_time_bins)
            for s in starts
        ]
    )
    return score_windows(model, device, wins)
