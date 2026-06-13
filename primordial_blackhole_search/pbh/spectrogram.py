"""Whitened strain -> log-frequency spectrogram, the model input.

Normalization: with unit-variance white noise input, every pixel has mean ~1
before the log. So the noise floor is flat and signals appear as excess power
— the network never needs to learn the detector's spectral shape (whitening
already removed it).
"""

from __future__ import annotations

import numpy as np

from . import config as C

_NPERSEG = int(C.STFT_SEC * C.SAMPLE_RATE)  # 4096
_HOP = int(C.STFT_HOP_SEC * C.SAMPLE_RATE)  # 2048
_WINDOW = np.hanning(_NPERSEG)
_WIN_NORM = float(np.sum(_WINDOW**2))

# log-spaced frequency bin edges, F_LOWER..F_HIGH
_EDGES = np.geomspace(C.F_LOWER, C.F_HIGH, C.N_FREQ_BINS + 1)
_FFT_FREQS = np.fft.rfftfreq(_NPERSEG, d=1.0 / C.SAMPLE_RATE)
_BIN_IDX = np.searchsorted(_FFT_FREQS, _EDGES)


def spectrogram(x: np.ndarray) -> np.ndarray:
    """(n_samples,) whitened strain -> (N_FREQ_BINS, n_frames) power, noise
    floor ~1 per pixel."""
    n_frames = 1 + (len(x) - _NPERSEG) // _HOP
    idx = np.arange(_NPERSEG)[None, :] + _HOP * np.arange(n_frames)[:, None]
    frames = x[idx] * _WINDOW[None, :]
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2 / _WIN_NORM  # (T, F)

    # aggregate into log-spaced frequency bins (mean keeps noise floor at 1)
    out = np.empty((C.N_FREQ_BINS, n_frames), dtype=np.float32)
    for k in range(C.N_FREQ_BINS):
        lo, hi = _BIN_IDX[k], max(_BIN_IDX[k] + 1, _BIN_IDX[k + 1])
        out[k] = power[:, lo:hi].mean(axis=1)
    return out


def model_input(window: np.ndarray, n_time_bins: int = C.N_TIME_BINS) -> np.ndarray:
    """Whitened window -> (N_FREQ_BINS, n_time_bins) float16 log-power. The
    default reproduces v1 (256 s -> 256 bins); a shorter window passes its own
    natural pooled length (e.g. 64 s -> 63 bins) so it is not padded out."""
    spec = spectrogram(window)
    # max-pool time by 2 (preserves the thin track better than mean)
    n = (spec.shape[1] // 2) * 2
    spec = spec[:, :n].reshape(C.N_FREQ_BINS, n // 2, 2).max(axis=2)
    # pad/crop to exactly n_time_bins
    if spec.shape[1] < n_time_bins:
        spec = np.pad(spec, ((0, 0), (0, n_time_bins - spec.shape[1])), mode="edge")
    spec = spec[:, :n_time_bins]
    return np.log1p(spec).astype(np.float16)
