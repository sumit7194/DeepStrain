"""Shared library for the echo search pipeline.

Design notes
------------
- Everything operates on WHITENED strain. Whitening is linear, so injecting a
  whitened template into whitened noise is equivalent to injecting raw strain
  and whitening afterward (v1 simplification, recorded in notes/lab_notebook.md).
  Amplitudes are therefore in units of whitened-noise standard deviations.
- The search statistic is deliberately SHAPE-AGNOSTIC: it never assumes the echo
  waveform, only that energy repeats at a fixed spacing. It works on the energy
  envelope's autocorrelation, summed at the first few multiples of a candidate
  spacing (a "comb").
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from gwosc.datasets import event_gps
from gwpy.timeseries import TimeSeries
from scipy.signal import hilbert

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
PROGRESS = RESULTS / "progress"


def progress(run: str, step: int, total: int, **metrics) -> None:
    """Heartbeat for the repo dashboard (same pattern as curvature's curvlib)."""
    import json
    import time

    PROGRESS.mkdir(parents=True, exist_ok=True)
    f = PROGRESS / f"{run}.json"
    hist = []
    if f.exists():
        try:
            hist = json.loads(f.read_text()).get("history", [])
        except json.JSONDecodeError:
            pass
    loss = metrics.get("loss")
    if loss is not None:
        hist = (hist + [[step, float(loss)]])[-200:]
    tmp = f.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(
            {"run": run, "step": int(step), "total": int(total),
             "metrics": {k: float(v) for k, v in metrics.items()},
             "history": hist, "updated": time.time()}
        )
    )
    tmp.replace(f)
    idx = PROGRESS / "index.json"
    try:
        names = json.loads(idx.read_text()) if idx.exists() else []
    except json.JSONDecodeError:
        names = []
    if run not in names:
        names.append(run)
        idx.write_text(json.dumps(names))

DETECTORS = ("H1", "L1")
BAND = (30.0, 350.0)  # Hz, audio band where BBH ringdowns (and their echoes) live

# Predicted echo spacing for GW150914 (Abedi et al., PRD 96, 082004, Table I).
GW150914_DT_PRED = 0.2925  # seconds, +-0.00916


# ---------------------------------------------------------------- data access
def fetch_block(det: str, event: str, span: float = 512.0) -> TimeSeries:
    """Fetch (cached) a long block of strain centred on `event` for `det`.

    One long block serves both the on-source segment and hundreds of
    off-source background segments, with a single download.
    """
    DATA.mkdir(exist_ok=True)
    gps = event_gps(event)
    cache = DATA / f"{event}_{det}_{int(span)}s.hdf5"
    if cache.exists():
        return TimeSeries.read(cache)
    ts = TimeSeries.fetch_open_data(det, gps - span / 2, gps + span / 2, cache=False)
    ts.write(cache, overwrite=True)
    return ts


def _longest_finite(ts: TimeSeries) -> TimeSeries:
    """Crop to the longest contiguous NaN-free span (GWOSC blocks can include
    data gaps when the science segment starts/ends inside the requested span)."""
    finite = np.isfinite(ts.value)
    if finite.all():
        return ts
    best_len, best_start, run_start = 0, 0, None
    for i, ok in enumerate(finite):
        if ok and run_start is None:
            run_start = i
        elif not ok and run_start is not None:
            if i - run_start > best_len:
                best_len, best_start = i - run_start, run_start
            run_start = None
    if run_start is not None and len(finite) - run_start > best_len:
        best_len, best_start = len(finite) - run_start, run_start
    fs = float(ts.sample_rate.value)
    t0 = float(ts.t0.value)
    return ts.crop(t0 + best_start / fs, t0 + (best_start + best_len) / fs)


def whiten_bp(ts: TimeSeries, band: tuple[float, float] = BAND) -> TimeSeries:
    """Whiten against the block's own PSD, then bandpass to the search band."""
    return ts.whiten(4, 2).bandpass(*band)


@dataclass
class Segments:
    """Whitened on-source and off-source segments for one detector."""

    on: np.ndarray  # the post-ringdown segment after the merger
    off: list[np.ndarray]  # event-free segments, the background population
    fs: float  # sample rate [Hz]


def load_segments(
    event: str,
    det: str,
    seg_dur: float = 3.0,
    on_start: float = 0.05,
    exclude: float = 8.0,
    span: float = 512.0,
    edge: float = 8.0,
    band: tuple[float, float] = BAND,
) -> Segments:
    """Whiten one long block and slice it into on-source + background segments.

    on-source: [merger + on_start, merger + on_start + seg_dur]
    off-source: non-overlapping seg_dur slices, excluding +-`exclude` s around
    the merger and `edge` s at the block boundaries (whitening edge effects).
    """
    gps = event_gps(event)
    raw = _longest_finite(fetch_block(det, event, span))
    block = whiten_bp(raw, band)
    fs = float(block.sample_rate.value)
    t0 = float(block.t0.value)
    x = block.value

    def slice_at(t_start: float) -> np.ndarray:
        i = int(round((t_start - t0) * fs))
        n = int(round(seg_dur * fs))
        return x[i : i + n]

    on = slice_at(gps + on_start)

    off = []
    t = t0 + edge
    while t + seg_dur < t0 + span - edge:
        # skip anything overlapping the exclusion zone around the merger
        if not (t < gps + exclude and t + seg_dur > gps - exclude):
            seg = slice_at(t)
            if len(seg) == int(round(seg_dur * fs)):
                off.append(seg)
        t += seg_dur
    return Segments(on=on, off=off, fs=fs)


# ------------------------------------------------------------------ injection
def echo_train(
    n: int,
    fs: float,
    dt: float,
    amp: float,
    f0: float = 250.0,
    tau: float = 0.02,
    gamma: float = 0.7,
    n_echoes: int = 6,
    t_first: float = 0.1,
    phase_flip: bool = True,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Synthetic echo train: damped sine-Gaussian pulses repeating every `dt`.

    Phenomenological model from the echo literature: each bounce is a damped
    ringdown-like pulse at ~the remnant's ringdown frequency `f0`, with
    amplitude decaying by the wall reflectivity `gamma` per bounce and
    (optionally) a sign flip per reflection. `amp` is the peak amplitude of the
    FIRST pulse in units of whitened-noise sigma (~ per-pulse SNR scale).
    """
    t = np.arange(n) / fs
    out = np.zeros(n)
    if rng is None:
        rng = np.random.default_rng()
    phase = rng.uniform(0, 2 * np.pi)
    for k in range(n_echoes):
        t_k = t_first + k * dt
        if t_k >= t[-1]:
            break
        sign = (-1) ** k if phase_flip else 1.0
        dt_k = t - t_k
        pulse = np.where(
            dt_k >= 0,
            np.exp(-dt_k / tau) * np.sin(2 * np.pi * f0 * dt_k + phase),
            0.0,
        )
        out += sign * amp * (gamma**k) * pulse
    return out


# ------------------------------------------------------------- the statistic
def envelope(x: np.ndarray, fs: float, smooth_ms: float = 10.0) -> np.ndarray:
    """Energy envelope: |analytic signal|, lightly smoothed, mean-removed."""
    env = np.abs(hilbert(x))
    w = max(3, int(round(smooth_ms * 1e-3 * fs)))
    kernel = np.ones(w) / w
    env = np.convolve(env, kernel, mode="same")
    env -= env.mean()
    return env


def _acf(env: np.ndarray) -> np.ndarray:
    """Normalised autocorrelation of the envelope (positive lags)."""
    n = len(env)
    f = np.fft.rfft(env, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    acf /= acf[0]
    return acf

def comb_on_env(
    env: np.ndarray, fs: float, dt_grid: np.ndarray, n_teeth: int = 3
) -> np.ndarray:
    """Comb statistic on an (already mean-removed) energy envelope: sum its
    autocorrelation at lags dt, 2*dt, ..., n_teeth*dt. Repeating energy -> all
    teeth positive; noise -> teeth average to ~0. (The v2 ML scorer feeds a
    reconstruction-error envelope through this same machinery.)"""
    acf = _acf(env)
    lags = np.arange(len(acf)) / fs
    score = np.zeros(len(dt_grid))
    for j, dt in enumerate(dt_grid):
        teeth = []
        for k in range(1, n_teeth + 1):
            lag = k * dt
            if lag >= lags[-1]:
                break
            teeth.append(np.interp(lag, lags, acf))
        score[j] = np.sum(teeth) if teeth else 0.0
    return score


def comb_score(
    x: np.ndarray, fs: float, dt_grid: np.ndarray, n_teeth: int = 3
) -> np.ndarray:
    """Shape-agnostic comb statistic on raw whitened strain (v1 baseline)."""
    return comb_on_env(envelope(x, fs), fs, dt_grid, n_teeth)


def detection_statistic(
    segs: dict[str, np.ndarray], fs: float, dt_grid: np.ndarray, n_teeth: int = 3
) -> tuple[float, float]:
    """Network statistic: sum the per-detector comb scores at the SAME spacing
    (a real echo must repeat identically in both detectors), then take the max
    over the spacing grid. Returns (max_score, best_dt)."""
    total = np.zeros(len(dt_grid))
    for x in segs.values():
        total += comb_score(x, fs, dt_grid, n_teeth)
    i = int(np.argmax(total))
    return float(total[i]), float(dt_grid[i])
