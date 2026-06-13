"""Noise segment discovery, fetching, caching, and whitening.

All sensitivity claims in this project rest on REAL detector noise, so this
module is the foundation: it finds stretches of O3a where the detector was
observing, excludes any stretch containing a catalogued GW event, downloads
them once, and caches locally as HDF5.
"""

from __future__ import annotations

import numpy as np

from . import config as C


def discover_segments(ifo: str, n_segments: int) -> list[int]:
    """Return GPS start times of `n_segments` clean SEGMENT_LEN stretches."""
    from gwosc import datasets
    from gwosc.timeline import get_segments

    scan_end = C.SCAN_START + C.SCAN_DAYS * 86400
    observing = get_segments(f"{ifo}_DATA", C.SCAN_START, scan_end)

    # GPS times of catalogued events in the scan window (to exclude)
    events = datasets.find_datasets(
        type="events", catalog="GWTC-2.1-confident", segment=(C.SCAN_START, scan_end)
    )
    event_gps = []
    for ev in events:
        try:
            event_gps.append(datasets.event_gps(ev))
        except ValueError:
            continue

    starts: list[int] = []
    margin = 256  # stay away from segment boundaries
    for seg_start, seg_end in observing:
        t = seg_start + margin
        while t + C.SEGMENT_LEN + margin <= seg_end:
            # exclude if any catalogued event within the stretch (+/- 64 s pad)
            if not any(t - 64 <= g <= t + C.SEGMENT_LEN + 64 for g in event_gps):
                starts.append(int(t))
            t += C.SEGMENT_LEN
            if len(starts) >= n_segments:
                return starts
    if len(starts) < n_segments:
        raise RuntimeError(
            f"only found {len(starts)} clean segments for {ifo} in scan window"
        )
    return starts


def segment_path(ifo: str, gps_start: int):
    return C.NOISE_DIR / f"{ifo}_{gps_start}_{C.SEGMENT_LEN}.hdf5"


def fetch_segment(ifo: str, gps_start: int, force: bool = False):
    """Download one segment from GWOSC and cache; returns the file path."""
    from gwpy.timeseries import TimeSeries

    path = segment_path(ifo, gps_start)
    if path.exists() and not force:
        return path
    C.NOISE_DIR.mkdir(parents=True, exist_ok=True)
    strain = TimeSeries.fetch_open_data(
        ifo, gps_start, gps_start + C.SEGMENT_LEN, sample_rate=C.SAMPLE_RATE
    )
    tmp = path.with_suffix(".tmp.hdf5")
    strain.write(tmp, path="strain", format="hdf5")
    tmp.rename(path)
    return path


def load_segment(ifo: str, gps_start: int) -> tuple[np.ndarray, float]:
    """Load cached strain; returns (array, gps_t0)."""
    from gwpy.timeseries import TimeSeries

    ts = TimeSeries.read(segment_path(ifo, gps_start), path="strain")
    return np.asarray(ts.value, dtype=np.float64), float(ts.t0.value)


# --------------------------------------------------------------- whitening


def estimate_psd(strain: np.ndarray):
    """Median-Welch PSD of a segment (robust to glitches and to any injected
    signal, which occupies a tiny fraction of the Welch averages)."""
    from pycbc.psd import interpolate, inverse_spectrum_truncation
    from pycbc.types import TimeSeries as PTS

    pts = PTS(strain, delta_t=1.0 / C.SAMPLE_RATE)
    psd = pts.psd(C.PSD_SEG_SEC, avg_method="median")
    # PSD frequency grid must match the grid of rfft(strain)
    psd = interpolate(psd, C.SAMPLE_RATE / len(strain))
    psd = inverse_spectrum_truncation(
        psd, int(C.PSD_SEG_SEC * C.SAMPLE_RATE), low_frequency_cutoff=C.F_LOWER - 10
    )
    return psd


def whiten(strain: np.ndarray, psd) -> np.ndarray:
    """Whiten so the output noise is unit-variance per sample.

    x_w = irfft( rfft(x) * sqrt(2 / (fs * S(f))) )  with one-sided PSD S.

    Two properties (verified in tests/test_injection.py):
      * noise drawn from S -> Var(x_w) = 1 per sample
      * a signal h whitened by the same operator satisfies
        sum(h_w**2) = SNR_opt**2  (the standard 4*int |h~|^2/S df)
    so whitened-domain energy IS matched-filter SNR^2 — labels come free.
    """
    n = len(strain)
    xf = np.fft.rfft(strain)
    if len(psd) != len(xf):  # guard: re-grid the PSD if caller mismatched it
        from pycbc.psd import interpolate as psd_interpolate

        psd = psd_interpolate(psd, C.SAMPLE_RATE / n)
    s = np.asarray(psd.data[: len(xf)], dtype=np.float64).copy()
    s[s <= 0] = np.inf
    w = np.sqrt(2.0 / (C.SAMPLE_RATE * s))
    # Band-limit to the analysis band. Below: detector noise wall. Above:
    # GWOSC 4k data is anti-alias filtered, so the measured PSD near Nyquist
    # is artificially tiny and 1/sqrt(S) explodes there. All SNRs in this
    # project are therefore BAND-LIMITED [F_LOWER-10, F_HIGH] by construction,
    # for data and templates alike.
    freqs = np.fft.rfftfreq(n, d=1.0 / C.SAMPLE_RATE)
    w[(freqs < (C.F_LOWER - 10)) | (freqs > C.F_HIGH)] = 0.0
    return np.fft.irfft(xf * w, n=n)


def whiten_segment(ifo: str, gps_start: int) -> tuple[np.ndarray, float, object]:
    """Load + whiten a cached segment. Returns (whitened, t0, psd).

    The first/last WHITEN_CROP_SEC seconds are corrupted by filter wraparound;
    callers must not draw windows from them.
    """
    strain, t0 = load_segment(ifo, gps_start)
    psd = estimate_psd(strain)
    return whiten(strain, psd), t0, psd
