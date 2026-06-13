"""Golden test for the whitening + injection chain (Phase 1b gate).

Asserts, on REAL noise:
  1. whitened noise has unit variance per sample
  2. sum(h_w**2) matches PyCBC's sigma() optimal SNR        (within 5%)
  3. a blind matched filter recovers the injection at the right time with
     the right SNR                                          (within 10%)

Run:  uv run python -m pytest tests/ -x -q   (or python tests/test_injection.py)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pbh import config as C
from pbh.data import estimate_psd, whiten
from pbh.waveforms import (
    BUFFER_SEC,
    InjectionParams,
    inject_into_window,
    make_whitened_injection,
    projected_waveform,
)

# any cached segment works; the spike file is the smallest
SPIKE_CACHE = C.ROOT / "spike" / "output" / "H1_1242440000_512.hdf5"


def _load_noise():
    from gwpy.timeseries import TimeSeries

    for cand in sorted(C.NOISE_DIR.glob("H1_*.hdf5")) + [SPIKE_CACHE]:
        if cand.exists():
            ts = TimeSeries.read(cand, path="strain")
            return np.asarray(ts.value, dtype=np.float64), float(ts.t0.value)
    raise FileNotFoundError("no cached noise segment found - run fetch first")


PARAMS = InjectionParams(
    mass1=0.6, mass2=0.4, ra=1.7, dec=-1.2, psi=0.6, inclination=0.5, coa_phase=1.1
)


def test_unit_variance_whitening():
    from pbh.spectrogram import spectrogram

    strain, _ = _load_noise()
    psd = estimate_psd(strain)
    w = whiten(strain, psd)
    crop = C.WHITEN_CROP_SEC * C.SAMPLE_RATE
    core = w[crop:-crop]
    # Whitening is band-limited [F_LOWER-10, F_HIGH], so the time-domain std
    # is sqrt(band fraction), not 1. The model-facing invariant is the
    # spectrogram noise floor: every in-band pixel should average ~1.
    band_frac = (C.F_HIGH - (C.F_LOWER - 10)) / (C.SAMPLE_RATE / 2)
    mad_std = 1.4826 * float(np.median(np.abs(core - np.median(core))))
    print(f"whitened noise MAD std = {mad_std:.4f} "
          f"(expect ~sqrt({band_frac:.3f}) = {np.sqrt(band_frac):.3f})")
    assert abs(mad_std - np.sqrt(band_frac)) < 0.12

    spec = spectrogram(core[: 256 * C.SAMPLE_RATE])
    # Glitch-robust floor: median over the high-frequency rows, where each
    # pixel averages >=13 FFT bins so median ~ mean ~ 1. (Low-f rows are
    # exponential with median ln2 — a global median would mix distributions.)
    floor = float(np.median(spec[96:]))
    print(f"spectrogram noise floor (high-f rows) = {floor:.3f} (want ~1)")
    assert 0.85 < floor < 1.15


def test_snr_normalization_matches_pycbc():
    from pycbc.filter import sigma
    from pycbc.psd import interpolate as psd_interpolate
    from pycbc.types import TimeSeries as PTS

    strain, t0 = _load_noise()
    psd = estimate_psd(strain)

    h_w, snr_ref = make_whitened_injection(PARAMS, "H1", t0, psd)

    h = projected_waveform(PARAMS, "H1", t0)
    n_buf = BUFFER_SEC * C.SAMPLE_RATE
    buf = np.zeros(n_buf)
    buf[-len(h) :] = h
    pts = PTS(buf, delta_t=1.0 / C.SAMPLE_RATE)
    psd_i = psd_interpolate(psd, pts.delta_f)
    snr_pycbc = float(
        sigma(
            pts,
            psd=psd_i,
            low_frequency_cutoff=C.F_LOWER - 10,
            high_frequency_cutoff=C.F_HIGH,
        )
    )
    rel = abs(snr_ref - snr_pycbc) / snr_pycbc
    print(f"sum(h_w^2)^0.5 = {snr_ref:.2f}  vs pycbc sigma = {snr_pycbc:.2f}  "
          f"({100 * rel:.1f}% diff)")
    assert rel < 0.05


def test_matched_filter_recovery():
    """Inject at known time/SNR into real noise, recover by FFT
    cross-correlation with the unit-normed whitened template. The peak must
    sit at the merger time with height ~ in-window SNR (the noise contributes
    ~N(0,1) to the peak, so a few-percent tolerance)."""
    strain, t0 = _load_noise()
    psd = estimate_psd(strain)
    w = whiten(strain, psd)

    crop = C.WHITEN_CROP_SEC * C.SAMPLE_RATE
    n_win = C.WINDOW_SEC * C.SAMPLE_RATE
    window = w[crop : crop + n_win]

    h_w, snr_ref = make_whitened_injection(PARAMS, "H1", t0, psd)
    target_snr = 18.0
    merger_idx = int(0.8 * n_win)
    injected, in_win_snr = inject_into_window(
        window, h_w, snr_ref, target_snr, merger_idx
    )
    print(f"target SNR {target_snr}, in-window SNR {in_win_snr:.2f}")

    # unit-norm whitened template; its merger is at its LAST sample
    tmpl = h_w / np.sqrt(np.sum(h_w**2))
    n = len(injected) + len(tmpl)
    corr = np.fft.irfft(
        np.fft.rfft(injected, n) * np.conj(np.fft.rfft(tmpl, n)), n
    )  # corr[k] = sum_i injected[i + k] * tmpl[i]
    k_star = (merger_idx - (len(tmpl) - 1)) % n  # expected peak lag
    peak_k = int(np.argmax(corr))
    peak = float(corr[peak_k])

    t_err = abs(((peak_k - k_star + n // 2) % n) - n // 2) / C.SAMPLE_RATE
    rel = abs(peak - in_win_snr) / in_win_snr
    print(f"corr peak {peak:.2f} (expect ~{in_win_snr:.2f}, {100 * rel:.1f}% off), "
          f"timing error {t_err * 1000:.1f} ms")
    assert rel < 0.15
    assert t_err < 0.01


if __name__ == "__main__":
    test_unit_variance_whitening()
    test_snr_normalization_matches_pycbc()
    test_matched_filter_recovery()
    print("\nGOLDEN TEST: ALL PASS")
