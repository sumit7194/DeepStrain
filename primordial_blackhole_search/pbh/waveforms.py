"""Subsolar waveform generation + whitened injection machinery.

Convention that makes everything simple: pbh.data.whiten() normalizes so that
whitened noise has unit variance per sample. A signal whitened by the same
operator then has  sum(h_w**2) == SNR_opt**2,  so:

  * scaling a whitened waveform to a target optimal SNR is just
    h_w * (target / sqrt(sum(h_w**2)))
  * the corresponding luminosity distance is d_ref * (snr_ref / target)

The golden test (tests/test_injection.py) verifies both against PyCBC's
sigma() and a blind matched-filter recovery.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config as C

REF_DISTANCE_MPC = 1.0
BUFFER_SEC = 512  # longest 0.2+0.2 signal from 50 Hz is ~350 s


@dataclass
class InjectionParams:
    mass1: float
    mass2: float
    ra: float
    dec: float
    psi: float
    inclination: float
    coa_phase: float

    @property
    def chirp_mass(self) -> float:
        m1, m2 = self.mass1, self.mass2
        return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


def sample_params(rng: np.random.Generator) -> InjectionParams:
    """Population: component masses log-uniform in [M_MIN, M_MAX], isotropic
    sky and orientation."""
    return InjectionParams(
        mass1=float(np.exp(rng.uniform(np.log(C.M_MIN), np.log(C.M_MAX)))),
        mass2=float(np.exp(rng.uniform(np.log(C.M_MIN), np.log(C.M_MAX)))),
        ra=float(rng.uniform(0, 2 * np.pi)),
        dec=float(np.arcsin(rng.uniform(-1, 1))),
        psi=float(rng.uniform(0, np.pi)),
        inclination=float(np.arccos(rng.uniform(-1, 1))),
        coa_phase=float(rng.uniform(0, 2 * np.pi)),
    )


def projected_waveform(p: InjectionParams, ifo: str, t_gps: float) -> np.ndarray:
    """Detector-frame time-domain waveform at REF_DISTANCE_MPC.

    Returns an array whose LAST sample is (approximately) the merger.
    """
    from pycbc.detector import Detector
    from pycbc.waveform import get_td_waveform

    hp, hc = get_td_waveform(
        approximant=C.APPROXIMANT,
        mass1=p.mass1,
        mass2=p.mass2,
        delta_t=1.0 / C.SAMPLE_RATE,
        f_lower=C.F_LOWER,
        distance=REF_DISTANCE_MPC,
        inclination=p.inclination,
        coa_phase=p.coa_phase,
    )
    fp, fx = Detector(ifo).antenna_pattern(p.ra, p.dec, p.psi, t_gps)
    h = fp * hp.numpy() + fx * hc.numpy()
    return h.astype(np.float64)


def whiten_waveform(h: np.ndarray, psd) -> np.ndarray:
    """Whiten a (short) waveform with the SAME convention as pbh.data.whiten,
    interpolating the PSD to the waveform buffer's frequency resolution."""
    from pycbc.psd import interpolate as psd_interpolate

    n_buf = BUFFER_SEC * C.SAMPLE_RATE
    if len(h) > n_buf:
        h = h[-n_buf:]
    buf = np.zeros(n_buf)
    buf[-len(h) :] = h  # merger at the end of the buffer

    df_buf = C.SAMPLE_RATE / n_buf
    psd_i = psd_interpolate(psd, df_buf)

    xf = np.fft.rfft(buf)
    s = np.asarray(psd_i.data[: len(xf)], dtype=np.float64).copy()
    s[s <= 0] = np.inf
    w = np.sqrt(2.0 / (C.SAMPLE_RATE * s))
    freqs = np.fft.rfftfreq(n_buf, d=1.0 / C.SAMPLE_RATE)
    # same band limits as pbh.data.whiten — SNR is band-limited by convention
    w[(freqs < (C.F_LOWER - 10)) | (freqs > C.F_HIGH)] = 0.0
    return np.fft.irfft(xf * w, n=n_buf)


def make_whitened_injection(
    p: InjectionParams, ifo: str, t_gps: float, psd
) -> tuple[np.ndarray, float]:
    """Returns (whitened waveform at 1 Mpc, optimal SNR at 1 Mpc).

    Trailing whitening corruption is trimmed; the merger sits at the array end
    minus a small pad.
    """
    h = projected_waveform(p, ifo, t_gps)
    h_w = whiten_waveform(h, psd)
    # trim the leading buffer region that is pure zero-padding artifact
    snr_ref = float(np.sqrt(np.sum(h_w**2)))
    return h_w, snr_ref


def inject_into_window(
    window: np.ndarray,
    h_w: np.ndarray,
    snr_ref: float,
    target_snr: float,
    merger_idx: int,
) -> tuple[np.ndarray, float]:
    """Add a whitened waveform (scaled to target_snr) into a whitened-noise
    window so the merger lands at merger_idx. Returns (window, in_window_snr).
    """
    scale = target_snr / snr_ref
    sig = h_w * scale

    out = window.copy()
    n_win = len(window)
    # h_w's merger is at its last sample; align it to merger_idx
    sig_end = merger_idx + 1
    sig_start = sig_end - len(sig)
    src_lo = max(0, -sig_start)
    dst_lo = max(0, sig_start)
    dst_hi = min(n_win, sig_end)
    src_hi = src_lo + (dst_hi - dst_lo)
    if dst_hi > dst_lo:
        out[dst_lo:dst_hi] += sig[src_lo:src_hi]
    in_window_snr = float(np.sqrt(np.sum(sig[src_lo:src_hi] ** 2)))
    return out, in_window_snr
