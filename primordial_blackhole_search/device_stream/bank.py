"""Chirp bank for on-device streaming training (DeepStrain phone node).

The phone cannot afford pbh.waveforms.make_whitened_injection (15.5 s/sample measured on-device, almost
all waveform generation). TaylorF2 here is dominant-mode only, so any orientation is a linear mix of the
face-on, zero-phase polarisations (verified: residual 2e-11):
    h+(i,phi) = (1+cos^2 i)/2 * ( cos2phi*hp0 - sin2phi*hc0 )
    hx(i,phi) =        cos i  * ( sin2phi*hp0 + cos2phi*hc0 )
Whitening is linear too, so per noise segment we whiten hp0, hc0 once and every injection is
    h_w = a*W(hp0) + b*W(hc0),  with a, b from (Fp, Fx, i, phi).
Golden test (test_bank.py) checks this against make_whitened_injection on a real O3a PSD.
"""
from __future__ import annotations

import numpy as np

from pbh import config as C
from pbh.waveforms import BUFFER_SEC, REF_DISTANCE_MPC, InjectionParams, whiten_waveform


def face_on(m1: float, m2: float) -> tuple[np.ndarray, np.ndarray]:
    """Raw (unwhitened) face-on, zero-phase polarisations at REF_DISTANCE_MPC, merger at the last sample."""
    from pycbc.waveform import get_td_waveform

    hp, hc = get_td_waveform(
        approximant=C.APPROXIMANT, mass1=m1, mass2=m2, delta_t=1.0 / C.SAMPLE_RATE,
        f_lower=C.F_LOWER, distance=REF_DISTANCE_MPC, inclination=0.0, coa_phase=0.0,
    )
    hp, hc = hp.numpy(), hc.numpy()
    # whiten_waveform only ever uses the last BUFFER_SEC and zero-fills the rest, so store exactly
    # that span without leading zeros (pycbc pads light systems to 1024 s): identical output, ~half the size.
    n_buf = BUFFER_SEC * C.SAMPLE_RATE
    hp, hc = hp[-n_buf:], hc[-n_buf:]
    nz = np.flatnonzero((hp != 0) | (hc != 0))
    lo = int(nz[0]) if len(nz) else len(hp)
    return hp[lo:].astype(np.float32), hc[lo:].astype(np.float32)


def whiten_pair(hp0: np.ndarray, hc0: np.ndarray, psd) -> tuple[np.ndarray, np.ndarray]:
    return (whiten_waveform(hp0.astype(np.float64), psd), whiten_waveform(hc0.astype(np.float64), psd))


def mix_coeffs(p: InjectionParams, fp: float, fx: float) -> tuple[float, float]:
    ci = np.cos(p.inclination)
    ap, ax = (1 + ci * ci) / 2, ci
    c2, s2 = np.cos(2 * p.coa_phase), np.sin(2 * p.coa_phase)
    return fp * ap * c2 + fx * ax * s2, -fp * ap * s2 + fx * ax * c2


def projected_whitened(wp: np.ndarray, wc: np.ndarray, p: InjectionParams, ifo: str, t_gps: float):
    """Same output contract as pbh.waveforms.make_whitened_injection: (h_w at 1 Mpc, snr_ref)."""
    from pycbc.detector import Detector

    fp, fx = Detector(ifo).antenna_pattern(p.ra, p.dec, p.psi, t_gps)
    a, b = mix_coeffs(p, fp, fx)
    h_w = a * wp + b * wc
    return h_w, float(np.sqrt(np.sum(h_w**2)))


# ---- fast path (phone): identical numerics to the functions above, measured cost 18.3 s/entry -> see bench

def whitening_filter(psd) -> np.ndarray:
    """The sqrt(2/(fs S)) band-limited filter whiten_waveform builds on every call; build it once per segment."""
    from pycbc.psd import interpolate as psd_interpolate

    n_buf = BUFFER_SEC * C.SAMPLE_RATE
    psd_i = psd_interpolate(psd, C.SAMPLE_RATE / n_buf)
    n_f = n_buf // 2 + 1
    s = np.asarray(psd_i.data[:n_f], dtype=np.float64).copy()
    s[s <= 0] = np.inf
    w = np.sqrt(2.0 / (C.SAMPLE_RATE * s))
    freqs = np.fft.rfftfreq(n_buf, d=1.0 / C.SAMPLE_RATE)
    w[(freqs < (C.F_LOWER - 10)) | (freqs > C.F_HIGH)] = 0.0
    return w


def whiten_with(h: np.ndarray, wfilt: np.ndarray) -> np.ndarray:
    n_buf = BUFFER_SEC * C.SAMPLE_RATE
    buf = np.zeros(n_buf)
    buf[-len(h):] = h
    return np.fft.irfft(np.fft.rfft(buf) * wfilt, n=n_buf)


class WhitenedEntry:
    """A bank entry whitened against one segment, with the Gram terms that give snr_ref in O(1)."""

    def __init__(self, hp0: np.ndarray, hc0: np.ndarray, wfilt: np.ndarray):
        self.wp = whiten_with(hp0.astype(np.float64), wfilt)
        self.wc = whiten_with(hc0.astype(np.float64), wfilt)
        self.pp, self.cc, self.pc = float(self.wp @ self.wp), float(self.wc @ self.wc), float(self.wp @ self.wc)

    def inject(self, window, p: InjectionParams, ifo: str, t_gps: float, target_snr: float, merger_idx: int):
        """Same contract and result as inject_into_window(window, *projected_whitened(...), target_snr, merger_idx),
        but mixes only the samples that land in the window."""
        from pycbc.detector import Detector

        fp, fx = Detector(ifo).antenna_pattern(p.ra, p.dec, p.psi, t_gps)
        a, b = mix_coeffs(p, fp, fx)
        snr_ref = float(np.sqrt(a * a * self.pp + 2 * a * b * self.pc + b * b * self.cc))
        scale = target_snr / snr_ref
        n_sig, n_win = len(self.wp), len(window)
        sig_end = merger_idx + 1
        sig_start = sig_end - n_sig
        src_lo = max(0, -sig_start)
        dst_lo = max(0, sig_start)
        dst_hi = min(n_win, sig_end)
        src_hi = src_lo + (dst_hi - dst_lo)
        out = window.copy()
        part = scale * (a * self.wp[src_lo:src_hi] + b * self.wc[src_lo:src_hi])
        if dst_hi > dst_lo:
            out[dst_lo:dst_hi] += part
        return out, float(np.sqrt(np.sum(part**2)))
