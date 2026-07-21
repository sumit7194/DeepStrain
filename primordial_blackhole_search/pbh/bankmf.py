"""Dense-bank matched filtering on MPS (follow-up A): FD cross-correlation in the whitened domain.

In this repo's whitening convention (sum h_w^2 = SNR^2), the matched-filter SNR time series against
whitened data d_w is a plain cross-correlation with the unit-normalized whitened template. Phase
maximization uses the analytic template (Hilbert): rho(t) = |ifft(fft(d) * conj(fft(h_analytic)))| / norm
with norm = sqrt(sum |h_analytic|^2 / 2) == snr_ref for a band-limited template. Everything is validated
by scripts/bank_golden.py (noiseless self-correlation == target SNR exactly; quadrature max recovers a
phase-rotated injection; MPS float32 == CPU float64 to <0.5%).
"""
from __future__ import annotations

import numpy as np
import torch


def analytic_spectrum(h_w: np.ndarray, n_fft: int) -> np.ndarray:
    """Full complex spectrum of the analytic (Hilbert) extension of a real template, zero-padded to n_fft."""
    H = np.fft.fft(h_w, n=n_fft)
    n = n_fft
    a = np.zeros(n, dtype=complex)
    a[0] = H[0]
    if n % 2 == 0:
        a[n // 2] = H[n // 2]
        a[1 : n // 2] = 2.0 * H[1 : n // 2]
    else:
        a[1 : (n + 1) // 2] = 2.0 * H[1 : (n + 1) // 2]
    return a


def template_norm(h_w: np.ndarray) -> float:
    """Quadrature norm: sqrt(sum h^2) (= snr_ref at 1 Mpc in this repo's convention)."""
    return float(np.sqrt(np.sum(h_w**2)))


def snr_series_cpu(d_w: np.ndarray, h_w: np.ndarray) -> np.ndarray:
    """Float64 reference: phase-maximized SNR time series rho(t); t = template END (merger) position."""
    n = len(d_w)
    A = analytic_spectrum(h_w, n)
    D = np.fft.fft(d_w.astype(np.float64), n=n)
    corr = np.fft.ifft(D * np.conj(A))
    # corr[t] pairs d_w[t + tau] with h[tau]; template end (merger) then sits at t + len(h) - 1
    rho = np.abs(corr) / template_norm(h_w)
    return np.roll(rho, len(h_w) - 1)


class BankMF:
    """Bank with analytic spectra precomputed ONCE on the device -> each query is 1 FFT + batched
    complex-multiply + iFFT. Templates are padded/cropped to n_fft (the analysis window length)."""

    def __init__(self, templates: list[np.ndarray], n_fft: int, device: str = "mps", batch: int = 64):
        self.n_fft, self.device, self.batch = n_fft, device, batch
        self.norms = torch.tensor([template_norm(h) for h in templates], dtype=torch.float32, device=device)
        specs = np.stack([analytic_spectrum(_fit(h, n_fft), n_fft) for h in templates]).astype(np.complex64)
        self.specs = torch.tensor(specs, device=device)                       # (n_tmpl, n_fft)

    @torch.no_grad()
    def peaks(self, d_w: np.ndarray) -> np.ndarray:
        """Phase-maximized peak SNR over time for every template against one chunk d_w (len n_fft)."""
        D = torch.fft.fft(torch.tensor(_fit(d_w, self.n_fft), dtype=torch.float32, device=self.device), n=self.n_fft)
        out = np.empty(len(self.norms), dtype=np.float64)
        for b in range(0, len(self.norms), self.batch):
            corr = torch.fft.ifft(D.unsqueeze(0) * self.specs[b : b + self.batch].conj(), dim=-1)
            out[b : b + self.batch] = (corr.abs().amax(dim=-1) / self.norms[b : b + self.batch]).float().cpu().numpy()
        return out


def _fit(x: np.ndarray, n: int) -> np.ndarray:
    """Right-align x into length n (crop the head / left-pad with zeros) — templates end at the merger."""
    if len(x) == n:
        return x
    return x[-n:] if len(x) > n else np.pad(x, (n - len(x), 0))


@torch.no_grad()
def snr_peaks_mps(d_w: np.ndarray, templates: list[np.ndarray], device: str = "mps",
                  batch: int = 32) -> np.ndarray:
    """Peak phase-maximized SNR over time for each template against one whitened chunk (batched on MPS)."""
    n = len(d_w)
    D = torch.fft.fft(torch.tensor(d_w, dtype=torch.float32, device=device), n=n)
    peaks = np.empty(len(templates), dtype=np.float64)
    for b in range(0, len(templates), batch):
        chunk = templates[b : b + batch]
        specs = np.stack([analytic_spectrum(h, n) for h in chunk]).astype(np.complex64)
        norms = torch.tensor([template_norm(h) for h in chunk], dtype=torch.float32, device=device)
        A = torch.tensor(specs, device=device)
        corr = torch.fft.ifft(D.unsqueeze(0) * A.conj(), dim=-1)
        rho = corr.abs().amax(dim=-1) / norms
        peaks[b : b + batch] = rho.float().cpu().numpy()
    return peaks
