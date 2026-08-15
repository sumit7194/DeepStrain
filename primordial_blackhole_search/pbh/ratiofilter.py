"""Ratio-filter dechirping (L1): derive a target template's SNR series from a NEARBY reference's, via a
short FIR, instead of running a full matched filter per template.

METHOD (arXiv:2601.18835, PRD 10.1103/k21q-wp8f — "Beyond FINDCHIRP"). Write a target template's analytic
spectrum as the reference's times a ratio, A_t(f) = A_r(f) R(f) with R = A_t / A_r. Because neighbouring
templates share almost all of their orbital phase evolution, R is SLOWLY varying in f, so its inverse
transform is a SHORT kernel. Our matched filter is a cross-correlation, so with

    c_x(t) = IFFT[ D(f) conj(A_x(f)) ]                     (the COMPLEX correlation series, pre-|.|)

substituting A_t = A_r R gives conj(A_t) = conj(A_r) conj(R), hence

    c_t = IFFT[ FFT(c_r) conj(R) ] = c_r  (*)  IFFT[conj(R)]

i.e. **the target's complex correlation series is the reference's convolved with one short FIR kernel.**
rho_t(t) = |c_t(t)| / norm_t as usual. This is exact up to truncating that kernel.

WHY IT MATTERS HERE — it is a MEMORY win for us, not (mainly) a FLOP win. bank_dense.py records the wall
plainly: "B=1617 cannot hold all analytic chunks in RAM (33 MB/template -> 53 GB)", which forced a
template-major rewrite that regenerates every template. Under ratio filtering we store analytic chunks only
for the ~1% reference templates and a ~256-tap kernel (a few KB) per target, so a bank an order of magnitude
denser fits in the memory that 1,617 templates used to need. The paper's headline 8x is a CPU
cache/memory-bandwidth result (its FLOP gain is only ~2x: O(N log K) at ~11 FLOP vs O(N log N) at ~20); on
MPS our bottleneck differs, so **the speedup must be measured on our hardware, never assumed** — that is
what scripts/bank_ratio_golden.py does alongside the accuracy check.

FIR DESIGN. Naive IFFT-and-truncate of conj(R) is poor because R is ill-conditioned wherever the reference
has little power (R = A_t/A_r blows up out of band). Instead fit the taps by WEIGHTED least squares over
frequency, weighting by the reference's own power |A_r|^2 — accuracy is only needed where the filter
actually sees signal. Minimise

    chi^2 = sum_f w(f) | R(f) - sum_k a_k exp(-2 pi i f t_k) |^2 ,   w = |A_r|^2

whose normal equations are TOEPLITZ, since (E^H W E)[j,k] = sum_f w(f) exp(2 pi i f (t_j - t_k)) depends
only on j-k. So both the matrix and the right-hand side come from single inverse FFTs of w and w*R, making
the fit O(n_fft log n_fft + n_taps^2) rather than O(n_fft * n_taps^2).

The kernel is allowed to be NON-CAUSAL (taps at negative lag): two templates of different chirp mass are
offset in time, and forcing causality would spend taps representing a pure delay.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_toeplitz


def ratio_taps(a_target: np.ndarray, a_ref: np.ndarray, n_taps: int = 257,
               n_fft: int | None = None) -> np.ndarray:
    """Weighted-least-squares FIR kernel k[t] with c_target ~= c_ref (*) k.

    a_target, a_ref: analytic (complex) chunk waveforms, same length. n_taps is forced odd so the kernel is
    centred on zero lag. Returns the kernel ordered from lag -(n_taps//2) to +(n_taps//2)."""
    if n_taps % 2 == 0:
        n_taps += 1
    n = n_fft or len(a_ref)
    A_t = np.fft.fft(a_target, n=n)
    A_r = np.fft.fft(a_ref, n=n)
    w = np.abs(A_r) ** 2
    tot = w.sum()
    if tot <= 0:
        return np.zeros(n_taps, dtype=complex)
    # R only where the reference has support; the weight makes the rest irrelevant anyway, but forming it
    # explicitly avoids inf/nan poisoning the FFTs below.
    keep = w > (1e-12 * w.max())
    R = np.zeros(n, dtype=complex)
    R[keep] = A_t[keep] / A_r[keep]

    # conj(R) is what convolves c_ref (see the module docstring), so fit THAT.
    target = np.conj(R)
    half = n_taps // 2
    acf = np.fft.ifft(w) * n                      # acf[m] = sum_f w(f) exp(2 pi i f m / n)
    rhs_full = np.fft.ifft(w * target) * n
    lags = np.arange(-half, half + 1)
    rhs = rhs_full[lags % n]
    col = acf[np.arange(0, n_taps) % n]
    reg = 1e-9 * float(np.real(col[0]))           # Tikhonov: the fit is rank-deficient out of band
    col = col.copy(); col[0] += reg
    try:
        taps = solve_toeplitz((col, np.conj(col)), rhs)
    except Exception:
        T = np.empty((n_taps, n_taps), dtype=complex)
        for j in range(n_taps):
            for k in range(n_taps):
                T[j, k] = acf[(j - k) % n]
        T[np.diag_indices(n_taps)] += reg
        taps = np.linalg.solve(T, rhs)
    return taps


def apply_taps(c_ref: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """Convolve a reference complex correlation series with a centred FIR kernel (circular, matching the
    circular correlation the FFT matched filter already produces)."""
    n = len(c_ref)
    half = len(taps) // 2
    K = np.fft.fft(np.roll(np.pad(taps, (0, n - len(taps))), -half), n=n)
    return np.fft.ifft(np.fft.fft(c_ref) * K)


def corr_series(d: np.ndarray, a: np.ndarray, n_fft: int | None = None) -> np.ndarray:
    """The COMPLEX correlation series c(t) = IFFT[D conj(A)] that ratio filtering operates on. Taking
    |c| / norm reproduces the usual phase-maximized rho(t)."""
    n = n_fft or len(d)
    D = np.fft.fft(d, n=n)
    A = np.fft.fft(a, n=n)
    return np.fft.ifft(D * np.conj(A))


def match(x: np.ndarray, y: np.ndarray) -> float:
    """Normalized agreement between two real series (1.0 = identical up to scale)."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    if nx == 0 or ny == 0:
        return 0.0
    return float(np.dot(x, y) / (nx * ny))
