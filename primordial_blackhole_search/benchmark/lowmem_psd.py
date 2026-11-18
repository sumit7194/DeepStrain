"""Memory-light equivalents of pbh.data.estimate_psd and pbh.data.whiten, for full 4096-s segments.

Why: measured on a 4096-s segment, estimate_psd raises peak RSS by ~1.6 GB (pycbc welch +0.47 GB,
inverse_spectrum_truncation +0.83 GB), which put Y0 at 3.5 GB and turned the shared box yellow (2026-11-18). These
functions do the same arithmetic in numpy with fewer temporaries:
  welch (median, hann, 16-s segments, 8-s stride, DC/Nyquist halved, median_bias, scale 2 df L / sum w^2)
  -> numpy.interp onto the data's frequency grid (pycbc.psd.interpolate)
  -> inverse_spectrum_truncation: inv_asd = psd^-1/2 on [kmin, N/2), q = irfft, zero q[L/2 : N - L/2],
     psd = 1/|rfft(q)|^2  (pycbc's fft/ifft scale factors delta_t and delta_f cancel over the round trip)
They must agree with the pbh versions; `python benchmark/lowmem_psd.py <npy>` checks PSD and whitened output and writes
benchmark/results/lowmem_psd_check.json. Nothing uses them until that check passes.
"""
import json
import resource
import sys
from pathlib import Path

import numpy as np

B = Path(__file__).resolve().parent
sys.path.insert(0, str(B.parent))
from pbh import config as C  # noqa: E402


def _median_bias(n):
    if n >= 1000:
        return np.log(2)
    return 1 + sum(1.0 / (2 * i + 1) - 1.0 / (2 * i) for i in range(1, (n - 1) // 2 + 1))


def estimate_psd(x):
    """Returns the PSD on the rfft grid of x (a numpy array, not a pycbc FrequencySeries)."""
    fs, n = C.SAMPLE_RATE, len(x)
    L = int(round(C.PSD_SEG_SEC * fs)); S = L // 2
    nseg = n // S
    if (nseg - 1) * S + L > n:
        nseg -= 1
    data_len = (nseg - 1) * S + L
    if data_len < n:                                   # pycbc's centring of the used data
        diff = n - data_len
        a = diff // 2 + (diff % 2)
        x = x[a:n - diff // 2]
    win = np.hanning(L)
    dt, df = 1.0 / fs, fs / L
    P = np.empty((nseg, L // 2 + 1))
    for i in range(nseg):
        P[i] = np.abs(np.fft.rfft(x[i * S:i * S + L] * win) * dt) ** 2
    P[:, 0] /= 2; P[:, -1] /= 2
    psd = np.median(P, axis=0) / _median_bias(nseg)
    del P
    psd *= 2 * df * L / (win * win).sum()
    # pycbc.psd.interpolate onto delta_f = fs / n
    new_df = fs / n
    new_n = int(np.rint((len(psd) - 1) * df / new_df + 1))
    psd = np.interp(np.arange(new_n) * new_df, np.arange(len(psd)) * df, psd)
    # inverse_spectrum_truncation(psd, L, low_frequency_cutoff=F_LOWER - 10)
    N = (len(psd) - 1) * 2
    kmin = int((C.F_LOWER - 10) / new_df)
    inv = np.zeros(len(psd))
    inv[kmin:N // 2] = psd[kmin:N // 2] ** -0.5
    del psd
    q = np.fft.irfft(inv, n=N)
    del inv
    q[L // 2:N - L // 2] = 0
    pt = np.fft.rfft(q)
    del q
    out = np.abs(pt) ** 2
    del pt
    return 1.0 / out


def whiten(x, psd):
    """Same as pbh.data.whiten for a numpy PSD already on x's rfft grid."""
    n = len(x)
    s = np.asarray(psd, dtype=np.float64).copy()
    s[s <= 0] = np.inf
    w = np.sqrt(2.0 / (C.SAMPLE_RATE * s))
    del s
    freqs = np.fft.rfftfreq(n, d=1.0 / C.SAMPLE_RATE)
    w[(freqs < (C.F_LOWER - 10)) | (freqs > C.F_HIGH)] = 0.0
    del freqs
    xf = np.fft.rfft(x)
    xf *= w
    del w
    return np.fft.irfft(xf, n=n)


def check(path):
    from pbh.data import estimate_psd as ref_psd, whiten as ref_whiten
    peak = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 2 ** 20  # macOS: bytes
    x = np.load(path).astype(np.float64)
    p0 = peak()
    psd = estimate_psd(x); w = whiten(x, psd)
    p1 = peak()
    rp = ref_psd(x); rw = ref_whiten(x, rp)
    f = np.fft.rfftfreq(len(x), 1 / C.SAMPLE_RATE)
    band = (f >= C.F_LOWER - 10) & (f <= C.F_HIGH)
    r = np.asarray(rp.data)
    res = dict(file=str(path), n_psd_bins=len(psd), ref_bins=len(r),
               psd_max_rel_diff_in_band=float(np.max(np.abs(psd[band] / r[band] - 1))),
               whitened_max_abs_diff=float(np.max(np.abs(w - rw))), whitened_rms=float(np.std(rw)),
               peak_rss_mb_after_load=p0, peak_rss_mb_lowmem=p1, peak_rss_mb_with_reference=peak())
    res["pass"] = res["n_psd_bins"] == res["ref_bins"] and res["psd_max_rel_diff_in_band"] < 1e-9 \
        and res["whitened_max_abs_diff"] < 1e-9 * res["whitened_rms"]
    (B / "results" / "lowmem_psd_check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1)); print("LOWMEM", "PASS" if res["pass"] else "FAIL")


if __name__ == "__main__":
    check(sys.argv[1])
