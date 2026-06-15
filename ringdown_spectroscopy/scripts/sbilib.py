"""Shared SBI components for the no-hair NPE (extracted from 09/10 to clear
the recorded duplication debt). NOTE the pickle constraint: posteriors saved
before this refactor reference Embed via the saving script's __main__, so
loading scripts keep a module-level `Embed` name (imported from here)."""

import numpy as np
import torch

import rdlib

FS = 4096.0
SEG = 0.04
N_SAMP = int(SEG * FS) + int(SEG * FS) % 2
T0_MAX_MS = 6.0
PEAK_AMP_RANGE = (2.0, 12.0)

kerr220, kerr221 = rdlib.KerrMap(2, 2, 0), rdlib.KerrMap(2, 2, 1)
CHI_GRID = np.linspace(0.01, 0.97, 200)
W220 = np.array([kerr220.f_tau(1.0, c) for c in CHI_GRID])
W221 = np.array([kerr221.f_tau(1.0, c) for c in CHI_GRID])


def simulate(mass, chi, delta, rng, n_det=2):
    """Whitened-domain segment: Kerr 220 + (1+delta)-shifted 221 + white noise."""
    i = min(max(np.searchsorted(CHI_GRID, chi), 0), len(CHI_GRID) - 1)
    f1, tau1 = W220[i][0] / mass, W220[i][1] * mass
    f2, tau2 = W221[i][0] / mass * (1.0 + delta), W221[i][1] * mass
    t = np.arange(N_SAMP) / FS
    t0 = rng.uniform(0, T0_MAX_MS / 1000.0)
    x = np.empty((n_det, N_SAMP), dtype=np.float32)
    a220 = rng.uniform(*PEAK_AMP_RANGE)
    for d in range(n_det):
        amp1 = a220 * rng.uniform(0.7, 1.3)
        amp2 = amp1 * rng.uniform(0.5, 1.5)
        params = [
            dict(f=f1, tau=tau1, amp=amp1, phi=rng.uniform(-np.pi, np.pi)),
            dict(f=f2, tau=tau2, amp=amp2, phi=rng.uniform(-np.pi, np.pi)),
        ]
        x[d] = rdlib.damped_sinusoids(t, t0, params) + rng.standard_normal(N_SAMP)
    return x


def simulate_tonecount(mass, chi, n_tones, amp_frac, rng, n_det=2, noise=None, snr_match=False):
    """v4 tone-count sim: 1-tone (220 only) or 2-tone (220 + Kerr-locked 221).

    delta=0 (tone-count tests PRESENCE of the overtone, not its frequency).
    amp_frac = 221/220 peak-amplitude ratio (ignored for 1-tone). Returns
    (whitened segment (n_det, N_SAMP), injected overtone matched-filter SNR over
    detectors -- 0 for 1-tone). Start time + per-detector amplitudes are drawn
    here, so both are marginalized by construction (the start-time dependence is
    the crux of the overtone controversy).

    noise: optional (n_det, N_SAMP) real whitened-noise windows to inject into
    (v4 fix B, domain-matched). If None, idealized unit white noise (the first
    cut). Whitened-domain injection (add in the whitened frame) keeps it cheap;
    the QNM band is in-band where whitening is ~flat.

    snr_match (v4 fix C'): rescale the 2-tone so its TOTAL signal energy equals
    a pure-220 of the same amp1 -> the two classes are SNR-matched, so the net
    must detect the overtone's spectral SPLITTING, not total loudness. Without
    this the net learns the shortcut 'high SNR => 2-tone' (diagnosed 2026-06-15)
    which does not transfer to real data."""
    i = min(max(np.searchsorted(CHI_GRID, chi), 0), len(CHI_GRID) - 1)
    f1, tau1 = W220[i][0] / mass, W220[i][1] * mass
    f2, tau2 = W221[i][0] / mass, W221[i][1] * mass
    t = np.arange(N_SAMP) / FS
    t0 = rng.uniform(0, T0_MAX_MS / 1000.0)
    a220 = rng.uniform(*PEAK_AMP_RANGE)
    x = np.empty((n_det, N_SAMP), dtype=np.float32)
    overtone_snr2 = 0.0
    for d in range(n_det):
        amp1 = a220 * rng.uniform(0.7, 1.3)
        s220 = rdlib.damped_sinusoids(t, t0, [dict(f=f1, tau=tau1, amp=amp1,
                                                   phi=rng.uniform(-np.pi, np.pi))])
        if n_tones == 2:
            s221 = rdlib.damped_sinusoids(t, t0, [dict(f=f2, tau=tau2, amp=amp1 * amp_frac,
                                                       phi=rng.uniform(-np.pi, np.pi))])
            sig = s220 + s221
            if snr_match:  # match total energy to the pure-220 -> remove the SNR cue
                e1, e2 = float(np.sum(s220 ** 2)), float(np.sum(sig ** 2))
                if e2 > 0:
                    scale = np.sqrt(e1 / e2)
                    sig, s221 = sig * scale, s221 * scale
            overtone_snr2 += float(np.sum(s221 ** 2))
        else:
            sig = s220
        nz = noise[d] if noise is not None else rng.standard_normal(N_SAMP)
        x[d] = sig + nz
    return x, float(np.sqrt(overtone_snr2))


class Embed(torch.nn.Module):
    def __init__(self, n_out=56):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Unflatten(1, (2, N_SAMP)),
            torch.nn.Conv1d(2, 16, 9, padding=4), torch.nn.ReLU(),
            torch.nn.Conv1d(16, 32, 9, padding=4, stride=2), torch.nn.ReLU(),
            torch.nn.Conv1d(32, 32, 9, padding=4, stride=2), torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.LazyLinear(128), torch.nn.ReLU(),
            torch.nn.Linear(128, n_out),
        )

    def forward(self, x):
        return self.net(x)
