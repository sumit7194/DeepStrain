"""Two-stage cascade against REAL background on LongChirp. Pre-registration: PREREG_realbg_cascade.md (0a849d7 +
amendment A1 9fa8b27), committed before this code ran.

  python benchmark/realbg_cascade.py templates            # child: the 64 stage-2 templates -> state/realbg/templates.npz
  python benchmark/realbg_cascade.py series <det> <gps> <real|twin>   # binned |z|^2 for 64 templates on one series
  python benchmark/realbg_cascade.py probe <gps>          # cost probe: templates + H1 real + H1 twin of one segment

Coherent statistic (per detector, whitened domain, benchmark lowmem whitening): for a template h cut at t_c,
  c(t)  = sum_n x_w[n+t] h_w[n]  = irfft(X conj(H) w^2),   c_q(t) = same with the quadrature (Hilbert) template,
  |z|^2 = (c^2 + c_q^2) / sum h_w^2,   merger index = t + len(h) - 1.
Binned by merger time in 102-sample (~25 ms) bins, max per bin (amendment A1).
"""
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

B = Path(__file__).resolve().parent
sys.path.insert(0, str(B.parent)); sys.path.insert(0, str(B))
from pbh import config as C  # noqa: E402
from lowmem_psd import estimate_psd  # noqa: E402

FS = C.SAMPLE_RATE
N = C.SEGMENT_LEN * FS                      # 2^24
BIN = 102                                    # ~25 ms
NB = N // BIN
TPL_SEC = 384                                # amendment A1
EDGE_LO, EDGE_HI = (TPL_SEC + 8) * FS, N - 8 * FS
N_G1 = 300                                   # sample-level draws per (template, series) for gate G1
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
STATE = B / "state" / "realbg"


def peak_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 2 ** 20


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg} | peak RSS {peak_mb()} MB", flush=True)


def stage2_template_params():
    """64 templates: 32 equal-mass + 32 at mass ratio 0.3, chirp masses log-spaced over the Y1 bank range."""
    import y1_bank as Y1
    out = []
    # q = 0.3 stays inside the population (both components in [0.2, 1.0]) only for Mc in [0.307, 0.461]
    for q, (lo, hi) in ((1.0, (Y1.MCS[0], Y1.MCS[-1])), (0.3, (0.307, 0.461))):
        for mc in np.geomspace(lo, hi, 32):
            m_tot = mc * (1 + q) ** 1.2 / q ** 0.6          # Mc = (m1 m2)^(3/5) / M^(1/5), m2 = q m1
            m1 = m_tot / (1 + q)
            out.append((float(m1), float(q * m1)))
    return out


def cut_template(h_full, t_c_index):
    """Keep samples up to and including t_c, then the last TPL_SEC seconds."""
    h = h_full[: t_c_index + 1]
    return h[-TPL_SEC * FS:].astype(np.float32)


def make_templates():
    """Child process (pycbc keeps waveform memory): projected H1 templates at T_REF, cut at t_c, last 384 s."""
    from dataclasses import replace
    from pycbc.detector import Detector
    from pycbc.waveform import get_td_waveform
    import y1_bank as Y1
    base = Y1.BASE
    arrs = {}
    for j, (m1, m2) in enumerate(stage2_template_params()):
        p = replace(base, mass1=m1, mass2=m2)
        hp, hc = get_td_waveform(approximant=C.APPROXIMANT, mass1=m1, mass2=m2, delta_t=1.0 / FS,
                                 f_lower=C.F_LOWER, distance=1.0, inclination=p.inclination, coa_phase=p.coa_phase)
        fp, fx = Detector("H1").antenna_pattern(p.ra, p.dec, p.psi, Y1.T_REF)
        h = fp * hp.numpy() + fx * hc.numpy()
        tc = int(np.searchsorted(hp.sample_times.numpy(), 0.5 / FS)) - 1
        arrs[f"t{j}"] = cut_template(h, tc)
    STATE.mkdir(parents=True, exist_ok=True)
    tmp = STATE / "templates.tmp.npz"
    np.savez(tmp, **arrs); os.replace(tmp, STATE / "templates.npz")
    log("templates saved")


def load_templates():
    p = STATE / "templates.npz"
    if not p.exists():
        subprocess.run([sys.executable, "-u", __file__, "templates"], check=True)
    z = np.load(p)
    return [z[f"t{j}"] for j in range(64)]


def whiten_filter(psd):
    s = np.asarray(psd, dtype=np.float64).copy()
    s[s <= 0] = np.inf
    w = np.sqrt(2.0 / (FS * s))
    f = np.fft.rfftfreq(N, d=1.0 / FS)
    w[(f < C.F_LOWER - 10) | (f > C.F_HIGH)] = 0.0
    return w


@torch.no_grad()
def series_spectrum(det, gps, mode, seed):
    """X * w^2 on the GPU (complex64) for real data or its phase-randomised Gaussian twin."""
    x = np.load(B / "data" / f"{det}_{gps}.npy").astype(np.float64)
    w = whiten_filter(estimate_psd(x))
    X = np.fft.rfft(x)
    del x
    if mode == "twin":
        rng = np.random.default_rng(seed)
        ph = np.exp(2j * np.pi * rng.random(len(X)))
        ph[0] = ph[-1] = 1.0
        X = np.abs(X) * ph
    Y = torch.tensor((X * w * w).astype(np.complex64), device=DEV)
    w_t = torch.tensor(w.astype(np.float32), device=DEV)
    return Y, w_t


@torch.no_grad()
def z2_series(Y, w_t, h):
    """|z|^2 indexed by MERGER sample, for one template (float32 on the GPU)."""
    H = torch.fft.rfft(torch.tensor(h, device=DEV), n=N)
    Hw = H * w_t
    sig2 = (2 * (Hw.abs() ** 2).sum() - (Hw[0].abs() ** 2) - (Hw[-1].abs() ** 2)) / N     # Parseval: sum h_w^2
    P = Y * H.conj()
    c = torch.fft.irfft(P, n=N)
    Pq = P * 1j
    Pq[0] = 0; Pq[-1] = 0
    cq = torch.fft.irfft(Pq, n=N)
    z2 = (c * c + cq * cq) / sig2
    return torch.roll(z2, len(h) - 1)          # c[t] has the template starting at t; merger at t + L - 1


@torch.no_grad()
def process_series(det, gps, mode, templates):
    out = STATE / f"bins_{mode}_{det}_{gps}.npz"
    if out.exists():
        return
    seed = int(gps) * 10 + (0 if det == "H1" else 1)
    Y, w_t = series_spectrum(det, gps, mode, seed)
    g1_idx = np.random.default_rng(1234).integers(EDGE_LO, EDGE_HI, N_G1)          # same indices for all series
    bins = np.empty((len(templates), NB), dtype=np.float16)
    g1 = np.empty((len(templates), N_G1), dtype=np.float32)
    for j, h in enumerate(templates):
        z2 = z2_series(Y, w_t, h)
        g1[j] = z2[torch.tensor(g1_idx, device=DEV)].cpu().numpy()
        bins[j] = z2[: NB * BIN].view(NB, BIN).amax(dim=1).cpu().numpy().astype(np.float16)
    STATE.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, bins=bins, g1=g1, g1_idx=g1_idx); os.replace(tmp, out)


def probe(gps):
    t0 = time.time()
    tpl = load_templates()
    t1 = time.time()
    log(f"templates ready ({t1 - t0:.0f} s; lengths {min(len(h) for h in tpl)}..{max(len(h) for h in tpl)})")
    times = {}
    for mode in ("real", "twin"):
        a = time.time()
        process_series("H1", gps, mode, tpl)
        times[mode] = time.time() - a
        log(f"H1 {gps} {mode}: {times[mode]:.0f} s")
    z = np.load(STATE / f"bins_twin_H1_{gps}.npz")
    g1 = z["g1"].ravel()
    res = dict(segment=gps, template_s=t1 - t0, series_s=times, peak_rss_mb=peak_mb(),
               mps_driver_mb=torch.mps.driver_allocated_memory() // 2 ** 20 if DEV == "mps" else 0,
               twin_single_det_mean_z2=float(g1.mean()),
               note="twin |z|^2 per detector should average ~2 (chi^2 with 2 dof); this is a sanity check, not gate G1")
    (B / "results" / "realbg_probe.json").write_text(json.dumps(res, indent=1))
    log(json.dumps(res))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "templates":
        make_templates()
    elif cmd == "series":
        process_series(sys.argv[2], int(sys.argv[3]), sys.argv[4], load_templates())
    elif cmd == "probe":
        probe(int(sys.argv[2]))
