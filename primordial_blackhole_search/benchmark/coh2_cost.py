"""Tuning step T-cost for PREREG_coherent2.md (frozen 715a0b3): G0 clause 3 (projected stage-2 cost <= 120 h).

Measures, on tuning data only:
  1. the coherent (Mc, q) template metric at length L (match maximised over time and phase, f_low = 50 Hz, a real
     O3a H1 PSD from tuning segment 0), from matches at small offsets;
  2. N_tmpl per candidate = (box area) x sqrt(det g) / ((3*sqrt(3)/2) * eps), eps = 1 - 0.97 (hexagonal covering),
     with the box from T1_mc_bias.json (99th percentile |dMc/Mc| per q stratum) and q over the population range;
  3. the per-template cost of a 2^21-sample complex correlation on MPS (median of repeats, warm);
  4. the projected stage-2 cost on a 24-segment set for r1 = 1/hour and 1/min, per L_max in {128, 256, 384}.

  python benchmark/coh2_cost.py          -> state/coh2/T_cost.json  (minutes, < 1.5 GB)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

B = Path(__file__).resolve().parent
sys.path.insert(0, str(B)); sys.path.insert(0, str(B.parent))
from pbh import config as C  # noqa: E402
from lowmem_psd import estimate_psd  # noqa: E402

FS, EPS = C.SAMPLE_RATE, 0.03
NSNIP = 2 ** 21
DF = FS / NSNIP
G_SUN = 4.925490947641267e-06  # G M_sun / c^3 in seconds
OUT = B / "state" / "coh2" / "T_cost.json"
L_OPTS = (128, 256, 384)
MC_PTS = (0.18, 0.26, 0.35, 0.50, 0.70, 0.86)


def f_at(mc, tau):
    """Leading-order GW frequency tau seconds before coalescence."""
    return (5.0 / (256.0 * tau)) ** 0.375 * (G_SUN * mc) ** -0.625 / np.pi


def masses(mc, q):
    m_tot = mc * (1 + q) ** 1.2 / q ** 0.6
    return m_tot / (1 + q), q * m_tot / (1 + q)


def q_range(mc):
    """Mass ratios with both components in [M_MIN, M_MAX] at this chirp mass."""
    qs = np.linspace(0.2, 1.0, 400)
    ok = [q for q in qs if C.M_MIN <= masses(mc, q)[1] and masses(mc, q)[0] <= C.M_MAX]
    return (min(ok), max(ok)) if ok else None


def fd_template(mc, q, L, freqs):
    from pycbc.waveform import get_fd_waveform
    m1, m2 = masses(mc, q)
    f0 = max(C.F_LOWER, f_at(mc, L))
    hp, _ = get_fd_waveform(approximant=C.APPROXIMANT, mass1=m1, mass2=m2, delta_f=DF, f_lower=f0,
                            f_final=C.F_HIGH, distance=1.0)
    h = np.zeros(len(freqs), complex)
    n = min(len(hp), len(h))
    h[:n] = hp.numpy()[:n]
    h[(freqs < f0) | (freqs > C.F_HIGH)] = 0
    return h


def match(a, b, inv_psd):
    w = inv_psd
    na, nb = np.sqrt(np.sum(np.abs(a) ** 2 * w)), np.sqrt(np.sum(np.abs(b) ** 2 * w))
    z = np.fft.ifft(np.conj(a) * b * w, n=2 * len(a))  # complex: max over time and phase
    return float(np.abs(z).max() * len(z) / (na * nb))


def metric(mc, q, L, freqs, inv_psd):
    """Quadratic fit of 1 - match in (ln Mc, q) from offsets sized to give mismatch ~1-3%."""
    h0 = fd_template(mc, q, L, freqs)
    def mm(dx, dy):
        return 1 - match(h0, fd_template(mc * np.exp(dx), q + dy, L, freqs), inv_psd)
    sx = 2e-5
    while mm(sx, 0) < 0.01 and sx < 0.05:
        sx *= 2
    d = 1.0 if q < 0.6 else -1.0
    sy = 2e-3 * d
    while mm(0, sy) < 0.01 and abs(sy) < 0.3:
        sy *= 2
    gxx = mm(sx, 0) / sx ** 2
    gyy = mm(0, sy) / sy ** 2
    gxy = (mm(sx, sy) / (sx * sy) - gxx * sx / sy - gyy * sy / sx) / 2
    det = max(gxx * gyy - gxy ** 2, 0.0)
    return dict(gxx=gxx, gyy=gyy, gxy=gxy, sqrt_det=float(np.sqrt(det)), sx=sx, sy=sy,
                mm_x=mm(sx, 0), mm_y=mm(0, sy))


def per_template_seconds(n_batch=64, reps=7):
    import torch
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    Y = torch.randn(NSNIP // 2 + 1, dtype=torch.complex64, device=dev)
    H = torch.randn(n_batch, NSNIP // 2 + 1, dtype=torch.complex64, device=dev)
    ts = []
    for _ in range(reps):
        if dev == "mps":
            torch.mps.synchronize()
        t = time.perf_counter()
        z = torch.fft.ifft(torch.nn.functional.pad(Y * H.conj(), (0, NSNIP // 2 - 1)), dim=1)
        top = z.abs().amax(dim=1)
        if dev == "mps":
            torch.mps.synchronize()
        ts.append((time.perf_counter() - t) / n_batch)
        del z, top
    return float(np.median(ts[2:])), dev


def candidate_mcs(per_day):
    import evaluate as E
    import y1_bank as Y1
    sub = B / "submissions" / "y1"
    bg, T, _ = E.background(sub)
    thr = float(bg[int(np.floor(per_day * T / 86400))])
    segs = json.loads((B / "manifest.json").read_text())["test"]
    mcs = []
    for k in range(E.N_SLIDES + 1):
        tr = E.read_triggers(sub / f"bg_k{k}.csv")
        for t, s in tr[tr[:, 1] > thr][:, :2]:
            i = next(j for j, g in enumerate(segs) if g <= t < g + C.SEGMENT_LEN)
            z = np.load(B / "results" / "y1_bins" / f"noise_{segs[i]}.npz")
            b = min(int((t - segs[i]) * FS) // Y1.BIN, len(z["H1_tid"]) - 1)
            mcs.append(float(Y1.MCS[z["H1_tid"][b]]))
    return np.array(mcs), T, thr


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    bias = json.loads((B / "state" / "coh2" / "T1_mc_bias.json").read_text())
    segs = json.loads((B / "manifest.json").read_text())["test"]
    x = np.load(B / "data" / f"H1_{segs[0]}.npy").astype(np.float64)
    psd_full = estimate_psd(x); del x
    f_full = np.fft.rfftfreq(C.SEGMENT_LEN * FS, d=1.0 / FS)
    freqs = np.arange(NSNIP // 2 + 1) * DF
    psd = np.interp(freqs, f_full, psd_full)
    inv = np.where((freqs >= C.F_LOWER) & (freqs <= C.F_HIGH), 1.0 / psd, 0.0)
    res = {"prereg": "PREREG_coherent2.md @ 715a0b3", "eps": EPS, "bias": bias, "metric": {}, "n_tmpl": {}}
    t0 = time.time()
    for L in L_OPTS:
        for mc in MC_PTS:
            qr = q_range(mc)
            if qr is None:
                continue
            qs = np.linspace(qr[0], qr[1], 4)
            rows = [dict(q=float(q), **metric(mc, float(q), L, freqs, inv)) for q in qs]
            sd = np.array([r["sqrt_det"] for r in rows])
            integ = float(np.trapezoid(sd, qs)) if len(qs) > 1 else 0.0
            n = {}
            for lab in ("q<0.5", "q>=0.5"):
                w = bias[lab]["p99"]
                n[lab] = 2 * w * integ / (1.5 * np.sqrt(3) * EPS)
            res["metric"][f"L{L}_mc{mc}"] = rows
            res["n_tmpl"][f"L{L}_mc{mc}"] = {"q_range": qr, "sqrt_det_integral": integ,
                                              "n_box_p99_qlt05": n["q<0.5"], "n_box_p99_qge05": n["q>=0.5"],
                                              "n_tmpl": max(n.values())}
            print(f"L={L} mc={mc}: N_tmpl ~ {max(n.values()):.0f}  ({time.time()-t0:.0f}s)", flush=True)
    tpl_s, dev = per_template_seconds()
    res["per_template_s"], res["device"] = tpl_s, dev
    print(f"per-template correlation {tpl_s*1e3:.2f} ms on {dev}", flush=True)
    res["projection"] = {}
    for lab, per_day in (("1/hour", 24.0), ("1/min", 1440.0)):
        mcs, T, thr = candidate_mcs(per_day)
        for L in L_OPTS:
            grid = sorted((mc, res["n_tmpl"][f"L{L}_mc{mc}"]["n_tmpl"]) for mc in MC_PTS
                          if f"L{L}_mc{mc}" in res["n_tmpl"])
            gm, gn = np.array([g[0] for g in grid]), np.array([g[1] for g in grid])
            n_each = np.exp(np.interp(np.log(mcs), np.log(gm), np.log(np.maximum(gn, 1))))
            hours_bg = float(2 * n_each.sum() * tpl_s / 3600)
            # PREREG S2 = real background + Gaussian twin (same candidates) + 600 decoys + ~300 nulls/fg looks
            total = hours_bg * (2 + 900 / len(mcs))
            res["projection"][f"{lab}_L{L}"] = {"n_candidates": int(len(mcs)), "stage1_thr": thr,
                                                "median_n_tmpl": float(np.median(n_each)),
                                                "bg_hours": hours_bg, "s2_total_hours": total}
            print(f"{lab} L_max={L}: {len(mcs)} cands, median N_tmpl {np.median(n_each):.0f}, bg {hours_bg:.1f} h, "
                  f"S2 total {total:.1f} h", flush=True)
    res["seconds"] = round(time.time() - t0)
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
