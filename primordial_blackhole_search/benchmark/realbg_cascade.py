"""Two-stage cascade against REAL background on LongChirp. Pre-registration: PREREG_realbg_cascade.md
(0a849d7; amendments A1 9fa8b27, A2 adopted 1dc5fb1, with gate parameters fixed in 9733bf8). This code was written after
the design was frozen.

Steps (each heavy one runs in its own short-lived process, driven by realbg_drive.py):
  gates <det> <gps>                 PyCBC detect_loud_glitches with its documented defaults -> state/realbg/gates_*.json
  series <det> <gps> <mode>         mode in {gated, ungated, twin}: binned |z|^2 for the 64 stage-2 templates
  caught                            Y1 stage 1 at one per minute: candidates per (slide, segment) + caught injections
  fg <gps>                          M3: realised true-template network SNR of caught injections (gated / ungated / null)
  g3                                linearity check against really injected, gated strain (5 injections)
  combine                           survival functions, thresholds, distances, gates G1-G4, verdict
  probe <gps>                       the cost probe that ran before A2 (kept for the record)

Coherent statistic per detector (whitened domain, lowmem whitening): c(t) = irfft(X conj(H) w^2), its quadrature
c_q = irfft(i X conj(H) w^2), |z|^2 = (c^2 + c_q^2) / sum h_w^2, indexed by MERGER sample (t + len(h) - 1).
Binned at 102 samples (~25 ms), max per bin. Network per bin: H1 bin + max of L1 bins b-1..b+1 (covers |delta| <= 10 ms).
"""
import csv
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

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
WIN_BINS = 40                                # +-1 s around a candidate
N_G1 = 300
STAGE1_PER_DAY = 1440.0                      # one candidate per minute (primary)
N_EFF = (1e4, 1e6, 1e8)
TARGETS = {"1/day": 365.25, "1/week": 52.18}
Y1_ALONE = {"1/day": 57.47, "1/week": 51.68}
PROBE_SEG = 1249001472
STATE = B / "state" / "realbg"
EDGES = np.concatenate([np.arange(0.0, 200.0, 0.05), np.geomspace(200.0, 1e9, 3000)])
MAN = json.loads((B / "manifest.json").read_text())
SEGS = MAN["test"]


def peak_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 2 ** 20


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg} | peak RSS {peak_mb()} MB", flush=True)


def atomic_json(path, obj):
    tmp = path.with_suffix(".tmp"); tmp.write_text(json.dumps(obj)); os.replace(tmp, path)


# ------------------------------------------------------------------ templates (unchanged since the probe)
def stage2_template_params():
    import y1_bank as Y1
    out = []
    # q = 0.3 stays inside the population (both components in [0.2, 1.0]) only for Mc in [0.307, 0.461]
    for q, (lo, hi) in ((1.0, (Y1.MCS[0], Y1.MCS[-1])), (0.3, (0.307, 0.461))):
        for mc in np.geomspace(lo, hi, 32):
            m_tot = mc * (1 + q) ** 1.2 / q ** 0.6
            m1 = m_tot / (1 + q)
            out.append((float(m1), float(q * m1)))
    return out


def cut_template(h_full, t_c_index):
    return h_full[: t_c_index + 1][-TPL_SEC * FS:].astype(np.float32)


def make_templates():
    from dataclasses import replace
    from pycbc.detector import Detector
    from pycbc.waveform import get_td_waveform
    import y1_bank as Y1
    arrs = {}
    for j, (m1, m2) in enumerate(stage2_template_params()):
        p = replace(Y1.BASE, mass1=m1, mass2=m2)
        hp, hc = get_td_waveform(approximant=C.APPROXIMANT, mass1=m1, mass2=m2, delta_t=1.0 / FS,
                                 f_lower=C.F_LOWER, distance=1.0, inclination=p.inclination, coa_phase=p.coa_phase)
        fp, fx = Detector("H1").antenna_pattern(p.ra, p.dec, p.psi, Y1.T_REF)
        tc = int(np.searchsorted(hp.sample_times.numpy(), 0.5 / FS)) - 1
        arrs[f"t{j}"] = cut_template(fp * hp.numpy() + fx * hc.numpy(), tc)
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


# ------------------------------------------------------------------ gating (PyCBC 2.10.0 defaults, A2)
def detect_gates(det, gps):
    from pycbc.strain.strain import detect_loud_glitches
    from pycbc.types import TimeSeries
    out = STATE / f"gates_{det}_{gps}.json"
    if out.exists():
        return
    x = np.load(B / "data" / f"{det}_{gps}.npy").astype(np.float64)
    times = detect_loud_glitches(TimeSeries(x, delta_t=1.0 / FS, epoch=gps))
    STATE.mkdir(parents=True, exist_ok=True)
    atomic_json(out, [float(t) for t in times])
    log(f"gates {det} {gps}: {len(times)}")


def gate_window(det, gps):
    """Multiplicative window equal to PyCBC's strain.gate(t, window=0.25, method='taper', taper_width=0.25)."""
    from pycbc.types import TimeSeries
    times = json.loads((STATE / f"gates_{det}_{gps}.json").read_text())
    g = TimeSeries(np.ones(N), delta_t=1.0 / FS, epoch=gps)
    for t in times:
        g = g.gate(t, window=0.25, method="taper", copy=False, taper_width=0.25)
    return g.numpy().copy(), times


def whiten_filter(psd):
    s = np.asarray(psd, dtype=np.float64).copy()
    s[s <= 0] = np.inf
    w = np.sqrt(2.0 / (FS * s))
    f = np.fft.rfftfreq(N, d=1.0 / FS)
    w[(f < C.F_LOWER - 10) | (f > C.F_HIGH)] = 0.0
    return w


# ------------------------------------------------------------------ M1 / M2: binned |z|^2 per series
def series_spectrum(det, gps, mode, seed):
    import torch
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    x = np.load(B / "data" / f"{det}_{gps}.npy").astype(np.float64)
    if mode in ("gated", "twin"):
        g, _ = gate_window(det, gps)
        x *= g
    psd = estimate_psd(x)
    w = whiten_filter(psd)
    if mode == "twin":          # Gaussian noise drawn from the (gated, median-Welch) PSD: unit-variance white in band
        rng = np.random.default_rng(seed)
        sig = np.sqrt(N * FS * psd / 4.0)
        X = sig * (rng.standard_normal(len(psd)) + 1j * rng.standard_normal(len(psd)))
        X[0] = X[0].real; X[-1] = X[-1].real
    else:
        X = np.fft.rfft(x)
    del x
    Y = torch.tensor((X * w * w).astype(np.complex64), device=dev)
    return Y, torch.tensor(w.astype(np.float32), device=dev), dev


def z2_series(Y, w_t, h, dev):
    import torch
    with torch.no_grad():
        H = torch.fft.rfft(torch.tensor(h, device=dev), n=N)
        Hw = H * w_t
        sig2 = (2 * (Hw.abs() ** 2).sum() - (Hw[0].abs() ** 2) - (Hw[-1].abs() ** 2)) / N
        P = Y * H.conj()
        c = torch.fft.irfft(P, n=N)
        Pq = P * 1j
        Pq[0] = 0; Pq[-1] = 0
        cq = torch.fft.irfft(Pq, n=N)
        return torch.roll((c * c + cq * cq) / sig2, len(h) - 1)


def process_series(det, gps, mode, templates):
    import torch
    out = STATE / f"bins_{mode}_{det}_{gps}.npz"
    if out.exists():
        return
    seed = int(gps) * 10 + (0 if det == "H1" else 1)
    Y, w_t, dev = series_spectrum(det, gps, mode, seed)
    g1_idx = np.random.default_rng(1234).integers(EDGE_LO, EDGE_HI, N_G1)
    idx_t = torch.tensor(g1_idx, device=dev)
    bins = np.empty((len(templates), NB), dtype=np.float16)
    g1 = np.empty((len(templates), N_G1), dtype=np.float32)
    with torch.no_grad():
        for j, h in enumerate(templates):
            z2 = z2_series(Y, w_t, h, dev)
            g1[j] = z2[idx_t].cpu().numpy()
            bins[j] = np.minimum(z2[: NB * BIN].view(NB, BIN).amax(dim=1).cpu().numpy(), 6e4).astype(np.float16)
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, bins=bins, g1=g1); os.replace(tmp, out)
    log(f"series {mode} {det} {gps} done")


# ------------------------------------------------------------------ stage 1: Y1 candidates and caught injections
def caught():
    import evaluate as E
    sub = B / "submissions" / "y1"
    bg, T, _ = E.background(sub)
    n1 = int(np.floor(STAGE1_PER_DAY * T / 86400))
    thr1 = float(bg[n1])
    cands = {}
    for k in range(E.N_SLIDES + 1):
        tr = E.read_triggers(sub / f"bg_k{k}.csv")
        for t, s in tr[tr[:, 1] > thr1][:, :2]:
            i = next(j for j, g in enumerate(SEGS) if g <= t < g + C.SEGMENT_LEN)
            cands.setdefault(f"{k}_{i}", []).append(int(np.floor((t - SEGS[i]) * FS / BIN)))
    inj = E.injection_table()
    loudest, _ = E.injection_loudest(sub, inj)
    y1thr = {}
    for far, per_year in TARGETS.items():
        k = int(np.floor(per_year * T / E.YEAR_S))
        y1thr[far] = float(bg[k])
    rows = [dict(id=int(x["id"]), segment=int(x["segment"]), t_h1=x["t_h1"], chirp_distance=x["chirp_distance"],
                 caught=bool(loudest[j] > thr1), y1_loudest=float(loudest[j])) for j, x in enumerate(inj)]
    atomic_json(STATE / "caught.json", dict(thr1=thr1, n1=n1, T=T, y1_thresholds=y1thr, candidates=cands, rows=rows))
    log(f"stage 1: thr1 {thr1:.3f}, {n1} candidates, {sum(r['caught'] for r in rows)} caught")


# ------------------------------------------------------------------ M3: realised foreground SNR (same trial definition)
def _injection_template(row, det, gps):
    import injections as I
    from pycbc.detector import Detector
    s, s0 = I.detector_signal(row, det)
    t_c = row["t_geo"] + Detector(det).time_delay_from_earth_center(row["ra"], row["dec"], row["t_geo"])
    ic = int(round((t_c - s0) * FS))
    h = cut_template(s, ic)
    merger = int(round((t_c - gps) * FS))
    return s, s0, h, merger


def _binned_at(z2, bH, nb):
    """Max of |z|^2 over bins bH-nb..bH+nb (each bin = 102 samples), from a merger-indexed GPU series."""
    lo, hi = (bH - nb) * BIN, (bH + nb + 1) * BIN
    return z2[lo:hi].view(2 * nb + 1, BIN).amax(dim=1).cpu().numpy()


def fg_segment(gps):
    """Per caught injection: network value in the SAME trial form as the background (H1 bin at the merger, L1 max
    over bins b-1..b+1), for signal+noise ('z') and noise only ('null'); gated (primary) and ungated."""
    import torch
    import injections as I
    out = STATE / f"fg_{gps}.json"
    if out.exists():
        return
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    st = json.loads((STATE / "caught.json").read_text())
    with open(I.TABLE) as f:
        table = {int(r["id"]): {k: (int(v) if k in ("id", "realization", "segment") else float(v)) for k, v in r.items()}
                 for r in csv.DictReader(f)}
    rows = [r for r in st["rows"] if r["caught"] and r["segment"] == gps]
    res = {r["id"]: {} for r in rows}
    bH = {r["id"]: None for r in rows}
    for det in ("H1", "L1"):
        x = np.load(B / "data" / f"{det}_{gps}.npy").astype(np.float64)
        g, _ = gate_window(det, gps)
        for name, gw in (("gated", g), ("ungated", None)):
            xx = x * gw if gw is not None else x
            w = whiten_filter(estimate_psd(xx))
            Yn = torch.tensor((np.fft.rfft(xx) * w * w).astype(np.complex64), device=dev)
            w_t = torch.tensor(w.astype(np.float32), device=dev)
            del xx
            for r in rows:
                row = table[r["id"]]
                s, s0, h, merger = _injection_template(row, det, gps)
                if det == "H1" and bH[r["id"]] is None:
                    bH[r["id"]] = merger // BIN
                b = bH[r["id"]]
                sfull = I.add_signal(np.zeros(N), gps, s, s0)
                if gw is not None:
                    sfull *= gw
                S = torch.tensor((np.fft.rfft(sfull) * w * w).astype(np.complex64), device=dev)
                nb = 0 if det == "H1" else 1
                zt = _binned_at(z2_series(Yn + S, w_t, h, dev), b, nb).max()
                zn = _binned_at(z2_series(Yn, w_t, h, dev), b, nb).max()
                hw = np.fft.irfft(np.fft.rfft(h.astype(np.float64), n=N) * w, n=N)
                res[r["id"]][f"{det}_{name}"] = dict(z2=float(zt), null2=float(zn), opt=float(np.sqrt(np.dot(hw, hw))))
        del x
    atomic_json(out, res)
    log(f"fg {gps}: {len(rows)} caught injections")


def g3(only_i=None, only_det=None):
    """Linearity check: really inject, detect gates on the INJECTED strain, gate, whiten, filter; compare the binned
    gated network value per detector with the shortcut (noise spectrum + gated signal spectrum).
    Fix F2: called with (i, det) it does ONE injection-detector pair in this process and writes g3_<i>_<det>.json, so
    pycbc's glitch detector (~3.1 GB peak) and its retained waveform memory never stack (the all-in-one version
    reached 3.77 GB and was stopped by the pressure guard on 2026-11-20)."""
    import torch
    import injections as I
    from pycbc.strain.strain import detect_loud_glitches
    from pycbc.types import TimeSeries
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    st = json.loads((STATE / "caught.json").read_text())
    with open(I.TABLE) as f:
        table = {int(r["id"]): {k: (int(v) if k in ("id", "realization", "segment") else float(v)) for k, v in r.items()}
                 for r in csv.DictReader(f)}
    picks = [r for r in st["rows"] if r["caught"]][:5]
    out = []
    for i, r in enumerate(picks):
        if only_i is not None and i != only_i:
            continue
        row = table[r["id"]]; gps = r["segment"]
        fg = json.loads((STATE / f"fg_{gps}.json").read_text())[str(r["id"])]
        bH = None
        for det in ("H1", "L1"):
            x = np.load(B / "data" / f"{det}_{gps}.npy").astype(np.float64)
            s, s0, h, merger = _injection_template(row, det, gps)
            if det == "H1":
                bH = merger // BIN
            I.add_signal(x, gps, s, s0)
            ts = TimeSeries(x, delta_t=1.0 / FS, epoch=gps)
            for t in detect_loud_glitches(ts):
                ts = ts.gate(t, window=0.25, method="taper", copy=False, taper_width=0.25)
            xg = ts.numpy().copy()
            w = whiten_filter(estimate_psd(xg))
            Y = torch.tensor((np.fft.rfft(xg) * w * w).astype(np.complex64), device=dev)
            w_t = torch.tensor(w.astype(np.float32), device=dev)
            direct = float(_binned_at(z2_series(Y, w_t, h, dev), bH, 0 if det == "H1" else 1).max())
            if only_det is not None and det != only_det:
                continue
            short = fg[f"{det}_gated"]["z2"]
            out.append(dict(id=r["id"], det=det, direct=direct, shortcut=short,
                            rel=abs(np.sqrt(short) / np.sqrt(direct) - 1)))
            log(f"g3 {r['id']} {det}: direct {direct:.2f} shortcut {short:.2f}")
    if only_i is not None:
        atomic_json(STATE / f"g3_{only_i}_{only_det}.json", out)
        return
    atomic_json(STATE / "g3.json", out)


def g3_merge():
    parts = [json.loads((STATE / f"g3_{i}_{d}.json").read_text()) for i in range(5) for d in ("H1", "L1")]
    atomic_json(STATE / "g3.json", [x for p in parts for x in p])


# ------------------------------------------------------------------ combine
def chi2_sf(x, dof):
    from scipy.stats import chi2
    return float(chi2.sf(x, dof))


def _load_bins(mode):
    return {(det, g): np.load(STATE / f"bins_{mode}_{det}_{g}.npz") for det in ("H1", "L1") for g in SEGS}


def _lmax3(L):
    out = L.copy()
    out[:, 1:] = np.maximum(out[:, 1:], L[:, :-1])
    out[:, :-1] = np.maximum(out[:, :-1], L[:, 1:])
    return out


def survival(mode, cands, exclude=None):
    """Conditional (candidate windows) and unconditional histograms of the binned network statistic."""
    data = _load_bins(mode)
    valid = np.zeros(NB, bool)
    valid[int(np.ceil(EDGE_LO / BIN)):EDGE_HI // BIN] = True
    cond = np.zeros(len(EDGES) - 1, np.int64); unc = np.zeros_like(cond)
    ks_sample = []
    rng = np.random.default_rng(7)
    n_excl = 0
    for key, bl in cands.items():
        k, i = map(int, key.split("_"))
        gh, gl = SEGS[i], SEGS[(i + k) % 24]
        if exclude is not None and exclude in (gh, gl):
            continue
        Hb = data[("H1", gh)]["bins"].astype(np.float32)
        Lb = _lmax3(data[("L1", gl)]["bins"].astype(np.float32))
        net = Hb + Lb
        unc += np.histogram(net[:, valid].ravel(), EDGES)[0]
        for b in bl:
            lo, hi = b - WIN_BINS, b + WIN_BINS + 1
            sel = np.arange(lo, hi)
            sel = sel[(sel >= 0) & (sel < NB)]
            sel = sel[valid[sel]]
            n_excl += (2 * WIN_BINS + 1) - len(sel)
            v = net[:, sel].ravel()
            cond += np.histogram(v, EDGES)[0]
            if len(v):
                ks_sample.append(rng.choice(v, size=min(20, len(v)), replace=False))
    return cond, unc, (np.concatenate(ks_sample) if ks_sample else np.zeros(0)), n_excl


def sf_from_hist(counts):
    tot = counts.sum()
    above = tot - np.cumsum(counts)          # count strictly above each upper edge
    return EDGES[1:], above / tot, above


def threshold_at(p, counts):
    """x such that S(x) = p: empirical if >= 100 exceedances exist at that depth, else a log-linear fit on the top 0.1%."""
    x, S, above = sf_from_hist(counts)
    ok = above >= 100
    if ok.any() and S[ok].min() <= p:
        j = np.argmax(S <= p)
        return float(x[j]), "empirical", None
    top = (S <= 1e-3) & (above >= 20)
    if top.sum() < 3:
        return float("nan"), "unfit", None
    slope, icpt = np.polyfit(x[top], np.log(S[top]), 1)
    return float((np.log(p) - icpt) / slope), "extrapolated", float(slope)


def distance(found, dc):
    import injections as I
    V = (I.DC_HI - I.DC_LO) * np.mean(4 * np.pi * dc ** 2 * found)
    return float((3 * V / (4 * np.pi)) ** (1 / 3))


def evaluate_variant(mode, st, fg, exclude=None):
    cond, unc, ks, n_excl = survival(mode, st["candidates"], exclude)
    rows = [r for r in st["rows"] if exclude is None or r["segment"] != exclude]
    dc = np.array([r["chirp_distance"] for r in rows])
    fgkey = "ungated" if mode == "ungated" else "gated"
    rho2 = np.full(len(rows), -np.inf); null2 = np.full(len(rows), -np.inf); opt = np.zeros(len(rows))
    for j, r in enumerate(rows):
        if r["caught"]:
            e = fg[str(r["id"])]
            rho2[j] = e[f"H1_{fgkey}"]["z2"] + e[f"L1_{fgkey}"]["z2"]
            null2[j] = e[f"H1_{fgkey}"]["null2"] + e[f"L1_{fgkey}"]["null2"]
            opt[j] = np.hypot(e["H1_ungated"]["opt"], e["L1_ungated"]["opt"])
    caught = np.array([r["caught"] for r in rows])
    y1found = {far: np.array([r["y1_loudest"] > st["y1_thresholds"][far] for r in rows]) for far in TARGETS}
    rate1 = st["n1"] / st["T"]
    out = {"n_excluded_window_bins": int(n_excl), "trials_conditional": int(cond.sum()),
           "trials_unconditional": int(unc.sum()), "by_n_eff": {}}
    for n_eff in N_EFF:
        row = {}
        for far, per_year in TARGETS.items():
            p = (per_year / (365.25 * 86400)) / (rate1 * n_eff)
            x, how, slope = threshold_at(p, cond)
            found = caught & (rho2 >= x)
            d, d1 = distance(found, dc), distance(y1found[far], dc)
            row[far] = dict(p=p, threshold_rho2=x, how=how, fit_slope=slope, distance_mpc=d,
                            y1_alone_mpc=d1, ratio=d / d1 if d1 > 0 else float("nan"),
                            null_pass_fraction=float(np.mean(null2[caught] >= x)),
                            loud_pass_fraction=float(np.mean((rho2 >= x)[caught & (opt >= 15)]))
                            if (caught & (opt >= 15)).any() else None)
        out["by_n_eff"][f"{n_eff:.0e}"] = row
    from scipy.stats import ks_2samp
    out["G2_ks_p"] = float(ks_2samp(null2[caught], ks).pvalue) if len(ks) else None
    xs, S, above = sf_from_hist(cond)
    xu, Su, _ = sf_from_hist(unc)
    out["tail_points"] = {f"S={q:g}": {"cond_x": float(xs[np.argmax(S <= q)]), "uncond_x": float(xu[np.argmax(Su <= q)])}
                          for q in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)}
    return out, cond


def combine():
    st = json.loads((STATE / "caught.json").read_text())
    fg = {}
    for g in SEGS:
        p = STATE / f"fg_{g}.json"
        if p.exists():
            fg.update(json.loads(p.read_text()))
    res = {"prereg": "PREREG_realbg_cascade.md (0a849d7, A1 9fa8b27, A2 1dc5fb1/9733bf8)", "thr1": st["thr1"],
           "n_candidates": st["n1"], "caught": int(sum(r["caught"] for r in st["rows"]))}
    # G1: twin sample-level calibration (chi^2_4 at fixed delta = 0)
    tw = _load_bins("twin")
    samp = np.concatenate([(tw[("H1", SEGS[i])]["g1"] + tw[("L1", SEGS[(i + k) % 24])]["g1"]).ravel()
                           for k in range(24) for i in range(24)])
    g1 = {}
    for q in (1e-3, 1e-5):
        from scipy.stats import chi2
        x = float(chi2.isf(q, 4)); emp = float(np.mean(samp > x))
        g1[f"{q:g}"] = dict(x=x, empirical=emp, ratio=emp / q, n=int(len(samp)))
    res["G1"] = g1
    res["G1_pass"] = bool(abs(g1["0.001"]["ratio"] - 1) <= 0.2 and 0.5 <= g1["1e-05"]["ratio"] <= 2)
    del tw
    g3p = STATE / "g3.json"
    g3 = json.loads(g3p.read_text()) if g3p.exists() else []
    res["G3"] = g3
    res["G3_pass"] = bool(g3) and all(x["rel"] <= 0.02 for x in g3)
    for label, excl in (("all_segments", None), ("without_probe_segment", PROBE_SEG)):
        res[label] = {}
        for mode in ("gated", "ungated", "twin"):
            if mode == "twin":
                v, _ = evaluate_variant("twin", st, fg, excl)
                res[label]["twin_tail_points"] = v["tail_points"]
                res[label]["twin_threshold_cells"] = {k: {f: (c["threshold_rho2"], c["how"], c["fit_slope"])
                                                          for f, c in r.items()} for k, r in v["by_n_eff"].items()}
                continue
            v, _ = evaluate_variant(mode, st, fg, excl)
            res[label][mode] = v
        prim = res[label]["gated"]["by_n_eff"]
        surv = all(prim["1e+06"][f]["ratio"] >= 1.10 for f in TARGETS)
        shr = all(prim["1e+04"][f]["ratio"] >= 1.10 for f in TARGETS)
        g2 = prim["1e+06"]["1/day"]["null_pass_fraction"] <= 0.01 and (res[label]["gated"]["G2_ks_p"] or 0) > 0.01
        lp = prim["1e+06"]["1/day"]["loud_pass_fraction"]
        g4 = lp is not None and lp >= 0.90
        gates_ok = res["G1_pass"] and res["G3_pass"] and g2 and g4
        res[label]["gates"] = dict(G1=res["G1_pass"], G2=bool(g2), G3=res["G3_pass"], G4=bool(g4))
        res[label]["verdict"] = ("NO VERDICT (gate failed)" if not gates_ok else
                                 "SURVIVES" if surv else "SHRINKS" if shr else "DIES")
    (B / "results" / "realbg_cascade.json").write_text(json.dumps(res, indent=1))
    log(f"VERDICT all={res['all_segments']['verdict']} | without probe seg={res['without_probe_segment']['verdict']}")


def probe(gps):
    """The cost probe that ran before A2 (phase-randomised twin, no gating). Kept for the record."""
    log("probe superseded by A2; see PREREG section 9")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "templates":
        make_templates()
    elif cmd == "gates":
        detect_gates(sys.argv[2], int(sys.argv[3]))
    elif cmd == "series":
        tpl = load_templates()
        for mode in sys.argv[4].split(","):
            process_series(sys.argv[2], int(sys.argv[3]), mode, tpl)
    elif cmd == "caught":
        caught()
    elif cmd == "fg":
        fg_segment(int(sys.argv[2]))
    elif cmd == "g3":
        g3(int(sys.argv[2]), sys.argv[3]) if len(sys.argv) > 3 else g3()
    elif cmd == "g3merge":
        g3_merge()
    elif cmd == "combine":
        combine()
    elif cmd == "probe":
        probe(int(sys.argv[2]))
