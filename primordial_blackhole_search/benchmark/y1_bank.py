"""Yardstick Y1: the adequate semi-coherent bank (bank_dense, 0.05 % chirp-mass spacing, 3,235 equal-mass templates,
n = 8 chunks, newSNR), run over whole segments as a benchmark submission.

Statistic: exactly scripts/semicoherent_oracle.segment_stats (8 analytic chunks, comb power A, chi^2 from
B = sum |rho_i|^4 / p_i, PyCBC-style newSNR), kept as a TIME SERIES over template positions instead of its maximum.
Computed on the GPU in float32 with the FFT length padded to 2^24 (agrees with the float64 numpy reference to ~1e-6,
results/y1_probe.json and the golden). For each detector the running maximum over templates is kept in BIN = 256-sample
(62.5 ms) bins, with the in-bin position and template index of the maximum, so checkpoints are a few MB.

Templates: equal-mass TaylorF2, bank_dense's fiducial extrinsics, projected for H1 at a FIXED reference time T_REF (the
first test segment), whitened with each series' own PSD exactly as pbh.waveforms.whiten_waveform does (512-s buffer),
last 64 s kept. bank_dense used each segment's own time for the antenna pattern; with phase-maximised, unit-normalised
chunks that changes only a constant phase and amplitude, and one T_REF keeps background and foreground identical.

Network statistic: sqrt(n_H1^2 + n_L1^2) per H1 bin, the L1 value maximised over the adjacent +-1 bin (light travel
<= 10 ms < one 62.5-ms bin). Triggers: loudest first, none within CLUSTER_S = 1 s of a louder one; time = H1
template-end (merger) time; var = 1 s by construction (template end IS the merger; measured offsets in the golden).

Background: whole H1 and L1 noise series of every test segment, all 3,235 templates. Slides pair H1 of test[i] with L1
of test[(i+k) mod 24] from the cached per-segment bins, so no extra filtering.
Foreground: only the +-SCAN_S window around each injection, with the +-48 chirp-mass neighbour templates of the
injection (bank_dense's convention). This can only LOWER a foreground statistic relative to the full bank, against a
full-bank background, so Y1's sensitivity is conservative; the golden measures how often it actually differs.

  python benchmark/y1_bank.py probe <dev gps>    # cost + numpy/MPS agreement on 3 templates (done, y1_probe.json)
  python benchmark/y1_bank.py golden             # dev-only checks -> results/y1_golden.json
  python benchmark/y1_bank.py noise              # background bins, all test segments (resumable)
  python benchmark/y1_bank.py fg [group size]    # foreground windows, realizations in groups (resumable)
  python benchmark/y1_bank.py submit             # -> submissions/y1/
"""
import csv
import gc
import importlib.util
import json
import os
import resource
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

B = Path(__file__).resolve().parent
ROOT = B.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(B))
from pbh import config as C  # noqa: E402
from pbh.waveforms import BUFFER_SEC, sample_params  # noqa: E402
from lowmem_psd import estimate_psd, whiten  # noqa: E402
import injections as I  # noqa: E402

_so = importlib.util.spec_from_file_location("so", ROOT / "scripts" / "semicoherent_oracle.py")
so = importlib.util.module_from_spec(_so); _so.loader.exec_module(so)
_bd = importlib.util.spec_from_file_location("bd", ROOT / "scripts" / "bank_dense.py")
bd = importlib.util.module_from_spec(_bd); _bd.loader.exec_module(bd)

FS = C.SAMPLE_RATE
WIN, CROP, N_CHUNK, EQ = so.WIN, so.CROP, bd.N_CHUNK, bd.EQ
SPACING, NEIGH = 0.0005, 48
NF = 1 << 24                  # series FFT length (power of two: 5x faster on MPS than the raw length)
NW = 1 << 19                  # foreground window FFT length
BIN = 256
SCAN_S, VAR_S, CLUSTER_S = 2.0, 1.0, 1.0
LOUD_SNR, MIN_REC = 10.0, 0.7    # golden G4: optimal network SNR that counts as loud; minimum stat / optimal SNR
DEV = "mps"
STATE = B / "state" / "y1"
SUB = B / "submissions" / "y1"
MAN = json.loads((B / "manifest.json").read_text())
T_REF = MAN["test"][0]
MCS = bd.bank_mcs(SPACING)
BASE = sample_params(np.random.default_rng(0))


def peak_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 2 ** 20    # macOS reports bytes


def log(msg):
    drv = torch.mps.driver_allocated_memory() // 2 ** 20 if torch.backends.mps.is_available() else 0
    print(f"{time.strftime('%H:%M:%S')} {msg} | peak RSS {peak_mb()} MB, MPS driver {drv} MB", flush=True)


def atomic_npz(path, **kw):
    tmp = path.with_suffix(".tmp.npz"); np.savez(tmp, **kw); os.replace(tmp, path)


# ---------------------------------------------------------------- series preparation
def prep_series(x):
    """Raw strain of one detector-segment -> (cropped whitened data, template whitening filter on the 512-s grid)."""
    psd = estimate_psd(x)
    wc = whiten(x, psd)[CROP:-CROP].astype(np.float32)
    n_buf = BUFFER_SEC * FS
    df_old, df_new = FS / ((len(psd) - 1) * 2), FS / n_buf          # pycbc.psd.interpolate, as whiten_waveform does
    new_n = int(np.rint((len(psd) - 1) * df_old / df_new + 1))
    psd_i = np.interp(np.arange(new_n) * df_new, np.arange(len(psd)) * df_old, psd)
    s = psd_i[: n_buf // 2 + 1].copy()
    s[s <= 0] = np.inf
    w = np.sqrt(2.0 / (FS * s))
    f = np.fft.rfftfreq(n_buf, d=1.0 / FS)
    w[(f < (C.F_LOWER - 10)) | (f > C.F_HIGH)] = 0.0
    return wc, w


def template_strain(mc):
    """Projected (H1, T_REF) template ending AT THE MERGER (t_c = 0). pycbc's time-domain TaylorF2 carries 5.000 s
    after t_c (measured for 0.2-1.0 Msun), so pbh.waveforms.projected_waveform's last sample is 5 s after the merger,
    not at it; that is harmless where templates and injections share the function (bank_dense), but here injections
    are placed by their true merger time, so the template is cut at t_c."""
    from pycbc.detector import Detector
    from pycbc.waveform import get_td_waveform
    p = replace(BASE, mass1=float(mc * EQ), mass2=float(mc * EQ))
    hp, hc = get_td_waveform(approximant=C.APPROXIMANT, mass1=p.mass1, mass2=p.mass2, delta_t=1.0 / FS,
                             f_lower=C.F_LOWER, distance=1.0, inclination=p.inclination, coa_phase=p.coa_phase)
    fp, fx = Detector("H1").antenna_pattern(p.ra, p.dec, p.psi, T_REF)
    keep = hp.sample_times.numpy() <= 0.5 / FS
    return (fp * hp.numpy() + fx * hc.numpy())[keep].astype(np.float64)


def raw_template(mc):
    """Template (merger at the end) in a 512-s buffer; returns its rfft."""
    h = template_strain(mc)
    n_buf = BUFFER_SEC * FS
    buf = np.zeros(n_buf)
    h = h[-n_buf:]
    buf[-len(h):] = h
    return np.fft.rfft(buf)


def chunks_for(hf, w):
    g = np.fft.irfft(hf * w, n=BUFFER_SEC * FS)[-WIN:]
    return so.analytic_chunks(g, N_CHUNK)


# ---------------------------------------------------------------- GPU statistic
@torch.no_grad()
def newsnr_gpu(Dt, chunks, n_fft, n_valid):
    """newSNR at every valid template-start position. Dt = rfft(data, n=n_fft) on the GPU, shape (bins,) or a
    batch (J, bins) of windows filtered by the same template; returns (n_valid,) or (J, n_valid)."""
    single = Dt.dim() == 1
    D = Dt.unsqueeze(0) if single else Dt
    A = torch.zeros(D.shape[0], n_valid, device=DEV); Bs = torch.zeros_like(A)
    for off, a, p in chunks:
        T = torch.fft.rfft(torch.tensor(np.stack([a.real, a.imag]), dtype=torch.float32, device=DEV), n=n_fft)
        c = torch.fft.irfft(D.unsqueeze(1) * T.conj().unsqueeze(0), n=n_fft)[..., off:off + n_valid]
        x = (c * c).sum(1)
        A += x; Bs += x * x / p
    n = len(chunks)
    rho = A.clamp_min(0).sqrt()
    if n >= 2:
        chi2r = (Bs / A.clamp_min(1e-30) - A).clamp_min(0) / (2 * n - 2)
        rho = rho / torch.where(chi2r > 1, ((1 + chi2r ** 3) / 2) ** (1 / 6), torch.ones_like(chi2r))
    return rho[0] if single else rho


class Binned:
    """Running max over templates, per BIN-sample bin: value, in-bin offset, template index."""

    def __init__(self, n_bins, state=None):
        self.n = n_bins
        if state is None:
            self.val = torch.full((n_bins,), -1.0, device=DEV)
            self.off = torch.zeros(n_bins, dtype=torch.int32, device=DEV)
            self.tid = torch.full((n_bins,), -1, dtype=torch.int32, device=DEV)
        else:
            self.val, self.off, self.tid = (torch.tensor(state[k], device=DEV) for k in ("val", "off", "tid"))

    def update(self, ns, ti):
        v, o = ns[: self.n * BIN].view(self.n, BIN).max(dim=1)
        m = v > self.val
        self.val = torch.where(m, v, self.val)
        self.off = torch.where(m, o.to(torch.int32), self.off)
        self.tid = torch.where(m, torch.full_like(self.tid, ti), self.tid)

    def numpy(self):
        return {k: getattr(self, k).cpu().numpy() for k in ("val", "off", "tid")}


# ---------------------------------------------------------------- background (whole series)
def noise_segment(g):
    out = STATE / f"noise_{g}.npz"
    if out.exists():
        return
    ck = STATE / f"ckpt_noise_{g}.npz"
    series = {}
    for d in ("H1", "L1"):
        wc, w = prep_series(np.load(B / "data" / f"{d}_{g}.npy").astype(np.float64))
        nv = len(wc) - WIN + 1
        series[d] = dict(Dt=torch.fft.rfft(torch.tensor(wc, device=DEV), n=NF), w=w, nv=nv)
        del wc
    n_bins = series["H1"]["nv"] // BIN
    start, state = 0, {d: None for d in series}
    if ck.exists():
        z = np.load(ck)
        start = int(z["next_ti"])
        state = {d: {k: z[f"{d}_{k}"] for k in ("val", "off", "tid")} for d in series}
    bins = {d: Binned(n_bins, state[d]) for d in series}
    log(f"noise {g}: start template {start}/{len(MCS)}")
    t0 = time.time()
    for ti in range(start, len(MCS)):
        hf = raw_template(MCS[ti])
        for d, s in series.items():
            bins[d].update(newsnr_gpu(s["Dt"], chunks_for(hf, s["w"]), NF, s["nv"]), ti)
        if ti % 100 == 99 or ti == len(MCS) - 1:
            atomic_npz(ck, next_ti=ti + 1, **{f"{d}_{k}": v for d in bins for k, v in bins[d].numpy().items()})
            log(f"noise {g}: {ti + 1}/{len(MCS)} templates ({(time.time() - t0) / 60:.1f} min)")
    atomic_npz(out, **{f"{d}_{k}": v for d in bins for k, v in bins[d].numpy().items()})
    ck.unlink(missing_ok=True)


def noise():
    STATE.mkdir(parents=True, exist_ok=True)
    for g in MAN["test"]:
        noise_segment(g)
    log("Y1 NOISE COMPLETE")


# ---------------------------------------------------------------- foreground (windows around injections)
def h1_time(r):
    from pycbc.detector import Detector
    return r["t_geo"] + Detector("H1").time_delay_from_earth_center(r["ra"], r["dec"], r["t_geo"])


def window_start(t_h1, g):
    """First template-start index (in the cropped series) of the scan window, aligned to the background's bins."""
    i = int(np.floor((t_h1 - SCAN_S - g - C.WHITEN_CROP_SEC) * FS)) - (WIN - 1)
    return (i // BIN) * BIN


M_SCAN = int(2 * SCAN_S * FS) + BIN                # template starts scanned per window (a whole number of bins)


def build_windows(rows):
    """For each injection row: per detector the whitened data window (float32) and that series' template filter."""
    wins, filters = [], {}
    by_seg = {}
    for j, r in enumerate(rows):
        by_seg.setdefault(r["segment"], []).append(j)
    out = [None] * len(rows)
    for g, js in by_seg.items():
        for d in ("H1", "L1"):
            x = np.load(B / "data" / f"{d}_{g}.npy").astype(np.float64)
            for j in js:
                sig, s0 = I.detector_signal(rows[j], d)
                I.add_signal(x, g, sig, s0)
            wc, w = prep_series(x)
            del x
            filters[(g, d)] = w
            for j in js:
                i0 = window_start(rows[j]["t_h1"], g)
                seg = wc[i0:i0 + M_SCAN + WIN - 1].copy()
                out[j] = out[j] or {}
                out[j][d] = (i0, seg)
            del wc
    return out, filters


def neighbours(rows):
    k = np.array([int(np.abs(MCS - I.chirp_mass(r["mass1"], r["mass2"])).argmin()) for r in rows])
    return k


def fg_windows(rows, wins, filters, templates, track_full=False):
    """Running binned max per injection and detector over the given template indices. Each injection only sees
    templates within +-NEIGH of its own chirp-mass index (the shortcut); with track_full it also keeps the full-bank
    max for comparison (golden only). Windows sharing a segment and detector are filtered in one batched GPU call."""
    kk = neighbours(rows)
    n_bins = M_SCAN // BIN
    res = {(j, d): Binned(n_bins) for j in range(len(rows)) for d in ("H1", "L1")}
    full = {(j, d): Binned(n_bins) for j in range(len(rows)) for d in ("H1", "L1")} if track_full else None
    groups = {}
    for j, r in enumerate(rows):
        for d in ("H1", "L1"):
            groups.setdefault((r["segment"], d), []).append(j)
    Dw = {key: torch.stack([torch.fft.rfft(torch.tensor(wins[j][key[1]][1], device=DEV), n=NW) for j in js])
          for key, js in groups.items()}
    t0 = time.time()
    for n, ti in enumerate(templates):
        near = np.abs(kk - ti) <= NEIGH
        if not (track_full or near.any()):
            continue
        hf = raw_template(MCS[ti])
        for key, js in groups.items():
            sel = [m for m, j in enumerate(js) if track_full or near[j]]
            if not sel:
                continue
            ns = newsnr_gpu(Dw[key][sel], chunks_for(hf, filters[key]), NW, M_SCAN)
            for row, m in enumerate(sel):
                j = js[m]
                if near[j]:
                    res[(j, key[1])].update(ns[row], ti)
                if track_full:
                    full[(j, key[1])].update(ns[row], ti)
        if n % 250 == 249:
            log(f"  fg templates {n + 1}/{len(templates)} ({(time.time() - t0) / 60:.1f} min)")
    return res, full


def network_trigger(h, l, i0, g):
    """Loudest network bin of one injection window -> (H1 merger time, statistic, best H1 template)."""
    lv = np.maximum.reduce([np.roll(l["val"], s) for s in (-1, 0, 1)])
    lv[0] = max(l["val"][0], l["val"][1]); lv[-1] = max(l["val"][-1], l["val"][-2])
    net = np.sqrt(np.maximum(h["val"], 0) ** 2 + np.maximum(lv, 0) ** 2)
    b = int(np.argmax(net))
    t = g + C.WHITEN_CROP_SEC + (i0 + b * BIN + int(h["off"][b]) + WIN - 1) / FS
    return t, float(net[b]), int(h["tid"][b])


def test_rows():
    with open(I.TABLE) as f:
        rows = [{k: (int(v) if k in ("id", "realization", "segment") else float(v)) for k, v in r.items()}
                for r in csv.DictReader(f)]
    for r in rows:
        r["t_h1"] = h1_time(r)
    return rows


def win_path(reals, part=None):
    return STATE / f"win_{'_'.join(map(str, reals))}{'' if part is None else f'_p{part}'}.npz"


WIN_PARTS = 6      # 4 segments per child build: pycbc waveform generation leaks ~50 MB per distinct-length waveform (measured 2026-11-20), so each child must stay small


def fgwin(reals, part=None):
    """Child-process step: build one pass's windows and filters (optionally only segment group `part`) and save them.
    pycbc's waveform generation keeps ~1.4 GB after the first injections and never returns it (measured 2026-11-20),
    so building inside the long-lived job pushed it past the 4 GB watchdog; a child process gives it back on exit."""
    sub = [r for r in test_rows() if r["realization"] in reals]
    if part is not None:
        segs = sorted({r["segment"] for r in sub})[part::WIN_PARTS]
        sub = [r for r in sub if r["segment"] in segs]
    wins, filters = build_windows(sub)
    kw = {}
    for r, w in zip(sub, wins):
        for d in ("H1", "L1"):
            kw[f"i0_{r['id']}_{d}"] = np.int64(w[d][0]); kw[f"w_{r['id']}_{d}"] = w[d][1]
    for (g, d), f in filters.items():
        kw[f"f_{g}_{d}"] = f
    atomic_npz(win_path(reals, part), **kw)
    log(f"fgwin {reals} part {part}: saved")


def load_windows(reals, sub):
    import subprocess
    zs = []
    for part in range(WIN_PARTS):
        p = win_path(reals, part)
        if not p.exists():
            subprocess.run([sys.executable, "-u", __file__, "fgwin", ",".join(map(str, reals)), str(part)], check=True)
        zs.append(np.load(p))
    get = lambda k: next(z[k] for z in zs if k in z.files)
    wins = [{d: (int(get(f"i0_{r['id']}_{d}")), get(f"w_{r['id']}_{d}")) for d in ("H1", "L1")} for r in sub]
    filters = {(r["segment"], d): get(f"f_{r['segment']}_{d}") for r in sub for d in ("H1", "L1")}
    return wins, filters


def fg(group=2, one_pass=False):
    STATE.mkdir(parents=True, exist_ok=True)
    rows = test_rows()
    for r0 in range(0, I.K, group):
        reals = list(range(r0, min(r0 + group, I.K)))
        out = STATE / f"fg_{'_'.join(map(str, reals))}.json"
        if out.exists():
            continue
        sub = [r for r in rows if r["realization"] in reals]
        wins, filters = load_windows(reals, sub)
        log(f"fg realizations {reals}: {len(sub)} windows built")
        res, _ = fg_windows(sub, wins, filters, range(len(MCS)))
        trig = []
        for j, r in enumerate(sub):
            h, l = res[(j, "H1")].numpy(), res[(j, "L1")].numpy()
            t, st, tid = network_trigger(h, l, wins[j]["H1"][0], r["segment"])
            trig.append(dict(id=r["id"], realization=r["realization"], time=t, stat=st, tid=tid,
                             dt=t - r["t_h1"]))
        tmp = out.with_suffix(".tmp"); tmp.write_text(json.dumps(trig)); os.replace(tmp, out)
        log(f"fg realizations {reals} done")
        for part in range(WIN_PARTS):
            win_path(reals, part).unlink(missing_ok=True)
        if one_pass:
            return
        # free this pass before the next builds its windows: holding them pushed the second build to 4.4 GB and the
        # watchdog stopped the run (2026-11-20); identical arithmetic, only the lifetime changes
        del wins, filters, res, trig
        gc.collect()
        torch.mps.empty_cache()
    log("Y1 FOREGROUND COMPLETE")


# ---------------------------------------------------------------- submission
def fgloop():
    """One foreground pass per fresh process (python's and MPS's retained memory is returned between passes)."""
    import subprocess
    while len(list(STATE.glob("fg_*.json"))) < I.K:
        subprocess.run([sys.executable, "-u", __file__, "fg", "1", "one"], check=True)
    log("Y1 FOREGROUND COMPLETE (fgloop)")


def series_triggers(h, l, g):
    """Network bins of one (H1 series, L1 series) pair -> clustered triggers (time, stat)."""
    lv = np.maximum.reduce([np.pad(l["val"], 1, mode="edge")[s:s + len(l["val"])] for s in (0, 1, 2)])
    net = np.sqrt(np.maximum(h["val"], 0) ** 2 + np.maximum(lv, 0) ** 2)
    t = g + C.WHITEN_CROP_SEC + (np.arange(len(net)) * BIN + h["off"] + WIN - 1) / FS
    half = int(round(CLUSTER_S * FS / BIN))
    taken = np.zeros(len(net), bool); keep = []
    for i in np.argsort(-net):
        if not taken[i]:
            keep.append(i); taken[max(0, i - half):i + half + 1] = True
    keep = np.sort(keep)
    return t[keep], net[keep]


def write_sub(path, t, s):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["time", "stat", "var"])
        for a, b in zip(t, s):
            w.writerow([f"{a:.4f}", f"{b:.5f}", VAR_S])
    os.replace(tmp, path)


def submit():
    import evaluate as E
    segs = MAN["test"]
    z = {g: np.load(STATE / f"noise_{g}.npz") for g in segs}
    get = lambda g, d: {k: z[g][f"{d}_{k}"] for k in ("val", "off", "tid")}
    SUB.mkdir(parents=True, exist_ok=True)
    for k in range(E.N_SLIDES + 1):
        tt, ss = [], []
        for i, g in enumerate(segs):
            t, s = series_triggers(get(g, "H1"), get(segs[(i + k) % len(segs)], "L1"), g)
            tt.append(t); ss.append(s)
        write_sub(SUB / f"bg_k{k}.csv", np.concatenate(tt), np.concatenate(ss))
    trig = [x for p in sorted(STATE.glob("fg_*.json")) for x in json.loads(p.read_text())]
    assert len(trig) == len(test_rows()), (len(trig), "foreground incomplete")
    for r in range(I.K):
        xs = [x for x in trig if x["realization"] == r]
        write_sub(SUB / f"fg_r{r}.csv", [x["time"] for x in xs], [x["stat"] for x in xs])
    log("Y1 SUBMISSION WRITTEN")


# ---------------------------------------------------------------- golden (dev data only)
def golden(inj_segments=None, step=1, tag=""):
    """G1 template whitening == pbh.waveforms.whiten_waveform. G2 GPU binned series == float64 numpy reference.
    G3 window statistic == whole-series statistic at the same positions. G4 on dev injections: recovered time vs H1
    merger, and the +-48-neighbour shortcut vs the full bank (how often and by how much it differs).
    G4 pass rule (2026-11-18, second version): every injection whose OPTIMAL network SNR (true template, the oracle's
    arithmetic) is >= LOUD_SNR must be recovered within VAR_S with stat >= MIN_REC x optimal, and there must be >= 3
    such injections. The first version used chirp distance < 70 Mpc as 'loud' and failed on dev[:2] because d_c
    ignores orientation (its d_c 58 Mpc injection has optimal SNR 5.9); that verdict is still reported as G4_dc_proxy.
    The rule was changed AFTER that failure, so it is confirmed on dev segments it was not tuned on (`golden confirm`)."""
    import y2_cnn as Y2
    from pbh.waveforms import whiten_waveform
    from pycbc.types import FrequencySeries
    res, ok = {}, True
    g = MAN["dev"][0]
    x = np.load(B / "data" / f"H1_{g}.npy").astype(np.float64)
    wc, w = prep_series(x)
    psd = estimate_psd(x); del x
    log("golden: series prepared")
    # G1
    psd_fs = FrequencySeries(psd, delta_f=FS / ((len(psd) - 1) * 2))
    g1 = []
    for mc in (MCS[0], MCS[len(MCS) // 2], MCS[-1]):
        ref = whiten_waveform(template_strain(mc), psd_fs)[-WIN:]
        got = np.fft.irfft(raw_template(mc) * w, n=BUFFER_SEC * FS)[-WIN:]
        g1.append(float(np.max(np.abs(got - ref)) / np.std(ref)))
    res["G1_template_whitening_max_rel"] = g1
    ok &= max(g1) < 1e-9
    # G2
    nv = len(wc) - WIN + 1
    Dt = torch.fft.rfft(torch.tensor(wc, device=DEV), n=NF)
    D64 = np.fft.rfft(wc.astype(np.float64))
    g2 = []
    for ti in (0, len(MCS) // 2, len(MCS) - 1):
        ch = chunks_for(raw_template(MCS[ti]), w)
        b = Binned(nv // BIN); b.update(newsnr_gpu(Dt, ch, NF, nv), ti)
        ref = _series_numpy(wc.astype(np.float64), ch, D64)[: (nv // BIN) * BIN].reshape(-1, BIN).max(1)
        g2.append(float(np.max(np.abs(b.numpy()["val"] - ref)) / ref.max()))
    res["G2_gpu_vs_numpy_binned_max_rel"] = g2
    ok &= max(g2) < 1e-4
    log("golden: G1 G2 done")
    # G3
    ti = len(MCS) // 2
    ch = chunks_for(raw_template(MCS[ti]), w)
    full = newsnr_gpu(Dt, ch, NF, nv).cpu().numpy()
    i0 = 1_000_000 // BIN * BIN
    seg = wc[i0:i0 + M_SCAN + WIN - 1]
    part = newsnr_gpu(torch.fft.rfft(torch.tensor(seg, device=DEV), n=NW), ch, NW, M_SCAN).cpu().numpy()
    res["G3_window_vs_series_max_rel"] = float(np.max(np.abs(part - full[i0:i0 + M_SCAN])) / full.max())
    ok &= res["G3_window_vs_series_max_rel"] < 1e-4
    del Dt, full, wc
    log("golden: G3 done")
    # G4
    inj_segments = inj_segments or MAN["dev"][:2]
    rows = [r for r in Y2.dev_rows() if r["realization"] == 0 and r["segment"] in inj_segments]
    opt = {}
    for gs in inj_segments:
        for d in ("H1", "L1"):
            xs = np.load(B / "data" / f"{d}_{gs}.npy").astype(np.float64)
            ps = estimate_psd(xs)
            for j, r in enumerate(rows):
                if r["segment"] == gs:
                    sig, t0 = I.detector_signal(r, d)
                    opt.setdefault(j, {})[d] = float(np.sqrt(np.sum(whiten(I.add_signal(np.zeros_like(xs), gs, sig, t0), ps) ** 2)))
            del xs, ps
    log("golden: optimal SNRs computed")
    for r in rows:
        r["t_h1"] = h1_time(r)
    wins, filters = build_windows(rows)
    log(f"golden: {len(rows)} dev injection windows built")
    nb, fb = fg_windows(rows, wins, filters, range(0, len(MCS), step), track_full=True)
    g4 = []
    for j, r in enumerate(rows):
        i0 = wins[j]["H1"][0]
        tn, sn, kn = network_trigger(nb[(j, "H1")].numpy(), nb[(j, "L1")].numpy(), i0, r["segment"])
        tf, sf, kf = network_trigger(fb[(j, "H1")].numpy(), fb[(j, "L1")].numpy(), i0, r["segment"])
        g4.append(dict(mc=I.chirp_mass(r["mass1"], r["mass2"]), d_c=r["chirp_distance"], neigh_stat=sn,
                       full_stat=sf, full_minus_neigh=sf - sn, full_tid_in_neigh=bool(abs(kf - neighbours([r])[0]) <= NEIGH),
                       neigh_dt_s=tn - r["t_h1"], full_dt_s=tf - r["t_h1"],
                       opt_net=float(np.hypot(opt[j]["H1"], opt[j]["L1"]))))
    res["G4_injections"] = g4
    loud = [x for x in g4 if x["d_c"] < 70]          # optimal network SNR >~ 20 at these distances
    res["G4_summary"] = dict(n=len(g4), n_differ=int(sum(x["full_minus_neigh"] > 1e-6 for x in g4)),
                             max_diff=max(x["full_minus_neigh"] for x in g4),
                             n_loud_dc_lt70=len(loud), loud_min_neigh_stat=min((x["neigh_stat"] for x in loud), default=None),
                             loud_max_abs_dt_s=max((abs(x["neigh_dt_s"]) for x in loud), default=None))
    # a loud injection must be recovered loudly and on time; the first version of this check passed vacuously
    # (no injection exceeded its bar) while every injection was 5 s off -- so it requires loud cases to exist
    res["G4_dc_proxy"] = dict(pass_=bool(len(loud) >= 3 and all(abs(x["neigh_dt_s"]) <= VAR_S and x["neigh_stat"] > 12
                                                                  for x in loud)), superseded="d_c ignores orientation")
    louds = [x for x in g4 if x["opt_net"] >= LOUD_SNR]
    res["G4_snr_rule"] = dict(loud_snr=LOUD_SNR, min_recovery=MIN_REC, n_loud=len(louds),
                              recovery=[x["neigh_stat"] / x["opt_net"] for x in louds],
                              abs_dt_s=[abs(x["neigh_dt_s"]) for x in louds])
    ok &= len(louds) >= 3 and all(abs(x["neigh_dt_s"]) <= VAR_S and x["neigh_stat"] >= MIN_REC * x["opt_net"]
                                  for x in louds)
    res["peak_rss_mb"] = peak_mb()
    res["template_step"] = step
    res["pass"] = bool(ok)
    res["inj_segments"] = [int(x) for x in inj_segments]
    (B / "results" / (f"y1_golden{tag}.json" if step == 1 else "y1_golden_smoke.json")).write_text(json.dumps(res, indent=1))
    log(f"golden: {'PASS' if ok else 'FAIL'} {json.dumps(res['G4_summary'])}")


def _series_numpy(wc, chunks, D):
    N = len(wc); nv = N - WIN + 1
    A = np.zeros(nv); Bs = np.zeros(nv)
    for off, a, p in chunks:
        cr = np.fft.irfft(D * np.conj(np.fft.rfft(np.real(a), n=N)), n=N)[off:off + nv]
        ci = np.fft.irfft(D * np.conj(np.fft.rfft(np.imag(a), n=N)), n=N)[off:off + nv]
        x = cr ** 2 + ci ** 2
        A += x; Bs += x ** 2 / p
    return so.newsnr(A, Bs, len(chunks))


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"golden": lambda: golden(step=int(sys.argv[2]) if len(sys.argv) > 2 else 1),
     "confirm": lambda: golden(inj_segments=MAN["dev"][2:4], tag="_confirm"), "noise": noise, "submit": submit,
     "fgwin": lambda: fgwin([int(v) for v in sys.argv[2].split(",")], int(sys.argv[3]) if len(sys.argv) > 3 else None),
     "fg": lambda: fg(int(sys.argv[2]) if len(sys.argv) > 2 else 2, one_pass=len(sys.argv) > 3 and sys.argv[3] == "one"),
     "fgloop": fgloop}[cmd]()
