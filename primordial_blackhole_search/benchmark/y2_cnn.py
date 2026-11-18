"""Yardstick Y2: cnn_w64 + H1xL1 coincidence (sum), run as a benchmark submission.

Per detector, the 64-s cnn_w64 window slides over the whitened segment with a 1-s hop (2 spectrogram frames), so the
score is a time series s_d(t) indexed by window END time. The spectrogram is computed once per segment (in chunks,
to bound memory) and windows are slices of it. Coincidence statistic: s_H1(t) + s_L1(t) at the same window index
(the 10-ms light travel time is far below the 1-s hop).

Triggers (Y2's own clustering): local maxima of the coincidence series, loudest first, none within CLUSTER_S = 64 s
(one window) of a louder kept one. Trigger time = window-end time - OFFSET, tolerance var = VAR, both fixed on the DEV
segments only (`calibrate`), never on test data: OFFSET = median (peak window end - H1 merger) over dev injections whose
peak beats every dev zero-lag noise peak; VAR = the largest deviation from OFFSET among them + 2 s, capped at 60 s.

  python benchmark/y2_cnn.py probe <dev segment gps>     # timing + memory probe on a dev segment (no test data)
  python benchmark/y2_cnn.py calibrate                   # OFFSET, VAR from dev -> results/y2_calibration.json
  python benchmark/y2_cnn.py run                         # submission -> submissions/y2/ (resumable per series)
"""
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

B = Path(__file__).resolve().parent
sys.path.insert(0, str(B.parent)); sys.path.insert(0, str(B))
from pbh import config as C  # noqa: E402
from lowmem_psd import estimate_psd, whiten  # noqa: E402  (== pbh.data versions, results/lowmem_psd_check.json)
from pbh.models import make_model  # noqa: E402
from pbh.spectrogram import spectrogram  # noqa: E402
from pbh.sweep import SweepGrid, pool_and_log, score_windows  # noqa: E402
import injections as I  # noqa: E402

GRID = SweepGrid.short(64)
F_WIN, NBINS = GRID.frames_per_win, GRID.n_time_bins
HOP_FR = 2                                    # 2 frames x 0.5 s = 1-s hop between scored windows
FR_S = C.STFT_HOP_SEC
CROP = C.WHITEN_CROP_SEC * C.SAMPLE_RATE
NPERSEG = int(C.STFT_SEC * C.SAMPLE_RATE)
HOP = int(C.STFT_HOP_SEC * C.SAMPLE_RATE)


def load_model():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    m = make_model("cnn"); m.load_state_dict(torch.load(C.MODEL_DIR / "cnn_w64.pt", map_location=dev, weights_only=True))
    return m.to(dev).eval(), dev


def chunked_spectrogram(w, chunk_frames=1024):
    """spectrogram(w) computed in frame chunks: identical output, bounded memory (the one-shot version holds
    the full frame matrix and its rfft, ~0.5 GB for a 4096-s segment)."""
    n_frames = 1 + (len(w) - NPERSEG) // HOP
    out = np.empty((C.N_FREQ_BINS, n_frames), dtype=np.float32)
    for a in range(0, n_frames, chunk_frames):
        b = min(a + chunk_frames, n_frames)
        out[:, a:b] = spectrogram(w[a * HOP:(b - 1) * HOP + NPERSEG])
    return out


def score_series(model, dev, strain, batch=512):
    """Raw strain of one 4096-s segment -> (window-end offsets in s from segment start, scores)."""
    w = whiten(strain, estimate_psd(strain))[CROP:-CROP]
    spec = chunked_spectrogram(w)
    del w
    starts = np.arange(0, spec.shape[1] - F_WIN + 1, HOP_FR)
    scores = np.empty(len(starts), dtype=np.float32)
    for a in range(0, len(starts), batch):
        s = starts[a:a + batch]
        feats = np.stack([pool_and_log(spec[:, k:k + F_WIN], NBINS) for k in s])
        scores[a:a + len(s)] = score_windows(model, dev, feats).ravel()
    t_end = C.WHITEN_CROP_SEC + (starts * HOP + NPERSEG + (F_WIN - 1) * HOP) / C.SAMPLE_RATE
    return t_end, scores


def probe(gps):
    """Inject loud signals (SNR ~20 per detector) into a dev segment and locate the score peak relative to the
    H1 merger. Dev data only; the injection rows are made here, not drawn from the benchmark table."""
    model, dev = load_model()
    rng = np.random.default_rng(1)
    t0 = time.time()
    x = np.load(B / "data" / f"H1_{gps}.npy").astype(np.float64)
    psd = estimate_psd(x)
    rows, clean = [], x.copy()
    for k, t in enumerate(range(gps + 500, gps + 4096 - 500, 480)):
        m1, m2 = sorted(np.exp(rng.uniform(np.log(0.2), np.log(1.0), 2)), reverse=True)
        r = dict(mass1=m1, mass2=m2, ra=1.0, dec=0.3, psi=0.2, inclination=0.4, coa_phase=0.5,
                 t_geo=t + 0.37, distance=1.0)
        sig, s0 = I.detector_signal(r, "H1")
        alone = I.add_signal(np.zeros_like(x), gps, sig, s0)
        snr1 = float(np.sqrt(np.sum(whiten(alone, psd) ** 2)))
        x += alone * (20.0 / snr1)
        from pycbc.detector import Detector
        r["t_h1"] = r["t_geo"] + Detector("H1").time_delay_from_earth_center(r["ra"], r["dec"], r["t_geo"])
        rows.append(r)
    t_end, s = score_series(model, dev, x)
    _, s0n = score_series(model, dev, clean)
    out = []
    for r in rows:
        rel = t_end + gps - r["t_h1"]
        near = np.abs(rel) < 120
        k = np.argmax(np.where(near, s, -np.inf))
        above = rel[near & (s > s0n.max())]
        out.append(dict(m1=round(r["mass1"], 3), m2=round(r["mass2"], 3), peak_offset_s=float(rel[k]),
                        peak_score=float(s[k]), above_noise_max_span_s=[float(above.min()), float(above.max())]
                        if len(above) else None))
        print(out[-1], flush=True)
    res = dict(segment=gps, n_windows=len(s), noise_max=float(s0n.max()), noise_median=float(np.median(s0n)),
               seconds=time.time() - t0, injections=out)
    (B / "results" / "y2_probe.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "injections"}))


CLUSTER_S = 64.0
CACHE = B / "state" / "y2"
SUB = B / "submissions" / "y2"
DEV_REAL, DEV_SEED, DEV_DC_HI = 3, 7, 250.0   # dev injections: denser near threshold than the test population


def dev_rows():
    """Dev-only injections, same population and placement as injections.table() but on the dev segments, with a
    different seed and d_c on [5, 250] Mpc so more of them sit near the detection threshold."""
    man = json.loads((B / "manifest.json").read_text())
    rng = np.random.default_rng(DEV_SEED)
    rows = []
    for r in range(DEV_REAL):
        for g in man["dev"]:
            for t in np.arange(g + I.EDGE_S, g + C.SEGMENT_LEN - I.EDGE_S - I.JITTER_S + 1, I.SPACING_S):
                m1, m2 = sorted(np.exp(rng.uniform(np.log(C.M_MIN), np.log(C.M_MAX), 2)), reverse=True)
                dc = rng.uniform(I.DC_LO, DEV_DC_HI)
                rows.append(dict(realization=r, segment=int(g), t_geo=float(t + rng.uniform(0, I.JITTER_S)),
                                 mass1=m1, mass2=m2, ra=rng.uniform(0, 2 * np.pi),
                                 dec=float(np.arcsin(rng.uniform(-1, 1))), psi=rng.uniform(0, np.pi),
                                 inclination=float(np.arccos(rng.uniform(-1, 1))), coa_phase=rng.uniform(0, 2 * np.pi),
                                 chirp_distance=dc, distance=dc * (I.chirp_mass(m1, m2) / I.MC_BNS) ** (5 / 6)))
    return rows


def test_rows():
    with open(I.TABLE) as f:
        return [{k: (int(v) if k in ("id", "realization", "segment") else float(v)) for k, v in r.items()}
                for r in csv.DictReader(f)]


def series(model, dev, tag, det, seg, rows):
    """Cached score series of one detector of one segment with `rows` injected (tag names the injection set)."""
    ck = CACHE / f"{tag}_{det}_{seg}.npz"
    if ck.exists():
        z = np.load(ck); return z["t"], z["s"]
    x = np.load(B / "data" / f"{det}_{seg}.npy").astype(np.float64)
    for r in rows:
        sig, s0 = I.detector_signal(r, det)
        I.add_signal(x, seg, sig, s0)
    t, s = score_series(model, dev, x)
    t = t + seg
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = ck.with_suffix(".tmp.npz"); np.savez(tmp, t=t, s=s); os.replace(tmp, ck)
    return t, s


def peaks(t, c):
    """Y2's clustering: loudest first, drop anything within CLUSTER_S of a louder kept peak."""
    keep, taken = [], np.zeros(len(c), bool)
    step = t[1] - t[0]
    half = int(round(CLUSTER_S / step))
    for i in np.argsort(-c):
        if taken[i]:
            continue
        keep.append(i); taken[max(0, i - half):i + half + 1] = True
    keep = np.array(sorted(keep))
    return t[keep], c[keep]


def h1_time(r):
    from pycbc.detector import Detector
    return r["t_geo"] + Detector("H1").time_delay_from_earth_center(r["ra"], r["dec"], r["t_geo"])


def calibrate():
    model, dev = load_model()
    man = json.loads((B / "manifest.json").read_text())
    rows = dev_rows()
    noise_max = max(peaks(*_coinc(model, dev, "devnoise", g, g, []))[1].max() for g in man["dev"])
    offs = []
    for r in range(DEV_REAL):
        for g in man["dev"]:
            sub = [x for x in rows if x["realization"] == r and x["segment"] == g]
            pt, pc = peaks(*_coinc(model, dev, f"dev{r}", g, g, sub))
            for x in sub:
                th = h1_time(x)
                near = np.abs(pt - th) <= 120
                if near.any() and pc[near].max() > noise_max:
                    k = np.flatnonzero(near)[np.argmax(pc[near])]
                    offs.append(float(pt[k] - th))
            print(f"dev{r} {g}: {len(offs)} detected so far", flush=True)
    offs = np.array(offs)
    offset = float(np.median(offs))
    var = float(min(60.0, np.max(np.abs(offs - offset)) + 2.0))
    res = dict(n_dev_injections=len(rows), n_detected=len(offs), dev_noise_max=float(noise_max), offset_s=offset,
               var_s=var, offsets_s=sorted(round(o, 2) for o in offs))
    (B / "results" / "y2_calibration.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "offsets_s"}))


def _coinc(model, dev, tag, seg_h, seg_l, rows):
    th, sh = series(model, dev, tag, "H1", seg_h, rows)
    _, sl = series(model, dev, tag, "L1", seg_l, rows)
    return th, sh + sl


def write_sub(path, t, c, cal):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["time", "stat", "var"])
        for a, b in zip(t, c):
            w.writerow([f"{a - cal['offset_s']:.3f}", f"{b:.5f}", cal["var_s"]])
    os.replace(tmp, path)


def run():
    import evaluate as E
    cal = json.loads((B / "results" / "y2_calibration.json").read_text())
    model, dev = load_model()
    segs = json.loads((B / "manifest.json").read_text())["test"]
    rows = test_rows()
    SUB.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for k in range(E.N_SLIDES + 1):
        out = SUB / f"bg_k{k}.csv"
        if out.exists():
            continue
        tt, cc = [], []
        for i, g in enumerate(segs):
            th, sh = series(model, dev, "noise", "H1", g, [])
            _, sl = series(model, dev, "noise", "L1", segs[(i + k) % len(segs)], [])
            pt, pc = peaks(th, sh + sl); tt.append(pt); cc.append(pc)
        write_sub(out, np.concatenate(tt), np.concatenate(cc), cal)
        print(f"bg k={k} ({time.time() - t0:.0f} s)", flush=True)
    for r in range(I.K):
        out = SUB / f"fg_r{r}.csv"
        if out.exists():
            continue
        tt, cc = [], []
        for g in segs:
            sub = [x for x in rows if x["realization"] == r and x["segment"] == g]
            pt, pc = peaks(*_coinc(model, dev, f"r{r}", g, g, sub)); tt.append(pt); cc.append(pc)
        write_sub(out, np.concatenate(tt), np.concatenate(cc), cal)
        print(f"fg r={r} ({time.time() - t0:.0f} s)", flush=True)
    print("Y2 SUBMISSION COMPLETE", flush=True)


if __name__ == "__main__":
    {"probe": lambda: probe(int(sys.argv[2])), "calibrate": calibrate, "run": run}[sys.argv[1]]()
