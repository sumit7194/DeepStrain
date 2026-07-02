"""Follow-up A3: the REAL dense-bank semi-coherent detector — bank_oracle at the density A2b demands.

A2b measured the n=8 newSNR recovery vs Mc spacing: 0.25% -> 0.858, 0.5% -> 0.566, 2% -> 0.366 (explains
bank_oracle's 0.000 at ~2.5%). This runs the SAME statistic (max over bank of newSNR_n8; zero-FA over the
same 6 real test segments; same injection prior/conventions) at 0.1% spacing (~1,617 templates) — the first
REALIZABLE matched-filter-style subsolar detector in this project. bank_oracle (B=64) is untouched as the
recorded baseline; this is a template-major rewrite because B=1617 cannot hold all analytic chunks in RAM
(33 MB/template -> 53 GB): each template is generated, scored against the segment noise (threshold) and its
Mc-NEIGHBORHOOD injections, then freed. Per-segment sharding (--seg-index) + atomic parquet -> parallel +
power-loss-resumable. --merge computes the density sweep (subset thresholds/detections via nan-aware max)
and the A5 comparison against cnn_w64.

Run:  .venv/bin/python scripts/bank_dense.py --seg-index 0..5   (shards, parallel)
      .venv/bin/python scripts/bank_dense.py --merge            (after all shards)
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pbh import config as C
from pbh.data import whiten_segment
from pbh.progress import progress
from pbh.waveforms import make_whitened_injection, sample_params

HERE = Path(__file__).resolve().parent
_so = importlib.util.spec_from_file_location("so", HERE / "semicoherent_oracle.py")
so = importlib.util.module_from_spec(_so); _so.loader.exec_module(so)

WIN, PAD, CROP = so.WIN, so.PAD, so.CROP
MC_LO, MC_HI = 0.173, 0.871
EQ = 2.0 ** 0.2
N_CHUNK = 8
SEED = C.SEED + 3131
OUT = C.RESULTS_DIR / "bank_dense"
OUT.mkdir(exist_ok=True, parents=True)
MASS_EDGES, MASS_LABELS = [0.17, 0.35, 0.55, 0.88], ["0.17-0.35", "0.35-0.55", "0.55-0.88"]
SNR_BINS = np.linspace(*C.EVAL_SNR_RANGE, 13)


def bank_mcs(spacing: float) -> np.ndarray:
    n = int(np.ceil(np.log(MC_HI / MC_LO) / np.log(1 + spacing))) + 1
    return np.geomspace(MC_LO, MC_HI, n)


def run_segment(gps: int, spacing: float, n_inj: int, n_neighbors: int) -> None:
    path = OUT / f"seg_{gps}_s{spacing}.parquet"
    if path.exists():
        print(f"{gps}: cached"); return
    ckpt_path = OUT / f"ckpt_{gps}_s{spacing}.npz"
    w, t0, psd = whiten_segment("H1", gps)
    wc = w[CROP:-CROP]
    mcs = bank_mcs(spacing)
    B = len(mcs)

    # all injections for this segment up front (windows kept in RAM as float32, ~2.1 GB at 250).
    # Deterministic from [SEED, gps] -> identical across resumes; fingerprinted in the checkpoint.
    rng = np.random.default_rng([SEED, int(gps)])
    injs = []
    for _ in range(n_inj):
        p = sample_params(rng)
        h_w, snr_ref = make_whitened_injection(p, "H1", t0, psd)
        target = float(rng.uniform(*C.EVAL_SNR_RANGE))
        sig = h_w * (target / snr_ref)
        m = int(rng.integers(max(len(sig), WIN) + PAD, len(wc) - PAD))
        dwin = wc[m - WIN - PAD : m + PAD].copy()
        dwin[PAD : PAD + WIN] += sig[-WIN:] if len(sig) >= WIN else np.pad(sig, (WIN - len(sig), 0))
        injs.append((p.chirp_mass, target, dwin.astype(np.float32)))
    fingerprint = float(np.sum([i[2][::4096].sum() for i in injs]))
    print(f"{gps}: {n_inj} injections staged; bank B={B} @ spacing {spacing}", flush=True)

    # template-major: generate one template, threshold + neighborhood-injection scores, free
    base = sample_params(np.random.default_rng(0))          # bank_oracle's fiducial extrinsics
    thr = np.empty(B)
    inj_scores = np.full((n_inj, B), np.nan, dtype=np.float32)
    ti_start = 0
    if ckpt_path.exists():                                  # resume mid-segment after power loss
        ck = np.load(ckpt_path)
        if abs(float(ck["fingerprint"]) - fingerprint) < 1e-3 and int(ck["B"]) == B:
            ti_start = int(ck["next_ti"])
            thr[:ti_start] = ck["thr"][:ti_start]
            inj_scores[:, :ti_start] = ck["inj_scores"][:, :ti_start]
            print(f"{gps}: RESUMING from template {ti_start}/{B}", flush=True)
        else:
            print(f"{gps}: checkpoint fingerprint mismatch — starting fresh", flush=True)
    t0_ = time.time()
    for ti, mc in enumerate(mcs):
        if ti < ti_start:
            continue
        h, _ = make_whitened_injection(replace(base, mass1=float(mc * EQ), mass2=float(mc * EQ)), "H1", t0, psd)
        g = h[-WIN:].copy()
        if len(g) < WIN:
            g = np.pad(g, (WIN - len(g), 0))
        ch = so.analytic_chunks(g, N_CHUNK)
        thr[ti] = so.segment_stats(wc, ch)[1]               # max newSNR over the whole real-noise segment
        for j, (imc, _, dwin) in enumerate(injs):           # score only Mc-neighborhood injections
            k = int(np.abs(mcs - imc).argmin())
            if abs(k - ti) <= n_neighbors:
                inj_scores[j, ti] = so.local_stats(dwin.astype(np.float64), ch, PAD)[1]
        if ti % 5 == 0:
            progress("bank_dense", ti + 1, B, elapsed_s=time.time() - t0_, segment=float(gps))
            print(f"  t{ti}/{B} ({(time.time()-t0_)/60:.1f} min)", flush=True)
        if ti % 50 == 49:                                   # atomic mid-segment checkpoint (~every 10 min)
            tmp = ckpt_path.with_suffix(".tmp.npz")
            np.savez(tmp, next_ti=ti + 1, thr=thr, inj_scores=inj_scores, B=B, fingerprint=fingerprint)
            os.replace(tmp, ckpt_path)

    df = pd.DataFrame(inj_scores, columns=[f"t{k}" for k in range(B)])
    df.insert(0, "chirp_mass", [i[0] for i in injs])
    df.insert(1, "target_snr", [i[1] for i in injs])
    tmp = path.parent / (path.stem + ".tmp.parquet")
    df.to_parquet(tmp); os.replace(tmp, path)
    np.save(OUT / f"thr_{gps}_s{spacing}.npy", thr)
    ckpt_path.unlink(missing_ok=True)                       # segment complete -> checkpoint obsolete
    print(f"{gps}: done in {(time.time()-t0_)/60:.0f} min -> {path.name}", flush=True)


def merge(spacing: float) -> None:
    segs = json.loads((C.DATA_DIR / "manifest.json").read_text())["H1"]["test"]
    mcs = bank_mcs(spacing)
    B = len(mcs)
    thr_mat = np.stack([np.load(OUT / f"thr_{g}_s{spacing}.npy") for g in segs])       # (Nseg, B)
    dfs = [pd.read_parquet(OUT / f"seg_{g}_s{spacing}.parquet") for g in segs]
    df = pd.concat(dfs, ignore_index=True)
    score_cols = df[[f"t{k}" for k in range(B)]].to_numpy()

    out = {"spacing": spacing, "B": B, "n_inj": len(df), "sweep": {}}
    for nsub in (83, 164, 326, 649, B):
        idx = np.unique(np.linspace(0, B - 1, nsub).astype(int))
        thr_sub = float(np.nanmax(thr_mat[:, idx]))                                     # zero-FA over subset
        det = np.nanmax(score_cols[:, idx], axis=1) > thr_sub
        fracs = {}
        for lo, hi, lab in zip(MASS_EDGES[:-1], MASS_EDGES[1:], MASS_LABELS):
            sub = df[(df.chirp_mass >= lo) & (df.chirp_mass < hi)]
            cen, eff = [], []
            for a, b in zip(SNR_BINS[:-1], SNR_BINS[1:]):
                m = (sub.target_snr >= a) & (sub.target_snr < b)
                if m.sum() >= 10:
                    cen.append((a + b) / 2); eff.append(float(det[sub.index][m].mean()))
            snr50 = float(np.interp(0.5, eff, cen)) if len(cen) > 1 and max(eff) >= 0.5 else np.nan
            fracs[lab] = round(8.0 / snr50, 4) if np.isfinite(snr50) else 0.0
        out["sweep"][str(len(idx))] = {"threshold": thr_sub, "frac": fracs}
        print(f"B_sub={len(idx):>5}: thr {thr_sub:.2f} | frac {fracs}", flush=True)

    cnn = {"0.17-0.35": 0.41, "0.35-0.55": 0.46, "0.55-0.88": 0.48}                     # v1 gated headline
    full = out["sweep"][str(B)]["frac"]
    out["cnn_w64"] = cnn
    out["beats_cnn"] = bool(np.mean(list(full.values())) > np.mean(list(cnn.values())))
    (C.RESULTS_DIR / "bank_dense.json").write_text(json.dumps(out, indent=2))
    print(f"full-bank frac {full} vs cnn_w64 {cnn} -> beats_cnn={out['beats_cnn']}")
    print("wrote bank_dense.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spacing", type=float, default=0.001)
    ap.add_argument("--n-inj", type=int, default=250)
    ap.add_argument("--n-neighbors", type=int, default=48)
    ap.add_argument("--seg-index", type=int, default=None)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny: spacing 0.02, 20 inj, seg 0")
    args = ap.parse_args()
    if args.smoke:
        run_segment(json.loads((C.DATA_DIR / "manifest.json").read_text())["H1"]["test"][0], 0.02, 20, 8)
        return
    if args.merge:
        merge(args.spacing); return
    segs = json.loads((C.DATA_DIR / "manifest.json").read_text())["H1"]["test"]
    todo = [segs[args.seg_index]] if args.seg_index is not None else segs
    for g in todo:
        run_segment(g, args.spacing, args.n_inj, args.n_neighbors)


if __name__ == "__main__":
    main()
