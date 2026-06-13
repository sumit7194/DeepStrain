"""Rung 3 stage 0: semi-coherent ORACLE ceiling (pre-registered + amended in RESULTS.md).

Before building any phase-aware model, measure what the architecture CLASS can
possibly achieve: per-chunk matched filtering with the TRUE per-injection
template (unit-norm analytic chunks), on the same 64-s window / zero-FA /
6-real-noise-segment convention as cnn_w64. The smoke run showed the raw
matched filter is glitch-dominated on real noise, so we measure THREE ceilings
per chunk count, in one pass:

  S(t)    = sum_i |rho_i(t)|^2                         (incoherent comb power)
  chi2(t) = sum_i (|rho_i|^2 - p_i S)^2 / (p_i S)      (chunk-consistency veto)
  newSNR  = sqrt(S) / [(1+chi2_r^3)/2]^(1/6)  (chi2_r>1; PyCBC reweighting)

  - clean    : max_t S      vs SYNTHETIC-noise zero-FA  -> pure phase ceiling
  - vetoed   : max_t newSNR vs REAL-noise newSNR zero-FA -> realistic ceiling
  - raw-real : max_t S      vs REAL-noise S zero-FA      -> glitch-limited ref

n_chunks=1 is the fully coherent in-window matched filter (no veto possible -> stays
glitch-vulnerable, by construction). p_i are renormalized over kept chunks. The
chi2 sum is streamed via A=sum|rho_i|^2, B=sum|rho_i|^4/p_i, so chi2 = B/S - S.

Run:  .venv/bin/python scripts/semicoherent_oracle.py [--selftest] [--smoke]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pbh import config as C
from pbh.data import whiten_segment
from pbh.metrics import MASS_LABELS, mass_bin_results
from pbh.progress import progress
from pbh.waveforms import make_whitened_injection, sample_params

FS = C.SAMPLE_RATE
CROP = C.WHITEN_CROP_SEC * FS
WIN = 64 * FS  # the 64-s window the rung-3 model would see
PAD = FS // 2  # +-0.5 s local time scan around the true alignment
N_CHUNKS = (1, 2, 4, 8, 16)
N_INJ_PER_SEG = 250
SEED = C.SEED + 1111
OUT_DIR = C.RESULTS_DIR / "oracle"
ENERGY_FLOOR = 1e-3  # drop chunks carrying <0.1% of in-window energy
CNN_W64 = {"0.17-0.35": 0.407, "0.35-0.55": 0.457, "0.55-0.88": 0.476}


def analytic_chunks(g: np.ndarray, n: int):
    """Split in-window template g (len WIN, merger at end) into n equal spans;
    return [(offset, unit-norm analytic chunk, energy fraction p_i)] for chunks
    above the energy floor, with p_i renormalized over the kept chunks."""
    L = WIN // n
    total = float(np.sum(g**2)) or 1.0
    kept = []
    for i in range(n):
        seg = g[i * L : (i + 1) * L]
        e = float(np.sum(seg**2))
        if e >= ENERGY_FLOOR * total:
            kept.append((i * L, hilbert(seg) / np.sqrt(e), e))
    es = sum(e for *_, e in kept) or 1.0
    return [(off, a, e / es) for off, a, e in kept]


def newsnr(S: np.ndarray, B: np.ndarray, n_kept: int) -> np.ndarray:
    """PyCBC-style reweighted SNR from comb power S and B=sum|rho|^4/p."""
    rho = np.sqrt(np.maximum(S, 0.0))
    if n_kept < 2:
        return rho  # one chunk -> no consistency test possible
    Ssafe = np.where(S > 0, S, np.inf)
    chi2 = np.maximum(B / Ssafe - S, 0.0)
    chi2_r = chi2 / (2 * n_kept - 2)
    w = np.where(chi2_r > 1.0, ((1.0 + chi2_r**3) / 2.0) ** (1.0 / 6.0), 1.0)
    return rho / w


def _corr_sq_local(d: np.ndarray, off: int, a: np.ndarray, ws: int) -> np.ndarray:
    """|rho(tau)|^2 for tau in [0, 2*PAD] of chunk a placed at ws+off in d."""
    n_tau = 2 * PAD + 1
    lo = ws + off - PAD
    seg = d[lo : lo + len(a) + 2 * PAD]
    F = np.fft.fft(seg)
    cr = np.fft.ifft(F * np.fft.fft(np.conj(np.real(a)[::-1]), n=len(seg)))
    ci = np.fft.ifft(F * np.fft.fft(np.conj(np.imag(a)[::-1]), n=len(seg)))
    s = len(a) - 1
    return np.abs(cr[s : s + n_tau]) ** 2 + np.abs(ci[s : s + n_tau]) ** 2


def local_stats(d: np.ndarray, chunks, ws: int) -> tuple[float, float]:
    """(max_t S, max_t newSNR) over the +-PAD local scan around window start ws."""
    A = np.zeros(2 * PAD + 1)
    Bsum = np.zeros(2 * PAD + 1)
    for off, a, p in chunks:
        x = np.real(_corr_sq_local(d, off, a, ws))
        A += x
        Bsum += x**2 / p
    return float(A.max()), float(newsnr(A, Bsum, len(chunks)).max())


def segment_stats(wc: np.ndarray, chunks) -> tuple[float, float]:
    """(max_t S, max_t newSNR) over a whole segment (streamed A,B; linear part)."""
    N = len(wc)
    n_valid = N - WIN + 1
    D = np.fft.rfft(wc)
    A = np.zeros(n_valid)
    Bsum = np.zeros(n_valid)
    for off, a, p in chunks:
        cr = np.fft.irfft(D * np.conj(np.fft.rfft(np.real(a), n=N)), n=N)[off : off + n_valid]
        ci = np.fft.irfft(D * np.conj(np.fft.rfft(np.imag(a), n=N)), n=N)[off : off + n_valid]
        x = cr**2 + ci**2
        A += x
        Bsum += x**2 / p
    return float(A.max()), float(newsnr(A, Bsum, len(chunks)).max())


def selftest() -> None:
    rng = np.random.default_rng(7)
    t = np.arange(WIN) / FS
    g = np.sin(2 * np.pi * (60 * t + 8 * t**2)) * np.hanning(WIN)
    target = 20.0
    sig = g * (target / np.sqrt(np.sum(g**2)))
    wc = rng.normal(0, 1, 600 * FS)
    dwin = wc[400 * FS - WIN - PAD : 400 * FS + PAD].copy()
    dwin[PAD : PAD + WIN] += sig
    # realistic glitch: loud energy CONCENTRATED in one chunk-width (matches part
    # of the template, inconsistent with a real signal's spread -> high chi2)
    tail = WIN // 16
    exc = np.zeros(WIN)
    exc[-tail:] = g[-tail:]
    exc *= 40.0 / np.sqrt(np.sum(exc**2))
    gwin = wc[100 * FS - PAD : 100 * FS + WIN + PAD].copy()
    gwin[PAD : PAD + WIN] += exc
    print(f"signal target^2 = {target**2:.0f}")
    ratio = {}
    for n in N_CHUNKS:
        ch = analytic_chunks(sig, n)
        sS, sN = local_stats(dwin, ch, PAD)
        gS, gN = local_stats(gwin, ch, PAD)
        ratio[n] = gN / np.sqrt(gS) if gS > 0 else 1.0
        print(f"n={n:2d} ({len(ch)} kept): signal S={sS:6.1f} newSNR={sN:5.1f} | "
              f"concentrated-glitch S={gS:7.1f} newSNR={gN:6.1f}  ratio={ratio[n]:.2f}")
        assert sS > 0.8 * target**2, f"n={n}: lost MF power"
        assert abs(sN - np.sqrt(sS)) < 0.25 * np.sqrt(sS), f"n={n}: veto penalized a real signal"
    assert ratio[1] > 0.95, "n=1 should have no veto (newSNR==sqrt(S))"
    assert min(ratio[4], ratio[8], ratio[16]) < 0.6, "chunking failed to suppress a concentrated glitch"
    print("real signal kept (newSNR~=sqrt(S)); chunk veto suppresses concentrated glitches. SELFTEST PASS")


def threshold_templates(t0, psd, rng):
    """3 spaced-Mc templates (equal-mass 0.2/0.45/1.0) for noise thresholds."""
    outs = []
    for mm in (0.2, 0.45, 1.0):
        p = replace(sample_params(rng), mass1=mm, mass2=mm)
        h_w, _ = make_whitened_injection(p, "H1", t0, psd)
        outs.append((p.chirp_mass, h_w[-WIN:].copy()))
    return outs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="8 injections/segment")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    tag = "_smoke" if args.smoke else ""
    n_inj = 8 if args.smoke else N_INJ_PER_SEG
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((C.DATA_DIR / "manifest.json").read_text())
    test_segs = manifest["H1"]["test"]

    # ---- stage 1: zero-FA thresholds (real S, real newSNR, synthetic S) cached
    tpath = OUT_DIR / "thresholds.json"
    if tpath.exists():
        tdata = json.loads(tpath.read_text())
    else:
        t0_ = time.time()
        thr = {str(n): {"real_S": 0.0, "real_newsnr": 0.0, "synth_S": 0.0}
               for n in N_CHUNKS}
        rng_t = np.random.default_rng([SEED, 0])
        for si, gps in enumerate(test_segs):
            w, t0, psd = whiten_segment("H1", gps)
            wc = w[CROP:-CROP]
            syn = np.random.default_rng([SEED, si, 99]).normal(0, 1, len(wc))
            for _mc, g in threshold_templates(t0, psd, rng_t):
                for n in N_CHUNKS:
                    ch = analytic_chunks(g, n)
                    rS, rN = segment_stats(wc, ch)
                    syS, _ = segment_stats(syn, ch)
                    d = thr[str(n)]
                    d["real_S"] = max(d["real_S"], rS)
                    d["real_newsnr"] = max(d["real_newsnr"], rN)
                    d["synth_S"] = max(d["synth_S"], syS)
            progress("oracle_thresholds", si + 1, len(test_segs), elapsed_s=time.time() - t0_)
            print(f"[thr] {gps} done", flush=True)
        tmp = tpath.with_suffix(".tmp")
        tmp.write_text(json.dumps(thr, indent=2))
        os.replace(tmp, tpath)
        tdata = thr
    thr = {int(k): v for k, v in tdata.items()}
    print("thresholds:", {n: {k: round(x, 1) for k, x in v.items()} for n, v in thr.items()})

    # ---- stage 2: injections (per-segment atomic cache, resumable)
    import pandas as pd

    t_b = time.time()
    n_total = len(test_segs) * n_inj
    for seg_i, gps in enumerate(test_segs):
        ipath = OUT_DIR / f"inj_{gps}{tag}.parquet"
        if ipath.exists():
            continue
        w, t0, psd = whiten_segment("H1", gps)
        wc = w[CROP:-CROP]
        rng = np.random.default_rng([SEED, int(gps)])
        rows = []
        for j in range(n_inj):
            p = sample_params(rng)
            h_w, snr_ref = make_whitened_injection(p, "H1", t0, psd)
            target = float(rng.uniform(*C.EVAL_SNR_RANGE))
            sig = h_w * (target / snr_ref)
            m = int(rng.integers(len(sig), len(wc) - PAD))
            dwin = wc[m - WIN - PAD : m + PAD].copy()
            dwin[PAD : PAD + WIN] += sig[-WIN:]
            g = sig[-WIN:]
            row = dict(gps=gps, chirp_mass=p.chirp_mass, target_snr=target)
            for n in N_CHUNKS:
                S, nsnr = local_stats(dwin, analytic_chunks(g, n), PAD)
                row[f"S_n{n}"] = S
                row[f"newsnr_n{n}"] = nsnr
            rows.append(row)
            if (j + 1) % 10 == 0:
                progress("oracle_inj" + tag, seg_i * n_inj + j + 1, n_total,
                         elapsed_s=time.time() - t_b, segment=float(seg_i + 1))
        tmp = ipath.parent / (ipath.stem + ".tmp.parquet")
        pd.DataFrame(rows).to_parquet(tmp)
        os.replace(tmp, ipath)
        print(f"[inj] {gps}: wrote {ipath.name}", flush=True)

    # ---- stage 3: metrics — three ceilings per n
    df = pd.concat([pd.read_parquet(OUT_DIR / f"inj_{g}{tag}.parquet") for g in test_segs])
    mc = 3 if args.smoke else 10
    ceilings = {"clean": {}, "vetoed": {}, "raw_real": {}}
    for n in N_CHUNKS:
        df[f"det_clean_n{n}"] = df[f"S_n{n}"] > thr[n]["synth_S"]
        df[f"det_vetoed_n{n}"] = df[f"newsnr_n{n}"] > thr[n]["real_newsnr"]
        df[f"det_rawreal_n{n}"] = df[f"S_n{n}"] > thr[n]["real_S"]
        ceilings["clean"][f"n{n}"] = mass_bin_results(df, f"det_clean_n{n}", mc)
        ceilings["vetoed"][f"n{n}"] = mass_bin_results(df, f"det_vetoed_n{n}", mc)
        ceilings["raw_real"][f"n{n}"] = mass_bin_results(df, f"det_rawreal_n{n}", mc)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    x = np.arange(len(MASS_LABELS))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(N_CHUNKS)))
    for ax, (cname, cdata) in zip(axes, ceilings.items()):
        for i, (n, col) in enumerate(zip(N_CHUNKS, colors)):
            f = [cdata[f"n{n}"]["mf_distance_fraction"][m] for m in MASS_LABELS]
            ax.bar(x + (i - 2) * 0.16, f, 0.16, color=col, label=f"n={n}")
        for i, lab in enumerate(MASS_LABELS):
            ax.plot([i - 0.45, i + 0.45], [CNN_W64[lab]] * 2, "r--", lw=1.5)
        ax.plot([], [], "r--", label="cnn_w64")
        ax.axhline(1.0, color="k", ls="--", alpha=0.4)
        ax.set_xticks(x, MASS_LABELS)
        ax.set_title(f"{cname} ceiling")
        ax.set_ylabel("fraction of ideal-MF distance")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(f"semi-coherent oracle — {len(df)} injections, 64-s window")
    fig.tight_layout()
    fig.savefig(C.RESULTS_DIR / f"oracle_semicoherent{tag}.png", dpi=120)

    out = {
        "protocol": "true-template chunked MF; clean=synthS, vetoed=newSNR(real), "
                    "raw_real=S(real); chi2 chunk-consistency veto",
        "window_sec": 64, "n_injections": int(len(df)), "seed": SEED,
        "thresholds": {str(k): v for k, v in thr.items()},
        "cnn_w64_gate": CNN_W64, "ceilings": ceilings,
    }
    jpath = C.RESULTS_DIR / f"oracle_semicoherent{tag}.json"
    jpath.write_text(json.dumps(out, indent=2))
    progress("oracle_inj" + tag, n_total, n_total, elapsed_s=time.time() - t_b)
    for cname, cdata in ceilings.items():
        print(f"--- {cname} ---")
        print(json.dumps({k: {m: round(v["mf_distance_fraction"][m], 3) for m in MASS_LABELS}
                          for k, v in cdata.items()}, indent=0))
    print(f"wrote {jpath.name} + oracle_semicoherent{tag}.png")


if __name__ == "__main__":
    main()
