"""Path F, milestone F0: the bank-mismatch gate.

The stage-0 oracle (semicoherent_oracle.py) matched-filtered with the TRUE
per-injection template and reached vetoed fractions 0.66/0.76/0.75. A real
detector can't use the true template — it uses a fixed BANK. This script swaps
the true template for a coarse equal-mass bank (mass-only grid) and re-measures,
on the SAME injections/noise/seeds and the SAME n=8 newSNR (vetoed) statistic,
how much sensitive distance survives template MISMATCH.

  detection statistic = max over bank templates of newSNR_n8(window)
  zero-FA threshold    = max over bank templates of newSNR_n8(real noise)

Bigger bank => less mismatch (helps) but a higher noise floor (trials, hurts);
the subset-size sweep traces that tradeoff in one run (per-template stats are
stored, so subsets are computed offline without rescoring).

GATE: does a coarse bank, with NO learned veto, already beat cnn_w64
(0.41/0.46/0.48)? If yes, path F is worth the learned-veto build (F1). If
mismatch crushes it to ~0, we've measured the bank-density wall instead.

Run:  .venv/bin/python scripts/bank_oracle.py [--bank 48] [--smoke]
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
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # to import the oracle module

from pbh import config as C
from pbh.data import whiten_segment
from pbh.metrics import MASS_LABELS, mass_bin_results
from pbh.progress import progress
from pbh.waveforms import make_whitened_injection, sample_params
from semicoherent_oracle import (
    CROP, FS, PAD, WIN, analytic_chunks, local_stats, segment_stats,
)

N_CHUNK = 8                  # the stage-0 sweet spot
SEED = C.SEED + 1111         # MUST match semicoherent_oracle.py -> same injections
OUT = C.RESULTS_DIR / "bank"
CNN_W64 = {"0.17-0.35": 0.407, "0.35-0.55": 0.457, "0.55-0.88": 0.476}
ORACLE_VETOED = {"0.17-0.35": 0.663, "0.35-0.55": 0.764, "0.55-0.88": 0.752}


def build_bank(t0, psd, n_templates: int):
    """Coarse equal-mass bank, mass-only grid (non-mass params fixed to one
    fiducial draw). Returns [(chirp_mass, whitened 64-s template)], mass-ordered,
    plus the precomputed n=8 analytic chunks per template."""
    base = sample_params(np.random.default_rng(0))  # fiducial sky/orientation
    masses = np.geomspace(C.M_MIN, C.M_MAX, n_templates)
    bank, chunks = [], []
    for m in masses:
        p = replace(base, mass1=float(m), mass2=float(m))
        h_w, _ = make_whitened_injection(p, "H1", t0, psd)
        g = h_w[-WIN:].copy()
        bank.append((p.chirp_mass, g))
        chunks.append(analytic_chunks(g, N_CHUNK))
    return bank, chunks


def subset_indices(B: int):
    """Evenly-spaced bank subsets (mass-ordered) for the density sweep."""
    sizes = [s for s in (3, 6, 12, 24, 48, 96) if s < B] + [B]
    return [(s, np.unique(np.linspace(0, B - 1, s).round().astype(int))) for s in sizes]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=int, default=48, help="number of bank templates")
    ap.add_argument("--smoke", action="store_true", help="8 injections/segment")
    args = ap.parse_args()
    B = args.bank
    tag = f"_B{B}" + ("_smoke" if args.smoke else "")
    n_inj = 8 if args.smoke else 250
    OUT.mkdir(parents=True, exist_ok=True)
    test_segs = json.loads((C.DATA_DIR / "manifest.json").read_text())["H1"]["test"]

    # ---- stage 1: per-template noise newSNR per segment (-> zero-FA thr per subset)
    tpath = OUT / f"thr{tag}.json"
    if tpath.exists():
        tdata = json.loads(tpath.read_text())
        seg_thr = {g: np.array(v) for g, v in tdata["seg_newsnr"].items()}
        bank_mc = np.array(tdata["bank_mc"])
    else:
        seg_thr, bank_mc, t0_ = {}, None, time.time()
        for si, gps in enumerate(test_segs):
            w, t0, psd = whiten_segment("H1", gps)
            wc = w[CROP:-CROP]
            bank, chunks = build_bank(t0, psd, B)
            bank_mc = np.array([mc for mc, _ in bank])
            vals = np.empty(B)
            for ti, ch in enumerate(chunks):  # per-template heartbeat (segment_stats is heavy)
                vals[ti] = segment_stats(wc, ch)[1]
                if ti % 4 == 0:
                    progress("bank_thr", si * B + ti + 1, len(test_segs) * B,
                             elapsed_s=time.time() - t0_, segment=float(si + 1))
            seg_thr[str(gps)] = vals
            print(f"[thr] {gps}: bank max newSNR {vals.max():.1f}", flush=True)
        tmp = tpath.with_suffix(".tmp")
        tmp.write_text(json.dumps({"seg_newsnr": {g: v.tolist() for g, v in seg_thr.items()},
                                   "bank_mc": bank_mc.tolist()}))
        os.replace(tmp, tpath)
        seg_thr = {g: v for g, v in seg_thr.items()}

    # ---- stage 2: per-template injection newSNR (per-segment parquet, resumable)
    t_b = time.time()
    for si, gps in enumerate(test_segs):
        ipath = OUT / f"inj_{gps}{tag}.parquet"
        if ipath.exists():
            continue
        w, t0, psd = whiten_segment("H1", gps)
        wc = w[CROP:-CROP]
        _, chunks = build_bank(t0, psd, B)
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
            nsnr = [local_stats(dwin, ch, PAD)[1] for ch in chunks]  # per template
            row = dict(chirp_mass=p.chirp_mass, target_snr=target)
            row.update({f"t{k}": nsnr[k] for k in range(B)})
            rows.append(row)
            if (j + 1) % 10 == 0:
                progress("bank_inj", si * n_inj + j + 1, len(test_segs) * n_inj,
                         elapsed_s=time.time() - t_b, segment=float(si + 1))
        tmp = ipath.parent / (ipath.stem + ".tmp.parquet")
        pd.DataFrame(rows).to_parquet(tmp)
        os.replace(tmp, ipath)
        print(f"[inj] {gps}: wrote {ipath.name}", flush=True)

    # ---- stage 3: density sweep — fraction vs bank subset size
    df = pd.concat([pd.read_parquet(OUT / f"inj_{g}{tag}.parquet") for g in test_segs])
    tmat = np.stack([df[f"t{k}"].to_numpy() for k in range(B)], axis=1)  # (Ninj, B)
    thr_mat = np.stack([seg_thr[str(g)] for g in test_segs], axis=0)     # (Nseg, B)
    mc = 3 if args.smoke else 10

    print(f"\n=== bank-oracle gate | B={B}, {len(df)} inj, n=8 vetoed statistic ===")
    print(f"{'subset':>7}  {'thr':>6}   " + "  ".join(f"{m:>11}" for m in MASS_LABELS))
    sweep = {}
    for k, idx in subset_indices(B):
        thr = float(thr_mat[:, idx].max())          # zero-FA over noise, this subset
        df["_det"] = tmat[:, idx].max(axis=1) > thr  # detection statistic = bank-max
        frac = mass_bin_results(df, "_det", mc)["mf_distance_fraction"]
        sweep[k] = {"threshold": thr, "frac": frac}
        print(f"{k:>7}  {thr:>6.1f}   " + "  ".join(f"{frac[m]:>11.3f}" for m in MASS_LABELS))
    print(f"{'oracle*':>7}  {'':>6}   " + "  ".join(f"{ORACLE_VETOED[m]:>11.3f}" for m in MASS_LABELS)
          + "   (* true template, n=8 vetoed)")
    print(f"{'cnn_w64':>7}  {'':>6}   " + "  ".join(f"{CNN_W64[m]:>11.3f}" for m in MASS_LABELS))

    full = sweep[B]["frac"]
    beats = {m: full[m] > CNN_W64[m] for m in MASS_LABELS}
    verdict = ("GATE CLEARED: coarse bank beats cnn_w64 in "
               f"{sum(beats.values())}/3 bins -> build F1 (learned veto)"
               if sum(beats.values()) >= 2 else
               "GATE NOT CLEARED: mismatch-limited -> measure denser bank / reconsider")
    print(f"\n{verdict}")

    out = {"bank_size": B, "n_chunk": N_CHUNK, "n_injections": int(len(df)),
           "statistic": "max over bank of newSNR_n8; zero-FA over real noise",
           "bank_mc_range": [float(bank_mc.min()), float(bank_mc.max())],
           "sweep": {str(k): v for k, v in sweep.items()},
           "full_bank_fraction": full, "beats_cnn_w64": beats,
           "oracle_vetoed": ORACLE_VETOED, "cnn_w64": CNN_W64, "seed": SEED}
    (C.RESULTS_DIR / f"bank_oracle{tag}.json").write_text(json.dumps(out, indent=2))

    fig, ax = plt.subplots(figsize=(8, 5))
    ks = sorted(sweep)
    for m in MASS_LABELS:
        ax.plot(ks, [sweep[k]["frac"][m] for k in ks], "o-", label=m)
        ax.axhline(CNN_W64[m], ls=":", alpha=0.4)
        ax.axhline(ORACLE_VETOED[m], ls="--", alpha=0.4)
    ax.set_xscale("log"); ax.set_xlabel("bank size (templates)")
    ax.set_ylabel("fraction of ideal-MF distance")
    ax.set_title(f"F0 bank-mismatch gate (dotted=cnn_w64, dashed=true-oracle)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(C.RESULTS_DIR / f"bank_oracle{tag}.png", dpi=120)
    print(f"wrote bank_oracle{tag}.json + .png")


if __name__ == "__main__":
    main()
