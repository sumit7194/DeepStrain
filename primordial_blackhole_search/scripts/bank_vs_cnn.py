"""Follow-up A5: airtight CNN vs real-bank-MF — score cnn_w64 on the IDENTICAL bank_dense injections.

bank_dense compared its sensitive distance to the v1 gated cnn_w64 number (a DIFFERENT injection realization).
To make "the real MF beats the CNN" load-bearing, this regenerates bank_dense's deterministic injections
(same [SEED, gps] streams, same order) and scores cnn_w64 on the exact same 64-s windows, with cnn_w64's own
zero-FA threshold over the same 6 test segments. Removes the injection-realization caveat -> the head-to-head
is then same injections + same segments + same zero-FA-per-livetime FAR.

Run:  .venv/bin/python scripts/bank_vs_cnn.py
"""
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pbh import config as C
from pbh.data import whiten_segment
from pbh.models import make_model
from pbh.spectrogram import spectrogram
from pbh.sweep import SweepGrid, pool_and_log, score_windows, segment_window_scores
from pbh.waveforms import make_whitened_injection, sample_params

HERE = Path(__file__).resolve().parent
_bd = importlib.util.spec_from_file_location("bd", HERE / "bank_dense.py")
bd = importlib.util.module_from_spec(_bd); _bd.loader.exec_module(bd)

WIN, PAD, CROP = bd.WIN, bd.PAD, bd.CROP
NBINS = SweepGrid.short(64).n_time_bins
SNR_BINS = np.linspace(*C.EVAL_SNR_RANGE, 13)
MASS_EDGES, MASS_LABELS = bd.MASS_EDGES, bd.MASS_LABELS


def frac_table(df, thr):
    det = df.score.to_numpy() > thr
    out = {}
    for lo, hi, lab in zip(MASS_EDGES[:-1], MASS_EDGES[1:], MASS_LABELS):
        sub = df[(df.chirp_mass >= lo) & (df.chirp_mass < hi)]
        cen, eff = [], []
        for a, b in zip(SNR_BINS[:-1], SNR_BINS[1:]):
            m = (sub.target_snr >= a) & (sub.target_snr < b)
            if m.sum() >= 10:
                cen.append((a + b) / 2); eff.append(float(det[sub.index][m].mean()))
        snr50 = float(np.interp(0.5, eff, cen)) if len(cen) > 1 and max(eff) >= 0.5 else np.nan
        out[lab] = round(8.0 / snr50, 4) if np.isfinite(snr50) else 0.0
    return out


def main() -> None:
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = make_model("cnn"); model.load_state_dict(torch.load(C.MODEL_DIR / "cnn_w64.pt", map_location=dev))
    model.to(dev).eval()
    segs = json.loads((C.DATA_DIR / "manifest.json").read_text())["H1"]["test"]

    rows, noise_max = [], []
    for gps in segs:
        w, t0, psd = whiten_segment("H1", gps)
        wc = w[CROP:-CROP]
        noise_max.append(float(segment_window_scores(model, dev, w).max()))   # cnn zero-FA contribution
        # regenerate bank_dense's injections EXACTLY (same rng stream + order)
        rng = np.random.default_rng([bd.SEED, int(gps)])
        wins, meta = [], []
        for _ in range(250):
            p = sample_params(rng)
            h_w, snr_ref = make_whitened_injection(p, "H1", t0, psd)
            target = float(rng.uniform(*C.EVAL_SNR_RANGE))
            sig = h_w * (target / snr_ref)
            m = int(rng.integers(max(len(sig), WIN) + PAD, len(wc) - PAD))
            win64 = wc[m - WIN : m].copy()                     # the injected 64-s window (merger at end)
            win64 += sig[-WIN:] if len(sig) >= WIN else np.pad(sig, (WIN - len(sig), 0))
            wins.append(pool_and_log(spectrogram(win64), NBINS)); meta.append((p.chirp_mass, target))
        sc = score_windows(model, dev, np.stack(wins))
        rows += [dict(chirp_mass=mc, target_snr=t, score=float(s)) for (mc, t), s in zip(meta, sc)]
        print(f"{gps}: scored 250 injections (noise max {noise_max[-1]:.3f})", flush=True)

    df = pd.DataFrame(rows)
    thr = float(max(noise_max))                                # cnn_w64 zero-FA over the 6 test segments
    cnn_frac = frac_table(df, thr)

    bankd = json.loads((C.RESULTS_DIR / "bank_dense.json").read_text())
    bank_frac = bankd["sweep"][str(bankd["B"])]["frac"]
    print(f"\n{'mass bin':>12} | {'CNN (same inj)':>14} | {'real-bank MF':>12} | MF/CNN")
    for lab in MASS_LABELS:
        r = bank_frac[lab] / cnn_frac[lab] if cnn_frac[lab] else float("nan")
        print(f"{lab:>12} | {cnn_frac[lab]:>14.3f} | {bank_frac[lab]:>12.3f} | {r:>5.2f}x")
    mc, mb = np.mean(list(cnn_frac.values())), np.mean(list(bank_frac.values()))
    print(f"{'mean':>12} | {mc:>14.3f} | {mb:>12.3f} | {mb/mc:>5.2f}x")

    (C.RESULTS_DIR / "bank_vs_cnn.json").write_text(json.dumps(
        {"cnn_thr": thr, "cnn_frac_same_inj": cnn_frac, "bank_frac": bank_frac,
         "cnn_mean": float(mc), "bank_mean": float(mb), "mf_beats_cnn": bool(mb > mc),
         "n_inj": len(df)}, indent=2))
    print(f"\nMF beats CNN on identical injections: {mb > mc} ({mb:.3f} vs {mc:.3f})")
    print("wrote bank_vs_cnn.json")


if __name__ == "__main__":
    main()
