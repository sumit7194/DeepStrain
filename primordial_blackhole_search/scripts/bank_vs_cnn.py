"""Follow-up A5: airtight CNN vs real-bank-MF — score cnn_w64 on the IDENTICAL bank_dense injections.

bank_dense compared its sensitive distance to the v1 gated cnn_w64 number (a DIFFERENT injection realization).
To make "the real MF beats the CNN" load-bearing, this regenerates bank_dense's deterministic injections
(same [SEED, gps] streams, same order) and scores cnn_w64 on the exact same 64-s windows, with cnn_w64's own
zero-FA threshold over the same 6 test segments. Removes the injection-realization caveat -> the head-to-head
is then same injections + same segments + same zero-FA-per-livetime FAR.

PRE-REGISTRATION for the ADEQUATE bank (2026-09-23, written before this was run at 0.05%): bank_adequacy found the
0.05% bank (B = 3,235, 9,000 injections) flat against its own 1,617 halving (ratios 1.01/0.97/0.98 < 1.10), so the
pre-registered rule makes a CNN-vs-MF verdict quotable at that density -- but only on IDENTICAL injections (the
merge's `beats_cnn` compares against hard-coded v1 numbers, the exact caveat Follow-up A's co-injection removed
when it shrank an apparent ~10% win to ~3%). Decision: MF BEATS the CNN iff the paired-bootstrap 90% CI of
mean(MF)/mean(CNN) sensitive-distance fraction EXCLUDES 1; otherwise TIE. Prediction: ratio 1.03-1.15, CI excluding
1 -- a modest MF win once the bank is adequate. Stated so it can be wrong.

Run:  .venv/bin/python scripts/bank_vs_cnn.py                                      # the original 0.1% comparison
      .venv/bin/python scripts/bank_vs_cnn.py --spacing 0.0005 --n-inj 1500        # the adequate bank
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


def fracs(mc, snr, det):
    """Sensitive-distance fraction 8/SNR50 per mass bin, from per-injection arrays (bank_dense's convention)."""
    out = {}
    for lo, hi, lab in zip(MASS_EDGES[:-1], MASS_EDGES[1:], MASS_LABELS):
        inb = (mc >= lo) & (mc < hi)
        cen, eff = [], []
        for a, b in zip(SNR_BINS[:-1], SNR_BINS[1:]):
            m = inb & (snr >= a) & (snr < b)
            if m.sum() >= 10:
                cen.append((a + b) / 2); eff.append(float(det[m].mean()))
        snr50 = float(np.interp(0.5, eff, cen)) if len(cen) > 1 and max(eff) >= 0.5 else np.nan
        out[lab] = round(8.0 / snr50, 4) if np.isfinite(snr50) else 0.0
    return out


def frac_table(df, thr):
    return fracs(df.chirp_mass.to_numpy(), df.target_snr.to_numpy(), df.score.to_numpy() > thr)


def mf_detections(segs, spacing):
    """Per-injection real-bank MF detection over the FULL bank at the global zero-FA threshold, in the same row
    order bank_dense wrote them (bank_dense.merge's rule: nan-aware max over templates and segments)."""
    B = len(bd.bank_mcs(spacing))
    thr = float(np.nanmax(np.stack([np.load(bd.OUT / f"thr_{g}_s{spacing}.npy") for g in segs])))
    dfs = [pd.read_parquet(bd.OUT / f"seg_{g}_s{spacing}.parquet") for g in segs]
    df = pd.concat(dfs, ignore_index=True)
    det = np.nanmax(df[[f"t{k}" for k in range(B)]].to_numpy(), axis=1) > thr
    return df.chirp_mass.to_numpy(), df.target_snr.to_numpy(), det, thr


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--spacing", type=float, default=0.001)
    ap.add_argument("--n-inj", type=int, default=250)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    tag = "" if args.spacing == 0.001 else f"_s{args.spacing}"
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
        for _ in range(args.n_inj):
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
        print(f"{gps}: scored {args.n_inj} injections (noise max {noise_max[-1]:.3f})", flush=True)

    df = pd.DataFrame(rows)
    thr = float(max(noise_max))                                # cnn_w64 zero-FA over the 6 test segments
    cnn_frac = frac_table(df, thr)

    bankd = json.loads((C.RESULTS_DIR / f"bank_dense{tag}.json").read_text())
    bank_frac = bankd["sweep"][str(bankd["B"])]["frac"]

    # ---- paired bootstrap: the SAME resampled injections score both methods -------------------------------
    mc_b, snr_b, det_b, _ = mf_detections(segs, args.spacing)
    mc_c, snr_c = df.chirp_mass.to_numpy(), df.target_snr.to_numpy()
    if not (len(mc_b) == len(mc_c) and np.allclose(mc_b, mc_c) and np.allclose(snr_b, snr_c)):
        raise SystemExit("regenerated injections do NOT match bank_dense's rows -- the comparison is not paired")
    if fracs(mc_b, snr_b, det_b) != bank_frac:
        raise SystemExit("per-injection MF detections do not reproduce bank_dense's merged fractions")
    det_c = df.score.to_numpy() > thr
    rb = np.random.default_rng(0)
    ratios = []
    for _ in range(args.n_boot):
        i = rb.integers(0, len(mc_c), len(mc_c))
        fb, fc = fracs(mc_b[i], snr_b[i], det_b[i]), fracs(mc_c[i], snr_c[i], det_c[i])
        mcn = np.mean(list(fc.values()))
        if mcn > 0:
            ratios.append(np.mean(list(fb.values())) / mcn)
    lo90, hi90 = (float(x) for x in np.percentile(ratios, [5, 95]))
    print(f"\n{'mass bin':>12} | {'CNN (same inj)':>14} | {'real-bank MF':>12} | MF/CNN")
    for lab in MASS_LABELS:
        r = bank_frac[lab] / cnn_frac[lab] if cnn_frac[lab] else float("nan")
        print(f"{lab:>12} | {cnn_frac[lab]:>14.3f} | {bank_frac[lab]:>12.3f} | {r:>5.2f}x")
    mc, mb = np.mean(list(cnn_frac.values())), np.mean(list(bank_frac.values()))
    print(f"{'mean':>12} | {mc:>14.3f} | {mb:>12.3f} | {mb/mc:>5.2f}x")

    beats = bool(lo90 > 1.0)
    (C.RESULTS_DIR / f"bank_vs_cnn{tag}.json").write_text(json.dumps(
        {"spacing": args.spacing, "cnn_thr": thr, "cnn_frac_same_inj": cnn_frac, "bank_frac": bank_frac,
         "cnn_mean": float(mc), "bank_mean": float(mb), "ratio": float(mb / mc),
         "boot90": [lo90, hi90], "n_boot": len(ratios), "mf_beats_cnn": beats,
         "verdict": "MF BEATS CNN (90% CI excludes 1)" if beats else "TIE (90% CI includes 1)",
         "n_inj": len(df)}, indent=2))
    print(f"\nmean MF/CNN {mb/mc:.3f}, paired-bootstrap 90% CI [{lo90:.3f}, {hi90:.3f}]  ->  "
          f"{'MF BEATS CNN' if beats else 'TIE'}")
    print(f"wrote bank_vs_cnn{tag}.json")


if __name__ == "__main__":
    main()
