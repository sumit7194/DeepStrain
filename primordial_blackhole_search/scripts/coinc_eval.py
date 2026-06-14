"""Path G milestone G1: does H1xL1 coincidence beat the single-detector noise floor?

Per-detector statistic = cnn_w64 score on 64-s windows (our best single-detector
detector, 0.41-0.48 single-det). Coincidence rides on it:
  - score every non-overlapping 64-s noise window in H1 and L1 (5 coincident
    TEST segments);
  - BACKGROUND via TIME-SLIDES: pair H1 window i with L1 window i+lag (lag!=0,
    so any real signal de-coincides) -> accidental-coincidence distribution ->
    a real false-alarm rate from little data;
  - coincident statistic = scoreH1 + scoreL1; zero-FA threshold = loudest slide;
  - INJECT coincident signals (proper antenna + light-travel delay) at a range of
    NETWORK SNR; an injection is detected (single) if scoreH1 > H1 zero-FA, or
    (coinc) if scoreH1+scoreL1 > slide zero-FA. Compare sensitive distance.

The question: does requiring two-detector agreement push cnn_w64 past its
single-detector wall by collapsing the noise floor? Same data, same FAR style.

Run:  .venv/bin/python scripts/coinc_eval.py [--smoke]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pbh import config as C
from pbh.data import whiten_segment
from pbh.metrics import MASS_EDGES, MASS_LABELS
from pbh.models import make_model
from pbh.progress import progress
from pbh.spectrogram import spectrogram
from pbh.sweep import HOP_SAMP, SweepGrid, pool_and_log, score_windows
from pbh.waveforms import make_whitened_injection, sample_params

GRID = SweepGrid.short(64)
WIN = GRID.win_samp                       # 262144 (64 s)
NBINS = GRID.n_time_bins                  # 63
SEED = C.SEED + 4242
NET_SNR_RANGE = (4.0, 40.0)               # NETWORK SNR (wider than per-det range so
SNR_BINS = np.linspace(*NET_SNR_RANGE, 13)  # single-det's SNR50 is on-scale here)


def bin_snr50(df, col, min_count):
    """Per mass bin: (SNR50 in network SNR, sensitive-distance fraction 8/SNR50)."""
    out = {}
    for lo_m, hi_m, lab in zip(MASS_EDGES[:-1], MASS_EDGES[1:], MASS_LABELS):
        sub = df[(df.chirp_mass >= lo_m) & (df.chirp_mass < hi_m)]
        cen, eff = [], []
        for blo, bhi in zip(SNR_BINS[:-1], SNR_BINS[1:]):
            s = sub[(sub.target_snr >= blo) & (sub.target_snr < bhi)]
            if len(s) >= min_count:
                cen.append((blo + bhi) / 2); eff.append(float(s[col].mean()))
        snr50 = float(np.interp(0.5, eff, cen)) if len(cen) > 1 and max(eff) >= 0.5 else np.nan
        out[lab] = (snr50, 8.0 / snr50 if np.isfinite(snr50) else 0.0)
    return out


def score_wins(model, device, wins):
    """List of (WIN,) whitened windows -> cnn_w64 scores (isolated-window
    spectrogram, exactly as the w64 shards were built/trained)."""
    feats = np.stack([pool_and_log(spectrogram(w), NBINS) for w in wins])
    return score_windows(model, device, feats)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    n_inj_per_seg = 30 if args.smoke else 300

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = make_model("cnn")
    model.load_state_dict(torch.load(C.MODEL_DIR / "cnn_w64.pt", map_location=device))
    model.to(device).eval()

    manifest = json.loads((C.DATA_DIR / "manifest.json").read_text())
    segs = [g for g in manifest["L1"]["coinc"] if g in manifest["H1"]["test"]]
    print(f"coincident TEST segments: {segs}")
    from pycbc.detector import Detector

    # ---- whiten both detectors; precompute noise-window scores + window starts
    data, noise = {}, {"H1": {}, "L1": {}}
    t0_ = time.time()
    for ifo in ("H1", "L1"):
        for g in segs:
            w, tgps, psd = whiten_segment(ifo, g)
            wc = w[C.WHITEN_CROP_SEC * C.SAMPLE_RATE : -C.WHITEN_CROP_SEC * C.SAMPLE_RATE]
            nwin = (len(wc) - WIN) // WIN
            starts = np.arange(nwin) * WIN
            data[(ifo, g)] = (wc, tgps, psd, starts)
            noise[ifo][g] = score_wins(model, device, [wc[s : s + WIN] for s in starts])
            progress("coinc_noise", (0 if ifo == "H1" else len(segs)) + segs.index(g) + 1,
                     2 * len(segs), elapsed_s=time.time() - t0_)
        print(f"  scored noise {ifo} ({time.time()-t0_:.0f}s)", flush=True)

    # ---- thresholds: single-det zero-FA, and coincident zero-FA via time-slides
    thr_single = float(np.concatenate([noise["H1"][g] for g in segs]).max())
    bg, n_zerolag = [], []
    for g in segs:
        h, l = noise["H1"][g], noise["L1"][g]
        n = min(len(h), len(l))
        h, l = h[:n], l[:n]
        n_zerolag.append(h + l)
        for lag in range(1, n):           # every non-zero slide = pure background
            bg.append(h + np.roll(l, lag))
    bg = np.concatenate(bg)
    zerolag = np.concatenate(n_zerolag)
    thr_coinc = float(bg.max())           # loudest accidental coincidence (very low FAR)
    # matched-FAR: single-det threshold = 0 FA in 1 livetime; the slide background
    # spans n_lag_total independent livetimes, so the n_lag_total-th loudest slide
    # is the coincident threshold at the SAME false-alarm rate as single-det.
    n_lag_total = sum(min(len(noise["H1"][g]), len(noise["L1"][g])) - 1 for g in segs)
    thr_coinc_matched = float(np.sort(bg)[-n_lag_total])
    n_bg_eff = len(bg)
    print(f"thr_single(H1 max) {thr_single:.3f} | thr_coinc(slide max, low FAR) {thr_coinc:.3f} "
          f"| thr_coinc(matched FAR) {thr_coinc_matched:.3f}")
    print(f"  {n_bg_eff} background coincidences over ~{n_lag_total} livetimes | "
          f"zero-lag max {zerolag.max():.3f}")

    # ---- coincident injections at a range of NETWORK SNR
    rows, t_b = [], time.time()
    for si, g in enumerate(segs):
        (wcH, tH, psdH, startsH) = data[("H1", g)]
        (wcL, tL, psdL, startsL) = data[("L1", g)]
        rng = np.random.default_rng([SEED, int(g)])
        nwin = min(len(startsH), len(startsL))
        winsH, winsL, metas = [], [], []
        for _ in range(n_inj_per_seg):
            p = sample_params(rng)
            t_geo = tH + C.SEGMENT_LEN // 2  # geocentric ref (cancels in network SNR)
            hwH, refH = make_whitened_injection(p, "H1", t_geo, psdH)
            hwL, refL = make_whitened_injection(p, "L1", t_geo, psdL)
            net_ref = float(np.hypot(refH, refL))
            target = float(rng.uniform(*NET_SNR_RANGE))  # NETWORK SNR
            scale = target / net_ref
            wi = int(rng.integers(0, nwin))
            m = int(rng.integers(WIN // 2, WIN))            # merger position in window
            wH = wcH[startsH[wi] : startsH[wi] + WIN].copy()
            wL = wcL[startsL[wi] : startsL[wi] + WIN].copy()
            wH[:m] += (hwH * scale)[-WIN:][WIN - m :]
            wL[:m] += (hwL * scale)[-WIN:][WIN - m :]
            winsH.append(wH); winsL.append(wL)
            metas.append((p.chirp_mass, target))
        sH = score_wins(model, device, winsH)
        sL = score_wins(model, device, winsL)
        rows += [dict(chirp_mass=mc, target_snr=t, sH1=float(a), sL1=float(b))
                 for (mc, t), a, b in zip(metas, sH, sL)]
        progress("coinc_inj", si + 1, len(segs), elapsed_s=time.time() - t_b,
                 n_injections=float(len(rows)))
        print(f"  injected+scored seg {si+1}/{len(segs)} ({time.time()-t_b:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df["det_single"] = df.sH1 > thr_single
    df["det_coinc"] = (df.sH1 + df.sL1) > thr_coinc
    df["det_coinc_matched"] = (df.sH1 + df.sL1) > thr_coinc_matched
    tag = "_smoke" if args.smoke else ""
    df.to_parquet(C.RESULTS_DIR / f"coinc_inj{tag}.parquet")  # raw scores -> free re-binning
    mc = 3 if args.smoke else 10
    rs = bin_snr50(df, "det_single", mc)
    rc = bin_snr50(df, "det_coinc", mc)
    rcm = bin_snr50(df, "det_coinc_matched", mc)

    print(f"\n=== G1 coincidence vs single-detector (cnn_w64) — SNR50 / distance-fraction ===")
    print(f"  matched-FAR = both methods at the SAME false-alarm rate (the fair comparison)")
    print(f"{'mass bin':>12}  {'single-det':>16}  {'coinc(matchFAR)':>18}  {'dist gain':>9}")
    for mlab in MASS_LABELS:
        (s50, sf), (c50, cf) = rs[mlab], rcm[mlab]
        gain = f"{s50/c50:.2f}x" if (np.isfinite(s50) and np.isfinite(c50) and c50 > 0) else \
               ("inf" if cf > 0 else "-")
        sstr = f"SNR50={s50:4.1f} f={sf:.3f}" if np.isfinite(s50) else "SNR50>40  f=0.000"
        cstr = f"SNR50={c50:4.1f} f={cf:.3f}" if np.isfinite(c50) else "SNR50>40  f=0.000"
        print(f"{mlab:>12}  {sstr:>16}  {cstr:>18}  {gain:>9}")
    res_single = {m: rs[m][1] for m in MASS_LABELS}
    res_coinc = {m: rc[m][1] for m in MASS_LABELS}
    res_coinc_m = {m: rcm[m][1] for m in MASS_LABELS}

    out = {"segments": segs, "n_injections": int(len(df)),
           "thr_single": thr_single, "thr_coinc_lowFAR": thr_coinc,
           "thr_coinc_matchedFAR": thr_coinc_matched,
           "n_background_coincidences": int(n_bg_eff), "n_livetimes": int(n_lag_total),
           "zerolag_max": float(zerolag.max()),
           "statistic": "coinc = sH1+sL1; background = time-slides; SNR axis = network",
           "single_det_fraction": res_single, "coinc_fraction_lowFAR": res_coinc,
           "coinc_fraction_matchedFAR": res_coinc_m, "seed": SEED}
    (C.RESULTS_DIR / f"coinc_eval{tag}.json").write_text(json.dumps(out, indent=2))

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(MASS_LABELS))
    ax.bar(x - 0.2, [res_single[m] for m in MASS_LABELS], 0.4, label="single-det H1", color="tab:gray")
    ax.bar(x + 0.2, [res_coinc_m[m] for m in MASS_LABELS], 0.4, label="H1xL1 coinc (matched FAR)", color="tab:green")
    ax.set_xticks(x, MASS_LABELS); ax.set_ylabel("fraction of ideal-MF distance (network SNR)")
    ax.set_title(f"G1: coincidence vs single-detector ({len(df)} inj, time-slide FAR)")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(C.RESULTS_DIR / f"coinc_eval{tag}.png", dpi=120)
    print(f"\nwrote coinc_eval{tag}.json + .png")


if __name__ == "__main__":
    main()
