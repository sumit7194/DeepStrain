"""Build C, step 2 (GPU VM): does the H1xL1 coincidence advantage hold at a REALISTIC
false-alarm rate?

The local result (+1.37x) was at a modest FAR (~1/6 h). Here, with ~24 coincident
segments (~26 h coincident livetime) + GLOBAL time-slides, the effective background
livetime is K x 26 h (K slides) -> we can set the coincident threshold down to
~1/month and below, and measure the sensitive distance there.

FAR accounting (the load-bearing part):
  - pool all H1 / L1 noise window scores (aligned by physical time at zero lag);
  - zero-lag foreground = sH1[j]+sL1[j] over the real livetime T_real = N*64 s;
  - background = K global slides: sH1 + roll(sL1, k), k=1..K -> K*N coincidences over
    T_bg = K * T_real;  FAR(T) = #(bg > T) / T_bg.
  - single-detector FAR floor = 1/T_real (no slides possible) -> can't reach 1/month
    from this data at all; coincidence can. That gap IS the result.

Run:  python coinc_far.py [--n-inj 3000] [--slides 2000]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
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
from pbh.models import make_model
from pbh.waveforms import make_whitened_injection, sample_params
from coinc_eval import GRID, NBINS, NET_SNR_RANGE, SNR_BINS, WIN, bin_snr50, score_wins
from pbh.spectrogram import spectrogram
from pbh.sweep import pool_and_log

CROP = C.WHITEN_CROP_SEC * C.SAMPLE_RATE
WHITE_DIR = C.DATA_DIR / "whitened_far"
SEED = C.SEED + 9001
FARS = {"1/6h": 1/(6*3600), "1/day": 1/86400, "1/week": 1/(7*86400),
        "1/month": 1/(30*86400), "1/year": 1/(365*86400)}


def _whiten_one(args):
    ifo, gps = args
    out = WHITE_DIR / f"{ifo}_{gps}.npy"
    if out.exists():
        return str(out)
    w, _, _ = whiten_segment(ifo, gps)
    np.save(out, w[CROP:-CROP].astype(np.float32))
    return str(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-inj", type=int, default=3000)
    ap.add_argument("--slides", type=int, default=2000)
    args = ap.parse_args()
    WHITE_DIR.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_model("cnn")
    model.load_state_dict(torch.load(C.MODEL_DIR / "cnn_w64.pt", map_location=device))
    model.to(device).eval()

    segs = json.loads((C.DATA_DIR / "manifest_far.json").read_text())["coinc"]
    print(f"device {device} | {len(segs)} coincident segments | {args.slides} slides", flush=True)

    # --- whiten all (parallel, cached) ---
    t0 = time.time()
    jobs = [(ifo, g) for g in segs for ifo in ("H1", "L1")]
    with ProcessPoolExecutor(max_workers=8) as ex:
        list(ex.map(_whiten_one, jobs))
    print(f"whitened {len(jobs)} segs ({time.time()-t0:.0f}s)", flush=True)

    # --- per-segment noise window scores (pooled, time-aligned across detectors) ---
    H1n, L1n, wc_cache, psd_cache = [], [], {}, {}
    for g in segs:
        for ifo, pool in (("H1", H1n), ("L1", L1n)):
            wc = np.load(WHITE_DIR / f"{ifo}_{g}.npy")
            nwin = (len(wc) - WIN) // WIN
            wins = [wc[i*WIN:(i+1)*WIN] for i in range(nwin)]
            pool.append(score_wins(model, device, wins))
        # keep H1/L1 cropped arrays + psd for injections (re-whiten once for psd)
    nwin_per = [min(len(h), len(l)) for h, l in zip(H1n, L1n)]
    sH1 = np.concatenate([h[:n] for h, n in zip(H1n, nwin_per)])
    sL1 = np.concatenate([l[:n] for l, n in zip(L1n, nwin_per)])
    N = len(sH1)
    T_real = N * 64.0
    print(f"pooled {N} coincident noise windows ({T_real/3600:.1f} h real livetime)", flush=True)

    # --- time-slide background (global slides) ---
    coinc_zl = sH1 + sL1                          # zero-lag foreground
    bg = np.concatenate([sH1 + np.roll(sL1, k) for k in range(1, args.slides + 1)])
    T_bg = args.slides * T_real
    print(f"background: {len(bg):,} coincidences over {T_bg/86400:.0f} days "
          f"(FAR floor 1/{T_bg/86400:.0f}d)", flush=True)

    def coinc_thr(far):                           # threshold at a target coincident FAR
        k = far * T_bg                            # expected background events above thr
        return float(np.sort(bg)[-max(1, int(round(k)))])
    thr_single = float(sH1.max())                 # single-det FAR floor = 1/T_real

    # --- coincident injections (proper geometry), scored once, thresholded per FAR ---
    from pycbc.detector import Detector
    rows, t_b = [], time.time()
    per_seg = max(1, args.n_inj // len(segs))
    for g in segs:
        wcH = np.load(WHITE_DIR / f"H1_{g}.npy"); wcL = np.load(WHITE_DIR / f"L1_{g}.npy")
        _, tH, psdH = whiten_segment("H1", g); _, tL, psdL = whiten_segment("L1", g)
        rng = np.random.default_rng([SEED, int(g)])
        nwin = min(len(wcH), len(wcL)) // WIN
        winsH, winsL, metas = [], [], []
        for _ in range(per_seg):
            p = sample_params(rng)
            hwH, refH = make_whitened_injection(p, "H1", tH + C.SEGMENT_LEN//2, psdH)
            hwL, refL = make_whitened_injection(p, "L1", tL + C.SEGMENT_LEN//2, psdL)
            net_ref = float(np.hypot(refH, refL))
            target = float(rng.uniform(*NET_SNR_RANGE))
            scale = target / net_ref
            wi = int(rng.integers(0, nwin)); m = int(rng.integers(WIN//2, WIN))
            wH = wcH[wi*WIN:wi*WIN+WIN].copy(); wL = wcL[wi*WIN:wi*WIN+WIN].copy()
            wH[:m] += (hwH*scale)[-WIN:][WIN-m:]; wL[:m] += (hwL*scale)[-WIN:][WIN-m:]
            winsH.append(wH); winsL.append(wL); metas.append((p.chirp_mass, target))
        siH = score_wins(model, device, winsH); siL = score_wins(model, device, winsL)
        rows += [dict(chirp_mass=mc, target_snr=t, sH1=float(a), sL1=float(b))
                 for (mc, t), a, b in zip(metas, siH, siL)]
        print(f"  injected seg {segs.index(g)+1}/{len(segs)} ({time.time()-t_b:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df["coinc"] = df.sH1 + df.sL1

    # --- sensitive distance vs FAR ---
    print(f"\n=== sensitive distance vs FAR ({len(df)} inj) ===")
    print(f"{'FAR':>8} | {'coinc thr':>9} | " + " ".join(f"{m:>10}" for m in
          ("0.17-0.35","0.35-0.55","0.55-0.88")))
    out_rows = {}
    for name, far in FARS.items():
        if far * T_bg < 1:          # not enough background to resolve this FAR
            print(f"{name:>8} | (FAR below background floor — skipped)"); continue
        thr = coinc_thr(far)
        df["_d"] = df.coinc > thr
        frac = bin_snr50(df, "_d", 10)
        out_rows[name] = {"thr": thr, "frac": {m: frac[m][1] for m in frac}}
        print(f"{name:>8} | {thr:>9.3f} | " + " ".join(f"{frac[m][1]:>10.3f}" for m in frac))
    df["_s"] = df.sH1 > thr_single
    sd = bin_snr50(df, "_s", 10)
    print(f"single-det @ FAR floor 1/{T_real/3600:.0f}h: " +
          " ".join(f"{sd[m][1]:.3f}" for m in sd) + "  (cannot reach lower FAR from this data)")

    res = {"n_segments": len(segs), "real_livetime_h": T_real/3600, "slides": args.slides,
           "bg_days": T_bg/86400, "n_injections": int(len(df)),
           "coinc_vs_far": out_rows, "single_det_floor": {m: sd[m][1] for m in sd},
           "single_det_far": f"1/{T_real/3600:.0f}h"}
    (C.RESULTS_DIR / "coinc_far.json").write_text(json.dumps(res, indent=2))

    ML = ("0.17-0.35","0.35-0.55","0.55-0.88")
    fig, ax = plt.subplots(figsize=(9,5.5))
    fars = [FARS[n] for n in out_rows]
    for m in ML:
        ax.plot([f*86400 for f in fars], [out_rows[n]["frac"][m] for n in out_rows], "o-", label=m)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xscale("log"); ax.set_xlabel("false-alarm rate [per day]")
    ax.set_ylabel("coincident sensitive distance (fraction of ideal MF)")
    ax.set_title(f"Build C: H1xL1 coincidence sensitivity vs FAR ({T_bg/86400:.0f}d background)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(C.RESULTS_DIR / "coinc_far.png", dpi=140)
    print(f"\nwrote coinc_far.json + coinc_far.png")


if __name__ == "__main__":
    main()
