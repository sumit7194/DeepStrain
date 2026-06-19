"""Build C, step 2 (GPU VM): does the H1xL1 coincidence advantage hold at a REALISTIC
false-alarm rate? Sensitive distance vs FAR via GLOBAL time-slides.

Efficient design: ONE worker per coincident segment (parallel over the 8 cores) whitens
that segment ONCE, builds its noise-window + injection spectrograms (the CPU bottleneck),
and returns features; the main process batch-scores everything on the GPU. Background uses
K global slides -> effective livetime K x T_real, reaching 1/month..1/year.

FAR accounting: pool H1/L1 noise scores (time-aligned at zero lag); foreground = sH1[j]+sL1[j]
over T_real = N*64 s; background = sH1 + roll(sL1, k), k=1..K -> K*N coincidences over
T_bg = K*T_real; FAR(T) = #(bg>T)/T_bg. Single-det FAR floor = 1/T_real (no slides) -> a
single detector cannot reach 1/month from this data; coincidence can. That gap is the result.

Run:  python coinc_far.py [--n-inj 2400] [--slides 4000]
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
from pbh.spectrogram import spectrogram
from pbh.sweep import pool_and_log
from pbh.waveforms import make_whitened_injection, sample_params
from coinc_eval import NBINS, NET_SNR_RANGE, WIN, bin_snr50

CROP = C.WHITEN_CROP_SEC * C.SAMPLE_RATE
SEED = C.SEED + 9001
N_INJ_PER = 100  # set per run via the global below (kept picklable for workers)
FARS = {"1/6h": 1/(6*3600), "1/day": 1/86400, "1/week": 1/(7*86400),
        "1/month": 1/(30*86400), "1/year": 1/(365*86400)}


def _feat(w):
    return pool_and_log(spectrogram(w), NBINS)


def _segment_worker(args):
    """One coincident segment: whiten H1+L1 once, return noise + injection spectrogram
    features (CPU). All CPU-bound work for this segment lives here (parallel-friendly)."""
    g, n_inj, seed = args
    wH_full, tH, psdH = whiten_segment("H1", g)
    wL_full, tL, psdL = whiten_segment("L1", g)
    wcH, wcL = wH_full[CROP:-CROP], wL_full[CROP:-CROP]
    nwin = min(len(wcH), len(wcL)) // WIN
    noiseH = np.array([_feat(wcH[i*WIN:(i+1)*WIN]) for i in range(nwin)])
    noiseL = np.array([_feat(wcL[i*WIN:(i+1)*WIN]) for i in range(nwin)])
    rng = np.random.default_rng(seed)
    fH, fL, metas = [], [], []
    for _ in range(n_inj):
        p = sample_params(rng)
        hwH, refH = make_whitened_injection(p, "H1", tH + C.SEGMENT_LEN // 2, psdH)
        hwL, refL = make_whitened_injection(p, "L1", tL + C.SEGMENT_LEN // 2, psdL)
        net = float(np.hypot(refH, refL))
        target = float(rng.uniform(*NET_SNR_RANGE))
        scale = target / net
        wi = int(rng.integers(0, nwin)); m = int(rng.integers(WIN // 2, WIN))
        a = wcH[wi*WIN:wi*WIN+WIN].copy(); b = wcL[wi*WIN:wi*WIN+WIN].copy()
        a[:m] += (hwH*scale)[-WIN:][WIN-m:]; b[:m] += (hwL*scale)[-WIN:][WIN-m:]
        fH.append(_feat(a)); fL.append(_feat(b)); metas.append((float(p.chirp_mass), target))
    return g, noiseH, noiseL, np.array(fH), np.array(fL), metas


@torch.no_grad()
def gpu_score(model, device, feats):
    out = []
    for i in range(0, len(feats), 4096):
        b = torch.from_numpy(feats[i:i+4096]).float().unsqueeze(1).to(device)
        out.append(model(b).float().cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-inj", type=int, default=2400)
    ap.add_argument("--slides", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_model("cnn")
    model.load_state_dict(torch.load(C.MODEL_DIR / "cnn_w64.pt", map_location=device))
    model.to(device).eval()
    segs = json.loads((C.DATA_DIR / "manifest_far.json").read_text())["coinc"]
    per = max(1, args.n_inj // len(segs))
    print(f"device {device} | {len(segs)} segs | {per}/seg inj | {args.slides} slides | "
          f"{args.workers} workers", flush=True)

    # --- parallel: each segment whitens + builds all its spectrograms (CPU) ---
    t0 = time.time()
    jobs = [(g, per, SEED + i) for i, g in enumerate(segs)]
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(_segment_worker, jobs):
            results[r[0]] = r
            print(f"  segment {len(results)}/{len(segs)} done ({time.time()-t0:.0f}s)", flush=True)

    # --- batch GPU scoring; pool noise (time-aligned), collect injections ---
    sH1, sL1, rows = [], [], []
    for g in segs:
        _, nH, nL, fH, fL, metas = results[g]
        scH, scL = gpu_score(model, device, nH), gpu_score(model, device, nL)
        n = min(len(scH), len(scL)); sH1.append(scH[:n]); sL1.append(scL[:n])
        iH, iL = gpu_score(model, device, fH), gpu_score(model, device, fL)
        rows += [dict(chirp_mass=mc, target_snr=t, sH1=float(a), sL1=float(b))
                 for (mc, t), a, b in zip(metas, iH, iL)]
    sH1 = np.concatenate(sH1); sL1 = np.concatenate(sL1); N = len(sH1)
    T_real = N * 64.0
    df = pd.DataFrame(rows); df["coinc"] = df.sH1 + df.sL1
    print(f"\npooled {N} noise windows ({T_real/3600:.1f} h) | {len(df)} injections "
          f"({time.time()-t0:.0f}s total)", flush=True)

    # --- time-slide background (global slides) ---
    bg = np.concatenate([sH1 + np.roll(sL1, k) for k in range(1, args.slides + 1)])
    bg.sort()
    T_bg = args.slides * T_real
    print(f"background {len(bg):,} coincidences over {T_bg/86400:.0f} days "
          f"(FAR floor 1/{T_bg/86400:.0f}d)", flush=True)

    def coinc_thr(far):
        k = max(1, int(round(far * T_bg)))
        return float(bg[-k])
    thr_single = float(sH1.max())

    ML = ("0.17-0.35", "0.35-0.55", "0.55-0.88")
    print(f"\n=== coincident sensitive distance vs FAR ({len(df)} inj) ===")
    print(f"{'FAR':>8} | {'thr':>7} | " + "  ".join(f"{m:>10}" for m in ML))
    out_rows = {}
    for name, far in FARS.items():
        if far * T_bg < 1:
            print(f"{name:>8} | (below background floor)"); continue
        thr = coinc_thr(far); df["_d"] = df.coinc > thr
        frac = bin_snr50(df, "_d", 10)
        out_rows[name] = {"thr": thr, "frac": {m: frac[m][1] for m in frac}}
        print(f"{name:>8} | {thr:>7.3f} | " + "  ".join(f"{frac[m][1]:>10.3f}" for m in ML))
    df["_s"] = df.sH1 > thr_single
    sd = bin_snr50(df, "_s", 10)
    print(f"single-det @ FAR floor 1/{T_real/3600:.0f}h: " +
          "  ".join(f"{sd[m][1]:.3f}" for m in ML) + "  (cannot go lower from this data)")

    res = {"n_segments": len(segs), "real_livetime_h": T_real/3600, "slides": args.slides,
           "bg_days": T_bg/86400, "n_injections": int(len(df)),
           "coinc_vs_far": out_rows, "single_det_floor_frac": {m: sd[m][1] for m in ML},
           "single_det_far": f"1/{T_real/3600:.0f}h"}
    (C.RESULTS_DIR / "coinc_far.json").write_text(json.dumps(res, indent=2))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    names = list(out_rows)
    for m in ML:
        ax.plot([FARS[n]*86400 for n in names], [out_rows[n]["frac"][m] for n in names], "o-", label=m)
    for m, c in zip(ML, ("tab:blue", "tab:orange", "tab:green")):
        ax.axhline(sd[m][1], ls=":", color=c, alpha=0.5)
    ax.set_xscale("log"); ax.set_xlabel("false-alarm rate [per day]  (left = stricter)")
    ax.set_ylabel("sensitive distance (fraction of ideal MF)")
    ax.set_title(f"Build C: H1xL1 coincidence sensitivity vs FAR\n"
                 f"({len(segs)} segs, {T_bg/86400:.0f}d background; dotted = single-det floor @ 1/{T_real/3600:.0f}h)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(C.RESULTS_DIR / "coinc_far.png", dpi=140)
    print(f"\nwrote coinc_far.json + coinc_far.png")


if __name__ == "__main__":
    main()
