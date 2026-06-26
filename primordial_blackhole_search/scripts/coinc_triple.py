"""N5 (PLAN.md): does adding VIRGO (H1×L1×V1 TRIPLE coincidence) beat the H1×L1 double-coincidence floor?

Extends coinc_eval (G1, +1.37× double-coincidence win) to a 3rd detector. cnn_w64 (H1-trained) scores 64-s
windows in H1, L1 AND V1 (transfer, as for L1); the triple statistic is sH1+sL1+sV1. Background via 3-way
time-slides — pair H1 window i with L1 i+lagL and V1 i+lagV (lagL,lagV≠0 → any real signal de-coincides) →
the accidental TRIPLE-coincidence distribution → a matched-FAR threshold (the n_livetime-th loudest slide, the
same false-alarm rate as single-det). Inject signals projected onto all 3 detectors (antenna + light-travel
delay); network SNR = √(snrH²+snrL²+snrV²). Compare single vs double (H1×L1) vs triple (H1×L1×V1) sensitive
distance at matched FAR. V1 is LESS sensitive than H1/L1, so the question is whether 3-way noise rejection
outweighs Virgo's weaker signal.

Run:  .venv/bin/python scripts/coinc_triple.py [--n-inj 2400] [--slides 2000] [--smoke]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from pbh import config as C
from pbh.data import whiten_segment
from pbh.models import make_model
from pbh.progress import progress
from pbh.spectrogram import spectrogram
from pbh.sweep import SweepGrid, pool_and_log, score_windows
from pbh.waveforms import make_whitened_injection, sample_params

GRID = SweepGrid.short(64)
WIN = GRID.win_samp                        # 262144 (64 s)
NBINS = GRID.n_time_bins                   # 63 — w64 grid (NOT the v1 default 256)
NET_SNR_RANGE = (4, 40)
SNR_BINS = np.linspace(*NET_SNR_RANGE, 10)
MASS_EDGES, MASS_LABELS = [0.17, 0.35, 0.55, 0.88], ["0.17-0.35", "0.35-0.55", "0.55-0.88"]
SEED = C.SEED
DETS = ("H1", "L1", "V1")


def bin_snr50(df, col, mc=10):
    out = {}
    for lo, hi, lab in zip(MASS_EDGES[:-1], MASS_EDGES[1:], MASS_LABELS):
        sub = df[(df.chirp_mass >= lo) & (df.chirp_mass < hi)]
        cen, eff = [], []
        for a, b in zip(SNR_BINS[:-1], SNR_BINS[1:]):
            s = sub[(sub.target_snr >= a) & (sub.target_snr < b)]
            if len(s) >= mc:
                cen.append((a + b) / 2); eff.append(float(s[col].mean()))
        snr50 = float(np.interp(0.5, eff, cen)) if len(cen) > 1 and max(eff) >= 0.5 else np.nan
        out[lab] = round(8.0 / snr50, 4) if np.isfinite(snr50) else 0.0
    return out


def score_wins(model, dev, wins):
    """List of (WIN,) whitened windows -> cnn_w64 scores (isolated-window spectrogram, w64 grid)."""
    feats = np.stack([pool_and_log(spectrogram(w), NBINS) for w in wins])
    return score_windows(model, dev, feats)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-inj", type=int, default=2400)
    ap.add_argument("--slides", type=int, default=2000)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = make_model("cnn"); model.load_state_dict(torch.load(C.MODEL_DIR / "cnn_w64.pt", map_location=dev))
    model.to(dev).eval()
    # fresh H1∩L1∩V1 triple-coincident segments (the manifest's H1∩L1 test segs are all Virgo
    # duty-cycle gaps; these are discovered by intersecting the 3 DATA flags, leakage-free vs cnn_w64).
    segs = json.loads((C.DATA_DIR / "triple_segs.json").read_text())
    if args.smoke:
        segs = segs[:2]
    n_inj_per_seg = (30 if args.smoke else args.n_inj) // max(1, len(segs))
    slides = 200 if args.smoke else args.slides
    print(f"N5 triple coincidence | {len(segs)} segs | {n_inj_per_seg}/seg inj | {slides} slides | {dev}", flush=True)

    # --- whiten H1,L1,V1; score every non-overlapping 64-s noise window ---
    # robust to partial fetches / Virgo gaps: keep only segments where ALL THREE detectors whiten cleanly
    data, noise = {}, {d: {} for d in DETS}
    crop = C.WHITEN_CROP_SEC * C.SAMPLE_RATE
    t0_ = time.time()
    usable = []
    for g in segs:
        tmp = {}
        try:
            for d in DETS:
                w, tgps, psd = whiten_segment(d, g)
                if not np.isfinite(w).all():
                    raise ValueError(f"{d} {g} non-finite")
                wc = w[crop:-crop]; nwin = (len(wc) - WIN) // WIN; starts = np.arange(nwin) * WIN
                tmp[d] = (wc, tgps, psd, starts)
        except Exception as e:
            print(f"  SKIP seg {g}: {type(e).__name__} {str(e)[:50]}", flush=True); continue
        for d in DETS:
            data[(d, g)] = tmp[d]
            noise[d][g] = score_wins(model, dev, [tmp[d][0][s:s + WIN] for s in tmp[d][3]])
        usable.append(g)
    segs = usable
    if len(segs) < 2:
        print(f"FATAL: only {len(segs)} fully-triple-coincident segments usable (need >=2). "
              f"GWOSC/Virgo data incomplete; rerun when more segments fetch.", flush=True)
        return
    print(f"  scored noise on {len(segs)} usable triple segs ({time.time()-t0_:.0f}s): {segs}", flush=True)
    n_inj_per_seg = (30 if args.smoke else args.n_inj) // len(segs)   # recompute on USABLE segs to hit target

    # --- thresholds: single (H1), double (H1×L1), triple (H1×L1×V1) at MATCHED single-det FAR ---
    rng = np.random.default_rng(SEED)
    thr_single = float(np.concatenate([noise["H1"][g] for g in segs]).max())
    bg2, bg3, n_live2, n_live3 = [], [], 0, 0
    for g in segs:
        h, l, v = (noise[d][g] for d in DETS); n = min(len(h), len(l), len(v)); h, l, v = h[:n], l[:n], v[:n]
        for lag in range(1, n):                                    # double slides
            bg2.append(h + np.roll(l, lag))
        n_live2 += n - 1
        for _ in range(slides):                                    # triple slides (random lag pairs)
            lagL, lagV = int(rng.integers(1, n)), int(rng.integers(1, n))
            bg3.append(h + np.roll(l, lagL) + np.roll(v, lagV))
        n_live3 += slides
    bg2, bg3 = np.concatenate(bg2), np.concatenate(bg3)
    thr2 = float(np.sort(bg2)[-n_live2]); thr3 = float(np.sort(bg3)[-n_live3])     # matched-FAR thresholds
    print(f"thr single {thr_single:.3f} | double {thr2:.3f} ({n_live2} live) | triple {thr3:.3f} ({n_live3} live)", flush=True)

    # --- coincident injections onto all 3 detectors at a range of network SNR ---
    # per-segment checkpoint (this run survived repeated power losses): each seg's injection rows are
    # seeded by gps and saved on completion, so a re-run resumes from the last finished segment.
    rows_path = C.RESULTS_DIR / f"coinc_triple_rows{'_smoke' if args.smoke else ''}.parquet"
    rows, done = [], set()
    if rows_path.exists():
        prev = pd.read_parquet(rows_path); prev = prev[prev.gps.isin(segs)]
        rows = prev.to_dict("records"); done = set(int(x) for x in prev.gps.unique())
        print(f"  resuming: {len(done)} seg(s) cached ({len(rows)} rows): {sorted(done)}", flush=True)
    tb = time.time()
    for si, g in enumerate(segs):
        if int(g) in done:
            print(f"  seg {si+1}/{len(segs)} ({g}) cached, skip", flush=True); continue
        wcs = {d: data[(d, g)][0] for d in DETS}; psds = {d: data[(d, g)][2] for d in DETS}
        starts = {d: data[(d, g)][3] for d in DETS}; tH = data[("H1", g)][1]
        rg = np.random.default_rng([SEED, int(g)]); nwin = min(len(starts[d]) for d in DETS)
        wins = {d: [] for d in DETS}; metas = []
        for _ in range(n_inj_per_seg):
            p = sample_params(rg); t_geo = tH + C.SEGMENT_LEN // 2
            hw, ref = {}, {}
            for d in DETS:
                hw[d], ref[d] = make_whitened_injection(p, d, t_geo, psds[d])
            net_ref = float(np.sqrt(sum(ref[d] ** 2 for d in DETS)))
            target = float(rg.uniform(*NET_SNR_RANGE)); scale = target / net_ref
            wi = int(rg.integers(0, nwin)); m = int(rg.integers(WIN // 2, WIN))
            for d in DETS:
                ww = wcs[d][starts[d][wi]:starts[d][wi] + WIN].copy()
                ww[:m] += (hw[d] * scale)[-WIN:][WIN - m:]
                wins[d].append(ww)
            metas.append((p.chirp_mass, target))
        sc = {d: score_wins(model, dev, wins[d]) for d in DETS}
        rows += [dict(gps=int(g), chirp_mass=mc, target_snr=t,
                      sH1=float(sc["H1"][i]), sL1=float(sc["L1"][i]), sV1=float(sc["V1"][i]))
                 for i, (mc, t) in enumerate(metas)]
        pd.DataFrame(rows).to_parquet(rows_path)                    # checkpoint after each segment
        progress("coinc_triple_inj", si + 1, len(segs), elapsed_s=time.time() - tb)
        print(f"  injected+scored seg {si+1}/{len(segs)} ({time.time()-tb:.0f}s, checkpointed)", flush=True)

    df = pd.DataFrame(rows)
    df["det_single"] = df.sH1 > thr_single
    df["det_double"] = (df.sH1 + df.sL1) > thr2
    df["det_triple"] = (df.sH1 + df.sL1 + df.sV1) > thr3
    fs, fd, ft = (bin_snr50(df, c, 3 if args.smoke else 10) for c in ("det_single", "det_double", "det_triple"))
    print(f"\n{'mass bin':>12} | {'single':>7} | {'double':>7} | {'triple':>7} | triple/double")
    for lab in MASS_LABELS:
        ratio = ft[lab] / fd[lab] if fd[lab] else float("nan")
        print(f"{lab:>12} | {fs[lab]:>7.3f} | {fd[lab]:>7.3f} | {ft[lab]:>7.3f} | {ratio:>6.2f}x")
    mean_s = np.mean(list(fs.values()))
    mean_d, mean_t = np.mean(list(fd.values())), np.mean(list(ft.values()))
    print(f"\nmean sensitive-distance fraction: single {mean_s:.3f} -> double {mean_d:.3f} -> triple {mean_t:.3f} "
          f"({mean_t/mean_d:.2f}x triple/double); Virgo {'HELPS' if mean_t > mean_d + 0.01 else 'does NOT clearly help'}")
    # WHY: per-detector signal responsiveness — mean score on loud (netSNR>25) minus faint (<10) injections.
    # V1 barely responds at subsolar masses, so it adds noise to the sum + raises the 3-way threshold rather
    # than contributing signal (a LEARNED triple statistic would also fail — no V1 signal to weight).
    loud, faint = df[df.target_snr > 25], df[df.target_snr < 10]
    resp = {d: float(loud[f"s{d}"].mean() - faint[f"s{d}"].mean()) for d in DETS}
    print(f"signal responsiveness (loud-faint mean score): H1 {resp['H1']:+.2f}  L1 {resp['L1']:+.2f}  V1 {resp['V1']:+.2f} "
          f"-> V1 {resp['V1']/np.mean([resp['H1'],resp['L1']]):.0%} of H1/L1 (too insensitive at subsolar)")
    (C.RESULTS_DIR / f"coinc_triple{'_smoke' if args.smoke else ''}.json").write_text(json.dumps(
        {"n_segs": len(segs), "single_frac": fs, "double_frac": fd, "triple_frac": ft,
         "mean_single": float(mean_s), "mean_double": float(mean_d), "mean_triple": float(mean_t),
         "double_over_single": float(mean_d / mean_s) if mean_s else None,
         "triple_over_double": float(mean_t / mean_d) if mean_d else None,
         "signal_responsiveness": resp, "virgo_helps": bool(mean_t > mean_d + 0.01)}, indent=2))
    print("wrote coinc_triple.json")


if __name__ == "__main__":
    main()
