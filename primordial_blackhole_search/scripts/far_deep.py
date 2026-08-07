"""Deep false-alarm-rate background: push the H1xL1 coincidence FAR far below Build C's 1/year.

WHY THIS IS COMPUTE, NOT A WALL: to claim "noise fakes this only once per N years" you must WATCH noise for
~N years. Global time-slides manufacture that: shift L1's whole score stream against H1's by a lag, and every
surviving coincidence is accidental by construction. With N_tot windows there are N_tot-1 distinct non-zero
circular lags, each an independent copy of the full livetime, so

    background_time = (N_tot - 1) x total_livetime        [verified: reproduces Build C's 1692 days exactly]

Both factors grow with the number of segments, so background scales as **N_segments^2**:
    24 segs -> 4.7 yr (Build C)   40 -> 13 yr   60 -> 29 yr   80 -> 52 yr   100 -> 82 yr

DESIGNED TO RUN FOR DAYS AND SURVIVE POWER LOSS:
  * The expensive work (fetch -> whiten -> score) is checkpointed PER SEGMENT as a tiny .npz of window
    scores (~63 floats/detector). A completed segment is never re-fetched or re-scored, ever.
  * Raw strain can be purged after scoring (--purge) so disk never becomes the limit — the scores are the
    scientific product; the segment list makes them re-derivable.
  * Interrupt at any point and re-run: it resumes from the cache and simply extends.
  * The background is recomputed from ALL cached segments each pass, so partial progress is always usable.

O4b data, so leakage-free by construction (cnn_w64 trained on O3a).

Run:  .venv/bin/python scripts/far_deep.py --target 60 [--purge]      (re-run anytime to extend)
      .venv/bin/python scripts/far_deep.py --report-only              (background from what's cached)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pbh import config as C
from pbh.data import fetch_segment, segment_path, whiten_segment
from pbh.models import make_model
from pbh.spectrogram import spectrogram
from pbh.sweep import SweepGrid, pool_and_log, score_windows

GRID = SweepGrid.short(64)
WIN, NBINS = GRID.win_samp, GRID.n_time_bins
CROP = C.WHITEN_CROP_SEC * C.SAMPLE_RATE
CACHE = C.RESULTS_DIR / "far_scores"          # per-segment score cache = the checkpoint
CACHE.mkdir(exist_ok=True, parents=True)
DETS = ("H1", "L1")


def score_segment(model, dev, gps):
    """Fetch (if needed) -> whiten -> score every non-overlapping 64-s window, both detectors."""
    out = {}
    for d in DETS:
        if not segment_path(d, gps).exists():
            fetch_segment(d, gps)
        w, _, _ = whiten_segment(d, gps)
        if not np.isfinite(w).all():
            raise ValueError(f"{d} {gps} non-finite")
        wc = w[CROP:-CROP]
        starts = np.arange((len(wc) - WIN) // WIN) * WIN
        feats = np.stack([pool_and_log(spectrogram(wc[s:s + WIN]), NBINS) for s in starts])
        out[d] = score_windows(model, dev, feats).astype(np.float32)
    n = min(len(out["H1"]), len(out["L1"]))
    return out["H1"][:n], out["L1"][:n]


def cached_segments():
    return sorted(int(p.stem.split("_")[1]) for p in CACHE.glob("seg_*.npz"))


def load_all():
    """Concatenate every cached segment's window scores into the global streams."""
    H, L, segs = [], [], []
    for g in cached_segments():
        z = np.load(CACHE / f"seg_{g}.npz")
        H.append(z["h"]); L.append(z["l"]); segs.append(g)
    if not H:
        return np.array([]), np.array([]), []
    return np.concatenate(H), np.concatenate(L), segs


def background_topk(sH, sL, keep=200_000):
    """Global time-slide background. N_tot-1 distinct non-zero circular lags (the honest count — using more
    repeats lags and re-injects zero-lag, which overcounts livetime). Keeps only the loudest `keep` values,
    so memory stays bounded as the background grows as N^2."""
    N = len(sH)
    top = np.full(keep, -np.inf, dtype=np.float32)
    t0 = time.time()
    for k in range(1, N):
        vals = sH + np.roll(sL, k)
        if vals.max() > top[0]:
            top = np.partition(np.concatenate([top, vals]), -keep)[-keep:]
            top.sort()
        if k % 500 == 0:
            print(f"    lag {k}/{N-1} ({time.time()-t0:.0f}s)", flush=True)
    return top, N - 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=60, help="total coincident segments to accumulate")
    ap.add_argument("--purge", action="store_true", help="delete raw strain after scoring (scores are the product)")
    ap.add_argument("--report-only", action="store_true", help="skip fetching; just recompute from the cache")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = make_model("cnn"); model.load_state_dict(torch.load(C.MODEL_DIR / "cnn_w64.pt", map_location=dev))
    model.to(dev).eval()

    have = cached_segments()
    print(f"cached segments: {len(have)} | target {args.target} | device {dev}", flush=True)

    if not args.report_only and len(have) < args.target:
        pool_f = C.DATA_DIR / "o4b_hl_coinc_pool.json"
        pool = json.loads(pool_f.read_text()) if pool_f.exists() else []
        if not pool:
            print(f"no candidate pool at {pool_f} — re-run the O4b H1nL1 discovery first"); return
        todo = [g for g in pool if g not in set(have)][: args.target - len(have)]
        print(f"fetching + scoring {len(todo)} new segments (each checkpointed on completion)\n", flush=True)
        for i, g in enumerate(todo):
            try:
                t0 = time.time()
                h, l = score_segment(model, dev, g)
                tmp = CACHE / f"seg_{g}.tmp.npz"
                np.savez(tmp, h=h, l=l); os.replace(tmp, CACHE / f"seg_{g}.npz")   # atomic
                if args.purge:
                    for d in DETS:
                        segment_path(d, g).unlink(missing_ok=True)
                    # gwpy's fetch_open_data(cache=True) ALSO stores each file in astropy's download
                    # cache — a second copy our purge previously missed (it grew ~0.25 GB/segment and
                    # would have exhausted disk before segment 100). Clear it too.
                    try:
                        from astropy.utils.data import clear_download_cache
                        clear_download_cache()
                    except Exception:
                        pass
                print(f"  [{len(have)+i+1}/{args.target}] {g}: {len(h)} windows "
                      f"({time.time()-t0:.0f}s){' purged' if args.purge else ''}", flush=True)
            except Exception as e:
                print(f"  {g}: SKIP {type(e).__name__}: {str(e)[:60]}", flush=True)

    sH, sL, segs = load_all()
    if len(sH) < 100:
        print(f"only {len(sH)} windows cached — need more segments before a background is meaningful"); return
    live_s = len(segs) * C.SEGMENT_LEN
    print(f"\ncomputing background from {len(segs)} segments, {len(sH)} windows "
          f"({live_s/3600:.1f} h real livetime)", flush=True)
    top, n_lags = background_topk(sH, sL)
    bg_s = n_lags * live_s
    bg_yr = bg_s / 3.156e7
    zero_lag = sH + sL

    # FAR ladder: the k-th loudest accidental coincidence defines the threshold at rate k/bg_time
    out = {"n_segments": len(segs), "n_windows": int(len(sH)), "real_livetime_h": live_s / 3600,
           "n_distinct_lags": int(n_lags), "background_years": bg_yr,
           "zero_lag_max": float(zero_lag.max()), "far_ladder": {}}
    print(f"\nbackground: {n_lags} distinct lags x {live_s/3600:.1f} h = **{bg_yr:.1f} years**")
    print(f"{'FAR':>14} {'threshold':>10}   (loudest accidental coincidence at that rate)")
    for label, per_year in (("1/month", 12), ("1/year", 1), ("1/decade", 0.1), ("1/century", 0.01)):
        n_expected = per_year * bg_yr           # how many bg events in the background at this FAR
        if n_expected < 1 or n_expected > len(top):
            print(f"{label:>14}   {'not reachable' if n_expected<1 else 'need larger top-k'}")
            continue
        thr = float(top[-int(round(n_expected))])
        out["far_ladder"][label] = thr
        print(f"{label:>14} {thr:>10.3f}")
    out["reach_far"] = min(out["far_ladder"], key=lambda k: {"1/month":12,"1/year":1,"1/decade":.1,"1/century":.01}[k]) if out["far_ladder"] else None
    print(f"\ndeepest FAR reached: {out['reach_far']}  (Build C reached 1/year at 4.6 yr background)")
    (C.RESULTS_DIR / "far_deep.json").write_text(json.dumps(out, indent=2))
    print("wrote far_deep.json")


if __name__ == "__main__":
    main()
