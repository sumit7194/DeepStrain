"""Stage 1: pre-generate a pool of whitened waveforms for on-the-fly injection.

The slow part of injection is waveform generation (~0.4 s each); doing it once
upfront lets the training DataLoader inject by a cheap array-add. Each pool entry
stores the LAST 64 s of the whitened waveform (the part that lands in a 64-s
window) plus its FULL optimal SNR (snr_ref), so a training sample is

    window += (target_snr / snr_ref) * pool_waveform_last64s

matching the oracle's convention exactly (target = full-signal SNR).

SIMPLIFICATION (pre-registered): all waveforms are whitened with ONE representative
train-segment PSD. Good enough for training (the model learns morphology); EVAL
re-whitens per-segment for a fair comparison with the oracle / cnn_w64.

Run:  .venv/bin/python scripts/build_waveform_pool.py [--n-pool 2500]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pbh import config as C
from pbh.data import whiten_segment
from pbh.progress import progress
from pbh.waveforms import make_whitened_injection, sample_params

WIN = 64 * C.SAMPLE_RATE
POOL_DIR = C.DATA_DIR / "waveform_pool"
SEED = C.SEED + 2222


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pool", type=int, default=2500)
    args = ap.parse_args()
    POOL_DIR.mkdir(parents=True, exist_ok=True)
    xpath, mpath = POOL_DIR / "x_pool.npy", POOL_DIR / "pool_meta.parquet"
    if xpath.exists() and mpath.exists():
        x = np.load(xpath, mmap_mode="r")
        if x.shape == (args.n_pool, WIN):
            print(f"pool already built: {x.shape}")
            return

    manifest = json.loads((C.DATA_DIR / "manifest.json").read_text())
    rep_gps = manifest["H1"]["train"][0]  # one representative PSD (declared)
    _, t0, psd = whiten_segment("H1", rep_gps)
    print(f"representative PSD from segment {rep_gps}; generating {args.n_pool} waveforms")

    x = np.lib.format.open_memmap(xpath, mode="w+", dtype=np.float16,
                                  shape=(args.n_pool, WIN))
    rng = np.random.default_rng(SEED)
    rows, t_start = [], time.time()
    for i in range(args.n_pool):
        p = sample_params(rng)
        h_w, snr_ref = make_whitened_injection(p, "H1", t0, psd)
        x[i] = h_w[-WIN:].astype(np.float16)
        rows.append(dict(idx=i, mass1=p.mass1, mass2=p.mass2,
                         chirp_mass=p.chirp_mass, snr_ref=snr_ref))
        if (i + 1) % 50 == 0:
            progress("waveform_pool", i + 1, args.n_pool, elapsed_s=time.time() - t_start)
            print(f"  {i + 1}/{args.n_pool}", flush=True)
    x.flush()
    tmp = mpath.with_suffix(".tmp.parquet")
    pd.DataFrame(rows).to_parquet(tmp)
    os.replace(tmp, mpath)
    progress("waveform_pool", args.n_pool, args.n_pool, elapsed_s=time.time() - t_start)
    print(f"wrote {xpath.name} {x.shape} + {mpath.name} "
          f"({xpath.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
