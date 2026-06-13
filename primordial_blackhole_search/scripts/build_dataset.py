"""Phase 1c: build spectrogram training shards from cached noise + injections.

For each segment: whiten once, then draw windows.
  positives: subsolar injection, masses log-uniform [0.2, 1.0] Msun, optimal
             SNR uniform in TRAIN_SNR_RANGE, merger placed so the window sees
             a meaningful piece of track (in-window SNR >= 4).
  negatives: the same real-noise windows untouched (glitches included).

Output per split: x_{split}.npy float16 memmap (N, 128, n_time), meta_{split}.parquet.

--window-sec selects the window length: the default (C.WINDOW_SEC) reproduces v1
into data/shards/ with 256 time bins; a shorter window (rung 2) writes to
data/shards_w{W}/ with its own natural bin count (e.g. 64 s -> 63 bins).

Run:  uv run python scripts/build_dataset.py [--window-sec 64]
"""

from __future__ import annotations

import json
import sys

from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pbh import config as C
from pbh.data import whiten_segment
from pbh.progress import progress
from pbh.spectrogram import model_input
from pbh.sweep import SweepGrid
from pbh.waveforms import (
    inject_into_window,
    make_whitened_injection,
    sample_params,
)

POS_PER_SEG = {"train": 1250, "val": 625}
NEG_PER_SEG = {"train": 1250, "val": 625}
MIN_IN_WINDOW_SNR = 4.0

CROP = C.WHITEN_CROP_SEC * C.SAMPLE_RATE


def layout(window_sec: int) -> tuple[int, int, Path]:
    """(n_win samples, n_time bins, shard_dir). window_sec == C.WINDOW_SEC
    reproduces v1 exactly (data/shards, 256 padded bins)."""
    if window_sec == C.WINDOW_SEC:
        return C.WINDOW_SEC * C.SAMPLE_RATE, C.N_TIME_BINS, C.SHARD_DIR
    g = SweepGrid.short(window_sec)
    return g.win_samp, g.n_time_bins, C.DATA_DIR / f"shards_w{window_sec}"


def build_segment(args: tuple[str, int, int, int], window_sec: int) -> list[dict]:
    """Worker: produce all examples for one segment, writing into the
    pre-allocated memmap at [offset, offset + n_pos + n_neg)."""
    split, gps, offset, seed = args
    n_pos, n_neg = POS_PER_SEG[split], NEG_PER_SEG[split]
    n_win, n_time, shard_dir = layout(window_sec)
    rng = np.random.default_rng(seed)

    w, t0, psd = whiten_segment("H1", gps)
    x = np.lib.format.open_memmap(shard_dir / f"x_{split}.npy", mode="r+")

    lo, hi = CROP, len(w) - CROP - n_win
    meta: list[dict] = []
    row = offset
    n_seg = n_pos + n_neg
    run = f"build_w{window_sec}_{split}_{gps}"

    for _ in range(n_pos):
        while True:
            p = sample_params(rng)
            h_w, snr_ref = make_whitened_injection(p, "H1", t0, psd)
            target = float(rng.uniform(*C.TRAIN_SNR_RANGE))
            start = int(rng.integers(lo, hi))
            # merger anywhere from 25% into the window to 30% past its end
            merger_idx = int(rng.uniform(0.25, 1.30) * n_win)
            window, in_snr = inject_into_window(
                w[start : start + n_win], h_w, snr_ref, target, merger_idx
            )
            if in_snr >= MIN_IN_WINDOW_SNR:
                break
        x[row] = model_input(window, n_time)
        meta.append(
            dict(
                row=row, split=split, label=1, gps=gps, start=start,
                mass1=p.mass1, mass2=p.mass2, chirp_mass=p.chirp_mass,
                target_snr=target, in_window_snr=in_snr,
                merger_frac=merger_idx / n_win,
            )
        )
        row += 1
        if (row - offset) % 100 == 0:
            progress(run, row - offset, n_seg)

    for _ in range(n_neg):
        start = int(rng.integers(lo, hi))
        x[row] = model_input(w[start : start + n_win], n_time)
        meta.append(
            dict(
                row=row, split=split, label=0, gps=gps, start=start,
                mass1=0.0, mass2=0.0, chirp_mass=0.0,
                target_snr=0.0, in_window_snr=0.0, merger_frac=0.0,
            )
        )
        row += 1
        if (row - offset) % 100 == 0:
            progress(run, row - offset, n_seg)

    progress(run, n_seg, n_seg)
    print(f"segment {gps} [{split}] done", flush=True)
    return meta


def job_list(manifest: dict) -> list[tuple[str, int, int, int]]:
    jobs = []
    for split in ("train", "val"):
        per_seg = POS_PER_SEG[split] + NEG_PER_SEG[split]
        for i, gps in enumerate(manifest["H1"][split]):
            # deterministic per-segment seed (hash() is process-salted)
            jobs.append((split, gps, i * per_seg, C.SEED + gps % 10**6))
    return jobs


def main() -> None:
    """Three modes (multiprocessing.Pool deadlocks on macOS, so jobs run as
    independent OS processes driven by xargs — see run_build.sh):
      --init        allocate memmaps, print number of jobs
      --job I       build segment job I, write its meta part
      --finalize    merge meta parts into meta_{split}.parquet
    """
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--init", action="store_true")
    ap.add_argument("--job", type=int, default=None)
    ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--window-sec", type=int, default=C.WINDOW_SEC)
    args = ap.parse_args()

    _, n_time, shard_dir = layout(args.window_sec)
    manifest = json.loads((C.DATA_DIR / "manifest.json").read_text())
    jobs = job_list(manifest)
    parts_dir = shard_dir / "meta_parts"

    if args.init:
        shard_dir.mkdir(parents=True, exist_ok=True)
        parts_dir.mkdir(exist_ok=True)
        for split in ("train", "val"):
            per_seg = POS_PER_SEG[split] + NEG_PER_SEG[split]
            n_total = per_seg * len(manifest["H1"][split])
            np.lib.format.open_memmap(
                shard_dir / f"x_{split}.npy",
                mode="w+",
                dtype=np.float16,
                shape=(n_total, C.N_FREQ_BINS, n_time),
            )
        print(len(jobs))
        return

    if args.job is not None:
        split, gps, offset, seed = jobs[args.job]
        meta = build_segment((split, gps, offset, seed), args.window_sec)
        pd.DataFrame(meta).to_parquet(parts_dir / f"{split}_{gps}.parquet")
        return

    if args.finalize:
        parts = [pd.read_parquet(p) for p in sorted(parts_dir.glob("*.parquet"))]
        meta = pd.concat(parts, ignore_index=True)
        for split in ("train", "val"):
            df = meta[meta.split == split].sort_values("row").reset_index(drop=True)
            df.to_parquet(shard_dir / f"meta_{split}.parquet")
            print(f"{split}: {len(df)} examples "
                  f"({int((df.label == 1).sum())} pos / "
                  f"{int((df.label == 0).sum())} neg)")
        print("DATASET BUILD DONE", flush=True)


if __name__ == "__main__":
    main()
