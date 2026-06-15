"""Path G milestone G2b: build an H1+L1 training set (self-contained, resumable).

cnn_w64 was trained on H1 only and TRANSFERRED to L1 in G1. This builds 64-s
spectrogram shards from BOTH detectors so a sibling model (cnn_hl) sees L1 noise
directly -- the one remaining tractable lever on the +1.37x coincidence gain.

NO LEAKAGE: trains on the 5 L1 segments at H1-TRAIN times; the G1 eval uses the 5
L1 segments at H1-TEST times. Writes to data/shards_w64_hl/ (existing shards_w64
and the gate-critical cnn_w64 are untouched). Resumable: completed (ifo,gps)
segments are recorded and skipped on restart (power-loss safe).

Run:  .venv/bin/python scripts/build_hl.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pbh import config as C
from pbh.data import whiten_segment
from pbh.progress import progress
from pbh.spectrogram import model_input
from pbh.sweep import SweepGrid
from pbh.waveforms import inject_into_window, make_whitened_injection, sample_params

POS_PER_SEG = {"train": 1250, "val": 625}
NEG_PER_SEG = {"train": 1250, "val": 625}
MIN_IN_WINDOW_SNR = 4.0
CROP = C.WHITEN_CROP_SEC * C.SAMPLE_RATE
GRID = SweepGrid.short(64)
N_WIN, N_TIME = GRID.win_samp, GRID.n_time_bins
SHARD = C.DATA_DIR / "shards_w64_hl"


def seg_lists(manifest):
    l1_train = [g for g in manifest["L1"]["coinc"] if g in manifest["H1"]["train"]]
    train = [("H1", g) for g in manifest["H1"]["train"]] + [("L1", g) for g in l1_train]
    val = [("H1", g) for g in manifest["H1"]["val"]]
    return {"train": train, "val": val}


def build_one(ifo, gps, split, offset, x, seed):
    rng = np.random.default_rng(seed)
    w, t0, psd = whiten_segment(ifo, gps)
    lo, hi = CROP, len(w) - CROP - N_WIN
    n_pos, n_neg = POS_PER_SEG[split], NEG_PER_SEG[split]
    meta, row = [], offset
    run = f"build_hl_{split}_{ifo}_{gps}"
    for _ in range(n_pos):
        while True:
            p = sample_params(rng)
            h_w, snr_ref = make_whitened_injection(p, ifo, t0, psd)
            target = float(rng.uniform(*C.TRAIN_SNR_RANGE))
            start = int(rng.integers(lo, hi))
            merger_idx = int(rng.uniform(0.25, 1.30) * N_WIN)
            window, in_snr = inject_into_window(w[start:start + N_WIN], h_w, snr_ref, target, merger_idx)
            if in_snr >= MIN_IN_WINDOW_SNR:
                break
        x[row] = model_input(window, N_TIME)
        meta.append(dict(row=row, split=split, label=1, ifo=ifo, gps=gps,
                         chirp_mass=p.chirp_mass, target_snr=target, in_window_snr=in_snr))
        row += 1
        if (row - offset) % 100 == 0:
            progress(run, row - offset, n_pos + n_neg)
    for _ in range(n_neg):
        start = int(rng.integers(lo, hi))
        x[row] = model_input(w[start:start + N_WIN], N_TIME)
        meta.append(dict(row=row, split=split, label=0, ifo=ifo, gps=gps,
                         chirp_mass=0.0, target_snr=0.0, in_window_snr=0.0))
        row += 1
        if (row - offset) % 100 == 0:
            progress(run, row - offset, n_pos + n_neg)
    progress(run, n_pos + n_neg, n_pos + n_neg)
    print(f"  {ifo} {gps} [{split}] done", flush=True)
    return meta


def main() -> None:
    SHARD.mkdir(parents=True, exist_ok=True)
    parts = SHARD / "meta_parts"
    parts.mkdir(exist_ok=True)
    manifest = json.loads((C.DATA_DIR / "manifest.json").read_text())
    segs = seg_lists(manifest)
    done_path = SHARD / "done.json"
    done = set(tuple(x) for x in json.loads(done_path.read_text())) if done_path.exists() else set()

    for split in ("train", "val"):
        per_seg = POS_PER_SEG[split] + NEG_PER_SEG[split]
        n_total = per_seg * len(segs[split])
        xpath = SHARD / f"x_{split}.npy"
        if not xpath.exists():
            np.lib.format.open_memmap(xpath, mode="w+", dtype=np.float16,
                                      shape=(n_total, C.N_FREQ_BINS, N_TIME))
        x = np.lib.format.open_memmap(xpath, mode="r+")
        t0 = time.time()
        for i, (ifo, gps) in enumerate(segs[split]):
            key = (split, ifo, int(gps))
            if key in done:
                continue
            meta = build_one(ifo, gps, split, i * per_seg, x, C.SEED + gps % 10**6 + (0 if ifo == "H1" else 7))
            pd.DataFrame(meta).to_parquet(parts / f"{split}_{ifo}_{gps}.parquet")
            done.add(key)
            done_path.write_text(json.dumps(sorted(done)))
        print(f"{split}: {len(segs[split])} segments ({time.time()-t0:.0f}s)", flush=True)

    # finalize meta
    allparts = [pd.read_parquet(p) for p in sorted(parts.glob("*.parquet"))]
    meta = pd.concat(allparts, ignore_index=True)
    for split in ("train", "val"):
        df = meta[meta.split == split].sort_values("row").reset_index(drop=True)
        df.to_parquet(SHARD / f"meta_{split}.parquet")
        print(f"{split}: {len(df)} examples ({int((df.label==1).sum())} pos / "
              f"{int((df.label==0).sum())} neg)")
    print("HL DATASET BUILD DONE", flush=True)


if __name__ == "__main__":
    main()
