"""Noise-variety experiment (PREREGISTRATION_DRAFT.md): train SpectrogramCNN on 64-s windows, segment by segment.

  --source stream    Arm V (phone): fresh O3a H1 segments from stream_segments.json, each used once, then purged.
  --source original  Arm C (control): the 16 cnn_w64 train segments, recycled in a seeded order (fetched once, kept).

Everything except the noise is identical between arms at the same --seed: every random draw (bank entries, injection
parameters, window starts, merger positions, ordering, augmentation) comes from default_rng([seed, segment_index]).
Injections use the golden-tested bank fast path (bank.WhitenedEntry == make_whitened_injection to 1.7e-8).
Resumable: an atomic checkpoint after every segment; rerunning the same command continues where it stopped.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from pbh import config as C
from pbh.data import fetch_segment, segment_path, whiten_segment
from pbh.models import SpectrogramCNN
from pbh.spectrogram import model_input
from pbh.sweep import SweepGrid
from pbh.waveforms import sample_params

from bank import WhitenedEntry, whitening_filter

HERE = Path(__file__).resolve().parent
PER_SEG = 1024  # 16 batches of 64: 512 injections + 512 noise windows
N_POS = PER_SEG // 2
MIN_IN_WINDOW_SNR = 4.0  # as build_dataset.py
GRID = SweepGrid.short(64)
CROP = C.WHITEN_CROP_SEC * C.SAMPLE_RATE


def segment_list(source: str, seed: int, n_seg: int) -> list[int]:
    if source == "stream":
        pool = json.load(open(HERE / "stream_segments.json"))[f"pool_seed{seed % 2}"]
        order = np.random.default_rng([seed, 999]).permutation(pool).tolist()
        if len(order) < n_seg:
            raise SystemExit(f"pool has {len(order)} segments, run needs {n_seg}")
        return [int(g) for g in order[:n_seg]]
    train = json.load(open(C.DATA_DIR / "manifest.json"))["H1"]["train"]
    out: list[int] = []
    cycle = 0
    while len(out) < n_seg:
        out += np.random.default_rng([seed, 1000 + cycle]).permutation(train).tolist()
        cycle += 1
    return [int(g) for g in out[:n_seg]]


def battery_ok(log) -> None:
    """Phone only: pause while the battery is low or hot (USB input is capped at ~2.5 W)."""
    vf, tf = Path("/sys/class/power_supply/battery/voltage_now"), Path("/sys/class/power_supply/battery/temp")
    if not vf.exists():
        return
    while True:
        v = int(vf.read_text()) / 1e6
        t = int(tf.read_text()) / 10
        if v >= 3.70 and t <= 42.0:
            return
        log(f"battery guard: {v:.3f} V, {t:.1f} C -> pausing 120 s")
        time.sleep(120)


def augment(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """train.py's augmentation, seeded."""
    k = int(rng.integers(-12, 13))
    if k:
        x = np.roll(x, k, axis=1)
        if k > 0:
            x[:, :k] = x[:, k : k + 1]
        else:
            x[:, k:] = x[:, k - 1 : k]
    f = int(rng.integers(-2, 3))
    if f:
        x = np.roll(x, f, axis=0)
    return x + rng.normal(0, 0.05, x.shape).astype(np.float32)


def make_samples(gps: int, rng: np.random.Generator, bank_files: list[Path], k: int, after_whiten=lambda: None):
    w, t0, psd = whiten_segment("H1", gps)
    after_whiten()  # only now start the next download: fetch + whiten peaks together exceed the phone's RAM
    wf = whitening_filter(psd)
    n_win, n_time = GRID.win_samp, GRID.n_time_bins
    lo, hi = CROP, len(w) - CROP - n_win
    x = np.empty((PER_SEG, 128, n_time), dtype=np.float32)
    meta = np.zeros((PER_SEG, 4), dtype=np.float32)  # label, chirp mass, target snr, in-window snr
    picks = rng.choice(len(bank_files), size=k, replace=False)
    per_entry = np.array_split(np.arange(N_POS), k)
    for e_i, rows in zip(picks, per_entry):
        z = np.load(bank_files[e_i])
        entry = WhitenedEntry(z["hp0"], z["hc0"], wf)
        mc = float((z["m1"] * z["m2"]) ** 0.6 / (z["m1"] + z["m2"]) ** 0.2)
        for r in rows:
            while True:
                p = sample_params(rng)  # sky + orientation (its masses are unused: the bank entry fixes them)
                target = float(rng.uniform(*C.TRAIN_SNR_RANGE))
                start = int(rng.integers(lo, hi))
                merger_idx = int(rng.uniform(0.25, 1.30) * n_win)
                win, in_snr = entry.inject(w[start : start + n_win], p, "H1", t0, target, merger_idx)
                if in_snr >= MIN_IN_WINDOW_SNR:
                    break
            x[r] = model_input(win, n_time)
            meta[r] = (1, mc, target, in_snr)
        del entry
    for r in range(N_POS, PER_SEG):
        start = int(rng.integers(lo, hi))
        x[r] = model_input(w[start : start + n_win], n_time)
    return x, meta


def purge(gps: int) -> None:
    try:
        segment_path("H1", gps).unlink()
    except FileNotFoundError:
        pass
    try:  # gwpy keeps a second copy in astropy's download cache (far_deep lesson: ~0.25 GB/segment)
        from astropy.utils.data import clear_download_cache

        clear_download_cache()
    except Exception:
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["stream", "original"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--samples", type=int, default=600_000)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--bank", default=str(HERE / "bank"))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=str(HERE / "runs"))
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    run = f"{a.source}_s{a.seed}"
    out = Path(a.out) / run
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.txt", "a")

    def log(msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    n_seg = -(-a.samples // PER_SEG)
    segs = segment_list(a.source, a.seed, n_seg)
    bank_files = sorted(Path(a.bank).glob("w*.npz"))
    assert len(bank_files) >= a.k, "bank too small"
    total_steps = n_seg * (PER_SEG // C.BATCH_SIZE)

    torch.manual_seed(a.seed)
    model = SpectrogramCNN()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    lossf = nn.BCEWithLogitsLoss()
    ck = out / "ckpt.pt"
    done = 0
    if ck.exists():
        s = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"])
        done = s["segments_done"]
        log(f"resumed after {done} segments")
    else:
        json.dump(dict(vars(a), n_seg=n_seg, total_steps=total_steps, per_seg=PER_SEG, segments=segs,
                       bank_n=len(bank_files)), open(out / "config.json", "w"), indent=1)
        log(f"start {run}: {n_seg} segments, {total_steps} steps, bank {len(bank_files)}")

    prefetch: dict[int, threading.Thread] = {}

    def start_fetch(j: int) -> None:
        if j < n_seg and j not in prefetch and not segment_path("H1", segs[j]).exists():
            segment_path("H1", segs[j]).with_suffix(".tmp.hdf5").unlink(missing_ok=True)  # stub left by a crash
            th = threading.Thread(target=lambda g=segs[j]: fetch_segment("H1", g), daemon=True)
            th.start()
            prefetch[j] = th

    model.train()
    for j in range(done, n_seg):
        t_seg = time.time()
        battery_ok(log)
        gps = segs[j]
        start_fetch(j)
        prefetch.pop(j).join() if j in prefetch else None
        if not segment_path("H1", gps).exists():
            fetch_segment("H1", gps)  # the prefetch thread failed; retry in the foreground (raises if GWOSC is down)
        t_fetch = time.time() - t_seg

        rng = np.random.default_rng([a.seed, j])
        # dropout uses torch's global RNG: reseed per segment, or a resumed run diverges (caught by the resume test)
        torch.manual_seed(int(rng.integers(2**62)))
        t0 = time.time()
        x, meta = make_samples(gps, rng, bank_files, a.k, after_whiten=lambda: start_fetch(j + 1))
        t_prep = time.time() - t0
        if a.source == "stream":
            purge(gps)

        order = rng.permutation(PER_SEG)
        y = (np.arange(PER_SEG) < N_POS).astype(np.float32)
        t0 = time.time()
        losses = []
        for b in range(0, PER_SEG, C.BATCH_SIZE):
            idx = order[b : b + C.BATCH_SIZE]
            xb = np.stack([augment(x[i].copy(), rng) for i in idx])
            xb_t = torch.from_numpy(xb).unsqueeze(1)
            yb_t = torch.from_numpy(y[idx])
            opt.zero_grad()
            loss = lossf(model(xb_t), yb_t)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(float(loss))
        t_train = time.time() - t0

        tmp = out / "ckpt.tmp"
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                        segments_done=j + 1), tmp)
        os.replace(tmp, ck)
        with open(out / "segments.jsonl", "a") as f:
            f.write(json.dumps(dict(j=j, gps=gps, loss=float(np.mean(losses)), fetch_wait_s=round(t_fetch, 1),
                                    prep_s=round(t_prep, 1), train_s=round(t_train, 1),
                                    mean_in_snr=float(meta[:N_POS, 3].mean()))) + "\n")
        log(f"seg {j + 1}/{n_seg} gps {gps} loss {np.mean(losses):.4f} wait {t_fetch:.0f}s prep {t_prep:.0f}s "
            f"train {t_train:.0f}s total {time.time() - t_seg:.0f}s")

    torch.save(model.state_dict(), out / "final.pt")
    log("DONE")


if __name__ == "__main__":
    main()
