"""v2 track-score aggregation eval (pre-registered in RESULTS.md).

Production-path protocol: the full whitened waveform is added to the whitened
SEGMENT, the model sweeps the same grid as the FAR scan, only signal-
contaminated windows are recomputed and spliced into the cached noise score
series, and aggregated statistics (pbh.aggregate) are evaluated at positions
touching the contamination against zero-FA thresholds from the same noise sweep.

--window-sec selects the regime:
  256 (default) -> rung 1: 256 s overlapping windows; stats max / boxcar_bank /
                   count_above; oracle = boxcar over the true-track windows.
  64 (rung 2)   -> short non-overlapping windows; stats max / sum_track
                   (sqrt(k)-normalized summed logits); oracle = sumnorm. Loads
                   --weights cnn_w64; artifacts under results/track_w64/.

Stages, each resumable with atomic (*.tmp -> os.replace) per-segment caches:
  A. noise sweep, B. injection campaign, C. metrics + plot.

Run:  .venv/bin/python scripts/track_eval.py [--window-sec 64 --weights cnn_w64]
                                             [--selftest] [--smoke]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pbh import config as C
from pbh.aggregate import (
    BANK_K,
    SUM_BANK_K,
    boxcar,
    make_aggregators,
    make_aggregators_short,
    sumnorm,
)
from pbh.data import whiten_segment
from pbh.models import make_model
from pbh.progress import progress
from pbh.spectrogram import spectrogram
from pbh.sweep import (
    HOP_SAMP,
    NPERSEG,
    SweepGrid,
    V1_GRID,
    pool_and_log,
    score_windows,
    segment_window_scores,
)
from pbh.waveforms import make_whitened_injection, sample_params

CROP = C.WHITEN_CROP_SEC * C.SAMPLE_RATE
N_INJ_PER_SEG = 250


def atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    tmp = path.parent / (path.stem + ".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


def atomic_save_parquet(path: Path, df) -> None:
    tmp = path.parent / (path.stem + ".tmp.parquet")
    df.to_parquet(tmp)
    os.replace(tmp, path)


def contaminated_range(
    sig_start: int, sig_end: int, n_win: int, grid: SweepGrid
) -> tuple[int, int]:
    """Window indices [i_min, i_max] whose samples overlap [sig_start, sig_end)
    (positions relative to the crop origin)."""
    i_min = max(0, (sig_start - grid.win_samp) // grid.win_hop_samp + 1)
    i_max = min(n_win - 1, (sig_end - 1) // grid.win_hop_samp)
    return int(i_min), int(i_max)


def recompute_windows(
    model, device, wc: np.ndarray, sig: np.ndarray, sig_end: int,
    i_min: int, i_max: int, grid: SweepGrid,
) -> np.ndarray:
    """Re-score windows i_min..i_max of crop-origin strain wc with sig added
    (sig's last sample lands at sig_end - 1). Frame grid matches the global
    sweep exactly because spectrogram() frames are pure strides (no padding)."""
    f0 = i_min * grid.hop_frames
    f1 = i_max * grid.hop_frames + grid.frames_per_win - 1
    r_lo, r_hi = f0 * HOP_SAMP, f1 * HOP_SAMP + NPERSEG
    region = wc[r_lo:r_hi].copy()

    sig_start = sig_end - len(sig)
    dst_lo = max(0, sig_start - r_lo)
    dst_hi = min(len(region), sig_end - r_lo)
    src_lo = dst_lo - (sig_start - r_lo)
    if dst_hi > dst_lo:
        region[dst_lo:dst_hi] += sig[src_lo : src_lo + (dst_hi - dst_lo)]

    spec = spectrogram(region)
    wins = np.stack(
        [
            pool_and_log(spec[:, s : s + grid.frames_per_win], grid.n_time_bins)
            for s in ((np.arange(i_min, i_max + 1) - i_min) * grid.hop_frames)
        ]
    )
    return score_windows(model, device, wins)


def signal_duration_s(sig: np.ndarray) -> float:
    """Duration holding 99.5% of the trailing waveform energy."""
    cum = np.cumsum(sig**2)
    if cum[-1] <= 0:
        return 0.0
    start = int(np.searchsorted(cum, 0.005 * cum[-1]))
    return (len(sig) - start) / C.SAMPLE_RATE


def load_model(arch: str, weights: str, device: str):
    model = make_model(arch)
    model.load_state_dict(
        torch.load(C.MODEL_DIR / f"{weights}.pt", map_location=device)
    )
    model.to(device).eval()
    return model


def selftest(model, device, gps: int, grid: SweepGrid) -> None:
    """Golden check: zero-amplitude splice must reproduce the cached sweep."""
    w, _, _ = whiten_segment("H1", gps)
    wc = w[CROP:-CROP]
    scores = segment_window_scores(model, device, w, grid)
    n_win = len(scores)
    sig = np.zeros(512 * C.SAMPLE_RATE)
    sig_end = len(wc) // 2
    i_min, i_max = contaminated_range(sig_end - len(sig), sig_end, n_win, grid)
    re = recompute_windows(model, device, wc, sig, sig_end, i_min, i_max, grid)
    diff = float(np.abs(re - scores[i_min : i_max + 1]).max())
    print(f"selftest: {i_max - i_min + 1} windows recomputed, max|diff| = {diff:.2e}")
    assert diff < 1e-4, f"frame-grid misalignment: {diff}"
    print("SELFTEST PASS")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="cnn", choices=["cnn", "transformer"])
    ap.add_argument("--weights", default=None, help="model .pt stem (default=arch)")
    ap.add_argument("--window-sec", type=int, default=C.WINDOW_SEC)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="8 injections/segment")
    args = ap.parse_args()
    weights = args.weights or args.arch

    is_short = args.window_sec != C.WINDOW_SEC
    grid = SweepGrid.short(args.window_sec) if is_short else V1_GRID
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = load_model(args.arch, weights, device)
    manifest = json.loads((C.DATA_DIR / "manifest.json").read_text())
    test_segs = manifest["H1"]["test"]

    base = f"track_w{args.window_sec}" if is_short else "track"
    cache_dir = C.RESULTS_DIR / base
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.selftest:
        selftest(model, device, test_segs[0], grid)
        return

    tag = "_smoke" if args.smoke else ""
    n_inj = 8 if args.smoke else N_INJ_PER_SEG
    inj_dir = C.RESULTS_DIR / f"{base}{tag}"
    inj_dir.mkdir(parents=True, exist_ok=True)

    # regime-dependent statistic family (pre-registered in RESULTS.md)
    oracle_combiner = sumnorm if is_short else boxcar
    hop_sec = grid.win_hop_samp / C.SAMPLE_RATE
    oracle_k_lo, oracle_k_hi = (1, 12) if is_short else (4, 200)
    seed = C.SEED + (999 if is_short else 888)

    # ---- stage A: noise sweep (cached, shared between smoke and full)
    import time

    run_name = f"{base}_{weights}{tag}"
    t_start = time.time()
    noise, held_w = {}, {}
    total_hours = 0.0
    for seg_i, gps in enumerate(test_segs):
        npath = cache_dir / f"noise_{gps}.npy"
        ipath = inj_dir / f"inj_{gps}.parquet"
        if npath.exists() and ipath.exists():
            noise[gps] = np.load(npath)
        else:
            w, t0, psd = whiten_segment("H1", gps)
            held_w[gps] = (w, t0, psd)
            if npath.exists():
                noise[gps] = np.load(npath)
            else:
                noise[gps] = segment_window_scores(model, device, w, grid)
                atomic_save_npy(npath, noise[gps])
            print(f"[A] {gps}: {len(noise[gps])} noise windows", flush=True)
        total_hours += (
            len(noise[gps]) * grid.win_hop_samp + grid.win_samp - grid.win_hop_samp
        ) / C.SAMPLE_RATE / 3600
        progress(f"{run_name}_noise", seg_i + 1, len(test_segs),
                 elapsed_s=time.time() - t_start)

    pooled = np.concatenate(list(noise.values()))
    aggs = make_aggregators_short(pooled) if is_short else make_aggregators(pooled)
    thresholds = {
        name: max(fn(noise[g]) for g in test_segs) for name, fn in aggs.items()
    }
    oracle_thresh = {
        k: max(oracle_combiner(noise[g], k) for g in test_segs)
        for k in range(oracle_k_lo, oracle_k_hi + 1)
    }
    print(f"thresholds: { {k: round(v, 4) for k, v in thresholds.items()} }")

    # ---- stage B: injections, spliced into the cached noise series
    import pandas as pd

    n_total = len(test_segs) * n_inj
    t_b = time.time()
    for seg_i, gps in enumerate(test_segs):
        ipath = inj_dir / f"inj_{gps}.parquet"
        if ipath.exists():
            progress(f"{run_name}_inj", (seg_i + 1) * n_inj, n_total,
                     elapsed_s=time.time() - t_b, segment=float(seg_i + 1))
            continue
        w, t0, psd = held_w[gps]
        wc = w[CROP:-CROP]
        series0 = noise[gps]
        n_win = len(series0)
        grid_end = (n_win - 1) * grid.win_hop_samp + grid.win_samp
        rng = np.random.default_rng([seed, int(gps)])
        rows = []
        for j in range(n_inj):
            p = sample_params(rng)
            h_w, snr_ref = make_whitened_injection(p, "H1", t0, psd)
            target = float(rng.uniform(*C.EVAL_SNR_RANGE))
            sig = h_w * (target / snr_ref)
            sig_end = int(rng.integers(len(sig), min(len(wc), grid_end)))
            i_min, i_max = contaminated_range(sig_end - len(sig), sig_end, n_win, grid)
            re = recompute_windows(model, device, wc, sig, sig_end, i_min, i_max, grid)
            series = series0.copy()
            series[i_min : i_max + 1] = re
            mask = np.zeros(n_win, dtype=bool)
            mask[i_min : i_max + 1] = True

            dur = signal_duration_s(sig)
            k_oracle = int(np.clip(round((dur + args.window_sec) / hop_sec),
                                   oracle_k_lo, oracle_k_hi))
            row = dict(
                gps=gps, chirp_mass=p.chirp_mass, target_snr=target,
                dur_s=dur, k_oracle=k_oracle,
                stat_oracle=oracle_combiner(series, k_oracle, mask),
            )
            for name, fn in aggs.items():
                row[f"stat_{name}"] = fn(series, mask)
            rows.append(row)
            if (j + 1) % 10 == 0:
                done = seg_i * n_inj + j + 1
                progress(f"{run_name}_inj", done, n_total,
                         elapsed_s=time.time() - t_b, segment=float(seg_i + 1))
            if (j + 1) % 50 == 0:
                print(f"[B] {gps}: {j + 1}/{n_inj}", flush=True)
        atomic_save_parquet(ipath, pd.DataFrame(rows))
        print(f"[B] {gps}: wrote {ipath.name}", flush=True)
        del held_w[gps]

    # ---- stage C: metrics
    df = pd.concat([pd.read_parquet(inj_dir / f"inj_{g}.parquet") for g in test_segs])
    df["det_oracle"] = df.apply(
        lambda r: r.stat_oracle > oracle_thresh[int(r.k_oracle)], axis=1
    )
    for name in aggs:
        df[f"det_{name}"] = df[f"stat_{name}"] > thresholds[name]

    snr_bins = np.linspace(*C.EVAL_SNR_RANGE, 11)
    mass_edges = [0.17, 0.35, 0.55, 0.88]
    mass_labels = ["0.17-0.35", "0.35-0.55", "0.55-0.88"]
    stat_names = list(aggs) + ["oracle"]

    def eff_curve(sub, det_col):
        centers, effs = [], []
        for lo, hi in zip(snr_bins[:-1], snr_bins[1:]):
            s = sub[(sub.target_snr >= lo) & (sub.target_snr < hi)]
            if len(s) >= (3 if args.smoke else 10):
                centers.append((lo + hi) / 2)
                effs.append(float(s[det_col].mean()))
        return centers, effs

    results = {}
    for name in stat_names:
        fracs, curves = {}, {}
        for lo_m, hi_m, lab in zip(mass_edges[:-1], mass_edges[1:], mass_labels):
            sub = df[(df.chirp_mass >= lo_m) & (df.chirp_mass < hi_m)]
            c, e = eff_curve(sub, f"det_{name}")
            curves[lab] = {"snr": c, "eff": e}
            snr50 = (
                float(np.interp(0.5, e, c)) if len(c) > 1 and max(e) >= 0.5 else np.nan
            )
            fracs[lab] = 8.0 / snr50 if np.isfinite(snr50) else 0.0
        results[name] = {"mf_distance_fraction": fracs, "eff_curves": curves}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    palette = {"max": "k", "boxcar_bank": "tab:blue", "count_above": "tab:green",
               "sum_track": "tab:purple", "oracle": "tab:red"}
    for ax, lab in zip(axes.flat[:3], mass_labels):
        for name in stat_names:
            cv = results[name]["eff_curves"][lab]
            ls = "--" if name == "oracle" else "-"
            ax.plot(cv["snr"], cv["eff"], ls, color=palette.get(name), marker="o",
                    ms=3, label=name)
        ax.axvline(8, color="gray", ls=":", label="ideal MF")
        ax.set_title(f"Mc {lab} Msun")
        ax.set_xlabel("total band-limited SNR")
        ax.set_ylabel("efficiency")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    ax = axes.flat[3]
    x = np.arange(len(mass_labels))
    n_stat = len(stat_names)
    for i, name in enumerate(stat_names):
        f = [results[name]["mf_distance_fraction"][m] for m in mass_labels]
        ax.bar(x + (i - (n_stat - 1) / 2) * 0.18, f, 0.18,
               color=palette.get(name), label=name)
    ax.axhline(1.0, color="k", ls="--", alpha=0.5)
    ax.set_xticks(x, mass_labels)
    ax.set_ylabel("fraction of ideal-MF sensitive distance")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.suptitle(f"{base} ({weights}, {len(df)} injections, "
                 f"zero-FA over {total_hours:.1f} h)")
    fig.tight_layout()
    fig.savefig(C.RESULTS_DIR / f"efficiency_{weights}_{base}{tag}.png", dpi=120)

    out = {
        "weights": weights,
        "arch": args.arch,
        "window_sec": args.window_sec,
        "protocol": "full-signal segment injection, spliced sweep, masked stats",
        "n_injections": int(len(df)),
        "test_hours": total_hours,
        "n_noise_windows": int(len(pooled)),
        "thresholds": {k: float(v) for k, v in thresholds.items()},
        "bank_k": list(SUM_BANK_K if is_short else BANK_K),
        "seed": seed,
        "results": results,
    }
    jpath = C.RESULTS_DIR / f"eval_{weights}_{base}{tag}.json"
    jpath.write_text(json.dumps(out, indent=2))
    progress(f"{run_name}_inj", n_total, n_total, elapsed_s=time.time() - t_b)
    df.to_parquet(C.RESULTS_DIR / f"injections_{weights}_{base}{tag}.parquet")
    summary = {n: results[n]["mf_distance_fraction"] for n in stat_names}
    print(json.dumps(summary, indent=2))
    print(f"wrote {jpath.name} + efficiency_{weights}_{base}{tag}.png")


if __name__ == "__main__":
    main()
