"""Phase 4: honest evaluation on held-out REAL noise.

1. FAR scan   — slide the detector over the 6 test segments (pure noise),
                cluster adjacent triggers, get threshold-vs-FAR.
2. Injections — 1500 injections across the mass grid into the same segments,
                efficiency vs band-limited optimal SNR at the FAR threshold.
3. Money plot — efficiency curves vs the ideal matched filter (step at SNR 8),
                and the recovered fraction of MF sensitive distance per mass.

Conventions stated in RESULTS.md: single-window scoring, band-limited
[50, 1024] Hz SNR, threshold set at zero false alarms over the test set.

Run:  uv run python scripts/evaluate.py --model cnn
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pbh import config as C
from pbh.data import whiten_segment
from pbh.models import make_model
from pbh.spectrogram import spectrogram
from pbh.sweep import pool_and_log, score_windows, segment_window_scores
from pbh.waveforms import inject_into_window, make_whitened_injection, sample_params

N_WIN = C.WINDOW_SEC * C.SAMPLE_RATE
CROP = C.WHITEN_CROP_SEC * C.SAMPLE_RATE
EVAL_SNR_RANGE = C.EVAL_SNR_RANGE
N_INJ_PER_SEG = 250


def cluster_count(scores: np.ndarray, thresh: float) -> int:
    """Number of distinct trigger clusters (adjacent windows merged)."""
    above = scores > thresh
    return int(np.sum(above[1:] & ~above[:-1]) + (1 if above[0] else 0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["cnn", "transformer"], required=True)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = make_model(args.model)
    model.load_state_dict(torch.load(C.MODEL_DIR / f"{args.model}.pt",
                                     map_location=device))
    model.to(device).eval()

    manifest = json.loads((C.DATA_DIR / "manifest.json").read_text())
    test_segs = manifest["H1"]["test"]

    # ---------------- 1. FAR scan on pure noise
    print(f"[1/3] FAR scan over {len(test_segs)} test segments...")
    noise_scores, total_hours = [], 0.0
    whitened = {}
    for gps in test_segs:
        w, t0, psd = whiten_segment("H1", gps)
        whitened[gps] = (w, t0, psd)
        s = segment_window_scores(model, device, w)
        noise_scores.append(s)
        total_hours += (len(w) / C.SAMPLE_RATE - 2 * C.WHITEN_CROP_SEC) / 3600
    noise_scores = np.concatenate(noise_scores)
    print(f"  {len(noise_scores)} windows over {total_hours:.2f} h of test noise")

    thresh_zero_fa = float(noise_scores.max())
    # threshold-vs-FAR curve (clustered triggers per hour)
    grid = np.quantile(noise_scores, np.linspace(0.5, 1.0, 200, endpoint=False))
    far_curve = [
        (float(t), cluster_count(noise_scores, t) / total_hours) for t in grid
    ]
    far_1h = min((t for t, far in far_curve if far <= 1.0), default=thresh_zero_fa)
    print(f"  zero-FA threshold {thresh_zero_fa:.3f}; FAR<=1/h threshold {far_1h:.3f}")

    # ---------------- 2. injection campaign
    print(f"[2/3] injecting {N_INJ_PER_SEG} per segment...")
    rng = np.random.default_rng(C.SEED + 777)
    rows = []
    for gps in test_segs:
        w, t0, psd = whitened[gps]
        lo, hi = CROP, len(w) - CROP - N_WIN
        wins, metas = [], []
        for _ in range(N_INJ_PER_SEG):
            p = sample_params(rng)
            h_w, snr_ref = make_whitened_injection(p, "H1", t0, psd)
            target = float(rng.uniform(*EVAL_SNR_RANGE))
            start = int(rng.integers(lo, hi))
            merger_idx = int(rng.uniform(0.30, 1.0) * N_WIN)
            window, in_snr = inject_into_window(
                w[start : start + N_WIN], h_w, snr_ref, target, merger_idx
            )
            spec = spectrogram(window)
            wins.append(pool_and_log(spec))
            metas.append((p.chirp_mass, target, in_snr))
        scores = score_windows(model, device, np.stack(wins))
        rows += [
            dict(gps=gps, chirp_mass=m, target_snr=t, in_snr=s, score=float(sc))
            for (m, t, s), sc in zip(metas, scores)
        ]
        print(f"  segment {gps} done", flush=True)

    # ---------------- 3. metrics + plots
    print("[3/3] metrics...")
    import pandas as pd

    df = pd.DataFrame(rows)
    df["detected"] = df.score > thresh_zero_fa
    C.RESULTS_DIR.mkdir(exist_ok=True)

    snr_bins = np.linspace(*EVAL_SNR_RANGE, 11)
    mass_edges = [0.17, 0.35, 0.55, 0.88]  # chirp-mass terciles-ish
    mass_labels = ["0.17-0.35", "0.35-0.55", "0.55-0.88"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    eff_curves = {}
    for lo_m, hi_m, lab in zip(mass_edges[:-1], mass_edges[1:], mass_labels):
        sub = df[(df.chirp_mass >= lo_m) & (df.chirp_mass < hi_m)]
        centers, effs = [], []
        for lo_s, hi_s in zip(snr_bins[:-1], snr_bins[1:]):
            s = sub[(sub.in_snr >= lo_s) & (sub.in_snr < hi_s)]
            if len(s) >= 10:
                centers.append((lo_s + hi_s) / 2)
                effs.append(float(s.detected.mean()))
        eff_curves[lab] = (centers, effs)
        axes[0].plot(centers, effs, "o-", label=f"Mc {lab} Msun")
    axes[0].axvline(8, color="k", ls="--", alpha=0.6, label="ideal MF (SNR 8)")
    axes[0].set_xlabel("band-limited optimal SNR in window")
    axes[0].set_ylabel("detection efficiency")
    axes[0].set_title(f"{args.model}: efficiency at zero-FA threshold "
                      f"({total_hours:.1f} h test noise)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # SNR at 50% efficiency per mass bin -> fraction of MF sensitive distance
    fracs = []
    for lab in mass_labels:
        c, e = eff_curves[lab]
        snr50 = float(np.interp(0.5, e, c)) if len(c) > 1 and max(e) >= 0.5 else np.nan
        fracs.append(8.0 / snr50 if np.isfinite(snr50) else 0.0)
    axes[1].bar(mass_labels, fracs, color="steelblue")
    axes[1].axhline(1.0, color="k", ls="--", alpha=0.6)
    axes[1].set_ylabel("fraction of ideal-MF sensitive distance")
    axes[1].set_xlabel("chirp mass bin [Msun]")
    axes[1].set_title("sensitive-distance fraction (1.0 = matched filter)")
    axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(C.RESULTS_DIR / f"efficiency_{args.model}.png", dpi=120)

    out = {
        "model": args.model,
        "test_hours": total_hours,
        "n_noise_windows": int(len(noise_scores)),
        "thresh_zero_fa": thresh_zero_fa,
        "thresh_far_1h": float(far_1h),
        "eff_curves": {k: {"snr": c, "eff": e} for k, (c, e) in eff_curves.items()},
        "mf_distance_fraction": dict(zip(mass_labels, fracs)),
        "overall_eff_above_snr10": float(df[df.in_snr > 10].detected.mean()),
        "overall_eff_above_snr15": float(df[df.in_snr > 15].detected.mean()),
    }
    (C.RESULTS_DIR / f"eval_{args.model}.json").write_text(json.dumps(out, indent=2))
    df.to_parquet(C.RESULTS_DIR / f"injections_{args.model}.parquet")
    print(json.dumps({k: v for k, v in out.items() if k != "eff_curves"}, indent=2))
    print(f"wrote results/eval_{args.model}.json + efficiency_{args.model}.png")


if __name__ == "__main__":
    main()
