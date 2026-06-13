"""Stage 1 eval: score the trained learned model through the oracle convention.

Same 64-s window / zero-FA / 6-real-noise-segment / per-segment-proper-whitening
protocol as semicoherent_oracle.py and cnn_w64 — so the learned model's
mf_distance_fraction is directly comparable to the oracle ceiling (0.66/0.76/0.75)
and to cnn_w64 (0.41/0.46/0.48). Per rung 2, the detector is per-window (max);
no aggregation.

Run:  .venv/bin/python scripts/eval_semicoherent.py [--weights semicoherent] [--smoke]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pbh import config as C
from pbh.data import whiten_segment
from pbh.metrics import MASS_LABELS, mass_bin_results
from pbh.models import make_model
from pbh.progress import progress
from pbh.waveforms import make_whitened_injection, sample_params

WIN = 64 * C.SAMPLE_RATE
CROP = C.WHITEN_CROP_SEC * C.SAMPLE_RATE
N_INJ_PER_SEG = 250
SEED = C.SEED + 3333
CNN_W64 = {"0.17-0.35": 0.407, "0.35-0.55": 0.457, "0.55-0.88": 0.476}
ORACLE = {"0.17-0.35": 0.663, "0.35-0.55": 0.764, "0.55-0.88": 0.752}


@torch.no_grad()
def score(model, device, windows: np.ndarray) -> np.ndarray:
    out = []
    for i in range(0, len(windows), 64):
        b = torch.from_numpy(windows[i : i + 64]).float().unsqueeze(1).to(device)
        out.append(model(b).float().cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="semicoherent")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    tag = "_smoke" if args.smoke else ""
    n_inj = 8 if args.smoke else N_INJ_PER_SEG

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = make_model("semicoherent")
    model.load_state_dict(torch.load(C.MODEL_DIR / f"{args.weights}.pt", map_location=device))
    model.to(device).eval()
    test_segs = json.loads((C.DATA_DIR / "manifest.json").read_text())["H1"]["test"]

    # --- noise: non-overlapping 64-s windows -> zero-FA threshold (oracle convention)
    import pandas as pd

    rows, t0 = [], time.time()
    thresh = -np.inf
    n_noise = 0
    for si, gps in enumerate(test_segs):
        w, tgps, psd = whiten_segment("H1", gps)
        wc = w[CROP:-CROP]
        nwin = len(wc) // WIN
        wins = wc[: nwin * WIN].reshape(nwin, WIN)
        ns = score(model, device, wins)
        thresh = max(thresh, float(ns.max()))
        n_noise += nwin
        # --- injections into this segment (proper per-segment whitening)
        rng = np.random.default_rng([SEED, int(gps)])
        iwins, metas = [], []
        for _ in range(n_inj):
            p = sample_params(rng)
            h_w, snr_ref = make_whitened_injection(p, "H1", tgps, psd)
            target = float(rng.uniform(*C.EVAL_SNR_RANGE))
            sig = (h_w * (target / snr_ref))[-WIN:]
            base = int(rng.integers(CROP, len(wc) - WIN))
            win = wc[base : base + WIN].copy()
            m = int(rng.integers(WIN // 2, WIN))
            win[:m] += sig[WIN - m :]
            iwins.append(win)
            metas.append((p.chirp_mass, target))
        isc = score(model, device, np.stack(iwins))
        rows += [dict(gps=gps, chirp_mass=mc, target_snr=t, score=float(s))
                 for (mc, t), s in zip(metas, isc)]
        progress("eval_semicoherent" + tag, si + 1, len(test_segs), elapsed_s=time.time() - t0)
        print(f"[{gps}] {nwin} noise win, {n_inj} inj done", flush=True)

    df = pd.DataFrame(rows)
    df["detected"] = df.score > thresh
    res = mass_bin_results(df, "detected", min_count=3 if args.smoke else 10)
    frac = res["mf_distance_fraction"]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(MASS_LABELS))
    ax.bar(x - 0.25, [frac[m] for m in MASS_LABELS], 0.25, label="learned (this)", color="tab:green")
    ax.bar(x, [CNN_W64[m] for m in MASS_LABELS], 0.25, label="cnn_w64", color="tab:gray")
    ax.bar(x + 0.25, [ORACLE[m] for m in MASS_LABELS], 0.25, label="oracle ceiling", color="tab:red", alpha=0.6)
    ax.axhline(1.0, color="k", ls="--", alpha=0.4)
    ax.set_xticks(x, MASS_LABELS); ax.set_ylabel("fraction of ideal-MF distance")
    ax.set_title(f"learned semi-coherent ({len(df)} inj, zero-FA)"); ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(C.RESULTS_DIR / f"eval_semicoherent{tag}.png", dpi=120)

    out = {"weights": args.weights, "n_injections": int(len(df)), "n_noise_windows": n_noise,
           "threshold": thresh, "mf_distance_fraction": frac,
           "cnn_w64": CNN_W64, "oracle_ceiling": ORACLE, "eff_curves": res["eff_curves"],
           "seed": SEED}
    (C.RESULTS_DIR / f"eval_semicoherent{tag}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({"learned": {m: round(frac[m], 3) for m in MASS_LABELS},
                      "cnn_w64": CNN_W64, "oracle": ORACLE}, indent=2))
    print(f"wrote eval_semicoherent{tag}.json + .png")


if __name__ == "__main__":
    main()
