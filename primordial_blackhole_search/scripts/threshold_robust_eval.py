"""Second-pass B: is stage-1's 0.000 an artifact of ONE glitch setting the bar?

Re-scores the SAME noise windows + injections as eval_semicoherent.py (identical
seeds/protocol — this is a re-thresholding, not a new experiment), then reports
mf_distance_fraction under a ladder of threshold policies from strict to glitch-
robust. If 0.000 -> meaningfully nonzero as we drop the top noise outlier(s), the
wall was partly the eval (no glitch veto); if it stays ~0, the model genuinely
can't separate signal from noise and the glitch was not the story.

Run: .venv/bin/python scripts/threshold_robust_eval.py --weights semicoherent_v2 --arch semicoherent_v2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pbh import config as C
from pbh.data import whiten_segment
from pbh.metrics import MASS_LABELS, mass_bin_results
from pbh.models import make_model
from pbh.waveforms import make_whitened_injection, sample_params

WIN = 64 * C.SAMPLE_RATE
CROP = C.WHITEN_CROP_SEC * C.SAMPLE_RATE
N_INJ_PER_SEG = 250
SEED = C.SEED + 3333  # MUST match eval_semicoherent.py for a like-for-like re-threshold
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
    ap.add_argument("--weights", default="semicoherent_v2")
    ap.add_argument("--arch", default="semicoherent_v2",
                    choices=["semicoherent", "semicoherent_v2"])
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = make_model(args.arch)
    model.load_state_dict(torch.load(C.MODEL_DIR / f"{args.weights}.pt", map_location=device))
    model.to(device).eval()
    test_segs = json.loads((C.DATA_DIR / "manifest.json").read_text())["H1"]["test"]

    noise_scores, rows, t0 = [], [], time.time()
    for gps in test_segs:
        w, tgps, psd = whiten_segment("H1", gps)
        wc = w[CROP:-CROP]
        nwin = len(wc) // WIN
        wins = wc[: nwin * WIN].reshape(nwin, WIN)
        noise_scores.append(score(model, device, wins))
        rng = np.random.default_rng([SEED, int(gps)])
        iwins, metas = [], []
        for _ in range(N_INJ_PER_SEG):
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
        rows += [dict(chirp_mass=mc, target_snr=t, score=float(s))
                 for (mc, t), s in zip(metas, isc)]
        print(f"[{gps}] {nwin} noise, {N_INJ_PER_SEG} inj  ({time.time()-t0:.0f}s)", flush=True)

    ns = np.sort(np.concatenate(noise_scores))[::-1]  # descending
    n_noise = len(ns)
    df = pd.DataFrame(rows)
    hours = n_noise * 64 / 3600.0

    # threshold ladder: strict -> glitch-robust, each with its implied false-alarm rate
    policies = [
        ("max (0 FA)", ns[0], "0 in %.1fh" % hours),
        ("drop top-1", ns[1], "1 in %.1fh" % hours),
        ("drop top-2", ns[2], "2 in %.1fh" % hours),
        ("drop top-1%%", ns[max(1, n_noise // 100)], "~1%% FA"),
        ("p99", float(np.percentile(np.concatenate(noise_scores), 99)), "1%% FA"),
        ("p95", float(np.percentile(np.concatenate(noise_scores), 95)), "5%% FA"),
    ]

    print(f"\n=== {args.weights} | {n_noise} noise win ({hours:.1f}h), {len(df)} inj ===")
    print(f"{'policy':<14}{'thresh':>9}   {'far':<12}  " + "  ".join(f"{m:>11}" for m in MASS_LABELS))
    table = {}
    for name, thr, far in policies:
        df["detected"] = df.score > thr
        frac = mass_bin_results(df, "detected")["mf_distance_fraction"]
        table[name] = {"threshold": float(thr), "far": far, "frac": frac}
        print(f"{name:<14}{thr:>9.3f}   {far:<12}  " +
              "  ".join(f"{frac[m]:>11.3f}" for m in MASS_LABELS))
    print(f"{'cnn_w64 ref':<14}{'':<9}   {'(max thr)':<12}  " +
          "  ".join(f"{CNN_W64[m]:>11.3f}" for m in MASS_LABELS))
    print(f"{'oracle ref':<14}{'':<9}   {'(+chi2 veto)':<12}  " +
          "  ".join(f"{ORACLE[m]:>11.3f}" for m in MASS_LABELS))

    out = {"weights": args.weights, "arch": args.arch, "n_noise_windows": n_noise,
           "noise_hours": hours, "n_injections": int(len(df)),
           "top8_noise_scores": [float(x) for x in ns[:8]],
           "policies": table, "cnn_w64": CNN_W64, "oracle_ceiling": ORACLE, "seed": SEED}
    (C.RESULTS_DIR / f"threshold_robust_{args.weights}.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote threshold_robust_{args.weights}.json")


if __name__ == "__main__":
    main()
