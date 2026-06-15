"""Path G milestone G2a: does a smarter coincidence STATISTIC beat the plain sum?

True 10-ms timing/phase coincidence needs matched-filter arrival times -> a dense
bank -> the intractable wall from G0/F0. So the achievable "finer coincidence" is
a better way to COMBINE the two per-detector cnn_w64 scores. G1 used sum (sH1+sL1).
This re-scores the noise (cheap) + reuses the saved injection scores
(results/coinc_inj.parquet) to compare combination rules at matched FAR, no
re-injection. Cheap: the expensive scoring already happened in G1.

Run:  .venv/bin/python scripts/coinc_stat.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pbh import config as C
from pbh.data import whiten_segment
from pbh.metrics import MASS_LABELS
from pbh.models import make_model
from coinc_eval import WIN, bin_snr50, score_wins


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


# combination rules: (name, elementwise fn of the two per-detector scores)
STATS = {
    "sum  (G1)": lambda a, b: a + b,
    "min": lambda a, b: np.minimum(a, b),
    "prod-prob": lambda a, b: sig(a) * sig(b),       # joint detection probability
    "max+min": lambda a, b: np.maximum(a, b) + 2 * np.minimum(a, b),  # weight the weaker site
}


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = make_model("cnn")
    model.load_state_dict(torch.load(C.MODEL_DIR / "cnn_w64.pt", map_location=device))
    model.to(device).eval()

    manifest = json.loads((C.DATA_DIR / "manifest.json").read_text())
    segs = [g for g in manifest["L1"]["coinc"] if g in manifest["H1"]["test"]]
    crop = C.WHITEN_CROP_SEC * C.SAMPLE_RATE

    noise = {"H1": {}, "L1": {}}
    t0 = time.time()
    for ifo in ("H1", "L1"):
        for g in segs:
            w, _, _ = whiten_segment(ifo, g)
            wc = w[crop:-crop]
            nwin = (len(wc) - WIN) // WIN
            noise[ifo][g] = score_wins(model, device, [wc[i * WIN:(i + 1) * WIN] for i in range(nwin)])
        print(f"  re-scored noise {ifo} ({time.time()-t0:.0f}s)", flush=True)

    df = pd.read_parquet(C.RESULTS_DIR / "coinc_inj.parquet")
    thr_single = float(np.concatenate([noise["H1"][g] for g in segs]).max())
    n_lag_total = sum(min(len(noise["H1"][g]), len(noise["L1"][g])) - 1 for g in segs)

    # single-det baseline (network-SNR axis)
    df["_s"] = df.sH1 > thr_single
    base = bin_snr50(df, "_s", 10)

    print(f"\n=== G2a: coincidence combination rules (matched FAR, network-SNR axis) ===")
    print(f"single-det baseline distance: " +
          "  ".join(f"{m}={base[m][1]:.3f}" for m in MASS_LABELS))
    print(f"\n{'statistic':>12}  " + "  ".join(f"{m:>11}" for m in MASS_LABELS) + "   mean-gain")
    out = {"single_det": {m: base[m][1] for m in MASS_LABELS}, "rules": {}}
    a_inj, b_inj = df.sH1.to_numpy(), df.sL1.to_numpy()
    for name, fn in STATS.items():
        bg = np.concatenate([fn(noise["H1"][g][:n], np.roll(noise["L1"][g][:n], lag))
                             for g in segs
                             for n in [min(len(noise["H1"][g]), len(noise["L1"][g]))]
                             for lag in range(1, n)])
        thr = float(np.sort(bg)[-n_lag_total])
        df["_d"] = fn(a_inj, b_inj) > thr
        r = bin_snr50(df, "_d", 10)
        gains = [r[m][1] / base[m][1] if base[m][1] > 0 else np.nan for m in MASS_LABELS]
        out["rules"][name.strip()] = {m: r[m][1] for m in MASS_LABELS}
        print(f"{name:>12}  " + "  ".join(f"{r[m][1]:>11.3f}" for m in MASS_LABELS)
              + f"   {np.nanmean(gains):.2f}x")
    (C.RESULTS_DIR / "coinc_stat.json").write_text(json.dumps(out, indent=2))
    print("\nwrote coinc_stat.json  (gain = vs single-det; G1 'sum' was ~1.3-1.5x)")


if __name__ == "__main__":
    main()
