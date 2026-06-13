"""Step 10 — v5: the fair head-to-head. Both statistics (v2 ML scorer, v1 raw
comb) on the SAME raw-injected, re-whitened segments. The production-path
sensitivity ratio becomes a measurement. Pre-registration Y1-Y2 (2026-06-13).
"""

import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from echolib import DETECTORS, GW150914_DT_PRED, RESULTS, comb_on_env, comb_score, envelope, fetch_block, progress

spec9 = importlib.util.spec_from_file_location(
    "raw9", Path(__file__).resolve().parent / "09_raw_injection.py"
)
raw9 = importlib.util.module_from_spec(spec9)
spec9.loader.exec_module(raw9)
ml = raw9.ml

FS = 4096.0
AMPS = (0.5, 1.0, 1.5, 2.0, 3.0)
N_BG, N_TRIALS = 30, 25
rng = np.random.default_rng(53)


def main() -> None:
    raws = {det: fetch_block(det, "GW150914") for det in DETECTORS}
    t0 = float(raws["H1"].t0.value)
    centers = t0 + 308 + 4 * np.arange(42)
    dt_grid = np.arange(0.05, 0.5, 0.005)
    j = int(np.argmin(np.abs(dt_grid - GW150914_DT_PRED)))
    slope = float(np.mean(list(
        json.loads((RESULTS / "09_raw_injection.json").read_text())["slopes"].values())))

    models = {}
    for det in DETECTORS:
        m = ml.ConvAE()
        m.load_state_dict(torch.load(RESULTS / f"07_scorer_{det}.pt", weights_only=True))
        m.eval()
        models[det] = m

    def both_scores(c, inj=None):
        tot_ml = np.zeros(len(dt_grid))
        tot_cb = np.zeros(len(dt_grid))
        for det in DETECTORS:
            seg = raw9.whitened_segment(raws[det], c, inj)
            tot_ml += comb_on_env(ml.error_envelope(models[det], seg, FS), FS, dt_grid)
            tot_cb += comb_score(seg, FS, dt_grid)
        return float(tot_ml[j]), float(tot_cb[j])

    bg_ml, bg_cb = [], []
    for i in range(N_BG):
        a, b = both_scores(float(centers[i % len(centers)]))
        bg_ml.append(a)
        bg_cb.append(b)
        progress("10_h2h_bg", i, N_BG)
    th_ml = float(np.quantile(bg_ml, 0.95))
    th_cb = float(np.quantile(bg_cb, 0.95))
    print(f"Y1 backgrounds (n={N_BG}): ML 95th = {th_ml:.3f}, comb 95th = {th_cb:.3f}")

    eff = {"ml": {}, "comb": {}}
    for amp in AMPS:
        A = amp / slope
        h_ml = h_cb = 0
        for i in range(N_TRIALS):
            c = float(centers[rng.integers(0, len(centers))])
            a, b = both_scores(c, lambda t: raw9.raw_train(t, c + 0.05, A))
            h_ml += a > th_ml
            h_cb += b > th_cb
            progress(f"10_h2h_{amp}", i, N_TRIALS)
        eff["ml"][amp] = h_ml / N_TRIALS
        eff["comb"][amp] = h_cb / N_TRIALS
        print(f"Y2 {amp:.1f} sigma-equiv: ML {100*eff['ml'][amp]:3.0f}%   "
              f"comb {100*eff['comb'][amp]:3.0f}%")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(AMPS, [eff["ml"][a] for a in AMPS], "o-", color="crimson",
            label="v2 ML scorer (production path)")
    ax.plot(AMPS, [eff["comb"][a] for a in AMPS], "s--", color="darkslategray",
            label="v1 comb (production path)")
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_xlabel("injected amplitude [whitened-σ equivalent, raw-strain injection]")
    ax.set_ylabel("recovery fraction (p < 0.05 vs own background)")
    ax.set_title("GW150914: the FAIR head-to-head — identical raw-injection path")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    fig.tight_layout()
    out = RESULTS / "10_head_to_head.png"
    fig.savefig(out, dpi=140)
    print(f"plot -> {out}")
    (RESULTS / "10_head_to_head.json").write_text(json.dumps(
        {"thresholds": {"ml": th_ml, "comb": th_cb}, "eff": eff}, indent=1))


if __name__ == "__main__":
    main()
