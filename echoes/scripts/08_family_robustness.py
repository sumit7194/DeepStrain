"""Step 08 — v3: is the ML scorer's 13× advantage family-robust?

Reuses the saved v2 scorers and the existing background threshold; injects echo
trains with varied waveform-family parameters (one at a time from the v2
baseline f0=250 Hz, tau=20 ms, gamma=0.7) into eval-pool noise. Includes a
deliberate out-of-band control (f0=450 Hz vs the 30-350 Hz bandpass) that MUST
collapse. Pre-registration (W1, W2) in notes/lab_notebook.md.
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
from echolib import DETECTORS, GW150914_DT_PRED, RESULTS, comb_on_env, echo_train, load_segments

spec = importlib.util.spec_from_file_location(
    "ml", Path(__file__).resolve().parent / "07_ml_scorer.py"
)
ml = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ml)

EVENT = "GW150914"
AMPS = (0.2, 0.5, 1.0)
N_TRIALS = 30
BASE = dict(f0=250.0, tau=0.02, gamma=0.7)
CONFIGS = [
    ("baseline", BASE),
    ("f0=150", dict(BASE, f0=150.0)),
    ("f0=320", dict(BASE, f0=320.0)),
    ("tau=10ms", dict(BASE, tau=0.01)),
    ("tau=50ms", dict(BASE, tau=0.05)),
    ("gamma=0.5", dict(BASE, gamma=0.5)),
    ("gamma=0.9", dict(BASE, gamma=0.9)),
    ("f0=450 (out-of-band control)", dict(BASE, f0=450.0)),
]


def main() -> None:
    rng = np.random.default_rng(31)
    dt_grid = np.arange(0.05, 0.5, 0.005)
    j_pred = int(np.argmin(np.abs(dt_grid - GW150914_DT_PRED)))

    all_segs = {det: load_segments(EVENT, det) for det in DETECTORS}
    fs = all_segs["H1"].fs
    n_off = min(len(all_segs[d].off) for d in DETECTORS)
    eval_list = list(range(ml.N_TRAIN_PAIRS, n_off))
    n_samp = len(all_segs["H1"].off[0])

    models = {}
    for det in DETECTORS:
        m = ml.ConvAE()
        m.load_state_dict(torch.load(RESULTS / f"07_scorer_{det}.pt", weights_only=True))
        m.eval()
        models[det] = m
    prev = np.load(RESULTS / "07_ml_scorer.npy", allow_pickle=True).item()
    thresh = prev["thresh"]
    print(f"reusing scorers + threshold {thresh:.3f} (99th pct of v2 background)")

    table = {}
    for name, cfg in CONFIGS:
        table[name] = {}
        for amp in AMPS:
            hits = 0
            for _ in range(N_TRIALS):
                i = eval_list[int(rng.integers(0, len(eval_list)))]
                inj = echo_train(
                    n_samp, fs, GW150914_DT_PRED, amp=amp,
                    f0=cfg["f0"], tau=cfg["tau"], gamma=cfg["gamma"], rng=rng,
                )
                total = np.zeros(len(dt_grid))
                for det in DETECTORS:
                    total += comb_on_env(
                        ml.error_envelope(models[det], all_segs[det].off[i] + inj, fs),
                        fs, dt_grid,
                    )
                hits += float(total[j_pred]) > thresh
            table[name][amp] = hits / N_TRIALS
        print(f"  {name:32s}: " + "  ".join(
            f"{amp}σ → {100 * table[name][amp]:3.0f}%" for amp in AMPS))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(CONFIGS))
    width = 0.25
    for k, amp in enumerate(AMPS):
        ax.bar(x + (k - 1) * width, [table[n][amp] for n, _ in CONFIGS], width,
               label=f"{amp}σ")
    ax.axhline(0.8, color="crimson", ls="--", lw=1, label="W1 gate (80% @ 0.5σ, in-band)")
    ax.set_xticks(x)
    ax.set_xticklabels([n for n, _ in CONFIGS], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("recovery fraction (p < 0.01)")
    ax.set_title("v3: family-robustness of the ML scorer's sensitivity")
    ax.legend()
    fig.tight_layout()
    out = RESULTS / "08_family_robustness.png"
    fig.savefig(out, dpi=140)
    print(f"plot -> {out}")
    (RESULTS / "08_family_robustness.json").write_text(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
