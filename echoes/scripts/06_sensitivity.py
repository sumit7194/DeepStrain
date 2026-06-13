"""Step 06 — the money plot: injection-recovery sensitivity curve.

For a grid of injection amplitudes, inject synthetic echo trains (at the
predicted spacing) into many real off-source noise segment pairs, run the
identical network statistic, and measure the fraction recovered above the
background's 99th percentile (i.e. detected at p < 0.01).

The result is the honest sentence this project exists to produce:
   "this pipeline detects echo trains of amplitude >= X (whitened-noise sigma)
    with Y% probability; at smaller amplitudes it is blind."
Any upper limit or detection claim inherits its meaning from this curve.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from echolib import (
    DETECTORS,
    GW150914_DT_PRED,
    RESULTS,
    comb_score,
    echo_train,
    load_segments,
)

EVENT = "GW150914"
AMPS = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0])
N_TRIALS = 60  # injections per amplitude


def main() -> None:
    rng = np.random.default_rng(2025)
    dt_grid = np.arange(0.05, 0.5, 0.005)
    j_pred = int(np.argmin(np.abs(dt_grid - GW150914_DT_PRED)))

    all_segs = {det: load_segments(EVENT, det) for det in DETECTORS}
    fs = all_segs["H1"].fs
    n_off = min(len(all_segs[d].off) for d in DETECTORS)
    n_samp = len(all_segs["H1"].off[0])

    # background threshold at p < 0.01, from step 05's saved scores
    bg = np.load(RESULTS / "05_background.npy", allow_pickle=True).item()
    thresh = float(np.quantile(bg["bg_pred"], 0.99))
    print(f"detection threshold (99th pct of background, statistic B): {thresh:.3f}")

    def trial(amp: float) -> bool:
        i = int(rng.integers(0, n_off))
        # the SAME astrophysical signal arrives in both detectors (same dt,
        # same morphology); detector noise differs.
        inj = echo_train(n_samp, fs, GW150914_DT_PRED, amp=amp, rng=rng)
        total = np.zeros(len(dt_grid))
        for det in DETECTORS:
            total += comb_score(all_segs[det].off[i] + inj, fs, dt_grid)
        return float(total[j_pred]) > thresh

    eff, err = [], []
    for amp in AMPS:
        hits = sum(trial(amp) for _ in range(N_TRIALS))
        p = hits / N_TRIALS
        eff.append(p)
        err.append(np.sqrt(p * (1 - p) / N_TRIALS))
        print(f"amp {amp:4.1f}: {hits}/{N_TRIALS} recovered ({100*p:.0f}%)")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(AMPS, eff, yerr=err, marker="o", lw=1.5, capsize=3,
                color="darkslategray")
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.axhline(0.9, color="gray", ls=":", lw=1)
    ax.set_xlabel("injected first-pulse amplitude [whitened-noise σ]")
    ax.set_ylabel("recovery fraction (p < 0.01 vs background)")
    ax.set_title(f"{EVENT}-like echo trains: injection-recovery curve\n"
                 f"(Δt = {GW150914_DT_PRED} s, reflectivity 0.7, 6 echoes, "
                 f"real {'+'.join(DETECTORS)} noise)")
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    out = RESULTS / "06_sensitivity.png"
    fig.savefig(out, dpi=140)

    np.save(RESULTS / "06_sensitivity.npy",
            {"amps": AMPS, "eff": np.array(eff), "err": np.array(err),
             "threshold": thresh}, allow_pickle=True)
    print(f"plot -> {out}")


if __name__ == "__main__":
    main()
