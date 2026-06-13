"""Step 03 — the injection framework, demonstrated on real detector noise.

Per the project ground rules this exists BEFORE any search runs on real
post-merger data: synthetic echo trains injected at controlled amplitudes into
real off-source noise are the only honest way to measure our own sensitivity.

Produces a figure: one real noise segment, the same segment with echo trains
injected at decreasing amplitude, and the comb statistic's response to each.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from echolib import RESULTS, GW150914_DT_PRED, comb_score, echo_train, load_segments

EVENT = "GW150914"
AMPS = (6.0, 3.0, 1.5, 0.0)


def main() -> None:
    rng = np.random.default_rng(7)
    segs = load_segments(EVENT, "H1")
    noise = segs.off[10]  # an arbitrary real off-source noise segment
    fs = segs.fs
    dt_grid = np.arange(0.05, 0.5, 0.005)

    fig, axes = plt.subplots(len(AMPS), 2, figsize=(14, 9), sharex="col")
    for row, amp in enumerate(AMPS):
        inj = echo_train(len(noise), fs, GW150914_DT_PRED, amp=amp, rng=rng)
        x = noise + inj
        score = comb_score(x, fs, dt_grid)

        t = np.arange(len(x)) / fs
        axes[row][0].plot(t, x, lw=0.4, color="steelblue")
        if amp > 0:
            axes[row][0].plot(t, inj, lw=0.8, color="crimson", alpha=0.8)
        label = f"amp = {amp}" if amp > 0 else "no injection (pure noise)"
        axes[row][0].set_ylabel(label)

        axes[row][1].plot(dt_grid, score, lw=1.2, color="darkslategray")
        axes[row][1].axvline(
            GW150914_DT_PRED, color="crimson", ls="--", lw=1, alpha=0.7
        )
        axes[row][1].set_ylim(-0.3, 1.6)

    axes[0][0].set_title("real H1 noise + injected echo train (red)")
    axes[0][1].set_title("comb statistic vs candidate spacing (red line = true Δt)")
    axes[-1][0].set_xlabel("time [s]")
    axes[-1][1].set_xlabel("candidate echo spacing Δt [s]")
    fig.tight_layout()
    out = RESULTS / "03_injection_demo.png"
    fig.savefig(out, dpi=140)
    print(f"plot -> {out}")


if __name__ == "__main__":
    main()
