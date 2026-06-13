"""Step 04 — the on-source search: GW150914's post-ringdown segment.

Runs the network comb statistic (H1 + L1 at the same candidate spacing) on the
3 s after the merger and plots the score across the spacing grid, with the
pre-registered prediction Δt = 0.2925 s marked.

This script reports the SCORE only. Significance comes exclusively from
step 05 (background estimation) — never from eyeballing this plot.
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
    detection_statistic,
    load_segments,
)

EVENT = "GW150914"


def main() -> None:
    dt_grid = np.arange(0.05, 0.5, 0.005)

    all_segs = {det: load_segments(EVENT, det) for det in DETECTORS}
    fs = all_segs["H1"].fs
    on = {det: s.on for det, s in all_segs.items()}

    fig, ax = plt.subplots(figsize=(10, 5))
    total = np.zeros(len(dt_grid))
    for det, x in on.items():
        s = comb_score(x, fs, dt_grid)
        total += s
        ax.plot(dt_grid, s, lw=1, alpha=0.6, label=f"{det}")
    ax.plot(dt_grid, total, lw=2, color="black", label="network (H1+L1)")
    ax.axvline(GW150914_DT_PRED, color="crimson", ls="--", lw=1.2,
               label=f"predicted Δt = {GW150914_DT_PRED} s")

    score, best_dt = detection_statistic(on, fs, dt_grid)
    # the pre-registered test: the network score AT the predicted spacing
    j = int(np.argmin(np.abs(dt_grid - GW150914_DT_PRED)))
    score_at_pred = float(total[j])

    ax.set_xlabel("candidate echo spacing Δt [s]")
    ax.set_ylabel("comb score")
    ax.set_title(f"{EVENT} post-ringdown — network comb statistic")
    ax.legend()
    fig.tight_layout()
    out = RESULTS / "04_onsource_search.png"
    fig.savefig(out, dpi=140)

    np.save(RESULTS / "04_onsource_scores.npy",
            {"dt_grid": dt_grid, "total": total,
             "max_score": score, "best_dt": best_dt,
             "score_at_pred": score_at_pred},
            allow_pickle=True)

    print(f"plot -> {out}")
    print(f"max network score: {score:.3f} at Δt = {best_dt:.3f} s")
    print(f"score at predicted Δt ({GW150914_DT_PRED} s): {score_at_pred:.3f}")
    print("significance: see step 05 — a score means nothing without background.")


if __name__ == "__main__":
    main()
