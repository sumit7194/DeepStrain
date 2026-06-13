"""Step 05 — background estimation: THE step that decides everything.

Runs the identical network statistic on every event-free off-source segment
pair, building the score's distribution under the noise-only hypothesis. The
on-source score's significance is then simply its rank among the background:

    p = (number of background scores >= on-source score + 1) / (N + 1)

Two statistics are scored, both fixed in advance:
  A. "max over grid"  — the score maximised over the whole spacing grid
                        (search without a prediction; pays a trials factor)
  B. "at predicted Δt" — the score at the pre-registered spacing for this
                        event (the sharper, theory-driven test)
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
    load_segments,
)

EVENT = "GW150914"


def network_scores(seg_pair: dict[str, np.ndarray], fs: float,
                   dt_grid: np.ndarray, j_pred: int) -> tuple[float, float]:
    total = np.zeros(len(dt_grid))
    for x in seg_pair.values():
        total += comb_score(x, fs, dt_grid)
    return float(total.max()), float(total[j_pred])


def main() -> None:
    dt_grid = np.arange(0.05, 0.5, 0.005)
    j_pred = int(np.argmin(np.abs(dt_grid - GW150914_DT_PRED)))

    all_segs = {det: load_segments(EVENT, det) for det in DETECTORS}
    fs = all_segs["H1"].fs
    n_off = min(len(all_segs[d].off) for d in DETECTORS)
    print(f"background population: {n_off} segment pairs")

    bg_max, bg_pred = [], []
    for i in range(n_off):
        pair = {det: all_segs[det].off[i] for det in DETECTORS}
        m, p = network_scores(pair, fs, dt_grid, j_pred)
        bg_max.append(m)
        bg_pred.append(p)
    bg_max = np.array(bg_max)
    bg_pred = np.array(bg_pred)

    on = {det: all_segs[det].on for det in DETECTORS}
    on_max, on_pred = network_scores(on, fs, dt_grid, j_pred)

    def pval(on_score: float, bg: np.ndarray) -> float:
        return (np.sum(bg >= on_score) + 1) / (len(bg) + 1)

    p_max = pval(on_max, bg_max)
    p_pred = pval(on_pred, bg_pred)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, bg, on_s, p, title in (
        (axes[0], bg_max, on_max, p_max, "statistic A: max over Δt grid"),
        (axes[1], bg_pred, on_pred, p_pred,
         f"statistic B: at predicted Δt = {GW150914_DT_PRED} s"),
    ):
        ax.hist(bg, bins=30, color="lightsteelblue", edgecolor="white")
        ax.axvline(on_s, color="crimson", lw=2,
                   label=f"on-source = {on_s:.3f}\np = {p:.3f}")
        ax.set_xlabel("network comb score")
        ax.set_title(title)
        ax.legend()
    axes[0].set_ylabel(f"count (N = {len(bg_max)} noise segments)")
    fig.suptitle(f"{EVENT}: on-source score vs noise-only background")
    fig.tight_layout()
    out = RESULTS / "05_background.png"
    fig.savefig(out, dpi=140)

    np.save(RESULTS / "05_background.npy",
            {"bg_max": bg_max, "bg_pred": bg_pred,
             "on_max": on_max, "on_pred": on_pred,
             "p_max": p_max, "p_pred": p_pred}, allow_pickle=True)

    print(f"plot -> {out}")
    print(f"A (max over grid):    on-source {on_max:.3f}, p = {p_max:.3f}")
    print(f"B (at predicted Δt):  on-source {on_pred:.3f}, p = {p_pred:.3f}")


if __name__ == "__main__":
    main()
