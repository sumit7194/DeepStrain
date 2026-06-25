"""Per-event runner: on-source search + background + p-values for any event.

This is the catalog-scaling entry point. Each event carries its own
pre-registered configuration (from Abedi et al., PRD 96, 082004, Table I, plus
a search band scaled to the remnant's ringdown frequency — lighter remnant,
higher frequency):

Usage:
    python scripts/run_event.py GW150914
    python scripts/run_event.py GW151226
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from echolib import DETECTORS, RESULTS, comb_score, load_segments

# Pre-registered per-event configuration. dt_pred from Abedi et al. Table I;
# f0 ~ remnant ringdown frequency (sets the search band, injections in step 06).
EVENTS = {
    "GW150914": {"dt_pred": 0.2925, "f0": 250.0, "band": (30.0, 350.0)},
    "LVT151012": {"dt_pred": 0.1778, "f0": 430.0, "band": (30.0, 600.0)},  # Abedi Table I (was wrongly 0.1013)
    "GW151226": {"dt_pred": 0.1013, "f0": 750.0, "band": (30.0, 900.0)},   # Abedi Table I (was wrongly 0.0579)
}


def main() -> None:
    event = sys.argv[1] if len(sys.argv) > 1 else "GW150914"
    cfg = EVENTS[event]
    dt_pred = cfg["dt_pred"]

    # spacing grid: wide agnostic range, but never narrower than half the
    # prediction so the comb's teeth fit inside the 3 s segment
    lo = max(0.03, dt_pred / 2)
    dt_grid = np.arange(lo, 0.5, 0.002)
    j_pred = int(np.argmin(np.abs(dt_grid - dt_pred)))

    all_segs = {
        det: load_segments(event, det, band=cfg["band"]) for det in DETECTORS
    }
    fs = all_segs["H1"].fs
    n_off = min(len(all_segs[d].off) for d in DETECTORS)

    def network(pair: dict[str, np.ndarray]) -> np.ndarray:
        total = np.zeros(len(dt_grid))
        for x in pair.values():
            total += comb_score(x, fs, dt_grid)
        return total

    # on-source
    on_curve = network({det: s.on for det, s in all_segs.items()})
    on_max, on_pred = float(on_curve.max()), float(on_curve[j_pred])

    # background
    bg_max, bg_pred = [], []
    for i in range(n_off):
        c = network({det: all_segs[det].off[i] for det in DETECTORS})
        bg_max.append(float(c.max()))
        bg_pred.append(float(c[j_pred]))
    bg_max, bg_pred = np.array(bg_max), np.array(bg_pred)

    def pval(s: float, bg: np.ndarray) -> float:
        return (np.sum(bg >= s) + 1) / (len(bg) + 1)

    p_max, p_pred = pval(on_max, bg_max), pval(on_pred, bg_pred)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    axes[0].plot(dt_grid, on_curve, lw=1.2, color="black")
    axes[0].axvline(dt_pred, color="crimson", ls="--", lw=1.2)
    axes[0].set_xlabel("candidate Δt [s]")
    axes[0].set_title(f"{event} on-source comb (red = predicted Δt)")

    for ax, bg, s, p, name in (
        (axes[1], bg_max, on_max, p_max, "A: max over grid"),
        (axes[2], bg_pred, on_pred, p_pred, f"B: at Δt = {dt_pred} s"),
    ):
        ax.hist(bg, bins=30, color="lightsteelblue", edgecolor="white")
        ax.axvline(s, color="crimson", lw=2, label=f"on-source\np = {p:.3f}")
        ax.set_title(name)
        ax.legend()
    fig.suptitle(f"{event}: echo search vs {n_off} background segment pairs")
    fig.tight_layout()
    out = RESULTS / f"{event}_run.png"
    fig.savefig(out, dpi=140)

    print(f"plot -> {out}")
    print(f"{event}:  A (max over grid) p = {p_max:.3f}   "
          f"B (at predicted Δt = {dt_pred}s) p = {p_pred:.3f}")


if __name__ == "__main__":
    main()
