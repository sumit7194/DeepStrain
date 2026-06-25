"""Step 11 — turn the echo NON-DETECTION into an UPPER LIMIT (roadmap P1 + P2).

06 measured recovery vs amplitude at the ONE predicted spacing. This generalises
it to a 2-D (amplitude x spacing Delta-t) efficiency map at HIGHER N, then reads
off, per Delta-t, the first-pulse amplitude A90 at which 90% of injections are
recovered above that spacing's p<0.01 background threshold.

Because the on-source data showed NO detection (05: p_pred>0.01 for both events),
any echo train louder than A90(Delta-t) would have been seen with >=90% prob and
was not -> we EXCLUDE first-pulse amplitude >= A90(Delta-t), at each spacing.
That converts "we found nothing" into a quantitative exclusion curve.

Per-Delta-t background (not just the predicted-Delta-t one from 05) makes the
limit honest at every spacing. Comb statistic (the v1 baseline that produced the
on-source null); the ML scorer (07) would tighten A90 by ~1.2x (v5).

Run:  .venv/bin/python scripts/11_upper_limits.py [--event GW150914] [--smoke]
"""

import argparse
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
    progress,
)

DT_PRED = {"GW150914": GW150914_DT_PRED, "GW151226": 0.1013}  # Abedi 2017 Table I (was wrongly 0.0579; verified by 14_echo_spacing.py)
AMPS = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", default="GW150914", choices=list(DT_PRED))
    ap.add_argument("--n-trials", type=int, default=300)   # P2: high N for decisive efficiency
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    n_trials = 30 if args.smoke else args.n_trials
    event, dt_pred = args.event, DT_PRED[args.event]
    rng = np.random.default_rng(2025)

    # spacing grid for the exclusion curve (subsample the search grid + pin the prediction)
    dt_ul = np.unique(np.append(np.arange(0.05, 0.49, 0.03), dt_pred))

    all_segs = {det: load_segments(event, det) for det in DETECTORS}
    fs = all_segs["H1"].fs
    n_off = min(len(all_segs[d].off) for d in DETECTORS)
    n_samp = len(all_segs["H1"].off[0])
    print(f"{event}: {n_off} off-source pairs, fs={fs:.0f}, {n_trials} trials/cell, "
          f"{len(dt_ul)} spacings x {len(AMPS)} amps")

    def score_at(x_pair, dt):
        """Network comb statistic (sum over detectors) at one spacing dt."""
        g = np.array([dt])
        return float(sum(comb_score(x_pair[d], fs, g)[0] for d in DETECTORS))

    # --- per-Δt background threshold (99th pct over off-source pairs) = p<0.01
    thr = {}
    for jdt, dt in enumerate(dt_ul):
        bg = [score_at({d: all_segs[d].off[i] for d in DETECTORS}, dt) for i in range(n_off)]
        thr[dt] = float(np.quantile(bg, 0.99))
        progress(f"11_ul_{event}_bg", jdt + 1, len(dt_ul))
    print(f"per-Δt thresholds ready ({len(dt_ul)} spacings)")

    # --- efficiency map + A90 / A50 per spacing
    a90, a50, eff_map = [], [], np.zeros((len(dt_ul), len(AMPS)))
    total_cells = len(dt_ul) * len(AMPS)
    for jdt, dt in enumerate(dt_ul):
        eff = np.empty(len(AMPS))
        for ja, amp in enumerate(AMPS):
            hits = 0
            for _ in range(n_trials):
                i = int(rng.integers(0, n_off))
                inj = echo_train(n_samp, fs, dt, amp=amp, rng=rng)
                pair = {d: all_segs[d].off[i] + inj for d in DETECTORS}
                hits += score_at(pair, dt) > thr[dt]
            eff[ja] = hits / n_trials
            progress(f"11_ul_{event}_eff", jdt * len(AMPS) + ja + 1, total_cells)
        eff_map[jdt] = eff
        # A_X = amplitude where efficiency first reaches X (monotone interp; nan if never)
        a90.append(float(np.interp(0.9, eff, AMPS)) if eff.max() >= 0.9 else np.nan)
        a50.append(float(np.interp(0.5, eff, AMPS)) if eff.max() >= 0.5 else np.nan)
        print(f"  Δt {dt:.3f}: A90={a90[-1]:.2f}  A50={a50[-1]:.2f}  (eff@2σ {eff[AMPS==2.0][0]:.2f})")
    a90, a50 = np.array(a90), np.array(a50)

    jpred = int(np.argmin(np.abs(dt_ul - dt_pred)))
    print(f"\n=== {event} UPPER LIMIT (p<0.01, comb, N={n_trials}) ===")
    print(f"at predicted Δt={dt_pred:.4f}s: exclude first-pulse amplitude >= "
          f"A90={a90[jpred]:.2f} σ (A50={a50[jpred]:.2f} σ)")
    print(f"best (tightest) A90 over the grid: {np.nanmin(a90):.2f} σ at Δt={dt_ul[np.nanargmin(a90)]:.3f}s")

    np.save(RESULTS / f"11_upper_limits_{event}.npy",
            {"event": event, "dt": dt_ul, "amps": AMPS, "eff_map": eff_map,
             "A90": a90, "A50": a50, "thresholds": np.array([thr[d] for d in dt_ul]),
             "dt_pred": dt_pred, "n_trials": n_trials, "statistic": "comb at-Δt, p<0.01"},
            allow_pickle=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(dt_ul, a90, "o-", color="crimson", label="A90 (exclude above)")
    ax.plot(dt_ul, a50, "s--", color="darkslategray", alpha=0.7, label="A50 (50% sensitive)")
    ax.axvline(dt_pred, color="gray", ls=":", label=f"predicted Δt = {dt_pred:.3f} s")
    ax.fill_between(dt_ul, a90, ax.get_ylim()[1], color="crimson", alpha=0.08)
    ax.set_xlabel("echo spacing Δt [s]")
    ax.set_ylabel("first-pulse amplitude [whitened-noise σ]")
    ax.set_title(f"{event}: echo amplitude EXCLUSION curve (p<0.01, comb, N={n_trials})\n"
                 f"shaded = excluded (would have been detected; on-source null)")
    ax.legend()
    fig.tight_layout()
    out = RESULTS / f"11_upper_limits_{event}.png"
    fig.savefig(out, dpi=140)
    print(f"plot -> {out}")


if __name__ == "__main__":
    main()
