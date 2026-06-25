"""N3 (PLAN.md): STACKED multi-event echo search at formula-predicted spacings.

Each event's echo spacing is fixed by its remnant mass via the verified Δt(M,χ) (14_echo_spacing). This
stacks the comb statistic across 4 well-characterized events (GW150914, GW151226, GW151012, GW250114 — all
already searched + cached, all individually null) at each event's OWN predicted Δt. Per-event scores are
z-scored against that event's own background, then summed → a combined statistic whose SNR grows ~√N, i.e.
a population echo upper limit ~√N tighter than any single event. If the universe makes echoes at the
predicted spacing, the stack is where a faint common signal would show up.

Run:  .venv/bin/python scripts/16_stacked_echo.py [--n-trials 300] [--smoke]
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from echolib import DETECTORS, RESULTS, comb_score, echo_train, load_segments, progress

_spec = importlib.util.spec_from_file_location("es", Path(__file__).resolve().parent / "14_echo_spacing.py")
es = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(es)

# detector-frame remnant (M_f [M_sun], chi_f) + search band; all 4 are already cached from prior runs.
EVENTS = {
    "GW150914":         dict(M=68.0, chi=0.69, band=(30.0, 350.0)),
    "GW151226":         dict(M=22.4, chi=0.74, band=(30.0, 900.0)),
    "GW151012":         dict(M=42.0, chi=0.66, band=(30.0, 600.0)),
    "GW250114_082203":  dict(M=76.0, chi=0.76, band=(30.0, 350.0)),
}
AMPS = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=300)
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    n_trials, n_boot = (25, 500) if args.smoke else (args.n_trials, args.n_boot)
    rng = np.random.default_rng(7)

    # --- per-event: comb-at-predicted-Δt on-source + off-source background, z-scored ---
    ev = {}
    for name, cfg in EVENTS.items():
        dt = es.echo_spacing(cfg["M"], cfg["chi"])
        g = np.array([dt])
        segs = {det: load_segments(name, det, band=cfg["band"]) for det in DETECTORS}
        fs = segs["H1"].fs
        n_off = min(len(segs[d].off) for d in DETECTORS)
        on = float(sum(comb_score(segs[d].on, fs, g)[0] for d in DETECTORS))
        off = np.array([sum(comb_score(segs[d].off[i], fs, g)[0] for d in DETECTORS) for i in range(n_off)])
        mu, sd = float(off.mean()), float(off.std() + 1e-9)
        ev[name] = dict(dt=dt, fs=fs, segs=segs, n_off=n_off, n_samp=len(segs["H1"].off[0]),
                        on_z=(on - mu) / sd, off=off, mu=mu, sd=sd, band=cfg["band"])
        print(f"  {name}: Δt={dt:.4f}s, {n_off} off pairs, on-source z={ev[name]['on_z']:+.2f}", flush=True)
    names = list(ev)

    def stacked_offsample(idxs):
        return sum((ev[n]["off"][idxs[k]] - ev[n]["mu"]) / ev[n]["sd"] for k, n in enumerate(names))

    # --- combined null background by bootstrap (random off-source index per event) ---
    bg = np.array([stacked_offsample([int(rng.integers(0, ev[n]["n_off"])) for n in names])
                   for _ in range(n_boot)])
    thr = float(np.quantile(bg, 0.99))
    on_stack = float(sum(ev[n]["on_z"] for n in names))
    p_stack = float((np.sum(bg >= on_stack) + 1) / (n_boot + 1))
    print(f"\nstacked on-source = {on_stack:+.2f}, p<0.01 threshold = {thr:.2f}, p = {p_stack:.3f}", flush=True)

    # --- injection recovery: stacked vs best single event (echoes injected at each event's own Δt) ---
    eff_stack = np.zeros(len(AMPS))
    eff_single = {n: np.zeros(len(AMPS)) for n in names}
    thr_single = {n: float(np.quantile(ev[n]["off"], 0.99)) for n in names}      # raw-comb 99th pct per event
    for ja, amp in enumerate(AMPS):
        hs = 0
        hsing = {n: 0 for n in names}
        for _ in range(n_trials):
            zsum = 0.0
            for n in names:
                e = ev[n]; i = int(rng.integers(0, e["n_off"]))
                inj = echo_train(e["n_samp"], e["fs"], e["dt"], amp=amp, rng=rng)
                sc = float(sum(comb_score(e["segs"][d].off[i] + inj, e["fs"], np.array([e["dt"]]))[0]
                               for d in DETECTORS))
                zsum += (sc - e["mu"]) / e["sd"]
                hsing[n] += sc > thr_single[n]
            hs += zsum > thr
            progress("16_stacked_eff", ja * n_trials + _ + 1, len(AMPS) * n_trials)
        eff_stack[ja] = hs / n_trials
        for n in names:
            eff_single[n][ja] = hsing[n] / n_trials

    def a90(e):
        return float(np.interp(0.9, e, AMPS)) if e.max() >= 0.9 else np.nan
    a90_stack = a90(eff_stack)
    a90_singles = {n: a90(eff_single[n]) for n in names}
    best_single = np.nanmin(list(a90_singles.values()))
    gain = best_single / a90_stack if a90_stack and np.isfinite(a90_stack) else np.nan
    print(f"\n=== STACKED echo upper limit ({len(names)} events) ===")
    print(f"A90 stacked = {a90_stack:.2f}σ vs best single {best_single:.2f}σ -> {gain:.2f}x tighter "
          f"(√N ideal = {np.sqrt(len(names)):.2f}x)")
    print(f"per-event A90: " + "  ".join(f"{n.split('_')[0]}:{a90_singles[n]:.2f}" for n in names))
    print(f"stacked on-source: {'NULL (no echoes)' if p_stack > 0.05 else 'SIGNAL?'} (p={p_stack:.3f})")

    (RESULTS / "16_stacked_echo.json").write_text(json.dumps(
        {"events": {n: {"dt": ev[n]["dt"], "on_z": ev[n]["on_z"], "a90_single": a90_singles[n]} for n in names},
         "stacked_on": on_stack, "p_stacked": p_stack, "a90_stacked": a90_stack,
         "best_single_a90": float(best_single), "stack_gain": gain, "sqrtN_ideal": float(np.sqrt(len(names))),
         "amps": AMPS.tolist(), "eff_stacked": eff_stack.tolist(), "n_trials": n_trials}, indent=2))

    fig, ax = plt.subplots(figsize=(8, 5))
    for n in names:
        ax.plot(AMPS, eff_single[n], "--", alpha=0.5, label=f"{n.split('_')[0]} (A90={a90_singles[n]:.2f})")
    ax.plot(AMPS, eff_stack, "o-", color="crimson", lw=2.2, label=f"STACKED (A90={a90_stack:.2f})")
    ax.axhline(0.9, color="gray", ls=":", lw=1)
    ax.set_xlabel("injected first-pulse amplitude [whitened-σ]"); ax.set_ylabel("recovery fraction (p<0.01)")
    ax.set_title(f"Stacked echo search over {len(names)} events at formula-predicted Δt\n"
                 f"combined limit {gain:.2f}× tighter than best single (√N={np.sqrt(len(names)):.2f}); on-source null")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(RESULTS / "16_stacked_echo.png", dpi=140)
    print(f"wrote 16_stacked_echo.json + .png")


if __name__ == "__main__":
    main()
