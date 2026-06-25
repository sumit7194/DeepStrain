"""N1 (PLAN.md): JOINT ringdown ↔ echo — a MASS-CONDITIONED echo search on GW250114.

The echo spacing Δt depends on the remnant mass that the RINGDOWN measures — and we now have a verified
Δt(M,χ) (14_echo_spacing.py, <2% vs Abedi). GW250114's loud ringdown gives a TIGHT mass posterior
(M=76 [68,85]), so propagating it through the formula yields a tight, physical Δt PRIOR — versus the usual
flat 0.05–0.5 s scan. Fewer search trials → a smaller look-elsewhere penalty → a MORE SENSITIVE echo search,
on the SAME event. This couples the two sub-projects: the first echo search conditioned on the event's own
ringdown. Deliverable: injection-recovery A90 for the conditioned vs flat search + the on-source GW250114 result.

Run:  .venv/bin/python scripts/15_joint_ringdown_echo.py [--n-trials 300] [--smoke]
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

EVENT = "GW250114_082203"
BAND = (30.0, 350.0)
RD_POST = Path(__file__).resolve().parent.parent.parent / "ringdown_spectroscopy" / "results" / "09_nohair_GW250114.json"
AMPS = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0])


def dt_prior_from_ringdown(rng, n=20000):
    """Propagate GW250114's ringdown (M, χ) posterior through the verified Δt(M,χ) → echo-Δt prior."""
    post = json.loads(RD_POST.read_text())["posterior"]
    (Mmed, Mlo, Mhi), (Cmed, Clo, Chi) = post["mass"], post["chi"]
    sM, sC = (Mhi - Mlo) / (2 * 1.645), (Chi - Clo) / (2 * 1.645)     # 90% CI -> sigma
    Ms = rng.normal(Mmed, sM, n)
    Cs = np.clip(rng.normal(Cmed, sC, n), 0.05, 0.95)
    dts = np.array([es.echo_spacing(m, c) for m, c in zip(Ms, Cs)])
    return float(np.median(dts)), float(np.percentile(dts, 5)), float(np.percentile(dts, 95)), (Mmed, Cmed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=300)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    n_trials = 25 if args.smoke else args.n_trials
    rng = np.random.default_rng(2025)

    dt_med, dt_lo, dt_hi, (Mmed, Cmed) = dt_prior_from_ringdown(rng)
    print(f"GW250114 ringdown M={Mmed:.0f}, χ={Cmed:.2f}  ->  echo Δt prior = {dt_med:.4f} s "
          f"[{dt_lo:.4f}, {dt_hi:.4f}] (90%)", flush=True)

    segs = {det: load_segments(EVENT, det, band=BAND) for det in DETECTORS}
    fs = segs["H1"].fs
    n_samp = len(segs["H1"].off[0])
    n_off = min(len(segs[d].off) for d in DETECTORS)
    dt_grid = np.arange(0.05, 0.5, 0.002)
    window = (dt_grid >= dt_lo) & (dt_grid <= dt_hi)          # the conditioned search band
    j_pred = int(np.argmin(np.abs(dt_grid - dt_med)))
    trials_ratio = len(dt_grid) / max(1, window.sum())
    print(f"{EVENT}: {n_off} off-source pairs, conditioned window = {window.sum()}/{len(dt_grid)} grid points "
          f"({trials_ratio:.1f}x fewer trials)", flush=True)

    def net(pair):
        return sum(comb_score(pair[d], fs, dt_grid) for d in DETECTORS)

    # --- background: max over FULL grid (flat) vs max over the conditioned WINDOW ---
    bg_flat, bg_cond = [], []
    for i in range(n_off):
        s = net({d: segs[d].off[i] for d in DETECTORS})
        bg_flat.append(float(s.max())); bg_cond.append(float(s[window].max()))
        progress("15_joint_bg", i + 1, n_off)
    bg_flat, bg_cond = np.array(bg_flat), np.array(bg_cond)
    thr_flat, thr_cond = float(np.quantile(bg_flat, 0.99)), float(np.quantile(bg_cond, 0.99))
    print(f"p<0.01 thresholds: flat {thr_flat:.3f}  conditioned {thr_cond:.3f} "
          f"(lower -> more sensitive)", flush=True)

    # --- injection recovery: A90 for flat vs conditioned (echoes injected at the predicted Δt) ---
    eff_flat, eff_cond = np.zeros(len(AMPS)), np.zeros(len(AMPS))
    for ja, amp in enumerate(AMPS):
        hf = hc = 0
        for _ in range(n_trials):
            i = int(rng.integers(0, n_off))
            inj = echo_train(n_samp, fs, dt_med, amp=amp, rng=rng)
            s = net({d: segs[d].off[i] + inj for d in DETECTORS})
            hf += s.max() > thr_flat
            hc += s[window].max() > thr_cond
            progress("15_joint_eff", ja * n_trials + _ + 1, len(AMPS) * n_trials)
        eff_flat[ja], eff_cond[ja] = hf / n_trials, hc / n_trials

    def a_at(eff, x):
        return float(np.interp(x, eff, AMPS)) if eff.max() >= x else np.nan
    a90_flat, a90_cond = a_at(eff_flat, 0.9), a_at(eff_cond, 0.9)
    a50_flat, a50_cond = a_at(eff_flat, 0.5), a_at(eff_cond, 0.5)

    # --- on-source GW250114 ---
    on = net({d: segs[d].on for d in DETECTORS})
    p_flat = float((np.sum(bg_flat >= on.max()) + 1) / (n_off + 1))
    p_cond = float((np.sum(bg_cond >= on[window].max()) + 1) / (n_off + 1))

    gain = a90_flat / a90_cond if a90_cond and np.isfinite(a90_cond) and np.isfinite(a90_flat) else np.nan
    print(f"\n=== GW250114 mass-conditioned echo search ===")
    print(f"sensitivity A90:  flat {a90_flat:.2f}σ   conditioned {a90_cond:.2f}σ   -> {gain:.2f}x tighter "
          f"(A50: {a50_flat:.2f} vs {a50_cond:.2f})")
    print(f"on-source p:      flat {p_flat:.3f}   conditioned {p_cond:.3f}   "
          f"-> {'NULL' if min(p_flat, p_cond) > 0.05 else 'signal?'}")

    (RESULTS / "15_joint_ringdown_echo.json").write_text(json.dumps(
        {"event": EVENT, "dt_prior": {"median": dt_med, "lo": dt_lo, "hi": dt_hi},
         "trials_ratio": trials_ratio, "thr_flat": thr_flat, "thr_cond": thr_cond,
         "A90": {"flat": a90_flat, "conditioned": a90_cond}, "A50": {"flat": a50_flat, "conditioned": a50_cond},
         "sensitivity_gain": gain, "on_source_p": {"flat": p_flat, "conditioned": p_cond},
         "amps": AMPS.tolist(), "eff_flat": eff_flat.tolist(), "eff_cond": eff_cond.tolist(),
         "n_trials": n_trials, "n_off": n_off}, indent=2))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(AMPS, eff_flat, "s--", color="darkslategray", label=f"flat scan (A90={a90_flat:.2f}σ)")
    ax.plot(AMPS, eff_cond, "o-", color="crimson", label=f"ringdown-conditioned (A90={a90_cond:.2f}σ)")
    ax.axhline(0.9, color="gray", ls=":", lw=1)
    ax.set_xlabel("injected first-pulse amplitude [whitened-σ]"); ax.set_ylabel("recovery fraction (p<0.01)")
    ax.set_title(f"GW250114: ringdown-conditioned echo search is {gain:.2f}× more sensitive\n"
                 f"(Δt prior {dt_med:.3f}s [{dt_lo:.3f},{dt_hi:.3f}] from the event's own ringdown mass)")
    ax.legend(); fig.tight_layout()
    fig.savefig(RESULTS / "15_joint_ringdown_echo.png", dpi=140)
    print(f"wrote 15_joint_ringdown_echo.json + .png")


if __name__ == "__main__":
    main()
