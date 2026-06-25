"""N2 (PLAN.md): does an H1×L1 CONSISTENCY-weighted echo statistic beat the plain comb-sum?

Build C-2 showed a LEARNED H1×L1 consistency statistic beats `sum` for the pbh CNN. The echo analog would
be a learned head — but the echo data is tiny (~159 off-source pairs/event), so a learned head would overfit
(the R2 lesson). Instead we test a ROBUST, training-free consistency statistic:

    net_λ(Δt) = combH(Δt) + combL(Δt) − λ·|combH(Δt) − combL(Δt)|        (λ=0 → plain sum; λ=1 → 2·min)

which penalises per-Δt disagreement between the detectors (a real echo has power at the SAME Δt in both; a
noise fluctuation does not). λ is swept; if λ>0 lowers A90 (more sensitive) the consistency idea helps,
robustly. Note the comb-sum is ALREADY Δt-consistency-aware (it sums per-Δt), so the prior is that λ≈0 wins.

Run:  .venv/bin/python scripts/17_echo_consistency.py [--n-trials 200]
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
es = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(es)

EVENTS = {"GW150914": dict(M=68.0, chi=0.69, band=(30.0, 350.0)),
          "GW250114_082203": dict(M=76.0, chi=0.76, band=(30.0, 350.0))}
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]
AMPS = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=200)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    n_trials = 20 if args.smoke else args.n_trials
    rng = np.random.default_rng(11)

    LAM_PRE = 0.5            # PRE-CHOSEN consistency penalty (avoids the best-λ selection bias)
    out = {}
    for event, cfg in EVENTS.items():
        dt = es.echo_spacing(cfg["M"], cfg["chi"])
        grid = np.arange(0.05, 0.5, 0.002)
        segs = {d: load_segments(event, d, band=cfg["band"]) for d in DETECTORS}
        fs = segs["H1"].fs; n_off = min(len(segs[d].off) for d in DETECTORS); n_samp = len(segs["H1"].off[0])

        def netmax(cH, cL, lam):
            return float((cH + cL - lam * np.abs(cH - cL)).max())       # max-over-Δt consistency statistic

        # background combs (once) -> per-λ p<0.01 threshold
        bgc = [(comb_score(segs["H1"].off[i], fs, grid), comb_score(segs["L1"].off[i], fs, grid))
               for i in range(n_off)]
        thr = {lam: float(np.quantile([netmax(cH, cL, lam) for cH, cL in bgc], 0.99)) for lam in LAMBDAS}

        # injection-recovery: store per-(amp,trial) statistic for EVERY λ (combs computed once/trial)
        stat = {lam: np.zeros((len(AMPS), n_trials)) for lam in LAMBDAS}
        for ja, amp in enumerate(AMPS):
            for k in range(n_trials):
                i = int(rng.integers(0, n_off)); inj = echo_train(n_samp, fs, dt, amp=amp, rng=rng)
                cH = comb_score(segs["H1"].off[i] + inj, fs, grid); cL = comb_score(segs["L1"].off[i] + inj, fs, grid)
                for lam in LAMBDAS:
                    stat[lam][ja, k] = netmax(cH, cL, lam)
                progress(f"17_{event}", ja * n_trials + k + 1, len(AMPS) * n_trials)

        def a90_from(det):                                              # det: (n_amp, n_trial) booleans
            eff = det.mean(1)
            return float(np.interp(0.9, eff, AMPS)) if eff.max() >= 0.9 else float("nan")
        a90 = {lam: a90_from(stat[lam] > thr[lam]) for lam in LAMBDAS}

        # SIGNIFICANCE: bootstrap ΔA90 = A90(λ_pre) − A90(sum) over resampled trials (paired)
        d0 = stat[0.0] > thr[0.0]; dP = stat[LAM_PRE] > thr[LAM_PRE]
        diffs = []
        for _ in range(500):
            bi = rng.integers(0, n_trials, n_trials)
            diffs.append(a90_from(dP[:, bi]) - a90_from(d0[:, bi]))
        diffs = np.array([x for x in diffs if np.isfinite(x)])
        lo, hi = np.percentile(diffs, [5, 95]); p_better = float((diffs < 0).mean())
        out[event] = {"dt": dt, "a90_vs_lambda": a90, "sum_a90": a90[0.0], "pre_a90": a90[LAM_PRE],
                      "dA90_ci90": [float(lo), float(hi)], "p_pre_better": p_better}
        sig = hi < 0                                                    # whole CI below 0 -> λ_pre significantly better
        print(f"{event}: A90 sum={a90[0.0]:.2f}, λ={LAM_PRE} {a90[LAM_PRE]:.2f}; "
              f"ΔA90 90%CI[{lo:+.2f},{hi:+.2f}] P(better)={p_better:.2f} -> {'SIGNIFICANT' if sig else 'not significant'}"
              + "  | full curve " + " ".join(f"{l}:{a90[l]:.2f}" for l in LAMBDAS), flush=True)

    sig_events = [e for e, o in out.items() if o["dA90_ci90"][1] < 0]
    helps = len(sig_events) == len(out)
    print(f"\nVERDICT: pre-chosen consistency penalty λ={LAM_PRE} significantly beats the plain comb-sum at "
          f"both events -> {'YES (robust, bootstrap-validated win)' if helps else 'NOT at all events (inconclusive/negative)'}"
          f"  [significant at: {sig_events or 'none'}]")

    (RESULTS / "17_echo_consistency.json").write_text(json.dumps(
        {"lambdas": LAMBDAS, "amps": AMPS.tolist(), "events": out, "consistency_helps": bool(helps),
         "n_trials": n_trials}, indent=2))

    fig, ax = plt.subplots(figsize=(8, 5))
    for event in out:
        ax.plot(LAMBDAS, [out[event]["a90_vs_lambda"][l] for l in LAMBDAS], "o-", label=event.split("_")[0])
    ax.set_xlabel("consistency penalty λ  (0 = plain sum, 1 = 2·min)"); ax.set_ylabel("A90 [whitened-σ] (lower = better)")
    ax.set_title("Echo H1×L1 consistency-weighted statistic vs plain comb-sum")
    ax.legend(); fig.tight_layout(); fig.savefig(RESULTS / "17_echo_consistency.png", dpi=140)
    print("wrote 17_echo_consistency.json + .png")


if __name__ == "__main__":
    main()
