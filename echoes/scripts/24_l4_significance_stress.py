"""Is L4's 3.2 sigma real? A paired bootstrap against three error terms the analytic bar ignores.

THE CLAIM UNDER TEST. 20_coherent_network concluded that coherent network combination helps by 1.12x at
3.17 sigma on physically-injected echoes, and verify.sh gates it at gain > 1.10 AND sigma > 2.0. Both
criteria are marginal (measured 1.1248 and 3.17), so a modest error-bar correction flips the verdict. That
alone justifies a harder look, but there are three specific reasons to distrust the quoted sigma:

  (1) THE SLOPE DENOMINATOR IS ESTIMATED FROM TWO ADJACENT GRID POINTS.
      errs = sqrt(0.25/n_trials) / slope, with slope = (y[i]-y[i-1])/(a[i]-a[i-1]) on a 6-point amplitude
      grid. A noisy denominator makes the error bar itself noisy, in an uncontrolled direction.

  (2) THE TWO STATISTICS ARE PAIRED, BUT COMBINED AS IF INDEPENDENT.
      `a, b = scores(c, amp, ...)` returns both statistics from the SAME trial -- same noise, same injection,
      same sky location -- yet the difference error is hypot(err_i, err_c). For positively correlated
      estimates that OVERSTATES the error, so this term could push significance UP. The sign is not knowable
      in advance, which is exactly why it must be measured rather than argued.

  (3) THE THRESHOLD UNCERTAINTY IS ENTIRELY ABSENT.
      th = 95th percentile of only n_bg = 60 background draws. That quantile has real sampling error, it
      shifts every efficiency point coherently, and it appears nowhere in the analytic bar.

WHY WE SHOULD EXPECT TROUBLE. The L2 deep-FAR audit found analytic Poisson bands an order of magnitude too
narrow against a resampling estimate of the same quantity, and the estimator audit found our jackknife
understating by 4.2x. Separately, L4's own history has the signature of an understated error: v3 reported
1.21x at 1.4 sigma, v4 reported 1.12x at 3.2 sigma -- the gain SHRANK as statistics improved, which is what
happens when the earlier uncertainty did not cover the spread.

THE TEST. Re-run the identical protocol but RETAIN per-trial outcomes, then bootstrap:
  * resample trial indices, the SAME indices for both statistics, so the pairing is preserved by construction
  * resample the background draws too, so the threshold moves as it really would
  * recompute both efficiency curves, both amp50 values, and the gain, on every resample
  * report the bootstrap SD of the difference and the CI on the gain, against the analytic 0.0324

PRE-REGISTERED:
  * the claim SURVIVES only if the bootstrap gain CI excludes 1.10 (the gate's own bar) and the bootstrap
    significance stays above 2.0
  * bootstrap_sd / analytic_sd is reported whatever happens -- >1.5 means the analytic bar was too narrow
    (the pattern from L2), <0.7 means the pairing was costing us real significance
  * a null here does NOT retract the injection-convention finding (physical vs identical, 1.12x vs 0.92x),
    which is a separate and much larger effect; it would retract only the "coherent helps" verdict.

Run:  .venv/bin/python scripts/24_l4_significance_stress.py [--n-trials 120] [--n-bg 60] [--boot 2000]
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from echolib import DETECTORS, GW150914_DT_PRED, RESULTS, comb_on_env, envelope, fetch_block, progress

HERE = Path(__file__).resolve().parent
_s = importlib.util.spec_from_file_location("net20", HERE / "20_coherent_network.py")
net20 = importlib.util.module_from_spec(_s); _s.loader.exec_module(net20)

OUT = RESULTS / "24_l4_significance_stress.json"
AMPS = net20.AMPS


def amp50_from(eff_y, amps):
    """Interpolate the 50% crossing; nan if the curve never reaches it (a real outcome under resampling)."""
    if np.max(eff_y) < 0.5:
        return np.nan
    return float(np.interp(0.5, eff_y, amps))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=120)
    ap.add_argument("--n-bg", type=int, default=60)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--conv", default="physical", choices=("physical", "identical"))
    args = ap.parse_args()

    # Rebuild 20_'s setup exactly. `scores` is nested inside its main(), so it cannot be imported; it is
    # reconstructed here from the same module-level pieces, including the geometry golden test, so this
    # measures THAT protocol rather than a lookalike.
    raw9 = net20.raw9
    FS = net20.FS
    raws = {d: fetch_block(d, "GW150914") for d in DETECTORS}
    t0 = float(raws["H1"].t0.value)
    centers = t0 + 308 + 4 * np.arange(42)
    dt_grid = np.arange(0.05, 0.5, 0.005)
    j = int(np.argmin(np.abs(dt_grid - GW150914_DT_PRED)))
    rng = np.random.default_rng(24)

    merger = {d: raw9.whitened_segment(raws[d], t0 + 316.0) for d in DETECTORS}
    n_use = int(0.2 * FS)
    d_samp, sign, cc = net20.measure_delay_sign(merger["H1"][:n_use], merger["L1"][:n_use], FS)
    d_ms = 1e3 * d_samp / FS
    if abs(abs(d_ms) - net20.LIT_DELAY_MS) >= 3.0:
        print(f"GOLDEN TEST FAILED (delay {d_ms:+.2f} ms) — refusing to proceed"); return
    print(f"GOLDEN TEST ok: delay {d_ms:+.2f} ms, sign {sign:+.0f}", flush=True)
    tau_s = -d_samp / FS

    def scores(c, amp=None, physical=True):
        pseed = int(rng.integers(1 << 30))
        segs = {}
        for det in DETECTORS:
            if amp is None:
                inj = None
            elif det == "L1" and physical:
                inj = (lambda t, c=c, A=amp, s_=pseed:
                       (setattr(raw9, "rng", np.random.default_rng(s_)),
                        sign * raw9.raw_train(t, c + 0.05 + tau_s, A))[1])
            else:
                inj = (lambda t, c=c, A=amp, s_=pseed:
                       (setattr(raw9, "rng", np.random.default_rng(s_)),
                        raw9.raw_train(t, c + 0.05, A))[1])
            segs[det] = raw9.whitened_segment(raws[det], c, inj)
        incoh = np.zeros(len(dt_grid))
        for x in segs.values():
            incoh += comb_on_env(envelope(x, FS), FS, dt_grid)
        coh = comb_on_env(envelope(net20.coherent_network(segs, d_samp, sign), FS), FS, dt_grid)
        return float(incoh[j]), float(coh[j])

    print(f"conv={args.conv} | n_trials {args.n_trials} | n_bg {args.n_bg} | boot {args.boot}", flush=True)

    # --- background, retained per draw (its quantile is one of the neglected error terms) ---
    bg_i, bg_c = [], []
    for i in range(args.n_bg):
        a, b = scores(float(centers[i % len(centers)]), amp=None)   # 20_ cycles deterministically here
        bg_i.append(a); bg_c.append(b)
        progress("24_bg", i, args.n_bg)
    bg_i, bg_c = np.array(bg_i), np.array(bg_c)
    print(f"background: incoherent 95th {np.quantile(bg_i,0.95):.3f} | coherent {np.quantile(bg_c,0.95):.3f}",
          flush=True)

    # --- trials, PER-TRIAL raw statistics retained (20_ keeps only counts, which is why it cannot bootstrap) ---
    slope = float(np.mean(list(
        json.loads((RESULTS / "09_raw_injection.json").read_text())["slopes"].values())))
    S = {"incoherent": np.zeros((len(AMPS), args.n_trials)),
         "coherent": np.zeros((len(AMPS), args.n_trials))}
    t_start = time.time()
    for ai, amp in enumerate(AMPS):
        A = amp / slope
        for k in range(args.n_trials):
            c = float(centers[rng.integers(0, len(centers))])
            a, b = scores(c, amp=A, physical=(args.conv == "physical"))
            S["incoherent"][ai, k] = a; S["coherent"][ai, k] = b
            progress(f"24_{amp}", k, args.n_trials)
        print(f"  amp {amp:4.2f} done ({time.time()-t_start:.0f}s)", flush=True)

    amps = np.array([float(a) for a in AMPS])

    def point_estimate(ti, bi):
        th_i = float(np.quantile(bg_i[bi], 0.95)); th_c = float(np.quantile(bg_c[bi], 0.95))
        ei = (S["incoherent"][:, ti] > th_i).mean(axis=1)
        ec = (S["coherent"][:, ti] > th_c).mean(axis=1)
        a_i, a_c = amp50_from(ei, amps), amp50_from(ec, amps)
        g = a_i / a_c if (np.isfinite(a_i) and np.isfinite(a_c) and a_c > 0) else np.nan
        return a_i, a_c, g

    all_t = np.arange(args.n_trials); all_b = np.arange(args.n_bg)
    a_i0, a_c0, g0 = point_estimate(all_t, all_b)
    print(f"\npoint estimate: amp50 incoherent {a_i0:.4f} | coherent {a_c0:.4f} | gain {g0:.4f}", flush=True)

    # --- PAIRED bootstrap: same trial indices for both statistics, background resampled too ---
    gains, diffs = [], []
    for _ in range(args.boot):
        ti = rng.integers(0, args.n_trials, args.n_trials)
        bi = rng.integers(0, args.n_bg, args.n_bg)
        a_i, a_c, g = point_estimate(ti, bi)
        if np.isfinite(g):
            gains.append(g); diffs.append(a_i - a_c)
    gains, diffs = np.array(gains), np.array(diffs)
    sd_boot = float(diffs.std(ddof=1))
    sig_boot = float(abs(np.mean(diffs)) / sd_boot) if sd_boot > 0 else float("nan")

    prev = json.loads((RESULTS / "20_coherent_network.json").read_text())
    sd_analytic = float(prev["significance"][args.conv]["diff_err"])
    sig_analytic = float(prev["significance"][args.conv]["n_sigma"])

    res = {"conv": args.conv, "n_trials": args.n_trials, "n_bg": args.n_bg,
           "boot_used": int(len(gains)), "boot_requested": args.boot,
           "point": {"amp50_incoherent": a_i0, "amp50_coherent": a_c0, "gain": g0},
           "analytic": {"diff_sd": sd_analytic, "n_sigma": sig_analytic},
           "bootstrap": {"diff_mean": float(diffs.mean()), "diff_sd": sd_boot, "n_sigma": sig_boot,
                         "gain_median": float(np.median(gains)),
                         "gain_ci90": [float(np.percentile(gains, 5)), float(np.percentile(gains, 95))],
                         "p_gain_gt_1.10": float((gains > 1.10).mean()),
                         "p_gain_gt_1": float((gains > 1.0).mean()),
                         "nan_fraction": float(1 - len(gains) / args.boot)},
           "sd_ratio_boot_over_analytic": sd_boot / sd_analytic if sd_analytic > 0 else None}

    print(f"\nanalytic : diff sd {sd_analytic:.4f} -> {sig_analytic:.2f} sigma")
    print(f"bootstrap: diff sd {sd_boot:.4f} -> {sig_boot:.2f} sigma   "
          f"(ratio {res['sd_ratio_boot_over_analytic']:.2f}x)")
    b = res["bootstrap"]
    print(f"gain {b['gain_median']:.4f}  90% CI [{b['gain_ci90'][0]:.4f}, {b['gain_ci90'][1]:.4f}]  "
          f"P(>1.10) = {b['p_gain_gt_1.10']:.2f}  P(>1) = {b['p_gain_gt_1']:.2f}")
    if b["nan_fraction"] > 0:
        print(f"  ({100*b['nan_fraction']:.1f}% of resamples never reached 50% efficiency — a real outcome, "
              f"excluded from the CI, and itself a sign of how close the curve sits to the bar)")

    survives = b["gain_ci90"][0] > 1.10 and sig_boot > 2.0
    res["survives"] = bool(survives)
    res["verdict"] = ("L4 HOLDS — gain CI excludes the 1.10 gate bar and significance stays above 2 sigma"
                      if survives else
                      "L4 DOES NOT SURVIVE a paired bootstrap — the analytic bar understated the spread; the "
                      "'coherent helps' verdict must be re-quoted (injection-convention finding unaffected)")
    print(f"\nVERDICT: {res['verdict']}")
    OUT.write_text(json.dumps(res, indent=2))
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
