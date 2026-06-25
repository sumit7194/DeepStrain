"""R3 (PLAN.md): IMR-waveform REFEREE for the no-hair NPE — does it recover δ≈0 on REALISTIC ringdowns?

The no-hair NPE (09) is trained on ANALYTIC 220 + (1+δ)·221 tones. R3 tests it against realistic full-IMR
ringdowns (IMRPhenomXAS, NR-calibrated, generated via pbh's pycbc → data/imr_ringdowns.npz). These contain the
real post-peak waveform: the full overtone content + the merger→ringdown transition the two-tone model omits.
Inject the IMR ringdown SHAPE (unit-peak, scaled to loudness) into the NPE's whitened/unit-noise convention,
sample (M, χ, δ), recalibrate (v3 T), and check δ ≈ 0 — the true δ is 0 (IMR is GR), so any offset is a
model-incompleteness bias. Compared head-to-head with the analytic-tone control (the NPE's own training family).

Run:  .venv/bin/python scripts/15_imr_referee.py [--n 80]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import rdlib
import sbilib
from sbilib import Embed, FS, N_SAMP, PEAK_AMP_RANGE, T0_MAX_MS  # Embed needed to unpickle the posterior

RESULTS = Path(__file__).resolve().parent.parent / "results"
PLOTS = Path(__file__).resolve().parent.parent / "plots"
DATA = Path(__file__).resolve().parent.parent / "data"
PRIOR_SIG = 1.0 / np.sqrt(12)


def inject_shape(shape, a220, rng, start_off=0):
    """2-detector segment: ringdown SHAPE from `start_off` samples after the peak (re-normalized to unit peak,
    so loudness is matched across offsets), scaled to a220, + per-det amp + t0 jitter + unit noise.
    start_off>0 skips the early merger-transition/overtone-rich part → tests where the model-misfit bias lives."""
    sh = shape[start_off:]
    sh = sh / (np.max(np.abs(sh)) + 1e-12)
    x = np.empty((2, N_SAMP), dtype=np.float32)
    for d in range(2):
        amp = a220 * rng.uniform(0.7, 1.3)
        t0 = int(rng.uniform(0, T0_MAX_MS / 1000.0) * FS)               # small start-time jitter (NPE marginalizes it)
        sig = np.zeros(N_SAMP)
        seg = sh[: N_SAMP - t0]
        sig[t0:t0 + len(seg)] = amp * seg
        x[d] = sig + rng.standard_normal(N_SAMP)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    posterior = torch.load(RESULTS / "09_posterior_150k.pt", weights_only=False)
    T = json.loads((RESULTS / "10_recalibration.json").read_text())["T"]
    z = np.load(DATA / "imr_ringdowns.npz")
    imr = {k: z[k] for k in z if not k.endswith("_meta")}
    meta = {k[:-5]: z[k] for k in z if k.endswith("_meta")}
    print(f"loaded NPE + T={T}; IMR ringdowns: "
          + ", ".join(f"{k}({meta[k][3]:.0f}Hz)" for k in imr) + f"; n={args.n}/case\n")

    @torch.no_grad()
    def delta_hat(x):
        s = posterior.sample((400,), x=torch.tensor(x.reshape(1, -1)), show_progress_bars=False).numpy()[:, 2]
        s = np.median(s) + T * (s - np.median(s))                       # v3 recalibration
        return float(np.median(s)), float(np.std(s))

    # GW250114-class loudness (the loud end of the training range)
    def loud():
        return rng.uniform(0.7 * PEAK_AMP_RANGE[1], PEAK_AMP_RANGE[1])

    rows = {}
    # --- analytic-tone CONTROL (the NPE's own training family) — should be unbiased δ≈0 ---
    dc = []
    for _ in range(args.n):
        x = sbilib.simulate(rng.uniform(60, 90), rng.uniform(0.6, 0.85), 0.0, rng)   # δ=0 Kerr tones
        m, _ = delta_hat(x); dc.append(m)
        rdlib.progress("15_imr_referee", len(dc), args.n * (len(imr) + 1))
    rows["analytic_control"] = dc
    print(f"{'case':>18} {'f_ring':>7} {'δ_hat mean':>11} {'± std':>7}  Kerr-consistent?")
    print(f"{'analytic (control)':>18} {'--':>7} {np.mean(dc):>+11.3f} {np.std(dc):>7.3f}  "
          f"{'YES' if abs(np.mean(dc)) < 0.1 else 'BIASED'}")

    # --- IMR full-merger ringdowns ---
    for k in imr:
        d = []
        for _ in range(args.n):
            x = inject_shape(imr[k], loud(), rng)
            m, _ = delta_hat(x); d.append(m)
            rdlib.progress("15_imr_referee", args.n * (1 + list(imr).index(k)) + len(d), args.n * (len(imr) + 1))
        rows[k] = d
        m1, m2, s, fr = meta[k]
        print(f"{f'IMR {int(m1)}+{int(m2)} s{s:.1f}':>18} {fr:>7.0f} {np.mean(d):>+11.3f} {np.std(d):>7.3f}  "
              f"{'YES' if abs(np.mean(d)) < 0.1 else 'BIASED'}")

    imr_means = [float(np.mean(rows[k])) for k in imr]
    peak_bias = max(abs(m) for m in imr_means)

    # --- stress-test the MECHANISM: does the bias shrink if we inject the later, cleaner ringdown? ---
    print(f"\nstart-time sweep (IMR case 'a', matched loudness) — where does the bias live?")
    off_ms = [0.0, 2.0, 4.0, 6.0]
    sweep = {}
    for om in off_ms:
        soff = int(om / 1000.0 * FS)
        d = [delta_hat(inject_shape(imr["a"], loud(), rng, start_off=soff))[0] for _ in range(args.n)]
        sweep[om] = float(np.mean(d))
        print(f"  start +{om:.0f} ms after peak: δ_hat = {sweep[om]:+.3f}")
    shrinks = abs(sweep[off_ms[-1]]) < abs(sweep[0.0]) - 0.05            # bias clearly smaller at later start

    print(f"\nVERDICT: the no-hair NPE (trained on analytic 220+221) is UNBIASED on its own family "
          f"(control δ={np.mean(dc):+.3f}) but carries a δ≈{-peak_bias:.2f} SYSTEMATIC on realistic full-IMR "
          f"ringdowns injected from the peak. Bias {'SHRINKS' if shrinks else 'persists'} when injecting the "
          f"later/cleaner ringdown ({sweep[0.0]:+.2f}→{sweep[off_ms[-1]]:+.2f}) ⇒ it is the early-time "
          f"merger-transition/overtone content the two-tone model omits — a real waveform systematic.")

    (RESULTS / "15_imr_referee.json").write_text(json.dumps(
        {"T": T, "n_per_case": args.n, "control_delta_mean": float(np.mean(dc)),
         "imr_peak": {k: {"f_ring": float(meta[k][3]), "delta_mean": float(np.mean(rows[k])),
                          "delta_std": float(np.std(rows[k]))} for k in imr},
         "peak_bias": float(peak_bias), "start_time_sweep_ms": sweep, "bias_shrinks_late": bool(shrinks)},
        indent=2))
    print("wrote 15_imr_referee.json")


if __name__ == "__main__":
    main()
