"""POST-HOC check of the noise-variety verdict (z = -0.15, bound ~ +-2.7%). Declared 2026-10-23 AFTER the verdict.

This is NOT part of the pre-registered decision and cannot change it. It asks one question: is the +-2.7% bound
honest, or is it inflated (or deflated) by the noisiest ingredient of the metric? That ingredient is the zero-FA
threshold, the MAXIMUM of only 384 noise-window scores; across the five controls it swings 1.50-3.27.

Uses only saved scores (per-injection stat_max and the 384 per-run noise scores), no re-scoring.
  G0  golden: recomputing each run's metric from the saved scores with thr = max(noise) reproduces the committed
      eval json to 1e-6. If not, nothing below is read.
  A   steadier thresholds: thr = k-th largest noise score, k = 2, 4, 8 (per-window FAR ~0.5/1/2%). For each k:
      delta, sd(C) (ddof=1), z, and an exploratory 95% interval (t4, SE = sd*sqrt(1/2+1/5)). k = 4 is the headline.
  B   threshold-noise share: within each run, bootstrap the 384 noise windows (B = 1000) and recompute the zero-FA
      metric; compare the mean within-run variance with the between-seed variance sd(C)^2.
  C   paired V_s - C_s at each k.
Declared reading (before running):
  - "bound honest" if, at k = 4, |z| < 2 holds AND the 95% half-width is <= 4% of mean(C);
  - "zero-FA result is threshold-fragile" if |z| >= 2 at any k, or the sign of delta flips between k = 1 and k = 4
    while |delta| > sd(C);
  - B's share is reported as measured, with no bar.
Output: results/device_stream_posthoc.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pbh import metrics  # noqa: E402
from pbh import config as C  # noqa: E402

RES = C.RESULTS_DIR
RUNS = {**{f"V{s}": f"dev_v{s}_stream_s{s}" for s in (0, 1)},
        **{f"C{s}": f"dev_c{s}_original_s{s}" for s in range(5)}}
T4 = 2.776


def load(stem):
    inj = pd.read_parquet(RES / f"injections_{stem}_track_w64_{stem}.parquet")
    noise = np.concatenate([np.load(p) for p in sorted((RES / f"track_w64_{stem}").glob("noise_*.npy"))])
    ref = json.loads((RES / f"eval_{stem}_track_w64_{stem}.json").read_text())["results"]["max"]["mf_distance_fraction"]
    return inj, noise, ref


def frac(inj, thr):
    df = inj.assign(det=inj.stat_max > thr)
    return float(np.mean(list(metrics.mass_bin_results(df, "det")["mf_distance_fraction"].values())))


def summary(m):
    v = np.array([m["V0"], m["V1"]]); c = np.array([m[f"C{s}"] for s in range(5)])
    d = v.mean() - c.mean(); sd = c.std(ddof=1); se = sd * np.sqrt(1 / 2 + 1 / 5)
    return dict(mean_V=v.mean(), mean_C=c.mean(), delta=d, sd_C=sd, z=d / sd,
                ci95=[d - T4 * se, d + T4 * se], halfwidth_rel=T4 * se / c.mean(),
                paired={s: m[f"V{s}"] - m[f"C{s}"] for s in (0, 1)})


def main() -> int:
    data = {k: load(v) for k, v in RUNS.items()}
    g0 = {k: abs(frac(inj, noise.max()) - float(np.mean(list(ref.values())))) for k, (inj, noise, ref) in data.items()}
    g0_ok = max(g0.values()) < 1e-6
    print("G0 max|diff|", max(g0.values()), "PASS" if g0_ok else "FAIL")
    if not g0_ok:
        return 1
    out = {"G0": g0, "A": {}}
    for k in (1, 2, 4, 8):
        m = {r: frac(inj, np.sort(noise)[-k]) for r, (inj, noise, _) in data.items()}
        s = summary(m); s["per_run"] = m; out["A"][k] = s
        print(f"k={k}: dV-C {s['delta']:+.4f}  sd(C) {s['sd_C']:.4f}  z {s['z']:+.2f}  "
              f"95% [{s['ci95'][0]:+.4f},{s['ci95'][1]:+.4f}] (+-{100 * s['halfwidth_rel']:.1f}%)  "
              f"paired s0 {s['paired'][0]:+.4f} s1 {s['paired'][1]:+.4f}")
    rng = np.random.default_rng(20261023)
    within = {}
    for r, (inj, noise, _) in data.items():
        bs = [frac(inj, rng.choice(noise, size=len(noise), replace=True).max()) for _ in range(1000)]
        within[r] = float(np.var(bs, ddof=1))
    wvar = float(np.mean([within[f"C{s}"] for s in range(5)])); bvar = out["A"][1]["sd_C"] ** 2
    out["B"] = dict(within_run_var=within, mean_within_C=wvar, between_seed_var=bvar, share=wvar / bvar)
    print(f"B: threshold-only within-run var {wvar:.2e} vs between-seed var {bvar:.2e} -> share {wvar / bvar:.2f}")
    a4, a1 = out["A"][4], out["A"][1]
    honest = abs(a4["z"]) < 2 and a4["halfwidth_rel"] <= 0.04
    fragile = any(abs(out["A"][k]["z"]) >= 2 for k in out["A"]) or (
        np.sign(a1["delta"]) != np.sign(a4["delta"]) and abs(a1["delta"]) > a1["sd_C"])
    out["reading"] = ("ZERO-FA RESULT THRESHOLD-FRAGILE" if fragile else
                      "BOUND HONEST" if honest else "NOT FRAGILE, BUT k=4 BOUND WIDER THAN 4%")
    print("READING (post hoc):", out["reading"])
    (RES / "device_stream_posthoc.json").write_text(json.dumps(out, indent=2, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
