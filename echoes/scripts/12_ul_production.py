"""Step 12 — E1 (honest): production-path echo amplitude UPPER LIMIT, comb vs ML scorer.

11_upper_limits gives the gated comb exclusion in the WHITENED domain. There the v2 ML scorer's
edge is the known v5 *convention artifact* (~13x), NOT a real tightening — so a "ML upper limit"
must NOT be read off the whitened harness. This computes the exclusion through the PRODUCTION path
(raw-strain injection + re-whitening, the v5 fair-comparison machinery in 09/10): both statistics
on the SAME re-whitened segments, so the comb-vs-ML A90 ratio reflects the honest ~1.2x.

Per Δt: build the p<0.01 off-source background for each statistic, then the amplitude-recovery
efficiency, and read off A90/A50. Backgrounds (whitening + ML envelope + full-grid comb) are
Δt-independent so they are computed ONCE per off-source center and reused across the grid.

Run:  .venv/bin/python scripts/12_ul_production.py [--n-trials 120] [--smoke]
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
import torch
from echolib import DETECTORS, GW150914_DT_PRED, RESULTS, comb_on_env, comb_score, fetch_block, progress

_spec = importlib.util.spec_from_file_location("raw9", Path(__file__).resolve().parent / "09_raw_injection.py")
raw9 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(raw9)
ml = raw9.ml

FS = 4096.0
EVENT = "GW150914"
AMPS = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=120)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    n_trials = 12 if args.smoke else args.n_trials
    rng = np.random.default_rng(2025)

    raws = {det: fetch_block(det, EVENT) for det in DETECTORS}
    t0 = float(raws["H1"].t0.value)
    centers = t0 + 308 + 4 * np.arange(45)             # off-source 4-s-spaced eval centers
    slope = float(np.mean(list(json.loads((RESULTS / "09_raw_injection.json").read_text())["slopes"].values())))
    dt_full = np.arange(0.05, 0.5, 0.005)
    dt_ul = np.unique(np.append(np.arange(0.05, 0.49, 0.04), GW150914_DT_PRED))   # coarser (production is slow)
    dt_idx = [int(np.argmin(np.abs(dt_full - d))) for d in dt_ul]

    models = {}
    for det in DETECTORS:
        m = ml.ConvAE(); m.load_state_dict(torch.load(RESULTS / f"07_scorer_{det}.pt", weights_only=True))
        m.eval(); models[det] = m

    def full_scores(center, inj=None):
        """Both statistics over the FULL dt grid on the re-whitened segment (summed over detectors)."""
        tm = np.zeros(len(dt_full)); tc = np.zeros(len(dt_full))
        for det in DETECTORS:
            seg = raw9.whitened_segment(raws[det], center, inj)
            tm += comb_on_env(ml.error_envelope(models[det], seg, FS), FS, dt_full)
            tc += comb_score(seg, FS, dt_full)
        return tm, tc

    # --- background ONCE per center (Δt-independent), then per-Δt p<0.01 thresholds ---
    print(f"{EVENT}: production-path UL | {len(centers)} bg centers, {len(dt_ul)} spacings x {len(AMPS)} amps,"
          f" {n_trials} trials/cell", flush=True)
    bg_ml = np.empty((len(centers), len(dt_full))); bg_cb = np.empty((len(centers), len(dt_full)))
    for i, c in enumerate(centers):
        bg_ml[i], bg_cb[i] = full_scores(float(c))
        progress("12_ul_bg", i + 1, len(centers))
    th_ml = {k: float(np.quantile(bg_ml[:, k], 0.99)) for k in dt_idx}
    th_cb = {k: float(np.quantile(bg_cb[:, k], 0.99)) for k in dt_idx}

    # --- efficiency + A90/A50 per Δt for both statistics ---
    res = {"ml": {"a90": [], "a50": []}, "comb": {"a90": [], "a50": []}}
    eff_ml_map, eff_cb_map = np.zeros((len(dt_ul), len(AMPS))), np.zeros((len(dt_ul), len(AMPS)))
    for jd, (dt, k) in enumerate(zip(dt_ul, dt_idx)):
        for ja, amp in enumerate(AMPS):
            A = amp / slope
            hm = hc = 0
            for _ in range(n_trials):
                c = float(centers[rng.integers(0, len(centers))])
                inj = (lambda t, c=c, A=A, dt=dt: raw9.raw_train(t, c + 0.05, A, dt=dt))
                tm, tc = full_scores(c, inj)
                hm += tm[k] > th_ml[k]; hc += tc[k] > th_cb[k]
            eff_ml_map[jd, ja] = hm / n_trials; eff_cb_map[jd, ja] = hc / n_trials
            progress("12_ul_eff", jd * len(AMPS) + ja + 1, len(dt_ul) * len(AMPS))
        for tag, emap in (("ml", eff_ml_map), ("comb", eff_cb_map)):
            e = emap[jd]
            res[tag]["a90"].append(float(np.interp(0.9, e, AMPS)) if e.max() >= 0.9 else np.nan)
            res[tag]["a50"].append(float(np.interp(0.5, e, AMPS)) if e.max() >= 0.5 else np.nan)
        print(f"  Δt {dt:.3f}: A90 ml={res['ml']['a90'][-1]:.2f}  comb={res['comb']['a90'][-1]:.2f}", flush=True)

    jpred = int(np.argmin(np.abs(dt_ul - GW150914_DT_PRED)))
    a90_ml, a90_cb = np.array(res["ml"]["a90"]), np.array(res["comb"]["a90"])
    ratio = a90_cb[jpred] / a90_ml[jpred]
    print(f"\n=== {EVENT} PRODUCTION-PATH UPPER LIMIT (p<0.01, {len(centers)} bg centers) ===")
    print(f"at predicted Δt={GW150914_DT_PRED:.4f}s: A90 comb={a90_cb[jpred]:.2f}σ  ML={a90_ml[jpred]:.2f}σ"
          f"  -> ML tighter by {ratio:.2f}x (honest production path; cf. v5 ~1.2x)")

    (RESULTS / "12_ul_production.json").write_text(json.dumps(
        {"event": EVENT, "dt": dt_ul.tolist(), "amps": AMPS.tolist(), "n_bg": len(centers),
         "n_trials": n_trials, "dt_pred": GW150914_DT_PRED, "a90": {"ml": a90_ml.tolist(), "comb": a90_cb.tolist()},
         "a50": {k: res[k]["a50"] for k in res}, "ratio_at_pred": ratio,
         "eff_ml": eff_ml_map.tolist(), "eff_comb": eff_cb_map.tolist(),
         "statistic": "production-path raw injection, comb vs ML, p<0.01"}, indent=1))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(dt_ul, a90_cb, "s--", color="darkslategray", label="comb A90 (production path)")
    ax.plot(dt_ul, a90_ml, "o-", color="crimson", label="ML scorer A90 (production path)")
    ax.axvline(GW150914_DT_PRED, color="gray", ls=":", label=f"predicted Δt = {GW150914_DT_PRED:.3f} s")
    ax.set_xlabel("echo spacing Δt [s]"); ax.set_ylabel("first-pulse amplitude [whitened-σ equivalent]")
    ax.set_title(f"{EVENT}: HONEST production-path echo exclusion — ML vs comb\n"
                 f"(raw-strain injection + re-whitening; ML tighter by {ratio:.2f}x at predicted Δt)")
    ax.legend(); fig.tight_layout()
    out = RESULTS / "12_ul_production.png"
    fig.savefig(out, dpi=140)
    print(f"plot -> {out}")


if __name__ == "__main__":
    main()
