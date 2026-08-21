"""Are threshold-spread and FAR-spread the SAME uncertainty? Test the relation that MUST hold.

WHY. far_null_precision left one thing unresolved: the FAR-of-observed spread falls much faster with n
(~n^-0.74 early) than the threshold spread (~n^-0.10), which naively contradicts the claim that the two
summaries carry identical information. Reasoning about which wins is exactly how a confident wrong number
gets produced, so this measures it instead.

THE FREE INVARIANT (ansatz's suggestion, and it is the right shape of check -- a quantity the machinery
cannot violate even when it is wrong). If S -> FAR is a deterministic monotone map, then the FAR uncertainty
is FORCED by the threshold uncertainty through the local derivative:

    sigma(ln FAR)  =  |d(ln FAR)/dS| * sigma(S)

Nothing else is permitted. So predict the FAR spread from the MEASURED threshold spread at each n, using the
derivative measured from our own probe counts, and compare against the DIRECTLY measured FAR spread:

  * they track   => same information, confirmed rather than argued. The differing exponents are just the
                    derivative changing with n, and the "no summary escapes the glitch limit" headline rests
                    on a relation that must hold rather than on a suspicion.
  * they diverge => the FAR estimate carries a noise source the threshold does not (the Poisson-at-small-
                    counts story), and WHERE it diverges locates the regime boundary instead of leaving it
                    to be guessed at.

The derivative is measured, not assumed: our probe counts at S0 = 7.5 / 9.5 / 11.295 / 13.0 give
d(ln count)/dS locally by finite difference, at each subset size independently.

Run:  .venv/bin/python scripts/far_jacobian_check.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pbh import config as C

SRC = C.RESULTS_DIR / "far_null_precision.json"
OUT = C.RESULTS_DIR / "far_jacobian_check.json"
S_OBS = 11.295368194580078
REF_FAR = "1/decade"


def main() -> None:
    d = json.loads(SRC.read_text())
    rows = d["rows"]
    sizes = sorted({r["n"] for r in rows})
    probes = sorted({float(r["name"]) for r in rows if r["kind"] == "far_at_S0"})

    print(f"probe values: {probes}")
    print(f"predicting sigma(ln FAR) at S0={S_OBS:.3f} from sigma(threshold at {REF_FAR})\n")
    print(f"{'n':>5} {'dlnC/dS':>8} {'sd(thr)':>8} {'predicted':>10} {'measured':>9} {'ratio':>6}")

    out = []
    for n in sizes:
        far_rows = {float(r["name"]): r for r in rows if r["kind"] == "far_at_S0" and r["n"] == n}
        thr = next((r for r in rows if r["kind"] == "threshold" and r["name"] == REF_FAR and r["n"] == n), None)
        if thr is None or not far_rows:
            continue
        # probe names are rounded to 3 dp when written, so match the NEAREST probe rather than the
        # full-precision constant -- an exact float lookup silently misses by 4e-4.
        s_obs = min(far_rows, key=lambda k: abs(k - S_OBS))
        if abs(s_obs - S_OBS) > 1e-2:
            continue
        obs = far_rows[s_obs]
        # local d(ln count)/dS by central difference over the probes bracketing S_OBS
        lo = max((p for p in probes if p < s_obs - 1e-9), default=None)
        hi = min((p for p in probes if p > s_obs + 1e-9), default=None)
        if lo is None or hi is None:
            continue
        c_lo, c_hi = far_rows[lo]["mean_count"], far_rows[hi]["mean_count"]
        if c_lo <= 0 or c_hi <= 0:
            continue
        deriv = (np.log(c_lo) - np.log(c_hi)) / (hi - lo)          # positive: counts fall with S
        sd_thr = thr["sd"]
        pred_nats = deriv * sd_thr
        pred_dex = pred_nats / np.log(10)
        meas_dex = obs["sd_dex"]
        ratio = meas_dex / pred_dex if pred_dex > 0 else np.nan
        out.append({"n": n, "dlnC_dS": float(deriv), "sd_threshold": float(sd_thr),
                    "pred_sd_dex": float(pred_dex), "meas_sd_dex": float(meas_dex),
                    "ratio_meas_over_pred": float(ratio),
                    "mean_count_at_S0": obs["mean_count"]})
        print(f"{n:5d} {deriv:8.3f} {sd_thr:8.3f} {pred_dex:10.3f} {meas_dex:9.3f} {ratio:6.2f}")

    r = np.array([o["ratio_meas_over_pred"] for o in out])
    tracks = bool(np.all((r > 0.5) & (r < 2.0)))
    trend = float(r[-1] / r[0]) if len(r) > 1 else float("nan")
    res = {"rows": out, "ratio_mean": float(r.mean()), "ratio_range": [float(r.min()), float(r.max())],
           "tracks_within_2x": tracks, "ratio_trend_large_over_small_n": trend,
           "reference_far": REF_FAR, "s_obs": S_OBS}
    print(f"\nmeasured/predicted: mean {r.mean():.2f}, range [{r.min():.2f}, {r.max():.2f}], "
          f"trend across n {trend:.2f}x")
    res["verdict"] = (
        "TRACKS — the FAR spread is the threshold spread pushed through a steep tail, so the two summaries "
        "carry the same information and neither escapes the glitch limit" if tracks else
        "DIVERGES — the FAR estimate carries a noise source the threshold does not; see where the ratio moves")
    print(f"VERDICT: {res['verdict']}")
    OUT.write_text(json.dumps(res, indent=2))
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
