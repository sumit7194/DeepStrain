"""Pre-registered verdict for the noise-variety experiment (device_stream/PREREGISTRATION.md).

Written and committed 2026-10-23 BEFORE C2, C3 and C4 were scored (C0, C1, V0 and V1 were already known; see the
interim entry in RESULTS.md). Nothing here is chosen after seeing the remaining controls.

Primary (as pre-registered):
  per run      m = mean over the 3 mass bins of the `max` sensitive-distance fraction
               (track_eval.py --window-sec 64, the committed protocol; golden gate passed 2026-10-13/15)
  effect       delta = mean(m_V over seeds 0,1) - mean(m_C over seeds 0..4)
  scale        sd(C) = SAMPLE standard deviation of m_C over seeds 0..4 (ddof=1). The prereg says
               "sd(C over seeds 0-4)" without fixing ddof; ddof=1 is fixed here before C2-C4 are seen, and the
               ddof=0 value is printed only as a sensitivity line (it can only make |z| larger).
  z = delta / sd(C);  |z| < 2 => no detectable variety effect;  z >= 2 => helps;  z <= -2 => hurts.
Reported, not decisive: paired V_s - C_s for s = 0,1, and per-bin values.
Refuses to give a verdict unless all 7 runs are present.
"""
import json
import sys
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parent.parent / "results"
RUNS = {**{f"V{s}": f"dev_v{s}_stream_s{s}" for s in (0, 1)},
        **{f"C{s}": f"dev_c{s}_original_s{s}" for s in range(5)}}


def load(stem):
    p = RES / f"eval_{stem}_track_w64_{stem}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())["results"]["max"]["mf_distance_fraction"]


def main() -> int:
    bins = {k: load(v) for k, v in RUNS.items()}
    missing = [k for k, v in bins.items() if v is None]
    m = {k: float(np.mean(list(v.values()))) for k, v in bins.items() if v is not None}
    for k in RUNS:
        print(f"{k}: " + ("MISSING" if k in missing else
                          " ".join(f"{b}={x:.4f}" for b, x in bins[k].items()) + f"  mean={m[k]:.4f}"))
    if missing:
        print(f"NO VERDICT: missing {missing}")
        return 2
    mV = np.array([m["V0"], m["V1"]]); mC = np.array([m[f"C{s}"] for s in range(5)])
    delta = mV.mean() - mC.mean(); sd1 = mC.std(ddof=1); sd0 = mC.std(ddof=0); z = delta / sd1
    verdict = "VARIETY HELPS" if z >= 2 else "VARIETY HURTS" if z <= -2 else "NO DETECTABLE VARIETY EFFECT"
    paired = {s: m[f"V{s}"] - m[f"C{s}"] for s in (0, 1)}
    out = dict(per_run_mean=m, per_run_bins=bins, mean_V=mV.mean(), mean_C=mC.mean(), delta=delta,
               sd_C_ddof1=sd1, z=z, verdict=verdict, sensitivity_z_ddof0=delta / sd0, paired_V_minus_C=paired)
    print(f"\nmean V {mV.mean():.4f}  mean C {mC.mean():.4f}  delta {delta:+.4f}  sd(C,ddof=1) {sd1:.4f}  z {z:+.2f}")
    print(f"VERDICT (pre-registered |z|<2 rule): {verdict}")
    print(f"sensitivity: z with ddof=0 = {delta / sd0:+.2f};  paired V-C: s0 {paired[0]:+.4f}, s1 {paired[1]:+.4f}")
    (RES / "device_stream_verdict.json").write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
