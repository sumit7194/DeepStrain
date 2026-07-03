#!/usr/bin/env python
"""Milestone 22 (B1, PLAN.md): field-standard START-TIME SWEEP — referee R3 with the ringdown package.

R3 (15_imr_referee) found our no-hair NPE carries a delta ~ -0.33 systematic when fit from the PEAK, decaying
to ~0 by ~6 ms post-peak — the early-time merger/overtone content the 220+221 model omits. This refits GW250114
with the field-standard FD coherent pipeline (ringdown 1.0.0) at a sweep of start-time offsets and watches:
  - A221 amplitude vs t0: a REAL overtone (tau221 ~ 1.4 ms) must DECAY fast as the start moves later;
  - M, chi vs t0: do they STABILIZE once the early-time content has damped (the R3 story), or drift?
Offsets in units of t_Mf = G M_f / c^3 = 68.1 * T_sun = 0.335 ms (GW250114 detector-frame remnant). This is
the same knob as our simplified 05/16 start-time studies, now with the coherent package -> an independent
cross-check of the start-time systematic at the heart of the overtone controversy.

Runs in .venv311. Strain from 20_extract_strain.py.
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import arviz as az
import ringdown

HERE = Path(__file__).resolve().parent
DATA, RESULTS = HERE.parent / "data", HERE.parent / "results"
PROG = RESULTS / "progress"; PROG.mkdir(exist_ok=True, parents=True)

# reuse 21's verified targets + conditioning
_cc = importlib.util.spec_from_file_location("cc", HERE / "21_ringdown_crosscheck.py")
cc = importlib.util.module_from_spec(_cc); _cc.loader.exec_module(cc)

EVENT = "GW250114_082203"
T_SUN = 4.925490947e-6
M_F = 68.1                              # detector-frame remnant (LSC / arXiv:2509.08099)
T_MF = M_F * T_SUN                      # 0.335 ms
OFFSETS_TMF = [0, 1, 2, 4, 6, 8, 10, 12, 16]   # in units of t_Mf; 16 t_Mf ~ 5.4 ms
DURATION = cc.DURATION


def heartbeat(i, n):
    (PROG / "22_starttime.json").write_text(json.dumps({"run": "22_starttime_sweep", "progress": i / n, "t": time.time()}))


def fit_at(offset_s):
    fit = ringdown.Fit(modes=[(1, -2, 2, 2, n) for n in range(2)])
    for d in cc.load_data(EVENT):
        fit.add_data(d)
    t = cc.TARGETS[EVENT]
    fit.set_target(t["t0"] + offset_s, ra=t["ra"], dec=t["dec"], psi=t["psi"], duration=DURATION)
    fit.condition_data(ds=cc.DS, f_min=cc.FLOW)
    fit.update_model(m_min=40.0, m_max=200.0, a_scale_max=1e-20)
    fit.run()
    p = fit.result.posterior
    m = np.asarray(p["m"]).reshape(-1)
    chi = np.asarray(p["chi"]).reshape(-1)
    a = np.asarray(p["a"]).reshape(-1, np.asarray(p["a"]).shape[-1])
    rhat = float(np.max([np.max(np.asarray(az.rhat(p[v]).to_array())) for v in ("m", "chi", "a")]))
    return {
        "m_med": float(np.median(m)), "m_lo": float(np.percentile(m, 5)), "m_hi": float(np.percentile(m, 95)),
        "chi_med": float(np.median(chi)), "chi_lo": float(np.percentile(chi, 5)), "chi_hi": float(np.percentile(chi, 95)),
        "a221_med": float(np.median(a[:, 1])), "a221_over_a220": float(np.median(a[:, 1] / a[:, 0])),
        "a221_frac_below_10pct_median": float(np.mean(a[:, 1] < 0.10 * np.median(a[:, 1]))),
        "rhat": rhat,
    }


def main() -> None:
    rows = []
    print(f"GW250114 start-time sweep (t_Mf = {T_MF*1e3:.3f} ms), 220+221, duration {DURATION}s\n")
    print(f"{'t0[t_Mf]':>9} {'t0[ms]':>7} | {'M_med':>6} {'chi_med':>7} {'A221/A220':>9} {'P(A221~0)':>9} {'rhat':>5}")
    for i, otmf in enumerate(OFFSETS_TMF):
        heartbeat(i, len(OFFSETS_TMF))
        off = otmf * T_MF
        r = fit_at(off); r["offset_tMf"] = otmf; r["offset_ms"] = off * 1e3
        rows.append(r)
        print(f"{otmf:>9} {off*1e3:>7.2f} | {r['m_med']:>6.1f} {r['chi_med']:>7.3f} "
              f"{r['a221_over_a220']:>9.3f} {r['a221_frac_below_10pct_median']:>9.3f} {r['rhat']:>5.3f}", flush=True)

    # trend summaries: does A221 decay, do M/chi stabilize?
    a_ratio = np.array([r["a221_over_a220"] for r in rows])
    p_zero = np.array([r["a221_frac_below_10pct_median"] for r in rows])
    m_med = np.array([r["m_med"] for r in rows])
    chi_med = np.array([r["chi_med"] for r in rows])
    out = {"event": EVENT, "t_Mf_ms": T_MF * 1e3, "duration": DURATION, "offsets_tMf": OFFSETS_TMF,
           "sweep": rows,
           "a221_decays": bool(a_ratio[-1] < 0.6 * a_ratio[0]),
           "overtone_significant_at_peak": bool(p_zero[0] < 0.05),
           "overtone_lost_by_end": bool(p_zero[-1] > 0.05),
           "m_drift_peak_to_end": float(m_med[-1] - m_med[0]),
           "chi_drift_peak_to_end": float(chi_med[-1] - chi_med[0])}
    (RESULTS / "22_starttime_sweep.json").write_text(json.dumps(out, indent=2))
    print(f"\nA221/A220: {a_ratio[0]:.2f} (peak) -> {a_ratio[-1]:.2f} (end)  decays={out['a221_decays']}")
    print(f"overtone P(A221~0): {p_zero[0]:.3f} (peak, sig={out['overtone_significant_at_peak']}) "
          f"-> {p_zero[-1]:.3f} (end, lost={out['overtone_lost_by_end']})")
    print(f"M drift peak->end: {out['m_drift_peak_to_end']:+.1f} Msun; chi drift: {out['chi_drift_peak_to_end']:+.3f}")
    print("wrote 22_starttime_sweep.json")


if __name__ == "__main__":
    main()
