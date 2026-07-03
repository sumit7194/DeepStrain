#!/usr/bin/env python
"""Milestone 23 (B3, PLAN.md): close the NPE loop — locate our amortized NPE in the field-standard family.

R2/21 showed the NPE median (M,chi) agrees with the ringdown package at a fixed peak start. B1/22 showed the
package's (M,chi) drift with start time (the R3 systematic). B3 ties them together: the NPE marginalizes the
start time BY CONSTRUCTION, so WHERE in the package's start-time family does its posterior actually land?
Pure synthesis of committed artifacts (09 NPE + 21 package-fixed + 22 package-sweep) — no new fits.

Closes the loop two ways:
  (1) agreement: NPE (M,chi) median vs the package -> the amortized network does real, field-consistent inference;
  (2) location: the NPE median vs the start-time sweep -> whether it weights the (biased) peak or the late-start,
      i.e. does the NPE inherit the R3/B1 early-time systematic?
"""
import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results"
M_TRUE = 68.1   # detector-frame remnant (LSC / arXiv:2509.08099)


def main() -> None:
    npe = json.loads((RESULTS / "09_nohair_GW250114.json").read_text())["posterior"]
    pkg = json.loads((RESULTS / "21_ringdown_crosscheck.json").read_text())["GW250114_082203_n2"]
    sweep = json.loads((RESULTS / "22_starttime_sweep.json").read_text())["sweep"]

    npe_m, npe_m_lo, npe_m_hi = npe["mass"]
    npe_c, npe_c_lo, npe_c_hi = npe["chi"]
    pkg_m, pkg_m_lo, pkg_m_hi = pkg["m"]["q50"], pkg["m"]["q5"], pkg["m"]["q95"]
    pkg_c = pkg["chi"]["q50"]

    # (1) agreement: median gap + CI nesting
    dm, dc = abs(npe_m - pkg_m), abs(npe_c - pkg_c)
    pkg_ci_in_npe = (npe_m_lo <= pkg_m_lo and pkg_m_hi <= npe_m_hi)

    # (2) location: interpolate the start-time offset (t_Mf) at which the package M = the NPE median.
    off = np.array([r["offset_tMf"] for r in sweep])
    ms = np.array([r["m_med"] for r in sweep])
    # package M is ~monotone-decreasing in start time; NPE median sits where it crosses (clip to the peak if above)
    if npe_m >= ms[0]:
        npe_location_tMf = 0.0   # at or before the peak -> the earliest, highest-SNR regime
    else:
        order = np.argsort(ms)
        npe_location_tMf = float(np.interp(npe_m, ms[order], off[order]))

    npe_bias = npe_m - M_TRUE
    peak_bias = ms[0] - M_TRUE
    out = {
        "npe": {"m": [npe_m, npe_m_lo, npe_m_hi], "chi": [npe_c, npe_c_lo, npe_c_hi], "delta": npe["delta"]},
        "package_peak": {"m": [pkg_m, pkg_m_lo, pkg_m_hi], "chi": pkg_c},
        "m_median_gap": dm, "chi_median_gap": dc, "package_ci_nested_in_npe": bool(pkg_ci_in_npe),
        "npe_location_in_sweep_tMf": npe_location_tMf,
        "npe_mass_bias_vs_true": float(npe_bias), "peak_mass_bias_vs_true": float(peak_bias),
        "npe_inherits_peak_systematic": bool(npe_location_tMf < 4.0 and npe_bias > 4.0),
    }
    print(f"NPE     M {npe_m:.1f} [{npe_m_lo:.1f},{npe_m_hi:.1f}]  chi {npe_c:.3f}  delta {npe['delta'][0]:+.2f}")
    print(f"package M {pkg_m:.1f} [{pkg_m_lo:.1f},{pkg_m_hi:.1f}]  chi {pkg_c:.3f}  (peak start)")
    print(f"(1) AGREEMENT: median gap M {dm:.1f} Msun / chi {dc:.3f}; package 90% CI nested in NPE's: {pkg_ci_in_npe}")
    print(f"(2) LOCATION: NPE median sits at ~{npe_location_tMf:.0f} t_Mf in the start-time sweep "
          f"(peak=0); NPE mass bias vs true {M_TRUE}: {npe_bias:+.1f} Msun (peak bias {peak_bias:+.1f})")
    print(f"-> NPE inherits the peak-start systematic: {out['npe_inherits_peak_systematic']}")
    (RESULTS / "23_npe_package_loop.json").write_text(json.dumps(out, indent=2))
    print("wrote 23_npe_package_loop.json")


if __name__ == "__main__":
    main()
