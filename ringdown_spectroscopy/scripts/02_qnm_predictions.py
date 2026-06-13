#!/usr/bin/env python
"""Milestone 2: the predicted Kerr tone table.

Given a remnant's (detector-frame) mass and spin, print each quasinormal mode's
frequency [Hz] and damping time [ms] — the "no-hair" fingerprint that the data
must match if the remnant is Einstein's black hole.

Usage:
    python 02_qnm_predictions.py 68.0 0.69     # ~GW150914 remnant (detector frame)

Note: observed frequencies scale with the DETECTOR-frame mass,
M_det = M_source x (1 + z).
"""
import sys

import qnm

M_SUN_SECONDS = 4.925490947641267e-06  # G * M_sun / c^3

mass = float(sys.argv[1]) if len(sys.argv) > 1 else 68.0
chi = float(sys.argv[2]) if len(sys.argv) > 2 else 0.69
t_m = mass * M_SUN_SECONDS

MODES = [(2, 2, 0), (2, 2, 1), (3, 3, 0), (4, 4, 0)]

print(f"Kerr remnant: M = {mass:.1f} M_sun (detector frame), chi = {chi:.3f}")
print(f"{'mode (l,m,n)':>14} | {'f [Hz]':>8} | {'tau [ms]':>8} | {'Q':>6}")
print("-" * 48)
for ell, m, n in MODES:
    omega, _, _ = qnm.modes_cache(s=-2, l=ell, m=m, n=n)(a=chi)
    f_hz = omega.real / (2 * 3.141592653589793 * t_m)
    tau_s = -t_m / omega.imag
    quality = 3.141592653589793 * f_hz * tau_s
    print(f"  ({ell},{m},{n})      | {f_hz:8.1f} | {tau_s * 1e3:8.2f} | {quality:6.2f}")

print(
    "\nThe no-hair test: tone (2,2,0) alone fixes (M, chi); tone (2,2,1) must then"
    "\nland exactly where this table says. If it doesn't, the bell isn't Kerr."
)
