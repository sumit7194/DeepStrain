#!/usr/bin/env python
"""Milestone 6: THE TEST — two tones, one black hole?

The no-hair logic: fit TWO damped sinusoids (220 + 221 overtone) at the peak
of GW250114. Invert tone 1 with the Kerr-220 map and tone 2 with the Kerr-221
map. If the remnant is Einstein's black hole, both must point at the SAME
(mass, spin).

Error bars are empirical: the same two-tone fit is run on N injections of a
KNOWN Kerr two-tone signal (amplitudes matched to GW250114) in real off-source
noise; the per-injection recovery errors, applied as displacements around the
real-event point, form the uncertainty clouds.

Usage:
    python 06_no_hair.py [n_injections]
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from gwpy.timeseries import TimeSeries

import rdlib

N_INJ = int(sys.argv[1]) if len(sys.argv) > 1 else 14
EVENT = "GW250114_082203"
PLOTS = Path(__file__).resolve().parent.parent / "plots"
RESULTS = Path(__file__).resolve().parent.parent / "results"
PLOTS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

# ---- truth for the calibration injections: Kerr remnant M=68, chi=0.69
kerr220, kerr221 = rdlib.KerrMap(2, 2, 0), rdlib.KerrMap(2, 2, 1)
M_TRUE, CHI_TRUE = 68.0, 0.69
F1_T, TAU1_T = kerr220.f_tau(M_TRUE, CHI_TRUE)
F2_T, TAU2_T = kerr221.f_tau(M_TRUE, CHI_TRUE)
AMP, RATIO = 4.0e-21, 1.0  # 220 raw amp (GW250114-like); 221/220 amplitude ratio
print(f"calibration truth: 220 f={F1_T:.1f} Hz tau={TAU1_T*1e3:.2f} ms | "
      f"221 f={F2_T:.1f} Hz tau={TAU2_T*1e3:.2f} ms")

TWO_TONE_KW = dict(
    n_modes=2,
    f_bounds=((180.0, 320.0), (180.0, 320.0)),
    tau_bounds=((2e-3, 0.012), (4e-4, 2.5e-3)),
    n_restarts=40,
    window=0.03,
)


def two_tone_fit(series, t0, seed):
    fit = rdlib.fit_modes(series, t0, seed=seed, **TWO_TONE_KW)
    (m1, m2) = fit["modes"]  # sorted: longest tau first = the 220 candidate
    return m1, m2


# ------------------------------------------------------------- 1) calibration
BASE = rdlib.event_gps("GW150914")  # off-source noise from the O1 run
rng = np.random.default_rng(7)
cal = []
for k in range(N_INJ):
    center = BASE - 600 - 128 * k
    t_inj = center + rng.uniform(-10, 10)
    series, ok = {}, True
    for det in ("H1", "L1"):
        try:
            raw = TimeSeries.fetch_open_data(det, center - 32, center + 32, cache=True)
            params = [
                dict(f=F1_T, tau=TAU1_T, amp=AMP, phi=rng.uniform(-np.pi, np.pi)),
                dict(f=F2_T, tau=TAU2_T, amp=AMP * RATIO, phi=rng.uniform(-np.pi, np.pi)),
            ]
            inj = rdlib.inject_ringdown(raw, t_inj, params)
            seg = inj.whiten(4, 2).crop(t_inj - 1, t_inj + 1)
            series[det] = (seg.times.value, seg.value)
        except Exception as e:
            print(f"  [{k}] {det} skipped: {e}")
            ok = False
            break
    if not ok:
        continue
    m1, m2 = two_tone_fit(series, t_inj, seed=100 + k)
    row = dict(f1=m1["f"], tau1=m1["tau"], f2=m2["f"], tau2=m2["tau"])
    try:
        row["mass1"], row["chi1"] = kerr220.mass_chi(m1["f"], m1["tau"])
    except ValueError:
        row["mass1"] = row["chi1"] = np.nan
    try:
        row["mass2"], row["chi2"] = kerr221.mass_chi(m2["f"], m2["tau"])
    except ValueError:
        row["mass2"] = row["chi2"] = np.nan
    cal.append(row)
    print(f"  [inj {k}] 220: f={m1['f']:.1f} tau={m1['tau']*1e3:.2f} -> "
          f"M={row['mass1']:.1f} chi={row['chi1']:.2f} | "
          f"221: f={m2['f']:.1f} tau={m2['tau']*1e3:.2f} -> "
          f"M={row.get('mass2', np.nan):.1f} chi={row.get('chi2', np.nan):.2f}")

# ------------------------------------------------------------- 2) real event
gps = rdlib.event_gps(EVENT)
series, peaks = {}, {}
for det in ("H1", "L1"):
    white = rdlib.fetch_whitened(det, gps, bandpass=False)
    peaks[det] = rdlib.find_peak(white.bandpass(*rdlib.BAND), gps)
    seg = white.crop(gps - 1, gps + 1)
    series[det] = (seg.times.value, seg.value)
peak = max(peaks.values())
m1, m2 = two_tone_fit(series, peak, seed=1)
mass1, chi1 = kerr220.mass_chi(m1["f"], m1["tau"])
try:
    mass2, chi2 = kerr221.mass_chi(m2["f"], m2["tau"])
    invertible = True
except ValueError as e:
    mass2 = chi2 = np.nan
    invertible = False
    print(f"  221 tone not Kerr-invertible: {e}")

print(f"\n{EVENT} two-tone fit at the peak:")
print(f"  tone 1 (220): f={m1['f']:.1f} Hz, tau={m1['tau']*1e3:.2f} ms -> "
      f"M={mass1:.1f} M_sun, chi={chi1:.2f}")
print(f"  tone 2 (221): f={m2['f']:.1f} Hz, tau={m2['tau']*1e3:.2f} ms -> "
      f"M={mass2:.1f} M_sun, chi={chi2:.2f}")

# ------------------------------------------------------------- 3) verdict + plot
d1 = np.array([[r["mass1"] - M_TRUE, r["chi1"] - CHI_TRUE] for r in cal
               if np.isfinite(r["mass1"])])
d2 = np.array([[r["mass2"] - M_TRUE, r["chi2"] - CHI_TRUE] for r in cal
               if np.isfinite(r.get("mass2", np.nan))])
n2_fail = sum(1 for r in cal if not np.isfinite(r.get("mass2", np.nan)))

cloud1 = np.array([mass1, chi1]) + d1
cloud2 = (np.array([mass2, chi2]) + d2) if invertible and len(d2) else np.empty((0, 2))

fig, ax = plt.subplots(figsize=(7.5, 6))
ax.scatter(cloud1[:, 0], cloud1[:, 1], s=28, alpha=0.55, color="steelblue",
           label=f"tone 220 cloud (n={len(cloud1)})")
if len(cloud2):
    ax.scatter(cloud2[:, 0], cloud2[:, 1], s=28, alpha=0.55, color="darkorange",
               label=f"tone 221 cloud (n={len(cloud2)}, {n2_fail} fail)")
ax.scatter([mass1], [chi1], s=140, marker="*", color="navy", zorder=5,
           label=f"220: M={mass1:.0f}, chi={chi1:.2f}")
if invertible:
    ax.scatter([mass2], [chi2], s=140, marker="*", color="darkred", zorder=5,
               label=f"221: M={mass2:.0f}, chi={chi2:.2f}")
ax.set_xlabel("remnant mass [M_sun, detector frame]")
ax.set_ylabel("remnant spin chi")
ax.set_title(f"{EVENT}: do both tones point at the same black hole?\n"
             "(clouds = injection-calibrated errors, real off-source noise)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(PLOTS / "06_no_hair_GW250114.png", dpi=140)
print(f"  wrote {PLOTS / '06_no_hair_GW250114.png'}")

out = dict(
    event=EVENT,
    tone220=dict(f=m1["f"], tau_ms=m1["tau"] * 1e3, mass=mass1, chi=chi1),
    tone221=dict(f=m2["f"], tau_ms=m2["tau"] * 1e3, mass=mass2, chi=chi2),
    calibration=cal,
    n_overtone_invert_fail=n2_fail,
)
(RESULTS / "06_no_hair_GW250114.json").write_text(json.dumps(out, indent=2, default=float))
print(f"  wrote {RESULTS / '06_no_hair_GW250114.json'}")
