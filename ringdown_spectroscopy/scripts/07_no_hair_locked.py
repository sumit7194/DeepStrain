#!/usr/bin/env python
"""Milestone 7: the no-hair test, done the honest way.

06 proved (with injections) that two fully-free tones cannot be separated by
least squares — the 220 and 221 are ~6 Hz apart and the overtone dies in
~1.4 ms. So we run the PARAMETERIZED test (the simplified spirit of the LVK
analysis): both tones are LOCKED to their Kerr positions as functions of
(M, chi), except the overtone frequency is scaled by (1 + delta).

    delta = 0           <=> the overtone sits exactly where Kerr predicts
    fit prefers delta=0 <=> the no-hair fingerprint passes

The (M, chi, amplitudes, phases) are re-fit at every delta on a grid; the
cost-vs-delta curve is the test. Injections with TRUE delta=0 calibrate how
precisely delta can be measured at this loudness.

Usage:
    python 07_no_hair_locked.py [n_injections]
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from gwpy.timeseries import TimeSeries
from scipy.optimize import least_squares

import rdlib

N_INJ = int(sys.argv[1]) if len(sys.argv) > 1 else 10
EVENT = "GW250114_082203"
PLOTS = Path(__file__).resolve().parent.parent / "plots"
RESULTS = Path(__file__).resolve().parent.parent / "results"
DELTAS = np.linspace(-0.4, 0.4, 21)
WINDOW = 0.03

kerr220, kerr221 = rdlib.KerrMap(2, 2, 0), rdlib.KerrMap(2, 2, 1)


def kerr_locked_cost(series, t0, delta, m_init, chi_init, seed=0, n_restarts=8):
    """Best least-squares cost of the Kerr-locked two-tone model at fixed delta.

    Free: M, chi, per-detector (A1, phi1, A2, phi2). Tones locked to Kerr(M,chi),
    overtone frequency scaled by (1+delta)."""
    rng = np.random.default_rng(seed)
    dets = sorted(series)
    crops = {}
    for det in dets:
        t, v = series[det]
        m = (t >= t0 - 0.002) & (t <= t0 + WINDOW)
        crops[det] = (t[m], v[m])

    def resid(x):
        mass, chi = x[0], x[1]
        f1, tau1 = kerr220.f_tau(mass, chi)
        f2, tau2 = kerr221.f_tau(mass, chi)
        f2 = f2 * (1.0 + delta)
        res = []
        for i, det in enumerate(dets):
            a1, p1, a2, p2 = x[2 + 4 * i: 6 + 4 * i]
            t, v = crops[det]
            model = rdlib.damped_sinusoids(
                t, t0,
                [dict(f=f1, tau=tau1, amp=a1, phi=p1),
                 dict(f=f2, tau=tau2, amp=a2, phi=p2)],
            )
            res.append(v - model)
        return np.concatenate(res)

    lo = [40.0, 0.0] + [0.0, -np.pi, 0.0, -np.pi] * len(dets)
    hi = [120.0, 0.99] + [50.0, np.pi, 50.0, np.pi] * len(dets)
    best = None
    for _ in range(n_restarts):
        x0 = [np.clip(rng.normal(m_init, 5), 45, 115),
              np.clip(rng.normal(chi_init, 0.08), 0.05, 0.95)]
        for _ in dets:
            x0 += [rng.uniform(0.5, 8), rng.uniform(-np.pi, np.pi),
                   rng.uniform(0.5, 8), rng.uniform(-np.pi, np.pi)]
        try:
            sol = least_squares(resid, x0, bounds=(lo, hi), max_nfev=3000)
        except Exception:
            continue
        if best is None or sol.cost < best.cost:
            best = sol
    return float(best.cost), float(best.x[0]), float(best.x[1])


def delta_scan(series, t0, m_init, chi_init, seed=0):
    costs, masses, chis = [], [], []
    for j, d in enumerate(DELTAS):
        c, m, x = kerr_locked_cost(series, t0, d, m_init, chi_init, seed=seed * 100 + j)
        costs.append(c)
        masses.append(m)
        chis.append(x)
    costs = np.array(costs)
    return costs, DELTAS[costs.argmin()], masses, chis


def get_series(det_data):
    return {d: det_data[d] for d in det_data}


# ---------------------------------------------------- step A: 220-only anchor
gps = rdlib.event_gps(EVENT)
series, peaks = {}, {}
for det in ("H1", "L1"):
    white = rdlib.fetch_whitened(det, gps, bandpass=False)
    peaks[det] = rdlib.find_peak(white.bandpass(*rdlib.BAND), gps)
    seg = white.crop(gps - 1, gps + 1)
    series[det] = (seg.times.value, seg.value)
peak = max(peaks.values())
anchor = rdlib.fit_modes(series, peak + 0.003, n_modes=1, seed=3, window=WINDOW)
f_a, tau_a = anchor["modes"][0]["f"], anchor["modes"][0]["tau"]
m_init, chi_init = kerr220.mass_chi(f_a, tau_a)
print(f"anchor (220-only @ peak+3ms): M={m_init:.1f}, chi={chi_init:.2f}")

# ---------------------------------------------------- step B: real-event scan
costs, d_best, _, _ = delta_scan(series, peak, m_init, chi_init, seed=1)
print(f"{EVENT}: best delta = {d_best:+.2f}  (0 = Kerr)")

# ---------------------------------------------------- step C: calibration
BASE = rdlib.event_gps(EVENT)  # calibrate in O4 noise around the event itself
F1_T, TAU1_T = kerr220.f_tau(68.0, 0.69)
F2_T, TAU2_T = kerr221.f_tau(68.0, 0.69)
rng = np.random.default_rng(11)
d_hats, inj_curves = [], []
for k in range(N_INJ):
    center = BASE - 300 - 128 * k
    t_inj = center + rng.uniform(-10, 10)
    s_inj, ok = {}, True
    for det in ("H1", "L1"):
        try:
            raw = TimeSeries.fetch_open_data(det, center - 32, center + 32, cache=True)
            params = [
                dict(f=F1_T, tau=TAU1_T, amp=2.0e-21, phi=rng.uniform(-np.pi, np.pi)),
                dict(f=F2_T, tau=TAU2_T, amp=2.0e-21, phi=rng.uniform(-np.pi, np.pi)),
            ]
            if not np.isfinite(raw.value).all():
                raise ValueError("segment contains NaNs")
            inj = rdlib.inject_ringdown(raw, t_inj, params)
            seg = inj.whiten(4, 2).crop(t_inj - 1, t_inj + 1)
            s_inj[det] = (seg.times.value, seg.value)
        except Exception as e:
            print(f"  [inj {k}] skipped: {e}")
            ok = False
            break
    if not ok:
        continue
    c_inj, d_hat, _, _ = delta_scan(s_inj, t_inj, 68.0, 0.69, seed=50 + k)
    d_hats.append(d_hat)
    inj_curves.append(c_inj - c_inj.min())
    print(f"  [inj {k}] delta_hat = {d_hat:+.2f}")

d_hats = np.array(d_hats)
sigma = d_hats.std()
print(f"\ncalibration: delta_hat = {d_hats.mean():+.3f} +- {sigma:.3f} (true 0)")
verdict = abs(d_best) <= 2 * sigma
print(f"VERDICT: real-event delta = {d_best:+.2f}, calibrated 2-sigma = {2*sigma:.2f} "
      f"=> {'CONSISTENT with Kerr (no-hair passes)' if verdict else 'TENSION with Kerr'}")

# ---------------------------------------------------- plot
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
ax = axes[0]
ax.plot(DELTAS, costs - costs.min(), "o-", color="navy", label=EVENT)
for c in inj_curves:
    ax.plot(DELTAS, c, color="gray", alpha=0.25, lw=0.8)
ax.axvline(0, color="g", ls="--", label="Kerr prediction (delta=0)")
ax.axvline(d_best, color="navy", ls=":", label=f"best fit {d_best:+.2f}")
ax.set_xlabel("overtone frequency deviation delta")
ax.set_ylabel("delta-cost (best-fit residual, offset)")
ax.set_title("Does the overtone sit where Kerr says?")
ax.legend(fontsize=8)

ax = axes[1]
ax.hist(d_hats, bins=8, color="gray", alpha=0.7, label=f"injections (true delta=0)")
ax.axvline(d_best, color="navy", lw=2, label=f"{EVENT}: {d_best:+.2f}")
ax.axvline(0, color="g", ls="--")
ax.set_xlabel("recovered delta")
ax.set_title(f"calibration: sigma(delta)={sigma:.2f} at this loudness")
ax.legend(fontsize=8)
fig.suptitle("No-hair test, parameterized: overtone locked to Kerr except a frequency slide")
fig.tight_layout()
fig.savefig(PLOTS / "07_no_hair_locked.png", dpi=140)
print(f"wrote {PLOTS / '07_no_hair_locked.png'}")

out = dict(event=EVENT, anchor=dict(mass=m_init, chi=chi_init),
           deltas=DELTAS.tolist(), costs=costs.tolist(), delta_best=float(d_best),
           injections_delta_hat=d_hats.tolist(), sigma=float(sigma),
           verdict="consistent-with-Kerr" if verdict else "tension")
(RESULTS / "07_no_hair_locked.json").write_text(json.dumps(out, indent=2))
print(f"wrote {RESULTS / '07_no_hair_locked.json'}")
