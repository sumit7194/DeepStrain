#!/usr/bin/env python
"""Milestone 4: the injection harness — the referee.

Inject analytic Kerr 220 ringdowns of KNOWN (f, tau) into REAL off-source
detector noise, run the identical whiten+bandpass+fit pipeline, and measure
the bias and scatter of the recovered parameters.

Everything downstream (window choice, error bars on real events, trust in the
no-hair test) is calibrated here, NOT tuned on the real events.

Usage:
    python 04_injections.py            # GW150914-like loudness
    python 04_injections.py loud       # GW250114-like loudness
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import rdlib

MODE = sys.argv[1] if len(sys.argv) > 1 else "gw150914"
PLOTS = Path(__file__).resolve().parent.parent / "plots"
RESULTS = Path(__file__).resolve().parent.parent / "results"
PLOTS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

# truth: a GW150914-like remnant tone
F_TRUE, TAU_TRUE = 251.0, 4.13e-3
# raw-strain peak amplitudes; "loud" mimics GW250114's whitened peak (~2x 150914)
AMP_RAW = {"gw150914": 1.5e-21, "loud": 4.0e-21}[MODE.lower() if MODE != "loud" else "loud"]

# off-source noise: clean stretches well before GW150914 (same run, same detectors)
BASE_GPS = rdlib.event_gps("GW150914")
N_INJ = 16
T0_OFFSET_MS = 3.0

rng = np.random.default_rng(42)
kerr = rdlib.KerrMap(2, 2, 0)
rows = []

for k in range(N_INJ):
    # space injections 128 s apart, starting 600 s before the event
    center = BASE_GPS - 600 - 128 * k
    t_inj = center + rng.uniform(-10, 10)
    series = {}
    ok = True
    for det in ("H1", "L1"):
        try:
            from gwpy.timeseries import TimeSeries

            raw = TimeSeries.fetch_open_data(det, center - 32, center + 32, cache=True)
            params = [
                dict(f=F_TRUE, tau=TAU_TRUE, amp=AMP_RAW, phi=rng.uniform(-np.pi, np.pi))
            ]
            injected = rdlib.inject_ringdown(raw, t_inj, params)
            white = injected.whiten(4, 2)  # NO bandpass for fitting (see rdlib)
            seg = white.crop(t_inj - 1, t_inj + 1)
            series[det] = (seg.times.value, seg.value)
        except Exception as e:
            print(f"  [{k}] {det}: skipped ({e})")
            ok = False
            break
    if not ok:
        continue

    t0 = t_inj + T0_OFFSET_MS / 1000.0
    fit = rdlib.fit_modes(series, t0, n_modes=1, seed=k, window=0.03)
    f_hat, tau_hat = fit["modes"][0]["f"], fit["modes"][0]["tau"]
    try:
        m_hat, chi_hat = kerr.mass_chi(f_hat, tau_hat)
    except ValueError:
        m_hat = chi_hat = np.nan
    rows.append(dict(f=f_hat, tau=tau_hat, mass=m_hat, chi=chi_hat))
    print(
        f"  [{k}] f={f_hat:6.1f} Hz (true {F_TRUE}), tau={tau_hat*1e3:5.2f} ms "
        f"(true {TAU_TRUE*1e3:.2f}), M={m_hat:5.1f}, chi={chi_hat:.2f}"
    )

f_arr = np.array([r["f"] for r in rows])
tau_arr = np.array([r["tau"] for r in rows]) * 1e3
m_arr = np.array([r["mass"] for r in rows])
chi_arr = np.array([r["chi"] for r in rows])

summary = {
    "mode": MODE,
    "amp_raw": AMP_RAW,
    "f_true": F_TRUE,
    "tau_true_ms": TAU_TRUE * 1e3,
    "n": len(rows),
    "f_mean": float(np.nanmean(f_arr)),
    "f_std": float(np.nanstd(f_arr)),
    "tau_mean_ms": float(np.nanmean(tau_arr)),
    "tau_std_ms": float(np.nanstd(tau_arr)),
    "mass_mean": float(np.nanmean(m_arr)),
    "mass_std": float(np.nanstd(m_arr)),
    "chi_mean": float(np.nanmean(chi_arr)),
    "chi_std": float(np.nanstd(chi_arr)),
    "true_mass_chi_if_unbiased": kerr.mass_chi(F_TRUE, TAU_TRUE),
}
print("\nSummary:")
for key, val in summary.items():
    print(f"  {key}: {val}")

out_json = RESULTS / f"04_injections_{MODE}.json"
out_json.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
print(f"  wrote {out_json}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(f_arr, bins=10, color="steelblue", alpha=0.8)
axes[0].axvline(F_TRUE, color="r", ls="--", label=f"truth {F_TRUE} Hz")
axes[0].set_xlabel("recovered f [Hz]")
axes[0].legend()
axes[1].hist(tau_arr, bins=10, color="darkorange", alpha=0.8)
axes[1].axvline(TAU_TRUE * 1e3, color="r", ls="--", label=f"truth {TAU_TRUE*1e3:.2f} ms")
axes[1].set_xlabel("recovered tau [ms]")
axes[1].legend()
fig.suptitle(f"Injection recovery, {MODE} loudness, n={len(rows)} (t0 = peak+3 ms)")
fig.tight_layout()
fig.savefig(PLOTS / f"04_injections_{MODE}.png", dpi=140)
print(f"  wrote {PLOTS / f'04_injections_{MODE}.png'}")
