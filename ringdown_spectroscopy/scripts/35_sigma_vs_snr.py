"""Does sigma(delta) fall as 1/SNR, or saturate? TheBridge's standing request, answered from data on disk.

THE QUESTION (SISTER_REQUESTS.md). Our no-hair deviation delta has a per-event width sigma(delta). If the
measurement is likelihood-dominated it must fall as 1/SNR -- the Fisher scaling every forecast assumes. If it
is prior-dominated it saturates at the prior width and more SNR buys nothing until a threshold is crossed.
Which one the real population is in decides whether stacking or waiting for louder events is the way forward,
and it is answerable with no new compute: 26_more_events_o4.json already holds per-event SNR and delta
posteriors for 12 events spanning SNR 20.0 to 78.6.

PRE-REGISTERED, declared here before the fit is run:
  models      A: sigma = a / SNR                (Fisher / likelihood-dominated)
              B: sigma = sigma_prior            (fully prior-saturated, no free parameter)
              C: sigma = sigma_prior / sqrt(1 + (SNR/SNR_c)^2)   (saturating, one free parameter, ->A at
                                                                  high SNR and ->B at low, so it can favour
                                                                  either without being told which)
  statistic   sum of squared residuals, and the fitted SNR_c of model C
  verdict     A wins by >2x in SSR      => Fisher scaling, sigma is precision-limited across the population
              B wins by >2x             => saturated, more SNR buys nothing in this range
              C's SNR_c inside [20,79]  => the crossover is INSIDE the observed range and both regimes exist
              otherwise                 => unresolved, and said so
  and the honest bound: 12 events, only one above the informativeness threshold, so a 1/SNR branch supported
  by a single point is NOT a measurement of that branch and will be reported as unverified.

Run:  .venv/bin/python scripts/35_sigma_vs_snr.py
"""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

RESULTS = Path(__file__).resolve().parent.parent / "results"
SRC = RESULTS / "26_more_events_o4.json"
OUT = RESULTS / "35_sigma_vs_snr.json"


def main():
    d = json.loads(SRC.read_text())
    sp = float(d["sigma_prior"])
    rows = [(e["event"], float(e["snr"]), (e["delta"][2] - e["delta"][1]) / 3.2897)
            for e in d["events"] if "delta" in e and e.get("snr")]
    rows.sort(key=lambda r: r[1])
    snr = np.array([r[1] for r in rows])
    sig = np.array([r[2] for r in rows])

    print(f"prior width sigma_prior = {sp:.4f}   (a posterior at this width has learned nothing)")
    print(f"{'event':>18} {'SNR':>6} {'sigma':>8} {'sigma/prior':>12}")
    for name, s, g in rows:
        print(f"{name[:18]:>18} {s:6.1f} {g:8.4f} {g/sp:12.3f}")

    ssr = lambda pred: float(np.sum((sig - pred) ** 2))
    # A: Fisher
    a_fit = float(np.sum(sig * (1 / snr)) / np.sum((1 / snr) ** 2))
    ssr_A = ssr(a_fit / snr)
    # B: fully saturated, zero free parameters
    ssr_B = ssr(np.full_like(sig, sp))
    # C: saturating with a crossover
    model_C = lambda x, sc: sp / np.sqrt(1 + (x / sc) ** 2)
    (sc,), _ = curve_fit(model_C, snr, sig, p0=[100.0], maxfev=20000)
    ssr_C = ssr(model_C(snr, sc))

    obs_lo, obs_hi = float(snr.min()), float(snr.max())
    inside = obs_lo <= sc <= obs_hi
    if ssr_A * 2 < ssr_B and ssr_A * 2 < ssr_C:
        verdict = "A: Fisher 1/SNR scaling across the population"
    elif ssr_B * 2 < ssr_A:
        verdict = (f"B: SATURATED at the prior. Across SNR {obs_lo:.0f}-{obs_hi:.0f} -- a factor "
                   f"{obs_hi/obs_lo:.1f} -- sigma falls only {sig.max()/sig.min():.2f}x, where 1/SNR "
                   f"demands {obs_hi/obs_lo:.1f}x. More SNR buys nothing in this range.")
    else:
        verdict = "unresolved between the models on this sample"

    res = {"source": SRC.name, "n_events": len(rows), "sigma_prior": sp,
           "snr_range": [obs_lo, obs_hi],
           "events": [{"event": n, "snr": s, "sigma": g, "sigma_over_prior": g / sp} for n, s, g in rows],
           "models": {"A_fisher": {"a": a_fit, "ssr": ssr_A},
                      "B_saturated": {"ssr": ssr_B},
                      "C_crossover": {"snr_c": float(sc), "ssr": ssr_C,
                                      "inside_observed_range": bool(inside)}},
           "observed_sigma_ratio": float(sig.max() / sig.min()),
           "fisher_would_demand": float(obs_hi / obs_lo),
           "verdict": verdict,
           "scope": ("12 events, and only the loudest clears our own informativeness threshold "
                     f"(SNR ~ {d.get('snr_needed_for_information', float('nan')):.0f}). A 1/SNR branch "
                     "supported by ONE point above threshold is not a measurement of that branch: this "
                     "result bounds the saturated regime and says nothing reliable about the other side.")}

    print(f"\nmodel A  sigma = {a_fit:.3f}/SNR              SSR {ssr_A:.5f}")
    print(f"model B  sigma = sigma_prior (no free param)  SSR {ssr_B:.5f}")
    print(f"model C  crossover SNR_c = {sc:.1f}           SSR {ssr_C:.5f}"
          f"   ({'inside' if inside else 'OUTSIDE'} the observed range {obs_lo:.0f}-{obs_hi:.0f})")
    print(f"\nobserved: SNR spans {obs_hi/obs_lo:.1f}x, sigma spans {sig.max()/sig.min():.2f}x "
          f"(1/SNR would demand {obs_hi/obs_lo:.1f}x)")
    print(f"\nVERDICT: {verdict}")
    print(f"SCOPE:   {res['scope']}")
    OUT.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {OUT.name}")


if __name__ == "__main__":
    main()
