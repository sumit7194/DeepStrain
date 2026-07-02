#!/usr/bin/env python
"""Milestone 21 (R2 v2): the PROPER tone-count test + NPE referee, via the `ringdown` package (Isi/Farr).

R2 was parked honestly: our simplified time-domain/white-noise Bayes factor could not separate 1 vs 2
tones, but the published GW250114 analysis (arXiv:2509.08099) detects the overtone — so ours was an
implementation limit, not an information limit. This runs the field-standard FD coherent pipeline
(`ringdown` v1.0.0, Python 3.11 venv) on both events:

  (a) VALIDATION — GW150914 with the documented example target (t0=1126259462.4083147, ra=1.95,
      dec=-1.27, psi=0.82; ringdown.readthedocs.io GW150914 example): does the (M, chi) posterior land
      on the known values (~68 Msun, ~0.63-0.69)?
  (b) THE R2 QUESTION — GW250114 220-only vs 220+221: is the 221 amplitude posterior bounded away from
      zero (the field's detection statistic, Isi/Farr style), where our simplified machinery saw nothing?
  (c) NPE REFEREE — does the package's (M, chi) posterior on GW250114 agree with our 09/10 NPE
      (M~68, chi~0.69)? First independent field-standard cross-check of the whole arc.

GW250114 target verified from arXiv:2601.05734 (Wang & Ma), which fixes the LVK maximum-likelihood
values: (ra, dec) = (2.35, 0.22) rad, psi = 1.37 rad, merger GPS t0 = 1420878141.2362.

Runs in .venv311 (jax 0.4.35 / numpyro 0.15.3 pins in .venv311-pins.txt). Strain comes from
20_extract_strain.py (128 s of raw 4 kHz GWOSC data per detector -> plenty of off-source for the ACF).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)   # float64 NUTS (default float32 warned)
import arviz as az
import ringdown

HERE = Path(__file__).resolve().parent.parent
DATA, RESULTS = HERE / "data", HERE / "results"
PROG = RESULTS / "progress"
PROG.mkdir(exist_ok=True, parents=True)

TARGETS = {
    # docs example values (ringdown.readthedocs.io GW150914 example)
    "GW150914": dict(t0=1126259462.4083147, ra=1.95, dec=-1.27, psi=0.82),
    # LVK max-likelihood values per arXiv:2601.05734
    "GW250114_082203": dict(t0=1420878141.2362, ra=2.35, dec=0.22, psi=1.37),
}
DURATION = 0.05          # s of analysis segment (docs example value; ~12 tau220 for both events)
DS, FLOW = 2, 20.0       # condition: 4096 -> 2048 Hz, 20 Hz high-pass


def heartbeat(stage, frac):
    (PROG / "21_ringdown.json").write_text(json.dumps(
        {"run": "21_ringdown_crosscheck", "stage": stage, "progress": frac, "t": time.time()}))


def load_data(ev):
    z = np.load(DATA / f"20_strain_{ev}.npz")
    out = []
    for det in ("H1", "L1"):
        t0, fs, h = float(z[f"{det}_t0"]), float(z[f"{det}_fs"]), z[f"{det}_h"]
        out.append(ringdown.Data(h, index=t0 + np.arange(len(h)) / fs, ifo=det))
    return out


def run_fit(ev, n_modes):
    fit = ringdown.Fit(modes=[(1, -2, 2, 2, n) for n in range(n_modes)])
    for d in load_data(ev):
        fit.add_data(d)
    t = TARGETS[ev]
    fit.set_target(t["t0"], ra=t["ra"], dec=t["dec"], psi=t["psi"], duration=DURATION)
    fit.condition_data(ds=DS, f_min=FLOW)
    fit.update_model(m_min=40.0, m_max=200.0, a_scale_max=1e-20)
    print(f"[{ev} n={n_modes}] running NUTS ...", flush=True)
    t0 = time.time()
    fit.run()
    print(f"[{ev} n={n_modes}] done in {time.time()-t0:.0f}s", flush=True)
    post = fit.result.posterior
    print(f"  posterior vars: {list(post.data_vars)}", flush=True)
    out = {}
    for name in ("m", "chi", "a", "f", "tau", "g"):
        if name in post:
            v = np.asarray(post[name]).reshape(-1, *np.asarray(post[name]).shape[2:])
            out[name] = v
    # convergence diagnostics on the physics parameters (keep chain structure)
    diag = {}
    for name in ("m", "chi", "a"):
        if name in post:
            diag[f"rhat_{name}"] = float(np.max(np.asarray(az.rhat(post[name]).to_array())))
            diag[f"ess_{name}"] = float(np.min(np.asarray(az.ess(post[name]).to_array())))
    print(f"  diagnostics: {diag}", flush=True)
    return fit, out, diag


def q(x, qs=(5, 25, 50, 75, 95)):
    return {f"q{p}": float(np.percentile(x, p)) for p in qs}


def main() -> None:
    results = {"duration": DURATION, "ds": DS, "flow": FLOW, "targets": TARGETS}
    stages = [("GW150914", 2), ("GW250114_082203", 1), ("GW250114_082203", 2)]
    for i, (ev, n) in enumerate(stages):
        heartbeat(f"{ev}_n{n}", i / len(stages))
        fit, post, diag = run_fit(ev, n)
        r = {"n_modes": n, "diagnostics": diag}
        for name, v in post.items():
            if v.ndim == 1:
                r[name] = q(v)
            else:                               # per-mode (samples, modes)
                r[name] = {f"mode{j}": q(v[:, j]) for j in range(v.shape[1])}
        if n == 2 and "a" in post:
            a1 = post["a"][:, 1]
            r["a221_frac_below_10pct_median"] = float(np.mean(a1 < 0.10 * np.median(a1)))
            r["a221_over_a220_median"] = float(np.median(post["a"][:, 1] / post["a"][:, 0]))
        results[f"{ev}_n{n}"] = r
        m, c = r.get("m", {}).get("q50"), r.get("chi", {}).get("q50")
        print(f"  {ev} n={n}: M50={m}, chi50={c}", flush=True)
        np.savez(RESULTS / f"21_post_{ev}_n{n}.npz", **post)
    heartbeat("done", 1.0)
    (RESULTS / "21_ringdown_crosscheck.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {RESULTS/'21_ringdown_crosscheck.json'}")


if __name__ == "__main__":
    main()
