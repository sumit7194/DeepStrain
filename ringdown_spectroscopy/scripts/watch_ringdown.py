#!/usr/bin/env python
"""Event-watcher stage (venv311): field-standard 220+221 ringdown fit for one event -> JSON.

Reads an event config {name, gps, ra, dec, psi} (path as argv[1]); expects strain at
data/20_strain_<name>.npz (the orchestrator runs 20_extract_strain first). Emits the remnant (M, chi),
the overtone significance (P(A221 bounded away from zero) at the peak), and NUTS convergence.
Reuses 21_ringdown_crosscheck's verified conditioning.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import arviz as az
import ringdown

HERE = Path(__file__).resolve().parent
_cc = importlib.util.spec_from_file_location("cc", HERE / "21_ringdown_crosscheck.py")
cc = importlib.util.module_from_spec(_cc); _cc.loader.exec_module(cc)


def main() -> None:
    cfg = json.loads(Path(sys.argv[1]).read_text())
    name = cfg["name"]
    fit = ringdown.Fit(modes=[(1, -2, 2, 2, n) for n in range(2)])
    z = np.load(HERE.parent / "data" / f"20_strain_{name}.npz")
    for det in ("H1", "L1"):
        t0, fs, h = float(z[f"{det}_t0"]), float(z[f"{det}_fs"]), z[f"{det}_h"]
        fit.add_data(ringdown.Data(h, index=t0 + np.arange(len(h)) / fs, ifo=det))
    fit.set_target(cfg["gps"], ra=cfg["ra"], dec=cfg["dec"], psi=cfg["psi"], duration=cc.DURATION)
    fit.condition_data(ds=cc.DS, f_min=cc.FLOW)
    fit.update_model(m_min=40.0, m_max=200.0, a_scale_max=1e-20)
    fit.run()
    p = fit.result.posterior
    m = np.asarray(p["m"]).reshape(-1); chi = np.asarray(p["chi"]).reshape(-1)
    a = np.asarray(p["a"]).reshape(-1, np.asarray(p["a"]).shape[-1])
    rhat = float(np.max([np.max(np.asarray(az.rhat(p[v]).to_array())) for v in ("m", "chi", "a")]))
    a221_p = float(np.mean(a[:, 1] < 0.10 * np.median(a[:, 1])))
    out = {
        "stage": "ringdown_package",
        "M": [float(np.median(m)), float(np.percentile(m, 5)), float(np.percentile(m, 95))],
        "chi": [float(np.median(chi)), float(np.percentile(chi, 5)), float(np.percentile(chi, 95))],
        "a221_over_a220": float(np.median(a[:, 1] / a[:, 0])),
        "overtone_p_below": a221_p, "overtone_detected": bool(a221_p < 0.05), "rhat": rhat,
    }
    Path(cfg["_out"]).write_text(json.dumps(out, indent=2))
    print(f"[ringdown] {name}: M {out['M'][0]:.1f}, chi {out['chi'][0]:.3f}, overtone P {a221_p:.3f}", flush=True)


if __name__ == "__main__":
    main()
