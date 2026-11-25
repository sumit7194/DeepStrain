"""Step T of PREREG_coherent2.md (amendment T1, d4c3ca4; registration reports/stepT_registration.md): build and gates.
[model: Claude Opus 5.5]

  python benchmark/coh2t.py s0 [det] [gps]     S0: the PyCBC call path for v (PSD variation) and newSNR      (CPU, < 1 min)

The ranking statistic is PyCBC's, imported and called, never reimplemented (registration §3):
  v      pycbc.psd.variation.calc_filt_psd_variation(strain, 8, 0.25, 512, 8, 4, 'median', 50, 480) on the GATED strain,
         read at a trigger with find_trigger_value
  rho^   pycbc.events.ranking.newsnr(rho * v^-1/2, chi2_r / v) with v < 0.65 -> 1 (the newsnr_sgveto_psdvar rule without
         its sine-Gaussian term)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

B = Path(__file__).resolve().parent
sys.path.insert(0, str(B)); sys.path.insert(0, str(B.parent)); sys.path.insert(0, str(B.parent.parent / "tools"))
from pbh import config as C  # noqa: E402

FS = C.SAMPLE_RATE
OUT = B / "results" / "coh2t"
V_ARGS = dict(segment=8, short_segment=0.25, psd_long_segment=512, psd_duration=8, psd_stride=4,
              psd_avg_method="median", low_freq=50.0, high_freq=480.0)
MIN_EXPECTED_PSDVAR = 0.65


def write(name, obj):
    import envstamp
    from footprint import footprint_mb
    OUT.mkdir(parents=True, exist_ok=True)
    obj = dict(obj, model="Claude Opus 5.5", peak_footprint_mb=round(footprint_mb()[1]), env=envstamp.stamp(str(B.parent.parent)))
    (OUT / name).write_text(json.dumps(obj, indent=1))
    print(f"wrote results/coh2t/{name} | peak footprint {obj['peak_footprint_mb']} MB", flush=True)


def psd_variation(det, gps):
    """PyCBC's v time series (1-s steps) for one gated tuning series."""
    import realbg_cascade as R
    from pycbc.psd.variation import calc_filt_psd_variation
    from pycbc.types import TimeSeries
    x = np.load(B / "data" / f"{det}_{gps}.npy").astype(np.float64)
    x *= R.gate_window(det, gps)[0]
    return calc_filt_psd_variation(TimeSeries(x, delta_t=1.0 / FS, epoch=gps), **V_ARGS)


def v_at(psd_var, gps, t_gps):
    """PyCBC's find_trigger_value: idx = trigger sample index from the series start, start = series GPS start."""
    from pycbc.psd.variation import find_trigger_value
    idx = np.array([(float(t_gps) - float(gps)) * FS])
    return float(np.atleast_1d(find_trigger_value(psd_var, idx, float(gps), FS))[0])


def rho_hat(rho, chi2_r, v):
    """newsnr_sgveto_psdvar without the sine-Gaussian factor: PyCBC's own newsnr on the v-scaled inputs."""
    from pycbc.events.ranking import newsnr
    v = np.where(np.asarray(v, float) < MIN_EXPECTED_PSDVAR, 1.0, np.asarray(v, float))
    return newsnr(np.asarray(rho, float) * v ** -0.5, np.asarray(chi2_r, float) / v)


def s0(det="H1", gps=None):
    """Gate S0 (registration §3): v finite with median in [0.9, 1.1]; rho^ == rho whenever chi2_r' <= 1; rho^ < rho
    when chi2_r' > 1; v < 0.65 treated as 1; and the result matches pycbc.events.ranking.newsnr_sgveto_psdvar with a
    neutral sine-Gaussian chi^2 (sgchisq = 1) wherever that function applies."""
    from pycbc.events.ranking import newsnr_sgveto_psdvar
    segs = json.loads((B / "manifest.json").read_text())["test"]
    gps = int(gps or segs[0])
    t0 = time.time()
    pv = psd_variation(det, gps)
    vals = np.asarray(pv.numpy(), float)
    finite = np.isfinite(vals)
    med = float(np.median(vals[finite]))
    tq = [gps + 600.0, gps + 2048.0, gps + 3500.0]
    v_tq = [v_at(pv, gps, t) for t in tq]
    rng = np.random.default_rng(0)
    rho = rng.uniform(5, 40, 2000)
    chi = rng.uniform(0.2, 6.0, 2000)
    v = rng.uniform(0.3, 3.0, 2000)
    ours = rho_hat(rho, chi, v)
    ref = newsnr_sgveto_psdvar(rho, chi, np.ones_like(rho), v)
    v_eff = np.where(v < MIN_EXPECTED_PSDVAR, 1.0, v)
    unweighted = chi / v_eff <= 1
    checks = dict(
        v_finite_fraction=float(finite.mean()), v_median=med, v_p01=float(np.quantile(vals[finite], 0.01)),
        v_p99=float(np.quantile(vals[finite], 0.99)), v_samples=int(len(vals)), v_at_three_times=v_tq,
        rho_hat_equals_scaled_rho_when_unweighted=bool(np.allclose(ours[unweighted], (rho * v_eff ** -0.5)[unweighted])),
        rho_hat_below_scaled_rho_when_weighted=bool(np.all(ours[~unweighted] < (rho * v_eff ** -0.5)[~unweighted])),
        low_v_treated_as_one=bool(np.allclose(rho_hat(rho[:5], chi[:5], np.full(5, 0.5)), rho_hat(rho[:5], chi[:5], np.ones(5)))),
        max_rel_diff_vs_pycbc_newsnr_sgveto_psdvar=float(np.max(np.abs(ours / ref - 1))))
    ok = (checks["v_finite_fraction"] > 0.99 and 0.9 <= med <= 1.1 and checks["rho_hat_equals_scaled_rho_when_unweighted"]
          and checks["rho_hat_below_scaled_rho_when_weighted"] and checks["low_v_treated_as_one"]
          and checks["max_rel_diff_vs_pycbc_newsnr_sgveto_psdvar"] < 1e-12)
    print(json.dumps(checks), flush=True)
    write(f"S0_{det}_{gps}.json", dict(gate="S0", det=det, segment=gps, v_args=V_ARGS, min_expected_psdvar=MIN_EXPECTED_PSDVAR,
                                       checks=checks, passed=bool(ok), seconds=round(time.time() - t0, 1)))
    print(f"S0 {'PASS' if ok else 'FAIL'}", flush=True)


if __name__ == "__main__":
    {"s0": s0}[sys.argv[1]](*sys.argv[2:])
