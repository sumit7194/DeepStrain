"""What does our BROKEN Domb-Sykes estimator return when it fails? The null distribution of our own instrument.

WHY (bridge, 2026-09-22). We found the estimator returns R = 1.1290 on 1/((1-2x)(1+x)), whose true R is 0.5,
and separately returned R = 1.13 on Kerr. We recorded that as "the Kerr value is indistinguishable from the
failure mode". That is correct and WEAKER THAN WHAT THE DATA SUPPORTS: two numbers agreeing to three digits
on inputs with very different true R suggests 1.13 may be what the estimator returns WHEN IT FAILS, more or
less regardless of input -- an attractor of the failure rather than a property of Kerr.

    cluster near ~1 across widely different true R  ->  1.13 is the estimator's fixed point under failure,
                                                        and the Kerr number carries NO INFORMATION AT ALL:
                                                        P0 is not "unresolved", it is "never measured"
    scatter, tracking true R                        ->  the agreement was coincidence; the Kerr value is
                                                        contaminated but not necessarily vacuous

We had characterised THAT the estimator fails. We had not characterised WHAT IT RETURNS when it does -- and a
failure mode with an unmeasured output distribution cannot tell you how much of your result it explains.

Exact arithmetic throughout, no Leaver needed: this is the existing control harness run twenty times instead
of four.

PRE-REGISTERED BEFORE RUNNING:
  statistic   Spearman correlation between returned R and true R, over the whole family
  fixed point |rho| < 0.3 AND median(returned) in [0.85, 1.35] AND IQR(returned) < 0.50
  tracking    rho > 0.7
  otherwise   partial -- reported as such, not rounded to either

Run:  .venv/bin/python scripts/35_estimator_null.py
"""
import importlib.util
import json
from pathlib import Path

import mpmath as mp
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
OUT = RESULTS / "35_estimator_null.json"

# Import the estimator UNFIXED, exactly as it stands -- this must measure THAT code, not a reimplementation.
_spec = importlib.util.spec_from_file_location("s33", HERE / "33_domb_sykes_controls.py")
_s33 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_s33)
nodes, stable_ratios, domb_sykes = _s33.nodes, _s33.stable_ratios, _s33.domb_sykes


def family():
    """Functions with known R, deliberately spread, and several singularity structures."""
    out = []
    for R in (0.2, 0.35, 0.5, 0.7, 1.0, 1.5, 2.0):                 # single real pole -- the easy case
        out.append((f"1/(1-x/{R})", lambda x, R=R: 1 / (1 - x / R), R, "single pole"))
    for R, k in ((0.25, 2), (0.4, 3), (0.5, 2), (0.8, 4), (1.0, 2), (1.5, 3)):   # competing second pole
        out.append((f"1/((1-x/{R})(1+x/{k*R}))",
                    lambda x, R=R, k=k: 1 / ((1 - x / R) * (1 + x / (k * R))), R, "two real poles"))
    for R in (0.3, 0.6, 1.0, 1.4):                                  # complex conjugate pair
        out.append((f"1/(1+(x/{R})^2)", lambda x, R=R: 1 / (1 + (x / R) ** 2), R, "complex pair"))
    for R in (0.4, 0.9, 1.3):                                       # pole plus a branch point further out
        out.append((f"sqrt(1-x/{2*R})/(1-x/{R})",
                    lambda x, R=R: mp.sqrt(1 - x / (2 * R)) / (1 - x / R), R, "pole + branch point"))
    return out


def main() -> None:
    xs = nodes()
    rows = []
    print(f"{'function':>26} {'class':>20} {'true R':>8} {'returned':>10} {'n_stable':>9}")
    for name, f, R_true, kind in family():
        with mp.workdps(60):
            vals = [f(mp.mpf(x)) for x in xs]
        try:
            rs, step = stable_ratios(vals, xs)
            d = domb_sykes(rs, step)
            got = d["R"]
        except Exception as exc:
            got, rs = None, []
            d = {"note": type(exc).__name__}
        rows.append({"function": name, "class": kind, "R_true": R_true, "R_returned": got,
                     "n_stable": len(rs)})
        shown = f"{got:10.4f}" if got is not None else "       n/a"
        print(f"{name:>26} {kind:>20} {R_true:8.2f} {shown} {len(rs):9}")

    got = [(r["R_true"], r["R_returned"]) for r in rows if r["R_returned"] is not None]
    t = np.array([a for a, _ in got]); g = np.array([b for _, b in got])
    rt, rg = np.argsort(np.argsort(t)), np.argsort(np.argsort(g))
    rho = float(np.corrcoef(rt, rg)[0, 1])
    med, iqr = float(np.median(g)), float(np.percentile(g, 75) - np.percentile(g, 25))

    fixed = abs(rho) < 0.3 and 0.85 <= med <= 1.35 and iqr < 0.50
    tracking = rho > 0.7
    verdict = ("FIXED POINT -- the estimator returns ~1 regardless of the true radius, so the Kerr value "
               "carries NO information and P0 is 'never measured', not 'unresolved'" if fixed else
               "TRACKING -- returned R follows true R, so the 1.13/1.13 agreement was coincidence and the "
               "Kerr value is contaminated but not vacuous" if tracking else
               "PARTIAL -- neither a clean fixed point nor clean tracking; reported as such")

    res = {"n_functions": len(rows), "rows": rows, "spearman_rho": rho,
           "median_returned": med, "iqr_returned": iqr,
           "kerr_provisional": 1.13, "broken_control_returned": 1.1290, "verdict": verdict,
           "prereg": {"fixed_point": "|rho|<0.3 and median in [0.85,1.35] and IQR<0.50", "tracking": "rho>0.7"}}
    print(f"\n  n = {len(got)} functions with a returned value, true R spanning "
          f"{t.min():.2f}-{t.max():.2f}")
    print(f"  Spearman rho(returned, true) = {rho:+.3f}")
    print(f"  returned: median {med:.4f}, IQR {iqr:.4f}  "
          f"(broken control 1.1290, Kerr provisional 1.13)")
    print(f"\nVERDICT: {verdict}")
    OUT.write_text(json.dumps(res, indent=2))
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
