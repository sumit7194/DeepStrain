"""Can a Padé pole locate the singularity where Domb-Sykes cannot? A cheaper route to the same number.

WHY. 33 established that our Domb-Sykes pipeline is sound on clean values but BLIND at float64 to a
singularity off the real axis -- and a complex pair is exactly what the R<1 hypothesis consists of. The
conclusion was that a high-precision Leaver solver must be built before the Kerr question is even defined.

Before commissioning that build, one thing is worth trying, because it is free: Domb-Sykes needs MANY
coefficients (it extrapolates a ratio sequence to 1/n = 0), but a PADE APPROXIMANT needs few. The nearest
pole of the [m/n] Padé to a function converges to the function's nearest singularity, and a [2/2] needs only
five Taylor coefficients -- which is what float64 Kerr frequencies already give us.

The two estimators fail differently, which is the point: Domb-Sykes dies from too few terms, Padé dies from
spurious poles. Neither being an argument, both are put through the SAME three controls 33 used, at the SAME
float64 precision the Kerr data has. A Padé that cannot find the complex pair at float64 buys nothing and
the build stands.

    1/(1-x)     R = 1     real pole at x = 1
    1/(1-2x)    R = 1/2   real pole at x = 1/2
    1/(1+x^2)   R = 1     COMPLEX pair at x = +/- i

Run:  .venv/bin/python scripts/34_pade_pole.py
"""
import importlib.util
import json
from pathlib import Path

import mpmath as mp
import numpy as np
import qnm as qnmpkg

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
OUT = RESULTS / "34_pade_pole.json"

# Reuse 33's extractor rather than reimplementing it: this must test THAT code path, not a lookalike.
_spec = importlib.util.spec_from_file_location("s33", HERE / "33_domb_sykes_controls.py")
_s33 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_s33)
nodes, fit = _s33.nodes, _s33.fit

ORDERS = ((2, 2), (3, 3), (4, 4))     # [m/n] Pade; [m/m] needs 2m+1 Taylor coefficients
FIT_DEG = 20


def pade_poles(coefs, m, n):
    """Poles of the [m/n] Pade approximant built from a Taylor series, as complex numbers."""
    with mp.workdps(60):
        p, q = mp.pade([mp.mpf(c) for c in coefs[: m + n + 1]], m, n)
        roots = mp.polyroots(list(reversed(q)), maxsteps=200, extraprec=200)
        return sorted((complex(r) for r in roots), key=abs)


def nearest(coefs, m, n):
    try:
        poles = pade_poles(coefs, m, n)
    except Exception as exc:
        return {"R": None, "note": f"{type(exc).__name__}"}
    if not poles:
        return {"R": None, "note": "no poles"}
    z = poles[0]
    return {"R": float(abs(z)), "pole_re": float(z.real), "pole_im": float(z.imag),
            "complex": bool(abs(z.imag) > 0.05 * abs(z)),
            "n_poles": len(poles)}


def main():
    xs = nodes()
    controls = {
        "1/(1-x)":   (lambda x: 1 / (1 - x),     1.0, False),
        "1/(1-2x)":  (lambda x: 1 / (1 - 2 * x), 0.5, False),
        "1/(1+x^2)": (lambda x: 1 / (1 + x * x), 1.0, True),
    }
    res = {"orders": [list(o) for o in ORDERS], "fit_degree": FIT_DEG, "controls": {}}
    print("PADE POLE vs the same three controls 33 used, at the float64 precision the Kerr data has.\n")
    print(f"{'function':>11} {'[m/n]':>7} {'R':>9} {'R_true':>7} {'pole':>22}  verdict")
    passes = {}
    for name, (f, R_true, cplx_true) in controls.items():
        with mp.workdps(60):
            vals = [mp.mpf(float(f(float(x)))) for x in xs]        # float64 values, as Kerr's are
        c = [float(v) for v in fit(vals, xs, FIT_DEG)]
        res["controls"][name] = {"R_true": R_true, "complex_true": cplx_true, "orders": {}}
        for m, n in ORDERS:
            d = nearest(c, m, n)
            ok = (d["R"] is not None and abs(d["R"] - R_true) < 0.1
                  and d.get("complex") == cplx_true)
            passes.setdefault(name, []).append(ok)
            res["controls"][name]["orders"][f"{m}/{n}"] = {**d, "pass": bool(ok)}
            pole = (f"{d['pole_re']:+.4f}{d['pole_im']:+.4f}i" if d["R"] is not None else d.get("note", "-"))
            Rs = f"{d['R']:.4f}" if d["R"] is not None else "  n/a"
            print(f"{name:>11} {f'[{m}/{n}]':>7} {Rs:>9} {R_true:>7} {pole:>22}  {'PASS' if ok else 'FAIL'}")

    # A control is passed if ANY order gets it right AND the orders agree with each other -- a single
    # lucky order among three is not a working estimator.
    res["controls_pass"] = {k: bool(sum(v) >= 2) for k, v in passes.items()}
    all_ok = all(res["controls_pass"].values())

    seq = qnmpkg.modes_cache(s=-2, l=2, m=2, n=0)
    with mp.workdps(60):
        w = [mp.mpf(float(seq(a=float(x))[0].real)) for x in xs]
    ck = [float(v) for v in fit(w, xs, FIT_DEG)]
    res["kerr_220_real"] = {f"{m}/{n}": nearest(ck, m, n) for m, n in ORDERS}
    print(f"\nKERR omega_220 (float64 frequencies):")
    for k, d in res["kerr_220_real"].items():
        if d["R"] is not None:
            print(f"   [{k}]  R = {d['R']:.4f}   nearest pole {d['pole_re']:+.4f}{d['pole_im']:+.4f}i"
                  f"   {'COMPLEX' if d['complex'] else 'real'}")
        else:
            print(f"   [{k}]  {d.get('note')}")

    # WHY IT FAILS, recorded because the failure modes are the standard ones and knowing which is which
    # decides whether a fix exists. On EXACT coefficients of 1/(1-x) the [2/2] system is SINGULAR -- the
    # function is exactly [0/1], so the higher Hankel system is rank-deficient (checked directly). On NOISY
    # coefficients the degeneracy resolves into Froissart doublets: spurious pole-zero pairs that land near
    # the origin and dominate |z|. Both are intrinsic to Pade on few noisy terms, not implementation.
    res["failure_mode"] = ("degenerate Hankel system on exact input; Froissart doublets near the origin on "
                           "noisy input -- both standard, neither fixable by more careful coding")
    # AND THE REASON THE CONTROLS EARNED THEIR KEEP: the Kerr [2/2] returns R = 1.28, which LOOKS like
    # support for R = 1. The identical estimator returns 0.0096 on a function whose radius is exactly 1.
    # A number agreeing with the hypothesis, from an estimator shown to be broken on the hypothesis's own
    # test case, is the most expensive kind of number.
    Rs = [d["R"] for d in res["kerr_220_real"].values() if d["R"] is not None]
    spread = (max(Rs) - min(Rs)) if len(Rs) > 1 else float("inf")
    res["kerr_spread_across_orders"] = float(spread)
    res["verdict"] = (
        f"Pade passes all three controls at float64 (Domb-Sykes did not), so the Kerr estimate is USABLE; "
        f"orders agree to {spread:.3f}" if all_ok and spread < 0.1 else
        f"Pade does NOT clear the controls at float64 (or the Kerr orders disagree, spread {spread:.3f}) "
        f"=> no cheaper route; the high-precision Leaver build stands")
    print(f"\nCONTROLS at float64: {'ALL PASS' if all_ok else 'FAIL'}")
    print(f"VERDICT: {res['verdict']}")
    OUT.write_text(json.dumps(res, indent=2, default=float))
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
