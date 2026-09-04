"""Before asking anyone to build a 50-digit Leaver solver: can our pipeline detect R<1 at all, and is the
wall the FREQUENCIES' precision or our own EXTRACTOR's conditioning?

WHY (ansatz's catch, 2026-09-05). Our extractor's only golden test was 1/(1-x), whose radius of convergence
is exactly 1 -- which is hypothesis H, one of the two verdicts the Kerr study exists to distinguish. A control
validated on the H side alone cannot see a bias toward H: "agrees with truth" and "pulled toward 1" are
indistinguishable on that function. So three controls, one per verdict band:

    1/(1-x)     R = 1      real branch point on the positive axis   (H)
    1/(1-2x)    R = 1/2    real branch point                        (not-H: R < 1)
    1/(1+x^2)   R = 1      COMPLEX pair at +/- i                    (the shape not-H would take)

The third matters most. not-H is a singularity off the real axis; if Domb-Sykes cannot report the
alternating-ratio signature of a complex pair, the study returns H by construction.

AND THE SECOND QUESTION, which decides who builds what. We reported that only three Taylor coefficients are
recoverable from double-precision Kerr frequencies. Two different walls give that answer: the frequencies'
own ~1e-16 noise (needs a high-precision Leaver -- a build nobody in the fleet has) or our extractor's
Chebyshev/monomial conversion in double precision (fixable for free by doing the linear algebra in mpmath on
the SAME values). Each control is therefore run twice: on high-precision values, which tests the extractor
alone, and on double-precision values, which adds exactly the noise Kerr has.

Run:  .venv/bin/python scripts/33_domb_sykes_controls.py
"""
import json
from pathlib import Path

import mpmath as mp
import numpy as np
import qnm as qnmpkg

RESULTS = Path(__file__).resolve().parent.parent / "results"
OUT = RESULTS / "33_domb_sykes_controls.json"
HI, NPTS = 0.25, 120   # inside R=1/2, so the not-H control's own pole is not sampled
DEGREES = (16, 20, 24, 28)
STABLE_TOL = 1e-3          # relative drift of a RATIO across fit degrees, below which it counts as recovered


def nodes():
    k = np.arange(NPTS)
    return (np.cos(np.pi * (2 * k + 1) / (2 * NPTS)) + 1) * HI / 2


def fit(vals, xs, deg):
    """Monomial least squares by QR at precision scaled to the degree.

    NOT the normal equations, which square the condition number -- mpmath called X^T X singular at 16 digits
    on the first control, one line into the run. And not double precision either: the monomial Vandermonde on
    [0, 0.5] is already singular at degree 24 in float64, which is the conditioning wall this script measures.
    """
    with mp.workdps(max(40, 6 * deg)):
        X = mp.matrix([[mp.mpf(x) ** j for j in range(deg + 1)] for x in xs])
        y = mp.matrix([mp.mpf(v) for v in vals])
        return [mp.mpf(v) for v in mp.qr_solve(X, y)[0]]


def series_step(coefs):
    """1 for an ordinary series, 2 for one in x^2.

    THIS IS THE FIX ansatz's third control forced, and the failure was worse than the bias they predicted.
    For 1/(1+x^2) every odd coefficient vanishes, so a_1/a_0 = 0 and a_2/a_1 is a division by zero: the
    ratio test returns NOTHING. Run without this branch, the complex-pair control yields zero stable ratios
    and reports "too few coefficients" -- indistinguishable from a precision limit, and the study would have
    concluded H by construction while looking like it had simply run out of digits. A complex-conjugate pair
    is exactly the singularity structure not-H consists of.
    """
    a = [abs(float(c)) for c in coefs[:5]]
    big = max(a) or 1.0
    # Only the FIRST two odd coefficients. Higher ones degrade smoothly with fit degree (at degree 16 the
    # seventh sits at 1.8e-6 while the first is 1.1e-17), so a blanket threshold over eight coefficients
    # rejects a genuinely even series -- which is what it did, and the symptom was zero stable ratios and a
    # "too few coefficients" message that reads exactly like a precision limit.
    return 2 if (a[1] < 1e-8 * big and a[3] < 1e-8 * big) else 1


def stable_ratios(vals, xs):
    """Successive coefficient ratios that do not move across fit degree, in the series' own step.

    A ratio that moves is the fit talking, not the series -- 31's lesson applied as a precondition.
    """
    fits = {d: fit(vals, xs, d) for d in DEGREES}
    step = series_step(next(iter(fits.values())))
    out = []
    # STRIDE by `step`, not by 1: with step 2 the intermediate index holds a vanishing coefficient, so
    # a_3/a_1 is a ratio of two numerical zeros. Iterating by 1 hit that on the second term and broke out
    # of the loop with a single ratio -- again looking like "not enough coefficients" rather than a bug.
    for n in range(step, min(DEGREES), step):
        vals_n = [f[n] / f[n - step] for f in fits.values() if f[n - step] != 0]
        if len(vals_n) < len(fits):
            break
        rs = [float(v) for v in vals_n]
        scale = max(abs(r) for r in rs) or 1.0
        if (max(rs) - min(rs)) / scale > STABLE_TOL:
            break
        out.append(float(np.median(rs)))
    return out, step


def domb_sykes(ratios, step, k=3):
    """r_n vs 1/n, extrapolated to 1/n = 0: the intercept is 1/R in the series' own variable.

    With step 2 the fit is in u = x^2, so R_x = sqrt(R_u). k=3 is THIN and is reported as such rather than
    padded -- three points fix a line with one degree of freedom left over, which is the least that can
    distinguish a trend from two points.
    """
    if len(ratios) < k:
        return {"R": None, "note": f"only {len(ratios)} stable ratios; need >= {k}"}
    r = np.array(ratios)
    alt = bool(np.all(r[-k:] < 0))
    n = np.arange(1, len(r) + 1) * step
    slope, icpt = np.polyfit(1.0 / n[-k:], np.abs(r[-k:]), 1)
    R_u = 1.0 / icpt if icpt > 0 else None
    R = (R_u ** (1.0 / step)) if R_u else None
    return {"R": float(R) if R else None, "intercept": float(icpt), "slope": float(slope),
            "step": step, "alternating": alt, "n_ratios_used": k,
            "singularity": ("complex pair off the real axis" if (step == 2 and alt) else
                            "negative real axis" if alt else "positive real axis")}


def main():
    xs = nodes()
    controls = {
        "1/(1-x)":   (lambda x: 1 / (1 - x),      1.0, "positive real axis"),
        "1/(1-2x)":  (lambda x: 1 / (1 - 2 * x),  0.5, "positive real axis"),
        "1/(1+x^2)": (lambda x: 1 / (1 + x * x),  1.0, "complex pair off the real axis"),
    }
    res = {"interval": [0, HI], "n_points": NPTS, "degrees": list(DEGREES),
           "stable_tol": STABLE_TOL, "controls": {}}

    print("CONTROLS -- 'exact' values are computed at 60 digits (tests the EXTRACTOR alone);")
    print("            'float64' values carry the same 1e-16 noise the Kerr frequencies have.\n")
    print(f"{'function':>11} {'values':>8} {'n_stable':>9} {'R':>9} {'R_true':>7}  singularity")
    verdicts = {}
    for name, (f, R_true, kind_true) in controls.items():
        res["controls"][name] = {"R_true": R_true, "singularity_true": kind_true}
        for tag in ("exact", "float64"):
            with mp.workdps(60):
                vals = ([f(mp.mpf(x)) for x in xs] if tag == "exact" else
                        [mp.mpf(float(f(float(x)))) for x in xs])
            rs, step = stable_ratios(vals, xs)
            ds = domb_sykes(rs, step)
            ok = (ds["R"] is not None and abs(ds["R"] - R_true) < 0.15
                  and ds.get("singularity") == kind_true)
            res["controls"][name][tag] = {"n_stable": len(rs), "ratios": rs, **ds, "pass": bool(ok)}
            verdicts[(name, tag)] = ok
            Rs = f"{ds['R']:.4f}" if ds["R"] else "  n/a"
            print(f"{name:>11} {tag:>8} {len(rs):>9} {Rs:>9} {R_true:>7}  "
                  f"{ds.get('singularity','-')}  {'PASS' if ok else 'FAIL'}")

    # --- Kerr, on the double-precision frequencies we already have -------------------------------------
    seq = qnmpkg.modes_cache(s=-2, l=2, m=2, n=0)
    with mp.workdps(60):
        w = [mp.mpf(float(seq(a=float(x))[0].real)) for x in xs]
    rs, step = stable_ratios(w, xs)
    ds = domb_sykes(rs, step)
    res["kerr_220_real"] = {"n_stable": len(rs), "ratios": rs, **ds}
    print(f"\nKERR omega_220 (double-precision frequencies, mpmath extraction):")
    print(f"  stable ratios: {len(rs)}  -> {[round(r, 6) for r in rs]}")
    print(f"  Domb-Sykes:    R = {ds['R']:.4f}" if ds["R"] else "  Domb-Sykes: not enough ratios")

    # --- which wall? -----------------------------------------------------------------------------------
    n_exact = res["controls"]["1/(1-x)"]["exact"]["n_stable"]
    n_f64 = res["controls"]["1/(1-x)"]["float64"]["n_stable"]
    res["wall"] = {
        "extractor_capacity_on_clean_values": n_exact,
        "capacity_with_float64_noise": n_f64,
        "kerr_capacity": len(rs),
        "verdict": ("the FREQUENCIES' precision: the extractor recovers far more from clean values than from "
                    "float64 ones, so a high-precision Leaver solver is genuinely required"
                    if n_exact > n_f64 + 2 else
                    "NOT the frequencies alone: float64 values give nearly as many coefficients as exact "
                    "ones, so the earlier limit was substantially the extractor's conditioning")}
    # THE OPERATIVE TEST IS THE float64 ROW, not the clean one: float64 is the condition the Kerr
    # frequencies are actually in. Passing on clean values says the METHOD is sound; passing on float64
    # would say the method is usable ON OUR DATA. They are different claims and only the second licenses
    # a Kerr number.
    res["controls_all_pass_clean"] = all(verdicts[(n, "exact")] for n in controls)
    res["controls_all_pass_float64"] = all(verdicts[(n, "float64")] for n in controls)
    res["kerr_220_real"]["status"] = (
        "PROVISIONAL -- not a result. At float64 the complex-pair control FAILS, so at this precision the "
        "pipeline cannot detect a singularity off the real axis, which is exactly what the R<1 hypothesis "
        "consists of. A number produced here could only ever come out as R~1. High-precision frequencies "
        "are required before this is quotable."
        if not res["controls_all_pass_float64"] else "usable at this precision")
    print(f"\nWALL: {res['wall']['verdict']}")
    print(f"CONTROLS on clean values : {'ALL PASS' if res['controls_all_pass_clean'] else 'FAIL'}"
          f"   => the METHOD is sound")
    print(f"CONTROLS at float64      : {'ALL PASS' if res['controls_all_pass_float64'] else 'FAIL'}"
          f"   => the method on OUR DATA")
    print(f"\nKERR STATUS: {res['kerr_220_real']['status']}")
    OUT.write_text(json.dumps(res, indent=2, default=float))
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
