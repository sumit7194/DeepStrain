"""How wrong is an O(chi^2) black hole? Measuring the spin-truncation error against exact Kerr.

WHY. A theory programme proposed working to SECOND ORDER IN SPIN. The question is whether that covers real
black holes. It is usually argued rather than measured, and the two arguments people reach for are both
wrong:

  "the first neglected term is chi^3, and chi^3/chi^2 = chi = 0.69, so the error is ~70%"
      -- a TERM RATIO IS NOT AN ERROR. It assumes the Taylor coefficients are O(1). They are not; they
         decay, and the measured O(chi^2) error at chi=0.69 is ~4%, not 70%. (This was our claim, and it
         was wrong by more than an order of magnitude.)

  "and therefore no finite order in chi is controlled, since the ratio is chi at every order"
      -- SAME UNCHECKED PREMISE, amplified. If the coefficients decay the series converges, and going from
         2nd to 6th order buys ~16x at remnant spins. (This was bridge's correction of our claim, and it
         inherited the assumption it was correcting -- the failure mode now filed as PROTOCOL section 24.)

So this measures it instead, on exact Kerr 220 frequencies from the `qnm` package.

WHAT IS MEASURED
  (1) truncation error of an order-n spin expansion, evaluated at chi = 0.69 (the universal remnant spin of
      equal-mass non-spinning mergers, reproducible to ~1% across NR codes) and chi = 0.90 (typical of the
      high-spin supermassive objects that EMRI programmes target);
  (2) its STABILITY under fit range, because an extrapolation error that moves with the fit is not a
      truncation error;
  (3) which Taylor coefficients are real at all, via a fit-DEGREE sweep -- the decisive test, and the one
      that overturned our own reported numbers.

THREE METHODS TRIED, AND THE FAILURE PATTERN IS THE TRANSFERABLE RESULT
  monomial polyfit        -- |a_5| wanders an order of magnitude while |a_6| climbs with fit range
  high-order finite diff  -- ill-conditioned; returned a_6 = 807312 and errors of 1e5. Discarded.
  Chebyshev fit           -- well conditioned at low order, and DEGREE-DEPENDENT above n=4

All three fail at the same orders. That is a statement about the DATA, not the tools: the coefficients are
simply not recoverable beyond n~3 from numerically-computed frequencies at this precision.

  A blow-up announces itself. A PLATEAU RECRUITS YOU -- the Chebyshev ratios looked like a converging
  sequence (0.669, 0.731, 0.775, 0.801, 0.812) and two of those five values do not exist: at fit degree 24
  the n=5 ratio is 0.671, at degree 28 it is 0.150, and n=6 spans 0.735 to 125.

  GENERAL FORM, which is worth more than the physics here: extracting SERIES COEFFICIENTS from numerics is
  inference with an unquoted error bar; EVALUATING a closed form at points is verification. Only the second
  is safe. If a programme plans to "derive a closed form and check it against numerics", that distinction
  decides whether the check means anything.

Run:  .venv/bin/python scripts/31_spin_truncation.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import qnm as qnmpkg
from numpy.polynomial import chebyshev as Ch
from numpy.polynomial import polynomial as Pl

RESULTS = Path(__file__).resolve().parent.parent / "results"
OUT = RESULTS / "31_spin_truncation.json"
CHI_REMNANT = 0.69       # universal final spin, equal-mass non-spinning (astro-ph/0609172, 1305.5991)
CHI_EMRI = 0.90          # representative high-spin supermassive central object
FIT_RANGES = (0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
DEGREES = (12, 16, 20, 24, 28)


def main() -> None:
    seq = qnmpkg.modes_cache(s=-2, l=2, m=2, n=0)

    def w(c):
        return seq(a=c)[0].real

    res = {"mode": "220 (s=-2, l=2, m=2, n=0), real part",
           "chi_remnant": CHI_REMNANT, "chi_emri": CHI_EMRI,
           "exact": {f"{c:.2f}": w(c) for c in (0.0, 0.5, CHI_REMNANT, CHI_EMRI, 0.95)}}
    print("exact Kerr 220 Re(omega M):")
    for k, v in res["exact"].items():
        print(f"   chi={k}: {v:.5f}")

    # ---- (1)+(2) truncation error and its stability under fit range ----------------------------------
    print(f"\nO(chi^n) truncation error, and its spread over fit ranges {FIT_RANGES}")
    print(f"{'order':>6} {'err@0.69':>22} {'err@0.90':>22}")
    res["truncation"] = {}
    for order in (2, 3, 4, 5, 6):
        e69, e90 = [], []
        for hi in FIT_RANGES:
            ch = np.linspace(0.0, hi, 80)
            co = np.polyfit(ch, np.array([w(c) for c in ch]), order)
            e69.append(abs(np.polyval(co, CHI_REMNANT) - w(CHI_REMNANT)) / w(CHI_REMNANT))
            e90.append(abs(np.polyval(co, CHI_EMRI) - w(CHI_EMRI)) / w(CHI_EMRI))
        res["truncation"][str(order)] = {
            "chi069": {"median": float(np.median(e69)), "min": float(min(e69)), "max": float(max(e69))},
            "chi090": {"median": float(np.median(e90)), "min": float(min(e90)), "max": float(max(e90))}}
        print(f"{order:>6}  {np.median(e69):8.3%} [{min(e69):.3%}, {max(e69):.3%}]"
              f"  {np.median(e90):8.3%} [{min(e90):.3%}, {max(e90):.3%}]")

    # successive error ratios -- only meaningful at orders whose coefficients are real (see below)
    res["error_ratio"] = {}
    print("\nsuccessive error ratio (order n -> n+1), fit range 0-0.30")
    ch = np.linspace(0.0, 0.30, 80); y = np.array([w(c) for c in ch])
    prev = None
    for order in (2, 3, 4, 5):
        co = np.polyfit(ch, y, order)
        e = {t: abs(np.polyval(co, t) - w(t)) / w(t) for t in (CHI_REMNANT, CHI_EMRI)}
        if prev:
            r69, r90 = e[CHI_REMNANT] / prev[CHI_REMNANT], e[CHI_EMRI] / prev[CHI_EMRI]
            res["error_ratio"][f"{order-1}->{order}"] = {"chi069": float(r69), "chi090": float(r90)}
            print(f"   {order-1}->{order}:  chi=0.69 {r69:.3f}   chi=0.90 {r90:.3f}")
        prev = e

    # ---- (3) WHICH COEFFICIENTS ARE REAL? the decisive test --------------------------------------------
    print("\ncoefficient ratio |a_(n+1)/a_n| vs Chebyshev fit degree -- a real coefficient does not move")
    lo, hi = 0.0, 0.5
    x = Ch.chebpts2(600); cc = lo + (x + 1) * (hi - lo) / 2
    y = np.array([w(v) for v in cc]); s = 2.0 / (hi - lo)
    tab = {}
    print(f"{'deg':>5}" + "".join(f"{'n=' + str(n):>10}" for n in range(2, 8)))
    for deg in DEGREES:
        P = Pl.Polynomial(Ch.cheb2poly(Ch.chebfit(x, y, deg)))
        a = P(Pl.Polynomial([-1 - s * lo, s])).coef
        r = [abs(a[n + 1] / a[n]) if n + 1 < len(a) else float("nan") for n in range(2, 8)]
        tab[deg] = r
        print(f"{deg:>5}" + "".join(f"{v:>10.3f}" for v in r))
    res["coefficient_stability"] = {}
    print("\n   drift across fit degree (max-min):")
    n_real = 0
    for i, n in enumerate(range(2, 8)):
        vals = [tab[d][i] for d in DEGREES if np.isfinite(tab[d][i])]
        drift = float(max(vals) - min(vals))
        real = drift < 0.02
        n_real += real
        res["coefficient_stability"][str(n)] = {"min": float(min(vals)), "max": float(max(vals)),
                                                "drift": drift, "is_real": bool(real)}
        tag = "REAL" if real else ("drifting" if drift < 0.15 else "FIT-DOMINATED")
        print(f"     n={n}: {min(vals):8.3f} - {max(vals):8.3f}   drift {drift:8.3f}   {tag}")
    res["n_recoverable_coefficients"] = int(n_real) + 2   # a_0, a_1 trivially, plus the stable ratios

    t2 = res["truncation"]["2"]
    res["verdict"] = {
        "ringdown": (f"O(chi^2) error {t2['chi069']['median']:.1%} at chi=0.69 is BELOW our sigma(delta) "
                     f"~ 0.14, so the spin truncation is NOT the binding constraint for ringdown"),
        "emri": (f"O(chi^2) error {t2['chi090']['median']:.1%} at chi=0.90, against phase accuracy over "
                 f"~1e5 cycles, is disqualifying -- the remedy there is non-perturbative in spin"),
        "limit_not_established": ("the asymptotic error ratio is NOT measurable: only the n=2 and n=3 "
                                  "coefficient ratios are stable under fit degree, so no statement about "
                                  "whether the ratio tends to chi or below it is supported")}
    print(f"\nRINGDOWN: {res['verdict']['ringdown']}")
    print(f"EMRI:     {res['verdict']['emri']}")
    print(f"NOT SHOWN: {res['verdict']['limit_not_established']}")
    OUT.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {OUT.name}")


if __name__ == "__main__":
    main()
