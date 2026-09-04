"""Does our spin-truncation measurement agree with a published, independent calibration? It did not, and that
found a method error in our own script.

WHY. 31_spin_truncation.py measured the O(chi^n) truncation error of the Kerr 220 with our own machinery and
nothing else. Every number in it -- including the one we used to WITHDRAW our own "disqualifying" claim --
rested on one implementation. A measurement that overturns a claim deserves an outside check more than one
that confirms it.

THE OUTSIDE NUMBER. Pierini & Gualtieri, PRD 106, 104009 (2022), arXiv:2207.11267, Sec. IV A -- the
second-order-in-spin EdGB quasinormal-mode paper, i.e. exactly the class of calculation a scalar-Gauss-Bonnet
programme proposes. Their Eq. (44) defines the discrepancy against EXACT Kerr, and they report:

    "at first order, the discrepancy of the Taylor expansion is smaller than 1% as long as a_bar <~ 0.22.
     Including the second order correction, the discrepancy is smaller than 1% for a_bar <~ 0.4."

Two numbers we did not use and did not tune to. In GR the slow-rotation expansion of the field equations is
the Taylor expansion of the exact Kerr frequency in chi, which is what makes the comparison legal -- and it
holds only in the Kerr limit, not for the sGB correction itself.

WHAT IT FOUND. Our first crossings were 0.35 and 0.50 against their 0.22 and 0.40 -- ours consistently LATER,
i.e. our errors smaller. The tempting reading was that the two measure different objects: theirs bundles the
slow-rotation method's own error (truncated angular system, direct integration) on top of the series
truncation, so theirs should be larger. That explanation is physically sensible, unfalsifiable from here, and
WRONG. It was our bug.

  31 obtained its coefficients with `np.polyfit` at the truncation order over [0, hi]. A degree-n
  least-squares fit is the best n-th degree approximation ON THAT INTERVAL -- it absorbs higher-order
  behaviour into its low coefficients -- so evaluating it outside the interval understates the truncation
  error of the actual series. Fitting HIGH in Chebyshev and truncating LOW returns the series' own
  coefficients, and then the published crossings come out at 0.219 and 0.400.

  The corrected O(chi^2) error at chi=0.69 is 6.36%, not the 4.43% first reported; at chi=0.90 it is 18.86%,
  not 16.15%. Both conclusions survive, with a smaller margin on the ringdown side. And the quoted
  "spread over fit ranges [3.44, 5.35]%" was never a systematic -- it was fit noise. The correct calculation
  has no spread at all: identical to six digits across every extraction range and degree.

THE GOLDEN TEST IS THE POINT, and it is tabula's precondition (relayed 2026-09-04): a sweep is evidence only
once the identical sweep has been shown to land correctly on a case whose answer is fixed in advance. For
1/(1-x) the order-n relative truncation error is exactly x^(n+1), so the 1% crossings are 0.1000 and 0.2154.
The low-degree fit returned 0.193 and 0.345 -- it fails a case it must pass. Had this script compared only
against the paper, the disagreement would have been argued about; the analytic control is what made it a bug.

Run:  .venv/bin/python scripts/32_spin_truncation_crosscheck.py
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import qnm as qnmpkg

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
OUT = RESULTS / "32_spin_truncation_crosscheck.json"

# Import 31's extractor rather than reimplementing it -- this must test THAT code, not a lookalike.
_spec = importlib.util.spec_from_file_location("s31", HERE / "31_spin_truncation.py")
_s31 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_s31)
taylor_coefficients = _s31.taylor_coefficients

PUBLISHED = {1: 0.22, 2: 0.40}        # quoted to two figures, so agreement is judged at that resolution
TOL = 0.02
GRID = np.linspace(0.005, 0.95, 946)


def crossing(err, level=0.01):
    """First spin at which the error exceeds `level`, linearly interpolated. NaN if it never does.

    First, not last: a `max`-based rule would report the end of the grid for a non-monotonic curve.
    """
    above = np.nonzero(err > level)[0]
    if above.size == 0 or above[0] == 0:
        return float("nan")
    i = above[0]
    return float(GRID[i - 1] + (level - err[i - 1]) * (GRID[i] - GRID[i - 1]) / (err[i] - err[i - 1]))


def crossings_of(f, orders=(1, 2)):
    exact = np.array([f(c) for c in GRID])
    co = taylor_coefficients(f)
    return {o: crossing(np.abs(np.polyval(co[:o + 1][::-1], GRID) - exact) / np.abs(exact)) for o in orders}


def golden(res):
    """1/(1-x): order-n relative truncation error is exactly x^(n+1). Fixed in advance, no fitting to it."""
    got = crossings_of(lambda x: 1.0 / (1.0 - x))
    out, ok = {}, True
    for o, x in got.items():
        want = 0.01 ** (1.0 / (o + 1))
        passed = abs(x - want) <= 0.005
        ok &= passed
        out[str(o)] = {"expected": want, "measured": x, "pass": bool(passed)}
        print(f"   order {o}: expected {want:.4f}  measured {x:.4f}  {'PASS' if passed else 'FAIL'}")
    res["golden_geometric"] = out
    if not ok:
        raise SystemExit("golden test FAILED -- the extractor is wrong, so no comparison is meaningful")


def main() -> None:
    res = {"published": {"source": "Pierini & Gualtieri, PRD 106, 104009 (2022), arXiv:2207.11267, Sec. IV A",
                         "definition": "Eq. (44): (omega_Taylor - omega_Kerr) / omega_Kerr",
                         "crossing_1pct": PUBLISHED, "tolerance": TOL}}

    print("golden test -- the extractor against an analytic truncation error")
    golden(res)

    # Their Fig. 1 carries four curves against one 1% line -- real and imaginary parts of (022) and (033) --
    # so the quoted crossing is the EARLIEST of the four, not whichever mode we happen to study.
    curves = {}
    for mname, (l, m) in (("022", (2, 2)), ("033", (3, 3))):
        seq = qnmpkg.modes_cache(s=-2, l=l, m=m, n=0)
        for pname, part in (("real", lambda z: z.real), ("imag", lambda z: abs(z.imag))):
            curves[f"{mname}_{pname}"] = crossings_of(lambda c, s=seq, p=part: p(s(a=c)[0]))

    res["measured"] = {k: {str(o): v for o, v in d.items()} for k, d in curves.items()}
    print("\nspin at which the truncation error first exceeds 1%")
    print(f"{'curve':>12} {'order 1':>9} {'order 2':>9}")
    for k, d in curves.items():
        print(f"{k:>12} {d[1]:9.3f} {d[2]:9.3f}")

    res["earliest"] = {str(o): {"crossing": min(d[o] for d in curves.values()),
                                "curve": min(curves, key=lambda k: curves[k][o])} for o in (1, 2)}
    agree = {o: bool(abs(res["earliest"][o]["crossing"] - PUBLISHED[int(o)]) <= TOL) for o in ("1", "2")}
    res["agrees_with_published"] = agree
    res["verdict"] = ("our truncation machinery reproduces both published crossings, so 31's corrected "
                      "numbers are externally validated" if all(agree.values()) else
                      "our crossings DISAGREE with the published calibration and the golden test passes -- "
                      "resolve before quoting 31 again")

    print("\nearliest curve vs published (a 1% line in a four-curve figure reports the earliest):")
    for o in ("1", "2"):
        e = res["earliest"][o]
        print(f"   order {o}: ours {e['crossing']:.3f} ({e['curve']})  published {PUBLISHED[int(o)]:.2f}"
              f"  -> {'agree' if agree[o] else 'DISAGREE'}")
    print(f"\nVERDICT: {res['verdict']}")

    OUT.write_text(json.dumps(res, indent=2))
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
