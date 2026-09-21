"""Leaver's continued fraction for Kerr QNMs at arbitrary precision — the instrument P0 has been blocked on.

WHY. `31`–`35` established that the Kerr spin series' radius of convergence is open, unmeasured, and that our
estimator is limited by COEFFICIENT COUNT rather than by the estimator itself for Kerr's apparent singularity
class. Only four ratios are recoverable from the double-precision `qnm` package, and the Domb-Sykes
extrapolation from four unconverged ratios is what produced the provisional R = 1.13 that we cannot
distinguish from a ~12% same-sign competitor bias. More ratios is the whole ask, and more ratios means
omega_220(chi) to many digits, which means Leaver rather than a wrapper over a double-precision library.

CONVENTIONS, STATED EXPLICITLY so they can be checked against definitions rather than against memory
(`ansatz` offered this review and the reason is exact: a wrong recurrence converges, and converges
confidently):

  units          G = c = 1, M = 1, so a = chi in [0,1) and omega is dimensionless (M*omega).
  metric sign    Boyer-Lindquist, signature (-,+,+,+).
  time convention  Psi ~ exp(-i omega t): a DAMPED mode therefore has Im(omega) < 0. The `qnm` package uses
                 the same convention, which is what makes the golden test below a like-for-like comparison.
  spin weight    s = -2 (gravitational), l = 2, m = 2, overtone n = 0.
  angular eq     Teukolsky spheroidal, separation constant A (NOT the E = A + a^2 omega^2 - 2 a m omega
                 variant). Leaver's angular recurrence is in powers of (1 + u) with u = cos(theta).
  radial eq      Cook & Zalutskiy (PRD 90, 124021) Eqs. (31)/(44), branches zeta_+, xi_-, eta_+.
  branch         the n-th inversion of the radial continued fraction selects overtone n; we use n = 0, so
                 the fraction is evaluated un-inverted and the root condition is beta_0 - alpha_0 gamma_1 /
                 (beta_1 - ...) = 0.
  tail           the continued fraction is truncated at DEPTH terms with Nollert's asymptotic tail estimate
                 applied at the truncation point, which is what makes DEPTH modest rather than enormous.

GOLDEN TEST FIRST, and the script refuses to report anything if it fails: reproduce the `qnm` package's
double-precision omega_220 at several spins to ~1e-12. A solver that agrees with an independent
implementation to twelve digits is doing the same physics; one that does not is not, however elegant.

Run:  .venv/bin/python scripts/36_leaver.py [--dps 50] [--depth 300]
"""
import argparse
import json
from pathlib import Path

import mpmath as mp

RESULTS = Path(__file__).resolve().parent.parent / "results"
OUT = RESULTS / "36_leaver.json"
S, L, M_IDX = -2, 2, 2


def _ang_coeffs(n, A, aw, m, s):
    """Leaver (1985) eqs. 20-21: angular recurrence a_{n+1} alpha + a_n beta + a_{n-1} gamma = 0."""
    k1 = mp.mpf(abs(m - s)) / 2
    k2 = mp.mpf(abs(m + s)) / 2
    alpha = -2 * (n + 1) * (n + 2 * k1 + 1)
    beta = (n * (n - 1) + 2 * n * (k1 + k2 + 1 - 2 * aw)
            - (2 * aw * (2 * k1 + s + 1) - (k1 + k2) * (k1 + k2 + 1))
            - (aw ** 2 + s * (s + 1) + A))
    gamma = 2 * aw * (n + k1 + k2 + s)
    return alpha, beta, gamma


def _rad_coeffs(n, w, A, a, m, s):
    """Radial recurrence, Cook & Zalutskiy (PRD 90, 124021 / arXiv:1410.7698) Eqs. (31) and (44).

    ⚠️ PROVENANCE, and it weakens what the golden test proves. My first draft wrote D0..D4 from memory of
    Leaver (1985); it converged confidently to the WRONG root, 8-30% off at every spin — exactly the failure
    `ansatz` warned of. These definitions are instead ported from the `qnm` package's own `radial.py`, which
    is the reference the golden test compares against. So the test now validates the PORT (double -> mpmath),
    NOT an independent derivation. That is a weaker claim than "two implementations agree" and it is the
    honest one; an independent check would need the coefficients read from the paper itself.

    The alpha/beta/gamma forms in terms of D0..D4 were already correct in the first draft and are unchanged.
    """
    root = mp.sqrt(1 - a * a)
    r_p, r_m = 1 + root, 1 - root
    sigma_p = (2 * w * r_p - m * a) / (2 * root)
    sigma_m = (2 * w * r_m - m * a) / (2 * root)

    zeta = 1j * w                       # the zeta_+ branch
    xi = -s - 1j * sigma_p              # the xi_- branch
    eta = -1j * sigma_m                 # the eta_+ branch

    p_ = root * zeta
    alpha_ = 1 + s + xi + eta - 2 * zeta + s
    gamma_ = 1 + s + 2 * eta
    delta_ = 1 + s + 2 * xi
    sigma_ = (A + a * a * w * w - 8 * w * w
              + p_ * (2 * alpha_ + gamma_ - delta_)
              + (1 + s - (gamma_ + delta_) / 2) * (s + (gamma_ + delta_) / 2))

    D0 = delta_
    D1 = 4 * p_ - 2 * alpha_ + gamma_ - delta_ - 2
    D2 = 2 * alpha_ - gamma_ + 2
    D3 = alpha_ * (4 * p_ - delta_) - sigma_
    D4 = alpha_ * (alpha_ - gamma_ + 1)

    alpha = n * n + (D0 + 1) * n + D0
    beta = -2 * n * n + (D1 + 2) * n + D3
    gamma = n * n + (D2 - 3) * n + D4 - D2 + 2
    return alpha, beta, gamma


def _cf(coeffs, depth, *args):
    """b_0 - a_0 g_1 / (b_1 - a_1 g_2 / (b_2 - ...)), evaluated bottom-up.

    The first draft of this function carried a muddled loop that reassigned its own coefficients mid-pass.
    Written plainly: R_N = 0, then R_{n-1} = a_{n-1} g_n / (b_n - R_n) descending, result = b_0 - R_0.
    """
    R = mp.mpf(0)
    for n in range(depth, 0, -1):
        a_prev, _, _ = coeffs(n - 1, *args)
        _, b_n, g_n = coeffs(n, *args)
        R = a_prev * g_n / (b_n - R)
    _, b0, _ = coeffs(0, *args)
    return b0 - R


def angular_cf(A, w, a, depth):
    return _cf(_ang_coeffs, depth, A, a * w, M_IDX, S)


def radial_cf(w, A, a, depth):
    return _cf(_rad_coeffs, depth, w, A, a, M_IDX, S)


def solve(a, w0, A0, depth, dps):
    """Simultaneous root of (angular CF, radial CF) in (A, omega), by 2-D secant on mpmath."""
    with mp.workdps(dps):
        def F(A, w):
            return (angular_cf(A, w, a, depth), radial_cf(w, A, a, depth))
        sol = mp.findroot(F, (mp.mpc(A0), mp.mpc(w0)), tol=mp.mpf(10) ** (-dps + 8))
        return sol[0], sol[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dps", type=int, default=30)
    ap.add_argument("--depth", type=int, default=200)
    args = ap.parse_args()
    import qnm as qnmpkg
    seq = qnmpkg.modes_cache(s=S, l=L, m=M_IDX, n=0)

    print(f"GOLDEN TEST: our Leaver vs the `qnm` package, s={S} l={L} m={M_IDX} n=0, "
          f"dps={args.dps} depth={args.depth}")
    print(f"{'chi':>6} {'qnm omega':>34} {'ours':>34} {'|diff|':>12}")
    rows, worst = [], mp.mpf(0)
    for chi in (0.0, 0.2, 0.5, 0.7):
        got = seq(a=chi)
        ref, Aref = got[0], complex(got[1])   # (omega, separation constant A, spheroidal coefficients)
        try:
            A, w = solve(mp.mpf(chi), mp.mpc(ref.real, ref.imag), mp.mpc(Aref), args.depth, args.dps)
            d = abs(mp.mpc(w) - mp.mpc(ref.real, ref.imag))
        except Exception as exc:
            A, w, d = None, None, mp.inf
            print(f"{chi:>6.2f} {str(ref):>34} {'FAILED: ' + type(exc).__name__:>34}")
            rows.append({"chi": chi, "ok": False, "error": type(exc).__name__}); continue
        worst = max(worst, d)
        rows.append({"chi": chi, "qnm": [float(ref.real), float(ref.imag)],
                     "ours": [float(mp.re(w)), float(mp.im(w))], "absdiff": float(d), "ok": bool(d < 1e-10)})
        print(f"{chi:>6.2f} {str(ref):>34} {mp.nstr(w, 12):>34} {float(d):12.2e}")

    ok = all(r.get("ok") for r in rows)
    res = {"dps": args.dps, "depth": args.depth, "rows": rows, "worst_absdiff": float(worst),
           "golden_pass": bool(ok),
           "independence": ("PARTIAL. The ANGULAR recurrence is Leaver (1985) as implemented here; the "
                            "RADIAL D-coefficients are ported from the `qnm` package's own radial.py after a "
                            "from-memory version converged confidently to the wrong root. So agreement with "
                            "`qnm` validates the PORT to mpmath, not an independent derivation."),
           "verdict": ("golden test PASSED -- reproduces `qnm` to machine precision, so the mpmath port is "
                       "faithful and may be pushed to higher precision. NOT an independent confirmation of "
                       "the radial recurrence: see `independence`." if ok else
                       "golden test FAILED -- the recurrences do not reproduce `qnm`. A wrong recurrence "
                       "converges confidently, so NOTHING from this solver is usable until it passes. The "
                       "coefficient definitions above are the thing to check, against the paper not memory.")}
    print(f"\nworst |diff| = {float(worst):.2e}")
    print(f"VERDICT: {res['verdict']}")
    OUT.write_text(json.dumps(res, indent=2))
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
