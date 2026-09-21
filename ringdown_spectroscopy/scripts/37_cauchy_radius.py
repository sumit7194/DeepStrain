"""Taylor coefficients of omega_220(chi) by CAUCHY INTEGRAL on a complex circle — no fitting, no Domb-Sykes.

WHY THIS REPLACES THE PREVIOUS ROUTE. `36` gave a working arbitrary-precision Leaver, and 60-digit
frequencies bought exactly ONE extra coefficient ratio over double precision (4 -> 5). That is a negative for
the whole approach and it relocates the blocker: the limit was never the FREQUENCY precision, it is the
EXTRACTION -- least-squares fitting on a real interval [0, 0.25] determines high-order Taylor coefficients
only weakly, however exactly the data is known, because x^n is ~0.25^n there.

Cauchy does not fit. For a function analytic in |z| < R,

    a_n = (1/2 pi i) * contour_integral( omega(z) / z^(n+1) )  ~=  (1/N) * sum_k omega(r e^{2 pi i k/N}) e^{-2 pi i k n/N} / r^n

on N equally spaced points of |z| = r, which is a discrete Fourier transform. The aliasing error is
|a_{n+N}| r^N, so with r safely inside R and N comfortably larger than the n we want, the coefficients come
out to nearly full working precision. This is only possible because the Leaver solver ANALYTICALLY CONTINUES
TO COMPLEX SPIN -- verified directly: chi = 0.2+0.05i, 0.15+0.15i and 0.0+0.2i all return finite omega.

AND IT PROBES THE RIGHT PLANE. The radius of convergence is set by the nearest singularity in the COMPLEX
chi-plane; every previous attempt inferred its distance from real-axis data. This walks the plane.

CONTINUATION, and why it is not optional: the root-finder is seeded from the previous point on the circle, so
the branch is tracked continuously. A jump to a different QNM overtone would look like a perfectly good
analytic function with the wrong coefficients, so a jump guard compares each step against its neighbour.

Run:  .venv/bin/python scripts/37_cauchy_radius.py [--r 0.2] [--n 64] [--dps 40]
"""
import argparse
import importlib.util
import json
from pathlib import Path

import mpmath as mp

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
OUT = RESULTS / "37_cauchy_radius.json"
_spec = importlib.util.spec_from_file_location("lv", HERE / "36_leaver.py")
lv = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(lv)


def ring(r, N, dps, depth, jump_bar):
    """omega on |z| = r, continued point to point, with a jump guard between neighbours."""
    import qnm as q
    seq = q.modes_cache(s=-2, l=2, m=2, n=0)
    g = seq(a=float(r))
    A, w = mp.mpc(complex(g[1])), mp.mpc(g[0].real, g[0].imag)
    vals, jumps = [], 0
    with mp.workdps(dps):
        for k in range(N):
            z = r * mp.exp(2j * mp.pi * k / N)
            A, w = lv.solve(z, w, A, depth, dps)
            if vals and abs(w - vals[-1]) > jump_bar:
                jumps += 1
            vals.append(w)
        # TRUE CLOSURE: step once more to angle 2*pi, which is the SAME POINT as angle 0, seeded from the
        # last point on the loop. The first version of this check compared vals[0] against vals[-1] -- but
        # vals[-1] sits at 2*pi*(N-1)/N, one step SHORT of the start, so it measured a neighbour step and
        # reported 4.2e-3 as a closure failure on data that was fine. An instrument that made me distrust a
        # good result, which is the rarer direction and no less wrong.
        _, w_close = lv.solve(r * mp.exp(2j * mp.pi), w, A, depth, dps)
        closure = abs(w_close - vals[0])
        step = max(abs(vals[k + 1] - vals[k]) for k in range(len(vals) - 1))
    return vals, jumps, closure, step


def coeffs(vals, r, nmax, dps):
    """a_n from the discrete Fourier transform of omega on the circle."""
    N = len(vals)
    with mp.workdps(dps):
        out = []
        for n in range(nmax + 1):
            s = mp.mpc(0)
            for k, v in enumerate(vals):
                s += v * mp.exp(-2j * mp.pi * k * n / N)
            out.append(s / N / r ** n)
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=str, default="0.2")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--dps", type=int, default=40)
    ap.add_argument("--depth", type=int, default=400)
    ap.add_argument("--nmax", type=int, default=24)
    a = ap.parse_args()
    r = mp.mpf(a.r)

    print(f"circle |chi| = {a.r}, {a.n} points, dps {a.dps}, depth {a.depth}")
    vals, jumps, closure, step = ring(r, a.n, a.dps, a.depth, mp.mpf("0.5"))
    print(f"  continuation: {jumps} neighbour jumps > 0.5  "
          f"{'(clean)' if jumps == 0 else '(BRANCH JUMP -- coefficients not trustworthy)'}")
    print(f"  closure |omega(2pi) - omega(0)| = {float(closure):.2e}   vs typical neighbour step "
          f"{float(step):.2e}  ->  {'SINGLE-VALUED' if closure < step / 10 else 'NOT single-valued: a branch '
          'point lies inside this contour and the Cauchy coefficients are invalid'}")

    c = coeffs(vals, r, a.nmax, a.dps)
    with mp.workdps(a.dps):
        print(f"\n{'n':>4} {'|a_n|':>26} {'|a_n/a_(n-1)|':>16}")
        ratios = []
        for n in range(1, a.nmax + 1):
            rt = abs(c[n] / c[n - 1])
            ratios.append(float(rt))
            print(f"{n:>4} {mp.nstr(abs(c[n]), 14):>26} {float(rt):16.10f}")

    res = {"r": a.r, "n_points": a.n, "dps": a.dps, "depth": a.depth,
           "jumps": jumps, "closure": float(closure), "typical_step": float(step),
           "single_valued": bool(closure < step / 10),
           "abs_a": [float(abs(v)) for v in c], "ratios": ratios,
           "note": ("ratio |a_n/a_(n-1)| -> 1/R as n grows, with NO extrapolation needed if enough "
                    "coefficients are clean; the aliasing error is |a_(n+N)| r^N")}
    OUT.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {OUT.name}")


if __name__ == "__main__":
    main()
