"""O(a^2) truncation of the sGB QNM CORRECTION itself -- by evaluation, not by fitting.

WHY. 38 measured the truncation of the sGB BACKGROUND (metric functions, horizon quantities). What a ringdown
test consumes is the frequency correction omega^(1) (omega = omega_Kerr + zeta omega^(1)). METRICS
(arXiv:2406.11986, Appendix B) tabulates omega^(1) NON-perturbatively at ten spins a = 0.005 ... 0.849 with a
numerical error per entry, and reprints the published SECOND-ORDER-IN-SPIN series of Pierini & Gualtieri
(PG, PRD 106 104009, arXiv:2207.11267) converted to its own normalisation (main.tex Eq. freq_Gultierai; "our
alpha is alpha/4 in PG") for the polar 022, 033 and 021 modes. Comparing the two at TABULATED spins is pure
evaluation of two published objects: no fit, no extraction, no estimator.

WHY NOT THE PAPER'S OWN FIT. METRICS fits a degree-8 polynomial to 10 points and says its w_0,1,2 are "close
to" PG's. Its printed numbers say otherwise: 022P w1 = -2.518 vs PG -0.335, w2 = +48.47 vs PG -0.32 (x150,
opposite sign), with alternating higher coefficients 157, -682, 1935 -- the ill-conditioned-least-squares
species that corrupted our own 31 (np.polyfit at truncation order). The fit reproduces its data points; its
low coefficients are NOT Taylor coefficients. Checked below (S1) rather than asserted.

GOLDEN (G1): two independent calculations -- METRICS (spectral) and PG (shooting) -- must agree at small spin,
and the residual |T2(a) - omega(a)| must grow like a^3 once above PG's printed-digit floor. That tests the
normalisation conversion and the table transcription at once; a factor-16 slip in zeta shows up at a^0.

PRE-REGISTRATION (written before this script was first run; committed in that state):
  * G1 PASS if the a^0 agreement at a = 0.005 is < 1% for all three modes and the log-log slope of the
    residual over a in {0.1, 0.2, 0.3} is in [2, 4].
  * MEASUREMENT: |T2 - omega_METRICS| / |omega_METRICS| at a = 0.6, 0.7, 0.8 (tabulated) and linearly
    interpolated to 0.69, per polar mode, with the METRICS numerical error as the error bar.
  * ⚠️ NOT BLIND for 022P at a = 0.7: while scoping I evaluated it by hand (~12.6%) before writing this.
    Blind for 033P, 021P and every other spin.
  * PREDICTION (033P, 021P at 0.69): 5-20%, i.e. comparable to or above Kerr 220's own 6.36%, because 38 found
    the sGB background converging more slowly than Kerr wherever the horizon matters, and a QNM is set by the
    near-horizon potential. Stated so it can be wrong.

Run:  .venv/bin/python scripts/39_sgb_qnm_truncation.py --tex <path to arXiv:2406.11986 main.tex>
"""
import argparse
import json
import math
import re
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
OUT = RESULTS / "39_sgb_qnm_truncation.json"

# PG second-order-in-spin polar series, BUILT FROM PG'S OWN TABLES (arXiv:2207.11267) rather than taken
# from METRICS' reprint. PG Eq. (40): omega = w0 + a m w1 + a^2 (w2a + m^2 w2b); METRICS normalisation is
# 16 x PG's O(zeta^2) coefficient (METRICS footnote; the x16 is what G1's a^0 agreement verifies).
#
# ⚠️ FIRST RUN USED THE REPRINT AND G1 FAILED ON 033P (residual slope 1.33, needs ~3). Rebuilding from PG's
# tables reproduces every reprinted coefficient EXCEPT 033P's linear imaginary term: PG gives
# 16 * 3 * (+1.18e-3) = +0.05664, METRICS prints -0.05664. A sign error in the reprint, caught by the gate.
#
# Two versions of PG's O(zeta^2) coefficient exist: Tables I-IV hold C2 of a degree-6 FIT in zeta (what METRICS
# used); Table V holds the TAYLOR coefficient omega_r(2). The Taylor one is the correct object -- a
# least-squares coefficient is not a Taylor coefficient, which is 31's lesson -- so it is primary; the fit
# version is kept as a sensitivity check (they differ by ~0.4%).
PG_TAYLOR = {  # Table V, rows "r 2": (Re, Im) for l = 2 and l = 3
    "0":  {2: (-1.411e-2, -4.70e-3), 3: (-5.463e-2, -7.21e-3)},
    "1":  {2: (-1.049e-2, 4e-5),     3: (-2.166e-2, 1.17e-3)},
    "2a": {2: (1.340e-2, 6.54e-3),   3: (2.947e-2, 1.777e-2)},
    "2b": {2: (-8.42e-3, 3.19e-3),   3: (-9.50e-3, 4.3e-4)},
}
PG_FITC2 = {   # Tables I-IV, row C2 -- what METRICS reprinted
    "0":  {2: (-1.406e-2, -4.70e-3), 3: (-5.453e-2, -7.19e-3)},
    "1":  {2: (-1.048e-2, 2e-5),     3: (-2.156e-2, 1.18e-3)},
    "2a": {2: (1.348e-2, 6.50e-3),   3: (2.941e-2, 1.857e-2)},
    "2b": {2: (-8.37e-3, 3.13e-3),   3: (-9.95e-3, 7.0e-4)},
}
MODES = {"022P": (2, 2), "033P": (3, 3), "021P": (2, 1)}


def pg_series(src, l, m):
    c = {k: complex(*v[l]) * 16 for k, v in src.items()}
    return [c["0"], m * c["1"], c["2a"] + m * m * c["2b"]]


PG = {k: pg_series(PG_TAYLOR, *lm) for k, lm in MODES.items()}
PG_FIT = {k: pg_series(PG_FITC2, *lm) for k, lm in MODES.items()}
# METRICS main.tex Eq. freq_Gultierai, verbatim -- compared against PG_FIT below, not used for measurement.
REPRINT = {
    "022P": [complex(-0.22496, -0.0752), complex(-0.33536, 0.00064), complex(-0.32, 0.30432)],
    "033P": [complex(-0.87248, -0.11504), complex(-1.03488, -0.05664), complex(-0.96224, 0.39792)],
    "021P": [complex(-0.22496, -0.0752), complex(-0.16768, 0.00032), complex(0.08176, 0.15408)],
}
# METRICS degree-8 fit, printed truncated at a^4 (main.tex Eq. METRICS_fitting_polynomials), polar modes.
FIT4 = {
    "022P": [complex(-0.215202, -0.0734094), complex(-2.51816, -0.411031), complex(48.4659, 9.59898),
             complex(-431.428, -82.4765), complex(1935.01, 378.832)],
    "033P": [complex(-0.872789, -0.113506), complex(-1.20198, -0.339974), complex(2.82484, 9.09087),
             complex(-35.6024, -75.7708), complex(165.435, 341.913)],
    "021P": [complex(-0.22612, -0.0754879), complex(0.0933968, 0.0504285), complex(-5.91521, -0.968127),
             complex(54.1801, 10.6838), complex(-252.83, -51.1785)],
}
CPLX = re.compile(r"\$\s*([-+]?\s*[\d.]+)\s*([-+])\s*([\d.]+)\s*i\s*\$")
ERR = re.compile(r"\$\s*\(\s*([\d.]+)\s*\+\s*([\d.]+)\s*i?\s*\)\s*\\times\s*10\^\{(-?\d+)\}\s*\$")


def table(tex, label):
    """Rows of an Appendix-B table: {a: {'A': (omega, err), 'P': (omega, err)}}."""
    end = tex.index(r"\label{" + label + "}")
    start = tex.rindex(r"\begin{tabular}", 0, end)
    rows = {}
    for line in tex[start:end].splitlines():
        cells = line.split("&")
        if len(cells) < 8 or not re.match(r"\s*\d", cells[0]):
            continue
        vals = []
        for c in (cells[3], cells[6]):
            m = CPLX.search(c)
            vals.append(complex(float(m.group(1).replace(" ", "")), (1 if m.group(2) == "+" else -1) * float(m.group(3))))
        errs = []
        for c in (cells[4], cells[7]):
            m = ERR.search(c)
            s = 10.0 ** int(m.group(3))
            errs.append(abs(complex(float(m.group(1)) * s, float(m.group(2)) * s)))
        rows[float(cells[0])] = {"A": (vals[0], errs[0]), "P": (vals[1], errs[1])}
    return rows


def poly(c, a):
    return sum(ck * a ** k for k, ck in enumerate(c))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", required=True)
    tex = Path(ap.parse_args().tex).read_text()
    tabs = {"022": table(tex, "tab:omega_1_022"), "033": table(tex, "tab:omega_1_033"),
            "021": table(tex, "tab:omega_1_021")}
    for k, t in tabs.items():
        if len(t) != 10:
            raise SystemExit(f"table {k}: parsed {len(t)} rows, expected 10 -- transcription unsafe")
    rep = {"source": "arXiv:2406.11986 main.tex Appendix B + Eq. freq_Gultierai", "modes": {}}

    audit = {}
    for mode in MODES:
        bad = [(k, part) for k in range(3) for part in ("re", "im")
               if abs(getattr(REPRINT[mode][k], "real" if part == "re" else "imag")
                      - getattr(PG_FIT[mode][k], "real" if part == "re" else "imag")) > 1e-4]
        audit[mode] = [f"a^{k} {part}: reprint {getattr(REPRINT[mode][k], 'real' if part == 're' else 'imag'):+.5f}"
                       f" vs PG tables {getattr(PG_FIT[mode][k], 'real' if part == 're' else 'imag'):+.5f}"
                       for k, part in bad]
        print(f"  reprint audit {mode}: {'matches PG tables' if not bad else 'MISMATCH ' + '; '.join(audit[mode])}")
    rep["reprint_audit"] = audit

    ok_g1 = True
    for mode, pg in PG.items():
        t = {a: v["P"] for a, v in tabs[mode[:3]].items()}
        # ---- G1: independent calculations agree at small spin; residual ~ a^3 ----------------------------
        a0 = abs(poly(pg, 0.005) - t[0.005][0]) / abs(t[0.005][0])
        res = {a: abs(poly(pg, a) - t[a][0]) for a in t}
        xs = [0.1, 0.2, 0.3]
        lx = [math.log(a) for a in xs]; ly = [math.log(res[a]) for a in xs]
        mx, my = sum(lx) / 3, sum(ly) / 3
        slope = sum((x - mx) * (y - my) for x, y in zip(lx, ly)) / sum((x - mx) ** 2 for x in lx)
        g1 = a0 < 0.01 and 2 <= slope <= 4
        ok_g1 &= g1
        # ---- measurement ----------------------------------------------------------------------------------
        rel = {a: {"rel": res[a] / abs(t[a][0]), "bar": t[a][1] / abs(t[a][0]), "T2": [poly(pg, a).real, poly(pg, a).imag],
                   "metrics": [t[a][0].real, t[a][0].imag]} for a in sorted(t)}
        r69 = rel[0.6]["rel"] + (rel[0.7]["rel"] - rel[0.6]["rel"]) * 0.9
        sens = {a: abs(poly(PG_FIT[mode], a) - t[a][0]) / abs(t[a][0]) for a in (0.6, 0.7, 0.8)}
        # ---- S1: the paper's printed fit, truncated at a^4, against its own table at small spin ----------
        s1 = {a: abs(poly(FIT4[mode], a) - t[a][0]) / abs(t[a][0]) for a in (0.005, 0.1, 0.2)}
        rep["modes"][mode] = {"G1": {"a0_rel_at_0.005": a0, "residual": {str(a): res[a] for a in sorted(res)},
                                     "loglog_slope_0.1_0.3": slope, "pass": g1},
                              "rel_err": {str(a): v for a, v in rel.items()}, "rel_err_0.69_interp": r69,
                              "sensitivity_fitC2_rel": {str(a): v for a, v in sens.items()},
                              "S1_fit4_vs_table": {str(a): v for a, v in s1.items()},
                              "fit_vs_PG_coeffs": {"w1": [FIT4[mode][1].real, FIT4[mode][1].imag],
                                                   "PG_w1": [pg[1].real, pg[1].imag],
                                                   "w2": [FIT4[mode][2].real, FIT4[mode][2].imag],
                                                   "PG_w2": [pg[2].real, pg[2].imag]}}
        print(f"  {mode}: G1 a^0 agreement {100*a0:.3f}%, residual slope over 0.1-0.3 = {slope:.2f}  "
              f"-> {'PASS' if g1 else 'FAIL'}")
        print(f"        O(a^2) truncation error:  " + "  ".join(
            f"a={a}: {100*rel[a]['rel']:.2f}% (+/-{100*rel[a]['bar']:.3f})" for a in (0.3, 0.5, 0.6, 0.7, 0.8, 0.849))
              + f"   | interp 0.69: {100*r69:.2f}%")
        print(f"        sensitivity (PG fit-C2 instead of Taylor): " + "  ".join(f"a={a}: {100*v:.2f}%" for a, v in sens.items()))
        print(f"        S1 paper's fit truncated at a^4 vs its own table: "
              + "  ".join(f"a={a}: {100*v:.1f}%" for a, v in s1.items()))
    rep["G1_all_pass"] = bool(ok_g1)
    OUT.write_text(json.dumps(rep, indent=2, default=str))
    print(f"wrote {OUT.name}   G1: {'PASS' if ok_g1 else 'FAIL -- measurement not reportable'}")


if __name__ == "__main__":
    main()
