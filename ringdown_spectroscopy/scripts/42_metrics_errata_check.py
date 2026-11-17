"""Reproduce the three METRICS (arXiv:2406.11986 v2) errata from the papers' own LaTeX sources. Read-only, ~1 s.

Run:  python scripts/42_metrics_errata_check.py --metrics <2406.11986v2 main.tex> --pg <2207.11267v2 .tex>
Needs only mpmath. Writes results/42_metrics_errata_check.json.

E1  GR horizon angular velocity. main.tex prints Omega_H^(0) = a/(2Mb), b = sqrt(1-a^2). For the Kerr metric in
    Boyer-Lindquist coordinates (the paper's background), Omega_H = -g_tphi/g_phiphi at r_+ = a/(2 M r_+)
    = a/(2M(1+b)) (M = 1). The printed form differs by the factor (1+b)/b. Its companion kappa^(0) = b/(2M(1+b))
    is printed correctly, which this script also checks.
E2  Reprint of Pierini & Gualtieri (arXiv:2207.11267), Eq. freq_Gultierai: the 033P linear-in-a imaginary term.
    Rebuilt here from PG's own table (r = 1, row C_2, Im l = 3), times 16 (coupling normalisation, footnote of
    main.tex) times m = 3.
E3  The printed fitting polynomials truncated at a^4 (Eq. METRICS_fitting_polynomials) against the paper's own
    Appendix-B tables at small spin, and their w_1, w_2 against PG's coefficients, which the text calls "close".
"""
import argparse
import importlib.util
import json
import re
from pathlib import Path

import mpmath as mp

HERE = Path(__file__).resolve().parent
_s = importlib.util.spec_from_file_location("s39", HERE / "39_sgb_qnm_truncation.py")
s39 = importlib.util.module_from_spec(_s); _s.loader.exec_module(s39)


def e1():
    mp.mp.dps = 30
    out = {}
    for a in (mp.mpf("0.3"), mp.mpf("0.69"), mp.mpf("0.9")):
        M = 1; b = mp.sqrt(1 - a * a); rp = M * (1 + b); th = mp.pi / 3
        sig = rp ** 2 + a ** 2 * mp.cos(th) ** 2
        g_tphi = -2 * M * a * rp * mp.sin(th) ** 2 / sig
        g_phph = (rp ** 2 + a ** 2 + 2 * M * a ** 2 * rp * mp.sin(th) ** 2 / sig) * mp.sin(th) ** 2
        omega = -g_tphi / g_phph
        kappa = (rp - M * (1 - b)) / (2 * (rp ** 2 + a ** 2))
        out[str(a)] = {"Omega_metric": float(omega), "printed_a_over_2Mb": float(a / (2 * M * b)),
                       "a_over_2M(1+b)": float(a / (2 * M * (1 + b))),
                       "kappa_metric": float(kappa), "printed_kappa_b_over_2M(1+b)": float(b / (2 * M * (1 + b)))}
    ok = all(abs(v["Omega_metric"] - v["a_over_2M(1+b)"]) < 1e-12 and
             abs(v["Omega_metric"] - v["printed_a_over_2Mb"]) > 1e-3 and
             abs(v["kappa_metric"] - v["printed_kappa_b_over_2M(1+b)"]) < 1e-12 for v in out.values())
    return {"values": out, "erratum_confirmed": ok}


def e2(mtex, pgtex):
    m = re.search(r"\(-1\.03488([-+])0\.05664i\)a", mtex)
    printed = (-1 if m.group(1) == "-" else 1) * 0.05664
    cap = pgtex.index("with $r=1$")
    line = next(ln for ln in pgtex[cap:].splitlines() if ln.strip().startswith("$C_2$"))
    cells = [c.strip().rstrip("\\").strip().strip("$") for c in line.split("&")]
    num = lambda s: float(re.sub(r"\s*\\cdot\s*10\^\{(-?\d+)\}", r"e\1", s))
    im_l3 = num(cells[4])
    rebuilt = im_l3 * 16 * 3
    return {"printed_im": printed, "PG_table_Im_l3_C2": im_l3, "rebuilt_im": rebuilt,
            "erratum_confirmed": abs(printed + rebuilt) < 1e-9 and abs(printed - rebuilt) > 1e-3}


def e3(mtex):
    for mode, c in s39.FIT4.items():
        lead = f"{c[0].real:g}"
        if lead not in mtex:
            raise SystemExit(f"transcription check failed: {mode} leading coefficient {lead} not in main.tex")
    out = {}
    for mode, c in s39.FIT4.items():
        t = {a: v["P"][0] for a, v in s39.table(mtex, f"tab:omega_1_{mode[:3]}").items()}
        out[mode] = {"fit_truncated_a4_vs_table_rel": {str(a): abs(s39.poly(c, a) - t[a]) / abs(t[a]) for a in (0.005, 0.1, 0.2)},
                     "w1_printed": [c[1].real, c[1].imag], "w1_PG": [s39.PG_FIT[mode][1].real, s39.PG_FIT[mode][1].imag],
                     "w2_printed": [c[2].real, c[2].imag], "w2_PG": [s39.PG_FIT[mode][2].real, s39.PG_FIT[mode][2].imag]}
    ok = out["022P"]["fit_truncated_a4_vs_table_rel"]["0.2"] > 1.0
    return {"modes": out, "erratum_confirmed": ok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True); ap.add_argument("--pg", required=True)
    a = ap.parse_args()
    mtex, pgtex = Path(a.metrics).read_text(), Path(a.pg).read_text()
    rep = {"metrics_source": "arXiv:2406.11986v2 main.tex", "pg_source": "arXiv:2207.11267v2",
           "E1": e1(), "E2": e2(mtex, pgtex), "E3": e3(mtex)}
    for k in ("E1", "E2", "E3"):
        print(k, "CONFIRMED" if rep[k]["erratum_confirmed"] else "NOT confirmed")
    print("E1 a=0.69:", rep["E1"]["values"]["0.69"])
    print("E2:", {k: rep["E2"][k] for k in ("printed_im", "rebuilt_im")})
    print("E3 022P:", rep["E3"]["modes"]["022P"])
    (HERE.parent / "results" / "42_metrics_errata_check.json").write_text(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
