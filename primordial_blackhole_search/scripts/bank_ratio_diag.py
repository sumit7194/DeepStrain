"""Why did bank_ratio_golden fail — a bug in our implementation, or the subsolar regime?

bank_ratio_golden.py returned match 0.964 at 0.1% Mc separation and 0.814 at 1%, against a pre-registered
>0.999. arXiv:2601.18835 reports >0.999 with 251 taps even when the reference and target match below 0.6, so
a result this poor is more likely OUR mistake than the method's failure. Two candidates, and they are
distinguishable:

  D1 IS THE ALGEBRA RIGHT?  Take the kernel UNTRUNCATED (full chunk length). Then c_t = c_r (*) IFFT[conj(R)]
     is an exact circular identity, so the reconstruction MUST match to numerical precision. If it does not,
     the derivation or the code is wrong and nothing downstream means anything. This is the control the
     golden test lacked.

  D2 HOW MANY TAPS DOES OUR REGIME NEED?  If D1 is clean, the failure is purely truncation, and the question
     becomes quantitative: how long must the kernel be for a subsolar pair? There is good reason to expect
     far more than the paper's ~250: a subsolar inspiral accumulates enormous orbital phase, so a given
     FRACTIONAL Mc offset produces a much larger phase difference than for the BNS case they tuned on, and
     R(f) is correspondingly less slowly-varying. That is a statement about our corner of parameter space,
     not about their method.

WHAT THE ANSWER DECIDES. The win we care about is MEMORY, not FLOPs: storing analytic chunks for ~1% of the
bank plus a kernel per target, instead of "33 MB/template -> 53 GB". That win survives a long kernel as long
as n_taps << chunk length (262,144). So:
    n_taps ~ 1e2-1e3  -> as advertised, huge win
    n_taps ~ 1e4      -> still a large memory win (16 KB vs 33 MB per target), worth building
    n_taps ~ 1e5+     -> the kernel is the template; no win, and L1 is dead FOR SUBSOLAR specifically

Run:  .venv/bin/python scripts/bank_ratio_diag.py
"""
import importlib.util
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pbh import config as C
from pbh import ratiofilter as rf
from pbh.data import whiten_segment
from pbh.waveforms import make_whitened_injection, sample_params

HERE = Path(__file__).resolve().parent
_so = importlib.util.spec_from_file_location("so", HERE / "semicoherent_oracle.py")
so = importlib.util.module_from_spec(_so); _so.loader.exec_module(so)

WIN, CROP = so.WIN, so.CROP
MC_REF = 0.30
EQ = 2.0 ** 0.2
TAPS = [257, 1025, 4097, 16385, 65537]
SEPS = [0.001, 0.005, 0.01]


def main() -> None:
    gps = json.loads((C.DATA_DIR / "manifest.json").read_text())["H1"]["test"][0]
    w, t0, psd = whiten_segment("H1", gps)
    wc = w[CROP:-CROP]
    base = sample_params(np.random.default_rng(0))
    rng = np.random.default_rng(C.SEED + 991)

    def templ(mc):
        h, _ = make_whitened_injection(replace(base, mass1=float(mc * EQ), mass2=float(mc * EQ)), "H1", t0, psd)
        g = h[-WIN:].copy()
        return g if len(g) == WIN else np.pad(g, (WIN - len(g), 0))

    p = replace(base, mass1=MC_REF * EQ, mass2=MC_REF * EQ)
    h_w, snr_ref = make_whitened_injection(p, "H1", t0, psd)
    m = int(rng.integers(max(len(h_w), WIN), len(wc)))
    d = wc[m - WIN:m].astype(np.float64).copy() + (h_w * (20.0 / snr_ref))[-WIN:]

    a_ref = so.analytic_chunks(templ(MC_REF), 1)[0][1]
    L = len(a_ref)
    c_ref = rf.corr_series(d, a_ref)
    print(f"chunk length L = {L}  (reference Mc {MC_REF})\n")

    out = {"L": int(L), "mc_ref": MC_REF, "rows": []}
    print(f"{'sep':>7} {'ref-tgt':>8} {'EXACT':>10} " + "".join(f"{t:>10}" for t in TAPS))
    print(f"{'':>7} {'match':>8} {'(D1)':>10} " + "".join(f"{'taps':>10}" for _ in TAPS))
    for sep in SEPS:
        a_t = so.analytic_chunks(templ(MC_REF * (1 + sep)), 1)[0][1]
        a_t = a_t[:L] if len(a_t) >= L else np.pad(a_t, (0, L - len(a_t)))
        rho_direct = np.abs(rf.corr_series(d, a_t))
        # how similar are the two TEMPLATES themselves? (the paper's "reference-target match")
        rt = abs(np.vdot(a_ref, a_t)) / (np.linalg.norm(a_ref) * np.linalg.norm(a_t))

        # D1: exact, untruncated kernel -- pure algebra check
        exact = rf.match(np.abs(rf.apply_taps(c_ref, rf.ratio_taps(a_t, a_ref, L - 1))), rho_direct)

        cells = []
        row = {"sep": sep, "ref_target_match": float(rt), "exact": float(exact), "taps": {}}
        for nt in TAPS:
            mm = rf.match(np.abs(rf.apply_taps(c_ref, rf.ratio_taps(a_t, a_ref, nt))), rho_direct)
            row["taps"][str(nt)] = float(mm)
            cells.append(f"{mm:>10.5f}")
        out["rows"].append(row)
        print(f"{sep:>7.3f} {rt:>8.4f} {exact:>10.6f} " + "".join(cells))

    # ---- verdict ---------------------------------------------------------------------------------------
    worst_exact = min(r["exact"] for r in out["rows"])
    out["D1_algebra_ok"] = bool(worst_exact > 0.999)
    print(f"\nD1 ALGEBRA: worst exact-kernel match = {worst_exact:.6f} -> "
          f"{'CORRECT (failure is truncation, not a bug)' if out['D1_algebra_ok'] else 'BUG — the derivation or code is wrong'}")

    if out["D1_algebra_ok"]:
        need = {}
        for r in out["rows"]:
            ok = [int(t) for t, v in r["taps"].items() if v > 0.999]
            need[str(r["sep"])] = min(ok) if ok else None
        out["D2_taps_needed_for_0999"] = need
        print("D2 TAPS NEEDED for match > 0.999:")
        for s, n in need.items():
            frac = f"{100*n/L:.1f}% of chunk" if n else f">{TAPS[-1]} ({100*TAPS[-1]/L:.0f}%+ of chunk)"
            print(f"    separation {float(s)*100:>5.1f}% Mc -> {n if n else 'NOT REACHED':>12}  ({frac})")
        vals = [v for v in need.values() if v]
        out["verdict"] = ("memory win intact" if vals and max(vals) <= L // 16 else
                          "kernel approaches template length -- no memory win for subsolar")
        print(f"  => {out['verdict']}")
    (C.RESULTS_DIR / "bank_ratio_diag.json").write_text(json.dumps(out, indent=2))
    print("\nwrote bank_ratio_diag.json")


if __name__ == "__main__":
    main()
