"""L1 gate: does ratio-filter dechirping actually reproduce our matched filter, and is it faster HERE?

Follow-up A left the subsolar arc at a wall it measured precisely: template-bank MISMATCH is the dominant
loss (real 1,619-template bank 0.489 vs CNN 0.472 vs true-template oracle 0.720), and 1,619 was our ceiling
because bank_dense.py could not hold the bank in memory ("33 MB/template -> 53 GB"). arXiv:2601.18835 says
a target template's SNR series can be produced from a NEARBY reference's by one short FIR, which would store
analytic chunks for only ~1% of the bank. That would remove the ceiling on the one axis that matters.

THIS SCRIPT DECIDES WHETHER THAT IS TRUE FOR US, before any of it is built on.

PRE-REGISTERED PASS CRITERIA (fixed before running; failing any of them means L1's premise fails here):
  G1 ACCURACY   reconstruction match > 0.999 vs the direct matched filter, using <= 512 taps, at a
                reference-target separation of ~1% of the bank (the paper's hierarchy: coarse reference bank
                ~1% the size of the dense target bank).
  G2 PEAK SNR   the recovered peak SNR — the number the detector actually thresholds on — within 1% of
                direct, over that same separation.
  G3 DEGRADATION accuracy must degrade GRACEFULLY with separation and improve with taps. A method that is
                only accurate at zero separation is useless: the whole point is covering many targets per
                reference.
  G4 SPEED      measured on THIS hardware (MPS/CPU), not assumed. The paper's 8x is a CPU cache result; our
                FLOP-level gain is only ~2x. **G4 is reported, NOT required** — the memory win alone
                (~100x fewer stored spectra) is what lifts the density ceiling, and a speedup that does not
                transfer would not invalidate that. Recording it honestly is the point.

Run:  .venv/bin/python scripts/bank_ratio_golden.py [--n-sep 6] [--quick]
"""
import argparse
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
MC_REF = 0.30                      # a mid-bank reference chirp mass
EQ = 2.0 ** 0.2
TAPS = [65, 129, 257, 513]
# separations as FRACTIONAL Mc offsets. At 0.1% target spacing a 1%-size reference bank puts the furthest
# target ~50 steps = ~5% away; we bracket well past that to see where the method breaks.
SEPS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]


def whitened_template(mc: float, base, t0, psd) -> np.ndarray:
    h, _ = make_whitened_injection(replace(base, mass1=float(mc * EQ), mass2=float(mc * EQ)), "H1", t0, psd)
    g = h[-WIN:].copy()
    return g if len(g) == WIN else np.pad(g, (WIN - len(g), 0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="one chunk, fewer separations")
    args = ap.parse_args()

    gps = json.loads((C.DATA_DIR / "manifest.json").read_text())["H1"]["test"][0]
    w, t0, psd = whiten_segment("H1", gps)
    wc = w[CROP:-CROP]
    base = sample_params(np.random.default_rng(0))
    rng = np.random.default_rng(C.SEED + 991)

    # one real-noise window with a loud injection at the REFERENCE mass, so peak SNR is meaningful
    p = replace(base, mass1=MC_REF * EQ, mass2=MC_REF * EQ)
    h_w, snr_ref = make_whitened_injection(p, "H1", t0, psd)
    m = int(rng.integers(max(len(h_w), WIN), len(wc)))
    d = wc[m - WIN:m].astype(np.float64).copy()
    d += (h_w * (20.0 / snr_ref))[-WIN:]

    g_ref = whitened_template(MC_REF, base, t0, psd)
    chunks_ref = so.analytic_chunks(g_ref, so.N_CHUNKS[-2] if not args.quick else 1)
    off_r, a_ref, _ = chunks_ref[len(chunks_ref) // 2]        # a mid, energetic chunk
    L = len(a_ref)
    d_chunk = d[off_r:off_r + L]
    c_ref = rf.corr_series(d_chunk, a_ref)

    seps = SEPS[:3] if args.quick else SEPS
    taps_list = TAPS[:3] if args.quick else TAPS
    print(f"segment {gps} | chunk len {L} | reference Mc {MC_REF}")
    print(f"{'sep':>7} {'dMc':>8} " + "".join(f"{'taps ' + str(t):>13}" for t in taps_list)
          + f"{'peakSNR err':>13}")

    rows = []
    for sep in seps:
        mc_t = MC_REF * (1 + sep)
        g_t = whitened_template(mc_t, base, t0, psd)
        ch_t = so.analytic_chunks(g_t, so.N_CHUNKS[-2] if not args.quick else 1)
        # align on the same chunk index so the comparison is like-for-like
        off_t, a_t, _ = ch_t[min(len(ch_t) // 2, len(ch_t) - 1)]
        c_direct = rf.corr_series(d[off_r:off_r + L], a_t[:L] if len(a_t) >= L else np.pad(a_t, (0, L - len(a_t))))
        rho_direct = np.abs(c_direct)

        cells, best = [], None
        for nt in taps_list:
            taps = rf.ratio_taps(a_t[:L] if len(a_t) >= L else np.pad(a_t, (0, L - len(a_t))), a_ref, nt)
            rho_rec = np.abs(rf.apply_taps(c_ref, taps))
            mm = rf.match(rho_rec, rho_direct)
            cells.append(f"{mm:>13.5f}")
            best = (nt, rho_rec, mm)
        pk_err = abs(best[1].max() - rho_direct.max()) / max(rho_direct.max(), 1e-12)
        rows.append({"sep": sep, "dMc": mc_t - MC_REF,
                     "match": {str(t): rf.match(np.abs(rf.apply_taps(c_ref, rf.ratio_taps(
                         a_t[:L] if len(a_t) >= L else np.pad(a_t, (0, L - len(a_t))), a_ref, t))), rho_direct)
                         for t in taps_list},
                     "peak_snr_rel_err_best_taps": float(pk_err), "best_taps": best[0]})
        print(f"{sep:>7.3f} {mc_t-MC_REF:>8.4f} " + "".join(cells) + f"{pk_err:>13.4f}")

    # ---- G4 speed, measured here ------------------------------------------------------------------------
    a_t_fix = whitened_template(MC_REF * 1.01, base, t0, psd)
    ch = so.analytic_chunks(a_t_fix, so.N_CHUNKS[-2] if not args.quick else 1)
    a_t_fix = ch[min(len(ch) // 2, len(ch) - 1)][1]
    a_t_fix = a_t_fix[:L] if len(a_t_fix) >= L else np.pad(a_t_fix, (0, L - len(a_t_fix)))
    N_REP = 20
    t0_ = time.time()
    for _ in range(N_REP):
        rf.corr_series(d_chunk, a_t_fix)
    t_direct = (time.time() - t0_) / N_REP
    taps = rf.ratio_taps(a_t_fix, a_ref, 257)
    t0_ = time.time()
    for _ in range(N_REP):
        rf.apply_taps(c_ref, taps)
    t_ratio = (time.time() - t0_) / N_REP
    print(f"\nG4 SPEED (per target template, {L}-sample chunk, CPU numpy):")
    print(f"  direct matched filter {1e3*t_direct:7.2f} ms | ratio filter (257 taps) {1e3*t_ratio:7.2f} ms"
          f" -> {t_direct/t_ratio:.2f}x")
    print("  (kernel FIT is a one-off per (ref,target) pair and is excluded — it is bank-build cost, not")
    print("   per-segment search cost. Reported separately below.)")
    t0_ = time.time()
    for _ in range(5):
        rf.ratio_taps(a_t_fix, a_ref, 257)
    t_fit = (time.time() - t0_) / 5
    print(f"  one-off kernel fit {1e3*t_fit:7.2f} ms/template")

    # ---- verdict against the pre-registered criteria -----------------------------------------------------
    ref_sep = 0.05
    at_ref = min(rows, key=lambda r: abs(r["sep"] - ref_sep))
    best_at_ref = max(float(v) for k, v in at_ref["match"].items() if int(k) <= 512)
    g1 = best_at_ref > 0.999
    g2 = at_ref["peak_snr_rel_err_best_taps"] < 0.01
    m_small = max(float(v) for v in rows[0]["match"].values())
    m_large = max(float(v) for v in rows[-1]["match"].values())
    g3 = m_small >= m_large
    out = {"gps": int(gps), "chunk_len": int(L), "mc_ref": MC_REF, "rows": rows,
           "speed": {"direct_ms": 1e3 * t_direct, "ratio_ms": 1e3 * t_ratio,
                     "speedup": t_direct / t_ratio, "kernel_fit_ms": 1e3 * t_fit},
           "G1_accuracy": bool(g1), "G1_match_at_5pct_sep": best_at_ref,
           "G2_peak_snr": bool(g2), "G2_rel_err": at_ref["peak_snr_rel_err_best_taps"],
           "G3_graceful": bool(g3), "passed": bool(g1 and g2 and g3)}
    print(f"\nG1 accuracy  match {best_at_ref:.5f} at {100*at_ref['sep']:.0f}% Mc separation -> "
          f"{'PASS' if g1 else 'FAIL'} (need > 0.999)")
    print(f"G2 peak SNR  rel err {at_ref['G2_rel_err'] if False else at_ref['peak_snr_rel_err_best_taps']:.4f} -> "
          f"{'PASS' if g2 else 'FAIL'} (need < 0.01)")
    print(f"G3 graceful  match {m_small:.5f} (closest) >= {m_large:.5f} (furthest) -> {'PASS' if g3 else 'FAIL'}")
    print(f"\nL1 PREMISE: {'HOLDS on our data — proceed to the dense bank' if out['passed'] else 'FAILS HERE — do not build on it'}")
    (C.RESULTS_DIR / "bank_ratio_golden.json").write_text(json.dumps(out, indent=2))
    print("wrote bank_ratio_golden.json")


if __name__ == "__main__":
    main()
