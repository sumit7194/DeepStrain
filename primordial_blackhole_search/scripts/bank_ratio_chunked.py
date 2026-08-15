"""L1 pre-build check: does ratio filtering reproduce the ACTUAL n=8 semi-coherent statistic, and what does
the real bank cost?

Everything measured so far used n=1 (the whole 64-s window as one chunk). The detector does not: bank_dense
scores the **n=8 semi-coherent newSNR**, so a target needs EIGHT kernels, not one. That is a flaw in
bank_ratio_costmodel.py's 6.5 GB figure and has to be corrected before a bank is built on it. It could go
either way, which is why it is measured:

  - each chunk is 1/8 the duration, so it accumulates ~1/8 the phase DIFFERENCE -> fewer taps per chunk
  - but there are 8 of them -> 8x as many kernels

If taps-per-chunk falls by ~8x, total storage is unchanged and the cost model survives intact. If it stays at
4,097, the real bank is 8x bigger than advertised and the plan needs re-pricing.

TWO THINGS THIS SETTLES
  C1  TAPS PER CHUNK at n=8, versus the 4,097 measured for n=1.
  C2  END-TO-END GOLDEN TEST — the one that actually matters. Reproduce `so.local_stats` (the exact statistic
      bank_dense thresholds on: A = sum_i |c_i|^2, B = sum_i |c_i|^4 / p_i, then PyCBC-style newSNR) for a
      TARGET template, computed entirely from a REFERENCE's per-chunk correlation series plus kernels, and
      compare against the direct computation. Agreement here means a ratio-filter bank is a drop-in for
      bank_dense; disagreement means the whole L1 build is void, however good the per-chunk matches look.

  Note the statistic is a NONLINEAR function of the correlation series (squares, fourth powers, a max over a
  local scan), so a good per-chunk match does NOT automatically give a good statistic match. That is exactly
  why C2 is run on the statistic itself rather than inferred from C1.

Run:  .venv/bin/python scripts/bank_ratio_chunked.py
"""
import importlib.util
import json
import sys
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

WIN, PAD, CROP = so.WIN, so.PAD, so.CROP
N_CHUNK = 8
EQ = 2.0 ** 0.2
MC_REF = 0.30
SEPS = [0.001, 0.005, 0.01]
TAPS = [129, 513, 2049, 8193]
BYTES_PER_TEMPLATE = 33e6
MC_LO, MC_HI = 0.173, 0.871


def main() -> None:
    gps = json.loads((C.DATA_DIR / "manifest.json").read_text())["H1"]["test"][0]
    w, t0, psd = whiten_segment("H1", gps)
    wc = w[CROP:-CROP]
    base = sample_params(np.random.default_rng(0))
    rng = np.random.default_rng(C.SEED + 7717)

    def templ(mc):
        h, _ = make_whitened_injection(replace(base, mass1=float(mc * EQ), mass2=float(mc * EQ)), "H1", t0, psd)
        g = h[-WIN:].copy()
        return g if len(g) == WIN else np.pad(g, (WIN - len(g), 0))

    # a real-noise window with a loud injection at the reference mass
    p = replace(base, mass1=MC_REF * EQ, mass2=MC_REF * EQ)
    h_w, snr_ref = make_whitened_injection(p, "H1", t0, psd)
    ws = int(rng.integers(PAD + WIN, len(wc) - WIN - PAD))
    d = wc.astype(np.float64).copy()
    d[ws:ws + WIN] += (h_w * (25.0 / snr_ref))[-WIN:]

    ch_ref = so.analytic_chunks(templ(MC_REF), N_CHUNK)
    print(f"segment {gps} | n={N_CHUNK} chunks | chunk len {len(ch_ref[0][1])} | reference Mc {MC_REF}\n")

    out = {"n_chunk": N_CHUNK, "chunk_len": int(len(ch_ref[0][1])), "rows": []}
    print("C1 TAPS PER CHUNK (worst chunk match at each tap count) and C2 the STATISTIC itself")
    print(f"{'sep':>7} " + "".join(f"{'taps ' + str(t):>13}" for t in TAPS)
          + f"{'taps@.999':>10}{'newSNR direct':>15}{'newSNR ratio':>14}{'rel err':>10}")

    for sep in SEPS:
        ch_t = so.analytic_chunks(templ(MC_REF * (1 + sep)), N_CHUNK)
        # pair chunks by offset so reference and target chunks are aligned spans
        pairs = []
        for off_t, a_t, p_t in ch_t:
            m = [c for c in ch_ref if c[0] == off_t]
            if m:
                pairs.append((off_t, a_t, p_t, m[0][1]))

        cells, need = [], None
        for nt in TAPS:
            worst = 1.0
            for off, a_t, _, a_r in pairs:
                lo = ws + off - PAD
                seg = d[lo:lo + len(a_t) + 2 * PAD]
                c_r = rf.corr_series(seg, a_r)
                c_t = rf.corr_series(seg, a_t)
                rec = rf.apply_taps(c_r, rf.ratio_taps(a_t, a_r, nt, n_fft=len(seg)))
                worst = min(worst, rf.match(np.abs(rec) ** 2, np.abs(c_t) ** 2))
            cells.append(f"{worst:>13.5f}")
            if need is None and worst > 0.999:
                need = nt

        # C2: the real statistic, direct vs reconstructed-from-reference
        nt_use = need or TAPS[-1]
        _, ns_direct = so.local_stats(d, [(o, a, pp) for o, a, pp, _ in pairs], ws)
        A = np.zeros(2 * PAD + 1); B = np.zeros(2 * PAD + 1)
        for off, a_t, p_t, a_r in pairs:
            lo = ws + off - PAD
            seg = d[lo:lo + len(a_t) + 2 * PAD]
            c_r = rf.corr_series(seg, a_r)
            rec = rf.apply_taps(c_r, rf.ratio_taps(a_t, a_r, nt_use, n_fft=len(seg)))
            # CONVENTION: so._corr_sq_local correlates via a reversed-conjugate CONVOLUTION, so its zero lag
            # sits at index len(a)-1. rf.corr_series uses IFFT[D conj(A)], whose zero lag is index 0. Slicing
            # the ratio path at len(a)-1 (as an earlier version did) reads the wrong window entirely and made
            # the statistic look broken precisely where the per-chunk match was BEST.
            x = (np.abs(rec) ** 2)[0:2 * PAD + 1]
            A += x; B += x ** 2 / p_t
        ns_ratio = float(so.newsnr(A, B, len(pairs)).max())
        rel = abs(ns_ratio - ns_direct) / max(ns_direct, 1e-12)
        out["rows"].append({"sep": sep, "taps_needed": need, "newsnr_direct": float(ns_direct),
                            "newsnr_ratio": ns_ratio, "rel_err": float(rel)})
        print(f"{sep:>7.3f} " + "".join(cells) + f"{str(need) if need else '>8193':>10}"
              f"{ns_direct:>15.4f}{ns_ratio:>14.4f}{rel:>10.5f}")

    # ---- re-price the bank with the REAL per-chunk kernel cost -------------------------------------------
    worst_taps = max((r["taps_needed"] or TAPS[-1]) for r in out["rows"])
    n1_taps = 4097                                   # bank_ratio_diag/mcscan, n=1
    out["taps_per_chunk"] = int(worst_taps)
    out["taps_n1_reference"] = n1_taps
    out["total_taps_ratio_vs_n1"] = N_CHUNK * worst_taps / n1_taps
    B_t = int(np.ceil(np.log(MC_HI / MC_LO) / np.log(1 + 0.0001))) + 1
    b_ref = int(np.ceil(np.log(MC_HI / MC_LO) / np.log(1 + 2 * 0.005))) + 1
    ram = b_ref * BYTES_PER_TEMPLATE + B_t * N_CHUNK * worst_taps * 16
    out["repriced"] = {"B_target": B_t, "B_ref": b_ref, "ram_gb": ram / 1e9,
                       "ram_direct_gb": B_t * BYTES_PER_TEMPLATE / 1e9,
                       "ram_win": (B_t * BYTES_PER_TEMPLATE) / ram}
    worst_rel = max(r["rel_err"] for r in out["rows"])
    out["C2_statistic_ok"] = bool(worst_rel < 0.01)
    out["fits_in_ram"] = bool(ram / 1e9 < 32)
    print(f"\nC1 per-chunk taps at n=8: {worst_taps} (vs {n1_taps} for n=1) -> total kernel storage is "
          f"{N_CHUNK*worst_taps/n1_taps:.2f}x the n=1 estimate")
    print(f"C2 statistic reproduced: worst relative newSNR error {worst_rel:.5f} -> "
          f"{'PASS -- ratio bank is a drop-in for bank_dense' if out['C2_statistic_ok'] else 'FAIL -- do not build'}")
    print(f"\nRE-PRICED at 0.01% spacing: {B_t} targets + {b_ref} references -> "
          f"{ram/1e9:.1f} GB (direct {B_t*BYTES_PER_TEMPLATE/1e9:.0f} GB, {out['repriced']['ram_win']:.0f}x), "
          f"{'FITS' if out['fits_in_ram'] else 'DOES NOT FIT'}")
    (C.RESULTS_DIR / "bank_ratio_chunked.json").write_text(json.dumps(out, indent=2))
    print("wrote bank_ratio_chunked.json")


if __name__ == "__main__":
    main()
