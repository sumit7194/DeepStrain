"""L1 VERDICT: priced against the real pipeline, ratio-filter dechirping does NOT help subsolar. 1.1x.

This supersedes bank_ratio_costmodel.py, which was WRONG. That script timed the matched filter on a single
262,144-sample CHUNK (7.8 ms) and concluded waveform generation (442 ms) was 56x the filtering, hence a 36x
win from not regenerating templates. But bank_dense.py's expensive step is `so.segment_stats`, which
correlates each template's 8 chunks across the ENTIRE ~16.7M-sample segment -- 64x more data per correlation.
Measured properly, generation is ~10% of the per-template cost, not 98%.

WHY THE METHOD DOES NOT TRANSFER -- the mechanism, which is the actual result here. Ratio filtering converts
an O(N log N) FFT into an O(N log K) block convolution, so the gain is roughly log N / log K. The published
8x rests on K ~ 250 taps for BNS. Subsolar needs **K ~ 16,385** (measured: bank_ratio_regime.py, to hit 1%
statistic accuracy) because these inspirals accumulate enormous orbital phase, so the same fractional
chirp-mass offset shifts the phase far more. With N = 16.7M and K = 16,385, log2(N)/log2(2K) = 24/15 = 1.6x
in theory -- and 1.1x measured. **The benefit is inversely tied to the kernel length the signal class
demands, and subsolar demands the longest kernels.** That is not a defect of their method; it is our regime
falling outside where it pays.

WHAT SURVIVES. The algebra is exact (untruncated kernel reproduces the matched filter to 1.000000) and
kernels really are ~31x smaller than stored analytic chunks. But memory was never the binding constraint:
bank_dense.py already worked around it by going template-major (generate, score, free). The binding
constraint is COMPUTE TIME, and ratio filtering does not reduce it. So the dense-bank wall stands -- now for
a properly understood reason rather than an unexamined cost assumption.

Run:  .venv/bin/python scripts/bank_ratio_realcost.py
"""
import importlib.util
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.signal import oaconvolve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pbh import config as C
from pbh import ratiofilter as rf
from pbh.data import whiten_segment
from pbh.waveforms import make_whitened_injection, sample_params

HERE = Path(__file__).resolve().parent
_so = importlib.util.spec_from_file_location("so", HERE / "semicoherent_oracle.py")
so = importlib.util.module_from_spec(_so); _so.loader.exec_module(so)

WIN, CROP = so.WIN, so.CROP
N_CHUNK, EQ, MC = 8, 2.0 ** 0.2, 0.30
TAPS = 16385                 # what bank_ratio_regime.py showed is needed for <1% statistic error
MC_LO, MC_HI = 0.173, 0.871
N_SEG = 6


def main() -> None:
    gps = json.loads((C.DATA_DIR / "manifest.json").read_text())["H1"]["test"][0]
    w, t0, psd = whiten_segment("H1", gps)
    wc = w[CROP:-CROP]
    base = sample_params(np.random.default_rng(0))

    h, _ = make_whitened_injection(replace(base, mass1=MC * EQ, mass2=MC * EQ), "H1", t0, psd)
    g = h[-WIN:].copy(); g = g if len(g) == WIN else np.pad(g, (WIN - len(g), 0))
    ch = so.analytic_chunks(g, N_CHUNK)
    N = len(wc)
    print(f"segment {gps}: N = {N} samples ({N/C.SAMPLE_RATE:.0f} s), {len(ch)} chunks, {TAPS} taps\n")

    t = time.time(); so.segment_stats(wc, ch); t_seg = time.time() - t
    t = time.time()
    make_whitened_injection(replace(base, mass1=MC * 1.01 * EQ, mass2=MC * 1.01 * EQ), "H1", t0, psd)
    t_gen = time.time() - t
    c_ref = rf.corr_series(wc, np.pad(ch[0][1], (0, N - len(ch[0][1]))))
    rng = np.random.default_rng(0)
    taps = rng.standard_normal(TAPS) + 1j * rng.standard_normal(TAPS)
    t = time.time(); oaconvolve(c_ref, taps, mode="same"); t_fir1 = time.time() - t
    t_fir = len(ch) * t_fir1

    direct = t_seg + t_gen
    speedup = direct / t_fir
    print(f"DIRECT  segment_stats {t_seg:5.1f} s + generation {t_gen:4.2f} s = {direct:5.1f} s / template / segment")
    print(f"RATIO   {len(ch)} x oaconvolve({TAPS} taps) = {t_fir:5.1f} s / template / segment")
    print(f"        -> speedup {speedup:.2f}x   (generation is only {100*t_gen/direct:.0f}% of the direct cost)\n")

    # the asymptotic explanation, so the number is understood rather than merely recorded
    theory = np.log2(N) / np.log2(2 * TAPS)
    print(f"MECHANISM: O(N log N) -> O(N log K) predicts ~log2({N})/log2(2*{TAPS}) = "
          f"{np.log2(N):.0f}/{np.log2(2*TAPS):.0f} = {theory:.1f}x ceiling; measured {speedup:.2f}x.")
    print(f"  The published 8x assumes K ~ 250 (BNS). Subsolar needs K = {TAPS} -> the advantage vanishes.")

    print(f"\nWALL-CLOCK for the dense bank ({N_SEG} segments):")
    print(f"{'spacing':>9} {'B':>7} {'direct':>12} {'ratio':>12}")
    plans = []
    for spacing in (0.001, 0.0005, 0.0001):
        B = int(np.ceil(np.log(MC_HI / MC_LO) / np.log(1 + spacing))) + 1
        hd, hr = B * direct * N_SEG / 3600, B * t_fir * N_SEG / 3600
        plans.append({"spacing": spacing, "B": B, "direct_h": hd, "ratio_h": hr})
        print(f"{spacing:>9.4f} {B:>7} {hd:>10.1f} h {hr:>10.1f} h")

    out = {"segment_samples": int(N), "taps": TAPS, "n_chunk": len(ch),
           "t_segment_stats_s": t_seg, "t_generation_s": t_gen, "t_ratio_s": t_fir,
           "speedup": speedup, "generation_frac_of_direct": t_gen / direct,
           "theory_ceiling": float(theory), "plans": plans,
           "helps": bool(speedup > 2.0),
           "supersedes": "bank_ratio_costmodel.json (timed correlation on a 262k CHUNK, not the 16.7M SEGMENT)",
           "verdict": ("NEGATIVE for subsolar. The algebra is exact and kernels are ~31x smaller than stored "
                       "analytic chunks, but memory was never the binding constraint (bank_dense.py already "
                       "went template-major to work around it) -- compute time is, and ratio filtering does "
                       "not reduce it. The gain scales as log N / log K, and subsolar's enormous phase "
                       "accumulation forces K ~ 16,385 taps instead of the paper's ~250, which is exactly why "
                       "the published 8x does not transfer.")}
    print(f"\nL1 VERDICT: {'HELPS' if out['helps'] else 'DOES NOT HELP'} -- speedup {speedup:.2f}x. "
          f"Dense bank stays blocked, now for a understood reason.")
    (C.RESULTS_DIR / "bank_ratio_realcost.json").write_text(json.dumps(out, indent=2))
    print("wrote bank_ratio_realcost.json")


if __name__ == "__main__":
    main()
