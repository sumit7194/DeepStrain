"""Is the ratio filter's gain LOOP ORDER rather than kernel length? The L1 benchmark, pre-registered.

WHY. L1 closed as a negative on a TIMED 0.94x -- real `oaconvolve`, 6.005 s vs 5.192 s at N=16,711,680,
K=16,385. Then a production O4a search built a 25-MILLION-template subsolar bank and credited the same
method family with 8x per core. Our *ceiling* was the modelled part: log N / log K gives 2.4x at K=1024 and
3.0x at K=256 and needs K=8 taps to reach 8x, so a model that cannot hit the target at ANY parameter value
was used to conclude the target does not apply to us.

THE HYPOTHESIS, and it is not about K. Both our paths stream the whole reference correlation PER TEMPLATE,
so both are memory-bound and ~1x is self-consistent at any kernel length. The production gain must be loop
order: hold a cache-sized block of the reference series resident and sweep MANY templates' kernels across it,
paying the memory stream once per block per pass instead of once per template. With 25M templates that is the
whole game, and our benchmark paid the full stream per template.

    (A) direct    per template, full-length FFT matched filter -- what ratio filtering replaces
    (B) streaming per template, oaconvolve over the full-length series -- what the 0.94x measured
    (C) blocked   outer loop over cache-sized blocks, inner loop over templates, overlap-save

Identical arithmetic in (B) and (C); only the memory access order differs. Bar (RESULTS.md): C < 0.8*B at
B=64 AND C/B falling monotonically with batch size, because a constant offset is not amortisation and only
amortisation scales to 25M templates.

Run:  .venv/bin/python scripts/bank_ratio_blocked.py [--n 4194304] [--taps 16385]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.signal import oaconvolve

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pbh import config as C

OUT = C.RESULTS_DIR / "bank_ratio_blocked.json"
BATCHES = (1, 4, 16, 64)
DTYPE = np.complex64          # halves memory traffic vs complex128; the effect under test IS traffic


def direct_one(d_fft, a, n):
    """Full-length FFT matched filter for one template. The data FFT is shared across templates, so the
    per-template cost is one forward transform of the template plus the product plus one inverse."""
    A = np.fft.fft(a, n=n)
    return np.fft.ifft(d_fft * np.conj(A))


def streaming(c_ref, kernels):
    """(B) one full-length overlap-add pass per template -- 2 x len(c_ref) x itemsize of traffic each."""
    return [oaconvolve(c_ref, k, mode="same") for k in kernels]


def blocked(c_ref, kernels, block):
    """(C) same convolutions, outer loop over blocks so the block stays resident across all templates.

    Overlap-SAVE: each block is extended by K-1 samples of its predecessor so the 'same'-mode output of the
    interior is exact; the halo is the only redundant work and is (K-1)/block of the total.
    """
    n, K = len(c_ref), len(kernels[0])
    halo = K - 1
    out = [np.empty(n, dtype=c_ref.dtype) for _ in kernels]
    for start in range(0, n, block):
        stop = min(start + block, n)
        lo = max(0, start - halo)
        hi = min(n, stop + halo)
        seg = c_ref[lo:hi]                      # <- resident for the whole inner loop
        for j, k in enumerate(kernels):
            full = oaconvolve(seg, k, mode="same")
            out[j][start:stop] = full[start - lo: start - lo + (stop - start)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4_194_304)
    ap.add_argument("--taps", type=int, default=16385)
    ap.add_argument("--block", type=int, default=1 << 19)      # 524,288 x 8 B = 4 MB, an L2 block
    args = ap.parse_args()
    n, K, block = args.n, args.taps, args.block

    rng = np.random.default_rng(20260921)
    c_ref = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(DTYPE)
    kernels = [(rng.standard_normal(K) + 1j * rng.standard_normal(K)).astype(DTYPE)
               for _ in range(max(BATCHES))]
    d = rng.standard_normal(n).astype(np.float32)
    d_fft = np.fft.fft(d, n=n)

    print(f"N={n:,}  K={K:,}  block={block:,} ({block * c_ref.itemsize / 2**20:.1f} MB)  "
          f"series={n * c_ref.itemsize / 2**20:.0f} MB  dtype={DTYPE.__name__}")

    # ---- correctness gate BEFORE any timing is reported -------------------------------------------------
    ref = streaming(c_ref, kernels[:2])
    got = blocked(c_ref, kernels[:2], block)
    err = max(float(np.max(np.abs(r - g)) / np.max(np.abs(r))) for r, g in zip(ref, got))
    print(f"correctness: blocked vs streaming  max|d|/max|c| = {err:.2e}  "
          f"{'PASS' if err < 1e-6 else 'FAIL'}")
    if err >= 1e-6:
        (OUT).write_text(json.dumps({"correctness_error": err, "timed": False,
                                     "verdict": "blocked path does not reproduce streaming; no timing "
                                                "reported, per the pre-registration"}, indent=2))
        raise SystemExit("correctness gate FAILED -- a faster path that computes something else is not a result")
    del ref, got

    rows = {}
    print(f"\n{'B':>4} {'direct/tmpl':>12} {'stream/tmpl':>12} {'blocked/tmpl':>13} {'C/B':>7} {'B/C':>7}")
    for B in BATCHES:
        ks = kernels[:B]
        t = time.perf_counter(); [direct_one(d_fft, np.pad(k, (0, n - K)), n) for k in ks]
        t_dir = (time.perf_counter() - t) / B
        t = time.perf_counter(); streaming(c_ref, ks); t_str = (time.perf_counter() - t) / B
        t = time.perf_counter(); blocked(c_ref, ks, block); t_blk = (time.perf_counter() - t) / B
        rows[B] = {"direct_s": t_dir, "streaming_s": t_str, "blocked_s": t_blk,
                   "blocked_over_streaming": t_blk / t_str, "speedup": t_str / t_blk}
        print(f"{B:>4} {t_dir:12.4f} {t_str:12.4f} {t_blk:13.4f} {t_blk/t_str:7.3f} {t_str/t_blk:7.2f}")

    ratios = [rows[B]["blocked_over_streaming"] for B in BATCHES]
    falling = all(ratios[i] >= ratios[i + 1] - 1e-3 for i in range(len(ratios) - 1))
    wins = rows[max(BATCHES)]["blocked_over_streaming"] < 0.8
    verdict = ("BLOCKING WINS AND AMORTISES -- loop order is the mechanism; L1's negative was a property of "
               "our access pattern, not of the method" if (wins and falling) else
               "constant-factor win only -- blocking helps but does NOT amortise with batch size, so it does "
               "not scale to 25M templates the way the production gain must" if wins else
               "BLOCKING DOES NOT WIN -- L1 closes again, this time on a measurement of the mechanism rather "
               "than on a cost model, and the published 8x is unexplained by anything we can reproduce here")
    res = {"n": n, "taps": K, "block": block, "dtype": DTYPE.__name__, "correctness_error": err,
           "batches": list(BATCHES), "rows": {str(k): v for k, v in rows.items()},
           "ratio_falls_with_batch": bool(falling), "wins_at_max_batch": bool(wins), "verdict": verdict}
    print(f"\nratios C/B across B: {[round(r, 3) for r in ratios]}  falling={falling}")
    print(f"VERDICT: {verdict}")
    OUT.write_text(json.dumps(res, indent=2))
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
