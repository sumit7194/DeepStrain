"""Is the ratio bank's residual error a SIGNAL-accuracy problem or a noise-max sampling effect?

bank_ratio_chunked.py (after the correlation-convention fix) reproduces the n=8 semi-coherent newSNR to 0.2%
where there IS signal, but missed the pre-registered <1% bar at one separation with 3.9%. Hypothesis: that
separation's newSNR (~5.8) is not a signal at all -- the template is too far from the injection to recover it,
so the statistic is the LOCAL NOISE MAXIMUM over a 4,097-point scan, and a tiny reconstruction error simply
changes which noise sample wins the max. Where signal dominates (newSNR 15.5) the agreement was 0.2%.

That is a plausible story and a single trial per separation is far too thin to assert it. It also matters:
bank_dense sets its threshold from the noise maximum, so if noise maxima are systematically mis-reproduced we
would be importing a real bias, not a harmless jitter.

WHAT THIS SEPARATES, over many independent noise windows:
  R1  SIGNAL REGIME  -- injection recovered by a near-matched template. Error here is what would corrupt
      detection efficiency. Pre-registered bar: median |rel err| < 1%.
  R2  NOISE REGIME   -- pure noise, no injection. Error here shifts the threshold. Two distinct questions:
      is it BIASED (a systematic shift in the mean, which would move the threshold), or merely NOISY (a
      symmetric jitter in which sample wins, which largely averages out over a segment-wide max)?
      Pre-registered bars: |median relative bias| < 1%, and the sign of the error roughly balanced.

A biased noise regime is the dangerous outcome and would block the build; a noisy-but-unbiased one is the
benign case the hypothesis predicts.

Run:  .venv/bin/python scripts/bank_ratio_regime.py [--n-trial 40]
"""
import argparse
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
N_CHUNK, EQ, MC_REF = 8, 2.0 ** 0.2, 0.30
SEP = 0.005                # the separation that missed the bar
TAPS = 8193


def stats_pair(d, ws, pairs, taps_cache):
    """(direct, ratio) newSNR for one window, using cached kernels."""
    _, ns_direct = so.local_stats(d, [(o, a, p) for o, a, p, _ in pairs], ws)
    A = np.zeros(2 * PAD + 1); B = np.zeros(2 * PAD + 1)
    for (off, a_t, p_t, a_r), k in zip(pairs, taps_cache):
        seg = d[ws + off - PAD: ws + off - PAD + len(a_t) + 2 * PAD]
        x = (np.abs(rf.apply_taps(rf.corr_series(seg, a_r), k)) ** 2)[0:2 * PAD + 1]
        A += x; B += x ** 2 / p_t
    return float(ns_direct), float(so.newsnr(A, B, len(pairs)).max())


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--n-trial", type=int, default=40)
    ap.add_argument("--taps", type=int, default=TAPS)
    ap.add_argument("--signal-only", action="store_true")
    args = ap.parse_args()
    gps = json.loads((C.DATA_DIR / "manifest.json").read_text())["H1"]["test"][0]
    w, t0, psd = whiten_segment("H1", gps)
    wc = w[CROP:-CROP]
    base = sample_params(np.random.default_rng(0))
    rng = np.random.default_rng(C.SEED + 31337)

    def templ(mc):
        h, _ = make_whitened_injection(replace(base, mass1=float(mc * EQ), mass2=float(mc * EQ)), "H1", t0, psd)
        g = h[-WIN:].copy()
        return g if len(g) == WIN else np.pad(g, (WIN - len(g), 0))

    ch_ref = so.analytic_chunks(templ(MC_REF), N_CHUNK)
    ch_t = so.analytic_chunks(templ(MC_REF * (1 + SEP)), N_CHUNK)
    pairs = [(o, a, p, m[0][1]) for o, a, p in ch_t if (m := [c for c in ch_ref if c[0] == o])]
    seg_len = len(pairs[0][1]) + 2 * PAD
    taps_cache = [rf.ratio_taps(a_t, a_r, args.taps, n_fft=seg_len) for _, a_t, _, a_r in pairs]
    print(f"{len(pairs)} chunk pairs | {args.taps} taps | separation {SEP*100:.1f}% | {args.n_trial} trials/regime\n")

    # injection at the TARGET's own mass, so the signal regime really is signal-dominated
    h_inj, snr_ref = make_whitened_injection(
        replace(base, mass1=MC_REF * (1 + SEP) * EQ, mass2=MC_REF * (1 + SEP) * EQ), "H1", t0, psd)

    out = {"sep": SEP, "taps": args.taps, "n_trial": args.n_trial, "regimes": {}}
    for regime in (("signal",) if args.signal_only else ("signal", "noise")):
        errs, direct_vals = [], []
        for _ in range(args.n_trial):
            ws = int(rng.integers(PAD + WIN, len(wc) - WIN - PAD))
            d = wc.astype(np.float64).copy()
            if regime == "signal":
                d[ws:ws + WIN] += (h_inj * (25.0 / snr_ref))[-WIN:]
            a, b = stats_pair(d, ws, pairs, taps_cache)
            errs.append((b - a) / max(a, 1e-12)); direct_vals.append(a)
        e = np.array(errs)
        out["regimes"][regime] = {
            "median_newsnr_direct": float(np.median(direct_vals)),
            "median_rel_err": float(np.median(e)), "median_abs_rel_err": float(np.median(np.abs(e))),
            "p90_abs_rel_err": float(np.percentile(np.abs(e), 90)),
            "frac_positive": float((e > 0).mean())}
        r = out["regimes"][regime]
        print(f"{regime.upper():>7}: median newSNR {r['median_newsnr_direct']:6.2f} | "
              f"median rel err {r['median_rel_err']:+.4f} | median |rel err| {r['median_abs_rel_err']:.4f} | "
              f"p90 {r['p90_abs_rel_err']:.4f} | {100*r['frac_positive']:.0f}% positive")

    if args.signal_only:
        (C.RESULTS_DIR / f"bank_ratio_regime_t{args.taps}.json").write_text(json.dumps(out, indent=2)); return
    sig, noi = out["regimes"]["signal"], out["regimes"]["noise"]
    out["R1_signal_ok"] = bool(sig["median_abs_rel_err"] < 0.01)
    out["R2_noise_unbiased"] = bool(abs(noi["median_rel_err"]) < 0.01 and 0.3 < noi["frac_positive"] < 0.7)
    out["hypothesis_confirmed"] = bool(out["R1_signal_ok"] and out["R2_noise_unbiased"])
    print(f"\nR1 SIGNAL  median |rel err| {sig['median_abs_rel_err']:.4f} -> "
          f"{'PASS (detection efficiency safe)' if out['R1_signal_ok'] else 'FAIL'}")
    print(f"R2 NOISE   median bias {noi['median_rel_err']:+.4f}, {100*noi['frac_positive']:.0f}% positive -> "
          f"{'PASS (jitter, not bias -- threshold safe)' if out['R2_noise_unbiased'] else 'FAIL (biased -> would shift the threshold)'}")
    print(f"\nVERDICT: {'hypothesis CONFIRMED -- the 3.9% was noise-max sampling; safe to build' if out['hypothesis_confirmed'] else 'hypothesis REJECTED -- real accuracy problem, do not build'}")
    (C.RESULTS_DIR / "bank_ratio_regime.json").write_text(json.dumps(out, indent=2))
    print("wrote bank_ratio_regime.json")


if __name__ == "__main__":
    main()
