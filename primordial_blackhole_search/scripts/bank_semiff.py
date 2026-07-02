"""Follow-up A2b: SEMI-COHERENT (n=8 newSNR) bank tolerance vs Mc spacing — sets A3's bank density.

A2 measures the COHERENT fitting factor (collapses at any tractable spacing — the 64-s chirp carries ~1e4
cycles; consistent with LVK's 3.45M-template O4 subsolar bank, arXiv:2412.10951). But the tractable-real-
detector candidate is the n=8 SEMI-coherent statistic (8-s chunk coherence, chi2-reweighted): each chunk
tolerates ~8x more dephasing, and the chunk |.| removes inter-chunk phase. This measures, per Mc spacing,
the RECOVERY RATIO = (best-bank-template local newSNR) / (true-template local newSNR) on injections into
real noise — the quantity that scales the detector's SNR50. bank_oracle's recorded 0.000 was at ~2.5%
spacing (64 templates); this finds the spacing where recovery approaches 1, i.e. the density at which a
REAL semi-coherent bank detector matches the semicoherent_oracle ceiling.

Run:  .venv/bin/python scripts/bank_semiff.py [--spacings 0.0025 0.005 0.01 0.02] [--n-inj 50]
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
from pbh.data import whiten_segment
from pbh.waveforms import make_whitened_injection, sample_params

HERE = Path(__file__).resolve().parent
_so = importlib.util.spec_from_file_location("so", HERE / "semicoherent_oracle.py")
so = importlib.util.module_from_spec(_so); _so.loader.exec_module(so)

WIN = 64 * C.SAMPLE_RATE
MC_LO, MC_HI = 0.173, 0.871
EQ = 2.0 ** 0.2
N_CHUNK = 8
TARGET = 25.0


def build_bank_chunks(t0, psd, spacing):
    base = sample_params(np.random.default_rng(0))
    n = int(np.ceil(np.log(MC_HI / MC_LO) / np.log(1 + spacing))) + 1
    mcs = np.geomspace(MC_LO, MC_HI, n)
    out = []
    for mc in mcs:
        h, _ = make_whitened_injection(replace(base, mass1=float(mc * EQ), mass2=float(mc * EQ)), "H1", t0, psd)
        g = h[-WIN:].copy()
        if len(g) < WIN:
            g = np.pad(g, (WIN - len(g), 0))
        out.append((float(mc), so.analytic_chunks(g, N_CHUNK)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spacings", type=float, nargs="+", default=[0.0025, 0.005, 0.01, 0.02])
    ap.add_argument("--n-inj", type=int, default=50)
    ap.add_argument("--n-neighbors", type=int, default=6, help="bank templates each side of the injection Mc")
    args = ap.parse_args()

    manifest = json.loads((C.DATA_DIR / "manifest.json").read_text())
    w, t0, psd = whiten_segment("H1", manifest["H1"]["test"][0])
    crop = C.WHITEN_CROP_SEC * C.SAMPLE_RATE
    wc = w[crop:-crop]

    # injections: EQUAL-MASS (isolate the Mc-spacing axis; eta axis handled by A2's corr) into real noise.
    # local_stats scans +-PAD around a window start INSIDE a segment, so keep each injection in segment
    # context: a (2*PAD)-padded slice around the injected window, with ws = PAD.
    rng = np.random.default_rng(C.SEED + 21)
    PAD = so.PAD
    injections = []
    for _ in range(args.n_inj):
        mc = float(np.exp(rng.uniform(np.log(MC_LO * 1.02), np.log(MC_HI * 0.98))))
        p = replace(sample_params(rng), mass1=mc * EQ, mass2=mc * EQ)
        h, ref = make_whitened_injection(p, "H1", t0, psd)
        sig = h * (TARGET / ref)
        # oracle convention exactly: merger at window END, local scan around ws=PAD
        m = int(rng.integers(max(len(sig), WIN) + PAD, len(wc) - PAD))
        seg = wc[m - WIN - PAD : m + PAD].copy()
        tail = sig[-WIN:] if len(sig) >= WIN else np.pad(sig, (WIN - len(sig), 0))
        seg[PAD : PAD + WIN] += tail
        # truth: local newSNR with the injection's own template
        g = h[-WIN:].copy()
        if len(g) < WIN:
            g = np.pad(g, (WIN - len(g), 0))
        _, true_ns = so.local_stats(seg, so.analytic_chunks(g, N_CHUNK), PAD)
        injections.append((mc, seg, true_ns))
    print(f"{args.n_inj} equal-mass injections at network SNR {TARGET} (true-template newSNR "
          f"median {np.median([t for _,_,t in injections]):.1f})\n")

    out = {}
    print(f"{'spacing':>8} {'n_tmpl':>7} {'rec_med':>8} {'rec_p10':>8} {'rec_min':>8}")
    for s in args.spacings:
        t0_ = time.time()
        bank = build_bank_chunks(t0, psd, s)
        mcs = np.array([m for m, _ in bank])
        recs = []
        for mc, seg, true_ns in injections:
            i = int(np.abs(mcs - mc).argmin())
            lo, hi = max(0, i - args.n_neighbors), min(len(bank), i + args.n_neighbors + 1)
            best = max(so.local_stats(seg, ch, PAD)[1] for _, ch in bank[lo:hi])
            recs.append(best / true_ns)
        recs = np.array(recs)
        out[f"{s}"] = {"n_templates": len(bank), "rec_median": float(np.median(recs)),
                       "rec_p10": float(np.percentile(recs, 10)), "rec_min": float(recs.min())}
        print(f"{s:>8.4f} {len(bank):>7} {np.median(recs):>8.3f} {np.percentile(recs,10):>8.3f} "
              f"{recs.min():>8.3f}   ({time.time()-t0_:.0f}s)", flush=True)

    out.update({"n_chunk": N_CHUNK, "target_snr": TARGET, "note": "recovery = best-bank newSNR / true-template newSNR, local scan"})
    (C.RESULTS_DIR / "bank_semiff.json").write_text(json.dumps(out, indent=2))
    print("wrote bank_semiff.json")


if __name__ == "__main__":
    main()
