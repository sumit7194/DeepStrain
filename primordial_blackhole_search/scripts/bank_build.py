"""Follow-up A2: build the dense subsolar template bank + MEASURE its coverage (validate the wall).

The parked claim is "subsolar needs <=0.1% Mc spacing (~1,650 templates) -> intractable". A2 tests that
EMPIRICALLY on our actual 64-s / whitened setup: build equal-mass banks at several Mc spacings and measure
the FITTING FACTOR (best normalized match over the bank, time+phase maximized) against injections drawn from
the FULL (m1,m2) prior -- so the injections vary in BOTH Mc and mass-ratio eta while the bank is eta=0.25.
Two questions answered:
  (1) what Mc spacing is actually required for FF >= 0.97 (the standard bank-loss threshold)?
  (2) does an equal-mass (eta=0.25) bank cover unequal-mass injections, or is a 2-D (Mc, eta) bank needed?

Output feeds A3 (build the chosen-spacing bank once, cache to disk for the noise scan).

Run:  .venv/bin/python scripts/bank_build.py [--n-inj 300] [--spacings 0.001 0.003 0.01 0.03]
"""
import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pbh import config as C
from pbh.bankmf import BankMF, template_norm
from pbh.data import whiten_segment
from pbh.waveforms import make_whitened_injection, sample_params

WIN = 64 * C.SAMPLE_RATE                      # 64-s templates: the CNN window + the eval's in-window SNR convention
MC_LO, MC_HI = 0.173, 0.871                   # equal-mass Mc endpoints (m in [0.2,1.0] -> Mc = m/2^0.2)
EQ = 2.0 ** 0.2                               # m -> Mc factor for equal mass


def build_bank(t0, psd, spacing: float):
    """Equal-mass templates geomspaced in Mc at fractional `spacing`; each = last 64 s of the whitened chirp."""
    base = sample_params(np.random.default_rng(0))
    n = int(np.ceil(np.log(MC_HI / MC_LO) / np.log(1 + spacing))) + 1
    mcs = np.geomspace(MC_LO, MC_HI, n)
    bank = []
    for mc in mcs:
        m = float(mc * EQ)                    # equal-mass component that yields this Mc
        h, _ = make_whitened_injection(replace(base, mass1=m, mass2=m), "H1", t0, psd)
        g = h[-WIN:].copy()
        if len(g) < WIN:
            g = np.pad(g, (WIN - len(g), 0))
        bank.append(g.astype(np.float32))
    return mcs, bank


def fitting_factors(bankmf, injs):
    """FF for each injection = max over bank of <h_inj, h_bank>/(|h_inj||h_bank|) (time+phase maximized)."""
    ffs = []
    for h_inj in injs:
        peaks = bankmf.peaks(h_inj)                   # max_t <h_inj, h_i>/|h_i| per bank template
        ffs.append(float(peaks.max() / template_norm(h_inj)))
    return np.array(ffs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-inj", type=int, default=300)
    ap.add_argument("--spacings", type=float, nargs="+", default=[0.001, 0.003, 0.01, 0.03])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n_inj, args.spacings = 30, [0.003, 0.03]

    manifest = json.loads((C.DATA_DIR / "manifest.json").read_text())
    w, t0, psd = whiten_segment("H1", manifest["H1"]["test"][0])

    # full-prior injection templates (vary Mc AND eta) -> the thing the bank must cover
    rng = np.random.default_rng(C.SEED + 7)
    injs, inj_mc, inj_eta = [], [], []
    for _ in range(args.n_inj):
        p = sample_params(rng)
        h, _ = make_whitened_injection(p, "H1", t0, psd)
        g = h[-WIN:].copy()
        if len(g) < WIN:
            g = np.pad(g, (WIN - len(g), 0))
        injs.append(g.astype(np.float32)); inj_mc.append(p.chirp_mass)
        inj_eta.append(p.mass1 * p.mass2 / (p.mass1 + p.mass2) ** 2)
    inj_mc, inj_eta = np.array(inj_mc), np.array(inj_eta)
    print(f"{args.n_inj} full-prior injections: Mc [{inj_mc.min():.3f},{inj_mc.max():.3f}], "
          f"eta [{inj_eta.min():.3f},{inj_eta.max():.3f}]\n")

    out = {}
    ff_by_spacing = {}
    print(f"{'spacing':>8} {'n_tmpl':>7} {'FF_med':>7} {'FF_p1':>7} {'FF_min':>7} {'frac<0.97':>9}")
    for s in args.spacings:
        t0_ = time.time()
        mcs, bank = build_bank(t0, psd, s)
        ff = fitting_factors(BankMF(bank, WIN), injs)
        ff_by_spacing[s] = ff
        # is any low-FF driven by eta rather than Mc gaps? correlate the loss with distance-from-equal-mass
        lowmask = ff < 0.97
        eta_low = float(inj_eta[lowmask].mean()) if lowmask.any() else float("nan")
        out[f"{s}"] = {
            "n_templates": len(bank), "ff_median": float(np.median(ff)),
            "ff_p1": float(np.percentile(ff, 1)), "ff_min": float(ff.min()),
            "frac_below_0.97": float(lowmask.mean()), "mean_eta_of_lowFF": eta_low,
        }
        print(f"{s:>8.3f} {len(bank):>7} {np.median(ff):>7.3f} {np.percentile(ff,1):>7.3f} "
              f"{ff.min():>7.3f} {lowmask.mean():>9.2f}   ({time.time()-t0_:.0f}s)", flush=True)

    # correlation of FF-loss with eta at the finest spacing (isolates the eta dimension from Mc coverage)
    fine = min(args.spacings)
    ff_fine = ff_by_spacing[fine]
    r_eta = float(np.corrcoef(ff_fine, inj_eta)[0, 1])
    print(f"\nAt finest spacing {fine}: corr(FF, eta) = {r_eta:+.2f} "
          f"({'eta-limited -> 2-D bank needed' if r_eta > 0.3 else 'eta NOT the limit -> 1-D Mc bank sufficient'})")
    out["corr_ff_eta_at_finest"] = r_eta
    out["win_sec"] = 64
    (C.RESULTS_DIR / "bank_build.json").write_text(json.dumps(out, indent=2))
    print("wrote bank_build.json")


if __name__ == "__main__":
    main()
