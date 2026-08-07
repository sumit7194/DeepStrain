"""O4-3 stress-test: bootstrap confidence intervals on the O4b-vs-O3a reach gain.

The O4-3 campaign quotes a distance gain (1.24x) and volume gain (1.93x) with no uncertainty. With ~65
injections per efficiency point, each point carries ~6% statistical error, which propagates through SNR50
into the ratio -- so the bin-to-bin spread (1.20/1.24/1.29) may be pure noise. This resamples the SAVED
injection rows (no re-running of the 4,800-waveform campaign) to put a 90% CI on every gain, and asks the
only question that matters for the claim: **is the gain significantly > 1?**

Requires o4_sensitive_distance_rows*.parquet (written by o4_sensitive_distance.py).

Run:  .venv/bin/python scripts/o4_reach_bootstrap.py [--tag _matched] [--n-boot 1000]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pbh import config as C
from pbh.metrics import MASS_EDGES, MASS_LABELS

NET_SNR_RANGE = (4.0, 40.0)
SNR_BINS = np.linspace(*NET_SNR_RANGE, 13)


def reach_for(sub: pd.DataFrame, col: str, min_count: int = 10) -> float:
    """Reach in Mpc = <SNR_ref at 1 Mpc> / SNR50, for one mass bin of one era."""
    cen, eff = [], []
    for lo, hi in zip(SNR_BINS[:-1], SNR_BINS[1:]):
        s = sub[(sub.target_snr >= lo) & (sub.target_snr < hi)]
        if len(s) >= min_count:
            cen.append((lo + hi) / 2); eff.append(float(s[col].mean()))
    if len(cen) < 2 or max(eff) < 0.5:
        return np.nan
    snr50 = float(np.interp(0.5, eff, cen))
    return float(sub.snr_ref_net.mean()) / snr50 if snr50 > 0 else np.nan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="artifact suffix, e.g. _matched")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--col", default="det_coinc", help="det_coinc (H1xL1) or det_single")
    args = ap.parse_args()

    path = C.RESULTS_DIR / f"o4_sensitive_distance_rows{args.tag}.parquet"
    if not path.exists():
        print(f"MISSING {path.name} — re-run o4_sensitive_distance.py (it now persists injection rows)")
        return
    df = pd.read_parquet(path)
    rng = np.random.default_rng(C.SEED + 31337)
    print(f"bootstrap on {len(df)} injections ({args.col}), B={args.n_boot}\n")
    print(f"{'mass bin':>12} | {'O3a Mpc':>8} {'O4b Mpc':>8} | {'d gain':>7} {'90% CI':>16} | {'V gain':>7} {'>1?':>4}")

    out = {}
    for lo, hi, lab in zip(MASS_EDGES[:-1], MASS_EDGES[1:], MASS_LABELS):
        m3 = df[(df.era == "O3a") & (df.chirp_mass >= lo) & (df.chirp_mass < hi)]
        m4 = df[(df.era == "O4b") & (df.chirp_mass >= lo) & (df.chirp_mass < hi)]
        d3, d4 = reach_for(m3, args.col), reach_for(m4, args.col)
        gains = []
        for _ in range(args.n_boot):
            b3 = m3.iloc[rng.integers(0, len(m3), len(m3))]
            b4 = m4.iloc[rng.integers(0, len(m4), len(m4))]
            g3, g4 = reach_for(b3, args.col), reach_for(b4, args.col)
            if np.isfinite(g3) and np.isfinite(g4) and g3 > 0:
                gains.append(g4 / g3)
        gains = np.array(gains)
        lo_ci, hi_ci = (float(np.percentile(gains, 5)), float(np.percentile(gains, 95))) if len(gains) else (np.nan, np.nan)
        p_gt1 = float(np.mean(gains > 1.0)) if len(gains) else np.nan
        dgain = d4 / d3 if d3 else np.nan
        out[lab] = {"d_o3a_mpc": d3, "d_o4b_mpc": d4, "d_gain": dgain,
                    "d_gain_ci90": [lo_ci, hi_ci], "v_gain": dgain ** 3 if np.isfinite(dgain) else None,
                    "v_gain_ci90": [lo_ci ** 3, hi_ci ** 3] if np.isfinite(lo_ci) else None,
                    "prob_gain_gt_1": p_gt1, "n_boot_valid": int(len(gains))}
        print(f"{lab:>12} | {d3:>8.2f} {d4:>8.2f} | {dgain:>7.2f} [{lo_ci:>5.2f},{hi_ci:>5.2f}] | "
              f"{dgain**3:>7.2f} {'YES' if lo_ci > 1 else 'no':>4}")

    sig = all(v["d_gain_ci90"][0] > 1.0 for v in out.values() if np.isfinite(v["d_gain_ci90"][0]))
    allbins = [v["d_gain"] for v in out.values() if np.isfinite(v["d_gain"])]
    overlap = max(v["d_gain_ci90"][0] for v in out.values()) < min(v["d_gain_ci90"][1] for v in out.values())
    print(f"\nmean distance gain {np.mean(allbins):.2f}x | every bin's 90% CI excludes 1: {sig}")
    print(f"bin-to-bin spread ({min(allbins):.2f}-{max(allbins):.2f}) consistent with noise "
          f"(all CIs mutually overlap): {overlap}")
    out["_summary"] = {"mean_d_gain": float(np.mean(allbins)), "all_bins_significant": bool(sig),
                       "bin_spread_consistent_with_noise": bool(overlap), "col": args.col,
                       "n_boot": args.n_boot, "tag": args.tag}
    (C.RESULTS_DIR / f"o4_reach_bootstrap{args.tag}.json").write_text(json.dumps(out, indent=2))
    print(f"wrote o4_reach_bootstrap{args.tag}.json")


if __name__ == "__main__":
    main()
