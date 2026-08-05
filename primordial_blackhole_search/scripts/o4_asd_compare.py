"""Per-detector ASD across eras — the measured mechanism behind the N5 Virgo result.

N5 found Virgo does NOT help subsolar triple-coincidence, in BOTH O3a and O4b, because V1's signal
responsiveness is ~12-19% of H1/L1. This measures WHY, from our own cached strain rather than quoted
detector ranges: the median amplitude spectral density in the subsolar band [50, 300] Hz per detector,
per era. Injections are scaled to a fixed NETWORK SNR and each detector's share is set by its own PSD,
so a detector 3x louder than LIGO contributes ~1/3 the amplitude SNR and cannot carry the signal.

Run:  .venv/bin/python scripts/o4_asd_compare.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pbh import config as C
from pbh.data import whiten_segment

BAND = (50.0, 300.0)
N_SEG = 3


def median_asd(det, segs):
    vals = []
    for g in segs:
        try:
            _, _, psd = whiten_segment(det, g)
            f = np.asarray(psd.sample_frequencies); v = np.asarray(psd)
            m = (f >= BAND[0]) & (f <= BAND[1])
            vals.append(float(np.sqrt(np.median(v[m]))))
        except Exception:
            continue
    return float(np.median(vals)) if vals else float("nan")


def main() -> None:
    eras = {
        "O3a": json.loads((C.DATA_DIR / "triple_segs.json").read_text())[:N_SEG],
        "O4b": json.loads((C.DATA_DIR / "o4b_triple_segs.json").read_text())[:N_SEG],
    }
    out = {"band_hz": list(BAND), "n_segments_per_era": N_SEG, "eras": {}}
    print(f"Median ASD in [{BAND[0]:.0f}, {BAND[1]:.0f}] Hz (lower = more sensitive)\n")
    for era, segs in eras.items():
        asd = {d: median_asd(d, segs) for d in ("H1", "L1", "V1")}
        best_ligo = min(asd["H1"], asd["L1"])
        ratio = asd["V1"] / best_ligo
        out["eras"][era] = {"asd": asd, "v1_over_best_ligo": ratio}
        print(f"  {era}: H1 {asd['H1']:.2e}  L1 {asd['L1']:.2e}  V1 {asd['V1']:.2e}"
              f"   -> V1 is {ratio:.1f}x louder than the best LIGO detector")

    o3, o4 = out["eras"]["O3a"], out["eras"]["O4b"]
    ligo_gain = min(o3["asd"]["H1"], o3["asd"]["L1"]) / min(o4["asd"]["H1"], o4["asd"]["L1"])
    virgo_gain = o3["asd"]["V1"] / o4["asd"]["V1"]
    out["ligo_improvement_o3a_to_o4b"] = ligo_gain
    out["virgo_improvement_o3a_to_o4b"] = virgo_gain
    out["gap_widened"] = bool(o4["v1_over_best_ligo"] > o3["v1_over_best_ligo"])
    print(f"\n  O3a -> O4b improvement: LIGO {ligo_gain:.2f}x, Virgo {virgo_gain:.2f}x")
    print(f"  => the V1/LIGO gap {'WIDENED' if out['gap_widened'] else 'narrowed'}: "
          f"{o3['v1_over_best_ligo']:.1f}x -> {o4['v1_over_best_ligo']:.1f}x")
    print("  (this is why V1's signal responsiveness fell 19% -> 12% of H1/L1 between the two N5 runs)")
    (C.RESULTS_DIR / "o4_asd_compare.json").write_text(json.dumps(out, indent=2))
    print("\nwrote o4_asd_compare.json")


if __name__ == "__main__":
    main()
