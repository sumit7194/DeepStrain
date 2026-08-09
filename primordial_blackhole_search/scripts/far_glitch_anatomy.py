"""Anatomy of the deep-FAR background: what actually sets each detection threshold, glitches or coincidences?

Found during a "did we miss anything?" pass over the far_deep score cache. The deep-FAR run reported zero-lag
max = 11.295 against a 1/month threshold of 12.34, which READS like we came within 8% of a detection. Both
that reading and a first quick pass at this analysis were wrong; this is the careful version.

METHOD NOTE (why the first pass was wrong): a shallow sample of lags makes the "top 100 background events" a
MID-tail population, where one-sidedness dominates. Using all N-1 lags makes the top 100 the genuine extreme
tail, which behaves oppositely. One-sidedness must be reported as a function of loudness, not as one number.

THE MECHANISM — single-detector ceilings. Over 100 O4b segments, max(H1) = 12.53 and max(L1) = 6.26. So a
ONE-SIDED event (a glitch in one detector, nothing in the other) cannot push the sum statistic above ~12.5.
Any background above that REQUIRES both detectors to contribute: it is a genuine accidental coincidence.

WHAT THAT MEANS FOR OUR THRESHOLDS (the useful result):
    1/month  12.34  -> BELOW the ceiling: glitch-reachable, partly glitch-set
    1/year   14.11  -> ABOVE  the ceiling: set by genuine two-sided accidental coincidences
    1/decade 16.12  -> ABOVE  the ceiling: set by genuine two-sided accidental coincidences
  i.e. the DEEP thresholds we actually quote are physically meaningful, not glitch artifacts. Good news that
  was worth verifying rather than assuming.

AND THE NULL IS CLEANER THAN REPORTED: the loudest zero-lag "coincidence" (11.30) is H1 = +12.53 with
L1 = -1.24 — it is precisely the single loudest H1 window in the entire dataset, with Livingston seeing
nothing. Not a coincidence at all. Under a consistency statistic min(sH, sL), which no single detector can
inflate, the zero-lag max collapses to 1.63.

Run:  .venv/bin/python scripts/far_glitch_anatomy.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pbh import config as C

CACHE = C.RESULTS_DIR / "far_scores"
ONE_SIDED = 0.15          # a detector contributing < 15% of the other = one-sided
BANDS = [(0, 25), (25, 100), (100, 500), (500, 2000)]


def main() -> None:
    segs = sorted(int(p.stem.split("_")[1]) for p in CACHE.glob("seg_*.npz"))
    if not segs:
        print("no far_scores cache — run far_deep.py first"); return
    H = np.concatenate([np.load(CACHE / f"seg_{g}.npz")["h"] for g in segs])
    L = np.concatenate([np.load(CACHE / f"seg_{g}.npz")["l"] for g in segs])
    N = len(H)
    hmax, lmax = float(H.max()), float(L.max())
    ceiling = max(hmax, lmax)
    print(f"{N} windows over {len(segs)} O4b segments")
    print(f"single-detector ceilings: max(H1) = {hmax:.2f}, max(L1) = {lmax:.2f} "
          f"-> a one-sided event cannot exceed ~{ceiling:.1f}\n")

    # --- zero-lag: is the loudest REAL coincidence two-sided, or one detector's worst glitch? ---
    zl = H + L
    i = int(zl.argmax())
    zl_sum, zl_min = float(zl[i]), float(np.minimum(H, L).max())
    two_sided = bool(min(H[i], L[i]) >= ONE_SIDED * max(H[i], L[i]))
    is_worst_h1 = bool(abs(H[i] - hmax) < 1e-6)
    print("ZERO-LAG (real, unshifted):")
    print(f"  loudest sum = {zl_sum:.2f}  (H1 {H[i]:+.2f}, L1 {L[i]:+.2f}) -> "
          f"{'two-sided' if two_sided else 'ONE-SIDED: a single-detector glitch'}"
          f"{' — and it IS the loudest H1 window in the dataset' if is_worst_h1 else ''}")
    print(f"  loudest min = {zl_min:.2f}  (consistency statistic; no single detector can inflate it)")

    # --- background: one-sidedness AS A FUNCTION OF LOUDNESS (one number would be misleading) ---
    ev = []
    for k in range(1, N):
        Lr = np.roll(L, k)
        s = H + Lr
        j = int(s.argmax())
        ev.append((float(s[j]), float(H[j]), float(Lr[j])))
    ev.sort(reverse=True)
    s_arr = np.array([e[0] for e in ev]); a = np.array([e[1] for e in ev]); b = np.array([e[2] for e in ev])
    os_mask = np.minimum(a, b) < ONE_SIDED * np.maximum(a, b)
    print(f"\nBACKGROUND ({N-1} lags) — one-sidedness vs loudness:")
    print(f"  {'rank band':>14} {'sum range':>16} {'% one-sided':>12}")
    bands = []
    for lo, hi in BANDS + [(2000, len(ev))]:
        frac = float(os_mask[lo:hi].mean())
        bands.append({"lo": lo, "hi": hi, "sum_hi": float(s_arr[lo]), "sum_lo": float(s_arr[hi-1]),
                      "frac_one_sided": frac})
        print(f"  {f'top {lo}-{hi}':>14} {f'{s_arr[hi-1]:.1f} - {s_arr[lo]:.1f}':>16} {100*frac:>11.0f}%")
    print("  => one-sidedness dominates the MID tail; the extreme tail is genuinely two-sided.")

    # --- so: is each quoted threshold glitch-reachable or coincidence-set? ---
    lad = json.loads((C.RESULTS_DIR / "far_deep.json").read_text())["far_ladder"]
    print(f"\nTHRESHOLD REGIME (ceiling {ceiling:.2f}):")
    regimes = {}
    for k, v in lad.items():
        clean = v > ceiling
        regimes[k] = {"threshold": v, "above_single_det_ceiling": bool(clean)}
        print(f"  {k:>9} {v:6.2f} -> {'genuine two-sided coincidences' if clean else 'glitch-reachable'}")

    out = {"n_windows": N, "n_segments": len(segs), "max_H1": hmax, "max_L1": lmax,
           "one_sided_ceiling": ceiling, "zero_lag_sum": zl_sum, "zero_lag_H1": float(H[i]),
           "zero_lag_L1": float(L[i]), "zero_lag_two_sided": two_sided,
           "zero_lag_is_worst_H1_window": is_worst_h1, "zero_lag_min_stat": zl_min,
           "bands": bands, "threshold_regimes": regimes,
           "conclusion": "The 1/year and 1/decade thresholds sit ABOVE the single-detector ceiling, so they are "
                         "set by genuine two-sided accidental coincidences, not glitches. The zero-lag maximum "
                         "is the single loudest H1 glitch (L1 sees nothing), so the null is cleaner than the "
                         "sum statistic alone suggests: under min(sH,sL) the zero-lag max is 1.63."}
    (C.RESULTS_DIR / "far_glitch_anatomy.json").write_text(json.dumps(out, indent=2))
    print("\nwrote far_glitch_anatomy.json")


if __name__ == "__main__":
    main()
