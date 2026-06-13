"""Track-score aggregation statistics over a window-score series.

Every aggregator maps (scores, mask) -> float, where `scores` is a full
per-segment series on the standard sweep grid (pbh.sweep) and `mask` marks
signal-contaminated windows. With mask=None the statistic ranges over all
positions (used for zero-FA thresholds on pure noise); with a mask it is
restricted to positions touching the contamination, so an injection is never
credited with a detection caused by distant noise.

Registry choices are pre-registered in RESULTS.md (v2 rung 1): `max` is the
control, `boxcar_bank` spans the track-presence lengths of the subsolar mass
range, `count_above` rewards sustained consistency over single-window
amplitude. The bank pays its trials factor inside its own noise threshold.
"""

from __future__ import annotations

import numpy as np

BANK_K = (32, 48, 64, 80, 96)  # rung 1: 256 s windows, redundant overlap
SUM_BANK_K = (2, 3, 4, 6, 8)  # rung 2: short non-overlapping windows, accumulate


def _running(scores: np.ndarray, k: int, mask, reduce: str) -> float:
    """Max over k-window running sum/mean, restricted to spans touching mask."""
    if len(scores) < k:
        return float("-inf")
    kernel = np.ones(k) if reduce == "sum" else np.ones(k) / k
    agg = np.convolve(scores, kernel, mode="valid")  # window j covers [j, j+k)
    if mask is not None:
        idx = np.flatnonzero(mask)
        lo = max(0, idx[0] - k + 1)
        hi = min(len(agg), idx[-1] + 1)
        agg = agg[lo:hi]
        if len(agg) == 0:
            return float("-inf")
    return float(agg.max())


def boxcar(scores: np.ndarray, k: int, mask: np.ndarray | None = None) -> float:
    """Max running mean over k consecutive windows (positions touching mask)."""
    return _running(scores, k, mask, "mean")


def sumcar(scores: np.ndarray, k: int, mask: np.ndarray | None = None) -> float:
    """Max running SUM over k consecutive windows — accumulates independent
    per-window log-odds (rung 2). Different k have different scale, so callers
    threshold per-k against noise."""
    return _running(scores, k, mask, "sum")


def agg_max(scores: np.ndarray, mask: np.ndarray | None = None) -> float:
    s = scores if mask is None else scores[mask]
    return float(s.max())


def agg_boxcar_bank(scores: np.ndarray, mask: np.ndarray | None = None) -> float:
    return max(boxcar(scores, k, mask) for k in BANK_K)


def longest_run(above: np.ndarray, mask: np.ndarray | None = None) -> int:
    """Length of the longest True-run (only runs containing a masked index)."""
    padded = np.concatenate(([False], above, [False]))
    starts = np.flatnonzero(padded[1:] & ~padded[:-1])
    ends = np.flatnonzero(~padded[1:] & padded[:-1])  # exclusive
    if len(starts) == 0:
        return 0
    if mask is not None:
        midx = np.flatnonzero(mask)
        keep = [
            (s, e) for s, e in zip(starts, ends)
            if np.any((midx >= s) & (midx < e))
        ]
        if not keep:
            return 0
        return int(max(e - s for s, e in keep))
    return int((ends - starts).max())


def make_count_above(per_window_thresh: float):
    """count_above bound to a per-window threshold (pooled noise quantile)."""

    def agg(scores: np.ndarray, mask: np.ndarray | None = None) -> float:
        return float(longest_run(scores > per_window_thresh, mask))

    return agg


def make_aggregators(pooled_noise_scores: np.ndarray) -> dict:
    """Rung-1 registry (256 s overlapping windows). count_above bound to q99."""
    q99 = float(np.quantile(pooled_noise_scores, 0.99))
    return {
        "max": agg_max,
        "boxcar_bank": agg_boxcar_bank,
        "count_above": make_count_above(q99),
    }


def sumnorm(scores: np.ndarray, k: int, mask: np.ndarray | None = None) -> float:
    """Summed logits over k windows, normalized by sqrt(k). Under noise the sum
    of k ~iid window scores has std ~sqrt(k), so sqrt(k)-normalizing keeps the
    noise scale ~constant across the bank (one threshold compares members) while
    a real track's sum grows as k -> the normalized statistic grows as sqrt(k):
    genuine SNR accumulation across independent windows."""
    s = sumcar(scores, k, mask)
    return s / np.sqrt(k) if np.isfinite(s) else s


def agg_sum_track(scores: np.ndarray, mask: np.ndarray | None = None) -> float:
    return max(sumnorm(scores, k, mask) for k in SUM_BANK_K)


def make_aggregators_short(pooled_noise_scores: np.ndarray) -> dict:
    """Rung-2 registry (short non-overlapping windows): the accumulator plus the
    single-window control."""
    return {"max": agg_max, "sum_track": agg_sum_track}
