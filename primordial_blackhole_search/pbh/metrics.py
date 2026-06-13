"""Shared sensitivity metrics — the eval convention used by every campaign:
efficiency vs injected SNR per chirp-mass tercile, SNR at 50% efficiency,
sensitive-distance fraction = 8 / snr50 (ideal matched filter = 1.0).
"""

from __future__ import annotations

import numpy as np

from . import config as C

SNR_BINS = np.linspace(*C.EVAL_SNR_RANGE, 11)
MASS_EDGES = (0.17, 0.35, 0.55, 0.88)
MASS_LABELS = ("0.17-0.35", "0.35-0.55", "0.55-0.88")


def eff_curve(sub, det_col: str, min_count: int = 10):
    centers, effs = [], []
    for lo, hi in zip(SNR_BINS[:-1], SNR_BINS[1:]):
        s = sub[(sub.target_snr >= lo) & (sub.target_snr < hi)]
        if len(s) >= min_count:
            centers.append((lo + hi) / 2)
            effs.append(float(s[det_col].mean()))
    return centers, effs


def mass_bin_results(df, det_col: str, min_count: int = 10) -> dict:
    """{'mf_distance_fraction': {bin: f}, 'eff_curves': {bin: {snr, eff}}}"""
    fracs, curves = {}, {}
    for lo_m, hi_m, lab in zip(MASS_EDGES[:-1], MASS_EDGES[1:], MASS_LABELS):
        sub = df[(df.chirp_mass >= lo_m) & (df.chirp_mass < hi_m)]
        c, e = eff_curve(sub, det_col, min_count)
        curves[lab] = {"snr": c, "eff": e}
        snr50 = float(np.interp(0.5, e, c)) if len(c) > 1 and max(e) >= 0.5 else np.nan
        fracs[lab] = 8.0 / snr50 if np.isfinite(snr50) else 0.0
    return {"mf_distance_fraction": fracs, "eff_curves": curves}
