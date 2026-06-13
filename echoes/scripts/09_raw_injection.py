"""Step 09 — v4: raw-strain injection, the production path.

Injects echo trains into RAW strain, re-whitens through the standard pipeline,
and scores with the saved v2 scorers. Closes the whitened-domain-injection
simplification: amplitude calibration (X0), background consistency (X1), a
sensitivity spot check (X2), and the out-of-band control done properly (X3).
Pre-registration in notes/lab_notebook.md (2026-06-13).
"""

import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from echolib import (
    BAND,
    DETECTORS,
    GW150914_DT_PRED,
    RESULTS,
    comb_on_env,
    fetch_block,
    progress,
)
from gwpy.timeseries import TimeSeries

spec = importlib.util.spec_from_file_location(
    "ml", Path(__file__).resolve().parent / "07_ml_scorer.py"
)
ml = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ml)

EVENT = "GW150914"
FS = 4096.0
SEG = 3.0
A_REF = 2.0e-21
N_CAL, N_BG, N_TRIALS = 8, 30, 30
rng = np.random.default_rng(41)


def raw_train(times, t_seg0, amp, f0=250.0, tau=0.02, gamma=0.7, dt=GW150914_DT_PRED):
    out = np.zeros_like(times)
    phase = rng.uniform(0, 2 * np.pi)
    for k in range(6):
        t_k = t_seg0 + 0.1 + k * dt
        rel = times - t_k
        out += ((-1) ** k * amp * gamma**k
                * np.where(rel >= 0, np.exp(-rel / tau) * np.sin(2 * np.pi * f0 * rel + phase), 0.0))
    return out


def whitened_segment(raw, center, inj=None):
    """64-s raw slice around center (+ optional injection) -> standard 3-s
    whitened+bandpassed segment starting at center+0.05."""
    sl = raw.crop(center - 30, center + 34)
    if inj is not None:
        sl = sl.copy()
        sl.value[:] = sl.value + inj(sl.times.value)
    w = sl.whiten(4, 2).bandpass(*BAND)
    seg = w.crop(center + 0.05, center + 0.05 + SEG + 0.01).value[: int(SEG * FS)]
    return seg


def main() -> None:
    raws = {det: fetch_block(det, EVENT) for det in DETECTORS}
    t0 = float(raws["H1"].t0.value)
    centers = t0 + 308 + 4 * np.arange(45)  # eval-time region, 4-s spacing
    dt_grid = np.arange(0.05, 0.5, 0.005)
    j_pred = int(np.argmin(np.abs(dt_grid - GW150914_DT_PRED)))

    models = {}
    for det in DETECTORS:
        m = ml.ConvAE()
        m.load_state_dict(torch.load(RESULTS / f"07_scorer_{det}.pt", weights_only=True))
        m.eval()
        models[det] = m

    # X0: calibration by DIFFERENCING — whiten(noise+inj) − whiten(noise)
    # isolates the whitened signal itself (whitening is linear at fixed PSD;
    # the injection's effect on the PSD estimate is negligible at these amps).
    slopes = {}
    for det in DETECTORS:
        amps_w = []
        for i in range(N_CAL):
            c = float(centers[rng.integers(0, len(centers))])
            inj = lambda t: raw_train(t, c + 0.05, A_REF)
            seg_inj = whitened_segment(raws[det], c, inj)
            seg_cln = whitened_segment(raws[det], c)
            diff = seg_inj - seg_cln
            i0 = int((0.1 - 0.005) * FS)
            i1 = int((0.1 + 0.010) * FS)
            amps_w.append(np.max(np.abs(diff[i0:i1])))
            progress("09_raw_cal", i, N_CAL)
        slopes[det] = float(np.median(amps_w)) / A_REF
        print(f"X0 {det}: whitened SIGNAL amp at A_ref={A_REF:.1e} -> median "
              f"{np.median(amps_w):.2f} (differenced — noise floor removed)")
    slope = float(np.mean(list(slopes.values())))
    print(f"X0 mean slope: {slope:.3e} per unit strain "
          f"(raw amp for 1.0 sigma-equiv: {1.0/slope:.2e})")

    def network_score(c, inj=None):
        total = np.zeros(len(dt_grid))
        for det in DETECTORS:
            seg = whitened_segment(raws[det], c, inj)
            total += comb_on_env(ml.error_envelope(models[det], seg, FS), FS, dt_grid)
        return float(total[j_pred])

    # X1: background through THIS path
    bg = []
    for i in range(N_BG):
        bg.append(network_score(float(centers[i % len(centers)])))
        progress("09_raw_bg", i, N_BG)
    bg = np.array(bg)
    thresh = float(np.quantile(bg, 0.95))
    print(f"X1 background (n={N_BG}): 95th pct = {thresh:.3f}, 99th = "
          f"{np.quantile(bg, 0.99):.3f} (v2 whitened-path 99th: 0.093)")

    # X2: sensitivity spot check at calibrated raw amplitudes
    eff = {}
    for target in (0.2, 0.5, 1.0):
        A = target / slope
        hits = 0
        for i in range(N_TRIALS):
            c = float(centers[rng.integers(0, len(centers))])
            hits += network_score(c, lambda t: raw_train(t, c + 0.05, A)) > thresh
            progress(f"09_raw_sens_{target}", i, N_TRIALS)
        eff[target] = hits / N_TRIALS
        print(f"X2 {target} sigma-equiv (raw A={A:.2e}): {hits}/{N_TRIALS} "
              f"({100 * hits / N_TRIALS:.0f}%)")

    # X3: the proper out-of-band control
    A_oob = 1.0 / slope
    hits = 0
    for i in range(N_TRIALS):
        c = float(centers[rng.integers(0, len(centers))])
        hits += network_score(c, lambda t: raw_train(t, c + 0.05, A_oob, f0=450.0)) > thresh
        progress("09_raw_oob", i, N_TRIALS)
    print(f"X3 450 Hz out-of-band at 1.0 sigma-equiv raw amp: {hits}/{N_TRIALS} fired "
          f"({100 * hits / N_TRIALS:.0f}%)  (gate <= 10%; whitened-domain version was 100%)")

    (RESULTS / "09_raw_injection.json").write_text(json.dumps(
        {"slopes": slopes, "bg95": thresh, "bg99": float(np.quantile(bg, 0.99)),
         "eff": eff, "oob_rate": hits / N_TRIALS}, indent=1))
    print(f"wrote {RESULTS / '09_raw_injection.json'}")


if __name__ == "__main__":
    main()
