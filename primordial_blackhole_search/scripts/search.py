"""Phase 5: end-to-end search demo — the full hierarchical pipeline.

  1. Inject ONE astrophysical event into BOTH detectors (H1 + L1) with the
     proper per-detector antenna response and light-travel time delay.
  2. ML model sweeps both detectors cheaply -> trigger windows.
  3. Matched-filter follow-up inside trigger windows -> precise time + SNR.
  4. Two-detector coincidence test: |t_H1 - t_L1| <= 15 ms.
  5. Bonus physics plot: SNR loss vs chirp-mass mismatch — WHY subsolar
     template banks are compute-bound (0.1% off dephases a minutes-long
     template), i.e. exactly the cost the ML trigger stage amortizes.

This validates the pipeline plumbing with the true template (stated openly);
a blind production bank is future work.

Run:  uv run python scripts/search.py --model cnn
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pbh import config as C
from pbh.data import estimate_psd, load_segment, whiten
from pbh.models import make_model
from pbh.spectrogram import spectrogram
from pbh.waveforms import InjectionParams, make_whitened_injection

from evaluate import (  # reuse the sweep machinery
    CROP,
    FRAMES_PER_WIN,
    HOP_FRAMES,
    N_WIN,
    pool_and_log,
    score_windows,
)

EVENT = InjectionParams(
    mass1=0.62, mass2=0.44, ra=2.1, dec=-0.9, psi=1.3, inclination=0.7, coa_phase=0.4
)
EVENT_SNR_H1 = 22.0  # at the v1 detector's ~95% efficiency point (SNR_50 ~ 18.6);
#                      a louder-than-typical but in-reach demo event


def whiten_ifo(ifo: str, gps: int):
    strain, t0 = load_segment(ifo, gps)
    psd = estimate_psd(strain)
    return whiten(strain, psd), t0, psd


def mf_followup(w: np.ndarray, tmpl_w: np.ndarray, lo: int, hi: int,
                min_snr: float = 8.0) -> list[tuple[float, int]]:
    """Correlate a unit-norm whitened template against w[lo:hi].

    Returns ALL significant peaks [(snr, sample_index_in_w), ...], loudest
    first — not just the maximum. A glitch inside the trigger window can
    out-correlate the real signal in one detector; keeping every peak lets
    the two-site coincidence test pick the physical pairing (and that
    rejection is the point of coincidence).
    """
    from scipy.signal import find_peaks

    seg = w[lo:hi]
    t = tmpl_w / np.sqrt(np.sum(tmpl_w**2))
    n = len(seg) + len(t)
    corr = np.fft.irfft(np.fft.rfft(seg, n) * np.conj(np.fft.rfft(t, n)), n)
    corr = corr[: len(seg)]
    idx, props = find_peaks(corr, height=min_snr, distance=C.SAMPLE_RATE)
    peaks = sorted(
        ((float(h), lo + int(k) + len(t) - 1)
         for k, h in zip(idx, props["peak_heights"])),
        reverse=True,
    )
    return peaks[:10]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["cnn", "transformer"], default="cnn")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = make_model(args.model)
    model.load_state_dict(torch.load(C.MODEL_DIR / f"{args.model}.pt",
                                     map_location=device))
    model.to(device).eval()

    manifest = json.loads((C.DATA_DIR / "manifest.json").read_text())
    gps = manifest["L1"]["coinc"][0]
    thresh = json.loads(
        (C.RESULTS_DIR / f"eval_{args.model}.json").read_text()
    )["thresh_zero_fa"]
    print(f"coincident stretch GPS {gps}; trigger threshold {thresh:.3f}")

    # ---- 1. inject one event into both detectors, consistent geometry
    from pycbc.detector import Detector

    t_geo = gps + C.SEGMENT_LEN - 900  # geocentric merger time
    data = {}
    true_t = {}
    h1_scale = None
    for ifo in ("H1", "L1"):
        w, t0, psd = whiten_ifo(ifo, gps)
        h_w, snr_ref = make_whitened_injection(EVENT, ifo, t_geo, psd)
        delay = Detector(ifo).time_delay_from_earth_center(EVENT.ra, EVENT.dec, t_geo)
        t_det = t_geo + delay
        merger_idx = int(round((t_det - t0) * C.SAMPLE_RATE))
        if ifo == "H1":
            h1_scale = EVENT_SNR_H1 / snr_ref
        sig = h_w * h1_scale  # SAME distance scaling both detectors
        sig_end = merger_idx + 1
        sig_start = sig_end - len(sig)
        w[sig_start:sig_end] += sig
        data[ifo] = (w, t0, psd)
        true_t[ifo] = t_det
        print(f"  {ifo}: injected SNR {snr_ref * h1_scale:.1f}, "
              f"merger at GPS {t_det:.4f} (delay {delay * 1000:+.2f} ms)")

    # ---- 2. ML sweep
    triggers = {}
    for ifo in ("H1", "L1"):
        w, t0, psd = data[ifo]
        spec = spectrogram(w[CROP:-CROP])
        starts = np.arange(0, spec.shape[1] - FRAMES_PER_WIN, HOP_FRAMES)
        wins = np.stack(
            [pool_and_log(spec[:, s : s + FRAMES_PER_WIN]) for s in starts]
        )
        scores = score_windows(model, device, wins)
        hits = np.where(scores > thresh)[0]
        # cluster adjacent hits -> trigger windows in sample coordinates
        clusters = []
        for h in hits:
            samp = CROP + h * HOP_FRAMES * int(C.STFT_HOP_SEC * C.SAMPLE_RATE)
            if clusters and samp - clusters[-1][1] <= N_WIN:
                clusters[-1] = (clusters[-1][0], samp + N_WIN,
                                max(clusters[-1][2], float(scores[h])))
            else:
                clusters.append((samp, samp + N_WIN, float(scores[h])))
        triggers[ifo] = clusters
        print(f"  {ifo}: {len(hits)} hot windows -> {len(clusters)} trigger(s)")

    # ---- 3. MF follow-up with the true template (ALL significant peaks)
    peaks = {}
    for ifo in ("H1", "L1"):
        w, t0, psd = data[ifo]
        h_w, _ = make_whitened_injection(EVENT, ifo, t_geo, psd)
        plist = []
        for lo, hi, sc in triggers[ifo]:
            pad = len(h_w)
            lo2, hi2 = max(0, lo - pad), min(len(w), hi + 4 * C.SAMPLE_RATE)
            plist += [
                (snr, t0 + idx / C.SAMPLE_RATE)
                for snr, idx in mf_followup(w, h_w, lo2, hi2)
            ]
        peaks[ifo] = sorted(plist, reverse=True)
        shown = ", ".join(
            f"SNR {s:.1f} @ {(t - true_t[ifo]) * 1000:+.0f} ms" for s, t in plist[:3]
        )
        print(f"  {ifo}: {len(plist)} follow-up peak(s): {shown}")

    # ---- 4. coincidence: best network-SNR pair with |dt| <= 15 ms
    candidates = {"H1": None, "L1": None}
    verdict = "NO COINCIDENT CANDIDATE"
    best_net = 0.0
    for s1, t1 in peaks["H1"]:
        for s2, t2 in peaks["L1"]:
            if abs(t1 - t2) <= 0.015 and np.hypot(s1, s2) > best_net:
                best_net = float(np.hypot(s1, s2))
                candidates["H1"], candidates["L1"] = (s1, t1), (s2, t2)
    if candidates["H1"]:
        dt = candidates["H1"][1] - candidates["L1"][1]
        true_dt = true_t["H1"] - true_t["L1"]
        n_rejected = len(peaks["H1"]) + len(peaks["L1"]) - 2
        verdict = (
            f"COINCIDENT CANDIDATE: H1 SNR {candidates['H1'][0]:.1f} + "
            f"L1 SNR {candidates['L1'][0]:.1f} -> network {best_net:.1f}; "
            f"dt = {dt * 1000:+.2f} ms (true {true_dt * 1000:+.2f} ms); "
            f"{n_rejected} non-coincident peak(s) rejected (incl. any glitches)"
        )
        for ifo in ("H1", "L1"):
            err = (candidates[ifo][1] - true_t[ifo]) * 1000
            print(f"  {ifo}: coincident peak timing error {err:+.1f} ms")
    print("\n" + verdict)

    # ---- 5. bank-density physics plot: SNR loss vs chirp-mass mismatch
    print("\nbank-mismatch curve (why subsolar banks are huge)...")
    w, t0, psd = data["H1"]
    lo = int((true_t["H1"] - t0) * C.SAMPLE_RATE) - len(make_whitened_injection(
        EVENT, "H1", t_geo, psd)[0]) - 4 * C.SAMPLE_RATE
    hi = int((true_t["H1"] - t0) * C.SAMPLE_RATE) + 4 * C.SAMPLE_RATE
    deltas = np.array([-1e-2, -3e-3, -1e-3, -3e-4, -1e-4, 0.0,
                       1e-4, 3e-4, 1e-3, 3e-3, 1e-2])
    rec = []
    for d in deltas:
        # perturb both masses by the same factor -> chirp mass scales by (1+d)
        p2 = InjectionParams(
            mass1=EVENT.mass1 * (1 + d), mass2=EVENT.mass2 * (1 + d),
            ra=EVENT.ra, dec=EVENT.dec, psi=EVENT.psi,
            inclination=EVENT.inclination, coa_phase=EVENT.coa_phase,
        )
        tmpl, _ = make_whitened_injection(p2, "H1", t_geo, psd)
        pk = mf_followup(w, tmpl, max(0, lo), hi, min_snr=2.0)
        # only count a peak within +-2 s of the true merger (dephasing shifts
        # the apparent peak slightly); otherwise the event is lost in noise
        merger_samp = int((true_t["H1"] - t0) * C.SAMPLE_RATE)
        near = [s for s, idx in pk if abs(idx - merger_samp) < 2 * C.SAMPLE_RATE]
        snr = near[0] if near else 0.0
        rec.append(snr)
        print(f"  dMc/Mc = {d:+.0e}: recovered SNR {snr:.2f}")
    rec = np.array(rec)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(deltas * 100, rec / rec[deltas == 0.0][0], "o-")
    ax.set_xscale("symlog", linthresh=1e-2)
    ax.set_xlabel("chirp-mass mismatch [%]")
    ax.set_ylabel("fraction of SNR recovered")
    ax.set_title("Template dephasing for a ~3-minute subsolar signal\n"
                 "(why classic subsolar searches need millions of templates)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(C.RESULTS_DIR / "bank_mismatch.png", dpi=120)

    def cand_json(ifo: str):
        if not candidates[ifo]:
            return None
        return {"snr": candidates[ifo][0], "t": candidates[ifo][1],
                "t_err_ms": (candidates[ifo][1] - true_t[ifo]) * 1000}

    (C.RESULTS_DIR / "search_demo.json").write_text(json.dumps({
        "verdict": verdict,
        "event_snr_h1": EVENT_SNR_H1,
        "H1": cand_json("H1"),
        "L1": cand_json("L1"),
        "bank_mismatch": {str(d): float(r) for d, r in zip(deltas, rec)},
    }, indent=2))
    print("wrote results/search_demo.json + bank_mismatch.png")


if __name__ == "__main__":
    main()
