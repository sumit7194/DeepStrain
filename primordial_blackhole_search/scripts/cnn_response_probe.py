"""What DOES the CNN respond to, given it demonstrably does not respond to transient power?

WHERE THIS COMES FROM. glitch_score_correlation established at n=1,860 that Spearman(score, max transient
excess) = +0.091, that the top-scoring windows and the top-excess windows are disjoint sets, and that a
window holding a max-excess-4708 transient scores -0.484. The detector setting our deep-FAR ceiling is
responding to something, and it is not bursts. This asks what.

TWO PROBES, because a correlation can be confounded and an occlusion map cannot say what a feature MEANS.

  (A) BAND-POWER CORRELATION. The CNN eats a 128 x 63 log-spaced [50, 1024] Hz map. For each window take the
      mean feature value in each of 8 frequency bands and correlate with score across windows. This says
      WHICH PART OF THE SPECTRUM tracks the score. Subsolar chirps accumulate most of their SNR at LOW
      frequency (they sweep slowly there), so a detector doing its job should key on the bottom of the band.

  (B) OCCLUSION. Replace one region of the input with the window's own median value, re-score, and record
      the drop. Unlike a correlation this interrogates the MODEL rather than the data, and it is run on
      three populations that matter:
          - the top-scoring NOISE windows   (what triggers a false alarm)
          - injected SUBSOLAR signals       (what the detector is supposed to use)
          - median-scoring noise            (the null contrast)
      If the false-alarm map and the signal map key on the SAME region, the noise trigger is signal-like and
      the detector is being honestly fooled. If they differ, the trigger is a separate feature -- and that
      would explain why H1xL1 coincidence helps (an artefact uncorrelated between detectors) while every
      single-detector consistency cut we tried did not.

PRE-REGISTERED: I expect (A) to peak at low frequency for signals. For the top-scoring noise I have no
prediction, which is the point of running it.

Run:  .venv/bin/python scripts/cnn_response_probe.py [--n-seg 20]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pbh import config as C
from pbh.data import whiten_segment
from pbh.models import make_model
from pbh.spectrogram import spectrogram
from pbh.sweep import SweepGrid, pool_and_log, score_windows
from pbh.waveforms import make_whitened_injection, sample_params

GRID = SweepGrid.short(64)
WIN, NBINS = GRID.win_samp, GRID.n_time_bins
CROP = C.WHITEN_CROP_SEC * C.SAMPLE_RATE
OUT = C.RESULTS_DIR / "cnn_response_probe.json"
NB_F = 8          # frequency bands for the correlation / occlusion
NB_T = 8          # time chunks for occlusion


def band_edges(n, nb):
    return [(int(round(i * n / nb)), int(round((i + 1) * n / nb))) for i in range(nb)]


def freq_of_bin(i):
    """The feature rows are log-spaced 50 -> 1024 Hz."""
    return float(C.F_LOWER * (C.F_HIGH / C.F_LOWER) ** (i / (C.N_FREQ_BINS - 1)))


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1]) if ra.std() > 0 and rb.std() > 0 else float("nan")


def occlusion_map(model, dev, feats):
    """Mean score drop when each frequency band / time chunk is replaced by the window's own median."""
    base = score_windows(model, dev, feats)
    fe, te = band_edges(feats.shape[1], NB_F), band_edges(feats.shape[2], NB_T)
    fdrop, tdrop = [], []
    for a, b in fe:
        occ = feats.copy()
        for k in range(len(occ)):
            occ[k, a:b, :] = np.median(feats[k])
        fdrop.append(float(np.mean(base - score_windows(model, dev, occ))))
    for a, b in te:
        occ = feats.copy()
        for k in range(len(occ)):
            occ[k, :, a:b] = np.median(feats[k])
        tdrop.append(float(np.mean(base - score_windows(model, dev, occ))))
    return fdrop, tdrop, float(np.mean(base))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seg", type=int, default=20)
    ap.add_argument("--n-inj", type=int, default=40)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = make_model("cnn"); model.load_state_dict(torch.load(C.MODEL_DIR / "cnn_w64.pt", map_location=dev))
    model.to(dev).eval()
    segs = sorted({int(p.stem.split("_")[1]) for p in C.NOISE_DIR.glob("H1_*.hdf5")})[: args.n_seg]
    print(f"{len(segs)} cached segments | device {dev}", flush=True)

    rng = np.random.default_rng(11)
    F, S, INJ = [], [], []
    t0 = time.time()
    for si, g in enumerate(segs):
        try:
            w, t_gps, psd = whiten_segment("H1", g)
        except Exception as e:
            print(f"  {g}: skip ({type(e).__name__})", flush=True); continue
        wc = w[CROP:-CROP]
        n = (len(wc) - WIN) // WIN
        f = np.array([pool_and_log(spectrogram(wc[i * WIN:(i + 1) * WIN]), NBINS) for i in range(n)])
        F.append(f); S.append(score_windows(model, dev, f))
        for _ in range(max(1, args.n_inj // len(segs))):
            p = sample_params(rng)
            hw, ref = make_whitened_injection(p, "H1", t_gps + C.SEGMENT_LEN // 2, psd)
            tgt = float(rng.uniform(30.0, 120.0))
            wi = int(rng.integers(0, n)); mm = int(rng.integers(WIN // 2, WIN))
            a = wc[wi * WIN:(wi + 1) * WIN].copy()
            a[:mm] += (hw * (tgt / ref))[-WIN:][WIN - mm:]
            INJ.append(pool_and_log(spectrogram(a), NBINS))
        print(f"  [{si+1}/{len(segs)}] {g} ({time.time()-t0:.0f}s)", flush=True)

    F = np.concatenate(F); S = np.concatenate(S); INJ = np.array(INJ)
    print(f"\n{len(F)} noise windows, {len(INJ)} injections", flush=True)

    # ---- (A) which part of the spectrum tracks the score? ----------------------------------------
    fe = band_edges(F.shape[1], NB_F)
    res = {"n_windows": int(len(F)), "n_inj": int(len(INJ)), "bands": [], "occlusion": {}}
    print(f"\n(A) BAND-POWER vs SCORE, over noise windows")
    print(f"{'band (Hz)':>16} {'Spearman':>10}")
    for a, b in fe:
        bp = F[:, a:b, :].mean(axis=(1, 2))
        r = spearman(bp, S)
        res["bands"].append({"lo_hz": freq_of_bin(a), "hi_hz": freq_of_bin(b - 1), "spearman": r})
        print(f"{freq_of_bin(a):7.0f}-{freq_of_bin(b-1):7.0f} {r:+10.3f}")

    # ---- (B) occlusion: false alarms vs real signals ----------------------------------------------
    k = max(20, len(F) // 100)
    top = np.argsort(S)[::-1][:k]
    mid = np.argsort(np.abs(S - np.median(S)))[:k]
    pops = {"top_scoring_noise": F[top], "median_noise": F[mid], "injections": INJ}
    print(f"\n(B) OCCLUSION -- mean score drop when a band is replaced by the window's median")
    print(f"{'band (Hz)':>16} " + "".join(f"{n:>20}" for n in pops))
    maps = {}
    for name, feats in pops.items():
        fd, td, base = occlusion_map(model, dev, feats.astype(np.float32))
        maps[name] = {"freq_drop": fd, "time_drop": td, "base_score": base}
        res["occlusion"][name] = maps[name]
    for i, (a, b) in enumerate(fe):
        print(f"{freq_of_bin(a):7.0f}-{freq_of_bin(b-1):7.0f} " +
              "".join(f"{maps[n]['freq_drop'][i]:>20.3f}" for n in pops))
    print(f"{'BASE SCORE':>16} " + "".join(f"{maps[n]['base_score']:>20.3f}" for n in pops))

    fe_lo = np.array([b["lo_hz"] for b in res["bands"]])
    fe_hi = np.array([b["hi_hz"] for b in res["bands"]])

    def peak_band(m):
        i = int(np.argmax(m["freq_drop"]))
        return f"{fe_lo[i]:.0f}-{fe_hi[i]:.0f} Hz"

    def centre(m):
        v = np.clip(np.array(m["freq_drop"]), 0, None)
        w = v / max(v.sum(), 1e-9)
        return float((w * np.sqrt(fe_lo * fe_hi)).sum())

    res["peak_band"] = {n: peak_band(m) for n, m in maps.items()}
    res["centre_hz"] = {n: centre(m) for n, m in maps.items()}
    res["frac_below_224hz"] = {n: float(np.clip(np.array(m["freq_drop"]), 0, None)[:4].sum()
                                        / max(np.clip(np.array(m["freq_drop"]), 0, None).sum(), 1e-9))
                               for n, m in maps.items()}
    # COMPARE PROFILES, NOT PEAKS. The first version of this verdict took argmax and declared the false
    # alarms "a different feature" because the peak sat one band away from the injections' -- while the two
    # profiles correlated at 0.92 and their sensitivity centres differed by 10 Hz. An argmax over adjacent
    # bins is a coin flip when the peak is broad; the shape is the thing that carries the claim.
    a = np.array(maps["top_scoring_noise"]["freq_drop"])
    b = np.array(maps["injections"]["freq_drop"])
    res["profile_corr_noise_vs_injections"] = float(np.corrcoef(a, b)[0, 1])
    print("\npeak band / sensitivity centre / fraction below 224 Hz:")
    for n in maps:
        print(f"  {n:>20}: peak {res['peak_band'][n]:>12} | centre {res['centre_hz'][n]:6.1f} Hz | "
              f"below-224 {res['frac_below_224hz'][n]:.3f}")
    print(f"  profile correlation, top-scoring noise vs injections = "
          f"{res['profile_corr_noise_vs_injections']:+.3f}")
    same = res["profile_corr_noise_vs_injections"] > 0.7
    res["false_alarm_matches_signal"] = bool(same)
    res["verdict"] = ("false alarms use the SAME spectral region as real signals (profile correlation "
                      f"{res['profile_corr_noise_vs_injections']:+.2f}) -- the detector is honestly fooled by "
                      "band-limited noise power, so no single-detector cut can separate them; coincidence "
                      "works because the fluctuation is independent between detectors"
                      if same else
                      "false alarms use a DIFFERENT spectral region from signals -- a separate artefact")
    print(f"\nVERDICT: {res['verdict']}")
    OUT.write_text(json.dumps(res, indent=2))
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
