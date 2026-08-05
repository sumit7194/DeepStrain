"""O4 scout: does the O3a-trained cnn_w64 transfer to O4b noise? (decides whether O4 work needs retraining)

Our whole PBH arc was trained + evaluated on O3a (May 2019). O4a/O4b are now public and more sensitive, and
O4b includes a far better Virgo (N5's "Virgo doesn't help" was measured on weak O3a-era V1). Before any bulk
fetch/retrain, this answers the cheap decisive question: applied to O4b noise, does cnn_w64 still behave?

Three checks, all on a handful of O4b segments:
  T1  NOISE distribution: are O4b noise scores comparable to O3a's? (a big shift = domain mismatch, and the
      zero-FA threshold would be set by the new noise, not by signal quality)
  T2  INJECTION recovery: inject the same subsolar waveform population into O4b noise and measure the
      sensitive-distance fraction the same way evaluate.py does. Compare to the O3a-gated 0.41-0.48.
  T3  PSD context: O4b vs O3a amplitude spectral density in-band (why any shift happens; also tells us how
      much MORE sensitive O4b actually is -> the reason to move at all).

Verdict: transfer OK -> run the N5 Virgo re-test directly on O4b. Transfer poor -> retrain on O4b first.

Run:  .venv/bin/python scripts/o4_transfer_scout.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pbh import config as C
from pbh.data import segment_path, whiten_segment
from pbh.models import make_model
from pbh.spectrogram import spectrogram
from pbh.sweep import SweepGrid, pool_and_log, score_windows
from pbh.waveforms import inject_into_window, make_whitened_injection, sample_params

GRID = SweepGrid.short(64)
WIN, NBINS = GRID.win_samp, GRID.n_time_bins
CROP = C.WHITEN_CROP_SEC * C.SAMPLE_RATE
MASS_EDGES, MASS_LABELS = [0.17, 0.35, 0.55, 0.88], ["0.17-0.35", "0.35-0.55", "0.55-0.88"]
SNR_BINS = np.linspace(*C.EVAL_SNR_RANGE, 13)
N_INJ = 250


def noise_scores(model, dev, wc):
    starts = np.arange((len(wc) - WIN) // WIN) * WIN
    wins = np.stack([pool_and_log(spectrogram(wc[s:s + WIN]), NBINS) for s in starts])
    return score_windows(model, dev, wins)


def inject_and_score(model, dev, wc, t0, psd, rng, n_inj):
    wins, meta = [], []
    for _ in range(n_inj):
        p = sample_params(rng)
        h_w, snr_ref = make_whitened_injection(p, "H1", t0, psd)
        target = float(rng.uniform(*C.EVAL_SNR_RANGE))
        start = int(rng.integers(0, len(wc) - WIN))
        w, in_snr = inject_into_window(wc[start:start + WIN].copy(), h_w, snr_ref, target,
                                       int(rng.uniform(0.30, 1.0) * WIN))
        wins.append(pool_and_log(spectrogram(w), NBINS)); meta.append((p.chirp_mass, in_snr))
    sc = score_windows(model, dev, np.stack(wins))
    return [dict(chirp_mass=m, in_snr=s, score=float(x)) for (m, s), x in zip(meta, sc)]


def dist_fraction(df, thr):
    det = df.score.to_numpy() > thr
    out = {}
    for lo, hi, lab in zip(MASS_EDGES[:-1], MASS_EDGES[1:], MASS_LABELS):
        sub = df[(df.chirp_mass >= lo) & (df.chirp_mass < hi)]
        cen, eff = [], []
        for a, b in zip(SNR_BINS[:-1], SNR_BINS[1:]):
            m = (sub.in_snr >= a) & (sub.in_snr < b)
            if m.sum() >= 8:
                cen.append((a + b) / 2); eff.append(float(det[sub.index][m].mean()))
        snr50 = float(np.interp(0.5, eff, cen)) if len(cen) > 1 and max(eff) >= 0.5 else np.nan
        out[lab] = round(8.0 / snr50, 4) if np.isfinite(snr50) else 0.0
    return out


def main() -> None:
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = make_model("cnn"); model.load_state_dict(torch.load(C.MODEL_DIR / "cnn_w64.pt", map_location=dev))
    model.to(dev).eval()

    o4b = [g for g in json.loads(Path("/tmp/o4b_triple.json").read_text())
           if segment_path("H1", g).exists()][:3]
    o3a = json.loads((C.DATA_DIR / "manifest.json").read_text())["H1"]["test"][:3]
    print(f"O4b segments: {o4b}\nO3a segments: {o3a}\n")

    out = {}
    for era, segs in (("O3a", o3a), ("O4b", o4b)):
        rng = np.random.default_rng(C.SEED + 4242)
        all_noise, rows, asds = [], [], []
        for g in segs:
            w, t0, psd = whiten_segment("H1", g)
            wc = w[CROP:-CROP]
            all_noise.append(noise_scores(model, dev, wc))
            rows += inject_and_score(model, dev, wc, t0, psd, rng, N_INJ // len(segs))
            f = np.asarray(psd.sample_frequencies) if hasattr(psd, "sample_frequencies") else None
            v = np.asarray(psd)
            if f is not None:
                band = (f >= 50) & (f <= 300)
                asds.append(float(np.sqrt(np.median(v[band]))))
        noise = np.concatenate(all_noise)
        df = pd.DataFrame(rows)
        thr = float(noise.max())
        frac = dist_fraction(df, thr)
        out[era] = {"n_noise_windows": len(noise), "noise_median": float(np.median(noise)),
                    "noise_p99": float(np.percentile(noise, 99)), "zeroFA_thresh": thr,
                    "dist_fraction": frac, "mean_frac": float(np.mean(list(frac.values()))),
                    "median_asd_50_300Hz": float(np.median(asds)) if asds else None}
        print(f"[{era}] noise: median {np.median(noise):+.3f}, p99 {np.percentile(noise,99):+.3f}, "
              f"zero-FA thr {thr:.3f} ({len(noise)} windows)")
        print(f"[{era}] sensitive-distance fraction {frac} -> mean {out[era]['mean_frac']:.3f}")
        if asds:
            print(f"[{era}] median ASD in [50,300] Hz: {np.median(asds):.3e}")
        print()

    ratio = out["O4b"]["mean_frac"] / out["O3a"]["mean_frac"] if out["O3a"]["mean_frac"] else float("nan")
    thr_shift = out["O4b"]["zeroFA_thresh"] - out["O3a"]["zeroFA_thresh"]
    transfers = out["O4b"]["mean_frac"] > 0.7 * out["O3a"]["mean_frac"]
    print(f"TRANSFER: O4b/O3a sensitive-distance ratio {ratio:.2f}; zero-FA threshold shift {thr_shift:+.3f}")
    print("VERDICT:", "cnn_w64 TRANSFERS to O4b -> N5 Virgo re-test can run directly"
          if transfers else "cnn_w64 does NOT transfer -> retrain on O4b before any O4 claim")
    out["transfer_ratio"] = ratio; out["threshold_shift"] = thr_shift; out["transfers"] = bool(transfers)
    (C.RESULTS_DIR / "o4_transfer_scout.json").write_text(json.dumps(out, indent=2))
    print("wrote o4_transfer_scout.json")


if __name__ == "__main__":
    main()
