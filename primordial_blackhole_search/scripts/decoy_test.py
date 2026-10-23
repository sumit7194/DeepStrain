"""Decoy test: does cnn_w64 respond to the PHYSICS of a chirp, or only to band power? (NN_REDESIGN.md §5)

PRE-REGISTERED 2026-10-23, written and committed before any decoy is scored. Mac only, nice 10.

Background. The occlusion probe (cnn_response_probe.json) found the CNN's sensitivity concentrated in band power at
~73-224 Hz, the same region its false alarms use. A network that had learned chirp physics should reject a signal
with the same power but a physically impossible time-frequency path. No training example ever contained one, so
nothing forced it to.

Families, all built from the SAME in-window piece s(t) of a real whitened TaylorF2 chirp, on the SAME noise window,
at the SAME in-window SNR (sum of squares), occupying the SAME samples of the window:
  R   real chirp s(t): frequency rises, amplitude grows into merger (merger in the window's second half, as in
      training).
  D1  pure time reversal s(T-t): identical |FFT|, so identical power per frequency and duration. Frequency FALLS
      and the envelope decays (the loud end moves to the start of the occupied interval).
  D2  reversed sweep, envelope kept: A(t) cos(-phi(T-t)), where A and phi come from the analytic signal of s.
      Same envelope vs time as R; frequency falls. Per-frequency power differs from R (unavoidable: a chirp ties
      amplitude to frequency).
  D3  wrong sweep law, envelope kept: A(t) cos(Phi(t)), with instantaneous frequency LINEAR in time from R's start
      to its end frequency. Rising, same envelope, same band, wrong law (R spends long at low f; D3 does not).
  (Tabula's critique, 2026-10-06: a naive reversal also flips the envelope, so D2/D3 hold the envelope fixed.
  Position is matched in all families.)

Metric: detection at the committed cnn_w64 zero-FA threshold 1.817 (track_eval, 384 non-overlapping O3a test
windows). r_D = detection(D) / detection(R), pooled over in-window SNR {18, 22, 26}, paired (same chirp, same noise).

Gates (must pass, or no reading is made):
  G0 golden: per-window scoring here reproduces the cached track_w64 noise scores on the same grid windows
     (max |diff| < 1e-3).
  G1 positive control: real-chirp detection rises with SNR and is >= 0.5 at SNR 22.
  G2 negative control: noise-only windows (same random windows, no signal) exceed the threshold <= 2%.

Pre-registered readings, per decoy family:
  r >= 0.7  => the CNN is (largely) BLIND to the property that decoy breaks;
  r <= 0.3  => the CNN DISCRIMINATES it;
  otherwise => partial.
Decomposition:
  D1 high => sweep direction + envelope ignored ("power shortcut").
  D2 vs D3 => separate sweep direction from sweep law, with the envelope fixed.
Prediction (mine, before running): D1, D2, D3 all r >= 0.7 (~60%). The likeliest alternative is D1 lower than D2/D3
if the CNN keys on the envelope or the loud end's position.
Secondary (reported, not decisive): per-chirp-mass tercile; mean scores.
Output: results/decoy_test.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import hilbert

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pbh import config as C  # noqa: E402
from pbh.data import whiten_segment  # noqa: E402
from pbh.models import make_model  # noqa: E402
from pbh.spectrogram import spectrogram  # noqa: E402
from pbh.sweep import CROP, SweepGrid, pool_and_log, score_windows  # noqa: E402
from pbh.waveforms import make_whitened_injection, sample_params  # noqa: E402

THRESH = 1.817  # committed cnn_w64 zero-FA threshold (eval_cnn_w64_track_w64.json)
SNRS = (14.0, 18.0, 22.0, 26.0)
PRIMARY_SNRS = (18.0, 22.0, 26.0)
N_PER_SEG = 100
GRID = SweepGrid.short(64)
WIN = GRID.win_samp
FS = C.SAMPLE_RATE


def model_in(window: np.ndarray) -> np.ndarray:
    return pool_and_log(spectrogram(window), GRID.n_time_bins)


def decoys(piece: np.ndarray) -> dict[str, np.ndarray]:
    z = hilbert(piece)
    amp, phi = np.abs(z), np.unwrap(np.angle(z))
    e = float(np.sum(piece ** 2))
    norm = lambda x: x * np.sqrt(e / max(float(np.sum(x ** 2)), 1e-30))  # noqa: E731
    d1 = piece[::-1].copy()
    d2 = norm(amp * np.cos(-phi[::-1]))
    sup = np.flatnonzero(amp > 0.05 * amp.max())
    a, b = int(sup[0]), int(sup[-1])
    finst = np.gradient(phi) * FS / (2 * np.pi)
    f0 = float(np.median(finst[a:min(a + FS, b)]))
    f1 = float(np.median(finst[max(b - FS // 5, a):b + 1]))
    t = np.arange(len(piece)) - a
    span = max(b - a, 1)
    f_lin = np.clip(f0 + (f1 - f0) * np.clip(t / span, 0, 1), C.F_LOWER, C.F_HIGH)
    d3 = norm(amp * np.cos(2 * np.pi * np.cumsum(f_lin) / FS))
    return {"R": piece, "D1": d1, "D2": d2, "D3": d3}


def main() -> int:
    import pandas as pd
    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = make_model("cnn")
    model.load_state_dict(torch.load(C.MODEL_DIR / "cnn_w64.pt", map_location=device, weights_only=True))
    model.to(device).eval()
    test = json.loads((C.DATA_DIR / "manifest.json").read_text())["H1"]["test"]
    rng = np.random.default_rng(C.SEED + 4242)
    rows, g0 = [], []
    for gps in test:
        w, t0, psd = whiten_segment("H1", gps)
        # G0: the first 8 grid windows, scored per window, against the cached sweep
        cache = C.RESULTS_DIR / "track_w64" / f"noise_{gps}.npy"
        if cache.exists():
            ref = np.load(cache)[:8]
            mine = score_windows(model, device, np.stack(
                [model_in(w[CROP + k * WIN: CROP + (k + 1) * WIN]) for k in range(8)]))
            g0.append(float(np.max(np.abs(mine - ref))))
        lo, hi = CROP, len(w) - CROP - WIN
        wins, meta = [], []
        for i in range(N_PER_SEG):
            p = sample_params(rng)
            h_w, _ = make_whitened_injection(p, "H1", t0, psd)
            start = int(rng.integers(lo, hi))
            noise = w[start:start + WIN]
            merger = int(rng.integers(WIN // 2, WIN))
            sig_start = merger + 1 - len(h_w)
            src_lo, dst_lo = max(0, -sig_start), max(0, sig_start)
            piece = h_w[src_lo:src_lo + (merger + 1 - dst_lo)].astype(np.float64)
            fam = decoys(piece)
            base = float(np.sqrt(np.sum(piece ** 2)))
            wins.append(model_in(noise)); meta.append((p.chirp_mass, "noise", 0.0, i))
            for snr in SNRS:
                for name, x in fam.items():
                    out = noise.copy()
                    out[dst_lo:merger + 1] += x * (snr / base)
                    wins.append(model_in(out)); meta.append((p.chirp_mass, name, snr, i))
        scores = score_windows(model, device, np.stack(wins).astype(np.float32))
        rows += [dict(gps=gps, chirp_mass=m, family=f, snr=s, chirp=i, score=float(sc))
                 for (m, f, s, i), sc in zip(meta, scores)]
        print(f"{gps}: {len(wins)} windows scored", flush=True)

    df = pd.DataFrame(rows); df["det"] = df.score > THRESH
    df.to_parquet(C.RESULTS_DIR / "decoy_test_rows.parquet")
    eff = df[df.family != "noise"].groupby(["family", "snr"]).det.mean().unstack(0)
    g1 = bool(eff["R"].is_monotonic_increasing and eff.loc[22.0, "R"] >= 0.5)
    g2_rate = float(df[df.family == "noise"].det.mean()); g2 = g2_rate <= 0.02
    g0_ok = bool(g0) and max(g0) < 1e-3
    prim = df[df.snr.isin(PRIMARY_SNRS)]
    r = {d: float(prim[prim.family == d].det.mean() / prim[prim.family == "R"].det.mean()) for d in ("D1", "D2", "D3")}
    read = {d: ("BLIND (r>=0.7)" if v >= 0.7 else "DISCRIMINATES (r<=0.3)" if v <= 0.3 else "PARTIAL")
            for d, v in r.items()}
    terc = np.quantile(df.chirp_mass, [1 / 3, 2 / 3])
    df["mbin"] = np.digitize(df.chirp_mass, terc)
    prim = df[df.snr.isin(PRIMARY_SNRS)]
    by_mass = {int(b): {d: float(g[g.family == d].det.mean() / max(g[g.family == "R"].det.mean(), 1e-9))
                        for d in ("D1", "D2", "D3")} for b, g in prim.groupby("mbin")}
    out = dict(threshold=THRESH, n_chirps=int(df.chirp.nunique() * len(test)), snrs=SNRS,
               gates=dict(G0_max_abs_diff=max(g0) if g0 else None, G0=g0_ok, G1=g1, G2_noise_rate=g2_rate, G2=g2),
               efficiency={f: {str(k): float(v) for k, v in eff[f].items()} for f in eff.columns},
               mean_score=df.groupby("family").score.mean().to_dict(),
               r_primary=r, reading=read if (g0_ok and g1 and g2) else "GATES FAILED - no reading",
               r_by_mass_tercile=by_mass, mass_tercile_edges=terc.tolist())
    (C.RESULTS_DIR / "decoy_test.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("gates", "efficiency", "r_primary", "reading", "r_by_mass_tercile")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
