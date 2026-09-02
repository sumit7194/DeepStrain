"""WHAT are the glitches that set our deep-FAR ceiling? Measure their morphology, don't guess.

WHY. Every limit the deep-FAR arc hit traces to a handful of loud H1 windows: 1/decade's 425 background
events come from 8 distinct windows, 1/century's 43 from 3, and 51x more background barely moved those
counts. We have treated them as an anonymous nuisance for the whole arc. We also know EXACTLY which windows
they are. Identifying what they are is the only route to vetoing them on principled grounds rather than the
post-hoc removal we correctly refused as tuning.

VERIFIED TAXONOMY (searched, not recited -- these are load-bearing so they carry sources):
  Blip            ~10 ms duration, broadband ~100 Hz, 30-300 Hz          [arXiv:2101.01200, 2511.05244]
  Koi Fish        short; low-frequency head, thin tail above 256 Hz;
                  believed a Blip subclass, origin still not understood   [arXiv:2208.12849]
  Scattered Light ~4 s arches, 8-64 Hz, peak frequencies 10-40 Hz        [arXiv:2605.14143, 2101.01200]
  O3 additions    Fast Scattering (Crown), Blip Low Frequency            [arXiv:2208.12849]
  Gravity Spy uses Omega scans at 4 durations because morphology spans very different timescales.

THE PREDICTION THIS MAKES, pre-registered. Our pipeline is band-limited to [50, 1024] Hz. Scattered light --
the most common class in both detectors -- lives at 8-64 Hz and is therefore almost entirely REMOVED by our
own band. So the windows setting our ceiling should be BLIP-FAMILY: ~10 ms, broadband, peaking inside
50-300 Hz. Concretely:
  * duration90 < 0.1 s AND bandwidth90 > 50 Hz      -> blip-consistent
  * duration90 > 1 s AND peak_freq < 100 Hz         -> scattered-light-like (would be surprising in-band)
  * a monotone rising frequency track               -> chirp-like, i.e. NOT obviously instrumental (would
                                                       be the interesting outcome and needs a hard look)
  * none of the above                               -> report as unclassified rather than forcing a label

THE CONTROL THAT MAKES IT A MEASUREMENT. Morphology numbers mean nothing without a comparison, so every
feature is computed identically on three populations:
  (a) the tail-setting loud H1 windows,
  (b) ordinary noise windows from the SAME segments (what typical background looks like),
  (c) injected subsolar signals at MATCHED CNN score (what the thing we are searching for looks like when
      it scores as loudly as these glitches do).
If (a) sits far from (c) on duration and frequency evolution, then the events limiting us look nothing like
the signal, which is precisely what licenses building a veto -- and (b) says whether the features separate
at all.

SCOPE, stated up front: this CHARACTERISES, it does not veto. Designing a veto on these windows and then
evaluating it on the same windows would be the tuning we refused. A veto built from this must be validated
on held-out segments, which is a separate item.

Run:  .venv/bin/python scripts/glitch_anatomy_morphology.py [--n-loud 8] [--n-inj 60]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import shutil
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pbh import config as C
from pbh.data import fetch_segment, segment_path, whiten_segment
from pbh.models import make_model
from pbh.progress import progress
from pbh.spectrogram import spectrogram
from pbh.sweep import SweepGrid, pool_and_log, score_windows
from pbh.waveforms import make_whitened_injection, sample_params

GRID = SweepGrid.short(64)
WIN, NBINS = GRID.win_samp, GRID.n_time_bins
CROP = C.WHITEN_CROP_SEC * C.SAMPLE_RATE
CACHE = C.RESULTS_DIR / "far_scores"
OUT = C.RESULTS_DIR / "glitch_morphology.json"
CKPT = C.RESULTS_DIR / "glitch_morph_ckpt"          # per-segment, so an outage costs one segment not the run
CKPT.mkdir(exist_ok=True, parents=True)
RETRIES = 4
BACKOFF_S = 20
FAIL_STREAK = 3          # consecutive failures that mean "GWOSC is down", not "this segment is bad"
OUTAGE_SLEEP = 1800      # GWOSC goes degraded for hours; the job is designed to outlast that
MAX_HOURS = 10
FS = C.SAMPLE_RATE
NPERSEG = 256                      # fine time resolution: 256/4096 = 62.5 ms hop for morphology
NOVER = 224


def fine_spec(x):
    """High-time-resolution spectrogram for morphology (NOT the pipeline's pooled feature)."""
    from scipy.signal import spectrogram as sp
    f, t, S = sp(x, fs=FS, nperseg=NPERSEG, noverlap=NOVER, scaling="spectrum", mode="magnitude")
    band = (f >= C.F_LOWER) & (f <= C.F_HIGH)
    return f[band], t, S[band]


def morphology(x, p_false=0.01):
    """Morphology of the largest CONNECTED excess cluster -- the two bugs this fixes are instructive.

    (1) FIRST VERSION measured the 90%-energy extent of the whole window. In 64 s of whitened noise, 90% of
        total energy covers essentially the whole window and band, so a 10 ms transient is a rounding error:
        blip, pure noise, and chirp ALL returned duration ~64 s, bandwidth ~960 Hz. It measured the noise
        floor identically in every case.

    (2) SECOND VERSION thresholded at a fixed 5x the per-row median. Whitened power is ~exponential, so
        P(X > 5*median) = 2^-5 = 3%, and with ~250,000 pixels that is thousands of scattered false pixels --
        whose time and frequency extents again span everything. Pure noise came back "scattered-light-like".
        A fixed threshold ignores the TRIALS FACTOR, which is the whole difficulty of transient finding.

    Both failures had the same signature: the same answer for signal and for nothing. Fixed by (a) setting
    the threshold from the pixel count so fewer than `p_false` false pixels are expected, and (b) describing
    only the largest CONNECTED component, because a transient is a cluster and noise is confetti."""
    from scipy import ndimage
    f, t, S = fine_spec(x)
    P = S ** 2
    med = np.median(P, axis=1, keepdims=True)
    R = P / np.maximum(med, 1e-30)
    npix = R.size
    # exponential tail: P(X > k*median) = 2^-k, so demand 2^-k * npix < p_false
    thresh = float(np.log2(max(npix, 2) / max(p_false, 1e-9)))
    mask = R > thresh
    base = {"has_excess": False, "duration_s": 0.0, "bandwidth_hz": 0.0, "peak_freq_hz": float("nan"),
            "peak_time_s": float("nan"), "chirp_corr": 0.0, "frac_energy_10ms": 0.0,
            "max_excess": float(R.max()), "thresh": thresh, "n_pixels": 0,
            "kurtosis": float(((x - x.mean()) ** 4).mean() / (x.var() ** 2 + 1e-30) - 3.0)}
    if not mask.any():
        return base
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    big = int(np.argmax(sizes)) + 1
    fi, ti = np.where(lab == big)
    dt = float(t[1] - t[0]) if len(t) > 1 else 1.0
    df = float(f[1] - f[0]) if len(f) > 1 else 1.0
    dur = float(t[ti.max()] - t[ti.min()]) + dt
    bw = float(f[fi.max()] - f[fi.min()]) + df
    pk = np.argmax(R[fi, ti])
    cen, tt = [], []
    for j in np.unique(ti):
        sel = fi[ti == j]
        w = R[sel, j]
        cen.append(float((w * f[sel]).sum() / w.sum())); tt.append(float(t[j]))
    if len(cen) >= 4:
        rt = np.argsort(np.argsort(tt)).astype(float)
        rc = np.argsort(np.argsort(cen)).astype(float)
        chirp = float(np.corrcoef(rt, rc)[0, 1]) if np.std(rc) > 0 else 0.0
    else:
        chirp = 0.0
    exc = np.zeros(len(t))
    np.add.at(exc, ti, R[fi, ti] - 1.0)
    k = max(1, int(round(0.010 / dt)))
    conc = float(np.convolve(exc, np.ones(k), "valid").max() / max(exc.sum(), 1e-30))
    return {**base, "has_excess": True, "duration_s": dur, "bandwidth_hz": bw,
            "peak_freq_hz": float(f[fi[pk]]), "peak_time_s": float(t[ti[pk]]), "chirp_corr": chirp,
            "frac_energy_10ms": conc, "n_pixels": int(sizes[big - 1])}


def classify(m):
    """Label ONLY where the verified numbers support it; otherwise say unclassified."""
    if m is None:
        return "no-energy"
    if not m.get("has_excess"):
        return "no-excess"
    if m["duration_s"] < 0.10 and m["bandwidth_hz"] > 50:
        return "blip-consistent"
    if m["duration_s"] > 1.0 and m["peak_freq_hz"] < 100:
        return "scattered-light-like"
    if m["chirp_corr"] > 0.5 and m["duration_s"] > 0.5:
        return "chirp-like (INVESTIGATE)"
    return "unclassified"


def loud_windows(n):
    H, seg, wid = [], [], []
    for p in sorted(CACHE.glob("seg_*.npz"), key=lambda q: int(q.stem.split("_")[1])):
        d = np.load(p); g = int(p.stem.split("_")[1])
        H.append(d["h"]); seg += [g] * len(d["h"]); wid += list(range(len(d["h"])))
    H = np.concatenate(H); seg = np.array(seg); wid = np.array(wid)
    o = np.argsort(H)[::-1][:n]
    return [{"gps": int(seg[i]), "win": int(wid[i]), "score": float(H[i])} for i in o]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-loud", type=int, default=8)
    ap.add_argument("--n-inj", type=int, default=60)
    ap.add_argument("--n-noise", type=int, default=40)
    ap.add_argument("--cached-only", action="store_true",
                    help="analyse only segments already on disk; do not fetch. Use when GWOSC is degraded "
                         "-- a partial result on real data beats waiting hours on 1 kB/s transfers.")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = make_model("cnn"); model.load_state_dict(torch.load(C.MODEL_DIR / "cnn_w64.pt", map_location=dev))
    model.to(dev).eval()

    loud = loud_windows(args.n_loud)
    segs = sorted({d["gps"] for d in loud})
    print(f"the {args.n_loud} windows that set the deep-FAR tail span {len(segs)} segments", flush=True)
    for d in loud:
        print(f"   H1 {d['score']:+7.3f}  seg {d['gps']}  win {d['win']}", flush=True)
    print(flush=True)

    rng = np.random.default_rng(77)
    rows_g, rows_n, rows_i = [], [], []
    MIN_FREE_GB = 1.5     # one segment is ~0.26 GB; refuse to be the job that fills a nearly-full disk
    t_start = time.time()
    consec = 0
    # ROUNDS, not one pass. GWOSC has gone degraded for ~12 h before; a single sweep that gives up on each
    # segment after 4 attempts finishes in minutes having measured nothing, and then REPORTS COMPLETION --
    # a success signal that does not mean success. Keep cycling the unfinished segments until they are done
    # or the wall-clock budget runs out.
    todo = list(segs)
    round_i = 0
    while todo and (time.time() - t_start) / 3600 <= MAX_HOURS:
        round_i += 1
        if round_i > 1:
            print(f"\n=== round {round_i}: {len(todo)} segment(s) still unfetched, sleeping "
                  f"{OUTAGE_SLEEP//60} min before retrying ===", flush=True)
            time.sleep(OUTAGE_SLEEP)
        failed = []
        for si, g in enumerate(todo):
            ck = CKPT / f"{g}.json"
            if ck.exists():                                   # resume: this segment is already done
                d = json.loads(ck.read_text())
                rows_g += d["glitches"]; rows_n += d["noise"]; rows_i += d["inj"]
                print(f"  [{si+1}/{len(segs)}] {g}: from checkpoint", flush=True)
                continue
            if (time.time() - t_start) / 3600 > MAX_HOURS:
                print(f"  !! wall-clock budget reached; {len(segs)-si} segments left, re-run to continue",
                      flush=True)
                break
            if args.cached_only and not segment_path("H1", g).exists():
                failed.append(g)
                print(f"  [{si+1}/{len(todo)}] {g}: not cached, skipped (--cached-only)", flush=True)
                continue
            free_gb = shutil.disk_usage("/").free / 1e9
            if free_gb < MIN_FREE_GB:
                print(f"  !! ABORTING before {g}: only {free_gb:.2f} GB free (< {MIN_FREE_GB}). "
                      f"Checkpoints kept; re-run after freeing space.", flush=True)
                break
            t0 = time.time()
            if consec >= FAIL_STREAK:
                print(f"  GWOSC appears down ({consec} consecutive failures) — sleeping "
                      f"{OUTAGE_SLEEP//60} min", flush=True)
                time.sleep(OUTAGE_SLEEP); consec = 0
            ok = False
            for attempt in range(RETRIES):
                if segment_path("H1", g).exists():
                    ok = True; break
                try:
                    fetch_segment("H1", g); ok = True; break
                except Exception as e:
                    print(f"     fetch {g} attempt {attempt+1}/{RETRIES} failed: {type(e).__name__}", flush=True)
                    for f in Path(C.DATA_DIR).glob(f"*{g}*"):   # orphan purge: a partial file is worse than none
                        try:
                            f.unlink()
                        except OSError:
                            pass
                    if attempt < RETRIES - 1:
                        time.sleep(BACKOFF_S * 2 ** attempt)
            if not ok:
                consec += 1
                failed.append(g)
                print(f"  [{si+1}/{len(todo)}] {g}: unavailable this round, will retry", flush=True)
                continue
            consec = 0
            w_full, t_gps, psd = whiten_segment("H1", g)
            wc = w_full[CROP:-CROP]
            nwin = (len(wc) - WIN) // WIN
            here = [d for d in loud if d["gps"] == g]
            loud_ids = {d["win"] for d in here}
            seg_g, seg_n, seg_i = [], [], []

            for d in here:
                x = wc[d["win"] * WIN:(d["win"] + 1) * WIN]
                m = morphology(x)
                seg_g.append({**(m or {}), "score": d["score"], "gps": g, "win": d["win"],
                              "class": classify(m)})

            quiet = [i for i in range(nwin) if i not in loud_ids]
            for wi in rng.choice(quiet, min(args.n_noise // len(segs) + 1, len(quiet)), replace=False):
                m = morphology(wc[wi * WIN:(wi + 1) * WIN])
                if m:
                    seg_n.append({**m, "gps": g, "win": int(wi)})

            lo, hi = min(d["score"] for d in loud), max(d["score"] for d in loud)
            feats, cand = [], []
            for _ in range(max(8, args.n_inj // len(segs))):
                p = sample_params(rng)
                hw, ref = make_whitened_injection(p, "H1", t_gps + C.SEGMENT_LEN // 2, psd)
                target = float(rng.uniform(20.0, 140.0))
                wi = int(rng.choice(quiet)); mm = int(rng.integers(WIN // 2, WIN))
                a = wc[wi * WIN:(wi + 1) * WIN].copy()
                a[:mm] += (hw * (target / ref))[-WIN:][WIN - mm:]
                feats.append(pool_and_log(spectrogram(a), NBINS)); cand.append((a, float(p.chirp_mass), target))
            sc = score_windows(model, dev, np.array(feats))
            for (a, mc, tgt), sv in zip(cand, sc):
                if lo - 2.0 <= sv <= hi + 2.0:
                    m = morphology(a)
                    if m:
                        seg_i.append({**m, "score": float(sv), "chirp_mass": mc, "target_snr": tgt,
                                      "class": classify(m)})

            tmp = ck.with_suffix(".tmp")
            tmp.write_text(json.dumps({"glitches": seg_g, "noise": seg_n, "inj": seg_i}))
            os.replace(tmp, ck)                                # atomic, like the score checkpoints
            rows_g += seg_g; rows_n += seg_n; rows_i += seg_i
            if not args.cached_only:          # keep the only copy we have while GWOSC is degraded
                for f in Path(C.DATA_DIR).glob(f"*{g}*"):
                    try:
                        f.unlink()
                    except OSError:
                        pass
            progress("glitch_morphology", si + 1, len(segs))
            print(f"  [{si+1}/{len(segs)}] {g}: {len(seg_g)} loud, {len(seg_i)} matched inj "
                  f"({time.time()-t0:.0f}s) purged", flush=True)
        todo = failed
    if todo:
        print(f"\n!! {len(todo)} segment(s) never fetched: {todo}", flush=True)

    def summarize(rows, keys=("duration_s", "bandwidth_hz", "peak_freq_hz", "chirp_corr",
                              "frac_energy_10ms", "max_excess")):
        return {k: {"median": float(np.median([r[k] for r in rows])),
                    "iqr": [float(np.percentile([r[k] for r in rows], 25)),
                            float(np.percentile([r[k] for r in rows], 75))]} for k in keys} if rows else {}

    res = {"loud_windows": rows_g, "n_noise": len(rows_n), "n_matched_inj": len(rows_i),
           "summary": {"glitches": summarize(rows_g), "noise": summarize(rows_n),
                       "matched_injections": summarize(rows_i)},
           "classes": {}}
    from collections import Counter
    res["classes"]["loud_windows"] = dict(Counter(r["class"] for r in rows_g))
    res["classes"]["matched_injections"] = dict(Counter(r["class"] for r in rows_i))

    print(f"\n{'feature':>18} {'GLITCHES':>22} {'noise':>22} {'matched injections':>22}")
    for k in ("duration_s", "bandwidth_hz", "peak_freq_hz", "chirp_corr", "frac_energy_10ms"):
        def fmt(d):
            return f"{d[k]['median']:8.3f} [{d[k]['iqr'][0]:6.2f},{d[k]['iqr'][1]:6.2f}]" if d else " " * 22
        print(f"{k:>18} {fmt(res['summary']['glitches'])} {fmt(res['summary']['noise'])} "
              f"{fmt(res['summary']['matched_injections'])}")
    print(f"\nloud-window classes:      {res['classes']['loud_windows']}")
    print(f"matched-injection classes: {res['classes']['matched_injections']}")

    if not rows_g:
        print("\n!! NO glitch windows measured -- refusing to write an artifact that would read as a "
              "completed null result. Checkpoints (if any) are kept; re-run when GWOSC recovers.")
        return
    OUT.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {OUT.name}")


if __name__ == "__main__":
    main()
