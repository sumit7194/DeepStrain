"""Is the deep-FAR tail VETOABLE? A cheap pilot before committing to an instrumented re-run.

WHERE THIS SITS. L2 (2026-08-19) pushed the background to 4,120 yr and answered the audit's precision
complaint (jackknife 33-44% -> 10-12%), but it also proved what does NOT scale: 1/decade's 425 background
events come from **8 distinct H1 windows** and 1/century's 43 from **3** -- the same count the 80.5-yr run
had. 51x the background barely moved the number of independent loud Hanford glitches that set the deep tail.
So the binding limit is effective sample size, and the field's answer is not more livetime but SIGNAL-
CONSISTENCY VETOES + data-quality flags. This asks whether that axis is actually open to us, cheaply, before
spending days on a fully instrumented re-run.

WHY A PILOT AND NOT THE REAL THING. The L2 run used --purge, so the retained cache holds only two scalars
per window (h, l). A real veto needs time-frequency structure, which means re-fetching. Rather than re-fetch
727 segments on a hunch, this re-fetches the **22 segments containing the top-50 loudest H1 windows** (~2 h)
and asks the one question that decides it.

THE QUESTION, and the reason it is not trivially yes. At MATCHED CNN score, does a template-free consistency
statistic separate the loud H1 windows (presumed glitches) from injected subsolar signals? Matching on score
is the whole point: without it we would only be re-measuring loudness, and any statistic correlated with
amplitude would "succeed" while carrying no consistency information at all.

THE PHYSICAL BASIS. A subsolar inspiral is LONG -- at these masses the chirp sweeps for minutes, so inside a
64-s window it is a continuous, slowly-rising track spread over the whole window. A blip glitch is ~ms and
broadband. That is a large, template-free difference in how energy is distributed in TIME, which is exactly
what a consistency statistic can see and what a per-window scalar score cannot.

  conc   (PRIMARY, pre-registered)  max energy in a 0.1-s block / total window energy -- ms glitch -> high
  dur                               number of 0.1-s blocks above 5x the median block energy
  chirp                             Spearman corr(block index, energy-weighted centroid frequency): a chirp
                                    sweeps UP monotonically (positive); a glitch has no preferred direction
  kurt                              excess kurtosis of the whitened samples

PRE-REGISTERED DECISION RULE (fixed before any number was produced):
  AUC >= 0.90 on `conc` at matched score  -> a veto exists; the instrumented re-run is justified
  AUC <= 0.70                             -> this axis is NOT available from single-window structure; NEGATIVE
  in between                              -> inconclusive; report as such, do not spin it either way

CAVEAT THAT MUST TRAVEL WITH THE RESULT. n=50 loud windows is a small sample and the 8 tail-setters are
smaller still, so the AUC carries a wide bootstrap CI; only a near-total separation can be called at this
size. The 8 tail-setting windows are reported SEPARATELY from the top-50 because they are the actual targets
-- a veto that works on the mid-tail but not on the 8 that set 1/decade would not help the rung we care about.

Run:  .venv/bin/python scripts/far_veto_pilot.py [--n-loud 50] [--n-inj 400]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
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
OUT = C.RESULTS_DIR / "far_veto_pilot.json"
BLOCK = int(0.1 * C.SAMPLE_RATE)          # 0.1 s energy blocks
STATS = ("conc", "dur", "chirp", "kurt")
PRIMARY = "conc"


def _feat(w):
    return pool_and_log(spectrogram(w), NBINS)


def consistency_stats(w):
    """Template-free descriptors of how a window's whitened energy is distributed in time and frequency."""
    n = (len(w) // BLOCK) * BLOCK
    b = w[:n].reshape(-1, BLOCK)
    e = (b ** 2).sum(axis=1)
    tot = float(e.sum())
    med = float(np.median(e))
    spec = spectrogram(w[:n])                              # (freq, time)
    f_idx = np.arange(spec.shape[0])[:, None]
    p = spec / np.maximum(spec.sum(axis=0, keepdims=True), 1e-30)
    centroid = (p * f_idx).sum(axis=0)                     # energy-weighted centre frequency per frame
    t = np.arange(len(centroid))
    rt = np.argsort(np.argsort(t)).astype(float)
    rc = np.argsort(np.argsort(centroid)).astype(float)
    chirp = float(np.corrcoef(rt, rc)[0, 1]) if len(rc) > 2 and rc.std() > 0 else 0.0
    x = (w - w.mean()) / (w.std() + 1e-30)
    return {
        "conc": float(e.max() / tot) if tot > 0 else 0.0,
        "dur": float((e > 5 * med).sum()),
        "chirp": chirp,
        "kurt": float((x ** 4).mean() - 3.0),
    }


def auc(pos, neg):
    """P(a random pos ranks above a random neg); ties count a half."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    d = pos[:, None] - neg[None, :]
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / (len(pos) * len(neg)))


def boot_ci(pos, neg, b=2000, seed=0):
    rng = np.random.default_rng(seed)
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    v = [auc(rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)) for _ in range(b)]
    return float(np.percentile(v, 5)), float(np.percentile(v, 95))


def loud_windows(n_loud):
    """The loudest H1 windows across the whole L2 cache, with the 1/decade tail-setters flagged."""
    H, seg, wid = [], [], []
    for p in sorted(CACHE.glob("seg_*.npz"), key=lambda q: int(q.stem.split("_")[1])):
        d = np.load(p); g = int(p.stem.split("_")[1])
        H.append(d["h"]); seg += [g] * len(d["h"]); wid += list(range(len(d["h"])))
    H = np.concatenate(H); seg = np.array(seg); wid = np.array(wid)
    o = np.argsort(H)[::-1][:n_loud]
    return [{"gps": int(seg[i]), "win": int(wid[i]), "score": float(H[i])} for i in o]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-loud", type=int, default=50)
    ap.add_argument("--n-inj", type=int, default=400)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = make_model("cnn"); model.load_state_dict(torch.load(C.MODEL_DIR / "cnn_w64.pt", map_location=dev))
    model.to(dev).eval()

    loud = loud_windows(args.n_loud)
    tail8 = {(d["gps"], d["win"]) for d in loud_windows(8)}
    segs = sorted({d["gps"] for d in loud})
    print(f"top-{args.n_loud} H1 windows span {len(segs)} segments | device {dev}", flush=True)
    print(f"score range {loud[-1]['score']:.2f} .. {loud[0]['score']:.2f}\n", flush=True)

    rng = np.random.default_rng(7)
    G, I = [], []                                   # glitch rows, injection rows
    for si, g in enumerate(segs):
        t0 = time.time()
        for attempt in range(4):
            if segment_path("H1", g).exists():
                break
            try:
                fetch_segment("H1", g); break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(20 * 2 ** attempt)
        w_full, t_gps, psd = whiten_segment("H1", g)
        wc = w_full[CROP:-CROP]
        nwin = (len(wc) - WIN) // WIN
        here = [d for d in loud if d["gps"] == g]
        loud_ids = {d["win"] for d in here}

        for d in here:
            w = wc[d["win"] * WIN:(d["win"] + 1) * WIN]
            G.append({**consistency_stats(w), "score": d["score"], "gps": g, "win": d["win"],
                      "tail8": (g, d["win"]) in tail8})

        # injections go into QUIET windows of the SAME segment, so noise realisation and PSD are matched
        quiet = [i for i in range(nwin) if i not in loud_ids]
        per_seg = max(1, args.n_inj // len(segs))
        feats, rows = [], []
        for _ in range(per_seg):
            p = sample_params(rng)
            hw, ref = make_whitened_injection(p, "H1", t_gps + C.SEGMENT_LEN // 2, psd)
            # span a wide SNR range so the matched-score band is populated from BOTH sides
            target = float(rng.uniform(8.0, 120.0))
            wi = int(rng.choice(quiet)); m = int(rng.integers(WIN // 2, WIN))
            a = wc[wi * WIN:(wi + 1) * WIN].copy()
            a[:m] += (hw * (target / ref))[-WIN:][WIN - m:]
            feats.append(_feat(a))
            rows.append({**consistency_stats(a), "target_snr": target,
                         "chirp_mass": float(p.chirp_mass), "gps": g})
        sc = score_windows(model, dev, np.array(feats))
        for r, s in zip(rows, sc):
            r["score"] = float(s)
        I += rows

        if os.path.exists(C.DATA_DIR):
            for f in Path(C.DATA_DIR).glob(f"*{g}*"):
                try:
                    f.unlink()
                except OSError:
                    pass
        progress("far_veto_pilot", si + 1, len(segs), glitches=len(G), injections=len(I))
        print(f"  [{si+1}/{len(segs)}] {g}: {len(here)} loud, {per_seg} inj ({time.time()-t0:.0f}s)",
              flush=True)

    gs = np.array([r["score"] for r in G]); isc = np.array([r["score"] for r in I])
    lo, hi = float(gs.min()), float(gs.max())
    band = (isc >= lo) & (isc <= hi)                        # MATCHED-SCORE comparison
    print(f"\nmatched-score band [{lo:.2f}, {hi:.2f}]: {int(band.sum())} of {len(I)} injections", flush=True)

    res = {"n_loud": len(G), "n_inj": len(I), "n_matched": int(band.sum()),
           "score_band": [lo, hi], "primary": PRIMARY, "stats": {}}
    print(f"\n{'stat':>7}  {'AUC':>6}  {'90% CI':>16}   glitch median   inj median")
    for k in STATS:
        pv = np.array([r[k] for r in G]); nv = np.array([r[k] for r in I])[band]
        a = auc(pv, nv); ci = boot_ci(pv, nv)
        res["stats"][k] = {"auc": a, "ci90": list(ci),
                           "glitch_median": float(np.median(pv)), "inj_median": float(np.median(nv))}
        print(f"{k:>7}  {a:6.3f}  [{ci[0]:6.3f}, {ci[1]:6.3f}]   {np.median(pv):12.4f}   {np.median(nv):10.4f}")

    t8 = np.array([r[PRIMARY] for r in G if r["tail8"]])
    nv = np.array([r[PRIMARY] for r in I])[band]
    res["tail8"] = {"n": int(len(t8)), "auc": auc(t8, nv), "median": float(np.median(t8)) if len(t8) else None}
    print(f"\nthe 8 tail-setting windows alone ({PRIMARY}): AUC {res['tail8']['auc']:.3f} "
          f"(n={res['tail8']['n']}) -- these are the ones that set 1/decade")

    a = res["stats"][PRIMARY]["auc"]
    res["verdict"] = ("veto exists — instrumented re-run justified" if a >= 0.90 else
                      "NEGATIVE — no veto from single-window structure" if a <= 0.70 else
                      "INCONCLUSIVE — between the pre-registered bars")
    res["decision_rule"] = {"veto_if_auc_ge": 0.90, "negative_if_auc_le": 0.70}
    print(f"\nVERDICT ({PRIMARY} AUC {a:.3f} vs pre-registered 0.90 / 0.70): {res['verdict']}")
    OUT.write_text(json.dumps(res, indent=2))
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
