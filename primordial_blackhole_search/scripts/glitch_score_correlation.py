"""Does the CNN respond to transient excess at all? n=3 said no; this asks it on ~1,800 windows.

THE OBSERVATION THAT PROMPTED IT. Characterising the 3 tail-setting windows available from the deep-FAR
segments gave a result that inverts the premise of the whole exercise: they contain NO supra-threshold
time-frequency cluster (max-excess 17.9-22.4, kurtosis ~0, i.e. 0.88x an ordinary noise window), while an
ordinary window in the same segment holding an enormous transient (max-excess 4708, kurtosis +2404) scores
-0.484 -- below average. If that holds, the events limiting our deep FAR are not glitches, there is nothing
to veto, and the veto programme is dead before it starts.

But n=3 from one segment cannot carry a structural claim, and GWOSC is degraded to ~1 kB/s so the other
four segments are hours away. THE ANSWER IS ALREADY ON DISK: 30 cached O3a segments, ~1,860 windows, the
data cnn_w64 was trained and tested on. The mechanism question -- does this detector respond to transient
excess? -- is answerable there without fetching anything.

WHAT IS MEASURED, per window: the CNN score, and the morphology of the largest connected excess cluster
(same instrument as glitch_anatomy_morphology, validated on synthetic blips/tones/noise). Then:
  * Spearman(score, max_excess) over all windows -- a detector that fires on transients must show this > 0
  * the morphology of the TOP-SCORING windows vs the HIGHEST-EXCESS windows: if these are disjoint sets,
    the detector and the transients are looking at different things
  * what fraction of top-scoring windows contain any supra-threshold cluster at all

PRE-REGISTERED:
  rho > 0.3 and top-scorers mostly have excess  -> the detector does track transients; the n=3 was a fluke
                                                   of one segment and the veto idea is alive
  rho ~ 0 and top-scorers show no excess        -> confirms n=3: the deep-FAR tail is NOT glitch-driven,
                                                   which independently explains why min/veto never bought
                                                   reach and why tail-normalising HURT
  rho < 0                                       -> stronger still: loud transients actively suppress the score

CAVEAT CARRIED FORWARD: this is O3a (in-domain for cnn_w64, which was trained on it) while the deep-FAR tail
is O4b. It answers "does this detector respond to transients", not "are the O4b tail windows glitch-free".
The second needs the remaining segments and stays open.

Run:  .venv/bin/python scripts/glitch_score_correlation.py [--n-seg 30]
"""
import argparse
import importlib.util
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

HERE = Path(__file__).resolve().parent
_s = importlib.util.spec_from_file_location("gm", HERE / "glitch_anatomy_morphology.py")
gm = importlib.util.module_from_spec(_s); _s.loader.exec_module(gm)

GRID = SweepGrid.short(64)
WIN, NBINS = GRID.win_samp, GRID.n_time_bins
CROP = C.WHITEN_CROP_SEC * C.SAMPLE_RATE
OUT = C.RESULTS_DIR / "glitch_score_correlation.json"


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1]) if ra.std() > 0 and rb.std() > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seg", type=int, default=30)
    args = ap.parse_args()

    segs = sorted({int(p.stem.split("_")[1]) for p in C.NOISE_DIR.glob("H1_*.hdf5")})[: args.n_seg]
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = make_model("cnn"); model.load_state_dict(torch.load(C.MODEL_DIR / "cnn_w64.pt", map_location=dev))
    model.to(dev).eval()
    print(f"{len(segs)} cached segments | device {dev}", flush=True)

    rows = []
    t0 = time.time()
    for si, g in enumerate(segs):
        try:
            w, _, _ = whiten_segment("H1", g)
        except Exception as e:
            print(f"  {g}: skip ({type(e).__name__})", flush=True); continue
        wc = w[CROP:-CROP]
        n = (len(wc) - WIN) // WIN
        feats = [pool_and_log(spectrogram(wc[i * WIN:(i + 1) * WIN]), NBINS) for i in range(n)]
        sc = score_windows(model, dev, np.array(feats))
        for i in range(n):
            m = gm.morphology(wc[i * WIN:(i + 1) * WIN])
            rows.append({"gps": g, "win": i, "score": float(sc[i]),
                         "max_excess": m["max_excess"], "has_excess": bool(m["has_excess"]),
                         "duration_s": m["duration_s"], "bandwidth_hz": m["bandwidth_hz"],
                         "peak_freq_hz": m["peak_freq_hz"], "n_pixels": m["n_pixels"],
                         "kurtosis": m["kurtosis"]})
        print(f"  [{si+1}/{len(segs)}] {g}: {n} windows ({time.time()-t0:.0f}s)", flush=True)

    s = np.array([r["score"] for r in rows]); e = np.array([r["max_excess"] for r in rows])
    rho = spearman(s, e)
    k = max(10, len(rows) // 100)
    top_s = set(np.argsort(s)[::-1][:k].tolist())
    top_e = set(np.argsort(e)[::-1][:k].tolist())
    overlap = len(top_s & top_e)
    frac_excess_top = float(np.mean([rows[i]["has_excess"] for i in top_s]))
    frac_excess_all = float(np.mean([r["has_excess"] for r in rows]))
    # what does the CNN say about the windows with the biggest real transients?
    score_of_top_excess = [rows[i]["score"] for i in top_e]

    res = {"n_windows": len(rows), "n_segments": len(segs), "top_k": k,
           "spearman_score_vs_maxexcess": rho,
           "top_score_top_excess_overlap": overlap,
           "frac_top_scorers_with_excess": frac_excess_top,
           "frac_all_windows_with_excess": frac_excess_all,
           "median_score_all": float(np.median(s)),
           "median_score_of_highest_excess": float(np.median(score_of_top_excess)),
           "max_excess_of_top_scorers": [rows[i]["max_excess"] for i in sorted(top_s, key=lambda i: -s[i])][:10],
           "rows": rows}
    print(f"\nn = {len(rows)} windows over {len(segs)} segments")
    print(f"  Spearman(score, max_excess)            = {rho:+.3f}")
    print(f"  top-{k} by score  ∩  top-{k} by excess  = {overlap} windows")
    print(f"  fraction of top scorers with ANY excess = {frac_excess_top:.2f}  (all windows: {frac_excess_all:.2f})")
    print(f"  median score, all windows               = {np.median(s):+.3f}")
    print(f"  median score of the HIGHEST-excess windows = {np.median(score_of_top_excess):+.3f}")
    res["verdict"] = ("detector TRACKS transient excess -- the n=3 result was segment-specific"
                      if rho > 0.3 and frac_excess_top > 0.5 else
                      "detector does NOT track transient excess -- confirms the n=3 observation: the "
                      "deep-FAR tail is not glitch-driven, so there is nothing to veto"
                      if abs(rho) < 0.3 else
                      "detector is ANTI-correlated with transient excess -- loud glitches suppress the score")
    print(f"\nVERDICT: {res['verdict']}")
    OUT.write_text(json.dumps(res, indent=2))
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
