"""LongChirp benchmark evaluator: trigger lists -> FAR and sensitive chirp-distance and chirp-volume.

Runs a submission makes (see README):
  background slide k = 0..S : H1 of test[i] paired with L1 of test[(i + k) mod 24], noise only (k = 0 is zero lag).
                              Pairing across segments avoids the wrap-around discontinuity of circular slides.
  foreground realization r  : test[i] H1 and L1 at zero lag, with realization r of injections.csv added.
A submission is a directory holding bg_k{k}.csv and fg_r{r}.csv, each with columns `time,stat`. `time` is the H1 GPS
time of the trigger. Triggers closer than CLUSTER_S are clustered here (the loudest is kept), so pipelines need not
cluster.

FAR(x) = (#background triggers with stat >= x) / T_bg, where T_bg = (number of slides) x 24 x 4096 s.
An injection is found at threshold x if a foreground trigger with stat >= x lies within +-WINDOW_S of its H1 merger
time. Sensitive chirp-volume (MLGWSC-1 convention; d_c uniform on [lo, hi]) is
V(x) = (hi - lo) * mean_i[ 4 pi d_c,i^2 * found_i(x) ], and the sensitive chirp-distance is (3 V / 4 pi)^(1/3).

  python benchmark/evaluate.py <submission dir> [--out results/<name>.json]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

B = Path(__file__).resolve().parent
sys.path.insert(0, str(B)); sys.path.insert(0, str(B.parent))
import injections as I  # noqa: E402

N_SLIDES = 23
WINDOW_S, CLUSTER_S = 1.0, 1.0
FARS_PER_YEAR = {"1/day": 365.25, "1/week": 52.18, "1/month": 12.0}
YEAR_S = 365.25 * 86400


def read_triggers(path):
    if not path.exists():
        return np.zeros((0, 2))
    with open(path) as f:
        rows = [(float(r["time"]), float(r["stat"])) for r in csv.DictReader(f)]
    return cluster(np.array(rows).reshape(-1, 2))


def cluster(tr):
    """Greedy: loudest first; drop any trigger within CLUSTER_S of an already-kept, louder one."""
    if len(tr) == 0:
        return tr
    import bisect
    tr = tr[np.argsort(-tr[:, 1])]
    keep = np.zeros(len(tr), bool)
    sorted_t = []
    for i, (t, _) in enumerate(tr):
        j = bisect.bisect_left(sorted_t, t)
        near = (j < len(sorted_t) and sorted_t[j] - t < CLUSTER_S) or (j > 0 and t - sorted_t[j - 1] < CLUSTER_S)
        if not near:
            keep[i] = True
            bisect.insort(sorted_t, t)
    return tr[keep]


def injection_table():
    from pycbc.detector import Detector
    h1 = Detector("H1")
    with open(I.TABLE) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("t_geo", "ra", "dec", "chirp_distance", "mass1", "mass2"):
            r[k] = float(r[k])
        r["realization"] = int(r["realization"])
        r["t_h1"] = r["t_geo"] + h1.time_delay_from_earth_center(r["ra"], r["dec"], r["t_geo"])
    return rows


def evaluate(sub, n_test_segments=24, seg_len=4096):
    bg = [read_triggers(sub / f"bg_k{k}.csv") for k in range(N_SLIDES + 1)]
    n_runs = sum(1 for k in range(N_SLIDES + 1) if (sub / f"bg_k{k}.csv").exists())
    t_bg = n_runs * n_test_segments * seg_len
    bg_stats = np.sort(np.concatenate([b[:, 1] for b in bg]))[::-1]
    inj = injection_table()
    loudest = np.full(len(inj), -np.inf)
    for r in sorted({x["realization"] for x in inj}):
        fg = read_triggers(sub / f"fg_r{r}.csv")
        if len(fg) == 0:
            continue
        order = np.argsort(fg[:, 0]); ft, fsx = fg[order, 0], fg[order, 1]
        for i, x in enumerate(inj):
            if x["realization"] != r:
                continue
            lo, hi = np.searchsorted(ft, [x["t_h1"] - WINDOW_S, x["t_h1"] + WINDOW_S])
            if hi > lo:
                loudest[i] = fsx[lo:hi].max()
    dc = np.array([x["chirp_distance"] for x in inj])
    res = {"background_livetime_days": t_bg / 86400, "n_background_runs": n_runs, "n_injections": len(inj),
           "at_far": {}}
    for name, per_year in FARS_PER_YEAR.items():
        n_allowed = per_year * t_bg / YEAR_S  # background events allowed at this FAR
        if n_allowed < 1:
            res["at_far"][name] = {"reachable": False, "why": f"background supports {n_allowed:.2f} events (<1)"}
            continue
        k = int(np.floor(n_allowed))  # at most k background triggers may exceed the threshold
        thr = bg_stats[k] if k < len(bg_stats) else -np.inf  # the (k+1)-th loudest: exactly k lie strictly above
        found = loudest > thr
        V = (I.DC_HI - I.DC_LO) * np.mean(4 * np.pi * dc ** 2 * found)
        res["at_far"][name] = {"reachable": True, "threshold": float(thr), "found_fraction": float(found.mean()),
                               "chirp_volume_mpc3": float(V), "chirp_distance_mpc": float((3 * V / (4 * np.pi)) ** (1 / 3))}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    res = evaluate(a.submission)
    print(json.dumps(res, indent=1))
    if a.out:
        a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
