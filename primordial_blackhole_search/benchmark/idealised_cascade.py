"""Idealised two-stage cascade on LongChirp: is there room for a "careful second look" at all?

QUESTION. A cheap first stage flags candidates; a second stage examines only those, fully coherently. If even a
PERFECT second stage would gain little over Y1 alone, the idea (learned or classical refinement) is dead on this
benchmark, because the loss is in first-stage recall. Prior-art context and why this test comes first:
research_notes/Prior art checks for two ideas/phase_coherent_refinement.md.

PRE-REGISTRATION (written and committed before this script is run; nothing below may change after the first run).
  Stage 1 (real): the committed Y1 and Y2 submissions. A candidate is a background trigger above the stage-1
    threshold set at a loose rate F1 in {1 per hour, 1 per minute}; an injection is "caught" if its own foreground
    trigger (within the submission's own var) exceeds the same threshold.
  Stage 2 (idealised, deliberately optimistic): an injection caught by stage 1 scores its true-template optimal
    network SNR (Y0, results/oracle_snr.csv). Each stage-1 background candidate is examined over N_eff independent
    trials (templates x times) of a 2-detector coherent statistic, whose SNR^2 in Gaussian noise is chi-square with
    4 degrees of freedom. The stage-2 threshold rho* at target FAR f solves  (rate1 * N_eff) * P(chi2_4 > rho*^2) = f,
    rate1 = (number of stage-1 background candidates) / T_bg. N_eff in {1e2, 1e4, 1e6, 1e8}; a real coherent
    subsolar neighbourhood (Mc to ~1e-5, mass ratio, time within var) is plausibly >= 1e6, so 1e2 is very generous.
  Output: sensitive chirp distance (evaluate.py's estimator) at 1/day and 1/week, versus Y1 alone (57.47 / 51.68 Mpc),
    plus the stage-1 recall ceiling (rho* = 0, i.e. every caught injection found).
  VERDICT RULE (on the gain G = distance / Y1-alone distance - 1, over both stage-1 searches and both F1):
    DEAD        if G < 0.10 at N_eff = 1e2 for every stage-1 search, F1 and FAR;
    ROOM        if G >= 0.10 at N_eff = 1e6 for some stage-1 search and F1 at BOTH FARs;
    AMBIGUOUS   otherwise (room only under an implausibly small N_eff).
  Why the idealisation is one-sided: real noise has heavier tails than chi-square (raising rho*), optimal SNR ignores
    template mismatch and noise fluctuation, and no real refiner reaches the true template every time. So DEAD is a
    robust verdict; ROOM only licenses a real (non-idealised) test, not a claim.

  python benchmark/idealised_cascade.py      (minutes, < 1 GB; writes results/idealised_cascade.json)
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

B = Path(__file__).resolve().parent
sys.path.insert(0, str(B)); sys.path.insert(0, str(B.parent))
import evaluate as E  # noqa: E402
import injections as I  # noqa: E402

STAGE1_PER_DAY = {"1/hour": 24.0, "1/minute": 1440.0}
N_EFF = (1e2, 1e4, 1e6, 1e8)
TARGETS = {"1/day": 365.25, "1/week": 52.18}      # per year, as in evaluate.FARS_PER_YEAR
Y1_ALONE = {"1/day": 57.47, "1/week": 51.68}
KILL = 0.10


def chi2_4_sf(x):
    return np.exp(-x / 2) * (1 + x / 2)


def rho_star(p):
    if p >= 1:
        return 0.0
    return brentq(lambda r: chi2_4_sf(r * r) - p, 0.0, 60.0)


def distance(found, dc):
    V = (I.DC_HI - I.DC_LO) * np.mean(4 * np.pi * dc ** 2 * found)
    return float((3 * V / (4 * np.pi)) ** (1 / 3))


def main():
    inj = E.injection_table()
    dc = np.array([x["chirp_distance"] for x in inj])
    with open(B / "results" / "oracle_snr.csv") as f:
        oracle = {int(r["id"]): float(r["snr_net"]) for r in csv.DictReader(f)}
    snr = np.array([oracle[int(x["id"])] for x in inj])
    out = {"y1_alone_mpc": Y1_ALONE, "n_eff": list(N_EFF), "searches": {}}
    gains = []
    for name in ("y1", "y2"):
        sub = B / "submissions" / name
        bg, T, _ = E.background(sub)
        loudest, var = E.injection_loudest(sub, inj)
        rec = {"var_s": var, "background_days": T / 86400, "stage1": {}}
        for f1name, per_day in STAGE1_PER_DAY.items():
            n1 = int(np.floor(per_day * T / 86400))
            thr1 = float(bg[n1]) if n1 < len(bg) else -np.inf
            caught = loudest > thr1
            rate1 = n1 / T
            r1 = {"threshold": thr1, "n_candidates": n1, "caught_fraction": float(caught.mean()),
                  "recall_ceiling_mpc": distance(caught, dc), "by_n_eff": {}}
            for n_eff in N_EFF:
                row = {}
                for far, per_year in TARGETS.items():
                    p = (per_year / E.YEAR_S) / (rate1 * n_eff)
                    rs = rho_star(p)
                    d = distance(caught & (snr >= rs), dc)
                    g = d / Y1_ALONE[far] - 1
                    row[far] = {"rho_star": rs, "distance_mpc": d, "gain": g}
                    gains.append((name, f1name, n_eff, far, g))
                r1["by_n_eff"][f"{n_eff:.0e}"] = row
            rec["stage1"][f1name] = r1
            print(name, f1name, f"caught {caught.mean():.3f}, recall ceiling {r1['recall_ceiling_mpc']:.1f} Mpc",
                  {k: {f: round(v[f]["distance_mpc"], 1) for f in v} for k, v in r1["by_n_eff"].items()}, flush=True)
        out["searches"][name] = rec
    g = {(a, b, n, f): v for a, b, n, f, v in gains}
    dead = all(v < KILL for (a, b, n, f), v in g.items() if n == 1e2)
    room = any(all(g[(a, b, 1e6, f)] >= KILL for f in TARGETS) for a in ("y1", "y2") for b in STAGE1_PER_DAY)
    out["verdict"] = "DEAD" if dead else ("ROOM" if room else "AMBIGUOUS")
    (B / "results" / "idealised_cascade.json").write_text(json.dumps(out, indent=1))
    print("VERDICT", out["verdict"])


if __name__ == "__main__":
    main()
