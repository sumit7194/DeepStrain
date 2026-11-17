"""Golden test for evaluate.py on synthetic submissions whose answers are known in advance.

E1 perfect detector (a stat-100 trigger on every injection, background stats < 10): found fraction 1 at every FAR,
   and V equals the analytic chirp-volume (4 pi / 3)(hi^3 - lo^3) to within the Monte-Carlo error of 1344 draws.
E2 blind detector (no foreground triggers): found 0, V = 0.
E3 FAR threshold: background of known stats; at 1/day exactly floor(rate * T) triggers lie strictly above the
   threshold.
E4 clustering: triggers 0.3 s apart collapse to the loudest.
"""
import csv
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

B = Path(__file__).resolve().parent
sys.path.insert(0, str(B)); sys.path.insert(0, str(B.parent))
import evaluate as E  # noqa: E402
import injections as I  # noqa: E402


def write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["time", "stat"]); w.writerows(rows)


def main():
    inj = E.injection_table()
    segs = json.loads((B / "manifest.json").read_text())["test"]
    rng = np.random.default_rng(1)
    ok, out = True, {}
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        bg_stat = {}
        for k in range(E.N_SLIDES + 1):
            t = np.concatenate([g + np.sort(rng.uniform(10, 4086, 200)) for g in segs])
            st = rng.uniform(0, 10, len(t))
            bg_stat[k] = st
            write(d / f"bg_k{k}.csv", zip(t, st))
        for r in range(I.K):
            write(d / f"fg_r{r}.csv", [(x["t_h1"] + 0.2, 100.0) for x in inj if x["realization"] == r])
        res = E.evaluate(d)
        dc = np.array([x["chirp_distance"] for x in inj])
        v_true = 4 * np.pi / 3 * (I.DC_HI ** 3 - I.DC_LO ** 3)
        mc_se = (I.DC_HI - I.DC_LO) * np.std(4 * np.pi * dc ** 2) / np.sqrt(len(dc))
        for name, r in res["at_far"].items():
            if r["reachable"]:
                ok &= r["found_fraction"] == 1.0 and abs(r["chirp_volume_mpc3"] - v_true) < 3 * mc_se
        out["E1"] = {"v_true": v_true, "mc_se": mc_se, "res": res["at_far"]}
        allst = np.concatenate(list(bg_stat.values()))
        r1 = res["at_far"]["1/day"]
        n_allowed = int(np.floor(365.25 * res["background_livetime_days"] / 365.25))
        above = int((allst > r1["threshold"]).sum())
        out["E3"] = {"allowed": n_allowed, "above": above}
        ok &= above == n_allowed
        for r in range(I.K):
            (d / f"fg_r{r}.csv").unlink()
        res0 = E.evaluate(d)
        ok &= all(v.get("chirp_volume_mpc3", 0) == 0 for v in res0["at_far"].values() if v["reachable"])
        out["E2"] = res0["at_far"]
    c = E.cluster(np.array([[100.0, 1.0], [100.3, 5.0], [100.6, 2.0], [105.0, 3.0]]))
    ok &= sorted(c[:, 1].tolist()) == [3.0, 5.0]
    out["E4"] = c.tolist()
    out["pass"] = bool(ok)
    (B / "results").mkdir(exist_ok=True)
    (B / "results" / "golden_evaluate.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("E3", "E4", "pass")}), "\nE1 1/day:", out["E1"]["res"]["1/day"], out["E1"]["v_true"], out["E1"]["mc_se"])
    print("GOLDEN", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
