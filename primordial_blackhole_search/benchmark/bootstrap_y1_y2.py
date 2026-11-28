"""Y1 / Y2 sensitive-distance ratio with a paired bootstrap over injections -> results/y1_vs_y2_bootstrap.json.

Added in v1.0.1. The v1.0.0 release committed this file's output but not the script that wrote it, and that script
was not recoverable. This one was rebuilt from the documented design and checked against the committed file, which it
reproduces byte for byte (`--check`). The design:
  * per injection, found = its loudest trigger within tolerance is above the threshold at that FAR, using
    evaluate.py's own background(), injection_loudest() and threshold rule (thresholds fixed, NOT resampled);
  * distance ratio = (V_Y1 / V_Y2)^(1/3), with V proportional to the mean over injections of 4 pi d_c^2 found;
  * 2,000 paired resamples of the 1,344 injections (both yardsticks see the same resampled set);
  * 90 % interval = numpy.quantile at 0.05 and 0.95 (default linear method);
  * one generator, numpy.random.default_rng(0), consumed in the order 1/day own, 1/day fixed1, 1/week own,
    1/week fixed1. This seed and order are the only combination of seeds 0-499 and all 24 orders that reproduces
    the committed intervals, which is how they were recovered.
So the interval resamples injections only; it does not include the threshold's uncertainty (README caveat 1).

  python benchmark/bootstrap_y1_y2.py                  # writes results/y1_vs_y2_bootstrap.json
  python benchmark/bootstrap_y1_y2.py --check          # recompute and compare with the committed file, byte for byte
Needs the Y1 submission (regenerate it: copy results/y1_bins/ to state/y1/, then `y1_bank.py submit`) and the
committed Y2 submission. Uses pycbc only through evaluate.injection_table().
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

B = Path(__file__).resolve().parent
sys.path.insert(0, str(B))
import evaluate as E  # noqa: E402

N_BOOT, SEED = 2000, 0
FARS = (("1/day", 365.25), ("1/week", 52.18))           # evaluate.FARS_PER_YEAR
OUT = B / "results" / "y1_vs_y2_bootstrap.json"


def found_flags(sub, inj, fixed_var=None):
    bg, t_bg, _ = E.background(sub)
    loudest, _ = E.injection_loudest(sub, inj, fixed_var)
    out = {}
    for far, per_year in FARS:
        k = int(np.floor(per_year * t_bg / E.YEAR_S))          # evaluate.py's threshold: exactly k triggers above
        out[far] = loudest > bg[k]
    return out


def main(y1, y2, out, check):
    inj = E.injection_table()
    w = 4 * np.pi * np.array([x["chirp_distance"] for x in inj]) ** 2
    n = len(inj)
    f1, f2o, f2f = found_flags(y1, inj), found_flags(y2, inj), found_flags(y2, inj, fixed_var=1.0)
    rng = np.random.default_rng(SEED)
    res = {}
    for far in ("1/day", "1/week"):
        for tag, f2 in (("own", f2o), ("fixed1", f2f)):
            a, b = f1[far], f2[far]
            idx = rng.integers(0, n, (N_BOOT, n))
            with np.errstate(divide="ignore"):
                r = ((w[idx] * a[idx]).mean(1) / (w[idx] * b[idx]).mean(1)) ** (1 / 3)
            res[f"{far} y1/y2_{tag}"] = dict(
                distance_ratio=round(float(((w * a).mean() / (w * b).mean()) ** (1 / 3)), 3),
                ci90=[round(float(np.quantile(r, 0.05)), 3), round(float(np.quantile(r, 0.95)), 3)],
                n_found=[int(a.sum()), int(b.sum())], both=int((a & b).sum()))
    text = json.dumps(res, indent=1)
    if check:
        ok = OUT.read_text() == text
        print(f"bootstrap reproduces {OUT.name} byte for byte: {ok}")
        return 0 if ok else 1
    Path(out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--y1", default=str(B / "submissions" / "y1"))
    ap.add_argument("--y2", default=str(B / "submissions" / "y2"))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    sys.exit(main(Path(a.y1), Path(a.y2), a.out, a.check))
