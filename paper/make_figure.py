"""Figure 1 of the LongChirp preprint, drawn only from files committed at the release tag longchirp-v1.0.0.

Reads (via `git show longchirp-v1.0.0:<path>`, so the figure cannot drift from the release):
  results/y1_eval_own.json, results/y2_eval_own.json, results/y2_eval_fixed1.json   (Y1, Y2 own var, Y2 at +-1 s)
  results/oracle_snr.csv + injections.csv                                            (Y0, recomputed here)
Run from the repo root:  primordial_blackhole_search/.venv/bin/python paper/make_figure.py
"""
import csv
import io
import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

TAG, BASE = "longchirp-v1.0.0", "primordial_blackhole_search/benchmark/"
OUT = Path(__file__).resolve().parent / "fig_distances.pdf"


def show(p):
    return subprocess.run(["git", "show", f"{TAG}:{BASE}{p}"], capture_output=True, text=True, check=True).stdout


def y0():
    inj = list(csv.DictReader(io.StringIO(show("injections.csv"))))
    snr = {int(r["id"]): float(r["snr_net"]) for r in csv.DictReader(io.StringIO(show("results/oracle_snr.csv")))}
    dc = np.array([float(r["chirp_distance"]) for r in inj])
    s = np.array([snr[int(r["id"])] for r in inj])
    out = {}
    for t in (8, 10, 12):                       # evaluate.py's chirp-volume convention, d_c uniform on [5, 500] Mpc
        v = (500 - 5) * np.mean(4 * np.pi * dc ** 2 * (s >= t))
        out[t] = (3 * v / (4 * np.pi)) ** (1 / 3)
    return out


rows = [("Y1  matched-filter bank\n(var = 1 s)", "results/y1_eval_own.json"),
        ("Y2  CNN\n(own var = 20.2 s)", "results/y2_eval_own.json"),
        ("Y2  CNN\n(fixed var = 1 s)", "results/y2_eval_fixed1.json")]
fig, ax = plt.subplots(figsize=(6.4, 2.9))
for j, (label, f) in enumerate(rows):
    r = json.loads(show(f))["at_far"]
    for far, mk, fc in (("1/day", "o", "C0"), ("1/week", "s", "none")):
        d = r[far]["chirp_distance_mpc"]
        ax.plot(d, j, mk, ms=8, mfc=fc, mec="C0", mew=1.5, label=f"FAR {far}" if j == 0 else None)
    ax.annotate(f'{r["1/day"]["chirp_distance_mpc"]:.1f} / {r["1/week"]["chirp_distance_mpc"]:.1f} Mpc',
                (r["1/day"]["chirp_distance_mpc"], j), xytext=(10, 6), textcoords="offset points", fontsize=8)
for t, d in y0().items():
    ax.axvline(d, color="0.5", ls=":", lw=1)
    ax.text(d + 1.5, 2.45, f"Y0: SNR$\\geq${t}, {d:.0f} Mpc", fontsize=7, rotation=90, ha="left", va="bottom",
            color="0.35")
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows], fontsize=8)
ax.set_ylim(-0.6, 2.5)
ax.invert_yaxis()
ax.set_xlim(0, 175)
ax.set_xlabel("sensitive chirp distance (Mpc)")
ax.legend(loc="lower left", bbox_to_anchor=(0.33, 0.0), fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(OUT)
print(f"wrote {OUT}")
