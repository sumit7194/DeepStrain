"""Did fixing the warm-up asymmetry just FLIP it? The symmetric check the inverted gate cannot make.

WHY (bridge, 2026-09-21). The L1 bug was an ASYMMETRY: one cold `oaconvolve` multiplied by 8, against a
`segment_stats` that ran first and amortised its warm-up across 32 transforms inside one timed region. The
fix warmed the FIR path. If the FFT path is now the cold one, the bias has flipped sign and KEPT ITS
MAGNITUDE — and the gate inverted to guard the positive would guard that just as faithfully, because it
checks that the positive does not regress, not that the positive is right.

So measure both paths in BOTH states, and in BOTH ORDERS, because whichever runs first warms shared state
(FFTW-style plan caches, the allocator's arenas, the page cache):

    cold  = first call in a fresh process, no warm-up
    warm  = median of 5 after a discard
    order = direct-first and ratio-first, run as separate processes

If the cold ratio and the warm ratio agree, the 3.74x is a property of the work. If cold and warm disagree
sharply, the number is a warm-up artifact and which direction it points depends only on who ran first.

Run:  .venv/bin/python scripts/bank_ratio_warmup.py --order direct   (and --order ratio)
"""
import argparse
import importlib.util
import json
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.signal import oaconvolve

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pbh import config as C
from pbh import ratiofilter as rf
from pbh.data import whiten_segment
from pbh.waveforms import make_whitened_injection, sample_params

HERE = Path(__file__).resolve().parent
_so = importlib.util.spec_from_file_location("so", HERE / "semicoherent_oracle.py")
so = importlib.util.module_from_spec(_so); _so.loader.exec_module(so)
TAPS, N_CHUNK, EQ, MC, REPS = 16385, 8, 2.0 ** 0.2, 0.30, 5
PAGEIN_FACTOR = 3.0          # timed window may exceed its own control by this much before it is discarded


def pageins() -> int:
    """Cumulative page-ins. A residency measurement taken while the machine faults from swap measures the
    wrong level of the hierarchy -- disk, not cache."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.startswith("Pageins"):
            return int(line.split(":")[1].strip().rstrip("."))
    return -1


def paired_pagein_gate(work, reps=REPS):
    """Time `work`, with a CONTROL window of the same length immediately before it.

    ansatz's design, 2026-09-21, and it replaces a threshold I could not have chosen. They measured this box
    idle-ish at +3,155 then +234 page-ins over consecutive 20 s windows -- a 13x drift with nothing different
    happening -- while tabula measured page-OUTs flat at zero. The box faults pages back from swap it
    populated earlier and never reclaimed. So an absolute bar is unknowable: "any movement" discards
    everything and "more than 3,000" would have passed the second window and failed the first.

    A PAIRED comparison fixes it -- the timed window must not be an outlier against its own neighbour, which
    is the same move as timing warm medians instead of one cold run. The deltas are recorded either way, so a
    later reader can tell whether a surviving number survived because the box was clean or in spite of it.
    """
    work()                                              # warm; the first call is not the steady state
    t = time.perf_counter()
    for _ in range(reps):
        work()
    span = (time.perf_counter() - t) / reps

    p0 = pageins(); time.sleep(span * reps); p1 = pageins()      # control: same length, same state, no work
    ctrl = p1 - p0

    ts = []
    p2 = pageins()
    for _ in range(reps):
        t = time.perf_counter(); work(); ts.append(time.perf_counter() - t)
    timed = pageins() - p2

    clean = timed <= PAGEIN_FACTOR * max(ctrl, 1)
    return {"median_s": float(np.median(ts)), "pageins_control": ctrl, "pageins_timed": timed,
            "pagein_clean": bool(clean)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", choices=("direct", "ratio"), default="direct")
    args = ap.parse_args()

    gps = json.loads((C.DATA_DIR / "manifest.json").read_text())["H1"]["test"][0]
    w, t0, psd = whiten_segment("H1", gps)
    wc = w[so.CROP:-so.CROP]
    base = sample_params(np.random.default_rng(0))
    h, _ = make_whitened_injection(replace(base, mass1=MC * EQ, mass2=MC * EQ), "H1", t0, psd)
    g = h[-so.WIN:].copy()
    g = g if len(g) == so.WIN else np.pad(g, (so.WIN - len(g), 0))
    ch = so.analytic_chunks(g, N_CHUNK)
    N = len(wc)
    c_ref = rf.corr_series(wc, np.pad(ch[0][1], (0, N - len(ch[0][1]))))
    taps = np.random.default_rng(0).standard_normal(TAPS) + 1j * np.random.default_rng(1).standard_normal(TAPS)

    def direct():
        return so.segment_stats(wc, ch)

    def ratio():
        for _ in range(len(ch)):
            oaconvolve(c_ref, taps, mode="same")

    fns = {"direct": direct, "ratio": ratio}
    first, second = (args.order, "ratio" if args.order == "direct" else "direct")

    out = {"order": args.order, "n": N, "taps": TAPS, "n_chunk": len(ch), "reps": REPS}
    print(f"N={N:,}  K={TAPS:,}  chunks={len(ch)}  order: {first} first\n")
    for name in (first, second):                       # COLD phase, in the declared order
        t = time.perf_counter(); fns[name](); out[f"{name}_cold_s"] = time.perf_counter() - t
        print(f"  {name:>6} COLD  {out[f'{name}_cold_s']:6.2f}s")
    for name in (first, second):                       # WARM phase, same order, paired page-in control
        g = paired_pagein_gate(fns[name])
        out[f"{name}_warm_s"] = g["median_s"]
        out[f"{name}_pageins_control"] = g["pageins_control"]
        out[f"{name}_pageins_timed"] = g["pageins_timed"]
        out[f"{name}_pagein_clean"] = g["pagein_clean"]
        print(f"  {name:>6} WARM  {g['median_s']:6.2f}s  (median of {REPS})  page-ins "
              f"{g['pageins_timed']} vs control {g['pageins_control']}  "
              f"{'CLEAN' if g['pagein_clean'] else 'OUTLIER -- discard and retry'}")

    out["cold_speedup"] = out["direct_cold_s"] / out["ratio_cold_s"]
    out["warm_speedup"] = out["direct_warm_s"] / out["ratio_warm_s"]
    out["cold_warm_disagreement"] = abs(out["cold_speedup"] - out["warm_speedup"]) / out["warm_speedup"]
    out["pagein_clean"] = bool(out.get("direct_pagein_clean") and out.get("ratio_pagein_clean"))
    if not out["pagein_clean"]:
        print("  ⚠️  page-ins were an outlier against their own control: this number is NOT reportable as a "
              "residency measurement -- it may be timing disk, not cache. Retry on a quieter box.")
    print(f"\n  speedup COLD {out['cold_speedup']:.2f}x   WARM {out['warm_speedup']:.2f}x   "
          f"disagreement {100*out['cold_warm_disagreement']:.0f}%")
    p = C.RESULTS_DIR / f"bank_ratio_warmup_{args.order}first.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"wrote {p.name}")


if __name__ == "__main__":
    main()
