"""A5 (SISTER_REQUESTS / TheBridge): export raw 220/221 ringdown tone fits per event.

The bridge's A5 (precise multi-event no-hair) needs per-event raw tone fits — f_220, τ_220, f_221, τ_221,
amplitudes — like §06 produces for GW250114; §13 exposes only the NPE δ posterior. This runs the §06-style
CLASSICAL fit (joint H1+L1, shared f/τ across detectors, per-detector amp/φ; rdlib.fit_modes) on each §13
event and exports the tone parameters + the 220 inversion (M, χ via the qnm map). The bridge inverts each
220 via exact Leaver → predicts the 221 → a per-event precise δ → stacks (extending Move B v2 to the catalog).

HONEST CAVEAT (from §06): the FREE two-tone fit cannot reliably split the ~6-Hz-separated 220/221 at this
SNR — only the 220 is robust. We export BOTH the 1-tone (reliable 220) and the 2-tone (220 + low-confidence
221) fits, with a `tone221_reliable: false` flag. The bridge's plan (invert the *220*, predict the 221) only
needs the robust 220 tone.

Run:  .venv/bin/python scripts/18_export_tonefits.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from gwpy.timeseries import TimeSeries

import rdlib

RESULTS = Path(__file__).resolve().parent.parent / "results"
kerr220, kerr221 = rdlib.KerrMap(2, 2, 0), rdlib.KerrMap(2, 2, 1)

# the §13 catalog (GWOSC names)
CANDIDATES = ["GW250114_082203", "GW150914", "GW170814", "GW170104",
              "GW200129_065458", "GW190828_063405", "GW170809", "GW170818"]
ONE_TONE_KW = dict(n_modes=1, f_bounds=((180.0, 320.0),), tau_bounds=((2e-3, 0.012),), n_restarts=40, window=0.03)
TWO_TONE_KW = dict(n_modes=2, f_bounds=((180.0, 320.0), (180.0, 320.0)),
                   tau_bounds=((2e-3, 0.012), (4e-4, 2.5e-3)), n_restarts=40, window=0.03)


def load_series(ev):
    """{det: (times, values)} of whitened strain around the ringdown peak (retry the flaky open-data fetch)."""
    gps = rdlib.event_gps(ev)
    series, pk = {}, None
    for det in ("H1", "L1"):
        w = None
        for _ in range(5):
            try:
                w = rdlib.fetch_whitened(det, gps, bandpass=False); break
            except Exception:
                continue
        if w is None:
            raise RuntimeError(f"fetch failed for {det}")
        p = rdlib.find_peak(w.bandpass(*rdlib.BAND), gps)
        pk = p if pk is None else pk
        seg = w.crop(p - 0.01, p + 0.06)
        series[det] = (seg.times.value, seg.value)
    return series, pk


def invert(km, f, tau):
    try:
        M, chi = km.mass_chi(f, tau)
        return float(M), float(chi)
    except ValueError:
        return None, None


def main() -> None:
    out = {}
    print(f"{'event':>18} | {'220: f/τ -> M,χ':>34} | {'221: f/τ (low-confidence)':>26}")
    for ev in CANDIDATES:
        try:
            series, pk = load_series(ev)
        except Exception as e:
            out[ev] = None
            print(f"{ev:>18} | skipped ({type(e).__name__})")
            continue
        dets = sorted(series)
        # 1-tone: the robust 220 (amp/phi live in per_det[det][mode_index])
        f1 = rdlib.fit_modes(series, pk, seed=1, **ONE_TONE_KW)
        m220 = f1["modes"][0]
        M1, c1 = invert(kerr220, m220["f"], m220["tau"])
        # 2-tone: 220 + (low-confidence) 221 (modes sorted: index 0 = longest-τ = 220, index 1 = 221)
        f2 = rdlib.fit_modes(series, pk, seed=1, **TWO_TONE_KW)
        a, b = f2["modes"]
        M2, c2 = invert(kerr220, a["f"], a["tau"])
        # reliability of the 220: railed if f or τ pinned at a bound, or the (M,χ) inversion failed
        fb, tb = ONE_TONE_KW["f_bounds"][0], ONE_TONE_KW["tau_bounds"][0]
        railed = (abs(m220["f"] - fb[0]) < 1 or abs(m220["f"] - fb[1]) < 1
                  or abs(m220["tau"] - tb[0]) < 1e-4 or abs(m220["tau"] - tb[1]) < 1e-4 or M1 is None)
        rec = {
            "gps": float(rdlib.event_gps(ev)),
            "tone220_1mode": {"f": float(m220["f"]), "tau": float(m220["tau"]),
                              "amp": {d: f1["per_det"][d][0]["amp"] for d in dets},
                              "phi": {d: f1["per_det"][d][0]["phi"] for d in dets},
                              "M_inv": M1, "chi_inv": c1, "cost": f1["cost"]},
            "twotone": {"f220": float(a["f"]), "tau220": float(a["tau"]),
                        "f221": float(b["f"]), "tau221": float(b["tau"]),
                        "amp220": {d: f2["per_det"][d][0]["amp"] for d in dets},
                        "amp221": {d: f2["per_det"][d][1]["amp"] for d in dets},
                        "phi220": {d: f2["per_det"][d][0]["phi"] for d in dets},
                        "phi221": {d: f2["per_det"][d][1]["phi"] for d in dets},
                        "M220_inv": M2, "chi220_inv": c2, "cost": f2["cost"]},
            "tone220_railed": bool(railed),                      # True -> 220 fit hit a bound, do NOT trust
            "tone221_reliable": False,                           # §06: free 2-tone can't split tones at this SNR
        }
        out[ev] = rec
        print(f"{ev:>18} | f={m220['f']:6.1f} τ={m220['tau']*1e3:4.2f}ms -> "
              f"M={M1 if M1 else float('nan'):5.1f} χ={c1 if c1 else float('nan'):.2f}"
              f"{'  <-- RAILED (untrustworthy)' if railed else ''} | f221={b['f']:6.1f} τ221={b['tau']*1e3:4.2f}ms")

    (RESULTS / "18_tonefits.json").write_text(json.dumps(out, indent=2))
    n_ok = sum(v is not None for v in out.values())
    n_rel = sum(v is not None and not v["tone220_railed"] for v in out.values())
    print(f"\nexported {n_ok}/{len(CANDIDATES)} events -> results/18_tonefits.json; "
          f"{n_rel} have a RELIABLE (non-railed) 220 tone. 221 flagged low-confidence per §06. For TheBridge A5.")


if __name__ == "__main__":
    main()
