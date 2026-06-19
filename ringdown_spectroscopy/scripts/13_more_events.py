#!/usr/bin/env python
"""Milestone 13 — STRESS-TEST: can we trust the no-hair NPE on MORE real events?

Before stacking more events into the δ test (12), we must verify the network is
TRUSTWORTHY per event — not just that δ "looks sensible". The check: does the NPE
recover each event's remnant (M, χ) consistent with the published value (allowing
the known +10% mass pull from v2/v3)? An event only earns a place in the stack if
it passes. Reports (M, χ, δ) with 90% CIs per event for the cross-check.

Robustness is the only north star (memory). Run:  python 13_more_events.py
"""
import json
from pathlib import Path

import numpy as np
import torch

import rdlib
import sbilib
from sbilib import Embed, N_SAMP, SEG  # Embed at module level: pickled posterior resolves it

rng = np.random.default_rng(0)
RESULTS = Path(__file__).resolve().parent.parent / "results"

# candidate confident BBH events (GWOSC names). Remnant masses cross-checked after.
CANDIDATES = [
    "GW250114_082203", "GW150914", "GW170814", "GW170104",
    "GW200129_065458", "GW190828_063405", "GW170809", "GW170818",
]


def main() -> None:
    posterior = torch.load(RESULTS / "09_posterior_150k.pt", weights_only=False)
    T = json.loads((RESULTS / "10_recalibration.json").read_text())["T"]
    print(f"loaded NPE + recalibration T={T}\n")

    def post(x_obs):
        s = posterior.sample((2000,), x=x_obs, show_progress_bars=False).numpy()
        # recalibrate each parameter (v3 temperature widening about its median)
        s = np.median(s, 0) + T * (s - np.median(s, 0))
        q = lambda j: np.percentile(s[:, j], [50, 5, 95])
        return q(0), q(1), q(2)

    out = {}
    print(f"{'event':>18} | {'M [90% CI]':>22} | {'chi [90% CI]':>20} | {'delta [90% CI]':>22}")
    for ev in CANDIDATES:
        try:
            gps = rdlib.event_gps(ev)
            segs = []
            for det in ("H1", "L1"):
                white = rdlib.fetch_whitened(det, gps, bandpass=False)
                pk = rdlib.find_peak(white.bandpass(*rdlib.BAND), gps)
                seg = white.crop(pk, pk + SEG + 0.01).value[:N_SAMP]
                if len(seg) != N_SAMP or not np.isfinite(seg).all():
                    raise ValueError("bad segment")
                segs.append(seg)
            x = torch.tensor(np.stack(segs).reshape(1, -1).astype(np.float32))
            m, c, d = post(x)
            out[ev] = dict(M=list(m), chi=list(c), delta=list(d))
            # rail flag: posterior pinned at the prior edge (M in [40,120], chi in [0.05,0.95])
            rail = (m[0] > 116 or m[0] < 44 or c[0] > 0.92 or c[0] < 0.08)
            print(f"{ev:>18} | {m[0]:5.1f} [{m[1]:5.1f},{m[2]:5.1f}] | "
                  f"{c[0]:.2f} [{c[1]:.2f},{c[2]:.2f}] | "
                  f"{d[0]:+.2f} [{d[1]:+.2f},{d[2]:+.2f}]" + ("   <-- RAILED" if rail else ""))
        except Exception as e:
            out[ev] = None
            print(f"{ev:>18} | skipped ({e})")

    (RESULTS / "13_more_events.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {RESULTS/'13_more_events.json'}")
    print("NEXT: cross-check recovered (M, chi) vs PUBLISHED remnant values; only "
          "events that match (allowing the +10% mass pull) earn a stack slot.")


if __name__ == "__main__":
    main()
