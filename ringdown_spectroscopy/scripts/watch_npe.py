#!/usr/bin/env python
"""Event-watcher stage (.venv 3.12): apply the amortized no-hair NPE (09) to one event -> JSON.

The NPE is amortized + start-time-marginalized by construction, so ANY event costs one forward pass (seconds).
Reads config {name, gps} (argv[1]); fetches+conditions the event strain exactly as 09's on-source path;
emits the (M, chi, delta) posterior + Kerr consistency. Reuses 09's SEG/N_SAMP + the pickled posterior.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

import rdlib
import sbilib
from sbilib import Embed  # load-bearing: the pickled 09 posterior resolves __main__.Embed at load

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
FS, SEG = 4096.0, 0.04
N_SAMP = int(SEG * FS) + int(SEG * FS) % 2


def main() -> None:
    cfg = json.loads(Path(sys.argv[1]).read_text())
    gps = cfg["gps"]
    posterior = torch.load(RESULTS / "09_posterior_150k.pt", weights_only=False)
    segs = []
    for det in ("H1", "L1"):
        white = rdlib.fetch_whitened(det, gps, bandpass=False)
        pk = rdlib.find_peak(white.bandpass(*rdlib.BAND), gps)
        segs.append(white.crop(pk, pk + SEG + 0.01).value[:N_SAMP])
    x = torch.tensor(np.stack(segs).reshape(1, -1).astype(np.float32))
    s = posterior.sample((3000,), x=x, show_progress_bars=False).numpy()
    q = lambda j: [float(v) for v in np.percentile(s[:, j], [50, 5, 95])]
    m_q, c_q, d_q = q(0), q(1), q(2)
    out = {"stage": "npe_nohair", "M": m_q, "chi": c_q, "delta": d_q,
           "kerr_consistent_90": bool(d_q[1] <= 0.0 <= d_q[2])}
    Path(cfg["_out"]).write_text(json.dumps(out, indent=2))
    print(f"[npe] {cfg['name']}: M {m_q[0]:.1f}, chi {c_q[0]:.2f}, delta {d_q[0]:+.2f} "
          f"(Kerr {'ok' if out['kerr_consistent_90'] else 'EXCLUDED'})", flush=True)


if __name__ == "__main__":
    main()
