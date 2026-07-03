#!/usr/bin/env python
"""Event-watcher stage (echoes venv): predict the echo Δt from the remnant + run the comb search -> JSON.

Reads config {name, M_f, chi_f, band?} (argv[1]). Δt from the verified Kerr-tortoise formula (14_echo_spacing)
on the event's remnant; comb search on-source vs the independent ±hour background (E2/E3 convention). Emits the
predicted Δt + the on-source p-value (a null unless there's a post-merger echo). Reuses 19_per_event_ml's
offset_off + echolib's comb.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from echolib import DETECTORS, comb_score, load_segments

_es = importlib.util.spec_from_file_location("es", HERE / "14_echo_spacing.py")
es = importlib.util.module_from_spec(_es); _es.loader.exec_module(es)
_pe = importlib.util.spec_from_file_location("pe", HERE / "19_per_event_ml.py")
pe = importlib.util.module_from_spec(_pe); _pe.loader.exec_module(pe)


def main() -> None:
    cfg = json.loads(Path(sys.argv[1]).read_text())
    name = cfg["name"]
    band = tuple(cfg.get("band", [30.0, 350.0]))
    dt_pred = float(es.echo_spacing(cfg["M_f"], cfg["chi_f"]))
    dt_grid = np.arange(max(0.05, dt_pred / 2), min(0.5, 2 * dt_pred) + 1e-9, 0.005)
    j = int(np.argmin(np.abs(dt_grid - dt_pred)))

    segs = {det: load_segments(name, det, band=band) for det in DETECTORS}
    fs = segs["H1"].fs
    off = {det: pe.offset_off(name, det, band) for det in DETECTORS}
    n_bg = min(len(off[d]) for d in DETECTORS)
    bg = np.array([sum(comb_score(off[det][i], fs, dt_grid)[j] for det in DETECTORS) for i in range(n_bg)])
    on = float(sum(comb_score(segs[det].on, fs, dt_grid)[j] for det in DETECTORS))
    p = float((1 + np.sum(bg >= on)) / (1 + len(bg)))
    out = {"stage": "echo_search", "dt_pred_s": dt_pred, "band": list(band), "n_bg": int(n_bg),
           "comb_p_value": p, "echo_detected": bool(p < 0.01)}
    Path(cfg["_out"]).write_text(json.dumps(out, indent=2))
    print(f"[echo] {name}: Δt {dt_pred*1e3:.1f} ms, comb p {p:.3f} (n_bg {n_bg}) "
          f"-> {'DETECTION' if out['echo_detected'] else 'null'}", flush=True)


if __name__ == "__main__":
    main()
