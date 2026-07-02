#!/usr/bin/env python
"""Milestone 19 (E3, PLAN.md): per-event ML scorers + per-event Δt across the broadened event set.

07 trained ONE ML scorer on GW150914; run_event ran the v1 COMB on 3 events. E3 closes the loop: train a
per-event autoencoder scorer (per detector, on that event's own off-source pool) and run the v2 ML network
comb at each event's formula-predicted Δt, for the broadened set GW150914 / GW151012 / GW151226 / GW250114.
Deliverable = an honest per-event table of on-source nulls (ML p-value vs the v1 comb baseline) + the
per-event detection floor.

Δt per event (verified): GW150914 0.2925, GW151012 0.1778, GW151226 0.1013 (Abedi Table I, reproduced by
14_echo_spacing to <2%); GW250114 computed from its detector-frame remnant (M_f=68.1 M☉, χ=0.68; LSC/
arXiv:2509.08099) via the verified echo_spacing formula. Reuses 07's ConvAE/train_scorer/error_envelope.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from gwpy.timeseries import TimeSeries

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from echolib import DATA, DETECTORS, RESULTS, comb_score, event_gps, load_segments, whiten_bp, _longest_finite

# ±hour offsets whose blocks were prefetched (13_independent_bg convention) -> independent background pool
OFFSETS = [o * s for o in (1800, 3600, 7200, 14400, 28800, 57600) for s in (-1, 1)]
SEG_DUR, SPAN, EDGE = 3.0, 512.0, 8.0


def offset_off(event: str, det: str, band: tuple) -> list:
    """Whitened 3-s off-source segments from every available ±hour block (band-aware; independent of the
    on-source block, so a truly out-of-time background). Reads only cached blocks; missing offsets skipped."""
    out = []
    for off in OFFSETS:
        cache = DATA / f"{event}_off{int(off)}_{det}_{int(SPAN)}s.hdf5"
        if not cache.exists():
            continue
        try:
            finite = _longest_finite(TimeSeries.read(cache))
            if float(finite.duration.value) < 64.0:
                continue
            block = whiten_bp(finite, band)
        except Exception:
            continue
        fs = float(block.sample_rate.value); t0 = float(block.t0.value); x = block.value
        t = t0 + EDGE
        n = int(round(SEG_DUR * fs))
        while t + SEG_DUR < t0 + (len(x) / fs) - EDGE:
            i = int(round((t - t0) * fs)); seg = x[i:i + n]
            if len(seg) == n:
                out.append(seg)
            t += SEG_DUR
    return out

# reuse 07's scorer machinery (module name starts with a digit -> importlib)
_spec = importlib.util.spec_from_file_location("ml07", HERE / "07_ml_scorer.py")
ml07 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(ml07)
_spec14 = importlib.util.spec_from_file_location("es14", HERE / "14_echo_spacing.py")
es14 = importlib.util.module_from_spec(_spec14); _spec14.loader.exec_module(es14)

BAND = {"GW150914": (30., 350.), "GW151012": (30., 600.), "GW151226": (30., 900.), "GW250114_082203": (30., 350.)}
DT_PRED = {  # verified Δt (s): O1 from Abedi Table I; GW250114 from the formula on its remnant
    "GW150914": 0.2925, "GW151012": 0.1778, "GW151226": 0.1013,
    "GW250114_082203": float(es14.echo_spacing(68.1, 0.68)),
}


def pval(stat: float, bg: np.ndarray) -> float:
    return float((1 + np.sum(bg >= stat)) / (1 + len(bg)))


def run_event(event: str, rng) -> dict:
    band = BAND[event]
    dt_pred = DT_PRED[event]
    dt_grid = np.arange(max(0.05, dt_pred / 2), min(0.5, 2 * dt_pred) + 1e-9, 0.005)
    j = int(np.argmin(np.abs(dt_grid - dt_pred)))
    segs = {det: load_segments(event, det, band=band) for det in DETECTORS}
    fs = segs["H1"].fs

    # per-detector scorers trained on this event's own off-source pool (whitened noise model)
    n_own = min(len(segs[d].off) for d in DETECTORS)
    if n_own < 40:
        return {"event": event, "error": f"too few own-block off-source segments to train ({n_own})"}
    models = {det: ml07.train_scorer(segs[det].off) for det in DETECTORS}

    # background from the INDEPENDENT ±hour blocks (each own-PSD whitened, E2-style -> out-of-time,
    # non-stationarity-safe, and far larger than the 512-s own block -> sharp p-values + rescues NaN-cropped events)
    off = {det: offset_off(event, det, band) for det in DETECTORS}
    n_bg = min(len(off[d]) for d in DETECTORS)
    if n_bg < 50:
        return {"event": event, "error": f"too few independent background segments ({n_bg})"}
    bg_ml, bg_comb = [], []
    for i in range(n_bg):
        ml_tot = cb_tot = 0.0
        for det in DETECTORS:
            ml_tot += ml07.comb_on_env(ml07.error_envelope(models[det], off[det][i], fs), fs, dt_grid)[j]
            cb_tot += comb_score(off[det][i], fs, dt_grid)[j]
        bg_ml.append(ml_tot); bg_comb.append(cb_tot)
    bg_ml, bg_comb = np.array(bg_ml), np.array(bg_comb)

    # on-source statistic at the predicted Δt
    on_ml = np.zeros(len(dt_grid)); on_cb = np.zeros(len(dt_grid))
    for det in DETECTORS:
        on_ml += ml07.comb_on_env(ml07.error_envelope(models[det], segs[det].on, fs), fs, dt_grid)
        on_cb += comb_score(segs[det].on, fs, dt_grid)
    return {
        "event": event, "dt_pred": dt_pred, "band": band, "n_train_own": n_own, "n_bg": n_bg,
        "ml_p_at_dt": pval(on_ml[j], bg_ml), "comb_p_at_dt": pval(on_cb[j], bg_comb),
    }


def main() -> None:
    rng = np.random.default_rng(2026)
    rows = []
    print(f"{'event':>16} {'Δt':>7} {'n_bg':>5} | {'ML p@Δt':>8} {'comb p@Δt':>9}")
    for ev in ("GW150914", "GW151012", "GW151226", "GW250114_082203"):
        r = run_event(ev, rng)
        rows.append(r)
        if "error" in r:
            print(f"{ev:>16}  {r['error']}"); continue
        print(f"{ev:>16} {r['dt_pred']:>7.4f} {r['n_bg']:>5} | {r['ml_p_at_dt']:>8.3f} {r['comb_p_at_dt']:>9.3f}", flush=True)
    (RESULTS / "19_per_event_ml.json").write_text(json.dumps({"background": "independent ±hour blocks, own-PSD whitened (E2-style)", "events": rows}, indent=2))
    print(f"wrote {RESULTS/'19_per_event_ml.json'}")


if __name__ == "__main__":
    main()
