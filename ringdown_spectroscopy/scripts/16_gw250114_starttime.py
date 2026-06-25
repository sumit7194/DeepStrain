"""R3 capstone (PLAN.md): apply the R3 lesson to the REAL GW250114 — the no-hair δ vs ringdown start time.

R3 (15_imr_referee) showed the no-hair NPE carries a δ≈−0.33 systematic on realistic ringdowns analyzed from
the peak, decaying to ~0 by ~6 ms post-peak (the early merger/overtone content the 220+221 model omits). This
runs the SAME start-time sweep on the real GW250114 data: crop the whitened ringdown starting at peak+offset for
offset ∈ {0…12 ms} and run the NPE at each. If the real δ drifts toward 0 as the start moves later (mirroring the
injection bias decay), the headline 09 value (δ=−0.16, peak-cropped) was partly biased, and the late-start value
is the systematic-mitigated δ (at the cost of ringdown SNR → a wider posterior — the v6 wall).

Run:  .venv/bin/python scripts/16_gw250114_starttime.py
"""
import json
from pathlib import Path

import numpy as np
import torch

import rdlib
from sbilib import Embed, FS, N_SAMP   # Embed needed to unpickle the 09 posterior

RESULTS = Path(__file__).resolve().parent.parent / "results"
PLOTS = Path(__file__).resolve().parent.parent / "plots"
EVENT = "GW250114_082203"
OFFSETS_MS = [0.0, 2.0, 4.0, 6.0, 8.0, 12.0]


def main():
    torch.manual_seed(0)
    posterior = torch.load(RESULTS / "09_posterior_150k.pt", weights_only=False)
    T = json.loads((RESULTS / "10_recalibration.json").read_text())["T"]
    gps = rdlib.event_gps(EVENT)

    # fetch + whiten once per detector, find the peak (retry the open-data fetch — recent event, flaky TLS)
    whites, peaks = {}, {}
    for det in ("H1", "L1"):
        for attempt in range(5):
            try:
                w = rdlib.fetch_whitened(det, gps, bandpass=False)
                whites[det] = w
                peaks[det] = rdlib.find_peak(w.bandpass(*rdlib.BAND), gps)
                break
            except Exception as e:
                print(f"  {det} fetch attempt {attempt+1}: {type(e).__name__}; retry", flush=True)
        else:
            raise RuntimeError(f"could not fetch {det}")
    print(f"{EVENT}: peaks found; sweeping start offset (NPE T={T})\n")

    @torch.no_grad()
    def delta_post(segs):
        x = np.stack(segs).reshape(1, -1)
        s = posterior.sample((2000,), x=torch.tensor(x), show_progress_bars=False).numpy()[:, 2]
        s = np.median(s) + T * (s - np.median(s))                  # v3 recalibration
        return float(np.median(s)), float(np.percentile(s, 5)), float(np.percentile(s, 95))

    rows = {}
    print(f"{'start +ms':>10} {'δ median':>9} {'90% CI':>20}  Kerr?")
    for om in OFFSETS_MS:
        segs = []
        for det in ("H1", "L1"):
            t0 = peaks[det] + om / 1000.0
            segs.append(whites[det].crop(t0, t0 + N_SAMP / FS + 0.01).value[:N_SAMP].astype(float))
        med, lo, hi = delta_post(segs)
        kerr = lo <= 0.0 <= hi
        rows[om] = dict(median=med, lo=lo, hi=hi, kerr_inside_90=kerr)
        print(f"{om:>10.0f} {med:>+9.3f} {f'[{lo:+.2f}, {hi:+.2f}]':>20}  {'YES' if kerr else 'no'}")

    d0, d_late = rows[0.0]["median"], rows[OFFSETS_MS[-1]]["median"]
    drift = d_late - d0
    print(f"\nGW250114 δ: peak-cropped {d0:+.3f}  ->  late-start ({OFFSETS_MS[-1]:.0f}ms) {d_late:+.3f}  "
          f"(drift {drift:+.3f})")
    print(f"interpretation: {'δ drifts toward 0 at later start, as R3 predicts -> the peak value carried the systematic; ' if drift > 0.05 and abs(d_late) < abs(d0) else 'δ is start-time-stable here -> '}"
          f"all offsets remain Kerr-consistent: {all(r['kerr_inside_90'] for r in rows.values())}")

    (RESULTS / "16_gw250114_starttime.json").write_text(json.dumps(
        {"event": EVENT, "T": T, "offsets_ms": OFFSETS_MS, "delta_vs_start": rows,
         "peak_to_late_drift": drift, "all_kerr_consistent": all(r["kerr_inside_90"] for r in rows.values())},
        indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    oms = OFFSETS_MS
    med = [rows[o]["median"] for o in oms]
    lo = [rows[o]["lo"] for o in oms]; hi = [rows[o]["hi"] for o in oms]
    ax.fill_between(oms, lo, hi, alpha=0.2, color="steelblue", label="90% CI")
    ax.plot(oms, med, "o-", color="navy", label="GW250114 δ (real)")
    ax.axhline(0, color="k", lw=1.2, label="Kerr (δ=0)")
    ax.set_xlabel("ringdown analysis start, ms after peak"); ax.set_ylabel("no-hair deviation δ")
    ax.set_title("GW250114 no-hair δ vs start time (R3 capstone)\n"
                 "R3 systematic decays by ~6ms; the late-start δ is systematic-mitigated")
    ax.legend(); fig.tight_layout()
    fig.savefig(PLOTS / "16_gw250114_starttime.png", dpi=140)
    print("wrote 16_gw250114_starttime.json + .png")


if __name__ == "__main__":
    main()
