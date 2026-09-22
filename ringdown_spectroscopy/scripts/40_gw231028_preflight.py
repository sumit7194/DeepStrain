"""Preflight for a GW231028 no-hair NPE: does our simulator's idealisation survive at 240 Msun?

WHY. GW231028_153006 (arXiv:2509.08657: 221 overtone at BF ~189, remnant ~246 Msun redshifted, chi 0.81) is a
candidate second informative event for delta stacking. Our 09 NPE cannot be applied to it: its mass prior is
[40, 120] Msun, its window 40 ms (~2.6 damping times of a 240 Msun 220 mode, vs ~10 at GW250114), and its
simulator writes IDEAL damped sinusoids straight into unit white noise -- an idealisation validated by real-noise
injections only at 251 Hz. A 240 Msun ringdown sits near 80 Hz where the O4a ASD is steep, and whitening reshapes
a ringdown (v4 measured raw-vs-whitened shape overlap 0.48 for its tone-count setup). So before any training:

  P1  event amplitude: the whitened peak |x| of GW231028 in each detector, and the off-source whitened noise
      std, so a retrained network's amplitude range brackets the real event in the network's own units.
  P2  whitening fidelity: inject a 220 ringdown into RAW O4a strain near the event, whiten exactly as the
      pipeline does, subtract the whitened no-injection strain (whitening is linear) to isolate the whitened
      SIGNAL, and measure its best overlap with the IDEAL damped sinusoid the simulator uses (free amplitude
      and phase via a quadrature basis, +-2 ms time shift). CONTROL: the same at GW250114's (68, 0.69), where
      09's R2 real-noise injections passed -- the number to beat is the control's, not an absolute bar.

DECISION RULE (fixed before running): if the 240 Msun overlap is within 0.02 of the control, the idealised
simulator is kept (only prior, window and start-time range change); otherwise the retrain must use a
whitening-shaped simulator (sbilib.whiten_shape), because a network trained on shapes the data never has is
exactly how v4's tone-count model failed to transfer.

Run:  .venv/bin/python scripts/40_gw231028_preflight.py
"""
import json
from pathlib import Path

import numpy as np
from gwpy.timeseries import TimeSeries

import rdlib
from sbilib import kerr220

RESULTS = Path(__file__).resolve().parent.parent / "results"
OUT = RESULTS / "40_gw231028_preflight.json"
EVENT = "GW231028_153006"
FS = 4096.0


def fidelity(det, center, f, tau, win_s):
    """Best overlap of the whitened injected 220 with the ideal damped sinusoid over a window win_s."""
    raw = TimeSeries.fetch_open_data(det, center - 32, center + 32, cache=True)
    if not np.isfinite(raw.value).all():
        raise ValueError("NaNs in raw strain")
    amp = 5e-21
    # ⚠️ phi = pi/2 (a damped SINE, continuous at t0), and the overlap window starts AT t0. The first version
    # injected phi = 0 -- a damped cosine that JUMPS from 0 to full amplitude -- and cropped from 10 ms before
    # t0. Whitening is acausal: it smeared the step's broadband content backwards, 39-44% of the whitened
    # energy landed BEFORE t0 where the template is zero, and the metric read 0.47 (event) / 0.35 (validated
    # control). A metric that scores the setup 09's real-noise injections already validated at 0.35 is
    # measuring the injection's discontinuity, not the whitening. Continuous start: 0.975 / 0.960.
    inj = rdlib.inject_ringdown(raw, center, [dict(f=f, tau=tau, amp=amp, phi=np.pi / 2)])
    w_sig = inj.whiten(4, 2) - raw.whiten(4, 2)            # whitening is linear: this is the whitened signal
    seg = w_sig.crop(center, center + win_s).value
    t = np.arange(len(seg)) / FS
    best = 0.0
    for shift in range(-8, 9):                              # +-2 ms
        t0 = shift / FS
        c = rdlib.damped_sinusoids(t, t0, [dict(f=f, tau=tau, amp=1.0, phi=0.0)])
        s = rdlib.damped_sinusoids(t, t0, [dict(f=f, tau=tau, amp=1.0, phi=np.pi / 2)])
        B = np.column_stack([c, s])
        coef, *_ = np.linalg.lstsq(B, seg, rcond=None)
        fit = B @ coef
        ov = float(np.dot(fit, seg) / (np.linalg.norm(fit) * np.linalg.norm(seg)))
        best = max(best, ov)
    return best


def main() -> None:
    gps = rdlib.event_gps(EVENT)
    rep = {"event": EVENT, "gps": gps, "P1": {}, "P2": {}}
    print(f"{EVENT}  gps {gps}")

    # ---- P1: the event's whitened amplitude in the network's own units ----------------------------------
    for det in ("H1", "L1"):
        white = rdlib.fetch_whitened(det, gps, bandpass=False)
        pk = rdlib.find_peak(white.bandpass(*rdlib.BAND), gps)
        near = white.crop(pk - 0.005, pk + 0.005).value
        off = white.crop(gps - 20, gps - 4).value
        rep["P1"][det] = {"peak_gps": pk, "peak_abs": float(np.abs(near).max()), "offsource_std": float(off.std())}
        print(f"  P1 {det}: whitened peak |x| {np.abs(near).max():.2f}, off-source std {off.std():.3f}, "
              f"peak at gps {pk:.4f}")

    # ---- P2: whitening fidelity, event mass vs the validated control --------------------------------------
    cases = {"GW231028 (240, 0.81)": (EVENT, 240.0, 0.81), "control GW250114 (68, 0.69)": ("GW250114_082203", 68.0, 0.69)}
    for name, (ev, m, chi) in cases.items():
        f, tau = kerr220.f_tau(m, chi)
        base = rdlib.event_gps(ev)
        ovs = []
        for k in range(4):
            center = base - 300 - 128 * k
            try:
                ovs.append(fidelity("H1", center, f, tau, win_s=6 * tau))
            except Exception as exc:                                     # noqa: BLE001 -- reported, not hidden
                print(f"    [{name} noise {k}] skipped: {exc}")
        rep["P2"][name] = {"f_Hz": f, "tau_ms": 1e3 * tau, "overlaps": ovs, "median": float(np.median(ovs))}
        print(f"  P2 {name}: f {f:.1f} Hz, tau {1e3*tau:.2f} ms  ->  whitened-vs-ideal overlap "
              f"{np.median(ovs):.4f}  (per noise draw {[round(o, 4) for o in ovs]})")

    ev, ctl = (rep["P2"][k]["median"] for k in cases)
    keep = ev >= ctl - 0.02
    rep["decision"] = ("KEEP idealised simulator (overlap within 0.02 of the validated control)" if keep else
                       "USE whitening-shaped simulator (overlap more than 0.02 below the validated control)")
    print(f"  DECISION: {rep['decision']}")
    OUT.write_text(json.dumps(rep, indent=2))
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
