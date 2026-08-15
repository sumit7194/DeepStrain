"""L4: does a COHERENT network combination beat our incoherent one for echoes?

arXiv:2512.24730 searches LVK data for echoes with a "generalized phase-marginalized likelihood that
coherently combines data for each QNM across a detector network", and reports null results on GW150914 /
GW231226 / GW250114 -- consistent with our own E3 nulls. Their method is a rung above ours, and RELATED_WORK
lists closing that gap as L4.

OUR STATISTIC IS DOUBLY INCOHERENT, which is the premise worth stating precisely:
  * WITHIN a detector, `comb_score` runs on `envelope(x)` -- the Hilbert envelope, so phase is discarded.
  * ACROSS detectors, `detection_statistic` SUMS the per-detector comb scores.
So there are two separate axes on which coherence could buy something, and they are worth separating rather
than bundled into one "go coherent" change.

PRE-REGISTERED PREDICTION (stated before running -- this is the falsifiable part).
For a signal present in both detectors, coherent amplitude summation and incoherent POWER summation give the
SAME network SNR: two equal detectors give sqrt(2) either way (coherent: 2A / (sqrt(2) sigma); incoherent:
sqrt(SNR_H^2 + SNR_L^2)). So the NETWORK axis alone should buy little on signal strength. Any gain must come
from the BACKGROUND side -- a coherent combination demands phase consistency that noise does not satisfy, so
its background may be tighter, and our threshold is set from the measured background. **We therefore expect
at most a modest gain from the network axis, and predict the larger loss in our pipeline is the ENVELOPE
(phase thrown away within each detector).** If net-coherent >> incoherent, that prediction is wrong and the
network axis matters more than the SNR algebra suggests.

THE DELAY AND SIGN ARE MEASURED, NOT RECITED. A coherent H1+L1 sum needs the inter-detector arrival delay and
relative polarity. Rather than hard-code numbers from memory (the standing rule in this repo), they are
measured by cross-correlating the WHITENED MERGER itself -- loud, unambiguous, and from the same sky position
the echoes would come from, so the same delay applies. GW150914's published value (~6.9 ms, L1 first, with the
detectors' responses roughly anti-aligned) then serves as a GOLDEN TEST on our own data handling: if we cannot
recover it, nothing downstream is trustworthy.

Run:  .venv/bin/python scripts/20_coherent_network.py [--n-bg 30] [--n-trials 25]
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from echolib import DETECTORS, GW150914_DT_PRED, RESULTS, comb_on_env, envelope, fetch_block, progress

HERE = Path(__file__).resolve().parent
_s9 = importlib.util.spec_from_file_location("raw9", HERE / "09_raw_injection.py")
raw9 = importlib.util.module_from_spec(_s9); _s9.loader.exec_module(raw9)

FS = 4096.0
# The 50% crossing lives near ~0.9 sigma, so the grid is concentrated there. An earlier coarse grid
# (0.5, 1.0, 1.5, 2.0, 3.0) put efficiency at 0.15 -> 0.53 -> 1.00, leaving the 50% point decided by a single
# interpolation between two widely spaced points -- far too blunt to resolve a ~20% difference.
AMPS = (0.5, 0.7, 0.85, 1.0, 1.2, 1.5)
LIT_DELAY_MS = 6.9          # GW150914 H1-L1 arrival delay, for the golden test only
OUT = RESULTS / "20_coherent_network.json"
SHARED_PHASE = True   # see scores(): 09's raw_train re-randomizes phase per call


def measure_delay_sign(h, l, fs, max_ms=12.0):
    """Cross-correlate two whitened merger segments; return (delay_samples, sign, peak_corr).

    The physical inter-detector delay cannot exceed the light travel time (~10 ms for H1-L1), so the search
    is capped -- an unconstrained argmax would happily lock onto a noise peak."""
    n = int(max_ms * 1e-3 * fs)
    h = (h - h.mean()) / (h.std() + 1e-12)
    l = (l - l.mean()) / (l.std() + 1e-12)
    lags = np.arange(-n, n + 1)
    cc = np.array([float(np.dot(h, np.roll(l, k)) / len(h)) for k in lags])
    i = int(np.argmax(np.abs(cc)))
    return int(lags[i]), float(np.sign(cc[i])), float(cc[i])


def coherent_network(segs, delay, sign):
    """Phase-preserving network strain: align L1 onto H1 and add with the measured relative polarity.

    Both inputs are whitened (unit-variance), so equal weights are the noise-optimal combination; the 1/sqrt(2)
    keeps the noise variance at 1 so scores stay comparable to a single detector's."""
    h, l = segs["H1"], segs["L1"]
    return (h + sign * np.roll(l, delay)) / np.sqrt(2.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-bg", type=int, default=30)
    ap.add_argument("--n-trials", type=int, default=25)
    args = ap.parse_args()
    rng = np.random.default_rng(20)

    raws = {det: fetch_block(det, "GW150914") for det in DETECTORS}
    t0 = float(raws["H1"].t0.value)
    centers = t0 + 308 + 4 * np.arange(42)
    dt_grid = np.arange(0.05, 0.5, 0.005)
    j = int(np.argmin(np.abs(dt_grid - GW150914_DT_PRED)))
    slope = float(np.mean(list(
        json.loads((RESULTS / "09_raw_injection.json").read_text())["slopes"].values())))

    # ---- GOLDEN TEST: recover the known H1-L1 geometry from the merger itself -------------------------
    merger = {det: raw9.whitened_segment(raws[det], t0 + 316.0) for det in DETECTORS}
    n_use = int(0.2 * FS)
    d_samp, sign, cc = measure_delay_sign(merger["H1"][:n_use], merger["L1"][:n_use], FS)
    d_ms = 1e3 * d_samp / FS
    ok = abs(abs(d_ms) - LIT_DELAY_MS) < 3.0
    print(f"GOLDEN TEST -- H1/L1 geometry measured from the merger (not recited):")
    print(f"  delay {d_ms:+.2f} ms | relative sign {sign:+.0f} | peak |corr| {abs(cc):.3f}")
    print(f"  literature GW150914 delay ~{LIT_DELAY_MS} ms -> {'PASS' if ok else 'FAIL'} (|diff| < 3 ms)\n")

    out = {"delay_ms": d_ms, "delay_samples": d_samp, "sign": sign, "peak_corr": cc,
           "golden_delay_ok": bool(ok), "n_bg": args.n_bg, "n_trials": args.n_trials}
    if not ok:
        print("Geometry golden test FAILED -- refusing to build a coherent statistic on it.")
        OUT.write_text(json.dumps(out, indent=2)); return

    tau_s = -d_samp / FS          # L1 arrival offset implied by the measured geometry

    def scores(c, amp=None, physical=True):
        """(incoherent, network-coherent) comb score at the pre-registered spacing.

        physical=True injects the echo train into L1 with the MEASURED delay and polarity, i.e. as a real
        source at GW150914's sky position would produce it. physical=False reproduces the convention every
        existing echo script uses -- the identical waveform added to both detectors with no delay and no sign
        flip -- which is harmless for an incoherent statistic but cancels under a coherent one."""
        # raw_train draws a FRESH random carrier phase on every call (module-level rng in 09), so calling it
        # once per detector produces two DIFFERENT waveforms -- correlated only ~0.37. That alone makes any
        # coherent combination impossible, independently of delay and polarity. Re-seed identically for the
        # pair so both detectors receive the SAME signal, which is what a real source produces.
        pseed = int(rng.integers(1 << 30))
        segs = {}
        for det in DETECTORS:
            if amp is None:
                inj = None
            elif det == "L1" and physical:
                inj = (lambda t, c=c, A=amp, s_=pseed:
                       (setattr(raw9, "rng", np.random.default_rng(s_)),
                        sign * raw9.raw_train(t, c + 0.05 + tau_s, A))[1])
            else:
                inj = (lambda t, c=c, A=amp, s_=pseed:
                       (setattr(raw9, "rng", np.random.default_rng(s_)),
                        raw9.raw_train(t, c + 0.05, A))[1])
            segs[det] = raw9.whitened_segment(raws[det], c, inj)
        incoh = np.zeros(len(dt_grid))
        for x in segs.values():
            incoh += comb_on_env(envelope(x, FS), FS, dt_grid)
        coh = comb_on_env(envelope(coherent_network(segs, d_samp, sign), FS), FS, dt_grid)
        return float(incoh[j]), float(coh[j])

    bg_i, bg_c = [], []
    for i in range(args.n_bg):
        a, b = scores(float(centers[i % len(centers)]))
        bg_i.append(a); bg_c.append(b)
        progress("20_coh_bg", i, args.n_bg)
    th_i, th_c = float(np.quantile(bg_i, 0.95)), float(np.quantile(bg_c, 0.95))
    out["bg"] = {"incoherent": {"mean": float(np.mean(bg_i)), "sd": float(np.std(bg_i)), "thr95": th_i},
                 "coherent": {"mean": float(np.mean(bg_c)), "sd": float(np.std(bg_c)), "thr95": th_c}}
    print(f"backgrounds (n={args.n_bg}): incoherent 95th {th_i:.3f} (sd {np.std(bg_i):.3f}) | "
          f"coherent 95th {th_c:.3f} (sd {np.std(bg_c):.3f})")

    out["efficiency"] = {}
    for conv in ("physical", "identical"):
        eff = {"incoherent": {}, "coherent": {}}
        print(f"\n[{conv} injections] {'amp(sigma)':>11} {'incoherent':>12} {'coherent':>10}")
        for amp in AMPS:
            A = amp / slope
            hi = hc = 0
            for i in range(args.n_trials):
                c = float(centers[rng.integers(0, len(centers))])
                a, b = scores(c, amp=A, physical=(conv == "physical"))
                hi += a > th_i; hc += b > th_c
                progress(f"20_coh_{conv}_{amp}", i, args.n_trials)
            eff["incoherent"][str(amp)] = hi / args.n_trials
            eff["coherent"][str(amp)] = hc / args.n_trials
            print(f"{'':>13} {amp:>11.1f} {hi/args.n_trials:>12.2f} {hc/args.n_trials:>10.2f}", flush=True)
        out["efficiency"][conv] = eff

    def amp50(e):
        xs = [float(a) for a in AMPS]; ys = [e[str(a)] for a in AMPS]
        return float(np.interp(0.5, ys, xs)) if max(ys) >= 0.5 else float("nan")

    out["amp50"], out["gain"] = {}, {}
    for conv, eff in out["efficiency"].items():
        a_i, a_c = amp50(eff["incoherent"]), amp50(eff["coherent"])
        g = float(a_i / a_c) if a_c and np.isfinite(a_c) and a_c > 0 else float("nan")
        out["amp50"][conv] = {"incoherent": a_i, "coherent": a_c}
        out["gain"][conv] = g
        print(f"\n[{conv}] 50% detection amplitude: incoherent {a_i:.3f} sigma | coherent {a_c:.3f} sigma "
              f"-> gain {g:.2f}x")
    # Significance, not just a ratio. Binomial error on each efficiency point propagates to the interpolated
    # 50% amplitude through the local slope; a 1.4-sigma difference is not a result, however good the ratio
    # looks. An earlier run printed "COHERENT NETWORK HELPS" at exactly that significance -- hence this guard.
    for conv, eff in out["efficiency"].items():
        amps = np.array([float(a) for a in AMPS])
        errs = {}
        for stat in ("incoherent", "coherent"):
            ys = np.array([eff[stat][str(a)] for a in amps])
            i = int(np.argmax(ys >= 0.5))
            if i == 0 or ys.max() < 0.5:
                errs[stat] = float("nan"); continue
            slope = (ys[i] - ys[i - 1]) / (amps[i] - amps[i - 1])
            errs[stat] = float(np.sqrt(0.25 / args.n_trials) / max(slope, 1e-6))
        a_, b_ = out["amp50"][conv]["incoherent"], out["amp50"][conv]["coherent"]
        sd = float(np.hypot(errs["incoherent"], errs["coherent"]))
        out.setdefault("significance", {})[conv] = {
            "amp50_err": errs, "diff": float(a_ - b_), "diff_err": sd,
            "n_sigma": float(abs(a_ - b_) / sd) if sd > 0 else float("nan")}
        print(f"[{conv}] amp50 {a_:.3f}+-{errs['incoherent']:.3f} vs {b_:.3f}+-{errs['coherent']:.3f} "
              f"-> diff {a_-b_:+.3f}+-{sd:.3f} = {abs(a_-b_)/sd:.1f} sigma")

    gp = out["gain"]["physical"]
    sig_p = out["significance"]["physical"]["n_sigma"]
    out["coherent_helps"] = bool(np.isfinite(gp) and gp > 1.10 and sig_p > 2.0)
    out["underpowered"] = bool(sig_p <= 2.0)
    out["injection_convention_matters"] = bool(
        np.isfinite(gp) and np.isfinite(out["gain"]["identical"])
        and abs(gp - out["gain"]["identical"]) > 0.15)
    print(f"\n=> physical-injection gain {gp:.2f}x at {sig_p:.1f} sigma: "
          f"{'COHERENT NETWORK HELPS (significant)' if out['coherent_helps'] else ('UNDERPOWERED -- suggestive but not significant, do not claim' if out['underpowered'] else 'no material gain')}")
    print(f"=> injection convention {'MATTERS' if out['injection_convention_matters'] else 'does not matter'} "
          f"({gp:.2f}x physical vs {out['gain']['identical']:.2f}x identical) -- every existing echo script "
          f"uses the IDENTICAL convention, which silently cancels under a coherent statistic")
    OUT.write_text(json.dumps(out, indent=2))
    print("wrote 20_coherent_network.json")


if __name__ == "__main__":
    main()
