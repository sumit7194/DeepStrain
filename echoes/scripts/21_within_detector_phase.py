"""L4 rung 2: the WITHIN-detector phase axis — and the full 2x2 with the network axis.

WHERE THIS SITS. 20_coherent_network.py tested the NETWORK axis and found coherent combination worth
**1.12x at 3.2 sigma** on physically-injected echoes — refuting my prediction that it would buy ~nothing. That
prediction rested on an argument I still believe (coherent amplitude summation and incoherent POWER summation
give the same network SNR), which means the gain came from somewhere the argument did not cover, and my
reasoning about *where the loss lives* is therefore suspect. The other half of that reasoning was:

    "the larger loss in our pipeline is the ENVELOPE (phase thrown away within each detector)"

This tests it. Being wrong twice in the same direction would say something useful about the intuition.

THE TWO STATISTICS, and why they genuinely differ. `comb_score` autocorrelates the **energy envelope**
|hilbert(x)|, so a candidate repeat only has to match in ENERGY. The phase-preserving version autocorrelates
the **complex analytic signal**, where a repeat must match in WAVEFORM:

    env : ACF of |hilbert(x)| (smoothed, mean-removed), sum teeth at dt, 2dt, 3dt      [current]
    mf  : matched filter against the TRUE injected template -- an ORACLE, not a search  [ceiling]

**A FIRST ATTEMPT AT THIS RUNG FAILED FOR A CONCEPTUAL REASON, AND THE FAILURE IS THE FINDING.** I first built
`coh` = |ACF of the complex analytic signal|, believing it preserved within-detector phase. It does not: for a
narrowband analytic signal |∫a(t)conj(a(t+tau))dt| is, to excellent approximation, the ENVELOPE
autocorrelation -- the carrier phase factors straight out of the magnitude. Measured directly on a loud
injection: correlation between the two statistics over 0-0.5 s of lag = **0.9969**, with teeth agreeing to
~1% (0.689 vs 0.696). Taking |.| per tooth discards exactly the phase it was meant to keep.

To actually keep within-detector phase you must combine the TEETH coherently, and that requires knowing the
phase relation BETWEEN successive echoes -- which depends on the reflection physics and is exactly the
model-dependence a shape-agnostic comb exists to avoid. **So the envelope is not a sloppy choice; it is what
makes the search model-independent.** It also explains the published method (arXiv:2512.24730): they combine
coherently ACROSS DETECTORS, where geometry hands you the phase relation for free, and not across echoes.

What can still be measured is the CEILING. An oracle matched filter against the exact injected waveform uses
all the phase information there is, so the envelope comb's loss is bounded by the gap to it. If that gap is
small, the model-independence is nearly free and this rung closes.

FULL 2x2 — the point of the rung. The two axes are independent and we now measure both together:

              incoherent network        coherent network
    env       current pipeline          20_'s 1.12x
    coh       this rung's question      both axes

which answers: which axis carries the gain, and do they COMPOSE or overlap?

CONVENTIONS INHERITED FROM 20_ (all three were bugs found there, don't regress them):
  * injections are PHYSICAL — measured H1/L1 delay and polarity, and a SHARED carrier phase across detectors
    (09's raw_train re-randomises phase per call, which made the detectors see different waveforms, r=0.366)
  * the H1/L1 geometry is MEASURED from the merger, not recited, and gated against the published ~6.9 ms
  * the verdict is SIGNIFICANCE-GATED at 2 sigma — an earlier run printed "helps" off a 1.4 sigma difference
  * the amplitude grid is concentrated on the 50% crossing, not spread to 3 sigma

Run:  .venv/bin/python scripts/21_within_detector_phase.py [--n-bg 60] [--n-trials 120]
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy.signal import hilbert

from echolib import DETECTORS, GW150914_DT_PRED, RESULTS, comb_on_env, envelope, fetch_block, progress

HERE = Path(__file__).resolve().parent
_s9 = importlib.util.spec_from_file_location("raw9", HERE / "09_raw_injection.py")
raw9 = importlib.util.module_from_spec(_s9); _s9.loader.exec_module(raw9)
_s20 = importlib.util.spec_from_file_location("net20", HERE / "20_coherent_network.py")
net20 = importlib.util.module_from_spec(_s20); _s20.loader.exec_module(net20)

FS = 4096.0
# Extended DOWNWARD: the oracle matched filter saturates below 0.5 sigma (it is handed the true waveform),
# so the old grid could not resolve its 50% point and the ceiling would have been reported as a lower bound.
AMPS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.85, 1.0, 1.2, 1.5)
N_TEETH = 3
OUT = RESULTS / "21_within_detector_phase.json"
STATS = ("env_incoh", "env_net", "mf_incoh", "mf_net")


def matched_filter_peak(x, template):
    """ORACLE: peak |correlation| of the whitened segment against the exact injected waveform.

    This is not a search -- it is handed the true template, so it upper-bounds what ANY phase-aware statistic
    could extract. Normalised so the value is the standard matched-filter SNR in whitened noise."""
    x = np.asarray(x, dtype=float)
    t = np.asarray(template, dtype=float)
    n = max(len(x), len(t))
    nt = float(np.sqrt(np.sum(t ** 2)))
    if nt <= 0:
        return 0.0
    X = np.fft.rfft(x, 2 * n); T = np.fft.rfft(t, 2 * n)
    corr = np.fft.irfft(X * np.conj(T), 2 * n)
    return float(np.max(np.abs(corr)) / nt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-bg", type=int, default=60)
    ap.add_argument("--n-trials", type=int, default=120)
    args = ap.parse_args()
    rng = np.random.default_rng(21)

    raws = {det: fetch_block(det, "GW150914") for det in DETECTORS}
    t0 = float(raws["H1"].t0.value)
    centers = t0 + 308 + 4 * np.arange(42)
    # Evaluate at the EXACT pre-registered spacing, not the nearest point of a 5 ms grid. The envelope ACF
    # peak is ~20 ms wide (pulse duration + 10 ms smoothing) so a 2.5 ms offset costs it nothing; the COHERENT
    # ACF peak is only ~1/bandwidth ~ 3 ms wide, so the same offset lands it off-peak and it detects nothing.
    # The smoke test showed exactly that -- coh at 0/5 where env was 5/5. Both statistics are scored at the
    # same single pre-registered lag, which is the protocol anyway (only index j was ever used).
    dt_grid = np.array([GW150914_DT_PRED])
    j = 0
    slope = float(np.mean(list(
        json.loads((RESULTS / "09_raw_injection.json").read_text())["slopes"].values())))

    # geometry: measured, not recited (same golden test as 20_)
    merger = {det: raw9.whitened_segment(raws[det], t0 + 316.0) for det in DETECTORS}
    n_use = int(0.2 * FS)
    d_samp, sign, cc = net20.measure_delay_sign(merger["H1"][:n_use], merger["L1"][:n_use], FS)
    d_ms = 1e3 * d_samp / FS
    ok = abs(abs(d_ms) - 6.9) < 3.0
    print(f"GOLDEN TEST -- geometry from the merger: delay {d_ms:+.2f} ms, sign {sign:+.0f} -> "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok:
        print("geometry failed; refusing to build coherent statistics on it"); return
    tau_s = -d_samp / FS

    def scores(c, amp=None):
        """All four statistics on one trial: {envelope comb, oracle matched filter} x {incoherent, coherent network}."""
        pseed = int(rng.integers(1 << 30))
        segs, tmpl = {}, None
        for det in DETECTORS:
            if amp is None:
                inj = None
            elif det == "L1":
                inj = (lambda t, c=c, A=amp, s_=pseed:
                       (setattr(raw9, "rng", np.random.default_rng(s_)),
                        sign * raw9.raw_train(t, c + 0.05 + tau_s, A))[1])
            else:
                inj = (lambda t, c=c, A=amp, s_=pseed:
                       (setattr(raw9, "rng", np.random.default_rng(s_)),
                        raw9.raw_train(t, c + 0.05, A))[1])
            segs[det] = raw9.whitened_segment(raws[det], c, inj)
        # The oracle template: the SAME waveform through the SAME whitening path, isolated by differencing
        # against the un-injected segment so only the filter's response to the signal remains. Built on EVERY
        # trial including background -- a matched filter scored against no template would return 0 on noise,
        # collapsing its threshold and making it look perfect. The null is noise correlated against a template.
        A_t = amp if amp is not None else (1.0 / slope)
        raw9.rng = np.random.default_rng(pseed)
        base = raw9.whitened_segment(raws["H1"], c)
        raw9.rng = np.random.default_rng(pseed)
        tmpl = raw9.whitened_segment(raws["H1"], c,
                                     lambda t, c=c, A=A_t: raw9.raw_train(t, c + 0.05, A)) - base
        net = net20.coherent_network(segs, d_samp, sign)
        e_i = np.zeros(len(dt_grid))
        for x in segs.values():
            e_i += comb_on_env(envelope(x, FS), FS, dt_grid, N_TEETH)
        mf_i = sum(matched_filter_peak(x, tmpl) for x in segs.values())
        return {"env_incoh": float(e_i[j]),
                "env_net": float(comb_on_env(envelope(net, FS), FS, dt_grid, N_TEETH)[j]),
                "mf_incoh": float(mf_i),
                "mf_net": float(matched_filter_peak(net, tmpl))}

    bg = {k: [] for k in STATS}
    for i in range(args.n_bg):
        s = scores(float(centers[i % len(centers)]))
        for k in STATS:
            bg[k].append(s[k])
        progress("21_bg", i, args.n_bg)
    thr = {k: float(np.quantile(bg[k], 0.95)) for k in STATS}
    out = {"delay_ms": d_ms, "sign": sign, "n_bg": args.n_bg, "n_trials": args.n_trials,
           "bg": {k: {"mean": float(np.mean(bg[k])), "sd": float(np.std(bg[k])), "thr95": thr[k]} for k in STATS}}
    print("\nbackgrounds (95th pct, sd):")
    for k in STATS:
        print(f"  {k:>10}: thr {thr[k]:8.4f}  sd {np.std(bg[k]):7.4f}")

    eff = {k: {} for k in STATS}
    print(f"\n{'amp':>6} " + "".join(f"{k:>11}" for k in STATS))
    for amp in AMPS:
        A = amp / slope
        hits = {k: 0 for k in STATS}
        for i in range(args.n_trials):
            c = float(centers[rng.integers(0, len(centers))])
            s = scores(c, amp=A)
            for k in STATS:
                hits[k] += s[k] > thr[k]
            progress(f"21_{amp}", i, args.n_trials)
        for k in STATS:
            eff[k][str(amp)] = hits[k] / args.n_trials
        print(f"{amp:>6.2f} " + "".join(f"{eff[k][str(amp)]:>11.2f}" for k in STATS), flush=True)
    out["efficiency"] = eff

    def amp50_err(e):
        amps = np.array([float(a) for a in AMPS]); ys = np.array([e[str(a)] for a in AMPS])
        if ys.max() < 0.5:
            return float("nan"), float("nan")
        i = int(np.argmax(ys >= 0.5))
        if i == 0:
            return float(amps[0]), float("nan")
        x0, x1, y0, y1 = amps[i-1], amps[i], ys[i-1], ys[i]
        a50 = float(x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0))
        slope_ = (y1 - y0) / (x1 - x0)
        return a50, float(np.sqrt(0.25 / args.n_trials) / max(slope_, 1e-6))

    print(f"\n{'statistic':>10} {'amp50':>9} {'err':>7} {'gain vs env_incoh':>19} {'sigma':>7}")
    base, base_e = amp50_err(eff["env_incoh"])
    out["amp50"] = {}; out["vs_baseline"] = {}
    for k in STATS:
        a50, e = amp50_err(eff[k])
        out["amp50"][k] = {"amp50": a50, "err": e}
        if k == "env_incoh":
            print(f"{k:>10} {a50:>9.3f} {e:>7.3f} {'(baseline)':>19} {'--':>7}"); continue
        gain = base / a50 if a50 else float("nan")
        sd = float(np.hypot(base_e, e))
        nsig = abs(base - a50) / sd if sd > 0 else float("nan")
        out["vs_baseline"][k] = {"gain": gain, "n_sigma": nsig}
        print(f"{k:>10} {a50:>9.3f} {e:>7.3f} {gain:>19.2f} {nsig:>7.1f}")

    w = out["vs_baseline"]
    out["network_helps"] = bool(w["env_net"]["gain"] > 1.10 and w["env_net"]["n_sigma"] > 2.0)
    # The ORACLE ceiling: how much does the envelope comb give up by being model-independent?
    out["oracle_ceiling"] = w["mf_incoh"]["gain"]
    out["oracle_ceiling_sigma"] = w["mf_incoh"]["n_sigma"]
    out["envelope_cost"] = w["mf_incoh"]["gain"]          # the comb's loss is bounded by the gap to the oracle
    out["model_independence_is_cheap"] = bool(w["mf_incoh"]["gain"] < 1.5)
    print(f"\nnetwork axis helps        : {out['network_helps']} "
          f"({w['env_net']['gain']:.2f}x at {w['env_net']['n_sigma']:.1f} sigma)")
    print(f"ORACLE ceiling (matched filter, handed the true waveform): "
          f"{w['mf_incoh']['gain']:.2f}x at {w['mf_incoh']['n_sigma']:.1f} sigma")
    print(f"  => the envelope comb gives up AT MOST {w['mf_incoh']['gain']:.2f}x by using no waveform model.")
    print(f"  => {'that is cheap -- model-independence is nearly free' if out['model_independence_is_cheap'] else 'that is a real cost -- a phase model would buy a lot, if you trusted one'}")
    print(f"network axis ON TOP of the oracle: {w['mf_net']['gain']:.2f}x "
          f"(vs oracle alone {w['mf_incoh']['gain']:.2f}x)")
    OUT.write_text(json.dumps(out, indent=2))
    print("wrote 21_within_detector_phase.json")


if __name__ == "__main__":
    main()
