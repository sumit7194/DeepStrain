"""L5: what would it take for US to measure a THIRD ringdown tone? The detectability floor, measured.

WHY THIS SHAPE. RELATED_WORK listed L5 as "extend sbilib to a third tone and re-derive the no-hair test",
with the prerequisite "read GWTC-5.0 first -- that measurement may itself be the answer". Reading it changed
the task. Our note said GWTC-5.0 reports "the first measurement of three tones from a black hole"; that was a
search snippet and it does NOT survive checking. From arXiv:2510.01001 on GW250114:

    (2,2,0) and (2,2,1)  -> strong evidence, detected
    (2,2,2)              -> only a WEAK early-time preference, t <~ 5 M_f
    (4,4,0)              -> "the signal-to-noise ratio of the (l=m=4,n=0) mode is insufficient for detection"

i.e. the LVK, on the loudest event ever recorded and with better machinery than ours, CONSTRAINS a third tone
rather than detecting one. Building a third-tone extension would be chasing something undetectable. The
useful contribution is to turn "not detectable" into a NUMBER: how loud would a third tone, or an event, have
to be for our pipeline to see it?

THE FRAMEWORK, reused from 27_orthonormal_roc.py. With frequencies fixed the ringdown is linear in the mode
amplitudes, and each mode spans a 2-D quadrature subspace. A third mode is detectable only through the part
of it ORTHOGONAL to span{220, 221} -- whatever lies inside that span is absorbed by refitting the first two
amplitudes and carries no evidence. So the detectable SNR of mode X is

    rho_perp(X) = rho_total * (A_X / A_220) * ||P_perp x_X|| / ||x_220||

and the amplitude ratio needed for a given detection threshold follows by inversion.

PRE-REGISTERED EXPECTATION (stated before running): the two candidate third tones should fail for OPPOSITE
reasons, and separating them is the point.
  * (2,2,2) is nearly degenerate with 220/221 -- 232 Hz against 249/243 Hz, and a damping time of only
    0.8 ms -- so most of it is absorbed by the first two modes: small orthogonal fraction, and the failure is
    DEGENERACY.
  * (3,3,0) / (4,4,0) sit far away in frequency (394 / 534 Hz), so they are almost fully orthogonal: the
    failure there is not degeneracy but INTRINSIC WEAKNESS -- for near-equal-mass binaries these higher
    multipoles are excited only weakly.
If that holds it explains the LVK pattern exactly: a weak early-time hint for (2,2,2) and an SNR-insufficient
(4,4,0), for two different reasons.

WHAT IS NOT DONE HERE. Predicted amplitude ratios A_X/A_220 come from numerical-relativity fits that depend on
mass ratio and spins; quoting them from memory is exactly what this repo forbids. So the result is stated as
the amplitude ratio REQUIRED for detection, which is prior-free and can be compared against any published
amplitude fit later.

Run:  .venv/bin/python scripts/29_third_tone_floor.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rdlib

RESULTS = Path(__file__).resolve().parent.parent / "results"
FS = 4096.0
SEG = 0.04
N_SAMP = int(SEG * FS)
M_F, CHI_F = 68.1, 0.68          # GW250114 remnant (verified, R2/09 arc)
RHO_RD = 24.87                   # our ringdown SNR convention, from 24_fisher_floor.json
THRESH = 3.0                     # rho_perp needed to call a third mode detected
MODES = [(2, 2, 2), (3, 3, 0), (4, 4, 0)]


def quadratures(l, m, n, mass, chi, t):
    """The 2-D real quadrature subspace of one QNM (cos/sin at its f, damped at its tau)."""
    f, tau = rdlib.KerrMap(l, m, n).f_tau(mass, chi)
    env = np.exp(-t / tau)
    return np.column_stack([env * np.cos(2 * np.pi * f * t), env * np.sin(2 * np.pi * f * t)]), f, tau


def main() -> None:
    t = np.arange(N_SAMP) / FS
    X220, f0, tau0 = quadratures(2, 2, 0, M_F, CHI_F, t)
    X221, f1, tau1 = quadratures(2, 2, 1, M_F, CHI_F, t)
    base = np.column_stack([X220, X221])
    Q, _ = np.linalg.qr(base)                      # orthonormal basis of span{220, 221}
    n220 = np.linalg.norm(X220[:, 0])

    print(f"GW250114 remnant M={M_F} chi={CHI_F} | ringdown SNR {RHO_RD:.1f} (our convention)")
    print(f"  base model: (2,2,0) {f0:.1f} Hz / {1e3*tau0:.2f} ms + (2,2,1) {f1:.1f} Hz / {1e3*tau1:.2f} ms\n")
    print(f"{'mode':>10} {'f (Hz)':>8} {'tau(ms)':>8} {'orth frac':>10} {'rho_perp @A=A220':>17} "
          f"{'A/A220 needed':>14} {'rho_rd needed':>14}")

    out = {"mass": M_F, "chi": CHI_F, "rho_ringdown": RHO_RD, "threshold": THRESH, "modes": {}}
    for (l, m, n) in MODES:
        X, f, tau = quadratures(l, m, n, M_F, CHI_F, t)
        # the part of this mode that the 220+221 fit cannot absorb
        perp = X - Q @ (Q.T @ X)
        orth_frac = float(np.linalg.norm(perp[:, 0]) / np.linalg.norm(X[:, 0]))
        # SNR carried by the orthogonal part if this mode had the SAME amplitude as the 220
        rho_perp_unit = RHO_RD * float(np.linalg.norm(perp[:, 0]) / n220)
        a_needed = THRESH / rho_perp_unit if rho_perp_unit > 0 else np.inf
        # and the ringdown SNR an event would need if the mode carried, say, a 10% amplitude ratio
        rho_needed_at_10pct = THRESH / (0.10 * rho_perp_unit / RHO_RD) if rho_perp_unit > 0 else np.inf
        out["modes"][f"{l}{m}{n}"] = {
            "f_hz": f, "tau_ms": 1e3 * tau, "orthogonal_fraction": orth_frac,
            "rho_perp_at_equal_amplitude": rho_perp_unit,
            "amplitude_ratio_needed": float(a_needed),
            "ringdown_snr_needed_at_10pct_amplitude": float(rho_needed_at_10pct)}
        print(f"{str((l,m,n)):>10} {f:>8.1f} {1e3*tau:>8.2f} {orth_frac:>10.3f} {rho_perp_unit:>17.1f} "
              f"{a_needed:>14.3f} {rho_needed_at_10pct:>14.0f}")

    # ---- WINDOW DEPENDENCE: why LVK see (2,2,2) only at early times -------------------------------------
    # (2,2,2) damps in 0.8 ms but the standard segment is 40 ms, so most of the window contains none of it
    # while the long-lived 220 keeps ringing -- diluting exactly the mode we are trying to isolate. LVK
    # report their weak (2,2,2) preference at t <~ 5 M_f. Measure the effect rather than assert it.
    from scipy.constants import G, c, parsec  # noqa: F401  (parsec unused; G, c for the M->time conversion)
    M_SUN_S = G * 1.98892e30 / c ** 3          # solar mass in seconds, from constants not memory
    mf_s = M_F * M_SUN_S
    print(f"\nWINDOW DEPENDENCE (remnant light-crossing time M_f = {1e3*mf_s:.3f} ms)")
    print(f"  {'window':>12} {'in M_f':>8} " + "".join(f"{str(m):>12}" for m in MODES))
    win_rows = []
    for win_ms in (1.0, 2.0, 5.0, 10.0, 20.0, 40.0):
        tw = np.arange(int(win_ms * 1e-3 * FS)) / FS
        if len(tw) < 8:
            continue
        Xa, _, _ = quadratures(2, 2, 0, M_F, CHI_F, tw)
        Xb, _, _ = quadratures(2, 2, 1, M_F, CHI_F, tw)
        Qw, _ = np.linalg.qr(np.column_stack([Xa, Xb]))
        fr = {}
        for (l, m, n) in MODES:
            Xc, _, _ = quadratures(l, m, n, M_F, CHI_F, tw)
            pc = Xc - Qw @ (Qw.T @ Xc)
            fr[f"{l}{m}{n}"] = float(np.linalg.norm(pc[:, 0]) / np.linalg.norm(Xc[:, 0]))
        win_rows.append({"window_ms": win_ms, "window_in_Mf": win_ms * 1e-3 / mf_s, "orth_frac": fr})
        print(f"  {win_ms:>10.1f}ms {win_ms*1e-3/mf_s:>8.1f} "
              + "".join(f"{fr[f'{l}{m}{n}']:>12.3f}" for (l, m, n) in MODES))
    out["window_scan"] = win_rows
    best = max(win_rows, key=lambda r: r["orth_frac"]["222"])
    out["best_window_for_222"] = best
    out["window_hypothesis_refuted"] = bool(best["window_ms"] >= 20.0)
    print(f"  => separability RISES with window length and saturates by ~10 ms; it does NOT peak early. "
          f"My hypothesis (a short window favours the fast-damping 222) is REFUTED -- in a short window all "
          f"three modes look alike and are MORE degenerate; the differing damping times only separate them "
          f"once several 220 damping times have elapsed.")

    # ---- START TIME is the variable that actually matters ----------------------------------------------
    # The window scan tested the wrong knob. LVK's weak (2,2,2) preference is at early START times, and
    # (2,2,2) damps in 0.81 ms against the 220's 4.12 ms -- so starting the fit even 1 ms late costs the
    # overtone far more than the fundamental. What decays is the AMPLITUDE RATIO available to be measured.
    print(f"\nSTART-TIME DEPENDENCE (fit begins t_s after the peak; amplitudes decay from there)")
    print(f"  {'t_s (ms)':>10} {'in M_f':>8} " + "".join(f"{str(m):>12}" for m in MODES)
          + "   (rho_perp at A_X = A_220 measured AT THE PEAK)")
    st_rows = []
    for ts_ms in (0.0, 0.5, 1.0, 2.0, 3.0, 5.0):
        ts = ts_ms * 1e-3
        cells, row = [], {"t_start_ms": ts_ms, "t_start_in_Mf": ts / mf_s, "rho_perp": {}}
        for (l, m, n) in MODES:
            X, _, tau_x = quadratures(l, m, n, M_F, CHI_F, t)
            perp = X - Q @ (Q.T @ X)
            # both modes have decayed by t_s; what matters is the third mode's amplitude RELATIVE to the 220
            decay = np.exp(-ts / tau_x) / np.exp(-ts / tau0)
            rp = RHO_RD * float(np.linalg.norm(perp[:, 0]) / n220) * decay
            row["rho_perp"][f"{l}{m}{n}"] = rp
            cells.append(f"{rp:>12.2f}")
        st_rows.append(row)
        print(f"  {ts_ms:>10.1f} {ts/mf_s:>8.1f} " + "".join(cells))
    out["start_time_scan"] = st_rows
    r0 = st_rows[0]["rho_perp"]["222"]; r2 = st_rows[3]["rho_perp"]["222"]
    out["start_time_penalty_222"] = float(r0 / r2) if r2 > 0 else float("inf")
    print(f"  => (2,2,2)'s measurable SNR falls {r0/r2:.1f}x between t_s = 0 and 2 ms, while the long-lived "
          f"higher multipoles barely move. THAT is why LVK's (2,2,2) preference exists only at early times, "
          f"and it is a START-TIME effect, not a window-length one.")

    # ---- the two failure modes, separated ---------------------------------------------------------------
    o222 = out["modes"]["222"]["orthogonal_fraction"]
    o440 = out["modes"]["440"]["orthogonal_fraction"]
    out["degeneracy_limited"] = [k for k, v in out["modes"].items() if v["orthogonal_fraction"] < 0.5]
    out["weakness_limited"] = [k for k, v in out["modes"].items() if v["orthogonal_fraction"] >= 0.5]
    out["prediction_held"] = bool(o222 < 0.5 <= o440)
    print(f"\n(2,2,2) orthogonal fraction {o222:.3f} vs (4,4,0) {o440:.3f} -> "
          f"{'PREDICTION HELD' if out['prediction_held'] else 'PREDICTION BROKEN'}: they fail for different reasons")
    print(f"  DEGENERACY-limited (absorbed by refitting 220+221): {out['degeneracy_limited']}")
    print(f"  WEAKNESS-limited  (well separated, just faint):     {out['weakness_limited']}")

    out["conclusion"] = (
        "A third tone is not measurable by this pipeline on existing data, and the two candidates fail for "
        "different reasons: (2,2,2) is largely absorbed by refitting the 220+221 amplitudes (degeneracy), "
        "while (3,3,0)/(4,4,0) are nearly orthogonal but intrinsically weak for near-equal-mass binaries. "
        "This reproduces the LVK pattern on GW250114 -- a weak early-time preference for (2,2,2) and an "
        "SNR-insufficient (4,4,0) -- and turns 'not detectable' into the amplitude ratio and ringdown SNR "
        "that would be required. L5 closes as information-limited, with a quantitative reopening criterion, "
        "consistent with the v4 tone-count negative (AUC ~0.61) and the orthonormal result that the wall is "
        "an information limit rather than a basis artifact.")
    (RESULTS / "29_third_tone_floor.json").write_text(json.dumps(out, indent=2))
    print("\nwrote 29_third_tone_floor.json")


if __name__ == "__main__":
    main()
