"""M11: re-score LVK's own O4a subsolar trigger list with our CNN + adequate bank + H1xL1 coincidence.

WHY. LVK's O4a sub-solar-mass search (arXiv:2605.05444, Table 1, read in full 2026-09-23) reports no significant
candidates but publishes every trigger with FAR < 2/yr, with UTC time, detector-frame masses and network SNR. O4a
strain is public. An independent group (Christensen et al., arXiv:2607.23119) says three triggers "may contain a
possible sub-solar mass component". We have an independent detector (cnn_w64), a bank shown ADEQUATE and ~6% better
than the CNN on identical injections, and a time-slide background method. This asks what OUR instruments say.

PRE-REGISTRATION -- written and committed before any trigger is scored. Trigger scoring (`--stage score`) refuses to
run until `frozen.json` (thresholds + sensitivity, computed from background and injections only) exists, and that
file is committed before the scoring run.

  TRIGGER CLASSES (by rule, from Table 1's masses, not by look):
    PRIMARY       both components in our validated injection population [0.2, 1.0] Msun and both H1+L1:
                  T1 = 2023-07-19 11:50:50.27 UTC (PyCBC, 0.74+0.24, FAR 1.3/yr, network SNR 9.47)
                  T2 = 2023-08-10 10:10:03.37 UTC (PyCBC, 0.60+0.21, FAR 1.4/yr, network SNR 9.43)
    OUT-OF-DOMAIN chirp mass inside the bank [0.173, 0.871] but a component outside [0.2, 1.0]: CNN scored and its
                  FAR reported, NO verdict (Sep 14, Oct 14 HL; Aug 10 07:13 L-only -> single-detector L1).
    OUT-OF-RANGE  chirp mass ~2 Msun, outside everything we built (May 28, May 29 = GW230529): not scored.

  STATISTICS (PRIMARY triggers), each on the 64-s window ENDING at the reported UTC time (our training convention
  puts the merger at the window end):
    X1  CNN coincidence       s_H1 + s_L1 of cnn_w64 (the gated `sum`)
    X2  targeted-bank coinc.  n_H1 + n_L1, n = max newSNR (n=8 semi-coherent) over the trigger's template set,
                              scanned +-0.5 s. Template set = the equal-mass bank template that best recovers a
                              noiseless waveform with the trigger's own masses, +-20 grid neighbours (41 templates,
                              ~+-1% in chirp mass). X2 is USED ONLY IF that best template recovers >= 80% of the
                              in-window optimal SNR; otherwise X2 = "inapplicable (template mismatch)".

  BACKGROUND: 30 O4a H1nL1 4096-s segments drawn with a fixed seed from the coincident-data timeline, excluding
    +-1 h of every Table-1 trigger and of every GWOSC O4a event. All non-zero circular lags of the tiled 64-s
    windows (far_deep's construction): T_bg = (N-1) N x 64 s. FAR(X) = #{background pairs >= X} / T_bg. The
    identical window definition is applied to background, injections and triggers.

  SENSITIVITY (the precondition that makes any null interpretable): injections with the trigger's own masses,
    random extrinsics, network SNR in OUR band (f >= 50 Hz) on the grid 6/8/10/12/14/16, 120 per grid point per
    trigger, into the same background segments. eps_X(rho) = fraction above the FAR = 1/yr threshold of X.
    The trigger's in-band network SNR rho_T = (Table-1 network SNR) x c, with c^2 the ratio of the inspiral
    SNR integral over [50, 1024] Hz to that over PyCBC's [45, 1000] Hz (paper Sec. 2.3), on an O4a PSD.

  LOOK-ELSEWHERE: 2 primary triggers x 2 statistics = 4 looks. A look is SIGNIFICANT iff 4 x FAR <= 1/yr.

  VERDICT per (trigger, statistic), with eps evaluated at rho_T:
    significant & eps >= 0.5      -> INDEPENDENT SUPPORT
    significant & eps <  0.5      -> HIGH SCORE, BUT A REAL SIGNAL THIS WEAK RARELY SCORES SO -- noise excursion favoured
    not significant & eps >= 0.5  -> TENSION (we would probably have seen a real signal)
    not significant & eps <  0.1  -> UNINFORMATIVE (our instruments cannot test a trigger this weak)
    not significant, 0.1 <= eps < 0.5 -> WEAKLY INFORMATIVE (a null at modest sensitivity)

  PREDICTION, stated so it can be wrong: both primary triggers UNINFORMATIVE under both statistics. rho_T ~ 9, and
    our 50%-detection SNRs sit near 15-18 at zero false alarms, so eps(rho_T) < 0.1 is expected; scores consistent
    with noise (4 x FAR > 1/yr). The bank should be somewhat more sensitive than the CNN, but not enough.

Run (resumable; each background segment checkpointed, raw strain purged):
    .venv/bin/python scripts/o4a_ssm_rescore.py --stage discover
    .venv/bin/python scripts/o4a_ssm_rescore.py --stage select
    nohup .venv/bin/python scripts/o4a_ssm_rescore.py --stage background > log 2>&1 &
    .venv/bin/python scripts/o4a_ssm_rescore.py --stage freeze        # then COMMIT frozen.json
    .venv/bin/python scripts/o4a_ssm_rescore.py --stage score
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pbh import config as C
from pbh.data import fetch_segment, segment_path, whiten_segment
from pbh.models import make_model
from pbh.spectrogram import spectrogram
from pbh.sweep import SweepGrid, pool_and_log, score_windows
from pbh.waveforms import make_whitened_injection, sample_params

HERE = Path(__file__).resolve().parent
_so = importlib.util.spec_from_file_location("so", HERE / "semicoherent_oracle.py")
so = importlib.util.module_from_spec(_so); _so.loader.exec_module(so)
_bd = importlib.util.spec_from_file_location("bd", HERE / "bank_dense.py")
bd = importlib.util.module_from_spec(_bd); _bd.loader.exec_module(bd)

FS = C.SAMPLE_RATE
WIN, PAD = so.WIN, so.PAD
CROP = C.WHITEN_CROP_SEC * FS
NBINS = SweepGrid.short(64).n_time_bins
DETS = ("H1", "L1")
OUT = C.RESULTS_DIR / "o4a_ssm"
SEGS = OUT / "segs"
OUT.mkdir(exist_ok=True, parents=True); SEGS.mkdir(exist_ok=True)
POOL, TSETS, FROZEN, RESULT = OUT / "pool.json", OUT / "tsets.json", OUT / "frozen.json", OUT / "rescore.json"

O4A = (1368975618, 1389456018)          # 2023-05-24 15:00 -> 2024-01-16 16:00 UTC (arXiv:2605.05444 abstract)
N_BG, SEED_BG = 30, 20260923
SNR_GRID, N_PER = (6.0, 8.0, 10.0, 12.0, 14.0, 16.0), 120
HALF_SET, MATCH_MIN = 20, 0.80
LOOKS, YEAR = 4, 365.25 * 86400

# arXiv:2605.05444 Table 1 ("Triggers recovered with a FAR below 2 yr^-1"), transcribed from the full text;
# masses are detector-frame (1+z)m in Msun, FAR per year, SNR is the network SNR.
TABLE1 = [
    dict(far=0.00052, pipe="GstLAL", utc="2023-05-29 18:15:00.75", m1=6.49, m2=0.98, ifos="L", snr=10.63),
    dict(far=0.0010, pipe="PyCBC", utc="2023-05-29 18:15:00.75", m1=7.73, m2=0.87, ifos="L", snr=11.10),
    dict(far=0.87, pipe="MBTA", utc="2023-09-14 21:02:19.50", m1=2.63, m2=0.29, ifos="HL", snr=9.44),
    dict(far=0.91, pipe="MBTA", utc="2023-05-29 18:15:00.75", m1=7.64, m2=0.88, ifos="L", snr=9.96),
    dict(far=1.2, pipe="GstLAL", utc="2023-05-28 11:07:32.09", m1=6.49, m2=0.98, ifos="HL", snr=9.48),
    dict(far=1.3, pipe="PyCBC", utc="2023-07-19 11:50:50.27", m1=0.74, m2=0.24, ifos="HL", snr=9.47),
    dict(far=1.3, pipe="MBTA", utc="2023-10-14 08:15:06.33", m1=2.28, m2=0.22, ifos="HL", snr=10.28),
    dict(far=1.3, pipe="MBTA", utc="2023-08-10 07:13:47.88", m1=1.86, m2=0.32, ifos="L", snr=12.16),
    dict(far=1.4, pipe="PyCBC", utc="2023-08-10 10:10:03.37", m1=0.60, m2=0.21, ifos="HL", snr=9.43),
]


def mchirp(m1, m2):
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


def classify(t):
    mc = mchirp(t["m1"], t["m2"])
    if not bd.MC_LO <= mc <= bd.MC_HI:
        return "out-of-range"
    if C.M_MIN <= t["m2"] <= t["m1"] <= C.M_MAX and t["ifos"] == "HL":
        return "primary"
    return "out-of-domain"


def triggers():
    from gwpy.time import to_gps
    out = []
    for t in TABLE1:
        t = dict(t, gps=float(to_gps(t["utc"])), mc=mchirp(t["m1"], t["m2"]))
        t["cls"] = classify(t)
        t["name"] = t["utc"][:19].replace(" ", "T")
        out.append(t)
    return out


def primaries():
    return [t for t in triggers() if t["cls"] == "primary"]


# ---- stage: discover --------------------------------------------------------------------------------------------
def discover():
    from gwosc import datasets
    from gwosc.timeline import get_segments
    h = get_segments("H1_DATA", *O4A)
    l = get_segments("L1_DATA", *O4A)
    both, i, j = [], 0, 0
    while i < len(h) and j < len(l):                        # interval intersection
        a, b = max(h[i][0], l[j][0]), min(h[i][1], l[j][1])
        if b > a:
            both.append((a, b))
        if h[i][1] < l[j][1]:
            i += 1
        else:
            j += 1
    avoid = [t["gps"] for t in triggers()]
    for ev in datasets.find_datasets(type="events", segment=O4A):
        try:
            avoid.append(float(datasets.event_gps(ev)))
        except Exception:                                  # noqa: BLE001 -- a version without a GPS is skipped
            continue
    starts, margin = [], 256
    for a, b in both:
        t = a + margin
        while t + C.SEGMENT_LEN + margin <= b:
            if not any(t - 3600 <= g <= t + C.SEGMENT_LEN + 3600 for g in avoid):
                starts.append(int(t))
            t += C.SEGMENT_LEN
    rng = np.random.default_rng(SEED_BG)
    pick = sorted(int(x) for x in rng.choice(starts, size=N_BG, replace=False))
    POOL.write_text(json.dumps({"n_candidates": len(starts), "coincident_days": sum(b - a for a, b in both) / 86400,
                                "n_avoid": len(avoid), "segments": pick}, indent=2))
    print(f"H1nL1 coincident {sum(b - a for a, b in both)/86400:.1f} d, {len(starts)} clean 4096-s stretches "
          f"(+-1 h of {len(avoid)} triggers/events excluded) -> {N_BG} drawn with seed {SEED_BG}")


# ---- helpers ----------------------------------------------------------------------------------------------------
def fetch(d, g, tries=4):
    for k in range(tries):
        if segment_path(d, g).exists():
            return
        try:
            fetch_segment(d, g); return
        except Exception:                                  # noqa: BLE001 -- GWOSC transients; last one re-raised
            if k == tries - 1:
                raise
            time.sleep(20 * 2 ** k)


def purge(g):
    for d in DETS:
        segment_path(d, g).unlink(missing_ok=True)
    try:
        from astropy.utils.data import clear_download_cache
        clear_download_cache()                              # gwpy keeps a 2nd copy there (far_deep, 2026-08-08)
    except Exception:                                      # noqa: BLE001
        pass


def template_chunks(mc, d, t0, psd):
    base = sample_params(np.random.default_rng(0))          # bank_oracle's fiducial extrinsics, as bank_dense
    h, _ = make_whitened_injection(replace(base, mass1=float(mc * bd.EQ), mass2=float(mc * bd.EQ)), d, t0, psd)
    g = h[-WIN:].copy()
    if len(g) < WIN:
        g = np.pad(g, (WIN - len(g), 0))
    return so.analytic_chunks(g, bd.N_CHUNK)


def window_newsnr(D, N, chunks, starts):
    """Per tiled window: max newSNR within +-PAD of the window start, from one template. D = rfft(wc)."""
    n_valid = N - WIN + 1
    A = np.zeros(n_valid); Bs = np.zeros(n_valid)
    for off, a, p in chunks:
        cr = np.fft.irfft(D * np.conj(np.fft.rfft(np.real(a), n=N)), n=N)[off:off + n_valid]
        ci = np.fft.irfft(D * np.conj(np.fft.rfft(np.imag(a), n=N)), n=N)[off:off + n_valid]
        x = cr ** 2 + ci ** 2
        A += x; Bs += x ** 2 / p
    ns = so.newsnr(A, Bs, len(chunks))
    return np.array([ns[max(0, s - PAD):min(n_valid, s + PAD + 1)].max() for s in starts])


def cnn_score(model, dev, x):
    return float(score_windows(model, dev, pool_and_log(spectrogram(x), NBINS)[None])[0])


def load_model():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    m = make_model("cnn"); m.load_state_dict(torch.load(C.MODEL_DIR / "cnn_w64.pt", map_location=dev))
    m.to(dev).train(False)                                  # inference mode (dropout/batchnorm frozen)
    return m, dev


def inject_window(wc, m, sig):
    dwin = wc[m - WIN - PAD:m + PAD].copy()
    s = sig[-WIN:] if len(sig) >= WIN else np.pad(sig, (WIN - len(sig), 0))
    dwin[PAD:PAD + WIN] += s
    return dwin


# ---- stage: select ----------------------------------------------------------------------------------------------
def select():
    """Template set per primary trigger, from a noiseless waveform with its own masses (H1, first bg segment PSD)."""
    g = json.loads(POOL.read_text())["segments"][0]
    fetch("H1", g)
    w, t0, psd = whiten_segment("H1", g)
    mcs = bd.bank_mcs(0.0005)
    out = {}
    for t in primaries():
        base = sample_params(np.random.default_rng(0))
        h, _ = make_whitened_injection(replace(base, mass1=t["m1"], mass2=t["m2"]), "H1", t0, psd)
        sig = h[-WIN:] if len(h) >= WIN else np.pad(h, (WIN - len(h), 0))
        opt = float(np.sqrt(np.sum(sig ** 2)))
        dwin = np.zeros(WIN + 2 * PAD); dwin[PAD:PAD + WIN] = sig
        cand = np.where(np.abs(mcs / t["mc"] - 1) <= 0.10)[0]
        rho = np.array([np.sqrt(so.local_stats(dwin, template_chunks(mcs[k], "H1", t0, psd), PAD)[0]) for k in cand])
        best = int(cand[int(rho.argmax())])
        sel = list(range(max(0, best - HALF_SET), min(len(mcs), best + HALF_SET + 1)))
        frac = float(rho.max() / opt)
        out[t["name"]] = {"mc_trigger": t["mc"], "mc_best_template": float(mcs[best]), "match_fraction": frac,
                          "bank_applicable": bool(frac >= MATCH_MIN), "template_mcs": [float(mcs[k]) for k in sel]}
        print(f"{t['name']}: Mc {t['mc']:.4f} -> best equal-mass template {mcs[best]:.4f}, recovers "
              f"{100*frac:.1f}% of in-window optimal SNR  ({'bank USED' if frac >= MATCH_MIN else 'bank INAPPLICABLE'})")
    TSETS.write_text(json.dumps(out, indent=2))


# ---- stage: background (+ injections), one checkpoint per segment -----------------------------------------------
def background():
    pool = json.loads(POOL.read_text())["segments"]
    tsets = json.loads(TSETS.read_text())
    prim = primaries()
    model, dev = load_model()
    for si, g in enumerate(pool):
        f = SEGS / f"seg_{g}.npz"
        if f.exists():
            continue
        t_start = time.time()
        try:
            for d in DETS:
                fetch(d, g)
            W = {d: whiten_segment(d, g) for d in DETS}
            wc = {d: W[d][0][CROP:-CROP] for d in DETS}
            if not all(np.isfinite(wc[d]).all() for d in DETS):
                raise ValueError("non-finite whitened data")
            n = min(len(wc[d]) for d in DETS)
            starts = np.arange((n - WIN) // WIN) * WIN
            rec = {}
            for d in DETS:
                rec[f"cnn_{d}"] = np.array([cnn_score(model, dev, wc[d][s:s + WIN]) for s in starts], np.float32)
            chunks = {}
            for t in prim:
                ts = tsets[t["name"]]
                for d in DETS:
                    t0, psd = W[d][1], W[d][2]
                    chunks[(t["name"], d)] = [template_chunks(mc, d, t0, psd) for mc in ts["template_mcs"]]
                if not ts["bank_applicable"]:
                    continue
                for d in DETS:
                    N = len(wc[d]); D = np.fft.rfft(wc[d])
                    rec[f"bank_{t['name']}_{d}"] = np.max(
                        [window_newsnr(D, N, ch, starts) for ch in chunks[(t["name"], d)]], axis=0).astype(np.float32)
            # injections: this segment's share of each trigger's SNR grid, trigger's own masses, random extrinsics
            rng = np.random.default_rng([SEED_BG, int(g)])
            per = N_PER // len(pool)
            rows = []
            for t in prim:
                for snr in SNR_GRID:
                    for _ in range(per):
                        p = replace(sample_params(rng), mass1=t["m1"], mass2=t["m2"])
                        m = int(rng.integers(WIN + PAD, n - PAD))
                        hs = {d: make_whitened_injection(p, d, W[d][1], W[d][2]) for d in DETS}
                        scale = snr / np.sqrt(sum(hs[d][1] ** 2 for d in DETS))
                        row = [SNR_GRID.index(snr), prim.index(t)]
                        for d in DETS:
                            dwin = inject_window(wc[d], m, hs[d][0] * scale)
                            row.append(cnn_score(model, dev, dwin[PAD:PAD + WIN]))
                            if tsets[t["name"]]["bank_applicable"]:
                                row.append(max(so.local_stats(dwin, ch, PAD)[1] for ch in chunks[(t["name"], d)]))
                            else:
                                row.append(np.nan)
                        rows.append(row)
            rec["inj"] = np.array(rows, np.float64)          # [snr_idx, trig_idx, cnnH, bankH, cnnL, bankL]
            tmp = SEGS / f"seg_{g}.tmp.npz"
            np.savez(tmp, **rec); os.replace(tmp, f)
            print(f"[{si+1}/{len(pool)}] {g}: {len(starts)} windows, {len(rows)} injections "
                  f"({(time.time()-t_start)/60:.1f} min)", flush=True)
        except Exception as exc:                           # noqa: BLE001 -- unscored segment retried next pass
            print(f"[{si+1}/{len(pool)}] {g}: SKIP {type(exc).__name__}: {str(exc)[:80]}", flush=True)
        finally:
            purge(g)


# ---- stage: freeze (thresholds + sensitivity; no trigger data touched) -------------------------------------------
def load_bg():
    zs = [np.load(p) for p in sorted(SEGS.glob("seg_*.npz"))]
    if not zs:
        raise SystemExit("no background segments cached")
    cat = lambda k: np.concatenate([z[k] for z in zs])
    return zs, cat


def far_of(sH, sL, x):
    """All non-zero circular lags of the concatenated streams = all ordered pairs i != j."""
    N = len(sH)
    tot = np.add.outer(sH, sL)
    n_ge = int((tot >= x).sum() - (np.diag(tot) >= x).sum())
    return n_ge / ((N - 1) * N * 64.0) * YEAR           # per year


def thr_at(sH, sL, rate_per_yr):
    N = len(sH)
    tot = np.add.outer(sH, sL)
    off = tot[~np.eye(N, dtype=bool)]
    k = max(1, int(round(rate_per_yr * (N - 1) * N * 64.0 / YEAR)))
    return float(np.partition(off, len(off) - k)[len(off) - k])


def band_factor(psd):
    f = np.asarray(psd.sample_frequencies); S = np.asarray(psd)
    ok = S > 0
    integ = lambda lo, hi: float(np.sum((f ** (-7 / 3) / np.where(ok, S, np.inf))[(f >= lo) & (f <= hi) & ok]))
    return float(np.sqrt(integ(50, 1024) / integ(45, 1000)))


def freeze():
    zs, cat = load_bg()
    tsets = json.loads(TSETS.read_text())
    prim = primaries()
    g0 = json.loads(POOL.read_text())["segments"][0]
    fetch("H1", g0)
    c = band_factor(whiten_segment("H1", g0)[2]); purge(g0)
    inj = np.concatenate([z["inj"] for z in zs])
    fr = {"n_segments": len(zs), "n_windows": int(len(cat("cnn_H1"))), "band_factor_c": c, "looks": LOOKS,
          "T_bg_yr": (len(cat("cnn_H1")) - 1) * len(cat("cnn_H1")) * 64.0 / YEAR, "stats": {}}
    for ti, t in enumerate(prim):
        rho_T = t["snr"] * c
        for stat in ("cnn", "bank"):
            if stat == "bank" and not tsets[t["name"]]["bank_applicable"]:
                fr["stats"][f"{t['name']}|{stat}"] = {"applicable": False}
                continue
            key = (lambda d: f"cnn_{d}") if stat == "cnn" else (lambda d: f"bank_{t['name']}_{d}")
            sH, sL = cat(key("H1")), cat(key("L1"))
            thr = thr_at(sH, sL, 1.0)
            cols = (2, 4) if stat == "cnn" else (3, 5)
            eps = []
            for si in range(len(SNR_GRID)):
                r = inj[(inj[:, 0] == si) & (inj[:, 1] == ti)]
                eps.append(float(np.mean(r[:, cols[0]] + r[:, cols[1]] >= thr)))
            fr["stats"][f"{t['name']}|{stat}"] = {"applicable": True, "thr_far_1_per_yr": thr, "snr_grid": list(SNR_GRID),
                                                  "eps": eps, "rho_T": rho_T,
                                                  "eps_at_rho_T": float(np.interp(rho_T, SNR_GRID, eps)),
                                                  "n_inj_per_point": int(np.sum((inj[:, 0] == 0) & (inj[:, 1] == ti)))}
            print(f"{t['name']} {stat}: threshold(1/yr) {thr:.3f}; eps over SNR {list(SNR_GRID)} = "
                  f"{[round(e, 3) for e in eps]}; rho_T = {rho_T:.2f} -> eps {np.interp(rho_T, SNR_GRID, eps):.3f}")
    FROZEN.write_text(json.dumps(fr, indent=2))
    print(f"wrote {FROZEN.name} -- COMMIT IT before --stage score")


# ---- stage: score (trigger data touched only here) --------------------------------------------------------------
def score():
    if not FROZEN.exists():
        raise SystemExit("frozen.json missing: thresholds and sensitivity must be frozen (and committed) first")
    from gwosc.timeline import get_segments
    fr = json.loads(FROZEN.read_text())
    tsets = json.loads(TSETS.read_text())
    zs, cat = load_bg()
    model, dev = load_model()
    res = {"frozen": fr, "triggers": []}
    for t in triggers():
        row = {k: t[k] for k in ("name", "utc", "gps", "pipe", "far", "m1", "m2", "mc", "ifos", "snr", "cls")}
        if t["cls"] == "out-of-range":
            row["note"] = "chirp mass outside [0.173, 0.871]: not scored"
            res["triggers"].append(row); continue
        dets = [d for d in DETS if d[0] in t["ifos"]]
        # a 4096-s stretch fully inside each detector's DATA segments with the trigger well inside the crop
        seg = {}
        for d in dets:
            ok = get_segments(f"{d}_DATA", int(t["gps"]) - 4200, int(t["gps"]) + 4200)
            for off in (2048, 1024, 3072, 512, 3584):
                g = int(t["gps"]) - off
                if any(a <= g and g + C.SEGMENT_LEN <= b for a, b in ok) and \
                        g + (CROP + WIN + PAD) / FS + 1 < t["gps"] < g + C.SEGMENT_LEN - CROP / FS - 1:
                    seg[d] = g; break
        if len(seg) < len(dets):
            row["note"] = f"no clean 4096-s stretch around the trigger for {[d for d in dets if d not in seg]}"
            res["triggers"].append(row); continue
        vals = {}
        for d in dets:
            g = seg[d]
            fetch(d, g)
            w, t0, psd = whiten_segment(d, g)
            wc = w[CROP:-CROP]
            e = int(round((t["gps"] - t0) * FS)) - CROP
            vals[f"cnn_{d}"] = cnn_score(model, dev, wc[e - WIN:e])
            if t["cls"] == "primary" and tsets[t["name"]]["bank_applicable"]:
                dwin = wc[e - WIN - PAD:e + PAD]
                vals[f"bank_{d}"] = max(so.local_stats(dwin.astype(np.float64), template_chunks(mc, d, t0, psd), PAD)[1]
                                        for mc in tsets[t["name"]]["template_mcs"])
            segment_path(d, g).unlink(missing_ok=True)
        row["scores"] = vals
        if t["cls"] == "primary":
            row["verdicts"] = {}
            for stat in ("cnn", "bank"):
                st = fr["stats"][f"{t['name']}|{stat}"]
                if not st["applicable"]:
                    row["verdicts"][stat] = "INAPPLICABLE (template mismatch)"; continue
                key = (lambda d: f"cnn_{d}") if stat == "cnn" else (lambda d: f"bank_{t['name']}_{d}")
                x = vals[f"{stat}_H1"] + vals[f"{stat}_L1"]
                far = far_of(cat(key("H1")), cat(key("L1")), x)
                sig, eps = LOOKS * far <= 1.0, st["eps_at_rho_T"]
                v = ("INDEPENDENT SUPPORT" if sig and eps >= 0.5 else
                     "HIGH SCORE, NOISE EXCURSION FAVOURED" if sig else
                     "TENSION" if eps >= 0.5 else
                     "UNINFORMATIVE" if eps < 0.1 else "WEAKLY INFORMATIVE")
                row["verdicts"][stat] = {"x": x, "far_per_yr": far, "far_x_looks": LOOKS * far, "eps_at_rho_T": eps,
                                         "verdict": v}
                print(f"{t['name']} [{stat}] x = {x:.3f}  FAR {far:.3g}/yr (x{LOOKS} = {LOOKS*far:.3g})  "
                      f"eps(rho_T={st['rho_T']:.2f}) = {eps:.3f}  ->  {v}")
        else:
            if dets == ["L1"]:
                sL = cat("cnn_L1")
                row["single_L1_far_per_yr"] = float(np.sum(sL >= vals["cnn_L1"]) / (len(sL) * 64.0) * YEAR)
            else:
                x = vals["cnn_H1"] + vals["cnn_L1"]
                row["cnn_far_per_yr"] = far_of(cat("cnn_H1"), cat("cnn_L1"), x)
            print(f"{t['name']} ({t['cls']}): {vals}  -- no verdict by pre-registration")
        res["triggers"].append(row)
    purge(0)
    RESULT.write_text(json.dumps(res, indent=2))
    print(f"wrote {RESULT.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["discover", "select", "background", "freeze", "score"])
    {"discover": discover, "select": select, "background": background, "freeze": freeze,
     "score": score}[ap.parse_args().stage]()


if __name__ == "__main__":
    main()
