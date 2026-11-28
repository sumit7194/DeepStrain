"""B4 measurement (registered in reports/stepT_runnotes.md, '10:55 B4 measurement' entry, BEFORE computing): how much
the hard top-5 selection's float32/float64 tie flips move the ranked statistic, and through it the 1/day threshold and
R_tune. Measurement only; scratch only (results/coh2t/c12_silent/b4/); BLIND: R_tune, its CI, thresholds and distances
are never written or printed, only their changes relative to the CI half-width.

  python benchmark/coh2t_c12_b4.py scores32                  # f32 score_p of every bg and fg candidate (stored arrays)
  python benchmark/coh2t_c12_b4.py sample <worker> <n_workers>   # blind 100-box sample (+ box 1481 control on worker 0)
  python benchmark/coh2t_c12_b4.py census <worker> <n_workers>   # every fg box + bg boxes within the census margin
  python benchmark/coh2t_c12_b4.py analyze
  python benchmark/coh2t_c12_b4.py bound                     # amendment (run notes ~13:30): MC + worst-case bound
"""
import json
import os
import sys

import numpy as np

import coh2t_c12_silent as S

D = S.D
B4 = S.SCR / "b4"
SEED_SAMPLE, N_SAMPLE = 20261128 + 4, 100
CONTROL = (1481, (6587, 19641, 34404))
PS = D.P_BINS
_np = np


# ------------------------------------------------------------------ capture of score()'s own H1 peak ordering
class _NP:
    """numpy, except that argsort inside score() also records the sorted H1 peak |z|^2 (score's own input)."""
    capture = None

    def __getattr__(self, k):
        return getattr(_np, k)

    def argsort(self, a, *args, **kw):
        if _NP.capture is not None and getattr(a, "ndim", 0) == 1:
            _NP.capture.append((-_np.sort(a)[:8]).tolist())          # a = -tpeak: the 8 largest peaks, descending
        return _np.argsort(a, *args, **kw)


def recompute_capture(tid, kind, meta):
    tops, orig = [], D.score

    def patched(box, fine, rH, rL, plain_only=False):
        _NP.capture = None if plain_only else tops
        try:
            return orig(box, fine, rH, rL, plain_only)
        finally:
            _NP.capture = None
    D.np, D.score = _NP(), patched
    try:
        meta64, a64, sec = S.recompute_f64(tid, kind, meta)
    finally:
        D.np, D.score = _np, orig
    assert len(tops) == len(meta["candidates"]), f"{len(tops)} captures for {len(meta['candidates'])} candidates"
    return meta64, a64, sec, tops


# ------------------------------------------------------------------ reduce's ranked statistic
_V, _SEGS = {}, None


def box_scores(meta, arr):
    import coh2t
    import coh2t_reduce as R
    from pycbc.psd.variation import find_trigger_value
    from pycbc.types import TimeSeries
    global _SEGS
    _SEGS = _SEGS or json.loads((S.B / "manifest.json").read_text())["test"]

    def v_at(det, gps, t):
        if (det, gps) not in _V:
            if len(_V) > 8:
                _V.clear()
            z = np.load(S.B / "state" / "coh2t" / "v" / f"v_{det}_{gps}.npz")
            _V[(det, gps)] = TimeSeries(z["values"], delta_t=float(z["delta_t"]), epoch=float(z["start"]))
        t = np.asarray(t, float)
        return np.atleast_1d(find_trigger_value(_V[(det, gps)], ((t - gps) * D.FS).ravel(), float(gps),
                                                D.FS)).reshape(t.shape)
    cands = meta["candidates"]
    dl = (np.arange(-D.DELTA, D.DELTA + 1) + D.DELTA) / D.FS
    vH = np.stack([v_at("H1", _SEGS[c["i"]], arr["H_gps"][a]) for a, c in enumerate(cands)])
    vL = np.stack([v_at("L1", _SEGS[c["iL"]], arr["L_gps0"][a][:, None] + dl[None, :]) for a, c in enumerate(cands)])
    args = [np.asarray(arr[k], np.float64) for k in ("H_rho", "H_chi2", "L_rho", "L_chi2")]
    return {p: R.network_score(*args, vH, vL, coh2t.rho_hat, pi, p) for pi, p in enumerate(PS)}


def one_box(tid, kind, role):
    meta, a32 = S.load_npz(S.PROD / f"box_{tid:04d}_{kind}.npz")
    assert "c12" not in meta, f"box {tid} {kind}: stored output is not float32"
    meta64, a64, sec, tops = recompute_capture(tid, kind, meta)
    s32, s64 = box_scores(meta, a32), box_scores(meta, a64)
    rows = []
    for a, c in enumerate(meta["candidates"]):
        t = tops[a]
        rows.append(dict(role=role, tid=tid, kind=kind, n=c["n"], inj=c.get("inj"), k=c.get("k"), i=c.get("i"),
                         same_selection=bool(np.array_equal(a32["tmpl"][a], a64["tmpl"][a])
                                             and np.array_equal(a32["lag"][a], a64["lag"][a])),
                         a={p: float(s32[p][a]) for p in PS}, b={p: float(s64[p][a]) for p in PS},
                         tpeak8=t, gap56=(t[4] - t[5]) / t[4] if len(t) > 5 and t[4] > 0 else None,
                         closure_f64=meta64.get("closure_max", 0.0), seconds=sec))
    return rows


def done(path):
    return {(r["tid"], r["kind"]) for r in S.done_rows(path)} if path.exists() else set()


# ------------------------------------------------------------------ commands
def scores32():
    pl = D.load_plan()
    out = []
    for kind in ("bg", "fg"):
        for tid in sorted({c["tid"] for c in pl["candidates"] if c["valid"] and c["kind"] == kind}):
            meta, a32 = S.load_npz(S.PROD / f"box_{tid:04d}_{kind}.npz")
            sc = box_scores(meta, a32)
            for a, c in enumerate(meta["candidates"]):
                out.append(dict(tid=tid, kind=kind, n=c["n"], k=c.get("k"), i=c.get("i"), inj=c.get("inj"),
                                arbitrated="c12" in meta, a={p: float(sc[p][a]) for p in PS}))
    B4.mkdir(parents=True, exist_ok=True)
    (B4 / "scores32.json").write_text(json.dumps(out))
    print(f"scores32: {sum(r['kind'] == 'bg' for r in out)} bg, {sum(r['kind'] == 'fg' for r in out)} fg candidates",
          flush=True)


def sample_boxes():
    pl = D.load_plan()
    prior = {r["tid"] for r in S.done_rows() if r.get("role") == "sample" and r["kind"] == "bg"}
    pop = sorted({c["tid"] for c in pl["candidates"] if c["valid"] and c["kind"] == "bg"})
    order = [int(t) for t in np.random.default_rng(SEED_SAMPLE).permutation(pop) if int(t) not in prior]
    return [t for t in order if "c12" not in S.load_npz(S.PROD / f"box_{t:04d}_bg.npz")[0]][:N_SAMPLE]


def sample(w, nw):
    w, nw = int(w), int(nw)
    path = B4 / f"sample_w{w}.jsonl"
    have = done(path)
    todo = ([(CONTROL[0], "control")] if w == 0 else []) + [(t, "sample") for j, t in enumerate(sample_boxes())
                                                             if j % nw == w]
    for tid, role in todo:
        if (tid, "bg") in have:
            continue
        for r in one_box(tid, "bg", role):
            S.record(r, path)


def census_boxes():
    rows = json.loads((B4 / "scores32.json").read_text())
    smp = [r for w in range(8) for r in S.done_rows(B4 / f"sample_w{w}.jsonl") if r["role"] == "sample"]
    assert len({r["tid"] for r in smp}) == N_SAMPLE, "the blind sample must finish before the census margin is set"
    jump = max((r["b"][str(p)] - r["a"][str(p)]) / r["a"][str(p)] for r in smp for p in PS)
    J = max(0.25, 1.5 * jump)
    bg = [r for r in rows if r["kind"] == "bg"]
    thr = {}
    for p in PS:
        s = np.array([r["a"][str(p)] for r in bg])
        thr[p] = D_thr(s)
    boxes = sorted({r["tid"] for r in bg if any(r["a"][str(p)] >= thr[p] / (1 + J) for p in PS)})
    fg = sorted({r["tid"] for r in rows if r["kind"] == "fg"})
    return [(t, "bg") for t in boxes] + [(t, "fg") for t in fg], J, jump


def D_thr(s):
    import coh2t_reduce as R
    return R.weighted_threshold(np.asarray(s), np.ones(len(s)), R.FAR_DAY, R.N_SLIDES * 24 * R.VALID_S)


def census(w, nw):
    w, nw = int(w), int(nw)
    path = B4 / f"census_w{w}.jsonl"
    have = {(r["tid"], r["kind"]) for x in range(8) for f in ("census", "sample")      # every worker's rows, so a
            for r in S.done_rows(B4 / f"{f}_w{x}.jsonl") if r["role"] in ("census", "sample")}  # re-split skips them
    boxes, J, jump = census_boxes()
    if os.environ.get("C12S_CENSUS_KINDS"):                   # run the census part by part (e.g. fg first)
        boxes = [b for b in boxes if b[1] in os.environ["C12S_CENSUS_KINDS"].split(",")]
    print(f"census: {len(boxes)} boxes (J = {J:.3f}, sample max jump {jump:.3f})", flush=True)
    for j, (tid, kind) in enumerate(boxes):
        if j % nw != w or (tid, kind) in have:
            continue
        for r in one_box(tid, kind, "census"):
            S.record(r, path)


def analyze():
    from scipy.stats import beta
    rows = [r for w in range(8) for f in ("sample", "census") for r in S.done_rows(B4 / f"{f}_w{w}.jsonl")]
    ctl = [r for r in rows if r["role"] == "control"]
    smp = [r for r in rows if r["role"] == "sample"]
    res = {}
    # control
    ok = all(any(r["n"] == n and abs(r["b"]["4"] - r["a"]["4"]) / r["b"]["4"] > 1e-3 and r["gap56"] < 1e-4
                 for r in ctl) for n in CONTROL[1])
    res["control_1481"] = dict(passed=bool(ok), rows=[{k: r[k] for k in ("n", "same_selection", "gap56")} for r in ctl])
    # sample
    n, x = len(smp), sum(not r["same_selection"] for r in smp)
    ci = [float(beta.ppf(0.025, x, n - x + 1)) if x else 0.0, float(beta.ppf(0.975, x + 1, n - x))]
    res["sample"] = dict(n_boxes=len({r["tid"] for r in smp}), n_cand=n, flips=x, flip_rate=x / n, cp95=ci)
    for p in PS:
        d = np.array([(r["b"][str(p)] - r["a"][str(p)]) / r["b"][str(p)] for r in smp])
        big = np.abs(d) > 1e-3
        res["sample"][f"p{p}"] = dict(frac_rel_gt_1e3=float(big.mean()), n_up=int((d > 1e-3).sum()),
                                      n_down=int((d < -1e-3).sum()),
                                      q=dict(zip(("1%", "5%", "50%", "95%", "99%"),
                                                 np.quantile(d, [.01, .05, .5, .95, .99]).tolist())),
                                      max_abs=float(np.abs(d).max()))
    g = np.array([r["gap56"] for r in smp if r["gap56"] is not None])
    fl = np.array([not r["same_selection"] for r in smp if r["gap56"] is not None])
    res["gap56"] = dict(n=len(g), frac_lt={e: float(np.mean(g < float(e))) for e in ("1e-3", "1e-4", "1e-5", "1e-6")},
                        flips_gap_max=float(g[fl].max()) if fl.any() else None,
                        nonflips_gap_min=float(g[~fl].min()) if (~fl).any() else None)
    # propagation (BLIND: only differences relative to the CI half-width)
    cen = [r for r in rows if r["role"] in ("census", "sample")]
    if any(r["role"] == "census" for r in rows):
        res["propagation"] = propagate(cen)
    out = B4 / "b4_summary.json"
    out.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))


def propagate(recomputed):
    import coh2t_reduce as R
    rows = json.loads((B4 / "scores32.json").read_text())
    f64 = {(r["kind"], r["n"]): r["b"] for r in recomputed}
    segs = json.loads((S.B / "manifest.json").read_text())["test"]
    st = json.loads((S.B / "state" / "realbg" / "caught.json").read_text())
    base = dict(segs=segs, rows=st["rows"], y1bg=R.y1_background(segs), oracle={}, q={})

    def ctx_of(use64):
        red = {}
        for kind in ("bg", "fg"):
            rs = [r for r in rows if r["kind"] == kind]
            meta = [dict(n=r["n"], k=r["k"], i=r["i"], inj=r["inj"]) for r in rs]
            arr = {f"score_p{p}": np.array([(f64.get((kind, r["n"])) or r["a"])[str(p)] if use64 else r["a"][str(p)]
                                            for r in rs]) for p in PS}
            red[kind] = (meta, arr)
        return dict(base, red=red)
    c32, c64 = ctx_of(False), ctx_of(True)
    out, verdict = {}, "NEGLIGIBLE"
    best = {}
    for p in PS:
        r32 = R.evaluate_config(128, p, c32, np.random.default_rng([20261126, 128, p]))
        r64 = R.evaluate_config(128, p, c64, np.random.default_rng([20261126, 128, p]))
        h = (r32["ci90"][1] - r32["ci90"][0]) / 2
        dR, dlo = (r64["R_tune"] - r32["R_tune"]) / h, (r64["ci90"][0] - r32["ci90"][0]) / h
        c1 = r32["clause1"] != r64["clause1"]
        s2 = (r32["R_tune"] < 1.0 or r32["ci90"][1] < 1.15) != (r64["R_tune"] < 1.0 or r64["ci90"][1] < 1.15)
        out[f"p{p}"] = dict(dR_over_h=dR, dCIlo_over_h=dlo, dthr_rel=(r64["thr_cascade"] - r32["thr_cascade"]) /
                            r32["thr_cascade"], clause1_changes=bool(c1), S2_changes=bool(s2))
        best[p] = (r32["R_tune"], r64["R_tune"])
        if abs(dR) > 0.5 or abs(dlo) > 0.5 or c1 or s2:
            verdict = "MATERIAL"
        elif verdict == "NEGLIGIBLE" and (abs(dR) > 0.25 or abs(dlo) > 0.25):
            verdict = "INCONCLUSIVE"
    out["best_config_changes"] = bool(max(best, key=lambda p: best[p][0]) != max(best, key=lambda p: best[p][1]))
    if out["best_config_changes"]:
        verdict = "MATERIAL"
    out["reading"] = verdict
    out["n_recomputed"] = dict(bg=sum(r["kind"] == "bg" for r in recomputed), fg=sum(r["kind"] == "fg" for r in recomputed))
    return out


# ------------------------------------------------------------------ amendment ~13:30: the bound (blind)
SEED_MC, N_MC, P_FLIP, J_WC = 20261128 + 5, 2000, 0.0030, 0.25
JUMPS = None                                                  # filled from box 1481's flips (the control rows)


def _static(ctx, p):
    """Everything in evaluate_config's one() that does not depend on the bg or fg scores, per bootstrap draw (the
    same draws: one(np.ones(24)) first, then N_BOOT draws from default_rng([20261126, 128, p]))."""
    import coh2t_reduce as R
    rows = ctx["rows"]
    seg_of = np.array([ctx["segs"].index(r["segment"]) for r in rows])
    dc = np.array([r["chirp_distance"] for r in rows])
    caught = np.array([r["caught"] for r in rows])
    y1l = np.array([r["y1_loudest"] for r in rows])
    y1 = ctx["y1bg"]
    y1_k, y1_i, y1_s = y1[:, 0].astype(int), y1[:, 1].astype(int), y1[:, 2]
    rng = np.random.default_rng([20261126, 128, p])
    ms = [np.ones(24)] + [np.bincount(rng.integers(0, 24, 24), minlength=24).astype(float) for _ in range(R.N_BOOT)]
    per = []
    for m in ms:
        wy = m[y1_i] * m[(y1_i + y1_k) % 24]
        pairs = sum(m[i] * m[(i + k) % 24] for k in range(R.N_SLIDES) for i in range(24))
        thr_y = R.weighted_threshold(y1_s, wy, R.FAR_DAY, pairs * R.SEG_S)
        wi = m[seg_of]
        per.append((m, pairs, wi, R.distance(y1l > thr_y, dc, wi)))
    return dict(per=per, dc=dc, caught=caught, ids=[r["id"] for r in rows])


def _fast(st, bg_s, bg_i, bg_k, fg_ids, fg_s):
    """evaluate_config's (R_tune, ci90, thr_cascade), exactly, for these scores."""
    import coh2t_reduce as R
    fs = dict(zip(fg_ids, fg_s))
    s_fg = np.array([fs.get(i, -np.inf) for i in st["ids"]])
    order = np.argsort(-bg_s, kind="stable")
    ss, bi, bk = bg_s[order], bg_i[order], bg_k[order]
    out = []
    for m, pairs, wi, d_y in st["per"]:
        cw = np.cumsum(m[bi] * m[(bi + bk) % 24])
        j = int(np.searchsorted(cw, np.floor(R.FAR_DAY * pairs * R.VALID_S), side="right"))
        thr = float(ss[j]) if j < len(ss) else -np.inf
        out.append((R.distance(st["caught"] & (s_fg > thr), st["dc"], wi) / d_y, thr))
    boot = np.array([o[0] for o in out[1:]])
    return out[0][0], [float(np.quantile(boot, 0.05)), float(np.quantile(boot, 0.95))], out[0][1]


def bound():
    import math
    import coh2t_reduce as R
    rows = json.loads((B4 / "scores32.json").read_text())
    bgr, fgr = [r for r in rows if r["kind"] == "bg"], [r for r in rows if r["kind"] == "fg"]
    cen = {r["n"]: r["b"] for w in range(8) for r in S.done_rows(B4 / f"census_w{w}.jsonl") if r["kind"] == "fg"}
    assert all(r["n"] in cen for r in fgr), f"fg census incomplete: {sum(r['n'] in cen for r in fgr)}/{len(fgr)}"
    ctl = [r for w in range(8) for r in S.done_rows(B4 / f"sample_w{w}.jsonl") if r["role"] == "control"
           and not r["same_selection"]]
    jumps = sorted({abs(r["b"][str(p)] / r["a"][str(p)] - 1) for r in ctl for p in PS} - {0.0})
    jumps = np.array([j for j in jumps if j > 1e-3])
    segs = json.loads((S.B / "manifest.json").read_text())["test"]
    st_ = json.loads((S.B / "state" / "realbg" / "caught.json").read_text())
    ctx = dict(segs=segs, rows=st_["rows"], y1bg=R.y1_background(segs), oracle={}, q={})
    bg_i, bg_k = np.array([r["i"] for r in bgr]), np.array([r["k"] for r in bgr])
    fg_ids = [r["inj"] for r in fgr]
    A = {p: np.array([r["a"][str(p)] for r in bgr]) for p in PS}
    F32 = {p: np.array([r["a"][str(p)] for r in fgr]) for p in PS}
    F64 = {p: np.array([cen[r["n"]][str(p)] for r in fgr]) for p in PS}
    n_f = math.ceil(P_FLIP * len(bgr))
    res = dict(n_bg=len(bgr), n_fg=len(fgr), n_flip=n_f, jumps=jumps.tolist(), p_flip=P_FLIP, j_wc=J_WC)
    base, st = {}, {}
    for p in PS:                                                # gate: the replica equals evaluate_config, exactly
        st[p] = _static(ctx, p)
        base[p] = _fast(st[p], A[p], bg_i, bg_k, fg_ids, F32[p])
        red = {"bg": ([dict(k=r["k"], i=r["i"]) for r in bgr], {f"score_p{p}": A[p]}),
               "fg": ([dict(inj=r["inj"]) for r in fgr], {f"score_p{p}": F32[p]})}
        ref = R.evaluate_config(128, p, dict(ctx, red=red), np.random.default_rng([20261126, 128, p]))
        ok = (abs(ref["R_tune"] - base[p][0]) <= 1e-12 * abs(ref["R_tune"]) and
              all(abs(a - b) <= 1e-12 * abs(a) for a, b in zip(ref["ci90"], base[p][1])) and
              ref["thr_cascade"] == base[p][2])
        res[f"replica_matches_evaluate_config_p{p}"] = bool(ok)
        del ref
        assert ok, f"p{p}: the fast replica does not reproduce evaluate_config; the bound is void"

    def verdicts(vals):
        """vals: {p: (R, ci90, thr)} -> per-p clause1 / S2 flags and the best config (in memory only)."""
        c1 = {p: bool(v[0] >= 1.15 and v[1][0] > 1.00) for p, v in vals.items()}
        s2 = {p: bool(v[0] < 1.00 or v[1][1] < 1.15) for p, v in vals.items()}
        bp = max(vals, key=lambda p: vals[p][0])
        return c1, s2, bp, s2[bp]
    b_c1, b_s2, b_bp, b_s2best = verdicts(base)
    h = {p: (base[p][1][1] - base[p][1][0]) / 2 for p in PS}

    def delta(vals):
        c1, s2, bp, s2b = verdicts(vals)
        d = {f"p{p}": dict(dR_over_h=(vals[p][0] - base[p][0]) / h[p], dCIlo_over_h=(vals[p][1][0] - base[p][1][0]) / h[p],
                           dthr_rel=(vals[p][2] - base[p][2]) / base[p][2], clause1_changes=c1[p] != b_c1[p],
                           S2_changes=s2[p] != b_s2[p]) for p in PS}
        d["best_config_changes"] = bp != b_bp
        d["S2_best_changes"] = s2b != b_s2best
        return d

    def read(d):
        dv = [d[f"p{p}"] for p in PS]
        if any(abs(x["dR_over_h"]) > 0.5 or abs(x["dCIlo_over_h"]) > 0.5 or x["clause1_changes"] or x["S2_changes"]
               for x in dv) or d["best_config_changes"] or d["S2_best_changes"]:
            return "MATERIAL"
        return "NEGLIGIBLE" if all(abs(x["dR_over_h"]) <= 0.25 and abs(x["dCIlo_over_h"]) <= 0.25 for x in dv) \
            else "INCONCLUSIVE"
    # fg only
    d = delta({p: _fast(st[p], A[p], bg_i, bg_k, fg_ids, F64[p]) for p in PS})
    res["fg_only"] = dict(d, reading=read(d))
    # (ii-b) worst cases
    for name in ("WC_down", "WC_up"):
        vals = {}
        for p in PS:
            a = A[p].copy()
            thr0 = D_thr(A[p])
            if name == "WC_down":
                idx = np.where(a < thr0)[0]
                idx = idx[np.argsort(-a[idx], kind="stable")[:n_f]]
                a[idx] *= 1 + J_WC
            else:
                idx = np.argsort(-a, kind="stable")[:n_f]
                a[idx] *= 1 - J_WC
            vals[p] = _fast(st[p], a, bg_i, bg_k, fg_ids, F64[p])
        d = delta(vals)
        res[name] = dict(d, reading=read(d))
    # (ii-a) Monte Carlo
    rng = np.random.default_rng(SEED_MC)
    draws = []
    for _ in range(N_MC):
        flip = rng.random(len(bgr)) < P_FLIP
        vals = {}
        for p in PS:
            a = A[p].copy()
            nfl = int(flip.sum())
            a[flip] *= 1 + rng.choice([-1.0, 1.0], nfl) * rng.choice(jumps, nfl)
            vals[p] = _fast(st[p], a, bg_i, bg_k, fg_ids, F64[p])
        draws.append(delta(vals))
    mc = {}
    for p in PS:
        dr = np.array([abs(x[f"p{p}"]["dR_over_h"]) for x in draws])
        dl = np.array([abs(x[f"p{p}"]["dCIlo_over_h"]) for x in draws])
        mc[f"p{p}"] = dict(p95_abs_dR_over_h=float(np.quantile(dr, 0.95)), p95_abs_dCIlo_over_h=float(np.quantile(dl, 0.95)),
                           frac_clause1_changes=float(np.mean([x[f"p{p}"]["clause1_changes"] for x in draws])),
                           frac_S2_changes=float(np.mean([x[f"p{p}"]["S2_changes"] for x in draws])))
    mc["frac_best_config_changes"] = float(np.mean([x["best_config_changes"] for x in draws]))
    mc["frac_S2_best_changes"] = float(np.mean([x["S2_best_changes"] for x in draws]))
    vchg = max([mc[f"p{p}"]["frac_clause1_changes"] for p in PS] + [mc[f"p{p}"]["frac_S2_changes"] for p in PS] +
               [mc["frac_best_config_changes"], mc["frac_S2_best_changes"]])
    worst = max(max(mc[f"p{p}"]["p95_abs_dR_over_h"], mc[f"p{p}"]["p95_abs_dCIlo_over_h"]) for p in PS)
    mc["reading"] = ("MATERIAL" if worst > 0.5 or vchg > 0.05 else "NEGLIGIBLE" if worst <= 0.25 else "INCONCLUSIVE")
    res["MC"] = mc
    res["overall"] = ("NO bg census needed" if all(res[k]["reading"] == "NEGLIGIBLE" for k in ("MC", "WC_down", "WC_up"))
                      else "run the full census (i)" if any(res[k]["reading"] != "NEGLIGIBLE" for k in ("WC_down", "WC_up"))
                      else "MC not NEGLIGIBLE: report")
    (B4 / "bound.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float), flush=True)


def wcbit():
    """Post-hoc one-bit check (run notes ~15:55, committed before running): under WC-down, does clause1 of the BEST
    config (argmax R over p) change against baseline? Prints and writes only that bit."""
    import math
    import coh2t_reduce as R
    rows = json.loads((B4 / "scores32.json").read_text())
    bgr, fgr = [r for r in rows if r["kind"] == "bg"], [r for r in rows if r["kind"] == "fg"]
    cen = {r["n"]: r["b"] for w in range(8) for r in S.done_rows(B4 / f"census_w{w}.jsonl") if r["kind"] == "fg"}
    assert all(r["n"] in cen for r in fgr)
    segs = json.loads((S.B / "manifest.json").read_text())["test"]
    st_ = json.loads((S.B / "state" / "realbg" / "caught.json").read_text())
    ctx = dict(segs=segs, rows=st_["rows"], y1bg=R.y1_background(segs), oracle={}, q={})
    bg_i, bg_k = np.array([r["i"] for r in bgr]), np.array([r["k"] for r in bgr])
    fg_ids = [r["inj"] for r in fgr]
    n_f = math.ceil(P_FLIP * len(bgr))
    base, wc = {}, {}
    for p in PS:
        st = _static(ctx, p)
        A = np.array([r["a"][str(p)] for r in bgr])
        base[p] = _fast(st, A, bg_i, bg_k, fg_ids, np.array([r["a"][str(p)] for r in fgr]))
        a = A.copy()
        idx = np.where(a < D_thr(A))[0]
        idx = idx[np.argsort(-a[idx], kind="stable")[:n_f]]
        a[idx] *= 1 + J_WC
        wc[p] = _fast(st, a, bg_i, bg_k, fg_ids, np.array([cen[r["n"]][str(p)] for r in fgr]))

    def c1_best(vals):
        bp = max(vals, key=lambda p: vals[p][0])
        return bool(vals[bp][0] >= 1.15 and vals[bp][1][0] > 1.00)
    bit = c1_best(base) != c1_best(wc)
    (B4 / "wcbit.json").write_text(json.dumps({"clause1_of_best_changes_under_WC_down": bit}))
    print(json.dumps({"clause1_of_best_changes_under_WC_down": bit}), flush=True)


if __name__ == "__main__":
    S.scratch_state()
    B4.mkdir(parents=True, exist_ok=True)
    S.PL = D.load_plan()
    cmd = sys.argv[1]
    {"scores32": scores32, "sample": sample, "census": census, "analyze": analyze, "bound": bound, "wcbit": wcbit}[cmd](*sys.argv[2:])
