"""Step T reducer and G0 evaluation (registration reports/stepT_registration.md §3-§6, §11; clarifications C1-C17).
[model: Claude Opus 5.5]  UNVERIFIED: code and synthetic self-test only.

reduce  (CPU, imports pycbc; never the engine process): for every scored box file, PyCBC's v at each stored point
        (find_trigger_value on the segment's own v series) and PyCBC's newSNR (coh2t.rho_hat), then the registered
        H-anchored network value, per p in {4, 8, 16}:
            score = max over the K = 5 H1 anchors of  rho^_H^2 + max over |delta| <= 10 ms of rho^_L^2      (C2)
        plus the ablations: chi^2 only (v = 1) and plain rho (the engine's exact plain_net).
evaluate: thresholds at 1/day, R_tune and its segment-block bootstrap (2,000 resamples, both thresholds re-estimated
        in every resample), and the gates F1, F2, P1, C11 (K1's second clause), K1's first clause (if the K1-(A) maxima
        exist). One configuration per (L_max, p); the choice is the largest R_tune among configurations whose gates
        pass (registration §4). G0 itself is decided by the Bridge from this output.

Conventions fixed here (all already registered or ruled; listed so nothing is implicit):
  * stage-1 threshold thr1 (one per minute) is FIXED: it defines the candidate set; it is not re-estimated in the
    bootstrap (only the Y1-alone 1/day threshold and the cascade 1/day threshold are, registration §6);
  * a threshold at FAR f over livetime T is the score at position floor(f T) of the scores sorted loudest first
    (idealised_cascade's convention); with bootstrap weights, the first sorted position whose cumulative weight
    exceeds floor(f T_r);
  * cascade livetime = pairs x (4088 - 392) s (the C4 valid window); Y1-alone livetime = pairs x 4096 s (Y1's own);
  * slide pair (H1 test[i], L1 test[(i + k) mod 24]) gets bootstrap weight m_i * m_(i+k), injections weight m_segment;
  * distance = (3 V / 4 pi)^(1/3), V = (DC_HI - DC_LO) * weighted mean of 4 pi d_c^2 found (idealised_cascade);
  * found (cascade) = caught at stage 1 AND valid AND stage-2 score above the cascade threshold.

  python benchmark/coh2t_reduce.py reduce <l_max>
  python benchmark/coh2t_reduce.py evaluate <l_max> [<l_max> ...]
  python benchmark/coh2t_reduce.py selftest          (synthetic arrays only; no data, no pycbc ranking needed)
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

B = Path(__file__).resolve().parent
sys.path.insert(0, str(B)); sys.path.insert(0, str(B.parent)); sys.path.insert(0, str(B.parent.parent / "tools"))

FS = 4096
P_BINS = (4, 8, 16)
DELTA = int(round(0.010 * FS))
FAR_DAY = 1.0 / 86400.0
VALID_S = 4088.0 - 392.0
SEG_S = 4096.0
N_SLIDES = 24
N_BOOT = 2000
OUT = B / "results" / "coh2t"


def log(msg):
    import time
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------ the statistic (pure numpy, testable)
def network_score(H_rho, H_chi2, L_rho, L_chi2, vH, vL, rho_hat, pi, p):
    """H-anchored network value for one p. Shapes: H_rho [n, K], H_chi2 [n, K, 3], L_rho [n, K, D], L_chi2
    [n, K, 3, D], vH [n, K], vL [n, K, D]. rho_hat(rho, chi2_r, v) is PyCBC's (coh2t.rho_hat)."""
    dof = 2 * p - 2
    rH = rho_hat(H_rho, H_chi2[:, :, pi] / dof, vH)
    rL = rho_hat(L_rho, L_chi2[:, :, pi, :] / dof, vL)
    s = rH ** 2 + np.max(rL ** 2, axis=-1)
    out = np.max(s, axis=-1)
    assert np.isfinite(out).all(), "non-finite network score"
    return out


def weighted_threshold(scores, weights, f, T):
    """Score at the first loudest-first position whose cumulative weight exceeds floor(f T) (= the unweighted
    k-th-largest convention when all weights are 1); -inf if the background is too short."""
    k = np.floor(f * T)
    order = np.argsort(-scores, kind="stable")
    cw = np.cumsum(weights[order])
    i = int(np.searchsorted(cw, k, side="right"))
    return float(scores[order][i]) if i < len(scores) else -np.inf


def distance(found, dc, w=None, dc_lo=5.0, dc_hi=500.0):
    w = np.ones(len(dc)) if w is None else w
    V = (dc_hi - dc_lo) * np.sum(w * 4 * np.pi * dc ** 2 * found) / np.sum(w)
    return float((3 * V / (4 * np.pi)) ** (1 / 3))


# ------------------------------------------------------------------ reduce
def reduce(l_max):
    import coh2t
    from pycbc.psd.variation import find_trigger_value
    from pycbc.types import TimeSeries
    l_max = int(l_max)
    segs = json.loads((B / "manifest.json").read_text())["test"]
    vcache = {}

    def vser(det, gps):
        if (det, gps) not in vcache:
            z = np.load(B / "state" / "coh2t" / "v" / f"v_{det}_{gps}.npz")
            vcache[(det, gps)] = TimeSeries(z["values"], delta_t=float(z["delta_t"]), epoch=float(z["start"]))
        return vcache[(det, gps)]

    def v_at(det, gps, t):
        t = np.asarray(t, float)
        idx = ((t - gps) * FS).ravel()
        v = np.atleast_1d(find_trigger_value(vser(det, gps), idx, float(gps), FS)).reshape(t.shape)
        assert np.isfinite(v).all(), "non-finite v"
        return v
    out = {}
    files = sorted(glob.glob(str(OUT / f"L{l_max}" / "box_*.npz")))
    for f in files:
        z = np.load(f, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        kind = meta["kind"]
        cands = meta["candidates"]
        if not cands or kind.startswith("allen"):
            if kind.startswith("allen") and cands:
                out.setdefault(kind, []).append(dict(meta=cands, arr={k: z[k] for k in z.files if k != "meta"}))
            continue
        n = len(cands)
        dl = (np.arange(-DELTA, DELTA + 1) + DELTA) / FS                  # L1 lag offsets from L_gps0
        vH = np.stack([v_at("H1", segs[c["i"]], z["H_gps"][a]) for a, c in enumerate(cands)])
        vL = np.stack([v_at("L1", segs[c["iL"]], z["L_gps0"][a][:, None] + dl[None, :]) for a, c in enumerate(cands)])
        rec = dict(n=np.array([c["n"] for c in cands]), plain_net=z["plain_net"], tid=np.full(n, meta["tid"]))
        for pi, p in enumerate(P_BINS):
            args = (z["H_rho"].astype(np.float64), z["H_chi2"].astype(np.float64), z["L_rho"].astype(np.float64),
                    z["L_chi2"].astype(np.float64))
            rec[f"score_p{p}"] = network_score(*args, vH, vL, coh2t.rho_hat, pi, p)
            rec[f"score_chi2only_p{p}"] = network_score(*args, np.ones_like(vH), np.ones_like(vL), coh2t.rho_hat, pi, p)
        for k in ("rho_opt_trunc_net", "rho_opt_trunc_H1", "rho_opt_trunc_L1", "extra_only_plain", "extra_scale"):
            if k in z.files:
                rec[k] = z[k]
        out.setdefault(kind, []).append(dict(meta=cands, arr=rec))
    red = OUT / "reduced"
    red.mkdir(parents=True, exist_ok=True)
    for kind, parts in out.items():
        meta = [c for p_ in parts for c in p_["meta"]]
        keys = parts[0]["arr"].keys()
        arr = {k: np.concatenate([np.atleast_1d(p_["arr"][k]) for p_ in parts]) for k in keys}
        np.savez(red / f"L{l_max}_{kind}.npz", meta=json.dumps(meta), **arr)
        log(f"reduced L{l_max} {kind}: {len(meta)} candidates from {len(parts)} boxes")


def load_red(l_max, kind):
    p = OUT / "reduced" / f"L{l_max}_{kind}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    return json.loads(str(z["meta"])), {k: z[k] for k in z.files if k != "meta"}


# ------------------------------------------------------------------ evaluate
def y1_background(segs):
    """Y1's clustered background triggers with their (slide, H1 segment) labels."""
    import evaluate as E
    out = []
    for k in range(N_SLIDES):
        tr = E.read_triggers(B / "submissions" / "y1" / f"bg_k{k}.csv")
        for t, s_ in tr[:, :2]:
            i = next(j for j, g in enumerate(segs) if g <= t < g + SEG_S)
            out.append((k, i, s_))
    return np.array(out)


def evaluate_config(l_max, p, ctx, rng):
    """R_tune, its bootstrap and the gates for one (L_max, p)."""
    key = f"score_p{p}"
    bg_m, bg = ctx["red"]["bg"]
    fg_m, fg = ctx["red"]["fg"]
    fg_score = dict(zip([c["inj"] for c in fg_m], bg_none(fg, key)))
    rows = ctx["rows"]
    seg_of = np.array([ctx["segs"].index(r["segment"]) for r in rows])
    dc = np.array([r["chirp_distance"] for r in rows])
    caught = np.array([r["caught"] for r in rows])
    y1l = np.array([r["y1_loudest"] for r in rows])
    s_fg = np.array([fg_score.get(r["id"], -np.inf) for r in rows])
    bg_k = np.array([c["k"] for c in bg_m]); bg_i = np.array([c["i"] for c in bg_m]); bg_s = bg[key]
    y1 = ctx["y1bg"]
    y1_k, y1_i, y1_s = y1[:, 0].astype(int), y1[:, 1].astype(int), y1[:, 2]

    def one(m):
        wb = m[bg_i] * m[(bg_i + bg_k) % 24]
        wy = m[y1_i] * m[(y1_i + y1_k) % 24]
        pairs = sum(m[i] * m[(i + k) % 24] for k in range(N_SLIDES) for i in range(24))
        thr_c = weighted_threshold(bg_s, wb, FAR_DAY, pairs * VALID_S)
        thr_y = weighted_threshold(y1_s, wy, FAR_DAY, pairs * SEG_S)
        wi = m[seg_of]
        d_c = distance(caught & (s_fg > thr_c), dc, wi)
        d_y = distance(y1l > thr_y, dc, wi)
        return d_c / d_y, thr_c, thr_y, d_c, d_y
    R, thr_c, thr_y, d_c, d_y = one(np.ones(24))
    boot = np.array([one(np.bincount(rng.integers(0, 24, 24), minlength=24).astype(float))[0] for _ in range(N_BOOT)])
    res = dict(l_max=l_max, p=p, R_tune=R, ci90=[float(np.quantile(boot, 0.05)), float(np.quantile(boot, 0.95))],
               thr_cascade=thr_c, thr_y1=thr_y, d_cascade=d_c, d_y1=d_y, gates={})
    res["clause1"] = bool(R >= 1.15 and res["ci90"][0] > 1.00)
    g = res["gates"]
    # F1
    from scipy.stats import binom
    for kind in ("null", "wrong"):
        if ctx["red"].get(kind):
            m_, a_ = ctx["red"][kind]
            n_ = len(m_)
            xs = next(x for x in range(n_ + 1) if binom.sf(x, n_, 1 / 1440) <= 0.01)
            passes = int(np.sum(a_[key] > thr_c))
            g[f"F1_{kind}"] = dict(n=n_, passes=passes, x_star=xs, passed=bool(passes <= xs))
    # F2
    for kind in ("dec_sg", "dec_train", "dec_drift"):
        if ctx["red"].get(kind):
            m_, a_ = ctx["red"][kind]
            frac = float(np.mean(a_[key] > thr_c))
            e = dict(n=len(m_), pass_fraction=frac, passed=bool(frac <= 0.02))
            if kind == "dec_drift":
                e["pass_fraction_without_v"] = float(np.mean(a_[f"score_chi2only_p{p}"] > thr_c))
            g[f"F2_{kind}"] = e
    # P1 (C15: rho_opt = truncated true template, network; the full-signal oracle reported alongside)
    if ctx["red"].get("twin") and "rho_opt_trunc_net" in fg:
        tw_m, tw = ctx["red"]["twin"]
        thr_tw = weighted_threshold(tw[key], np.ones(len(tw[key])), FAR_DAY, 24 * N_SLIDES * VALID_S)
        ro = dict(zip([c["inj"] for c in fg_m], fg["rho_opt_trunc_net"]))
        ids = [c["inj"] for c in fg_m]
        rh = np.sqrt(np.maximum(fg[key], 0))
        ro_t = np.array([ro[i] for i in ids])
        q = np.array([ctx["q"][i] for i in ids])
        full = np.array([ctx["oracle"][i] for i in ids])
        strata = {}
        for name, msk in (("q<0.5", q < 0.5), ("q>=0.5", q >= 0.5)):
            strata[name] = dict(n=int(msk.sum()), median_ratio=float(np.median(rh[msk] / ro_t[msk])),
                                median_ratio_full=float(np.median(rh[msk] / full[msk])))
        loud = ro_t >= 15
        band = (ro_t >= 11) & (ro_t <= 14)
        g["P1"] = dict(strata=strata, thr_twin=thr_tw, n_loud=int(loud.sum()),
                       frac_loud_pass_twin=float(np.mean(fg[key][loud] > thr_tw)) if loud.any() else None,
                       band_11_14=dict(n=int(band.sum()), frac_pass_twin=float(np.mean(fg[key][band] > thr_tw)) if band.any() else None))
        g["P1"]["passed"] = bool(all(v_["median_ratio"] >= 0.90 for v_ in strata.values()) and loud.any()
                                 and g["P1"]["frac_loud_pass_twin"] >= 0.90)
        # K1 first clause (twin vs the K1-(A) maxima), only if they exist
        if ctx["red"].get("k1mc"):
            _, mc = ctx["red"]["k1mc"]
            x02 = float(np.quantile(tw[key], 1 - 1e-2))
            s_mc = float(np.mean(mc[key] > x02))
            g["K1_clause1"] = dict(x_at_twin_survival_1e2=x02, mc_survival=s_mc, ratio=s_mc / 1e-2,
                                   passed=bool(abs(s_mc / 1e-2 - 1) <= 0.15))
            # C18 (a): the deepest level with >= 20 MC events; the 10-event level descriptively; twin's own sd there
            n_mc, n_tw = len(mc[key]), len(tw[key])
            depth = {}
            for ev in (20, 10):
                x_d = float(np.sort(mc[key])[::-1][ev - 1])
                s_mc_d, s_tw_d = ev / n_mc, float(np.mean(tw[key] >= x_d))
                k_tw = s_tw_d * n_tw
                depth[f"{ev}_events"] = dict(x_d=x_d, s_mc=s_mc_d, s_twin=s_tw_d, ratio=s_tw_d / s_mc_d,
                                             twin_rel_sd=float(1 / np.sqrt(k_tw)) if k_tw > 0 else None)
            g["K1_depth"] = dict(depth, passed=bool(0.5 <= depth["20_events"]["ratio"] <= 2.0))
    res["gates_passed"] = bool(g) and all(v_.get("passed", True) for v_ in g.values())
    return res


def bg_none(a, key):
    return a[key] if a is not None else np.array([])


def allen(l_max):
    """C11: per family, detector and p: mean chi^2 against the bounds, with the registered bars."""
    out = {}
    for fam in ("allen_off", "allen_on", "allen_fail"):
        r = load_red(l_max, fam)
        if r is None:
            continue
        _, a = r
        for det in ("H1", "L1"):
            chi2, bnd = a[f"{det}_chi2"], a[f"{det}_bound"]
            nom = a[f"{det}_bound_nominal"]
            for pi, p in enumerate(P_BINS):
                c_, b_, n_ = chi2[:, pi], bnd[:, pi], nom[:, pi]
                se = float(np.std(c_ - b_, ddof=1) / np.sqrt(len(c_)))
                e = dict(n=len(c_), mean_chi2=float(c_.mean()), mean_bound=float(b_.mean()),
                         mean_nominal=float(n_.mean()), se=se)
                if fam == "allen_on":
                    se0 = float(np.std(c_, ddof=1) / np.sqrt(len(c_)))
                    e.update(noise_mean=2 * (p - 1), passed=bool(abs(c_.mean() - 2 * (p - 1)) <= 3 * se0),
                             found_rate=float(np.mean(a[f"{det}_on_found"])))
                elif fam == "allen_off":
                    e["passed"] = bool(c_.mean() <= b_.mean() + 3 * se)
                else:
                    sen = float(np.std(c_ - n_, ddof=1) / np.sqrt(len(c_)))
                    e["must_fail_exceeds_nominal"] = bool(c_.mean() > n_.mean() + 3 * sen)
                    if p == 16:
                        e["passed"] = e["must_fail_exceeds_nominal"]
                out[f"{fam}_{det}_p{p}"] = e
    return out


def k2eval(l_max, pi=2, p=16):
    """C18 (b): every K2 injection within 2 % on rho^_net, and on chi^2_r at the H1 and the L1 point of the anchor that
    sets the score. The shortcut takes v from the gated NOISE series, the real side from the injected strain's own
    v series (state/coh2t/k2/v_<det>_<inj>.npz). Reported for every p; the bar applies to every p."""
    import coh2t
    from pycbc.psd.variation import find_trigger_value
    from pycbc.types import TimeSeries
    f = OUT / f"K2_L{l_max}.npz"
    if not f.exists():
        return None
    z = np.load(f)
    segs = json.loads((B / "manifest.json").read_text())["test"]
    pl = json.loads((B / "state" / "coh2t" / "plan.json").read_text())
    fgc = {c["inj"]: c for c in pl["candidates"] if c["kind"] == "fg"}

    def vser(path):
        q = np.load(path)
        return TimeSeries(q["values"], delta_t=float(q["delta_t"]), epoch=float(q["start"]))
    rows, worst = [], dict(dev=0.0)
    dl = (np.arange(-DELTA, DELTA + 1) + DELTA) / FS
    for a_, inj in enumerate(z["inj"]):
        c = fgc[int(inj)]
        out = {}
        for side in ("short", "real"):
            vs = {det: vser(B / "state" / "coh2t" / ("v" if side == "short" else "k2") /
                            (f"v_{det}_{segs[c['i']]}.npz" if side == "short" else f"v_{det}_{int(inj)}.npz"))
                  for det in ("H1", "L1")}
            g = segs[c["i"]]
            Hg, Lg = z[f"{side}_H_gps"][a_], z[f"{side}_L_gps0"][a_][:, None] + dl[None, :]
            vH = np.atleast_1d(find_trigger_value(vs["H1"], (Hg - g) * FS, float(g), FS)).reshape(Hg.shape)
            vL = np.atleast_1d(find_trigger_value(vs["L1"], ((Lg - g) * FS).ravel(), float(g), FS)).reshape(Lg.shape)
            res = {}
            for pj, pp in enumerate(P_BINS):
                dof = 2 * pp - 2
                rH = coh2t.rho_hat(z[f"{side}_H_rho"][a_], z[f"{side}_H_chi2"][a_][:, pj] / dof, vH)
                rL = coh2t.rho_hat(z[f"{side}_L_rho"][a_], z[f"{side}_L_chi2"][a_][:, pj, :] / dof, vL)
                tot = rH ** 2 + np.max(rL ** 2, axis=-1)
                k = int(np.argmax(tot)); d = int(np.argmax(rL[k] ** 2))
                res[pp] = dict(rho_net=float(np.sqrt(tot[k])), chi2r_H=float(z[f"{side}_H_chi2"][a_][k, pj] / dof),
                               chi2r_L=float(z[f"{side}_L_chi2"][a_][k, pj, d] / dof), anchor=k)
            out[side] = res
        r = dict(inj=int(inj))
        for pp in P_BINS:
            s_, q_ = out["short"][pp], out["real"][pp]
            devs = {k: abs(s_[k] / q_[k] - 1) for k in ("rho_net", "chi2r_H", "chi2r_L")}
            r[f"p{pp}"] = dict(devs, short=s_, real=q_, passed=bool(max(devs.values()) <= 0.02))
            m = max(devs.values())
            if m > worst["dev"]:
                worst = dict(dev=m, inj=int(inj), p=pp, field=max(devs, key=devs.get))
        rows.append(r)
    return dict(rows=rows, worst=worst, passed=bool(all(r_[f"p{pp}"]["passed"] for r_ in rows for pp in P_BINS)))


def evaluate(*l_maxes):
    import csv
    segs = json.loads((B / "manifest.json").read_text())["test"]
    st = json.loads((B / "state" / "realbg" / "caught.json").read_text())
    with open(B / "results" / "oracle_snr.csv") as f:
        oracle = {int(r["id"]): float(r["snr_net"]) for r in csv.DictReader(f)}
    with open(B / "injections.csv") as f:
        q = {int(r["id"]): min(float(r["mass1"]), float(r["mass2"])) / max(float(r["mass1"]), float(r["mass2"]))
             for r in csv.DictReader(f)}
    ctx = dict(segs=segs, rows=st["rows"], y1bg=y1_background(segs), oracle=oracle, q=q)
    report = dict(configs=[], allen={})
    for l_max in map(int, l_maxes):
        ctx["red"] = {k: load_red(l_max, k) for k in ("bg", "fg", "twin", "null", "wrong", "dec_sg", "dec_train",
                                                     "dec_drift", "k1mc")}
        assert ctx["red"]["bg"] and ctx["red"]["fg"], f"L{l_max}: bg/fg not reduced"
        for p in P_BINS:
            r = evaluate_config(l_max, p, ctx, np.random.default_rng([20261126, l_max, p]))
            report["configs"].append(r)
            log(f"L{l_max} p{p}: R_tune {r['R_tune']:.3f} CI90 {r['ci90']} clause1 {r['clause1']} gates {r['gates_passed']}")
        report["allen"][f"L{l_max}"] = allen(l_max)
        report.setdefault("K2", {})[f"L{l_max}"] = k2eval(l_max)
    ok = [r for r in report["configs"] if r["gates_passed"]]
    report["chosen"] = max(ok, key=lambda r: r["R_tune"])["l_max"] if ok else None
    report["n_configs_tried"] = len(report["configs"])
    import envstamp
    report.update(model="Claude Opus 5.5", status="UNVERIFIED", env=envstamp.stamp(str(B.parent.parent)))
    (OUT / "evaluate.json").write_text(json.dumps(report, indent=1, default=float))


# ------------------------------------------------------------------ synthetic self-test
def selftest():
    rng = np.random.default_rng(0)

    def fake_rho_hat(rho, chi2_r, v):           # newsnr with q = 6, n = 2 and the v rules, written out for the test
        v = np.where(v < 0.65, 1.0, v)
        r, c = rho * v ** -0.5, chi2_r / v
        return np.where(c > 1, r * (0.5 * (1 + c ** 3)) ** (-1 / 6), r)
    n, K, D = 7, 5, 2 * DELTA + 1
    Hr = rng.uniform(3, 10, (n, K)); Hc = rng.uniform(5, 80, (n, K, 3))
    Lr = rng.uniform(1, 8, (n, K, D)); Lc = rng.uniform(5, 80, (n, K, 3, D))
    vH = rng.uniform(0.5, 1.5, (n, K)); vL = rng.uniform(0.5, 1.5, (n, K, D))
    got = network_score(Hr, Hc, Lr, Lc, vH, vL, fake_rho_hat, 2, 16)
    want = np.array([max(fake_rho_hat(Hr[a, k], Hc[a, k, 2] / 30, vH[a, k]) ** 2 +
                         max(fake_rho_hat(Lr[a, k, d], Lc[a, k, 2, d] / 30, vL[a, k, d]) ** 2 for d in range(D))
                         for k in range(K)) for a in range(n)])
    assert np.allclose(got, want), "network_score differs from the explicit loop"
    s = rng.normal(size=1000)
    for kk in (0, 5, 999):
        assert weighted_threshold(s, np.ones(1000), 1.0, kk) == np.sort(s)[::-1][kk]
    assert weighted_threshold(s, np.ones(1000), 1.0, 1000) == -np.inf
    w = np.ones(1000); w[np.argmax(s)] = 3           # the loudest counted three times
    assert weighted_threshold(s, w, 1.0, 2) == np.sort(s)[::-1][0]   # [L, L, L, s2, ...][2] = L
    assert weighted_threshold(s, w, 1.0, 3) == np.sort(s)[::-1][1]   # [L, L, L, s2, ...][3] = s2
    dc = rng.uniform(5, 500, 1000)
    assert abs(distance(np.ones(1000, bool), dc) - distance(np.ones(1000, bool), dc, np.full(1000, 2.0))) < 1e-12
    assert distance(np.zeros(1000, bool), dc) == 0.0
    print("selftest PASS: network_score == explicit loop; thresholds; weighted thresholds; distance", flush=True)


if __name__ == "__main__":
    a = sys.argv
    dict(reduce=reduce, evaluate=evaluate, selftest=selftest)[a[1]](*a[2:])
