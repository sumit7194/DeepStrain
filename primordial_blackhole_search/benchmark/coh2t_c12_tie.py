"""Box-1481 diagnostic (registered in reports/stepT_runnotes.md, '10:46 box-1481 diagnostic' entry, BEFORE computing):
is the float32-vs-float64 selection change a harmless near-tie or a real error? Per candidate, reduce's own ranked
statistic score_p (p = 4, 8, 16) from (a) the stored float32 arrays, (b) the float64 recompute, (c) float64 band pieces
at float32's anchors. Harmless iff |a-b|/b <= 1e-3 and |c-b|/b <= 1e-3 for every p. Scratch only (as coh2t_c12_silent).

  python benchmark/coh2t_c12_tie.py [tid] [kind]       (default 1481 bg)
"""
import os
for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import json
import sys

import numpy as np

import coh2t_c12_silent as S

D = S.D
TOL = 1e-3


def eval_at(box, fine, rH, rL, tmpls, lags):
    """The tail of D.score() (anchor values) at GIVEN (template, lag) anchors."""
    K, nd = D.K_PEAKS, 2 * D.DELTA + 1
    out = dict(H_rho=np.zeros(K), H_chi2=np.zeros((K, 3)), L_rho=np.zeros((K, nd)), L_chi2=np.zeros((K, 3, nd)))
    for a, (tidx, lg) in enumerate(zip(tmpls, lags)):
        n0 = rH["i_s"] - box.eng.half + int(lg)
        rho_h, c_h = D.chi2_of(D.band_series(box, fine, rH, int(tidx), n0)[D.DELTA])
        rho_l, c_l = D.chi2_of(D.band_series(box, fine, rL, int(tidx), n0))
        out["H_rho"][a], out["H_chi2"][a] = rho_h, [c_h[p] for p in D.P_BINS]
        out["L_rho"][a], out["L_chi2"][a] = rho_l, np.stack([c_l[p] for p in D.P_BINS])
    return out


def main(tid=1481, kind="bg"):
    S.scratch_state()
    S.PL = D.load_plan()
    tid = int(tid)
    meta_st, a32 = S.load_npz(S.PROD / f"box_{tid:04d}_{kind}.npz")
    assert "c12" not in meta_st, "stored output is not float32"
    n = len(meta_st["candidates"])
    forced, extra, calls, orig = list(zip(a32["tmpl"], a32["lag"])), [], [0], D.score

    def patched(box, fine, rH, rL, plain_only=False):
        o = orig(box, fine, rH, rL, plain_only)
        if not plain_only:
            extra.append(eval_at(box, fine, rH, rL, *forced[calls[0]]))
            calls[0] += 1
        return o
    D.score = patched
    try:
        meta64, a64, sec = S.recompute_f64(tid, kind, meta_st)
    finally:
        D.score = orig
    assert calls[0] == n, f"{calls[0]} score calls for {n} candidates"
    ac = {k: np.stack([e[k] for e in extra]) for k in ("H_rho", "H_chi2", "L_rho", "L_chi2")}

    import coh2t
    import coh2t_reduce as R
    from pycbc.psd.variation import find_trigger_value
    from pycbc.types import TimeSeries
    segs = json.loads((S.B / "manifest.json").read_text())["test"]
    vc = {}

    def v_at(det, gps, t):
        if (det, gps) not in vc:
            z = np.load(S.B / "state" / "coh2t" / "v" / f"v_{det}_{gps}.npz")
            vc[(det, gps)] = TimeSeries(z["values"], delta_t=float(z["delta_t"]), epoch=float(z["start"]))
        t = np.asarray(t, float)
        return np.atleast_1d(find_trigger_value(vc[(det, gps)], ((t - gps) * D.FS).ravel(), float(gps),
                                                D.FS)).reshape(t.shape)
    cands = meta_st["candidates"]
    dl = (np.arange(-D.DELTA, D.DELTA + 1) + D.DELTA) / D.FS

    def scores(arr, H_gps, L_gps0):
        vH = np.stack([v_at("H1", segs[c["i"]], H_gps[a]) for a, c in enumerate(cands)])
        vL = np.stack([v_at("L1", segs[c["iL"]], L_gps0[a][:, None] + dl[None, :]) for a, c in enumerate(cands)])
        args = [np.asarray(arr[k], np.float64) for k in ("H_rho", "H_chi2", "L_rho", "L_chi2")]
        return {p: R.network_score(*args, vH, vL, coh2t.rho_hat, pi, p) for pi, p in enumerate(D.P_BINS)}

    def argmax_anchor(arr, H_gps, L_gps0, p=16):
        pi = D.P_BINS.index(p)
        vH = np.stack([v_at("H1", segs[c["i"]], H_gps[a]) for a, c in enumerate(cands)])
        vL = np.stack([v_at("L1", segs[c["iL"]], L_gps0[a][:, None] + dl[None, :]) for a, c in enumerate(cands)])
        dof = 2 * p - 2
        rH = coh2t.rho_hat(np.asarray(arr["H_rho"], float), np.asarray(arr["H_chi2"], float)[:, :, pi] / dof, vH)
        rL = coh2t.rho_hat(np.asarray(arr["L_rho"], float), np.asarray(arr["L_chi2"], float)[:, :, pi, :] / dof, vL)
        return np.argmax(rH ** 2 + np.max(rL ** 2, axis=-1), axis=-1)
    sa, sb = scores(a32, a32["H_gps"], a32["L_gps0"]), scores(a64, a64["H_gps"], a64["L_gps0"])
    sc = scores(ac, a32["H_gps"], a32["L_gps0"])
    ma, mb = argmax_anchor(a32, a32["H_gps"], a32["L_gps0"]), argmax_anchor(a64, a64["H_gps"], a64["L_gps0"])
    rows, harmless = [], True
    for a in range(n):
        same = bool(np.array_equal(a32["tmpl"][a], a64["tmpl"][a]) and np.array_equal(a32["lag"][a], a64["lag"][a]))
        r = dict(cand=a, n=cands[a]["n"], y1=cands[a]["y1"], same_selection=same,
                 tmpl_f32=a32["tmpl"][a].tolist(), lag_f32=a32["lag"][a].tolist(),
                 tmpl_f64=a64["tmpl"][a].tolist(), lag_f64=a64["lag"][a].tolist(),
                 H_rho_f32=np.round(a32["H_rho"][a], 5).tolist(), H_rho_f64=np.round(a64["H_rho"][a], 5).tolist(),
                 max_anchor_f32=int(a32["tmpl"][a][ma[a]]), max_anchor_f64=int(a64["tmpl"][a][mb[a]]),
                 score={})
        for p in D.P_BINS:
            A, Bv, Cv = float(sa[p][a]), float(sb[p][a]), float(sc[p][a])
            dab, dcb = abs(A - Bv) / Bv, abs(Cv - Bv) / Bv
            r["score"][p] = dict(a_f32=A, b_f64=Bv, c_f64_at_f32sel=Cv, rel_ab=dab, rel_cb=dcb)
            if not same and (dab > TOL or dcb > TOL):
                harmless = False
        rows.append(r)
    diff = [r for r in rows if not r["same_selection"]]
    res = dict(tid=tid, kind=kind, n_cand=n, n_differing=len(diff), tol=TOL,
               reading=("HARMLESS near-tie" if harmless else "REAL ERROR") if diff else "no selection change",
               f64_closure=meta64.get("closure_max"), seconds=sec, rows=rows)
    out = S.SCR / f"tie_{tid}_{kind}.json"
    out.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))
    for r in diff:
        print(json.dumps(r, default=float))


if __name__ == "__main__":
    main(*sys.argv[1:])
