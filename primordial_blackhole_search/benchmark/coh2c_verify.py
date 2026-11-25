"""DeepStrain's independent checks of the Fable (c) round (reports/coh2c_handover.md, 4f01c22).  [model: Claude Opus 5.5]

Checks the handover's own tests cannot make:
  V1  whole-box brute force: build the constructed grid for a real box, evaluate EVERY grid template with the local
      engine and with an independent numpy brute force (pycbc template, one-sided product, full 2^21 inverse FFT), and
      compare the box maximum over templates and +-1 s of lags.
  V3  reference assignment over a real grid: every grid template's group-delay difference to its assigned reference
      must be <= dtau_max (1.5 s at tw = 4).
  V2  L1: the same as V1 on an L1 tuning segment.
  D2  start-frequency deviation: the box maximum with each template at its OWN start frequency (the frozen text) vs at
      its reference's (the engine), on the same data and grid.
Full rate only (tw = 4, dec = 1: the handover's recommended D1 setting).

  python benchmark/coh2c_verify.py box <det> <gps> <inj ids (first = target, all = cached series) or -> <mc_star or -> <l_max> [n_sub]   (one box, JSON line)
Device: COH2C_DEV (cpu / mps), as the engine.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

B = Path(__file__).resolve().parent
sys.path.insert(0, str(B)); sys.path.insert(0, str(B.parent))
import coh2c_local as L  # noqa: E402

OUT = B / "results" / "coh2c_verify"


def grid_points(mc_star, half_width, l_max):
    """coh2c_grid.build, unmodified except that it also returns the grid's (x, q) points."""
    src = (B / "coh2c_grid.py").read_text()
    src = src.replace("    return dict(mc_star=mc_star,", "    return grid_xq, dict(mc_star=mc_star,", 1)
    ns = {"__name__": "coh2c_grid_copy", "__file__": str(B / "coh2c_grid.py")}
    exec(compile(src, "coh2c_grid_copy", "exec"), ns)
    pts, summ = ns["build"](mc_star, half_width, l_max, 1.0, np.random.default_rng(1))
    return [(mc_star * float(np.exp(x)), float(q)) for x, q in pts], summ


def box(det, gps, inj, mc_star, l_max, n_sub=None):
    import torch
    import y1_bank as Y1
    dev = L.pick_dev()
    sign = json.loads((L.OUT / "E1_phase_model.json").read_text())["sign"]
    hw = json.loads((B / "results" / "coh2" / "T1_mc_bias.json").read_text())["all"]["p99"]
    st = json.loads((B / "state" / "realbg" / "caught.json").read_text())
    segs = json.loads((B / "manifest.json").read_text())["test"]
    inj_ids = () if inj is None else tuple(inj)
    inj = None if inj is None else inj[0]
    y, w_snip, rows = L.load_ww(det, gps, inj_ids)
    if inj is None:
        i = segs.index(gps)
        cand = [b for b in st["candidates"][f"0_{i}"] if (L.PRE + 1) * L.FS < b * 102 < 4096 * L.FS - 65 * L.FS][0] * 102
        i_c = cand
    else:
        i_c = rows[inj]["merger"] + 333
        if mc_star is None:
            mc_star = float(Y1.MCS[np.argmin(np.abs(np.log(Y1.MCS / rows[inj]["mc"])))])
    t0 = time.time()
    pts, gsum = grid_points(mc_star, hw, l_max)
    if n_sub and n_sub < len(pts):
        pts = [pts[k] for k in np.random.default_rng(7).choice(len(pts), n_sub, replace=False)]
    eng = L.Engine(dev, 4.0, 1)
    refs, lat_worst = L.reference_lattice(mc_star, hw, l_max, L.dtau_max(4.0))
    ysn = L.snippet_spectrum(y, i_c)
    kappa = 1.0 / np.median(np.abs(ysn[L.NS // 2048: L.NS // 8]))
    ysn_t = torch.tensor((ysn * kappa).astype(np.complex64), device=dev)
    i_s = L.PRE * L.FS
    x0s, sig0 = [], []
    for mc_r, q_r, f_lo, _ in refs:
        h0 = L.fd_template(mc_r, q_r, f_lo)
        sig0.append(np.sqrt(2 * np.sum(np.abs(h0 * w_snip) ** 2) / L.NS))
        x0 = L.analytic_corr(torch, ysn_t, torch.tensor(np.conj(h0).astype(np.complex64), device=dev))
        x0s.append(eng.window(x0, i_s)); del x0
    # V3 + local engine, batched per reference
    assign = [L.assign(L.coeffs(mc, q), refs) for mc, q in pts]
    v3_worst = max(d for _, d in assign)
    loc = np.empty(len(pts))
    for j in range(len(refs)):
        idx = [k for k, (jj, _) in enumerate(assign) if jj == j]
        for s in range(0, len(idx), 256):
            part = idx[s:s + 256]
            dco = torch.tensor(np.stack([L.coeffs(*pts[k]) - refs[j][3] for k in part]).astype(np.float32), device=dev)
            z = eng.z(x0s[j], dco, sign).cpu().numpy().astype(np.complex128) / (sig0[j] * kappa)
            loc[part] = (np.abs(z) ** 2).max(axis=1)
    t_loc = time.time() - t0
    # independent brute force (numpy), at the reference's start frequency (engine) and at the template's own (frozen)
    P = np.zeros(L.NS, complex)
    bf, own = np.empty(len(pts)), np.full(len(pts), np.nan)

    def peak(h):
        sig = np.sqrt(2 * np.sum(np.abs(h * w_snip) ** 2) / L.NS)
        P[: L.NS // 2 + 1] = ysn * np.conj(h)
        return float((np.abs(2 * np.fft.ifft(P)[i_s - L.FS: i_s + L.FS + 1] / sig) ** 2).max())
    for k, ((mc, q), (j, _)) in enumerate(zip(pts, assign)):
        bf[k] = peak(L.fd_template(mc, q, refs[j][2]))
        f_own = L.f_start(mc, l_max)
        own[k] = bf[k] if abs(f_own - refs[j][2]) < 1e-9 else peak(L.fd_template(mc, q, f_own))
    rel = np.abs(loc / bf - 1)
    out = dict(check="V1/V2/V3/D2", det=det, segment=gps, injection=inj, mc_star=mc_star, l_max=l_max, device=dev,
               n_templates=len(pts), n_grid_full=gsum["n_grid"], n_refs=len(refs), lattice_worst_dtau=lat_worst,
               v3_worst_assigned_dtau=float(v3_worst), dtau_rule=L.dtau_max(4.0), v3_pass=bool(v3_worst <= L.dtau_max(4.0)),
               box_max_local=float(loc.max()), box_max_brute=float(bf.max()),
               box_max_rel_err=float(abs(loc.max() / bf.max() - 1)),
               same_argmax=bool(int(np.argmax(loc)) == int(np.argmax(bf))),
               per_template_rel_err_max=float(rel.max()), per_template_rel_err_median=float(np.median(rel)),
               d2_box_max_own=float(np.nanmax(own)), d2_box_max_change=float(np.nanmax(own) / bf.max() - 1),
               d2_n_affected=int(np.sum(np.abs(own - bf) > 0)),
               d2_per_template_change_median=float(np.nanmedian(own / bf - 1)),
               d2_per_template_change_absmax=float(np.nanmax(np.abs(own / bf - 1))),
               seconds_local=round(t_loc, 1), seconds_total=round(time.time() - t0, 1),
               peak_footprint_mb=L.peak_footprint_mb(), model="Claude Opus 5.5 (DeepStrain verification)")
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{det}_{gps}_{'i%d' % inj if inj is not None else 'noise'}_mc{mc_star:.3f}_L{l_max}_{dev}" + \
          (f"_sub{n_sub}" if n_sub else "")
    (OUT / f"box_{tag}.json").write_text(json.dumps(dict(out, env=L.stamp()), indent=1))
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    a = sys.argv
    if a[1] == "box":
        box(a[2], int(a[3]), None if a[4] == "-" else [int(v) for v in a[4].split(",")], None if a[5] == "-" else float(a[5]), int(a[6]),
            int(a[7]) if len(a) > 7 else None)
