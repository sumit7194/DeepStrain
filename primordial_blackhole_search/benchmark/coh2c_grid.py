"""Redesign (c): CONSTRUCTED stage-2 grids and a fitting-factor Monte Carlo.   [model: Fable 5.2]  UNVERIFIED

Why: the brief's N_tmpl is a covering ESTIMATE (box area x sqrt(det g) / hexagon), never a built grid. Cost and loss
both hang on it. This builds real grids for the frozen box (chirp mass +-p99 bias, the population's q range, minimal
match 0.97, f_low 50 Hz, L_max 128 / 384 s) and measures what they actually cover. Analytic only: no strain is read,
only the whitening filter of tuning segment 0 (H1), as coh2_cost.py used. CPU only.

Method.
  * Phase differences are EXACTLY linear in the lalsimulation coefficient differences dc (coh2c_local.coeffs), so the
    mismatch maximised over time and phase is, to second order, dc^T G dc with
        G = (1/2) cov_W(basis_i, basis_j | projected off {1, f}),   W(f) ~ f^(-7/3) / S(f) on [f_start, 1024] Hz.
    One matrix G per box gives a flat metric in a 10-dimensional coefficient space; the templates are a curved
    two-dimensional sheet in it. (The form of the phase metric is textbook; it is CHECKED here against real matches,
    not trusted.)
  * xi = dc . sqrt(G); principal plane of the box's sheet; hexagonal lattice in that plane with covering radius
    sqrt(1 - 0.97); lattice points inside the box's image or within one radius of its edge are kept; each is pulled
    back to (chirp mass, q) by Newton iteration (edge-layer points land on the box edge).
  * Monte Carlo: random box points; fitting factor = best match over the grid, each match the peak of the
    noise-free correlation (the same arithmetic as the local engine).

  python benchmark/coh2c_grid.py          -> results/coh2c/G_grids.json   (CPU, minutes, < 1 GB)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

B = Path(__file__).resolve().parent
sys.path.insert(0, str(B))
import coh2c_local as L  # noqa: E402

EPS = 0.03
R_COV = np.sqrt(EPS)
M = 8192                                   # coarse frequency grid: 1/8 Hz over 1024 Hz
F_GRID = 537.0 + np.fft.fftfreq(M, d=4.0 / L.FS)
N_MC = 400


def weights(f_lo):
    w_snip = np.load(L.STATE / "wsnip_H1_1249001472.npy")
    f_snip = np.arange(L.NS // 2 + 1) * (L.FS / L.NS)
    w2 = np.interp(F_GRID, f_snip, w_snip.astype(np.float64) ** 2)
    W = np.where((F_GRID >= f_lo) & (F_GRID <= L.F_HI), F_GRID ** (-7.0 / 3.0) * w2, 0.0)
    return W / W.sum()


def metric_root(W):
    """sqrt(G) as a [NK, NK] matrix Lm with |dc @ Lm|^2 = predicted mismatch."""
    band = W > 0
    Bm = L.basis(F_GRID[band])
    w = W[band]
    Q = np.stack([np.ones(band.sum()), F_GRID[band] / 100.0])
    A = (Bm * w) @ Q.T @ np.linalg.inv((Q * w) @ Q.T)
    Rm = Bm - A @ Q
    G = (Rm * w) @ Rm.T / 2
    lam, V = np.linalg.eigh(G)
    return V * np.sqrt(np.clip(lam, 0, None))


UP = 4                                      # lags at 4096 per second, the frozen design's full rate


def matches(c_r, c_set, W, bas):
    """Match (maximised over phase, and over time at FULL-RATE lags) between template c_r and each row of c_set.
    The baseband spectrum is zero-padded UP times before the inverse FFT. (Run 1 of this script read the peak at
    1,024 lags per second, which understates every match by 1-3%: its numbers are kept only as a stopped log.)"""
    zf = W * np.exp(1j * ((c_set - c_r) @ bas))
    pad = np.zeros((len(zf), UP * M), complex)
    pad[:, : M // 2] = zf[:, : M // 2]
    pad[:, -M // 2:] = zf[:, M // 2:]
    return np.abs(np.fft.ifft(pad, axis=1)).max(axis=1) * UP * M


def build(mc_star, half_width, l_max, shrink=1.0, rng=None):
    from matplotlib.path import Path as MplPath
    from scipy.interpolate import LinearNDInterpolator
    from scipy.spatial import cKDTree
    qr = L.q_range(mc_star)
    f_lo = L.f_start(mc_star, l_max)
    W = weights(f_lo)
    Lm = metric_root(W)
    c0 = L.coeffs(mc_star, 0.5 * (qr[0] + qr[1]))

    def cf(x, q):
        return L.coeffs(mc_star * np.exp(x), q)

    def xi(c):
        return (c - c0) @ Lm

    # the box's sheet, sampled; its principal plane
    xs, qs = np.linspace(-half_width, half_width, 81), np.linspace(qr[0], qr[1], 41)
    xq = np.array([(x, q) for x in xs for q in qs])
    X = np.array([xi(cf(x, q)) for x, q in xq])
    mean = X.mean(axis=0)
    _, sv, Vt = np.linalg.svd(X - mean, full_matrices=False)
    U = Vt[:2].T
    u_fine = (X - mean) @ U
    off_plane = np.sqrt(np.mean(np.sum(((X - mean) - u_fine @ U.T) ** 2, axis=1)))

    def uv(x, q):
        return (xi(cf(x, q)) - mean) @ U

    # the image of the box edge, densely sampled
    n_e = 3000
    edge_xq = np.concatenate([np.stack([np.linspace(-half_width, half_width, n_e), np.full(n_e, qr[0])], 1),
                              np.stack([np.full(n_e, half_width), np.linspace(qr[0], qr[1], n_e)], 1),
                              np.stack([np.linspace(half_width, -half_width, n_e), np.full(n_e, qr[1])], 1),
                              np.stack([np.full(n_e, -half_width), np.linspace(qr[1], qr[0], n_e)], 1)])
    edge_u = np.array([uv(x, q) for x, q in edge_xq])
    area = 0.5 * abs(np.dot(edge_u[:, 0], np.roll(edge_u[:, 1], -1)) - np.dot(edge_u[:, 1], np.roll(edge_u[:, 0], -1)))
    poly, etree = MplPath(edge_u), cKDTree(edge_u)
    # hexagonal lattice
    a = np.sqrt(3) * R_COV * shrink
    lo, hi = edge_u.min(axis=0) - 2 * a, edge_u.max(axis=0) + 2 * a
    js = np.arange(int((hi[1] - lo[1]) / (a * np.sqrt(3) / 2)) + 2)
    is_ = np.arange(int((hi[0] - lo[0]) / a) + 2)
    pts = np.array([(lo[0] + i * a + (j % 2) * a / 2, lo[1] + j * a * np.sqrt(3) / 2) for j in js for i in is_])
    inside = poly.contains_points(pts)
    d_edge, i_edge = etree.query(pts)
    keep = inside | (d_edge <= R_COV * shrink)
    pts, inside, i_edge = pts[keep], inside[keep], i_edge[keep]
    # pull back to (x, q)
    guess = LinearNDInterpolator(u_fine, xq)(pts)
    grid_xq = np.empty((len(pts), 2))
    for k, (p, ins) in enumerate(zip(pts, inside)):
        if not ins or np.isnan(guess[k, 0]):
            grid_xq[k] = edge_xq[i_edge[k]]
            continue
        x, q = guess[k]
        for _ in range(4):
            u0 = uv(x, q)
            hx, hq = 1e-6, 1e-4
            J = np.stack([(uv(x + hx, q) - u0) / hx, (uv(x, q + hq) - u0) / hq], axis=1)
            dx, dq = np.linalg.solve(J, p - u0)
            x, q = x + dx, q + dq
            x, q = float(np.clip(x, -half_width, half_width)), float(np.clip(q, qr[0], qr[1]))
        grid_xq[k] = (x, q)
    grid_c = np.array([cf(x, q) for x, q in grid_xq])
    grid_u = (np.array([xi(c) for c in grid_c]) - mean) @ U
    # Monte Carlo fitting factor
    bas = L.basis(np.clip(F_GRID, 10.0, None))
    tree = cKDTree(grid_u)
    ff, quad = [], []
    for _ in range(N_MC):
        x, q = rng.uniform(-half_width, half_width), rng.uniform(*qr)
        c_r = cf(x, q)
        u_r = (xi(c_r) - mean) @ U
        near = tree.query_ball_point(u_r, 5 * R_COV)
        if not near:
            near = [int(tree.query(u_r)[1])]
        m = matches(c_r, grid_c[near], W, bas)
        ff.append(float(m.max()))
        j = near[int(np.argmax(m))]
        quad.append((float(np.sum((xi(grid_c[j]) - xi(c_r)) ** 2)), float(1 - m.max())))
    ff = np.array(ff)
    quad = np.array(quad)
    self_match = float(matches(c0, c0[None], W, bas)[0])
    assert abs(self_match - 1) < 1e-9, self_match
    sel = quad[:, 1] > 0.003
    return dict(mc_star=mc_star, l_max=l_max, f_start=f_lo, q_range=list(qr), shrink=shrink, n_grid=int(len(pts)),
                n_interior=int(inside.sum()), n_edge_layer=int((~inside).sum()), area_xi=float(area),
                n_area_estimate=float(area / (1.5 * np.sqrt(3) * EPS)), off_plane_rms=float(off_plane),
                singular_values=[float(s) for s in sv[:4]],
                ff_min=float(ff.min()), ff_p01=float(np.quantile(ff, 0.01)), ff_p05=float(np.quantile(ff, 0.05)),
                ff_median=float(np.median(ff)), frac_below_097=float(np.mean(ff < 0.97)),
                quadratic_over_true_mismatch_median=float(np.median(quad[sel, 0] / quad[sel, 1])) if sel.any() else None)


def main():
    tc = json.loads((B / "results" / "coh2" / "T_cost.json").read_text())
    half_width = tc["bias"]["all"]["p99"]
    rng = np.random.default_rng(20261126)
    res = dict(eps=EPS, half_width=half_width, n_mc=N_MC, rows=[])
    t0 = time.time()
    for l_max in (128, 384):
        for mc in (0.18, 0.26, 0.35, 0.50, 0.70, 0.86):
            est = tc["n_tmpl"][f"L{l_max}_mc{mc}"]["n_tmpl"]
            for shrink in (1.0, 0.85, 0.7):
                r = build(mc, half_width, l_max, shrink, rng)
                r["brief_estimate"] = est
                r["constructed_over_estimate"] = r["n_grid"] / est
                res["rows"].append(r)
                print(f"L {l_max} Mc* {mc}: shrink {shrink} -> {r['n_grid']} templates ({r['n_interior']} interior + "
                      f"{r['n_edge_layer']} edge) vs estimate {est:.0f} ({r['constructed_over_estimate']:.2f}x; my area "
                      f"estimate {r['n_area_estimate']:.0f}) | FF min {r['ff_min']:.4f} p01 {r['ff_p01']:.4f} median "
                      f"{r['ff_median']:.4f} below 0.97: {100*r['frac_below_097']:.1f}% | off-plane rms "
                      f"{r['off_plane_rms']:.3f} | quad/true {r['quadratic_over_true_mismatch_median']} | "
                      f"{time.time()-t0:.0f} s", flush=True)
                if r["frac_below_097"] <= 0.01:
                    break
    res["seconds"] = round(time.time() - t0)
    L.write("G_grids.json", res)


if __name__ == "__main__":
    main()
