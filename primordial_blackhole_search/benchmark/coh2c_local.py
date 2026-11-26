"""Redesign (c) for PREREG_coherent2.md: the LOCAL HETERODYNED REFINER.   [model: Fable 5.2]  UNVERIFIED

Same stage-2 search as the frozen design (same box, grid, per-candidate maximum), computed a cheaper way.
Sealed design and predictions: reports/coh2c_sealed_predictions.md (commit 09a68e0), written before this file.

Identity. With double-whitened data Y(f) and TaylorF2 templates H_theta = A(f) exp(-i Psi_theta(f)) that share a
start frequency, the complex correlation is
    z(t; theta) = sum_f Y H_theta^* e^{2 pi i f t} = sum_f [Y H_0^*] e^{+i (Psi_theta - Psi_0)(f)} e^{2 pi i f t},
i.e. the reference correlation x0 = IFFT[Y H_0^*] passed through an all-pass filter whose group-delay spread is
dtau(f) = (1/2pi) d(Psi_theta - Psi_0)/df. Only x0 within (search half-width + max |dtau|) of the candidate matters,
so x0 is cut to a +-TW-s window, shifted to baseband, decimated to the band, and every box template then costs one
length-M multiply and inverse FFT (M ~ 8192) instead of a 2^21-sample one.

Phase model: the PN coefficients come from lalsimulation (SimInspiralTaylorF2AlignedPhasing), never from memory.
Psi(f) = sum_k (v_k + vlogv_k ln v) v^(k-5), v = (pi M f)^(1/3); gate E1 checks it against pycbc get_fd_waveform.

  python benchmark/coh2c_local.py e1          gate E1: phase model vs pycbc                       (CPU, seconds)
  python benchmark/coh2c_local.py e2          gate E2: local vs brute force on real tuning strain (MPS, minutes)
  python benchmark/coh2c_local.py time        P1: per-template time, paired with the old baseline (MPS, ~1 min)
Outputs: benchmark/results/coh2c/*.json, each with an environment stamp.
"""
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

B = Path(__file__).resolve().parent
sys.path.insert(0, str(B)); sys.path.insert(0, str(B.parent)); sys.path.insert(0, str(B.parent.parent / "tools"))
from pbh import config as C  # noqa: E402

FS = C.SAMPLE_RATE
NS = 2 ** 21                       # the frozen design's snippet length (512 s)
F_LO, F_HI = C.F_LOWER, C.F_HIGH
G_SUN = 4.925490947641267e-06      # G M_sun / c^3 in seconds (same constant as coh2_cost.py)
F0 = 100.0                         # frequency scale for the phase basis (keeps coefficients moderate)
NK = 10                            # 8 powers u^((k-5)/3), k = 0..7, plus ln(u) and ln(u) u^(1/3)
PRE = 448                          # seconds of snippet before the candidate (template <= 384 s must fit)
OUT = B / "results" / "coh2c"


def peak_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 2 ** 20


def peak_footprint_mb():
    """Lifetime-peak physical footprint (top's MEM), the number peaks are declared in: RSS hides MPS memory."""
    from footprint import footprint_mb
    return round(footprint_mb()[1])


def footprint_now_mb():
    from footprint import footprint_mb
    return round(footprint_mb()[0])


def pick_dev():
    """MPS by default; COH2C_DEV=cpu runs the same engine code on the CPU (exactness gates then need ~1 GB of
    footprint instead of ~2 GB: a 2^21-sample FFT on MPS alone holds ~1.1 GB of driver memory)."""
    import os
    import torch
    return os.environ.get("COH2C_DEV") or ("mps" if torch.backends.mps.is_available() else "cpu")


def stamp():
    import envstamp
    return envstamp.stamp(str(B.parent.parent))


def write(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    obj = dict(obj, model="Fable 5.2", status="UNVERIFIED", peak_rss_mb=peak_mb(),
               peak_footprint_mb=peak_footprint_mb(), env=stamp())
    (OUT / name).write_text(json.dumps(obj, indent=1))
    print(f"wrote results/coh2c/{name} | peak footprint {peak_footprint_mb()} MB (RSS {peak_mb()} MB)", flush=True)


# ------------------------------------------------------------------ masses and the phase model
def masses(mc, q):
    m_tot = mc * (1 + q) ** 1.2 / q ** 0.6
    return m_tot / (1 + q), q * m_tot / (1 + q)


def q_range(mc):
    qs = np.linspace(0.2, 1.0, 400)
    ok = [q for q in qs if C.M_MIN <= masses(mc, q)[1] and masses(mc, q)[0] <= C.M_MAX]
    return (min(ok), max(ok)) if ok else None


def f_at(mc, tau):
    """Leading-order GW frequency tau seconds before coalescence (as coh2_cost.py)."""
    return (5.0 / (256.0 * tau)) ** 0.375 * (G_SUN * mc) ** -0.625 / np.pi


def f_start(mc, l_max):
    return max(F_LO, f_at(mc, l_max))


_LAL = None
_K8 = np.arange(8)


def coeffs(mc, q):
    """Float64 phase coefficients: Psi(f) = sum_j coeffs[j] * basis_j(f / F0)."""
    global _LAL
    if _LAL is None:
        import lal
        import lalsimulation as ls
        _LAL = (ls.SimInspiralTaylorF2AlignedPhasing, lal.CreateDict())
    m1, m2 = masses(mc, q)
    pfa = _LAL[0](m1, m2, 0.0, 0.0, _LAL[1])
    v, vl = np.array(pfa.v[:8]), np.array(pfa.vlogv[:8])
    v0 = (np.pi * (m1 + m2) * G_SUN * F0) ** (1.0 / 3.0)
    c = np.empty(NK)
    c[:8] = (v + vl * np.log(v0)) * v0 ** (_K8 - 5.0)
    c[8] = vl[5] / 3.0
    c[9] = vl[6] * v0 / 3.0
    return c


def coeffs_reference(mc, q):
    """The first, slow implementation (the one gates E1 and E2 ran with); kept to check coeffs() against. It also
    asserts that lalsimulation has no log terms other than the two the basis carries."""
    import lal
    import lalsimulation as ls
    m1, m2 = masses(mc, q)
    pfa = ls.SimInspiralTaylorF2AlignedPhasing(m1, m2, 0.0, 0.0, lal.CreateDict())
    v0 = (np.pi * (m1 + m2) * G_SUN * F0) ** (1.0 / 3.0)
    c = np.zeros(NK)
    for k in range(8):
        c[k] = (pfa.v[k] + pfa.vlogv[k] * np.log(v0)) * v0 ** (k - 5)
        assert pfa.vlogvsq[k] == 0.0
    c[8] = pfa.vlogv[5] / 3.0
    c[9] = pfa.vlogv[6] * v0 / 3.0
    assert all(pfa.vlogv[k] == 0.0 for k in (0, 1, 2, 3, 4, 7))
    return c


def basis(f):
    """[NK, len(f)] float64 basis functions of u = f / F0."""
    u = np.asarray(f, dtype=np.float64) / F0
    lu = np.log(u)
    rows = [u ** ((k - 5) / 3.0) for k in range(8)] + [lu, lu * u ** (1.0 / 3.0)]
    return np.stack(rows)


def dbasis_df(f):
    u = np.asarray(f, dtype=np.float64) / F0
    lu = np.log(u)
    rows = [((k - 5) / 3.0) * u ** ((k - 5) / 3.0 - 1.0) for k in range(8)]
    rows += [1.0 / u, u ** (-2.0 / 3.0) * (1.0 + lu / 3.0)]
    return np.stack(rows) / F0


F_TAU = np.geomspace(F_LO, F_HI, 48)
_DB_TAU = None


def max_dtau(c_a, c_b, f_lo):
    """Largest |group-delay difference| (s) between two phase-coefficient vectors over [f_lo, F_HI]."""
    global _DB_TAU
    if _DB_TAU is None:
        _DB_TAU = dbasis_df(F_TAU)
    d = (np.asarray(c_a) - np.asarray(c_b)) @ _DB_TAU / (2 * np.pi)
    return float(np.abs(d[F_TAU >= f_lo * 0.999]).max())


def fd_template(mc, q, f_lo, n=NS):
    """pycbc TaylorF2 on the n-sample rfft grid, zero outside [f_lo, F_HI]; the brute-force yardstick's template."""
    from pycbc.waveform import get_fd_waveform
    m1, m2 = masses(mc, q)
    df = FS / n
    hp, _ = get_fd_waveform(approximant=C.APPROXIMANT, mass1=m1, mass2=m2, delta_f=df, f_lower=f_lo,
                            f_final=F_HI, distance=1.0)
    h = np.zeros(n // 2 + 1, complex)
    m = min(len(hp), len(h))
    h[:m] = hp.numpy()[:m]
    f = np.arange(n // 2 + 1) * df
    h[(f < f_lo) | (f > F_HI)] = 0
    return h


# ------------------------------------------------------------------ gate E1: phase model vs pycbc
def e1():
    rng = np.random.default_rng(20261126)
    f = np.arange(NS // 2 + 1) * (FS / NS)
    band = (f >= F_LO) & (f <= F_HI)
    bas = basis(f[band])
    rows, sign = [], None
    for i in range(20):
        mc = float(np.exp(rng.uniform(np.log(0.18), np.log(0.86))))
        qr = q_range(mc)
        q0, q1 = rng.uniform(*qr), rng.uniform(*qr)
        mc1 = mc * float(np.exp(rng.uniform(-0.0215, 0.0215)))
        h0, h1 = fd_template(mc, q0, F_LO)[band], fd_template(mc1, q1, F_LO)[band]
        dpsi = (coeffs(mc1, q1) - coeffs(mc, q0)) @ bas
        ratio = h1 * np.conj(h0)
        errs = {}
        for s in (+1, -1):                   # h ~ exp(s i Psi): find the convention, then hold it fixed
            r = np.angle(ratio * np.exp(-1j * s * dpsi))
            errs[s] = float(np.abs(r - np.median(r)).max())
        best = min(errs, key=errs.get)
        sign = best if sign is None else sign
        amp = np.abs(h1) / np.abs(h0)
        rows.append(dict(mc=mc, q=q0, mc1=mc1, q1=q1, sign=best, max_phase_err_rad=errs[sign],
                         amp_ratio_spread=float(amp.max() / amp.min() - 1), dpsi_span_rad=float(np.ptp(dpsi))))
        print(f"E1 {i:2d} mc {mc:.3f} q {q0:.2f}->{q1:.2f} dMc {mc1/mc-1:+.4f}: sign {best:+d} "
              f"max err {errs[sign]:.2e} rad over a {np.ptp(dpsi):.0f}-rad span", flush=True)
    worst = max(r["max_phase_err_rad"] for r in rows)
    def rel(a, b):                       # entries that are exactly zero in the reference must be exactly zero
        nz = b != 0
        return float(np.max(np.abs(a[nz] / b[nz] - 1))) if np.all(a[~nz] == 0) else np.inf
    same = max(rel(coeffs(r["mc"], r["q"]), coeffs_reference(r["mc"], r["q"])) for r in rows)
    t0 = time.perf_counter()
    for i in range(5000):
        coeffs(0.3 + 1e-5 * i, 0.5)
    us = (time.perf_counter() - t0) / 5000 * 1e6
    print(f"coeffs() vs the first implementation: largest relative difference {same:.1e}; {us:.1f} us per call", flush=True)
    ok = worst <= 1e-3 and len({r["sign"] for r in rows}) == 1 and same <= 1e-13
    write("E1_phase_model.json", dict(gate="E1", bar_rad=1e-3, worst_rad=worst, sign=sign, passed=bool(ok),
                                      coeffs_fast_vs_reference_max_rel=same, coeffs_us_per_call=us, rows=rows))
    print(f"E1 {'PASS' if ok else 'FAIL'}: worst phase error {worst:.2e} rad, h ~ exp({sign:+d} i Psi)", flush=True)


# ------------------------------------------------------------------ the local engine (GPU)
TW, DEC, TAPER_S, SEARCH_S = 4.0, 4, 0.5, 1.0
F_SHIFT = {4: 537.0, 2: 1024.0, 1: 0.0}            # baseband shift per decimation (band 50-1024 Hz stays unaliased)


def dtau_max(tw):
    """Largest allowed group-delay difference between a template and its reference: what is left of the window after
    the search half-width, the taper, and a margin (1 s at tw = 4, the sealed 1.5-s rule; 0.15 s at tw = 2)."""
    return tw - SEARCH_S - TAPER_S - (1.0 if tw >= 4 else 0.15)


class Engine:
    """Cuts a reference correlation to +-tw s around a candidate, shifts it to baseband, decimates it by dec, and
    evaluates any template whose phase differs from the reference by sum_j dcoef[j] basis_j(f)."""

    def __init__(self, dev, tw=TW, dec=DEC):
        import torch
        self.t, self.dev, self.tw, self.dec, self.f_c = torch, dev, tw, dec, F_SHIFT[dec]
        self.nw = int(round(2 * tw * FS))
        self.m = self.nw // dec
        n = np.arange(self.nw)
        ramp = int(TAPER_S * FS)
        win = np.ones(self.nw)
        win[:ramp] = 0.5 - 0.5 * np.cos(np.pi * n[:ramp] / ramp)
        win[-ramp:] = win[:ramp][::-1]
        down = win * np.exp(-2j * np.pi * self.f_c * (n - self.nw // 2) / FS)
        self.down = torch.tensor(down[::dec].astype(np.complex64), device=dev)
        self.f_abs = self.f_c + np.fft.fftfreq(self.m, d=dec / FS)
        # x0 is analytic and band-limited to [F_LO, F_HI], so bins outside the band carry (almost) nothing; the basis
        # is evaluated at a clipped frequency there only to stay finite
        self.basis = torch.tensor(basis(np.clip(self.f_abs, 10.0, None)).astype(np.float32), device=dev)
        self.half = int(round(SEARCH_S * FS / dec))            # +-1 s in decimated samples
        self.lo, self.hi = self.m // 2 - self.half, self.m // 2 + self.half + 1
        t_rel = (np.arange(self.lo, self.hi) - self.m // 2) * dec / FS
        self.up = np.exp(2j * np.pi * self.f_c * t_rel)        # undoes the baseband shift (only for comparisons)

    def window(self, x0, i_c):
        """x0: complex64 reference correlation on the device (full rate); i_c: candidate sample. -> spectrum [m]."""
        sl = x0[i_c - self.nw // 2: i_c + self.nw // 2: self.dec]
        return self.t.fft.fft(sl * self.down)

    def z(self, ytil, dcoef, sign):
        """dcoef: [B, NK] float32 on the device (template minus reference). Returns complex z [B, 2*half+1] in
        baseband (unnormalised: divide by the reference's sigma)."""
        t = self.t
        ph = (dcoef @ self.basis) * float(-sign)
        zz = t.fft.ifft(ytil.unsqueeze(0) * t.complex(t.cos(ph), t.sin(ph)), dim=1)
        return zz[:, self.lo:self.hi]


def analytic_corr(t, y_snip, h0_conj):
    """2 * ifft of the one-sided product: c + i c_q, indexed by merger sample."""
    n = 2 * (y_snip.shape[-1] - 1)
    return 2 * t.fft.ifft(t.nn.functional.pad(y_snip * h0_conj, (0, n // 2 - 1)))


# ------------------------------------------------------------------ reference lattice (the declared rule)
def reference_lattice(mc_star, half_width, l_max, bound):
    """Smallest (n_mc x n_q) lattice of references such that every probe template in the box has a reference with
    max |dtau| <= bound over the reference's band. Returns the list of (mc, q, f_start, coeffs) and the worst dtau."""
    qr = q_range(mc_star)
    xs = np.linspace(-half_width, half_width, 9)
    qs = np.linspace(qr[0], qr[1], 9)
    probes = [(mc_star * np.exp(x), q, coeffs(mc_star * np.exp(x), q)) for x in xs for q in qs]
    best = None
    for n_q in (1, 2, 3, 4, 6, 8):
        for n_mc in range(1, 257):
            if best is not None and n_q * n_mc >= best[0]:
                break
            rx = (np.arange(n_mc) + 0.5) / n_mc * 2 * half_width - half_width
            rq = qr[0] + (np.arange(n_q) + 0.5) / n_q * (qr[1] - qr[0])
            refs = [(mc_star * np.exp(x), q) for x in rx for q in rq]
            rc = [(mc, q, f_start(mc, l_max), coeffs(mc, q)) for mc, q in refs]
            worst = max(min(max_dtau(pc, r[3], r[2]) for r in rc) for _, _, pc in probes)
            if worst <= bound:
                best = (n_q * n_mc, rc, worst)
                break
    assert best is not None, "no lattice meets the rule"
    return best[1], best[2]


def assign(c, refs):
    d = [max_dtau(c, r[3], r[2]) for r in refs]
    j = int(np.argmin(d))
    return j, d[j]


# ------------------------------------------------------------------ real tuning strain, double-whitened
STATE = B / "state" / "coh2c"


def _table():
    import csv
    import injections as I
    with open(I.TABLE) as f:
        return {int(r["id"]): {k: (int(v) if k in ("id", "realization", "segment") else float(v))
                               for k, v in r.items()} for r in csv.DictReader(f)}


def ww_tag(det, gps, inj_ids=()):
    return f"{det}_{gps}" + "".join(f"_i{i}" for i in inj_ids)


def prep(det, gps, inj_ids=()):
    """Own process (no torch): gated, double-whitened series as float32, the whitening filter on the snippet grid, and
    the injected rows. Injections are added by linearity exactly as realbg_cascade.fg_segment does (gated signal
    spectrum added to the gated noise spectrum; the PSD is the gated noise's)."""
    import injections as I
    import realbg_cascade as R
    from lowmem_psd import estimate_psd
    STATE.mkdir(parents=True, exist_ok=True)
    tag = ww_tag(det, gps, inj_ids)
    x = np.load(B / "data" / f"{det}_{gps}.npy").astype(np.float64)
    g = R.gate_window(det, gps)[0].astype(np.float32)
    x *= g
    w = R.whiten_filter(estimate_psd(x))
    X = np.fft.rfft(x)
    del x
    rows = {}
    table = _table() if inj_ids else {}
    for i in inj_ids:
        row = table[i]
        s, s0, _, merger = R._injection_template(row, det, gps)
        sf = I.add_signal(np.zeros(R.N), gps, s, s0)
        sf *= g
        X += np.fft.rfft(sf)
        del sf, s
        rows[i] = dict(merger=int(merger), mc=float(I.chirp_mass(row["mass1"], row["mass2"])),
                       q=float(min(row["mass1"], row["mass2"]) / max(row["mass1"], row["mass2"])))
    del g
    X *= w; X *= w
    y = np.fft.irfft(X, n=R.N).astype(np.float32)
    del X
    np.save(STATE / f"ww_{tag}.npy", y)
    np.save(STATE / f"wsnip_{tag}.npy", w[::8].copy())          # segment grid (1/4096 Hz) -> snippet grid (1/512 Hz)
    (STATE / f"rows_{tag}.json").write_text(json.dumps(rows))
    print(f"prep {tag}: rms of the double-whitened series {np.sqrt(np.mean(y[::64].astype(np.float64) ** 2)):.3e} "
          f"| peak RSS {peak_mb()} MB", flush=True)


def load_ww(det, gps, inj_ids=()):
    import subprocess
    tag = ww_tag(det, gps, inj_ids)
    if not (STATE / f"rows_{tag}.json").exists():
        subprocess.run([sys.executable, "-u", __file__, "prep", det, str(gps), *map(str, inj_ids)], check=True)
    rows = {int(k): v for k, v in json.loads((STATE / f"rows_{tag}.json").read_text()).items()}
    return np.load(STATE / f"ww_{tag}.npy", mmap_mode="r"), np.load(STATE / f"wsnip_{tag}.npy"), rows


def snippet_spectrum(y, i_c):
    lo = i_c - PRE * FS
    assert lo >= 0 and lo + NS <= len(y), "candidate too close to the segment edge for a 512-s snippet"
    return np.fft.rfft(np.asarray(y[lo: lo + NS], dtype=np.float64))


# ------------------------------------------------------------------ gate E2: local vs brute force
E2_CONFIGS = ((4.0, 4), (4.0, 2), (4.0, 1), (2.0, 1))     # (window half-width s, decimation); the first is the sealed primary
E2_BARS = dict(overlap_median=0.9999, overlap_worst=0.999, peak_err_median=1e-3, peak_err_worst=5e-3)


def e2():
    """Local engine vs the frozen brute-force definition, on real tuning strain. Cases: a real stage-1 noise candidate
    with its own Y1 box; three FORCED boxes on that noise (lowest / low / highest chirp mass: the longest templates
    and the extremes of the lattice); three caught injections (mid mass; the lowest chirp mass caught; a q = 0.25
    one), each with the box centred on the nearest Y1 bank template."""
    import torch
    import y1_bank as Y1
    dev = pick_dev()
    sign = json.loads((OUT / "E1_phase_model.json").read_text())["sign"]
    half_width = json.loads((B / "results" / "coh2" / "T1_mc_bias.json").read_text())["all"]["p99"]
    segs = json.loads((B / "manifest.json").read_text())["test"]
    st = json.loads((B / "state" / "realbg" / "caught.json").read_text())
    rng = np.random.default_rng(20261126)
    engines = {c: Engine(dev, *c) for c in E2_CONFIGS}
    g0 = segs[0]
    bins = np.load(B / "results" / "y1_bins" / f"noise_{g0}.npz")
    cand = [b for b in st["candidates"]["0_0"] if (PRE + 1) * FS < b * 102 < C.SEGMENT_LEN * FS - 65 * FS][0] * 102
    mc_y1 = float(Y1.MCS[bins["H1_tid"][min(cand // Y1.BIN, len(bins["H1_tid"]) - 1)]])
    plan = [("noise, Y1 box", g0, (), None, mc_y1), ("noise, forced box", g0, (), None, 0.18),
            ("noise, forced box", g0, (), None, 0.26), ("noise, forced box", g0, (), None, 0.86),
            ("signal", g0, (676,), 676, None), ("signal, lowest Mc caught", 1253212160, (471, 473), 471, None),
            ("signal, q = 0.25", 1253212160, (471, 473), 473, None)]
    res = dict(gate="E2", det="H1", device=dev, half_width=half_width, bars=E2_BARS, configs=[list(c) for c in E2_CONFIGS],
               cases=[])
    t0 = time.time()
    for label, gps, inj_ids, inj, mc_star in plan:
        y, w_snip, rows = load_ww("H1", gps, inj_ids)
        if inj is None:
            i_c, truth = cand, None
        else:
            i_c, truth = rows[inj]["merger"] + int(rng.integers(-800, 800)), (rows[inj]["mc"], rows[inj]["q"])
            mc_star = float(Y1.MCS[np.argmin(np.abs(np.log(Y1.MCS / truth[0])))])
        ysn = snippet_spectrum(y, i_c)
        kappa = 1.0 / np.median(np.abs(ysn[NS // 2048: NS // 8]))   # the device copy is scaled to order one (float32)
        ysn_t = torch.tensor((ysn * kappa).astype(np.complex64), device=dev)
        i_s = PRE * FS
        qr = q_range(mc_star)
        for l_max in (128, 384):
            for ci, cfg in enumerate(E2_CONFIGS):
                eng = engines[cfg]
                if ci > 0 and l_max == 128 and f_start(mc_star, 128) <= F_LO:
                    continue                                        # identical to the L_max = 384 box: skip repeats
                n_t = 60 if ci == 0 else 16
                refs, lat_worst = reference_lattice(mc_star, half_width, l_max, dtau_max(cfg[0]))
                x0s, sig0 = [], []
                for mc_r, q_r, f_lo, _ in refs:
                    h0 = fd_template(mc_r, q_r, f_lo)
                    sig0.append(np.sqrt(2 * np.sum(np.abs(h0 * w_snip) ** 2) / NS))
                    x0 = analytic_corr(torch, ysn_t, torch.tensor(np.conj(h0).astype(np.complex64), device=dev))
                    x0s.append(eng.window(x0, i_s))
                    del x0
                out, P = [], np.zeros(NS, complex)
                for k in range(n_t):
                    if truth is not None and k == 0:
                        mc, q = truth
                    else:
                        mc = mc_star * float(np.exp(rng.uniform(-half_width, half_width)))
                        q = float(rng.uniform(*qr))
                    c = coeffs(mc, q)
                    j, dt = assign(c, refs)
                    f_lo = refs[j][2]
                    dco = torch.tensor((c - refs[j][3]).astype(np.float32), device=dev).unsqueeze(0)
                    z_loc = eng.z(x0s[j], dco, sign)[0].cpu().numpy().astype(np.complex128) * eng.up / (sig0[j] * kappa)
                    h = fd_template(mc, q, f_lo)                # brute force, starting at the reference's frequency
                    sig = np.sqrt(2 * np.sum(np.abs(h * w_snip) ** 2) / NS)
                    P[: NS // 2 + 1] = ysn * np.conj(h)
                    zf = 2 * np.fft.ifft(P) / sig
                    z_bf = zf[i_s - FS: i_s + FS + 1]
                    far2 = float(np.mean(np.abs(zf[i_s - 300 * FS: i_s - 20 * FS: 97]) ** 2))    # sanity: ~2 in noise
                    shared = z_bf[::cfg[1]]
                    ov = abs(np.vdot(shared, z_loc)) / np.sqrt(np.vdot(shared, shared).real * np.vdot(z_loc, z_loc).real)
                    p_loc, p_sh, p_full = ((np.abs(a) ** 2).max() for a in (z_loc, shared, z_bf))
                    own = None                                      # the frozen text: the template's OWN start frequency
                    f_own = f_start(mc, l_max)
                    if abs(f_own - f_lo) > 1e-9:
                        h2 = fd_template(mc, q, f_own)
                        s2 = np.sqrt(2 * np.sum(np.abs(h2 * w_snip) ** 2) / NS)
                        P[: NS // 2 + 1] = ysn * np.conj(h2)
                        own = float((np.abs(2 * np.fft.ifft(P)[i_s - FS: i_s + FS + 1] / s2) ** 2).max() / p_full - 1)
                    out.append(dict(mc=mc, q=q, ref=j, dtau=dt, overlap=float(ov), peak_ratio=float(p_loc / p_sh),
                                    peak_full=float(p_full), loss_dec=float(1 - p_sh / p_full),
                                    loss_1024=float(1 - (np.abs(z_bf[::4]) ** 2).max() / p_full),
                                    loss_2048=float(1 - (np.abs(z_bf[::2]) ** 2).max() / p_full),
                                    far_mean_z2=far2, own_fstart_change=own))
                ovs = np.array([o["overlap"] for o in out]); pr = np.abs(np.array([o["peak_ratio"] for o in out]) - 1)
                owns = [abs(o["own_fstart_change"]) for o in out if o["own_fstart_change"] is not None]
                summ = dict(case=label, segment=gps, injection=inj, l_max=l_max, tw=cfg[0], dec=cfg[1], m=eng.m,
                            mc_star=mc_star, f_start_centre=f_start(mc_star, l_max), n_refs=len(refs),
                            lattice_worst_dtau=lat_worst, dtau_rule=dtau_max(cfg[0]), n=len(out),
                            overlap_median=float(np.median(ovs)), overlap_worst=float(ovs.min()),
                            peak_err_median=float(np.median(pr)), peak_err_worst=float(pr.max()),
                            loss_1024_mean=float(np.mean([o["loss_1024"] for o in out])),
                            loss_1024_max=float(np.max([o["loss_1024"] for o in out])),
                            loss_2048_mean=float(np.mean([o["loss_2048"] for o in out])),
                            loss_2048_max=float(np.max([o["loss_2048"] for o in out])),
                            far_mean_z2=float(np.median([o["far_mean_z2"] for o in out])),
                            peak_full_max=float(max(o["peak_full"] for o in out)),
                            true_template_peak=out[0]["peak_full"] if truth is not None else None,
                            own_fstart_change_max=max(owns, default=None),
                            own_fstart_change_median=float(np.median(owns)) if owns else None, templates=out)
                summ["passed"] = bool(summ["overlap_median"] >= E2_BARS["overlap_median"] and
                                      summ["overlap_worst"] >= E2_BARS["overlap_worst"] and
                                      summ["peak_err_median"] <= E2_BARS["peak_err_median"] and
                                      summ["peak_err_worst"] <= E2_BARS["peak_err_worst"])
                res["cases"].append(summ)
                print(f"E2 {label} Mc* {mc_star:.3f} L {l_max} tw {cfg[0]:.0f} dec {cfg[1]} M {eng.m}: {len(refs)} refs "
                      f"(dtau {lat_worst:.2f}/{dtau_max(cfg[0]):.2f}) | 1-overlap median {1-np.median(ovs):.1e} worst "
                      f"{1-ovs.min():.1e} | peak err worst {pr.max():.1e} | {'PASS' if summ['passed'] else 'FAIL'} | "
                      f"far <|z|^2> {summ['far_mean_z2']:.2f} | max {summ['peak_full_max']:.0f} | own-f "
                      f"{summ['own_fstart_change_max']} | {time.time()-t0:.0f} s | RSS {peak_mb()} MB", flush=True)
                del x0s
        del ysn_t
    prim = [k for k in res["cases"] if (k["tw"], k["dec"]) == E2_CONFIGS[0]]
    res["passed_primary"] = bool(all(k["passed"] for k in prim))
    res["passed_by_config"] = {f"tw{c[0]:.0f}_dec{c[1]}": bool(all(k["passed"] for k in res["cases"]
                                                                   if (k["tw"], k["dec"]) == c)) for c in E2_CONFIGS}
    res["seconds"] = round(time.time() - t0)
    write("E2_local_vs_brute.json", res)
    print(f"E2 primary {'PASS' if res['passed_primary'] else 'FAIL'}; by config {res['passed_by_config']}", flush=True)


# ------------------------------------------------------------------ P1: per-template time, paired with the baseline
CONFIGS = ((2.0, 4), (4.0, 4), (4.0, 2), (2.0, 1), (4.0, 1))   # (half-width s, decimation) -> M = 4096 ... 32768


def _sync(t, dev):
    if dev == "mps":
        t.mps.synchronize()


def _warm_median(fn, t, dev, reps=7):
    ts = []
    for _ in range(reps):
        _sync(t, dev)
        t0 = time.perf_counter()
        fn()
        _sync(t, dev)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts[2:])), ts


def chi2_at_peak(eng, ytil, dcoef, sign, lag, edges):
    """Allen chi^2 pieces for a few templates at one lag each: z_j = sum over coarse bins in band j. Returns [n, p]."""
    t = eng.t
    ph = (dcoef @ eng.basis) * float(-sign)
    zt = ytil.unsqueeze(0) * t.complex(t.cos(ph), t.sin(ph))                       # [n, m]
    k = t.arange(eng.m, device=eng.dev, dtype=t.float32)
    rot = 2 * np.pi * k.unsqueeze(0) * lag.unsqueeze(1).to(t.float32) / eng.m
    zt = zt * t.complex(t.cos(rot), t.sin(rot)) / eng.m
    return t.stack([(zt * e.unsqueeze(0)).sum(dim=1) for e in edges], dim=1)


# ------------------------------------------------------------------ gate E3: Allen chi^2 pieces at the peak
def chi2_fine(t, p0, dcoef, sign, n0, kedges, basis_fine, k_idx):
    """EXACT route: band sums at the snippet's full frequency resolution, heterodyned.
    p0 = Y conj(H0) on the band bins [nb]; dcoef [n, NK]; n0 [n] int64 snippet sample of each template's peak;
    kedges = p (start, stop) index pairs into the band bins. Returns z_j, [n, p] (unnormalised, as Engine.z)."""
    ph = (dcoef @ basis_fine) * float(-sign)
    lag = ((k_idx.unsqueeze(0) * n0.unsqueeze(1)) % NS).to(t.float32) * (2 * np.pi / NS)   # exact integer phase
    ph = ph + lag
    zt = p0.unsqueeze(0) * t.complex(t.cos(ph), t.sin(ph))
    return t.stack([zt[:, a:b].sum(dim=1) for a, b in kedges], dim=1) * (2.0 / NS)


def e3():
    """chi^2 pieces z_j (p equal-SNR frequency bands) at each template's peak lag: the exact fine-resolution route and
    the cheap coarse-mask route, both against brute force (pycbc template, float64, the full 2^21-sample spectrum).
    Bars, written before the first run: fine route |z_j - brute| <= 1e-3 * max(|z|, 1) in every band and chi^2 within
    1e-3 (relative, where chi^2 >= 1). The coarse route is EXPECTED to miss these (band edges are blurred by the
    window's ~1/8-Hz resolution, my estimate ~0.05 per band); it is measured to say by how much."""
    import torch
    import y1_bank as Y1
    dev = pick_dev()
    sign = json.loads((OUT / "E1_phase_model.json").read_text())["sign"]
    half_width = json.loads((B / "results" / "coh2" / "T1_mc_bias.json").read_text())["all"]["p99"]
    segs = json.loads((B / "manifest.json").read_text())["test"]
    st = json.loads((B / "state" / "realbg" / "caught.json").read_text())
    rng = np.random.default_rng(20261127)
    eng = Engine(dev, 4.0, 1)
    f_fine = np.arange(NS // 2 + 1) * (FS / NS)
    kb = np.where((f_fine >= F_LO - 12) & (f_fine <= F_HI))[0]            # band bins (covers every start frequency)
    basis_fine = torch.tensor(basis(f_fine[kb]).astype(np.float32), device=dev)
    k_idx = torch.tensor(kb, device=dev, dtype=torch.int64)
    g0 = segs[0]
    bins = np.load(B / "results" / "y1_bins" / f"noise_{g0}.npz")
    cand = [b for b in st["candidates"]["0_0"] if (PRE + 1) * FS < b * 102 < C.SEGMENT_LEN * FS - 65 * FS][0] * 102
    mc_y1 = float(Y1.MCS[bins["H1_tid"][min(cand // Y1.BIN, len(bins["H1_tid"]) - 1)]])
    plan = [("noise, Y1 box", g0, (), None, mc_y1, 384), ("signal", g0, (676,), 676, None, 384),
            ("signal, q = 0.25", 1253212160, (471, 473), 473, None, 384),
            ("signal, lowest Mc caught", 1253212160, (471, 473), 471, None, 384),
            ("signal, lowest Mc caught", 1253212160, (471, 473), 471, None, 128)]
    res = dict(gate="E3", device=dev, bars=dict(fine_band_err=1e-3, fine_chi2_rel=1e-3), cases=[])
    t0 = time.time()
    i_s = PRE * FS
    t_fine = None
    for label, gps, inj_ids, inj, mc_star, l_max in plan:
        y, w_snip, rows = load_ww("H1", gps, inj_ids)
        if inj is None:
            i_c, truth = cand, None
        else:
            i_c, truth = rows[inj]["merger"] + int(rng.integers(-800, 800)), (rows[inj]["mc"], rows[inj]["q"])
            mc_star = float(Y1.MCS[np.argmin(np.abs(np.log(Y1.MCS / truth[0])))])
        ysn = snippet_spectrum(y, i_c)
        kappa = 1.0 / np.median(np.abs(ysn[NS // 2048: NS // 8]))
        ysn_t = torch.tensor((ysn * kappa).astype(np.complex64), device=dev)
        qr = q_range(mc_star)
        refs, _ = reference_lattice(mc_star, half_width, l_max, dtau_max(eng.tw))
        tpl = []
        for k in range(8):
            mc, q = truth if (truth is not None and k == 0) else (
                mc_star * float(np.exp(rng.uniform(-half_width, half_width))), float(rng.uniform(*qr)))
            c = coeffs(mc, q)
            tpl.append((mc, q, c, assign(c, refs)[0]))
        rows_out = []
        for j, (mc_r, q_r, f_lo, c_r) in enumerate(refs):
            mine = [x for x in tpl if x[3] == j]
            if not mine:
                continue
            h0 = fd_template(mc_r, q_r, f_lo)
            pw = np.abs(h0 * w_snip) ** 2
            sig0 = np.sqrt(2 * pw.sum() / NS)
            h0c = torch.tensor(np.conj(h0).astype(np.complex64), device=dev)
            ytil = eng.window(analytic_corr(torch, ysn_t, h0c), i_s)
            p0 = (ysn_t * h0c)[k_idx]
            cum = np.cumsum(pw[kb]) / pw[kb].sum()
            P = np.zeros(NS, complex)
            for mc, q, c, _ in mine:
                h = fd_template(mc, q, f_lo)
                sig = np.sqrt(2 * np.sum(np.abs(h * w_snip) ** 2) / NS)
                prod = ysn * np.conj(h)
                P[: NS // 2 + 1] = prod
                zf = 2 * np.fft.ifft(P) / sig
                n0 = int(i_s - FS + np.argmax(np.abs(zf[i_s - FS: i_s + FS + 1]) ** 2))
                z_tot = zf[n0]
                dco = torch.tensor((c - c_r).astype(np.float32), device=dev).unsqueeze(0)
                for p_bins in (4, 8, 16):
                    ke = np.searchsorted(cum, np.arange(1, p_bins) / p_bins)
                    ke = np.concatenate([[0], ke, [len(kb)]])
                    kedges = [(int(a), int(b)) for a, b in zip(ke[:-1], ke[1:])]
                    rot = np.exp(2j * np.pi * ((kb.astype(np.int64) * n0) % NS) / NS)
                    zj_bf = np.array([2 * np.sum(prod[kb[a:b]] * rot[a:b]) / NS for a, b in kedges]) / sig
                    n0_t = torch.tensor([n0], device=dev, dtype=torch.int64)
                    zj_f = chi2_fine(torch, p0, dco, sign, n0_t, kedges, basis_fine, k_idx)[0].cpu().numpy() \
                        .astype(np.complex128) / (sig0 * kappa)
                    # coarse route: masks on the window's own frequency bins, the lag as a window sample
                    f_edge = [f_fine[kb[a]] for a, _ in kedges] + [f_fine[kb[-1]] + 1.0]
                    masks = [torch.tensor(((eng.f_abs >= lo) & (eng.f_abs < hi)).astype(np.float32), device=dev)
                             for lo, hi in zip(f_edge[:-1], f_edge[1:])]
                    lag_w = torch.tensor([n0 - (i_s - eng.nw // 2)], device=dev)
                    zj_c = chi2_at_peak(eng, ytil, dco, sign, lag_w, masks)[0].cpu().numpy().astype(np.complex128) \
                        / (sig0 * kappa)

                    def chi2(zj):
                        return float(p_bins * np.sum(np.abs(zj - zj.sum() / p_bins) ** 2))
                    scale = max(abs(z_tot), 1.0)
                    rows_out.append(dict(mc=mc, q=q, p=p_bins, z_abs=float(abs(z_tot)),
                                         closure_bf=float(abs(zj_bf.sum() - z_tot) / scale),
                                         fine_band_err=float(np.abs(zj_f - zj_bf).max() / scale),
                                         coarse_band_err=float(np.abs(zj_c - zj_bf).max() / scale),
                                         coarse_band_err_abs=float(np.abs(zj_c - zj_bf).max()),
                                         chi2_bf=chi2(zj_bf), chi2_fine=chi2(zj_f), chi2_coarse=chi2(zj_c)))
                if t_fine is None:                                   # P3 input: 5 templates, p = 16, warm median
                    dco5 = dco.repeat(5, 1)
                    n05 = torch.tensor([n0] * 5, device=dev, dtype=torch.int64)
                    t_fine = _warm_median(lambda: chi2_fine(torch, p0, dco5, sign, n05, kedges, basis_fine, k_idx),
                                          torch, dev)[0]
            del ytil, p0, h0c
        fe = max(r["fine_band_err"] for r in rows_out)
        fc = max(abs(r["chi2_fine"] / r["chi2_bf"] - 1) for r in rows_out if r["chi2_bf"] >= 1)
        ce = max(r["coarse_band_err_abs"] for r in rows_out)
        cc = max(abs(r["chi2_coarse"] / r["chi2_bf"] - 1) for r in rows_out if r["chi2_bf"] >= 1)
        summ = dict(case=label, segment=gps, injection=inj, l_max=l_max, mc_star=mc_star, n_refs=len(refs),
                    fine_band_err_max=fe, fine_chi2_rel_max=fc, coarse_band_err_abs_max=ce, coarse_chi2_rel_max=cc,
                    closure_bf_max=max(r["closure_bf"] for r in rows_out),
                    passed=bool(fe <= 1e-3 and fc <= 1e-3), rows=rows_out)
        res["cases"].append(summ)
        print(f"E3 {label} Mc* {mc_star:.3f} L {l_max}: FINE band err {fe:.1e}, chi2 rel {fc:.1e} -> "
              f"{'PASS' if summ['passed'] else 'FAIL'} | coarse band err {ce:.3f} (abs), chi2 rel {cc:.2f} | closure "
              f"{summ['closure_bf_max']:.1e} | {time.time()-t0:.0f} s | footprint {peak_footprint_mb()} MB", flush=True)
        del ysn_t
    res["passed"] = bool(all(c["passed"] for c in res["cases"]))
    res["chi2_fine_top5_p16_s"] = t_fine
    res["seconds"] = round(time.time() - t0)
    write("E3_chi2.json", res)
    print(f"E3 {'PASS' if res['passed'] else 'FAIL'} (fine route); fine chi^2 for 5 templates, p = 16: {t_fine*1e3:.2f} ms",
          flush=True)


SCALE = np.array([500, 50, 300, 200, 100, 50, 30, 10, 20, 10], dtype=np.float32)   # realistic |dcoef| sizes
MEM_BAR_MB = 1900                                  # a child is not launched if its predicted footprint is above this


def _child_report(d):
    import torch
    d.update(peak_footprint_mb=peak_footprint_mb(), peak_rss_mb=peak_mb(),
             mps_driver_mb=round(torch.mps.driver_allocated_memory() / 2 ** 20) if torch.backends.mps.is_available()
             else None, load_avg=list(__import__("os").getloadavg()))
    print("RESULT " + json.dumps(d), flush=True)


def tbase(n_batch):
    """Child: the OLD baseline, coh2_cost.per_template_seconds, at a given batch (the stored 3.36 ms used 64, which
    needs ~5.8 GB of unified memory and cannot be repeated under the 2-GB rule)."""
    import coh2_cost
    s, dev = coh2_cost.per_template_seconds(n_batch=n_batch)
    _child_report(dict(kind="baseline", batch=n_batch, samples_per_batch=n_batch * NS, per_template_s=s, device=dev))


def tlocal(tw, dec, total):
    """Child: the local engine at one window setting and one batch (total = samples per batch)."""
    import torch
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    rng = np.random.default_rng(0)
    eng = Engine(dev, tw, dec)
    nb = total // eng.m
    ytil = torch.tensor((rng.standard_normal(eng.m) + 1j * rng.standard_normal(eng.m)).astype(np.complex64), device=dev)
    dco = torch.tensor(rng.uniform(-1, 1, (nb, NK)).astype(np.float32) * SCALE, device=dev)

    def step():
        z = eng.z(ytil, dco, -1)
        return (z.real ** 2 + z.imag ** 2).max(dim=1)
    med, ts = _warm_median(step, torch, dev)
    _child_report(dict(kind="local", tw=tw, dec=dec, m=eng.m, batch=nb, samples_per_batch=nb * eng.m,
                       per_template_s=med / nb, all_reps_s=[x / nb for x in ts], device=dev))


def tovh():
    """Child: the costs the old model left out."""
    import torch
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    rng = np.random.default_rng(0)
    eng = Engine(dev, 4.0, 1)
    ysn = torch.tensor((rng.standard_normal(NS // 2 + 1) + 1j * rng.standard_normal(NS // 2 + 1)).astype(np.complex64),
                       device=dev)
    h0 = torch.tensor((rng.standard_normal(NS // 2 + 1) + 1j * rng.standard_normal(NS // 2 + 1)).astype(np.complex64),
                      device=dev)
    ov = {}
    ov["reference_corr_and_window_s"] = _warm_median(lambda: eng.window(analytic_corr(torch, ysn, h0), PRE * FS),
                                                     torch, dev)[0]
    snip = rng.standard_normal(NS)
    snip_t = torch.tensor(snip.astype(np.float32), device=dev)
    ov["snippet_rfft_mps_s"] = _warm_median(lambda: torch.fft.rfft(snip_t), torch, dev)[0]
    ww = sorted(STATE.glob("ww_*.npy"))
    if ww:                                   # a real cached series: memory-mapped read of one snippet + transfer + rfft
        mm = np.load(ww[0], mmap_mode="r")
        offs = iter(rng.integers(0, len(mm) - NS, 16).tolist())
        ov["snippet_read_transfer_rfft_s"] = _warm_median(
            lambda: torch.fft.rfft(torch.tensor(np.ascontiguousarray(mm[(o := next(offs)): o + NS]), device=dev)),
            torch, dev)[0]
    t0 = time.perf_counter()
    for i in range(2000):
        coeffs(0.3 + 1e-5 * i, 0.5)
    ov["coeffs_cpu_s"] = (time.perf_counter() - t0) / 2000
    t0 = time.perf_counter()
    for i in range(5):
        fd_template(0.3 + 0.01 * i, 0.6, F_LO)
    ov["reference_template_pycbc_s"] = (time.perf_counter() - t0) / 5
    ytil = torch.tensor((rng.standard_normal(eng.m) + 1j * rng.standard_normal(eng.m)).astype(np.complex64), device=dev)
    dco5 = torch.tensor(rng.uniform(-1, 1, (5, NK)).astype(np.float32) * SCALE, device=dev)
    lag = torch.tensor([eng.m // 2 + 3, eng.m // 2 - 7, eng.m // 2, eng.m // 2 + 100, eng.m // 2 - 50], device=dev)
    order = np.argsort(eng.f_abs)
    edges = []
    for j in range(16):
        e = np.zeros(eng.m, np.float32); e[order[j * eng.m // 16:(j + 1) * eng.m // 16]] = 1
        edges.append(torch.tensor(e, device=dev))
    ov["chi2_coarse_top5_p16_s"] = _warm_median(lambda: chi2_at_peak(eng, ytil, dco5, -1, lag, edges), torch, dev)[0]
    _child_report(dict(kind="overheads", overheads=ov, device=dev))


LOCAL_CONFIGS = ((2.0, 4), (4.0, 4), (4.0, 2), (2.0, 1), (4.0, 1))   # (half-width s, decimation): M = 4096 ... 32768


def timeall():
    """Driver (no torch here): short children, ONE at a time, baseline and local engine alternated so they see the
    same load. A child is only launched if its footprint, predicted from smaller runs already measured, is under
    MEM_BAR_MB. Headline speed-up is quoted against the STORED 3.36 ms (batch 64); the fresh baselines at small batch
    are secondary and probably slower per template than batch 64 would be (so ratios against them flatter us)."""
    import subprocess
    stored = json.loads((B / "results" / "coh2" / "T_cost.json").read_text())["per_template_s"]
    res = dict(prediction="P1", stored_baseline_s=stored, stored_baseline_batch=64, mem_bar_mb=MEM_BAR_MB, runs=[],
               skipped=[])

    def child(*args):
        out = subprocess.run([sys.executable, "-u", __file__, *map(str, args)], capture_output=True, text=True)
        line = [x for x in out.stdout.splitlines() if x.startswith("RESULT ")]
        if not line:
            print(out.stdout[-2000:], out.stderr[-2000:], flush=True)
            raise RuntimeError(f"child {args} gave no result")
        r = json.loads(line[-1][7:])
        res["runs"].append(r)
        tag = (f"baseline batch {r['batch']}" if r["kind"] == "baseline" else
               f"local M {r['m']} batch {r['batch']}" if r["kind"] == "local" else "overheads")
        t = f"{r['per_template_s']*1e6:9.1f} us per template | " if "per_template_s" in r else ""
        print(f"{tag:28s} {t}footprint peak {r['peak_footprint_mb']} MB (MPS driver {r['mps_driver_mb']} MB, RSS "
              f"{r['peak_rss_mb']} MB) | load {r['load_avg'][0]:.1f}", flush=True)
        return r

    def predicted(kind, samples, **match):
        """Linear extrapolation in samples-per-batch from the two largest runs of this kind already measured."""
        runs = sorted((r["samples_per_batch"], r["peak_footprint_mb"]) for r in res["runs"]
                      if r["kind"] == kind and all(r.get(k) == v for k, v in match.items()))
        runs = sorted({a: b for a, b in runs}.items())
        if len(runs) < 2:
            return None
        (s1, f1), (s2, f2) = runs[-2], runs[-1]
        return f2 + (samples - s2) * (f2 - f1) / (s2 - s1)

    def base_round():
        for nb in (2, 4, 8, 16):
            pred = predicted("baseline", nb * NS)
            if pred is not None and pred > MEM_BAR_MB:
                res["skipped"].append(dict(kind="baseline", batch=nb, predicted_footprint_mb=round(pred)))
                print(f"baseline batch {nb}: SKIPPED, predicted footprint {pred:.0f} MB > {MEM_BAR_MB}", flush=True)
                continue
            child("tbase", nb)

    base_round()
    for i, (tw, dec) in enumerate(LOCAL_CONFIGS):
        for total in (2 ** 20, 2 ** 22, 2 ** 24):
            pred = predicted("local", total, tw=tw, dec=dec)
            if pred is not None and pred > MEM_BAR_MB:
                res["skipped"].append(dict(kind="local", tw=tw, dec=dec, samples_per_batch=total,
                                           predicted_footprint_mb=round(pred)))
                print(f"local tw {tw} dec {dec} total 2^{int(np.log2(total))}: SKIPPED, predicted {pred:.0f} MB", flush=True)
                continue
            child("tlocal", tw, dec, total)
        if i in (1, 3):
            base_round()
    child("tovh")
    base_round()
    res["overheads"] = [r for r in res["runs"] if r["kind"] == "overheads"][0]["overheads"]
    base = {}
    for r in res["runs"]:
        if r["kind"] == "baseline":
            base.setdefault(r["batch"], []).append(r["per_template_s"])
    res["baseline_fresh"] = {str(k): dict(median_s=float(np.median(v)), all_s=v) for k, v in sorted(base.items())}
    res["configs"] = []
    for tw, dec in LOCAL_CONFIGS:
        rows = [r for r in res["runs"] if r["kind"] == "local" and (r["tw"], r["dec"]) == (tw, dec)]
        best = min(rows, key=lambda r: r["per_template_s"])
        matched = [r for r in rows if r["samples_per_batch"] == 8 * NS]          # same samples per batch as baseline-8
        res["configs"].append(dict(tw=tw, dec=dec, m=best["m"], per_template_s=best["per_template_s"],
                                   batch=best["batch"], peak_footprint_mb=best["peak_footprint_mb"],
                                   matched_batch8_per_template_s=matched[0]["per_template_s"] if matched else None,
                                   speedup_vs_stored=stored / best["per_template_s"]))
    prim = [c for c in res["configs"] if (c["tw"], c["dec"]) == (4.0, 4)][0]
    t = prim["per_template_s"]
    res["p1"] = dict(primary_m=8192, per_template_s=t, interval_s=[10e-6, 70e-6], point_s=25e-6,
                     grade="HIT" if 10e-6 <= t <= 70e-6 else ("MISS-fast" if t < 10e-6 else "MISS-slow"),
                     speedup_vs_stored=stored / t)
    res["max_child_footprint_mb"] = max(r["peak_footprint_mb"] for r in res["runs"])
    write("P1_timing.json", res)
    print(f"P1 {res['p1']['grade']}: {t*1e6:.1f} us per template at M = 8192 ({stored/t:.0f}x vs the stored 3.36 ms); "
          f"largest child footprint {res['max_child_footprint_mb']} MB", flush=True)


# ------------------------------------------------------------------ P3: the cost model, with what the old one left out
def cost():
    """Stage-2 total at one per minute (and one per hour), from the stored N_tmpl table (unchanged), the candidate
    chirp masses (as coh2_cost.py), and the times measured in P1_timing.json. Golden check first: with 3.36 ms per
    template and no overheads this code must reproduce the stored 584 / 694 / 703 h."""
    import coh2_cost as CC
    tc = json.loads((B / "results" / "coh2" / "T_cost.json").read_text())
    tm = json.loads((OUT / "P1_timing.json").read_text())
    ov = tm["overheads"]
    half_width = tc["bias"]["all"]["p99"]
    res = dict(prediction="P3", inputs=dict(n_tmpl="results/coh2/T_cost.json (covering estimate, unchanged)",
                                            timing="results/coh2c/P1_timing.json"), golden=[], lattices={}, rows=[])
    cands = {}
    for lab, per_day in (("1/hour", 24.0), ("1/min", 1440.0)):
        mcs, T, thr = CC.candidate_mcs(per_day)
        cands[lab] = mcs
    t_base = tc["per_template_s"]

    def n_each(mcs, l_max):
        grid = sorted((mc, tc["n_tmpl"][f"L{l_max}_mc{mc}"]["n_tmpl"]) for mc in CC.MC_PTS
                      if f"L{l_max}_mc{mc}" in tc["n_tmpl"])
        gm, gn = np.array([g[0] for g in grid]), np.array([g[1] for g in grid])
        return np.exp(np.interp(np.log(mcs), np.log(gm), np.log(np.maximum(gn, 1))))

    ok = True
    for lab, mcs in cands.items():
        for l_max in CC.L_OPTS:
            n = n_each(mcs, l_max)
            mine = 2 * n.sum() * t_base / 3600 * (2 + 900 / len(mcs))
            stored = tc["projection"][f"{lab}_L{l_max}"]["s2_total_hours"]
            ok &= abs(mine / stored - 1) < 1e-6
            res["golden"].append(dict(rate=lab, l_max=l_max, reproduced_h=mine, stored_h=stored))
    res["golden_passed"] = bool(ok)
    print(f"golden (old model reproduced): {'PASS' if ok else 'FAIL'} "
          f"{[round(g['reproduced_h'], 1) for g in res['golden']]}", flush=True)
    assert ok
    mc_pts = np.geomspace(0.175, 0.868, 12)
    t_snip = ov.get("snippet_read_transfer_rfft_s", ov["snippet_rfft_mps_s"])
    # chi^2 at 5 peaks: the exact fine-resolution route if gate E3 has timed it, else the coarse one
    e3p = OUT / "E3_chi2.json"
    t_chi2 = json.loads(e3p.read_text())["chi2_fine_top5_p16_s"] if e3p.exists() else ov["chi2_coarse_top5_p16_s"]
    res["inputs"]["chi2_s"] = t_chi2
    # constructed grids (coh2c_grid.py): the accepted (last) row per box; counts per candidate by the same log-log
    # interpolation over the six chirp-mass points that the old model used for its estimate
    built, gp = {}, OUT / "G_grids.json"
    if gp.exists():
        for r in json.loads(gp.read_text())["rows"]:
            built[(r["l_max"], r["mc_star"])] = r["n_grid"]
    GRID_BUILD_S = 8.0          # per distinct box, measured upper value incl. the Monte Carlo (G_grids.json: 62 s / 14)

    def n_built(mcs, l_max):
        gm = np.array(sorted(m for (l, m) in built if l == l_max))
        gn = np.array([built[(l_max, m)] for m in gm])
        return np.exp(np.interp(np.log(mcs), np.log(gm), np.log(gn)))
    for cfg in tm["configs"]:
        t_loc, tw, dec = cfg["per_template_s"], cfg["tw"], cfg["dec"]
        for l_max in CC.L_OPTS:
            key = f"tw{tw:.0f}_L{l_max}"
            if key not in res["lattices"]:
                res["lattices"][key] = [len(reference_lattice(float(m), half_width, l_max, dtau_max(tw))[0])
                                        for m in mc_pts]
            for lab, mcs in cands.items():
                n = n_each(mcs, l_max)
                n_ref = np.interp(np.log(mcs), np.log(mc_pts), res["lattices"][key])
                per_cd = n * t_loc + n_ref * ov["reference_corr_and_window_s"] + t_snip + t_chi2
                bg_h = 2 * per_cd.sum() / 3600
                passes = 2 + 900 / len(mcs)
                # one-off, shared by every pass: coefficient tables per distinct box, reference templates per lattice
                boxes = np.unique(np.round(mcs, 6))
                once_h = (n_each(boxes, l_max).sum() * ov["coeffs_cpu_s"] +
                          np.interp(np.log(boxes), np.log(mc_pts), res["lattices"][key]).sum()
                          * ov["reference_template_pycbc_s"]) / 3600
                total = bg_h * passes + once_h
                old = tc["projection"][f"{lab}_L{l_max}"]["s2_total_hours"]
                res["rows"].append(dict(rate=lab, l_max=l_max, tw=tw, dec=dec, m=cfg["m"], per_template_us=t_loc * 1e6,
                                        n_candidates=int(len(mcs)), median_n_tmpl=float(np.median(n)),
                                        median_n_refs=float(np.median(n_ref)),
                                        hours=dict(templates=float(2 * (n * t_loc).sum() / 3600 * passes),
                                                   references=float(2 * (n_ref * ov["reference_corr_and_window_s"]).sum()
                                                                    / 3600 * passes),
                                                   snippets=float(2 * len(mcs) * t_snip / 3600 * passes),
                                                   chi2=float(2 * len(mcs) * t_chi2 / 3600 * passes),
                                                   one_off_tables=float(once_h)),
                                        s2_total_hours=float(total), old_model_hours=old, speedup=float(old / total),
                                        total_if_n_tmpl_x2p5=float(total + 1.5 * 2 * (n * t_loc).sum() / 3600 * passes)))
                if (l_max, 0.35) in built:      # the same total with CONSTRUCTED grid counts, plus building the grids
                    nb_ = n_built(mcs, l_max)
                    r = res["rows"][-1]
                    grid_h = len(boxes) * GRID_BUILD_S / 3600
                    r["constructed"] = dict(
                        median_n_tmpl=float(np.median(nb_)), sum_ratio_to_estimate=float(nb_.sum() / n.sum()),
                        grid_build_hours=float(grid_h), n_distinct_boxes=int(len(boxes)),
                        s2_total_hours=float(total + 2 * ((nb_ - n) * t_loc).sum() / 3600 * passes + grid_h +
                                             (n_built(boxes, l_max) - n_each(boxes, l_max)).sum() * ov["coeffs_cpu_s"] / 3600))
                    r["constructed"]["speedup"] = float(old / r["constructed"]["s2_total_hours"])
                if lab == "1/min":
                    r = res["rows"][-1]
                    print(f"1/min L_max {l_max} tw {tw:.0f} dec {dec} (M {cfg['m']}): {r['s2_total_hours']:.1f} h "
                          f"(templates {r['hours']['templates']:.1f}, references {r['hours']['references']:.1f}, snippets "
                          f"{r['hours']['snippets']:.1f}, chi2 {r['hours']['chi2']:.2f}, one-off {r['hours']['one_off_tables']:.2f})"
                          f" | old {old:.0f} h | {r['speedup']:.0f}x | at 2.5x N_tmpl: {r['total_if_n_tmpl_x2p5']:.1f} h"
                          + (f" | CONSTRUCTED grids: {r['constructed']['s2_total_hours']:.1f} h "
                             f"({r['constructed']['sum_ratio_to_estimate']:.2f}x the templates, "
                             f"{r['constructed']['grid_build_hours']:.1f} h to build {r['constructed']['n_distinct_boxes']} grids)"
                             if "constructed" in r else ""), flush=True)
    write("P3_cost.json", res)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "prep":
        prep(sys.argv[2], int(sys.argv[3]), tuple(int(a) for a in sys.argv[4:]))
    elif cmd == "tbase":
        tbase(int(sys.argv[2]))
    elif cmd == "tlocal":
        tlocal(float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    else:
        {"e1": e1, "e2": e2, "e3": e3, "tovh": tovh, "time": timeall, "cost": cost}[cmd]()
