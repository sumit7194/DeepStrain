"""Nuisance-prior sensitivity of the GWTC-5.0 pSEOBNR (df220, dtau220) combined result. Rules: PREREG.md (395ea44).

  python pseob_prior.py <metafile dir> <release dir with joint/ and hierarchical/>   -> results/pseob_prior.json
"""
import gzip
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from scipy.stats import gaussian_kde

EVENTS = ["GW150914_095045", "GW170104_101158", "GW190519_153544", "GW190521_074359", "GW190630_185205",
          "GW190828_063405", "GW190910_112807", "GW200129_065458", "GW200224_222234", "GW200311_115853",
          "GW230628_231200", "GW230914_111401", "GW230927_153832", "GW231028_153006", "GW231102_071736",
          "GW231206_233901", "GW231226_101520", "GW240511_031507", "GW240514_121713", "GW240615_113620",
          "GW240621_195059", "GW240705_053215", "GW240716_034900", "GW240919_061559", "GW240920_124024",
          "GW241006_015333", "GW241127_061008", "GW241129_021832", "GW241225_082815", "GW250114_082203",
          "GW250119_025138"]
LO, HI = -0.8, 2.0
N_SUB, N_GRID, N_BOOT, SEED = 3000, 120, 20, 20261124
FIELDS = ["domega220", "dtau220", "luminosity_distance", "a_1", "a_2", "chirp_mass", "mass_ratio", "psi",
          "network_matched_filter_snr"]
OUT = Path(__file__).resolve().parent / "results" / "pseob_prior_A2.json"


def label(i):
    return "pSEOBNRv5PHM-reweighted" if i < 17 else ("pSEOBNRv5PHM" if i == 29 else "pseob-rd")


NO_PRIOR = ("GW240705_053215", "GW250119_025138")  # PREREG A1: no stored prior; documented family assumed


def load(mdir):
    ev = []
    for i, name in enumerate(EVENTS):
        with h5py.File(Path(mdir) / f"pSEOB_{name}_Prod0.h5", "r") as f:
            g = f[label(i)]
            ps = g["posterior_samples"][:]
            d = {k: np.asarray(ps[k], float) for k in FIELDS}
            pa = g["priors"].get("analytic")
            if pa is not None and "luminosity_distance" in pa:
                d["prior_dl"] = pa["luminosity_distance"][0].decode()
                d["prior_mc"] = pa["chirp_mass"][0].decode()
                d["prior_q"] = pa["mass_ratio"][0].decode()
                d["prior_a"] = pa["a_1"][0].decode() + pa["a_2"][0].decode()
            else:
                assert name in NO_PRIOR, name
                d["prior_dl"] = d["prior_mc"] = d["prior_q"] = d["prior_a"] = None
        d["name"] = name
        ev.append(d)
    ref = next(d for d in ev if d["prior_dl"] is not None)
    for d in ev:
        if d["prior_dl"] is None:
            d["prior_dl"], d["prior_mc"] = ref["prior_dl"], ref["prior_mc"]
            d["prior_q"], d["prior_a"] = ref["prior_q"], ref["prior_a"]
        # bilby's Planck15_LAL is FlatLambdaCDM(H0=67.90, Om0=0.3065): the same cosmology the O1-O4a files spell out
        assert "UniformSourceFrame" in d["prior_dl"] and (
            "H0=67.9, Om0=0.3065" in d["prior_dl"] or "'Planck15_LAL'" in d["prior_dl"]), d["name"]
        assert "UniformInComponentsChirpMass" in d["prior_mc"], d["name"]
        assert "UniformInComponentsMassRatio" in d["prior_q"], d["name"]
        assert d["prior_a"].count("Uniform(minimum=0") == 2, d["name"]
    return ev


def dl_prior_pdf(dl, text):
    assert "UniformSourceFrame" in text, text
    cosmo = FlatLambdaCDM(H0=67.9, Om0=0.3065)
    zg = np.linspace(1e-4, 1.6, 20000)
    dlg = cosmo.luminosity_distance(zg).value
    dens = cosmo.differential_comoving_volume(zg).value / (1 + zg) / np.gradient(dlg, zg)
    return np.interp(dl, dlg, dens)


def weights(d, variant):
    if variant in ("base", "boot"):
        return np.ones_like(d["dtau220"])
    if variant == "V1":
        return 1 / dl_prior_pdf(d["luminosity_distance"], d["prior_dl"])
    if variant == "V2":
        return d["luminosity_distance"] ** -2 / dl_prior_pdf(d["luminosity_distance"], d["prior_dl"])
    if variant == "V3":
        return d["a_1"] ** 2 * d["a_2"] ** 2
    if variant == "V4":
        assert "UniformInComponentsChirpMass" in d["prior_mc"]
        q = d["mass_ratio"]
        return 1 / (d["chirp_mass"] * (1 + q) ** 0.4 * q ** -1.2)
    if variant == "F1":
        return 1 + 0.5 * np.cos(4 * d["psi"])
    if variant == "P1":
        return 1 / (1 + d["dtau220"])
    raise ValueError(variant)


def ess(w):
    return float(w.sum() ** 2 / (w ** 2).sum())


def subsample(d, w, rng, boot):
    """PREREG A2: all posterior samples, weights passed to the KDE (run 1 resampled to N_SUB draws)."""
    n = len(w)
    idx = rng.integers(0, n, n) if boot else np.arange(n)
    return np.vstack([d["domega220"][idx], d["dtau220"][idx]]), w[idx]


def joint_logL(samples, grid):
    """Sum over events of log 2-D KDE (scipy gaussian_kde, as utils_o3b.multiply_likelihoods), on grid + at (0,0)."""
    pts = np.column_stack([grid, [0.0, 0.0]])
    tot = np.zeros(pts.shape[1])
    for s, w in samples:
        tot += np.log(np.maximum(gaussian_kde(s, weights=w)(pts), 1e-300))
    return tot[:-1], tot[-1]


def q2d(logL, logL0, jac=None):
    m = np.exp(logL - logL.max())
    if jac is not None:
        m = m * jac
    return float(m[logL > logL0].sum() / m.sum())


def make_grid(samples):
    allpts = np.hstack([s for s, _ in samples])
    lo = np.maximum(np.percentile(allpts, 0.5, axis=1) - 0.05, LO)
    hi = np.minimum(np.percentile(allpts, 99.5, axis=1) + 0.05, HI)
    lo, hi = np.minimum(lo, -0.05), np.maximum(hi, 0.05)
    gx = np.linspace(lo[0], hi[0], N_GRID)
    gy = np.linspace(lo[1], hi[1], N_GRID)
    X, Y = np.meshgrid(gx, gy, indexing="ij")
    return np.vstack([X.ravel(), Y.ravel()]), gx, gy


def released_q2d(rel):
    """K1a: utils_o3b.get_2d_quantile_from_samples(reflect_around_x=False) on LVK's released joint samples."""
    x = np.loadtxt(gzip.open(Path(rel) / "joint/joint_posterior_results/domega_220_comb.dat.gz"))
    y = np.loadtxt(gzip.open(Path(rel) / "joint/joint_posterior_results/dtau_220_comb.dat.gz"))
    pts = np.vstack([x, y])
    kde = gaussian_kde(pts)
    dens, d0 = kde(pts), kde([[0.0], [0.0]])[0]
    return float(np.count_nonzero(d0 < dens) / len(dens)), float(np.median(y))


def run_joint(ev, variant, rng, boot=False, grid=None):
    samples, esses = [], []
    for d in ev:
        w = weights(d, variant)
        esses.append(ess(w))
        samples.append(subsample(d, w, rng, boot))
    if grid is None:
        grid = make_grid(samples)
    G, gx, gy = grid
    logL, logL0 = joint_logL(samples, G)
    p = np.exp(logL - logL.max()).reshape(len(gx), len(gy))
    mtau = p.sum(0)
    cdf = np.cumsum(mtau) / mtau.sum()
    med_tau = float(np.interp(0.5, cdf, gy))
    out = {"Q2D": q2d(logL, logL0), "dtau_median": med_tau, "min_ess": min(esses),
           "ess": {d["name"]: round(e) for d, e in zip(ev, esses)}}
    Y = G[1]
    out["Q2D_flat_in_rate"] = q2d(logL, logL0, jac=(1 + Y) ** -2)
    return out, grid


def describe(ev):
    rows = {}
    for d in ev:
        name = d["name"]
        rows[name] = {"corr_dL_dtau": float(np.corrcoef(d["luminosity_distance"], d["dtau220"])[0, 1]),
                      "dtau_median": float(np.median(d["dtau220"])),
                      "snr_net_median": float(np.median(d["network_matched_filter_snr"])),
                      "n": int(len(d["dtau220"]))}
    return rows


def main():
    t0 = time.time()
    mdir, rel = sys.argv[1], sys.argv[2]
    rng = np.random.default_rng(SEED)
    res = {"prereg": "PREREG.md @ 395ea44 + A1 57cbbbd + A2 52f2ca5 (non-blind)", "n_events": len(EVENTS),
           "samples": "all, weighted KDE (A2)", "n_grid": N_GRID}
    k1a_q, k1a_med = released_q2d(rel)
    res["K1a"] = {"Q2D_released": k1a_q, "dtau_median_released": k1a_med, "target": 0.981,
                  "pass": abs(k1a_q - 0.981) <= 0.010}
    print("K1a", res["K1a"], f"{time.time()-t0:.0f}s", flush=True)
    ev = load(mdir)
    res["describe"] = describe(ev)
    base, grid = run_joint(ev, "base", rng)
    res["base"] = base
    res["K1b"] = {"Q2D": base["Q2D"], "dtau_median": base["dtau_median"],
                  "pass": abs(base["Q2D"] - 0.981) <= 0.015 and abs(base["dtau_median"] - k1a_med) <= 0.01}
    print("base", {k: base[k] for k in ("Q2D", "dtau_median", "Q2D_flat_in_rate")}, f"{time.time()-t0:.0f}s",
          flush=True)
    boots = [run_joint(ev, "boot", rng, boot=True, grid=grid)[0]["Q2D"] for _ in range(N_BOOT)]
    res["sigma_MC"] = float(np.std(boots, ddof=1))
    print("sigma_MC", res["sigma_MC"], f"{time.time()-t0:.0f}s", flush=True)
    for v in ("V1", "V2", "V3", "V4", "F1", "P1"):
        r, _ = run_joint(ev, v, rng, grid=grid)
        res[v] = r
        print(v, {k: r[k] for k in ("Q2D", "dtau_median", "min_ess")}, f"{time.time()-t0:.0f}s", flush=True)
    res["V5"] = {"Q2D": base["Q2D_flat_in_rate"]}
    ev29 = [d for d in ev if d["name"] not in NO_PRIOR]
    r29 = {}
    for v in ("base", "V1", "V2", "V3", "V4"):
        r, _ = run_joint(ev29, v, rng)
        r29[v] = {"Q2D": r["Q2D"], "dtau_median": r["dtau_median"], "min_ess": r["min_ess"]}
    r29["V5"] = {"Q2D": run_joint(ev29, "base", rng)[0]["Q2D_flat_in_rate"]}
    res["excl_no_prior_29"] = r29
    print("29-event", {v: round(r29[v]["Q2D"], 4) for v in r29}, f"{time.time()-t0:.0f}s", flush=True)
    res["K1c"] = "not attempted (PREREG A1.3)"
    b = base["Q2D"]
    res["F1_pass"] = abs(res["F1"]["Q2D"] - b) <= 2 * res["sigma_MC"]
    res["P1_pass"] = (b - res["P1"]["Q2D"]) >= 0.05
    ess_ok = {v: res[v]["min_ess"] >= 200 for v in ("V1", "V2", "V3", "V4")}
    res["ess_ok"] = ess_ok
    gates = res["K1a"]["pass"] and res["K1b"]["pass"] and res["F1_pass"] and res["P1_pass"]
    phys = {v: res[v]["Q2D"] for v in ("V1", "V3", "V4", "V5") if ess_ok.get(v, True)}
    if not gates:
        verdict = "NO VERDICT (gate failed)"
    elif any(q < 0.90 for q in phys.values()):
        verdict = "PRIOR-SENSITIVE"
    elif all(abs(q - b) < 0.03 for q in phys.values()):
        verdict = "ROBUST"
    else:
        verdict = "MODEST"
    res["verdict"] = verdict
    res["seconds"] = round(time.time() - t0)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    print("VERDICT", verdict, f"{res['seconds']}s")


if __name__ == "__main__":
    main()
