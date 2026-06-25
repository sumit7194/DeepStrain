"""A1 (SISTER_REQUESTS / TheBridge §9, "most original"): does an NPE's AMORTIZATION GAP predict sim→real TRANSFER?

Train N no-hair δ NPE variants spanning amortization (lever: training-set size N_TRAIN — the cleanest single
lever; all else fixed = the §09 architecture). For each, on a COMMON protocol report two numbers:
  - amortization_gap (on SIM): mean over params of |empirical 90% coverage − 0.90| (SBC/coverage deviation —
    how much amortization compromised the per-x posterior; an under-trained NPE is over-confident → big gap).
  - transfer (sim→real): mean per-param coverage on REAL-O4-noise injections − mean coverage on SIM injections
    (how much the inference degrades sim→real; §09's R2 protocol, generalized to random θ).
Save each checkpoint + a JSON {variant, n_train, amortization_gap, transfer}. The bridge correlates them
(read-only): does a bigger amortization gap predict worse — or better — transfer? Genuinely unknown.

Run:  .venv/bin/python scripts/19_amortization_transfer.py [--n-cov 200] [--n-real 60] [--smoke]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from gwpy.timeseries import TimeSeries
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn
from sbi.utils import BoxUniform

import rdlib
import sbilib
from sbilib import Embed, FS, N_SAMP, SEG, kerr220, kerr221, simulate

RESULTS = Path(__file__).resolve().parent.parent / "results"
MODELS = RESULTS  # checkpoints alongside results
EVENT = "GW250114_082203"
PRIOR = BoxUniform(torch.tensor([40.0, 0.05, -0.5]), torch.tensor([120.0, 0.95, 0.5]))
N_TRAIN_VARIANTS = [5000, 15000, 40000, 90000, 150000]
LO, HI = np.array([40.0, 0.05, -0.5]), np.array([120.0, 0.95, 0.5])


def train_npe(n_train, rng):
    thetas = np.column_stack([rng.uniform(*b, n_train) for b in zip(LO, HI)])
    xs = np.empty((n_train, 2 * N_SAMP), dtype=np.float32)
    for k, (m, c, d) in enumerate(thetas):
        xs[k] = simulate(m, c, d, rng).reshape(-1)
        if k % 5000 == 0:
            rdlib.progress(f"19_sims_{n_train}", k, n_train)
    density = posterior_nn(model="maf", embedding_net=Embed(), hidden_features=80, num_transforms=5)
    npe = NPE(prior=PRIOR, density_estimator=density)
    npe.append_simulations(torch.tensor(thetas, dtype=torch.float32), torch.tensor(xs))
    est = npe.train(training_batch_size=256, max_num_epochs=80, stop_after_epochs=10)
    return npe.build_posterior(est)


@torch.no_grad()
def coverage(post, x_builder, n, rng):
    """per-param 90% coverage: draw θ, build x via x_builder(m,c,d,rng), check θ in the 5–95% CI."""
    hits, used = np.zeros(3), 0
    for _ in range(n):
        m, c, d = rng.uniform(40, 120), rng.uniform(0.05, 0.95), rng.uniform(-0.5, 0.5)
        x = x_builder(m, c, d, rng)
        if x is None:
            continue
        s = post.sample((300,), x=torch.tensor(x.reshape(1, -1).astype(np.float32)),
                        show_progress_bars=False).numpy()
        for j, truth in enumerate((m, c, d)):
            lo, hi = np.percentile(s[:, j], [5, 95])
            hits[j] += lo <= truth <= hi
        used += 1
    return hits / max(used, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cov", type=int, default=200)
    ap.add_argument("--n-real", type=int, default=60)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    n_cov, n_real = (40, 15) if args.smoke else (args.n_cov, args.n_real)
    variants = [5000, 40000] if args.smoke else N_TRAIN_VARIANTS
    rng = np.random.default_rng(0)

    # real-O4-noise pool (raw segments around the event; §09 R2 scheme) — fetch once, retry flaky TLS
    base = rdlib.event_gps(EVENT)
    pool = []
    k = 0
    while len(pool) < 15 and k < 40:
        center = base - 300 - 128 * k; k += 1
        try:
            seg = {}
            for det in ("H1", "L1"):
                raw = TimeSeries.fetch_open_data(det, center - 32, center + 32, cache=True)
                if not np.isfinite(raw.value).all():
                    raise ValueError("NaN")
                seg[det] = raw
            pool.append((center, seg))
        except Exception:
            continue
    print(f"real-noise pool: {len(pool)} clean O4 segments\n", flush=True)

    def sim_x(m, c, d, rng):
        return simulate(m, c, d, rng)

    def real_x(m, c, d, rng):
        center, seg = pool[rng.integers(0, len(pool))]
        f1, t1 = kerr220.f_tau(m, c); f2, t2 = kerr221.f_tau(m, c)
        segs = []
        for det in ("H1", "L1"):
            params = [dict(f=f1, tau=t1, amp=2.0e-21, phi=rng.uniform(-np.pi, np.pi)),
                      dict(f=f2 * (1 + d), tau=t2, amp=2.0e-21, phi=rng.uniform(-np.pi, np.pi))]
            inj = rdlib.inject_ringdown(seg[det], center, params).whiten(4, 2)
            s = inj.crop(center - 0.002, center - 0.002 + SEG + 0.01).value[:N_SAMP]
            if len(s) != N_SAMP or not np.isfinite(s).all():
                return None
            segs.append(s)
        return np.stack(segs)

    out = []
    print(f"{'N_train':>8} {'sim cov (M,χ,δ)':>22} {'real cov':>22} {'amort_gap':>10} {'transfer':>9}")
    for n_train in variants:
        post = train_npe(n_train, np.random.default_rng(100 + n_train))
        torch.save(post, MODELS / f"19_npe_n{n_train}.pt")
        sc = coverage(post, sim_x, n_cov, np.random.default_rng(1))
        rc = coverage(post, real_x, n_real, np.random.default_rng(2))
        gap = float(np.mean(np.abs(sc - 0.90)))
        transfer = float(np.mean(rc) - np.mean(sc))
        out.append({"n_train": n_train, "sim_cov": sc.tolist(), "real_cov": rc.tolist(),
                    "amortization_gap": gap, "transfer": transfer})
        print(f"{n_train:>8} {str(np.round(sc,2)):>22} {str(np.round(rc,2)):>22} {gap:>10.3f} {transfer:>+9.3f}",
              flush=True)

    # correlation hint (the bridge does the full read-only version)
    g = np.array([r["amortization_gap"] for r in out]); t = np.array([r["transfer"] for r in out])
    corr = float(np.corrcoef(g, t)[0, 1]) if len(out) > 2 else float("nan")
    (RESULTS / "19_amortization_transfer.json").write_text(json.dumps(
        {"variants": out, "amortization_gap_vs_transfer_corr": corr,
         "note": "lever=N_train; gap=mean|cov_sim-0.9|; transfer=mean(cov_real)-mean(cov_sim). "
                 "Checkpoints 19_npe_n{N}.pt. For TheBridge §9 read-only correlation."}, indent=2))
    print(f"\namortization_gap vs transfer across variants: corr = {corr:+.2f} (bridge does the real test). "
          f"wrote 19_amortization_transfer.json + {len(variants)} checkpoints.")


if __name__ == "__main__":
    main()
