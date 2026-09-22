#!/usr/bin/env python
"""GW231028: a no-hair NPE built for a ~240 Msun remnant, validated at that mass BEFORE the event is read.

WHY. v5 stacking is parked on one finding: only GW250114 measures delta; every fainter event returns the prior.
arXiv:2509.08657 reports GW231028_153006's 221 overtone at BF ~189 (remnant 246 Msun redshifted, chi 0.81); the
event is not in our 12-event sample. 09's network cannot be applied (prior M in [40, 120], 40 ms window). 40's
preflight kept the idealised simulator: whitening fidelity at 240 Msun is 0.981 vs 0.959 for the validated 251 Hz
control. So this is 09's machinery with the scale-dependent settings moved to the event's mass:

  prior       M in [150, 350] Msun (detector frame; GWOSC 144 x 1.64 = 236, Wang 246), chi in [0.05, 0.95],
              delta in [-0.5, 0.5]  -- the SAME delta prior as 09, so delta_sigma/prior is comparable.
  window      SEG = 0.10 s  (~6.3 tau_220 at 240 Msun, chi 0.81)
  start time  t0 in [0, 20] ms  (09's 6 ms is ~18 M at 68 Msun; 20 ms is ~17 M at 240 Msun)
  amplitude   (1, 12) whitened units (event peak |x| 3.7 H1 / 2.9 L1 from 40; 09's (2, 12) puts it near an edge)
  training    150,000 simulations (09's fix-round budget), same MAF + Embed

PERSISTENCE: WEIGHTS ONLY. 09 serialises the whole posterior object and reloads it with arbitrary-object
loading. Here only the density estimator's state_dict is saved and loaded with weights_only=True; the
architecture is rebuilt from code and re-bound to the prior.

PRE-REGISTRATION (committed before the first full run):
  R1  per-parameter 90% coverage on 200 held-out simulations, each in [0.85, 0.95] (09's gate).
  R2  8 injections per delta_true in {0, +0.3} into REAL O4a noise before the event (H1+L1), at (240, 0.81),
      raw amplitude calibrated per noise draw so the whitened signal peak is 3.3 (the mean event peak; the event's
      own peak includes noise, so this is an approximation stated as one).
        R2a  |mean delta_hat(delta_true = 0)| <= 0.10                                  (unbiased at this mass)
        R2b  mean delta_hat(0.3) - mean delta_hat(0) > 2 SE                            (the estimate MOVES)
      R2b is the lesson of 30: an interval evaluated at the prior's centre cannot tell a measurement from a
      prior-returning estimator. The response (difference / 0.3) is reported beside GW250114-loudness's ~0.3
      (09/v2: medians ~+0.09 for true +0.3), so "moves at all" is the bar, not "recovers the truth".
  R3  GW231028 itself:
        trust  median M in [212, 283] (published 236-246, allowing 09's known +10% pull) and median chi in
               0.81 +/- 0.15 -- 13's per-event trust check. Failing it, delta is not reported as a measurement.
        informative  delta_sigma / prior_sigma < 0.88 (the project's gate; GW250114 = 0.82) AND R2b passed
                     AND below the 10th percentile of R0.
      PREDICTION: NOT informative. GW231028's network SNR is ~22 (GWOSC) vs GW250114's ~80, and every event
      below ~30 in our 12-event sample returned the prior; a heavy system puts more of its SNR in the ringdown,
      which is the only reason this is worth running. Stated so it can be wrong.
  R0  AMENDED BEFORE THE FULL RUN, after the --smoke run: the NO-SIGNAL FLOOR of THIS network -- delta sd/prior on
      30 pure white-noise inputs. Reason: the smoke network (3 epochs, barely trained) returned 0.824 on the event,
      i.e. it PASSED the 0.88 bar while knowing almost nothing, because an untrained flow emits a narrower-than-prior
      blob for any input. Checked on 09 the same night: its trained no-signal floor is 0.921 [P10 0.886, P90 0.968],
      so 0.88 was a sound bar FOR 09 and GW250114's 0.82 sits below the whole band -- but the bar is a property of
      a network, not a constant, so each network's own floor must be measured before its event is read.

Run:  .venv/bin/python scripts/41_gw231028_npe.py --smoke          # plumbing check, minutes, separate cache
      nohup .venv/bin/python scripts/41_gw231028_npe.py > log 2>&1 &   # the pre-registered run
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from gwpy.timeseries import TimeSeries

import rdlib
import sbilib
from sbilib import Embed, kerr220, kerr221

torch.manual_seed(0)
torch.set_num_threads(4)             # shared box: leave cores for the bank run and sister projects
rng = np.random.default_rng(0)

RESULTS = Path(__file__).resolve().parent.parent / "results"
EVENT = "GW231028_153006"
FS = 4096.0
SEG = 0.10
N_SAMP = int(SEG * FS) + int(SEG * FS) % 2
T0_MAX_MS = 20.0
AMP_RANGE = (1.0, 12.0)
M_LO, M_HI = 150.0, 350.0
M_INJ, CHI_INJ, PEAK_TARGET = 240.0, 0.81, 3.3
PRIOR_SD = 1.0 / np.sqrt(12.0)       # sd of U(-0.5, 0.5)


def simulate(m, c, d):
    return sbilib.simulate(m, c, d, rng, n_samp=N_SAMP, t0_max_ms=T0_MAX_MS, amp_range=AMP_RANGE)


def sims(n, tag):
    th = np.column_stack([rng.uniform(M_LO, M_HI, n), rng.uniform(0.05, 0.95, n), rng.uniform(-0.5, 0.5, n)])
    xs = np.empty((n, 2 * N_SAMP), dtype=np.float32)
    for k, (m, c, d) in enumerate(th):
        xs[k] = simulate(m, c, d).reshape(-1)
        if k % 5000 == 0:
            rdlib.progress(f"41_gw231028_sims{tag}", k, n)
    return torch.tensor(th, dtype=torch.float32), torch.tensor(xs)


def posterior(n_train, epochs, cache, tag):
    from sbi.inference import NPE
    from sbi.neural_nets import posterior_nn
    from sbi.utils import BoxUniform
    prior = BoxUniform(torch.tensor([M_LO, 0.05, -0.5]), torch.tensor([M_HI, 0.95, 0.5]))
    build = posterior_nn(model="maf", embedding_net=Embed(n_samp=N_SAMP), hidden_features=80, num_transforms=5)
    if cache.exists():
        print(f"loading cached weights {cache.name}", flush=True)
        # Rebuilding needs a batch to size the lazy layers; the z-scoring buffers it computes are then
        # overwritten by the saved state, so the batch's statistics do not leak into the loaded network.
        th, xs = sims(512, tag + "_rebuild")
        net = build(th, xs)
        net.load_state_dict(torch.load(cache, weights_only=True))
    else:
        th, xs = sims(n_train, tag)
        npe = NPE(prior=prior, density_estimator=build)
        npe.append_simulations(th, xs)
        print("training NPE ...", flush=True)
        with rdlib.heartbeat(f"41_gw231028_train{tag}"):
            net = npe.train(training_batch_size=256, max_num_epochs=epochs, stop_after_epochs=10,
                            show_train_summary=True)
        torch.save(net.state_dict(), cache)
    net.train(False)
    return NPE(prior=prior).build_posterior(density_estimator=net, prior=prior)


def summarize(s):
    q = lambda j: [float(x) for x in np.percentile(s[:, j], [50, 5, 95])]
    return {"M": q(0), "chi": q(1), "delta": q(2), "delta_sd_over_prior": float(s[:, 2].std() / PRIOR_SD)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    smoke = ap.parse_args().smoke
    tag = "_smoke" if smoke else ""
    n_train, epochs, n_r1, n_r2 = (3000, 3, 20, 1) if smoke else (150_000, 80, 200, 8)
    out = RESULTS / f"41_gw231028_npe{tag}.json"
    post = posterior(n_train, epochs, RESULTS / f"41_npe_weights{tag}.pt", tag)
    rep = {"event": EVENT, "smoke": smoke, "settings": {"prior_M": [M_LO, M_HI], "SEG": SEG, "t0_max_ms": T0_MAX_MS,
                                                        "amp_range": AMP_RANGE, "n_train": n_train}}

    # ---- R1 --------------------------------------------------------------------------------------------------
    hits = np.zeros(3)
    for k in range(n_r1):
        th = (rng.uniform(M_LO, M_HI), rng.uniform(0.05, 0.95), rng.uniform(-0.5, 0.5))
        s = post.sample((300,), x=torch.tensor(simulate(*th).reshape(1, -1)), show_progress_bars=False).numpy()
        for j, truth in enumerate(th):
            lo, hi = np.percentile(s[:, j], [5, 95])
            hits[j] += lo <= truth <= hi
    cov = hits / n_r1
    r1 = bool(all(0.85 <= c <= 0.95 for c in cov))
    rep["R1"] = {"coverage": {"M": cov[0], "chi": cov[1], "delta": cov[2]}, "pass": r1}
    print(f"R1 coverage M {cov[0]:.2f} chi {cov[1]:.2f} delta {cov[2]:.2f}  -> {'PASS' if r1 else 'FAIL'}", flush=True)

    # ---- R0: this network's no-signal floor (amendment; see docstring) ---------------------------------------
    floor = []
    for _ in range(30 if not smoke else 5):
        x = torch.tensor(rng.standard_normal((1, 2 * N_SAMP)).astype(np.float32))
        floor.append(summarize(post.sample((1000,), x=x, show_progress_bars=False).numpy())["delta_sd_over_prior"])
    p10 = float(np.percentile(floor, 10))
    rep["R0"] = {"noise_only_sd_over_prior": floor, "median": float(np.median(floor)), "p10": p10}
    print(f"R0 no-signal floor: delta sd/prior median {np.median(floor):.3f}, P10 {p10:.3f}  (09's: 0.921, P10 0.886)",
          flush=True)

    # ---- R2: real O4a noise, calibrated loudness --------------------------------------------------------------
    f1, tau1 = kerr220.f_tau(M_INJ, CHI_INJ)
    f2, tau2 = kerr221.f_tau(M_INJ, CHI_INJ)
    base = rdlib.event_gps(EVENT)
    inj = {0.0: [], 0.3: []}
    k_noise = 0
    for d_true in (0.0, 0.3):
        while len(inj[d_true]) < n_r2 and k_noise < 40:
            center = base - 300 - 128 * k_noise
            k_noise += 1
            try:
                segs = []
                for det in ("H1", "L1"):
                    raw = TimeSeries.fetch_open_data(det, center - 32, center + 32, cache=True)
                    if not np.isfinite(raw.value).all():
                        raise ValueError("NaNs")
                    ph = rng.uniform(-np.pi, np.pi, 2)
                    unit = [dict(f=f1, tau=tau1, amp=1e-21, phi=ph[0]),
                            dict(f=f2 * (1 + d_true), tau=tau2, amp=1e-21, phi=ph[1])]
                    w0 = raw.whiten(4, 2)
                    wsig = rdlib.inject_ringdown(raw, center, unit).whiten(4, 2) - w0
                    scale = PEAK_TARGET / np.abs(wsig.crop(center, center + SEG).value).max()
                    for p in unit:
                        p["amp"] *= scale
                    w = rdlib.inject_ringdown(raw, center, unit).whiten(4, 2)
                    seg = w.crop(center, center + SEG + 0.01).value[:N_SAMP]
                    assert len(seg) == N_SAMP
                    segs.append(seg)
                s = post.sample((400,), x=torch.tensor(np.stack(segs).reshape(1, -1).astype(np.float32)),
                                show_progress_bars=False).numpy()
                inj[d_true].append(summarize(s))
                r = inj[d_true][-1]
                print(f"  R2 d_true={d_true:+.1f}  M {r['M'][0]:6.1f}  chi {r['chi'][0]:.2f}  delta {r['delta'][0]:+.3f} "
                      f"[{r['delta'][1]:+.2f},{r['delta'][2]:+.2f}]  sd/prior {r['delta_sd_over_prior']:.2f}", flush=True)
            except Exception as exc:                                     # noqa: BLE001 -- reported, not hidden
                print(f"  [noise {k_noise}] skipped: {exc}", flush=True)
    d0 = np.array([r["delta"][0] for r in inj[0.0]]); d3 = np.array([r["delta"][0] for r in inj[0.3]])
    diff = d3.mean() - d0.mean()
    se = np.sqrt(d0.var(ddof=1) / len(d0) + d3.var(ddof=1) / len(d3)) if min(len(d0), len(d3)) > 1 else float("nan")
    r2a = bool(abs(d0.mean()) <= 0.10)
    r2b = bool(diff > 2 * se)
    rep["R2"] = {"injections": {str(k): v for k, v in inj.items()}, "mean_delta_0": float(d0.mean()),
                 "mean_delta_03": float(d3.mean()), "difference": float(diff), "se": float(se),
                 "response": float(diff / 0.3), "R2a": r2a, "R2b": r2b,
                 "reference_response_GW250114_loudness": 0.3}
    print(f"R2a mean delta_hat(0) {d0.mean():+.3f} -> {'PASS' if r2a else 'FAIL'};  R2b difference {diff:+.3f} "
          f"(2 SE = {2*se:.3f}), response {diff/0.3:.2f} (GW250114-loudness ~0.3) -> {'PASS' if r2b else 'FAIL'}", flush=True)

    # ---- R3: the event --------------------------------------------------------------------------------------
    gps = rdlib.event_gps(EVENT)
    segs = []
    for det in ("H1", "L1"):
        white = rdlib.fetch_whitened(det, gps, bandpass=False)
        pk = rdlib.find_peak(white.bandpass(*rdlib.BAND), gps)
        seg = white.crop(pk, pk + SEG + 0.01).value[:N_SAMP]
        assert len(seg) == N_SAMP
        segs.append(seg)
    s = post.sample((3000,), x=torch.tensor(np.stack(segs).reshape(1, -1).astype(np.float32)),
                    show_progress_bars=False).numpy()
    ev = summarize(s)
    trust = bool(212 <= ev["M"][0] <= 283 and abs(ev["chi"][0] - 0.81) <= 0.15)
    informative = bool(trust and r2b and ev["delta_sd_over_prior"] < min(0.88, p10))
    rep["R3"] = dict(ev, trust=trust, informative=informative, kerr_inside_90=bool(ev["delta"][1] <= 0 <= ev["delta"][2]))
    print(f"R3 {EVENT}: M {ev['M'][0]:.1f} [{ev['M'][1]:.1f},{ev['M'][2]:.1f}]  chi {ev['chi'][0]:.2f} "
          f"[{ev['chi'][1]:.2f},{ev['chi'][2]:.2f}]  delta {ev['delta'][0]:+.3f} [{ev['delta'][1]:+.2f},{ev['delta'][2]:+.2f}]")
    print(f"   trust (M in [212,283], chi 0.81+/-0.15): {trust};  delta sd/prior {ev['delta_sd_over_prior']:.3f} "
          f"(informative < min(0.88, R0 P10 {p10:.3f}); GW250114 0.82);  INFORMATIVE: {informative}")
    out.write_text(json.dumps(rep, indent=2))
    print(f"wrote {out.name}")


if __name__ == "__main__":
    main()
