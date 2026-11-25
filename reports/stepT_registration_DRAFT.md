# DRAFT registration: tuning step T and gate G0 for the coherent second look (LongChirp)

*DeepStrain [model: Claude Opus 5.5], 2026-11-26. **DRAFT, nothing runs from it.***
- On approval it becomes the dated amendment **T1** to `primordial_blackhole_search/benchmark/PREREG_coherent2.md`
  (frozen 715a0b3), committed on main before any T run.
- Builds on the redesign (c) round (`fable/2026-11-26-coh2c` @ 4f01c22) and its verification
  (`verify/2026-11-26-coh2c`, `reports/coh2c_verification.md`).
- Tuning set only (the 24 old LongChirp test segments). The 24 fresh segments stay unfetched and unopened unless G0
  passes.

## 1. What T decides

The frozen plan (§6, §6b) requires T to fix the tuned values, then evaluate G0 on the tuning set.

- **If G0 passes**, the fresh run is licensed.
- **If it fails**, stop. The fresh segments stay unopened, and the result is recorded as "G0 not passed" with the
  failing clause and its numbers.

## 2. Amendments adopted with T1 (each measured; numbers from the round and its verification)

| # | Change from the frozen text | Why | Measured effect |
|---|---|---|---|
| D1 | Stage 2 evaluated by the **local heterodyned refiner** at full rate (±4-s window, 4,096 samples/s) | Same statistic, same lags; only the arithmetic is cheaper | Box max equal to brute force within ≤ 5·10⁻⁶ on 6 real grids (7,178 templates), same argmax |
| D2 | Templates sharing a reference use **the reference's start frequency** f_start(Mc_ref, L_max) | The engine needs a common band per reference | Signal ρ² ±1.4% at the box edge (0 at centre); box max on the lowest-Mc caught injection −0.02%; template durations 0.965–1.036 × L_max |
| D3 | **r₁ = one per minute** (39,321 tuning background candidates) | The frozen §5 rule picks it when stage 2 ≤ 96 h; §9/§10's "one per hour forced" used the old cost | Constructed-grid cost 17.8 h (L_max 128) / 22.3 h (384) |
| D4 | Thin boxes keep the over-placed constructed grids | Cheap; a 1-D chain is not needed | Grids at Mc 0.18 / 0.86 hold 5–11× the estimate, but are small in absolute terms |
| D5 | Box half-width a **uniform 2.20%** in chirp mass (the larger q-stratum p99), not the pooled 2.15% the code used | The frozen text says "p99 stratified by q" (2.20% for q < 0.5, 1.90% for q ≥ 0.5); uniform-max is the conservative single box | ~+2% templates |
| D6 | **Memory is declared as footprint** (top MEM), ≤ 2 GB, on MPS | RSS hides MPS memory; T-cost's 1.31 GB was RSS | One box measured 1,859 MB footprint, including pycbc in-process. Production keeps pycbc out of the engine process. |

## 3. The statistic (fixed by this registration; formulas read from PyCBC 2.10.0 source, 2026-11-26)

**Per detector, per template and lag:** complex coherent z, normalised (D1/D2), with PyCBC-default autogating as in the
real-background test.

**Allen χ²:**
- frequency-domain, p equal-SNR bins, reduced as χ²_r = χ² / (2p − 2);
- computed at the **top K = 5 (template, lag) peaks of |z|²** per detector per candidate, by the exact route of gate E3.

**PSD variation v:** `pycbc.psd.variation.calc_filt_psd_variation`, using its documented parameters:
- segment 8 s, short_segment 0.25 s, psd_long_segment 512 s;
- psd_duration 8 s, psd_stride 4 s, median average;
- low_freq = **50 Hz** (our band; the docstring default is 20), high_freq 480 Hz.

v is evaluated at the peak time with `find_trigger_value`.

**Ranking, exactly `pycbc.events.ranking.newsnr_sgveto_psdvar` without the sine-Gaussian term:**
- **Small-variation floor:** if v < 0.65 (`min_expected_psdvar`), set v = 1.
- **Rescaling:** ρ' = ρ · v^(−1/2) and χ²_r' = χ²_r / v.
- **Down-weighting:** ρ̂ = ρ' · [½(1 + (χ²_r')^(q/n))]^(−1/q) when χ²_r' > 1, otherwise ρ̂ = ρ', with q = 6, n = 2 (the
  `newsnr` defaults).

The sine-Gaussian χ² is omitted, as accepted in review: it targets short, high-mass templates.

**Network and candidate score:**
- ρ̂²_net = ρ̂_H² + max over |δ| ≤ 10 ms of ρ̂_L²;
- the candidate score is the maximum over the box's top peaks.

**Gate before use (S0):**
- `newsnr` and `calc_filt_psd_variation` are **imported from PyCBC and called, not reimplemented**.
- A golden check confirms our call path on one tuning segment: the v series is finite with a median within [0.9, 1.1];
  ρ̂ = ρ whenever χ²_r ≤ 1.

## 4. Tuned values and how they are chosen (on the tuning set only)

**Search space:** L_max ∈ {128, 384} s × p ∈ {4, 8, 16} = **6 primary configurations**, all with v.

- **Ablations, reported for every configuration, never selectable:** χ² only (no v), and plain ρ.
- L_max = 256 is dropped as dominated: its cost and statistic lie between the other two.

**Why the cost is two passes, not six:** p and v enter only at the top-K peaks, so one stage-2 pass per L_max yields
all three p values and both ablations.

**Objective:** maximise tuning-set cascade distance at 1/day, subject to the tuning-set gates of §5 passing. Every
configuration tried is logged, and the count (6) is reported beside R_tune.

**Also fixed in T, before any fresh step:**
- the decoy amplitude calibration: each decoy family is scaled to the median plain-coherent ρ of caught tuning
  signals;
- the fresh-injection seed: `20261127`, declared now.

## 5. Gates on the tuning set (the frozen §4 bars, with these concrete settings)

- **K1, Gaussian-twin Monte Carlo of the real grid.**
  - The same refiner on PSD-drawn twin noise at the same 39,321 candidate times.
  - The per-candidate-maximum survival must match an independent Monte Carlo through the same grids: ±15% at
    survival 10⁻², and within a factor of 2 at the deepest populated level.
  - Off-grid twin injections: the mean χ² must stay at or below Allen's mismatch bound.
- **K2, linearity:** 15 caught injections, stratified by Mc, q and SNR (≥ 3 with ρ_opt > 30, one on a gated glitch).
  The shortcut must match real injected strain within 2% on ρ̂_net and χ²_r.
- **F1, must fail:**
  - nulls (ρ_opt = 0), and the wrong-template null (box offset to Mc +5%);
  - passes ≤ x*, the smallest count with P(X > x* | Binomial(n, 1/1440)) ≤ 0.01.
- **F2, must fail: 200 decoys per family at candidate times,** at the median plain-coherent ρ of caught signals:
  - (i) single sine-Gaussians;
  - (ii) glitch trains (3–10 sine-Gaussians within the template span);
  - (iii) power drift (a 10–30% multiplicative modulation over 10–100 s).
  - Each passes ≤ 2% under the primary statistic. (iii) is also reported without v.
- **P1, recovery:**
  - median ρ̂ / ρ_opt ≥ 0.90 in each q stratum, over caught tuning injections;
  - ≥ 90% of caught injections with ρ_opt ≥ 15 pass the twin threshold;
  - the 11–14 band is reported.
- **N1, numerics:** float32 vs float64 match ≥ 0.9999 for a 384-s light template through the identical chain.

## 6. G0 (frozen §6b), evaluated once on the chosen configuration

**Clause 1, headroom:**
- **R_tune** = cascade / Y1-alone chirp distance at 1/day on the tuning set.
- Segment-block bootstrap with 2,000 resamples, re-estimating both thresholds in every resample.
- Pass: **R_tune ≥ 1.15 and CI_lo > 1.00.**

**Clause 2, gates:** all §5 gates pass.

**Clause 3, cost:** the projected fresh stage-2 cost, from T's own measured wall time, must be **≤ 120 h**.

**Outcomes:**
- **Any clause fails:** STOP, fresh segments unopened.
- **G0 passes:** licenses the fresh run (download D, Y0, Y1-bg, Y1-fg, S2) under the frozen §5 verdict, and nothing
  else.

## 7. Sealed prediction (committed with this registration, graded after G0)

- **R_tune point prediction 1.12, 80% interval [0.85, 1.40].**
- **P(G0 passes) ≈ 35%.**
- **Reasoning:**
  - The idealised ceiling at one per minute is 1.89×.
  - Grid mismatch, the χ² and v penalties on signals, and the real-noise tail that χ² and v leave behind should take
    most of it.
  - The plain coherent tail was catastrophic (ρ² ≈ 1,254 at survival 10⁻³, against 33 for the twin).
- **Grading:** HIT if the measured R_tune lies in the interval. The direction of my error is recorded either way.

## 8. Build list before T (each a short run, under 15 min, with its gate; tuning data only)

1. **Driver:** box-major loop (build a grid and its references once, then every candidate of that box: real, twin,
   decoys, foreground). Checkpoint per box. Memory-mapped whitened series. pycbc in a separate process.
2. v via PyCBC (S0).
3. H1 + L1 combination.
4. Twin generator (K1).
5. Decoy generators (F2), and the wrong-template null (F1).
6. A quiet-core MPS re-timing of the engine at full rate. Needed to cost T's slot; the verification could not re-time
   on a contended box.

## 9. Compute (the fleet rule: limits listed beside estimates)

| Step | Estimate | Peak (footprint) | Device | Remaining limits |
|---|---|---|---|---|
| Build + gates (§8) | ~6–10 short runs, < 15 min each | < 2 GB | CPU / MPS | none that end a job |
| T pass, L_max 128 | ~18 h | ≤ 2 GB declared (measured 1.86 GB for one box with pycbc in-process) | MPS + 1 core | none; job_guard pauses on pressure; checkpoint per box |
| T pass, L_max 384 | ~22 h | same | MPS + 1 core | same |
| G0 evaluation | minutes | < 1 GB | CPU | none |

**Total: ~40 h of MPS** plus build time.

- It needs a Bridge slot with the GPU quiet (≥ 60% memory free).
- The two passes are independent, so they can be split across nights. Each resumes from its last finished box.
- No wall-clock or step limit is used anywhere. Correctness gates stop a run only at a box checkpoint.

## 10. What this cannot tell us

- G0 on the tuning set is not a result. The tuning set has been looked at by every earlier test, and the 1.15 bar is
  the only correction for that.
- The verdict belongs to the fresh run.
- Other noise eras, spins and eccentricity are out of scope, as in the frozen §7.
