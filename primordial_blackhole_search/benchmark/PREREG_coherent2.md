# Pre-registration (FROZEN 2026-11-25): a consistency-tested coherent second look on FRESH O3a segments

**FROZEN 2026-11-25** on Sumit's go (relayed by the Bridge), together with the staged gate G0 (§6b).

- Nothing has been run, and no fresh data has been downloaded.
- The tuning values are fixed later by amendment T (§6), committed before any fresh strain is opened.
- Any other change after this commit is a dated amendment, and the pre-amendment result is reported too.

**Lineage:**
- `PLAN_next_2026-11-23.md`, revised by the outside review (`reports/PLAN_next_2026-11-23_review.md`, accepted points
  marked ✔ below).
- The idealised cascade (ROOM, upper bound) and the real-background cascade `PREREG_realbg_cascade.md` (NO VERDICT; a
  plain coherent SNR has a heavy real-noise tail at candidate times, ρ² ≈ 1,254 at survival 10⁻³ against 33 for a
  Gaussian twin).

## 1. Question, and why it matters

**Question:** does a second look that is coherent, χ²-tested and PSD-variation-normalised, with a **real** local
template search, beat Y1 alone on LongChirp in real O3a noise, on segments nobody has looked at?

**What each outcome buys:**
- **SURVIVES:** LongChirp gains a realisable two-stage yardstick, Y3, between Y1 (~1/3 of the oracle) and the oracle.
- **DIES:** a measured statement that laptop-scale coherent follow-up of minutes-long chirps is limited by real-noise
  non-stationarity, even with the standard consistency tests.

**Not new:**
- the cascade architecture (CW hierarchical follow-up; Soni & Nitz 2024, arXiv:2409.11317);
- χ² (Allen 2005) and v_s (Mozzon et al. 2020, arXiv:2002.09407);
- production O4 subsolar searches already do fully coherent χ² filtering at scale (arXiv:2602.12115, 2605.05444,
  2610.19329, 2610.20276).

**What it adds:** an open, per-injection measurement on a public benchmark. Modest audience, and stated as such.

## 2. Data

- **Tuning set** (allowed to be looked at; already contaminated by earlier tests): the 24 original LongChirp test
  segments and their 1,344 injections.
  - Y1's per-segment bins and per-injection foreground triggers already exist (`results/y1_bins`), so tuning needs no
    new Y1 run.
- **Fresh validation set ✔:** 24 new O3a H1∩L1 coincident 4096-s segments.
  - **Selection rule** (deterministic, the same code path as `fetch.py discover`):
    - GWOSC H1_DATA and L1_DATA both;
    - ≥ 64 s from any GWTC-2.1 confident or marginal event;
    - GPS in [1249000000, 1253977218];
    - **excluding every GPS already used anywhere in this repo**: benchmark test and dev, `data/manifest.json`, the
      Build C and N5 coincident lists, and the device_stream O3a pool;
    - from the remaining eligible list, 24 picked evenly spaced.
  - If fewer than 24 are eligible, extend backwards into O3a after DEV_FROM, using the same rule, and record it.
- **Fresh injections:** the same generator and distribution as LongChirp (TaylorF2, m₁,₂ log-uniform [0.2, 1.0] M☉,
  d_c uniform [5, 500] Mpc, K = 8 realisations, ≥ 480 s spacing), with a **new seed** declared at freeze.
- **Blinding:** fresh strain is fetched and hashed only. No plot, statistic or spectrum of it is computed until T is
  committed.
- **Background:** the benchmark's own construction on the fresh set: zero lag plus 23 cross-segment slides, ≈ 27 days.

## 3. Method

1. **Stage 1** = Y1, run unchanged on the fresh set: 3,235-template semi-coherent bank, newSNR, H1×L1.
   - The candidate rate r₁ is declared in §5 (one per hour, or one per minute).
   - Candidates are **clustered at ≥ 2 s** ✔; overlapping ±1-s search windows are merged.
2. **Stage-2 statistic, per detector:**
   - **Band:** f_low = **50 Hz ✔**. The injections start at 50 Hz, and so does Y0, so a lower f_low would add only
     noise.
   - **Template length:** mass-dependent L = min(the template's time above 50 Hz, L_max) ✔, with L_max tuned.
     Heavy templates (24 s at 1+1 M☉) are naturally short.
   - **Complex coherent SNR z** from 2²¹-sample snippets around each candidate. Same lowmem_psd whitening and PyCBC
     2.10.0-default autogating as in the real-background test.
   - **Allen frequency-domain χ² ✔** with p equal-SNR bins (p tuned), computed only at the top peaks of |z| (as in
     PyCBC; Usman et al. 2016), then newSNR reweighting.
   - **v_s ✔:** the PSD-variation statistic of Mozzon et al. 2020 enters the primary statistic. Its exact form and
     parameters are read from the PyCBC 2.10.0 source (`pycbc.psd.variation` and its ranking use) and recorded in T.
     Nothing is recited from memory.
   - **Network value:** ρ̂²_net = ρ̂_H² + max over |δ| ≤ 10 ms of ρ̂_L².
3. **Real local refiner ✔** (this replaces the "N_eff" idealisation):
   - **Search box:** around each candidate's Y1 best template, a grid in (chirp mass, mass ratio q ∈ [0.2, 1]) and
     merger time (±1 s, full rate).
   - **Grid spacing:** set for minimal match ≥ 0.97 at length L, measured on tuning data.
   - **Chirp-mass box half-width:** the 99th percentile of the **stage-1 chirp-mass bias** ✔ of caught tuning
     injections, stratified by q. It is read from Y1's stored winning-template indices.
   - **Score:** the candidate's stage-2 value is the maximum over the box. Trials are counted by construction.
   - **Count:** N_tmpl per candidate is measured and reported per mass bin.
4. **Threshold:** at FAR f, the per-candidate maximum exceeded at rate f over the fresh background, **measured
   directly** with no tail extrapolation.
   - At r₁ = 1/hour and f = 1/day, that is a per-candidate probability of 1/24, i.e. ≈ 27 background events.
   - **1/week is reported, not gated ✔** (≈ 4 events).
5. **Foreground:** caught fresh injections are scored with the linearity shortcut, z = ⟨T, h_inj⟩ + ⟨T, n⟩ piece by
   piece, so no strain is rebuilt. K2 validates it.
6. **Uncertainty: segment-block bootstrap ✔**, 2,000 resamples.
   - Resample the 24 fresh segments with replacement.
   - Rebuild both the background (slide pairs whose H1 and L1 segments are both drawn, with multiplicity) and the
     injection set from the drawn segments.
   - **Re-estimate both thresholds** (Y1-alone and cascade) in every resample.
7. **Ablation ✔** (reported, not deciding): plain coherent ρ / + χ² newSNR / + χ² + v_s (primary).

## 4. Gates (all must pass on the fresh set, or no verdict)

- **K1, known answer by Monte Carlo of the real grid ✔.**
  - The identical refiner, run on a PSD-drawn **Gaussian twin** at the same candidate times, must give a
    per-candidate-maximum survival function matching a direct Monte Carlo of Gaussian noise through the same grid:
    within ±15% at survival 10⁻², and within a factor of 2 at the deepest populated level.
  - **Off-grid injections** in the twin: the χ² mean must stay at or below Allen's mismatch upper bound for their
    measured mismatch.
- **K2, linearity ✔:** 10–20 injections, stratified by chirp mass, q and SNR, including at least 3 with ρ_opt > 30 and
  one placed on a gated glitch.
  - The shortcut must match real injected strain within 2% on both ρ̂_net and χ²/dof.
- **F1, must fail ✔:**
  - **(a) Null injections:** ρ_opt set to 0, at the same times and true templates.
  - **(b) Wrong-template null:** real injections scored with a box offset to chirp mass +5%, which should find
    nothing.
  - **Bar for each:** passes ≤ x*, where x* is the smallest count with P(X > x* | Binomial(n, f/r₁)) ≤ 0.01. That is
    a bound consistent with the FAR, replacing the old loose 1%.
- **F2, must fail: decoys ✔.** 200 each, at candidate times, in real noise, at the same plain-coherent ρ as caught
  signals:
  - (i) single sine-Gaussians;
  - (ii) **glitch trains** (3–10 sine-Gaussians at random separations within the template span);
  - (iii) **power-drift decoys** (a slow multiplicative amplitude modulation of the noise, 10–30% over 10–100 s).
  - **Bar:** each family passes at ≤ 2% under the primary statistic. Family (iii) is also reported without v_s; it is
    expected to pass more there, and that is what shows v_s working.
- **P1, positive control ✔:**
  - Among caught fresh injections, the **median ρ̂/ρ_opt ≥ 0.90 in each q stratum** (q < 0.5, q ≥ 0.5).
  - ≥ 90% of caught injections with ρ_opt ≥ 15 pass the **Gaussian-twin** threshold.
  - The near-threshold band ρ_opt 11–14 is reported.
- **N1, numerics:** float32 vs float64 match ≥ 0.9999 for a 384-s light template through the identical chain.

## 5. Verdict (fresh set, f = 1/day)

**The statistic:** R = cascade sensitive chirp distance / Y1-alone distance, on identical fresh injections, with a
segment-block bootstrap 90% CI.

**Verdict bands:**
- **SURVIVES** if R ≥ 1.10 and CI_lo > 1.00. This licenses Y3 as a benchmark yardstick (its publication is Sumit's
  call).
- **DIES** if R < 1.00, or CI_hi < 1.10.
- **INCONCLUSIVE** otherwise.

R is also reported against the idealised bound and Y0.

**Stage-1 rate r₁, decided by measured cost alone, after tuning and before any fresh number (recorded in T):**
- r₁ = **one per minute** (39k candidates) if tuning's measured N_tmpl puts the confirmatory stage 2 at ≤ 96 h;
- **otherwise one per hour** (655 candidates).

**Warning, stated now.** The idealised gain at one per hour was only +17% at 1/day, against +90% at one per minute.
So a one-per-hour run has little room above the 1.10 bar, even before real-noise losses.

**Prediction, written now:**
- at one per hour: DIES 55%, INCONCLUSIVE 30%, SURVIVES 15%;
- at one per minute: SURVIVES 30%.

The reason: test 2's tail came from sub-threshold transients over minutes-long templates. χ² and v_s are built for
exactly those, but our refiner will not reach the true template every time.

## 6. Tuning amendment T (on the tuning set only; written and committed before any fresh strain is opened)

Fixes and records:
- L_max ∈ {128, 256, 384} s;
- p ∈ {4, 8, 16};
- the grid spacing and box half-widths per mass bin;
- the measured N_tmpl, and hence r₁ (§5);
- the v_s formula and parameters as read from source;
- the decoy amplitude calibration;
- the fresh-injection seed.

**Tuning objective:** maximise tuning-set cascade distance at 1/day, subject to the tuning-set versions of F1, F2 and
P1 passing. Every candidate configuration tried is logged, so that the number of configurations explored is on the
record.

## 6b. Staged go: the go/no-go gate G0, evaluated on the TUNING set at the end of T (added 2026-11-25 at Sumit's request)

The multi-day fresh run (download D, Y0, Y1-bg, Y1-fg, S2) starts **only if all of the following hold** on the tuning
set (the old 24 segments) with the frozen T configuration:

1. **Projected gain.** R_tune = cascade / Y1-alone sensitive chirp distance at 1/day, with a segment-block bootstrap
   (2,000 resamples, both thresholds re-estimated), must satisfy **R_tune ≥ 1.15 and CI_lo > 1.00**.
   - The bar is 1.15, stricter than the fresh verdict's 1.10, because T *chooses* the configuration that maximises
     R_tune, which biases it upward.
   - The number of configurations tried is reported beside it.
2. **The tuning-set versions of K1, K2, F1, F2, P1 and N1 all pass**, with the §4 bars.
3. **Cost.** The projected stage-2 cost on the fresh set at the chosen r₁, from the measured per-template time and
   N_tmpl, must be **≤ 120 h**.

**If any fails: STOP and report.** The fresh segments stay unfetched and unopened, and the result is recorded as
"G0 not passed", with the failing clause and its numbers. G0 passing licenses the fresh run only; it is not a result.

## 7. What this cannot tell us

- Other noise eras or detectors.
- Spinning or eccentric signals.
- Whether a learned refiner beats this classical one.
- Behaviour below the measured background depth.
- Anything about time-slide independence beyond the repo's known limit: slides add livetime, not independent
  samples.

## 8. Compute plan (for the Bridge to slot; nothing runs before the freeze and the Bridge's slot)

| Step | What | Basis | Wall time | Peak RAM | Cores / GPU |
|---|---|---|---|---|---|
| 0 | Free scratch: delete this session's 1.7 GB pSEOB scratch copy (re-downloadable). Ask Sumit first. | measured size | minutes | — | — |
| T | Tuning on the old 24 segments: bias from stored Y1 indices; spacing; L_max/p/v_s choices; tuning-set F1/F2/P1 | estimate | 6–12 h | < 2 GB | 1–2 CPU + MPS |
| D | **Download** 48 detector-segment files from GWOSC, one at a time | the original fetch: ≈ 3 GB for 64 files | **≈ 2.3 GB download, 3.1 GB stored** (64 MB npy each); 1–3 h at normal GWOSC speed | < 0.5 GB | network |
| Y0 | Oracle SNR for the fresh injections (lowmem_psd) | the Y0 run peaked 3.53 GB *before* lowmem_psd | 1–2 h | < 2 GB | 1–2 CPU |
| Y1-bg | Y1 background on 24 fresh segments | **measured** 68–70 min per segment, peak 2.56 GB | **≈ 28 h** | 2.6 GB | 2 CPU + MPS |
| Y1-fg | Y1 foreground, child processes of 4 segments | **measured** ≈ 4.5 h, peak 2.7–3.2 GB | ≈ 4.5 h | 3.2 GB | 2 CPU + MPS |
| S2 | Stage 2 on fresh candidates: background, Gaussian twin, F1, F2 (600 decoys), P1 | estimate: snippet correlation ~10 ms per template per detector (to be measured in T) | **one per hour: ≈ 12 h at N_tmpl 10³, ≈ 115 h at 10⁴; one per minute: ≤ 96 h by rule** | < 2 GB (snippets only) | 1–2 CPU + MPS |
| B | Bootstrap and verdict from stored per-candidate maxima | — | minutes | < 1 GB | 1 CPU |

**Total:**
- ≈ **2.5 days** of Mac time at the optimistic end (one per hour, N_tmpl ≈ 10³);
- ≈ **7 days** at the pessimistic end (N_tmpl ≈ 10⁴).

Every long step is nohup-detached, checkpointed per segment or per candidate block, and run under
`pressure_guard.sh` (TERM on memory pressure or swap > 1.5 GB), at nice 15 with ≤ 2 cores.

**Disk:** ~11 GB free now. Stored fresh strain (3.1 GB), plus Y1 bins and stage-2 maxima (< 0.5 GB), stays above the
5 GB floor after step 0. Raw GWOSC files and astropy's download cache are purged after each conversion (fetch.py
already does this).

**GWOSC etiquette:** one transfer at a time, with backoff of 10, 30 and 60 min.

**GPU:** Y1 and S2 use MPS. Per the box rules, they run only when ≥ 60% of memory is free.

## 9. Tuning log, part 1 (2026-11-25): cost and stage-1 rate (tuning set only; no fresh data touched)

**T1, stage-1 chirp-mass bias** (caught tuning injections, from Y1's stored winning-template indices; `results/coh2/T1_mc_bias.json`):

| | n | p90 | p99 |
|---|---|---|---|
| all | 220 | 0.36% | 2.15% |
| q < 0.5 | 84 | 0.28% | 2.20% |
| q ≥ 0.5 | 136 | 0.66% | 1.90% |

⇒ By the frozen rule, the box half-width is ~2%.

**T-cost** (`coh2_cost.py`, `results/coh2/T_cost.json`; 301 s, 1 core, nice 12, peak 1.31 GB):
- Coherent (ln Mc, q) metric, min match 0.97, f_low 50 Hz, real O3a H1 PSD.
- Sanity: self-match 1.000; a 0.01% Mc offset costs 17.5% at L = 384 s.
- N_tmpl per candidate, by chirp mass 0.18 / 0.26 / 0.35 / 0.50 / 0.70 / 0.86:
  - L_max = 128: 212 / 5,440 / 12,067 / 2,497 / 457 / 14;
  - L_max = 384: 465 / 7,669 / 12,067 / 2,497 / 457 / 14.
- **Per-template correlation: 3.36 ms** (MPS, 2²¹ samples, warm median).

**Projected stage-2 total** (background, twin, decoys and foreground):

| r₁ | L_max 128 / 256 / 384 |
|---|---|
| one per hour (655 candidates) | **17.5 / 20.6 / 20.9 h** |
| one per minute (39,321 candidates) | **584 / 694 / 703 h** |

**⇒ By the §5 rule** (one per minute only if ≤ 96 h), **r₁ = one per hour is FORCED**. G0 clause 3 (≤ 120 h) is
met at one per hour.

**Consequence, stated before any tuning-set gain is measured.** At one per hour, the idealised cascade's *upper bound*
at 1/day is 67.2 / 57.47 = **1.169**: Gaussian stage-2 noise, the true template every time, and expected SNR. G0
clause 1 needs R_tune ≥ 1.15. So the whole real-world loss budget (grid mismatch up to 3% in SNR, real-noise tails,
χ² and v_s penalties) must fit inside **1.6%** of distance. I expect G0 clause 1 to fail.
