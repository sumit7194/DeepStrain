# Pre-registration: is the GWTC-5 pSEOBNR damping-time tension sensitive to nuisance priors?

**Written 2026-11-24, before any number in §3–5 was computed.** It is committed before the analysis script exists.
Any change after the first result is a dated amendment, and the pre-amendment result is reported too.

## 1. Question and why it matters

- **The tension.** LVK's pSEOBNR remnant test puts GR at the edge of the combined (δf̂₂₂₀, δτ̂₂₂₀) posterior:
  - GWTC-4.0: 98.6% joint;
  - GWTC-5.0: **98.1% joint (Q2D), 98.7% hierarchical (μ_δτ̂)** (arXiv:2607.19293, Table 9).
  - The shift is toward longer damping times.
- **The proposed mechanism.** Kyutoku (arXiv:2607.05486) shows that at finite SNR the damping time is biased *long*
  through prior gradients and likelihood geometry, and that this could create a false catalog-level violation.
- **What LVK already did.** Their GR mock catalogs (2603.19021) already contain whatever bias their priors create.
  Those mocks reproduce the qualitative shift, but give quantiles this large in only 0.2–3.5% of catalogs.
- **The question here:** how much does the *published* combined result move under defensible alternative priors?
  - "Not at all" means this channel cannot be the explanation, which is useful to know.
  - "A lot" means the 98% is a prior-choice statement, which is worth stating publicly.

## 2. Fixed inputs

- **Data:** LVK GWTC-5.0 TGR release, Zenodo 21454847, `pSEOB.tar.gz` (md5 2a3f8ec08a52a9c277c45859ec547e47).
  - Per-event PESummary metafiles.
  - The 31-event list and per-event analysis labels are taken **exactly** from the release's
    `pSEOBNR_main_figure.ipynb`: events 0–16 use `pSEOBNRv5PHM-reweighted`, GW250114 uses `pSEOBNRv5PHM`, and the
    rest use `pseob-rd`.
- **Per-event priors (as stored in the files):**
  - δf̂, δτ̂ ~ U(−0.8, 2.0);
  - d_L ~ UniformSourceFrame(100, 10⁴ Mpc), with the stored cosmology;
  - chirp mass and mass ratio uniform in components;
  - spin magnitudes U(0, 0.99), isotropic.
- **Combination methods (LVK's own, from the release's `utils_o3b.py`):**
  - **Joint:** product of per-event 2-D Gaussian KDEs of (δf̂, δτ̂), with flat priors that cancel, on a grid over the
    prior box. Q2D = fraction of joint-posterior mass at density above the density at (0, 0).
  - **Hierarchical (secondary):** a 2-D Gaussian population (μ_f, μ_τ, σ_f, σ_τ, ρ). Q1D = the one-sided quantile of
    μ_δτ̂ at 0, with the exact definition fixed by K1c below.

## 3. Variants (importance reweighting; weight = π_new/π_old evaluated on each posterior sample)

**Physical alternatives** (these decide the verdict):
- **V1:** d_L uniform in d_L on [100, 10⁴] Mpc.
- **V3:** spin magnitudes "volumetric", π ∝ a₁² a₂² (orientations unchanged).
- **V4:** masses flat in (detector-frame chirp mass, q) instead of flat in components.
- **V5:** the joint posterior with a prior flat in the damping *rate* γ ∝ 1/τ. This is Kyutoku's suggested
  parameterisation, applied **once** to the joint (factor (1 + δτ̂)⁻²), with Q2D computed in (δf̂, δγ̂) coordinates.

**Extreme, reported but not deciding:**
- **V2:** d_L ∝ d_L⁻², i.e. flat in amplitude ∝ 1/d_L, the n_A = 0 analogue.

**Descriptive only** (no verdict): per-event Corr(d_L, δτ̂), and δτ̂ median vs network matched-filter SNR. Full-signal
SNR is not ringdown SNR.

## 4. Gates (all must pass, or no verdict)

- **K1a (known answer, instrument):** Q2D computed from LVK's released joint samples (`joint_posterior_results`,
  treated as paired draws) must equal 98.1% ± 1.0 point.
- **K1b (known answer, pipeline):** our own baseline joint from the 31 per-event files must give Q2D within ±1.5 points
  of 98.1%, and a δτ̂ marginal median within ±0.01 of the released joint samples.
- **K1c (known answer, secondary):**
  - The released hierarchical μ_δτ̂ samples must reproduce 98.7% ± 0.5 under one of the two definitions
    P(μ > 0) / HPD. That fixes the definition.
  - Our own hierarchical baseline must reproduce it within ±2 points.
  - If K1c fails, or our hierarchical takes more than ~5 min of CPU, the hierarchical result is dropped and that is
    reported. The joint stays primary.
- **ESS:** Kish effective sample size ≥ 200 for every event under every variant. A variant failing this is reported as
  unreliable and does not enter the verdict.
- **F1 (must NOT move):** reweighting by π(ψ) ∝ 1 + 0.5 cos 4ψ, where ψ is the polarisation angle and has no physical
  link to δτ̂. It must move Q2D by ≤ 2 σ_MC.
- **P1 (must move):** a deliberately wrong per-event factor (1 + δτ̂)⁻¹ on all 31 events must lower Q2D by ≥ 5 points.
  This shows the pipeline can respond.
- **σ_MC:** the standard deviation of baseline Q2D over 20 bootstrap resamplings of each event's posterior samples.

## 5. Verdict (on joint Q2D, GWTC-5.0, 31 events; physical variants V1, V3, V4, V5)

- **ROBUST:** every physical variant moves Q2D by < 3 points.
- **PRIOR-SENSITIVE:** any physical variant puts Q2D below 90%, i.e. GR inside the 90% region.
- **MODEST:** otherwise.

V2 is reported beside the verdict. If only V2 moves Q2D, that is said explicitly.

**Prediction, written now:** ROBUST (my probability ~60%). pSEOBNR pins the amplitude through the inspiral, so I expect
|Corr(d_L, δτ̂)| < 0.2 for most events. V5 is the most likely to move Q2D, by 1–4 points, because density-contour
quantiles are not invariant under reparameterisation.

## 6. What this cannot tell us

- The likelihood-geometry half of Kyutoku's bias: it lives in the likelihood, so reweighting cannot touch it.
- Waveform systematics.
- Non-Gaussian noise.
- Whether LVK's mock catalogs are representative.
- Whether any alternative prior is *more correct*. This is a sensitivity analysis, not a re-analysis.

## 7. Compute

Short scripts at nice 12: reading 31 HDF5 files (~1 GB extracted, scratchpad only) and numpy/scipy KDEs on a grid. Each
step is expected to take seconds to a few minutes. Anything longer stops and is reported to the Bridge first.

## 8. Amendment A1 (2026-11-24, after a structural crash in data loading, before any variant or baseline number existed)

**What happened.** The first run stopped while loading files.
- **2 of the 31 events, GW240705_053215 and GW250119_025138 (both labelled `pseob-rd`), store NO prior:**
  `priors/analytic` is absent and `priors/samples` is an empty dataset.
- Their config shows the same `default-prior = BBHPriorDict` as the 29 events that do document a prior.
- The run printed only K1a (LVK's released joint, a published known answer) before crashing. No baseline, bootstrap
  or variant number existed.

**Changes:**
1. **For these two events, assume the same prior family as the 29 documented events.** That family is asserted
   identical across all 29 by the script: UniformSourceFrame d_L with H0 = 67.9, Om0 = 0.3065; UniformInComponents
   chirp mass. Reweighting uses only the prior *shape* within the support, so unknown bounds do not matter as long as
   the posterior lies inside them.
2. **Robustness addition:** the verdict quantities are also reported with these two events excluded (29 events). If
   the 29-event result changes the verdict, both are reported and the 31-event verdict is labelled conditional on
   assumption 1.
3. **Hierarchical (secondary) is not attempted in this pass.** Only the joint (primary) is run; K1c is reported as
   "not attempted". This is a scope reduction, declared before any number.

## 9. Run 1 result (2026-11-24): NO VERDICT by the frozen rule

Result file: `results/pseob_prior_run1_NOVERDICT.json`. The O4b files name their cosmology 'Planck15_LAL', which in
bilby is the same H0 = 67.90 / Om0 = 0.3065 as the O1–O4a files. The family check was widened to accept both
spellings; no prior changed.

**Gates:**
- K1a **PASS**: 98.125%.
- **K1b FAIL**: baseline 95.46% vs 98.1 ± 1.5; δτ̂ median 0.060 vs 0.070 ± 0.01.
- **F1 FAIL**: the ψ null moved Q2D by +2.36 points, against 2 σ_MC = 1.53.
- P1 PASS: −7.5 points.
- V2 ESS 183 < 200.

**Variant numbers, read only as instrument diagnostics, not results:** V1 97.1, V2 98.7, V3 93.6, V4 96.6, V5 95.5
(31 events). The 29-event subset: base 96.7, V3 84.7. Both are within the run's demonstrated noise (F1), so nothing
is resolved.

**Exploratory diagnosis, after run 1:**
- The same baseline using **all** posterior samples instead of 3,000 resampled draws per event gives Q2D 98.38% and a
  δτ̂ median 0.062. That is inside both K1b tolerances.
- The 3,000-draw subsample was an implementation choice not in §2. It widens every KDE through Scott's n^(−1/6)
  bandwidth, and adds resampling noise. That is consistent with both failures.

## 10. Amendment A2 (2026-11-24, after run 1; NOT blind, because run-1 variant numbers were seen)

**Changes, dictated by the failed known-answer and null gates, not by any variant outcome:**
1. **All posterior samples, weights entering scipy `gaussian_kde(weights=…)` directly** (deterministic). This
   replaces weighted resampling to 3,000 draws.
2. **σ_MC is unchanged in definition:** 20 bootstrap resamplings of each event's posterior samples.

Everything else in §2–5 is unchanged: gates, bars, variants and the verdict rule. The A2 verdict is reported as
"post-amendment, non-blind" beside run 1's NO VERDICT. Both results are kept.

**Cost:** ~40 s per joint × ~33 joints ≈ 20–25 min on one core at nice 12. That exceeds "a few minutes", so it waits
for a slot from the Bridge.
