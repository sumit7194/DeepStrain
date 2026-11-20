# Pre-registration: the two-stage cascade against REAL background (LongChirp)

Written 2026-11-20, before any code for this test exists and before any number is computed. It is committed before
the run. Nothing in sections 2–6 may change after the first number is seen; any change after that is an amendment,
dated and committed, and the pre-amendment result is reported too.

## 1. Question and why it matters

The idealised cascade (`idealised_cascade.py`, result d15bef0) said **ROOM**. Using Y1 at one candidate per minute
plus a perfect coherent second look gives at most about 109 Mpc, against Y1's 57.5 Mpc alone (1/day). That bound
rested on three idealisations:

- (i) a Gaussian χ²₄ stage-2 background;
- (ii) a refiner that always reaches the true template;
- (iii) expected rather than realised SNR.

This test removes (i) and (iii) using real O3a noise, and keeps (ii) as the one stated idealisation. Removing (ii)
needs a real refiner, which is the follow-up this test would license. The question is: **does the cascade's
advantage survive real noise?**

It matters for the same reason the bound did. If it survives, effort on long faint chirps belongs in the second-look
step. If it collapses, the idealised headroom was a Gaussian artefact, and that is worth knowing before anyone builds
a refiner.

## 2. Fixed inputs (no re-tuning)

- **Data:** the 24 LongChirp test segments, H1 and L1. Background pairing is the benchmark's own: H1 of test[i]
  with L1 of test[(i+k) mod 24], k = 0…23. That is 27.3 days, the same livetime as every yardstick.
- **Stage 1:** the committed Y1 submission (`results/y1_bins`, regenerated `submissions/y1`). The stage-1 threshold
  is set at **one candidate per minute** (10.65; 39,321 background candidates). The foreground "caught" set is the
  Y1 injections whose own trigger exceeds it (16.2%, from `idealised_cascade.json`). One per hour (655 candidates)
  is reported as a secondary.
- **Stage-2 statistic:** fully coherent matched filter per detector (complex SNR z_d(t), whitened domain, the
  benchmark's `lowmem_psd` whitening, band [40, 1024] Hz). The network statistic is
  ρ²_net = |z_H(t)|² + max over |δ| ≤ 10 ms of |z_L(t+δ)|². In Gaussian noise, at a single time with δ fixed, this
  is χ² with 4 degrees of freedom. Templates are cut at t_c (the 5-s tail lesson).
- **Gain is measured against** Y1 alone at 57.47 Mpc (1/day) and 51.68 Mpc (1/week).

## 3. Measurements

**M1 — conditional real-noise tail at stage-1 candidate times.**
- **Template sample:** 64 templates spanning the population: 32 equal-mass at log-spaced chirp masses across the Y1
  bank, plus 32 at mass ratio ≈ 0.3, all cut at t_c.
- **Trials:** for every background slide k and segment i, evaluate ρ²_net in 25-ms bins over ±1 s around each Y1
  stage-1 background candidate. Each bin takes its maximum over samples, which is conservative. That gives 64
  templates × 80 bins ≈ 5,120 trials per candidate, and ≈ 2×10⁸ trials over the 39,321 candidates.
- **Output:** the per-trial survival function S_real(x) = P(ρ²_net > x).
- **Why at candidate times:** stage 1 selects noise that already looks signal-like, so the tail must be measured
  there, not at random times. The unconditional tail (all times) is recorded too, and the ratio of conditional to
  unconditional is reported.

**M2 — Gaussian twin of the same data, the same way.** Each whitened series is phase-randomised: the same amplitude
spectrum with uniformly random phases, which gives Gaussian, glitch-free noise with an identical PSD. Run M1 on it,
using the same candidate times and bins. The ratio R(x) = S_real(x) / S_twin(x) is the heavy-tail factor and does
not depend on bin correlation.

**M3 — realised foreground SNR with the true template.**
- **Shortcut:** whitening and filtering are linear, so for each caught injection and detector the realised complex
  SNR is z_d = ρ_opt,d + z_noise,d, where z_noise,d is the true template filtered against the noise-only series at
  the true arrival time. ρ_opt,d comes from `oracle_snr.csv` (per detector), and is real and positive by construction
  at the true template and time.
- **Statistic:** ρ²_net = |z_H|² + |z_L|². No signal is ever added to strain, which also avoids pycbc's
  memory-retention problem for injection building.

**Threshold and distance.**
- **Threshold:** ρ*²(N_eff) solves rate1 × N_eff × S_stage2(x) = f, where rate1 = 39,321 / T_bg,
  f ∈ {1/day, 1/week} and N_eff ∈ {1e4, 1e6, 1e8} trials per candidate.
- **Tail extrapolation:** S_stage2 uses the measured conditional S_real wherever the needed probability is within
  the measured range, which means at least 100 exceedances. Beyond that it uses a fit of log S_real against x over
  the top 0.1% of trials, and the fitted slope is reported next to the Gaussian value of −1/2. If the needed depth is
  more than 100× beyond the measured range, that cell is reported as extrapolated and flagged.
- **Distance:** an injection is found if it is caught at stage 1 and ρ_net,realised ≥ ρ*. Distance uses
  `evaluate.py`'s chirp-volume estimator.

## 4. Validity gates (all must pass, or no verdict is drawn and the failure is reported)

- **G1, statistic calibration.** On the Gaussian twin, at single samples (no binning, fixed δ = 0, 10⁷ random
  (time, template) draws), the survival of ρ²_net must match χ²₄ within ±20% at P = 10⁻³ and within a factor of 2
  at P = 10⁻⁵. This catches errors in normalisation and whitening.
- **G2, MUST-FAIL control (null injection).** Rerun M3 with ρ_opt set to 0: the same injection times, the same true
  templates, noise only. These "injections" must **not** separate. The fraction passing ρ* at N_eff = 1e6, 1/day,
  must be ≤ 1%, and its distribution must agree with background trials (two-sample KS, p > 0.01). If the null
  injections pass, the pipeline is seeing something other than signal, and the test stops.
- **G3, linearity check.** For 5 injections, build the real injected strain (child process), whiten and filter it
  with the true template. The realised |z_d| must match the M3 shortcut within 2%.
- **G4, positive control.** At least 90% of caught injections with ρ_opt,net ≥ 15 must pass ρ* at N_eff = 1e6,
  1/day. If they do not, the threshold machinery is broken, not the idea.

## 5. Verdict rule (on Y1 at one per minute, both FARs)

- **SURVIVES** if the cascade distance ≥ 1.10 × Y1-alone at N_eff = 1e6, at both 1/day and 1/week.
- **SHRINKS** if it is < 1.10 × at N_eff = 1e6 but ≥ 1.10 × at N_eff = 1e4.
- **DIES** if it is < 1.10 × at N_eff = 1e4.

The cascade distance is also reported as a fraction of the idealised bound (109.3 / 108.5 Mpc). Under any verdict,
the remaining idealisation (ii), perfect template recovery, is stated beside the number. SURVIVES licenses building
a real refiner. It is not a sensitivity claim.

## 6. What this cannot tell us (stated in advance)

- Whether any real refiner reaches the true template (idealisation ii). N_eff stays a parameter, not a measurement.
- Behaviour beyond the measured tail depth (it is extrapolated and flagged).
- Other noise eras or detectors.
- Whether learned refinement beats classical coherent follow-up (the prior-art question; see
  `research_notes/Prior art checks for two ideas/phase_coherent_refinement.md`).

## 7. Cost estimate (to be measured on one segment before the slot request)

- **Expected cost:** about 48 series × (load and whiten ≈ 5 s + 64 templates × one 2²⁴ GPU correlation ≈ 0.05 s)
  ≈ 7 min for real noise, and the same again for the twin. M3 is about 1,344 template generations in child processes
  plus 2,688 correlations, ≈ 10 min. G3 is about 5 injected segments. **Total roughly 30–45 min.**
- **Expected peak:** about 3 GB (one whitened series and its spectrum on the GPU, plus the 2²⁴ template spectrum).
- **Disk:** binned per-series arrays (float16, 25 ms) ≈ 2 GB for real plus twin, deleted after the survival functions
  are computed.

The slot request will quote one segment's measured time and peak, not this estimate.
