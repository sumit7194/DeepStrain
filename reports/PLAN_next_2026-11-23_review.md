# Verification of the Gemini "Neon" review of PLAN_next_2026-11-23 (DeepStrain, 2026-11-24)

**Inputs:**
- the Bridge's condensation (`~/Github/neon_work/deepstrain/T56_answer.md`);
- citations checked against the arXiv API directly (titles, dates, abstracts).

Abstract-level checks only, unless stated. Nothing was run.

## 1. Citations

| Cited as | Found | Says what Neon claims? |
|---|---|---|
| Mozzon et al. 2020, v_s | arXiv:2002.09407, *Dynamic Normalization for CBC Searches in Non-Stationary Noise* | **Yes.** Statistic tracks broadband noise variation over tens of seconds; re-ranking O2 PyCBC triggers gave +5% volume for low-mass binaries. |
| Soni & Nitz 2024 | arXiv:2409.11317, *Hierarchical searches for subsolar-mass binaries and the 3G era* | **Exists.** Two-stage subsolar search; f_low 35 Hz gives +6% SNR and +10–20% volume at 2.5× lower cost. The abstract says nothing about sub-segment slides or ≥1–5 yr backgrounds, so that attribution is **unverified**. |
| "Soni et al. 2024", conditioning | Nearest match: Soni, Dhurandhar & Mitra, arXiv:2309.00019 (posted 2023) | **Plausibly.** Background for two-stage searches from time-shifted triggers with tail extrapolation; matches PyCBC VT at IFAR 10 yr. The year differs (journal year?). |
| Usman et al. 2016 | arXiv:1508.02357, *The PyCBC search…* (CQG 2016) | Exists. "χ² only at peaks above threshold" is standard PyCBC practice, but **not confirmed in the body**. |
| Allen 2005 | gr-qc/0405045, PRD 71, 062001 | **Yes.** The frequency-bin χ²; the abstract itself gives mismatch upper bounds on E[χ²] and says the test "works well only for broad-band signals". |
| Nitz 2018 | arXiv:1709.08974, CQG 35, 035016 | **Yes.** Sine-Gaussian test for blip glitches vs short, high-mass (>60 M☉) templates. |
| Kyutoku 2026 | arXiv:2607.05486 | **Yes** (finite-SNR bias for damped oscillations; catalog-level warning). |
| arXiv:2603.19021 | *GWTC-4.0: Tests of GR III. Tests of the Remnants* | **Yes.** |

**Bycatch, relevant to our "beyond published" paragraph.** Two new O4 subsolar searches are on arXiv since our last
sweep, titles seen, abstracts not read:
- arXiv:2610.19329, O4a+O4b;
- arXiv:2610.20276, LVK O4b.

They shrink LongChirp's novelty margin further.

## 2. Technical claims, against our own numbers and code

| Claim | Verdict | Why |
|---|---|---|
| F1's 1% ceiling vs the calibrated 0.069% | **TRUE** (accept) | At 1/day the per-candidate pass probability is 1/1440 ≈ 0.069%. With ~1,344 nulls the expected count is ~0.9, and 1% (≈13) catches only gross leakage. "Cannot fail" is overstated: it still fails on the bug class it was built for. Fix: a binomial bound (≤ 4 passes; false-fail P ≈ 0.002 if calibrated, Poisson λ = 0.93) plus a wrong-template null. |
| p× χ² cost missing from S3 | **TRUE** (accept) | A time-piece χ² at every sample needs p correlations per template, and S3 counted one. Fix: compute ρ over the ±1-s window, then χ² only at the top peaks (dot products, no FFTs). |
| float32 phase at L = 384 s | **FALSE premise** (reject; keep a check) | We never store the phase in float32. pycbc generates waveform samples in float64 and we cast the samples, so the per-sample error is ~6e-8 relative with no accumulation. Y1's float32 2²⁴ correlations agree with float64 to ~1e-6. A one-line 384-s float32/float64 match check is worth adding anyway. |
| Stage-1 Mc bias for unequal-mass injections | **PLAUSIBLE, unmeasured** (accept) | The Y1 bank is equal-mass, and q enters the phase at 1PN, so the best equal-mass match shifts Mc. Whether it exceeds the plan's ±0.1% box is unknown. Y1's foreground stores the winning template index per injection, so it is **measurable from existing dev output without a new run**. |
| N_tmpl 3–10× too low | **PLAUSIBLE** (accept) | Our own bank-dephasing number (±0.01% Mc → −28% SNR on a full-length light chirp) implies ~3e-5 Mc spacing for 0.97 match. Across a ±0.1% box that is ~70 Mc points times tens of correlated q points, i.e. ~10³–10⁴ at the light end. My 10²–10³ was a guess and was low there; heavy templates are much shorter and cheaper. |
| S0 as a single-segment kill switch | **TRUE** (accept) | One segment, plus 64 fixed templates extrapolated to 10³ correlated trials, is the N_eff idealisation again. Fix: a diagnostic over 2–4 dev segments with the real neighbourhood. |
| "L > 64 s adds nothing for heavy templates unless f_low drops" | **TRUE** | 1+1 M☉ spends 24 s above 50 Hz (Bridge checked). The plan also omitted f_low (test 2 used 40 Hz) and must state it. |

## 3. The other criticisms

**Accepted:**
- **v_s (PSD-variation normalisation) in the primary statistic**, with an ablation (ρ / newSNR / newSNR + v_s).
  Our own probe found the CNN's false alarms are short-term power drift, which is exactly v_s's target.
- **Segment-block bootstrap that re-estimates the threshold.** Our README caveat 1 already admits the threshold is
  treated as exact.
- **Drop 1/week as a gate** (it rests on ~4 events); report it instead.
- **Fresh validation segments.** The same 24 segments and 1,344 injections have now been reused across Y1, Y2 and
  three cascade tests. This is the strongest point in the review.
- **Cluster stage-1 candidates at ≥ 2 s**, or merge overlapping ±1-s windows, to avoid double-counting trials.
- **Harder decoys:** glitch trains, plus power-drift decoys once v_s is in. Sine-Gaussians are the easy case.
- **Rename PARTIAL → INCONCLUSIVE.**
- **K1 via Monte Carlo of the actual correlated grid** on Gaussian noise, plus off-grid injections against Allen's
  mismatch bound.
- **K2 with 10–20 stratified injections**, including loud ones and one on a gated glitch.
- **A per-q recovery gate, median ρ̂/ρ_opt ≥ 0.9.**

**Rejected or modified:**
- **≥ 1–5 yr background via sub-segment shifts.** Stage 1 could be slid cheaply from cached bins, but every extra
  stage-1 candidate costs a full neighbourhood search: 1 yr at one per minute is ~5×10⁵ candidates, ~13× S3, which
  is infeasible on the Mac. Our own deep-FAR work also showed that slides add pairs, not independent noise samples.
  Instead: drop 1/week and quote 1/day with the block bootstrap.
- **"Gate on all injections" in P1.** That mixes stage-1 and stage-2 failures, so gate on the caught set, stratified
  by q.
- **Frequency-domain vs time-piece χ².** For a monotonic chirp, equal-SNR frequency bins and time pieces are nearly
  the same partition. Use Allen's frequency-domain form because it is cheaper and standard, but this is not a
  substantive change.

**Unclear:** mass-bin heterogeneity, i.e. one glitchy bin dominating a global threshold. Per-bin thresholds add
their own trials factor, so the right treatment is not obvious.

## 4. My independent view: Q6 vs the revised LongChirp plan

**Revised LongChirp got more expensive and less novel.** It now needs:
- 24 fresh segments: fetch, plus a Y1 rerun (~28 h), Y0 and injections;
- v_s;
- a bigger neighbourhood, with peak-only χ².

That is roughly 1–2 weeks of Mac time. Its likely outcome is also partly predictable: production searches already
make χ² + v_s coherent filtering work on real noise. What is not predictable is whether a laptop-scale neighbourhood
captures it. The value is real but modest, and two new O4 subsolar searches narrow the novelty further.

**Q6 is higher-impact, but Neon's "cheap" is not established.**
- LVK's tension comes from **pSEOB**, a full inspiral–merger–ringdown waveform with fractional deviations, analysed
  hierarchically.
- Kyutoku's bias is derived for **damped-oscillation** fits.
- A damped-sinusoid mock-catalog study is cheap, but it largely re-derives Kyutoku.
- The decisive version, pSEOB-like mock catalogs at GWTC-4 SNRs, needs parameter estimation on many mocks. That may
  be cluster-scale, and it is a new toolchain for us (`pyseobnr`); our ringdown tools are the `ringdown` package and
  an NPE, not pSEOB.
- Whether LVK's own mock-catalog results are public is unknown.

**My recommendation.** Before Sumit chooses, spend a **2–3 h read** (no compute) of:
- the body of Kyutoku 2026;
- the pSEOB and mock-catalog sections of 2603.19021.

The read should answer three questions:
1. Does Kyutoku's mechanism apply to pSEOB's parameterisation?
2. Are LVK's mock-catalog outputs public?
3. Is a reduced-order version (e.g. linear-signal / Fisher plus higher-order bias terms with pyseobnr waveforms)
   Mac-feasible?

**Decision rule:**
- If the answers are yes / yes-or-no / yes, Q6 first.
- If the decisive version needs cluster-scale PE, LongChirp-revised first, validated on fresh segments.

Q6 opens a new line, so it is Sumit's call either way.
