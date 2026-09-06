# DeepStrain — Roadmap (forward-looking)

> Complements [CLAUDE.md](CLAUDE.md) (current status per sub-project) and
> [JOURNAL.md](JOURNAL.md) (dated history). This file captures the **next
> high-leverage moves** and the standing guardrails. Each item says *what*,
> *why it's high-leverage*, *which sub-project / where the detail lives*, and
> *status*. Added 2026-06-20 from a cross-project review ("legs" analysis).

All three arcs are currently PARKED with honest results (two wins, one modest,
one honest negative). These items are the cleanest ways to **strengthen what we
already have on the same data**, plus the one guardrail on a parked thread.

---

## P1 — Echo non-detections → real UPPER LIMITS  *(highest leverage)*
- **What:** add an **injection-efficiency curve** to the echo comb search — i.e.
  measure, per echo spacing Δt (≡ λ), the amplitude you *would have detected*.
  That converts the current honest "non-detection" into a quantitative
  **"we exclude λ above X"** exclusion.
- **Why high-leverage:** same data, but a genuinely stronger and more publishable
  result — an exclusion is a constraint, a non-detection is not. Flagged honestly
  in the leg-8 review.
- **Where:** `echoes/`. Detail + the v1/v5 sensitivity machinery to extend:
  [echoes/notes/lab_notebook.md](echoes/notes/lab_notebook.md). (v1 already has a
  sensitivity-curve harness `06`; this generalizes it to a per-Δt efficiency → λ map.)
- **Status:** ✅ **DONE (2026-06-20, v6).** `scripts/11_upper_limits.py`: per-Δt exclusion
  curve at N=300. **GW150914: exclude amplitude ≥ A90=1.65σ** at predicted Δt (A50 1.33σ);
  GW151226: ≥1.55σ at its canonical Δt=0.0579 s. Smooth across all spacings; stress-tested
  (statistic verified, threshold not glitch-driven). (γ=0.7 fixed.) **Update (E1, 2026-06-25): the ML
  scorer does NOT tighten this — through the honest production path A90 ML≈comb (0.98×); v5's ~1.2× edge
  is a 50%-point effect that vanishes at the 90%-exclusion level. The comb UL stands. See PLAN.md E1 /
  echoes lab notebook.**

## P1 — Multi-event no-hair δ STACKING
- **What:** combine the no-hair deviation δ across **multiple events**, not just the
  single spine event (GW250114). Single-event σ(δ) ≈ 0.24; stacking is the clean way
  to sharpen it.
- **Why high-leverage:** the one place more data **directly tightens a real GR test**.
  Already noted as a v2 direction; the amortized SBI network is built and calibrated,
  so this is mostly a hierarchical-combination layer on top of proven infra.
- **Where:** `ringdown_spectroscopy/` (no-hair arc, v2/v3 — COMPLETE & calibrated).
  Detail: [ringdown_spectroscopy/notes/lab_notebook.md](ringdown_spectroscopy/notes/lab_notebook.md).
- **Status:** ⚠️ **METHOD ✅ / real payoff ❌ PARKED (2026-06-20, v5 + stress-test).** `12_stacking.py`
  validated the stacking METHOD — σ(δ) tightens as **√N** on informative injections (N=8 → 0.095 vs
  ideal 0.097, unbiased, calibrated), gated. **BUT** the north-star stress-test (`13_more_events.py`)
  showed **only GW250114 actually measures δ**; all 7 fainter public events return ≈ the prior. So the
  v5 "GW250114+GW150914 → 1.3× tighter" was a Gaussian-approx-of-prior **artifact** (corrected) — there
  is effectively ONE informative real event. **Real multi-event sharpening is blocked by the per-event
  SNR information wall** (only SNR~80-class events measure δ). Come-back-later = more very-loud events, or
  an NPE that extracts δ at lower SNR (likely information-limited, like tone-count).
  **v6 (2026-06-20) MAPPED THE WALL:** `14_delta_threshold.py` swept injected ringdown loudness and
  measured σ(δ) vs SNR — δ only becomes informative (σ/prior < 0.90) at **ringdown SNR ≳ 37**, and even
  at the top of the NPE's trained loudness it's just **~13% tighter than the prior**; GW250114 (real,
  σ/prior 0.83) sits right at that edge. So the stacking starvation is now quantitative, not anecdotal:
  every public event lands at-or-below the informative threshold. Seed-robust, gated.

## P2 — Higher-N injection campaigns where claims are UNDERPOWERED
- **What:** re-run the underpowered claims at **N ≈ 300–500** injections. Specifically
  the leg-8b "sensitivity reversal" rested on **N = 25** (confidence intervals overlap,
  so it's not yet real or refuted).
- **Why:** cheap and **decisive** — settles whether the effect is real instead of leaving
  it ambiguous. Low effort, high clarity.
- **Where:** `echoes/`. Detail: [echoes/notes/lab_notebook.md](echoes/notes/lab_notebook.md).
- **Status:** ✅ **DONE (2026-06-20).** (a) Upper limits run at **N=300**. (b) The specific leg-8b
  "sensitivity reversal" SETTLED: re-ran `08 --n-trials 300` → the in-band family differences are REAL
  & physically sensible (f0=320/γ=0.9 genuinely easier, +6–7σ; f0=150/γ=0.5 harder), NOT a pathology —
  the N=30 overlap was just underpower. The one true anomaly (out-of-band control not collapsing) is the
  known whitened-domain artifact (valid v4 raw test = 10%). No pathological reversal survives.

## GUARDRAIL — Do NOT throw more ML at the tone-count gap
- **What:** keep the ringdown **v4 tone-count thread PARKED**. Do not iterate more
  classifier architectures on the same data.
- **Why:** the gap is **information-limited, not legibility-limited** — independently
  confirmed (leg 2 and leg 7, and our own six-attempt diagnostic chain ending in a
  calibrated-but-weak AUC ~0.61). The real lever is **more SNR / a coherent multi-
  detector model**, not a fancier net on the same parked data.
- **Where:** `ringdown_spectroscopy/` v4 (PARKED — honest negative). Detail + the full
  six-attempt table: [ringdown_spectroscopy/notes/lab_notebook.md](ringdown_spectroscopy/notes/lab_notebook.md).
- **Status:** 🅿️ PARKED intentionally. Revisit only with more data / a coherent model /
  multi-event stacking / explicit Bayesian model selection.

---

## LONG-HORIZON PROJECTS (L1–L7) — tracked, never dropped for size
Full scoping in **[RELATED_WORK.md](RELATED_WORK.md)** (§ Long-horizon projects), added 2026-08-15 from a
literature sweep. **Standing rule: effort is not a reason to drop an item.** There is no deadline; the Mac
runs unattended; long-running jobs are normal. An item leaves the list when it is *done* or *measured to be
impossible* — never because it looked big.

**Status as of 2026-09-05: six of the seven are closed.** Five returned negatives or limits rather than
wins, which is the expected yield and is why they were worth doing. Detail for each lives in CLAUDE.md.

| | project | outcome |
|---|---|---|
| **L1** | cheap-template dense bank (ratio-filter de-chirping) | ❌ **CLOSED, honest negative (2026-08-15).** The algebra is exact but the gain is ≈ log N / log K, and subsolar needs K ≈ 16,385 taps → **0.94×, marginally slower**. The published 8× assumes K ≈ 250 (BNS). Reopens only for a signal class needing K ≲ 1,000 taps. |
| **L2** | deep background → 1/century | ✅ **DONE (2026-08-19).** 727 segments → **4,120 yr**, 1/century reached, **null 4/4**. Precision 33–44% → 10–12%, though the estimator audit then showed the jackknife understates by 4.2×, so quote 1/decade **14.53 ± ~1.7**. Remaining limit is independent loud-noise samples, not livetime. |
| **L3** | orthonormal-mode adoption across the ringdown arc | ❌ **CLOSED, negative (2026-08-15).** Max \|ΔAUC\| = **0.00000** bit-for-bit — the basis carries no detection information. Does **not** explain the v4 tone-count wall, which stands as an information limit. |
| **L4** | coherent network echo search | ⚠️ **RETRACTED (2026-08-22).** The 1.12× at "3.2σ" was 0.96σ under a paired bootstrap; the analytic bar had omitted three terms. **The injection-convention finding stands** (physical 1.12× vs identical 0.92×). |
| **L5** | three-tone spectroscopy | ❌ **CLOSED, information-limited (2026-08-15).** (2,2,2) is degeneracy-limited (needs A/A220 = 1.45); (3,3,0)/(4,4,0) are weakness-limited. **Reopening criterion: ρ_rd ≈ 38 ≈ 1.5× GW250114.** Also corrected our own [S] note — GWTC-5.0 *constrains* a third tone, it does not detect one. |
| **L6** | larger unlabeled pool for the SSL backbone | ❌ **CLOSED, saturates (2026-08-15).** Fully achieved by **2,500 specs**, 8× less than N4 used; slope 5k→20k is **−0.020**. Cross-detector pool addition is null (+0.0009). A bigger fetch is not justified. |
| **L7** | S251112cm (FAR 1/6.2 yr subsolar candidate) | 🔒 **STILL BLOCKED: O4c not public.** `o4c_release_watch.py` is the trigger; last checked 2026-09-04, only `O4c1DiscC00` (1.14 h) released and it does not cover the event. **The one item here that is waiting on the world rather than on us.** |

**Added 2026-09-05 — P0 (cross-project): THE RADIUS OF CONVERGENCE OF THE KERR QNM SPIN SERIES.**
*A hypothesis two oracles can test on one object, born from our own instrument's limit.*
- **The object.** ω₂₂₀(χ), the fundamental Kerr quasinormal frequency as a function of dimensionless spin —
  the quantity we measure on real remnants (GW250114 at χ ≈ 0.69–0.77) and `ansatz` evaluates exactly with a
  Leaver oracle (`qnm_precise.py`). Genuinely the same object; already joined once by TheBridge leg B.
- **What we measured and where we stopped.** The slow-rotation series' successive error ratio *rises* with
  order — 0.7235 → 0.7485 at χ = 0.90, separation 21:1 from its own drift — but only three Taylor
  coefficients are recoverable from double-precision `qnm` frequencies (three methods, same wall), so
  **direction resolved, trend shape not, limit not.**
- **The hypothesis, stated with both live outcomes because both are consequential.** The series' radius of
  convergence R is set by the nearest singularity of ω₂₂₀ in the *complex* χ-plane. **H: R = 1, with the
  singularity at extremality of square-root type, so the error ratio → χ.** Then every O(χⁿ) programme is
  convergent for all sub-extremal spin, error_n ~ χⁿ asymptotically, and our 0.7235/0.7485 climb to 0.90.
  **¬H: R < 1** — a complex singularity closer than extremality — and the slow-rotation series *diverges*
  above R; EMRI central objects at χ ≈ 0.9 would then sit past it, and "no finite order is controlled" —
  the claim we refuted on 2026-09-02 — comes back true at high spin for a reason nobody argued.
- **Who does what.** `ansatz` has the instrument we lack: arbitrary-precision Leaver. Domb–Sykes on the
  series (a_n/a_{n−1} vs 1/n → intercept 1/R, slope → exponent) needs ~20 reliable coefficients, which is a
  precision problem, not a method problem — we hit the wall at 3 in double precision. We supply the
  observation, the validated extractor (Chebyshev-high/truncate-low, golden-tested on 1/(1−x)), the
  pre-registration, and the physical anchor. TheBridge joins them under a gate.
- **✅ CONTROLS BUILT AND RUN (2026-09-05, `33_domb_sykes_controls.py`), and they change the ask.** `ansatz`
  objected that the extractor's only golden test was 1/(1−x), whose R is exactly 1 — i.e. hypothesis H — so a
  bias toward H would be invisible. Three controls, one per verdict band, on clean values and on float64:

  | control | truth | clean values | float64 values |
  |---|---|---|---|
  | 1/(1−x) | R=1, real | **PASS**, 8 ratios | PASS, 4 ratios |
  | 1/(1−2x) | R=1/2, real | **PASS**, R=0.5000 | PASS, R=0.5000 |
  | 1/(1+x²) | R=1, **complex pair** | **PASS**, pair identified | **FAIL**, 2 ratios |

  **The method is sound and can detect R<1** — the 1/(1−2x) control returns 0.5000 at both precisions, so the
  feared "returns H by construction" is not a property of the method. **But at float64 the complex-pair
  control FAILS**, and a complex pair is exactly what ¬H consists of ⇒ **on our data the pipeline could only
  ever have returned R≈1.** The objection was right, for a reason neither of us had: not bias, *blindness*.
- **⇒ THE HIGH-PRECISION LEAVER BUILD IS JUSTIFIED, and now by a measurement rather than an expectation.**
  A provisional Kerr number exists (4 stable ratios 0.3366/0.5704/0.6694/0.7315, Domb–Sykes R = 1.13) and is
  **explicitly not a result** — gated as PROVISIONAL, because the control that would license it fails at the
  precision it was computed in.
- **Three bugs the controls found in our own pipeline, all of which failed silently as "too few
  coefficients"** — a message indistinguishable from a precision limit: normal equations (XᵀX) singular at
  16 digits, fixed with QR; step-detection thresholding eight coefficients instead of two, rejecting a
  genuinely even series; and iterating the ratio index by 1 when the series has stride 2, dividing two
  numerical zeros. **Each would have been read as "we need more precision" and none was.**
- **✅ PRIOR-ART RISK RETIRED (2026-09-06, body read).** arXiv:2607.27043 computes its slow-rotation
  expansion **only to a⁸** (eqs. 41–46), in Mathematica at unstated precision, and performs **no ratio test,
  no Domb–Sykes analysis, no radius-of-convergence statement, and no search for singularities in the complex
  spin plane**. It notes only that "the end result … is still a finite Taylor series in the spin, which is
  not appropriate to model black holes with moderate rotations" — awareness of the limitation, no analysis of
  it. Their Padé is applied to the **WKB** series in ξ̄, a different expansion. ⇒ **the question is open at
  the level of the one paper most likely to have closed it.** Scope: one body read, not a proof of novelty.
- **❌ AND THE CHEAP ROUTE IS CLOSED (`34_pade_pole.py`).** Before commissioning the Leaver build, tested
  whether a **Padé pole** could locate the singularity from the few coefficients float64 already gives —
  Domb–Sykes needs many terms, a [2/2] needs five. Put through the same three controls at the same float64
  precision: **1/(1−x) returns R = 0.0096 where the truth is 1**, and 1/(1−2x) returns 0.0081 where the truth
  is 1/2. Standard failure modes, both intrinsic: a degenerate Hankel system on exact input, Froissart
  doublets near the origin on noisy input. **The controls earned their keep here** — the Kerr [2/2] returns
  **R = 1.28**, which looks like support for H, from an estimator that gets a known R = 1 wrong by two orders
  of magnitude. A number agreeing with the hypothesis, from an instrument broken on the hypothesis's own test
  case, is the most expensive kind of number.
- **⇒ NO CHEAPER ROUTE EXISTS. The high-precision Leaver build is the only path, and it is OURS to own** —
  `ansatz` is right that nothing in GF(p) nullspace machinery helps and the skill needed is numerical. What
  we would want from them is review of the continued-fraction recurrences, since a wrong one converges to a
  confident wrong number.
- **Pre-register before anyone runs it.** (i) coefficient count declared in advance and coefficients shown
  stable under extraction degree, per our own 2026-09-04 lesson; (ii) the Domb–Sykes fit range fixed;
  (iii) verdict bands: R ∈ [0.98, 1.02] ⇒ H; R < 0.95 ⇒ ¬H; between ⇒ unresolved and say so.
- **Why it is physics and not numerics.** The near-extremal Kerr spectrum *branches* (zero-damped vs damped
  families, Yang et al. 2013 **[S]**); whether that branching is what limits the slow-rotation series, or
  something nearer, is a statement about the analytic structure of the Kerr spectrum in spin — and it is the
  number every second-order-in-spin beyond-GR programme silently depends on.
- **Prior art, checked before writing this.** arXiv:2607.27043 (2026) **[A]** resums Kerr QNMs with Padé /
  Borel–Padé, including a slow-rotation implementation, and traces a breakdown near extremality to the
  near-horizon potential — but its abstract states nothing about the spin series' radius of convergence or
  the singularity location in the complex-spin plane. **Body not read; the residual risk that it does this
  is real and must be retired by reading it before any claim of novelty.** No Domb–Sykes analysis of the
  Kerr spin series found in two targeted searches.
- **What it feeds.** The sGB programme's O(χ²) substrate (`ansatz`), the 6.4%/18.9% truncation numbers
  (ours), and the open item "the sGB correction's own spin series" — if R = 1 for Kerr, the next question is
  whether the *correction's* R is smaller, which is answerable by the same instrument on the same object.

**✅ DONE 2026-09-06 — σ(δ) SATURATES; it does not follow 1/SNR.** Across SNR 20.0–78.6 (a 3.9× span) σ falls
**1.15×** where Fisher demands 3.9×; a zero-free-parameter "σ = σ_prior" model beats the Fisher fit by 5× in
SSR, and a crossover model puts its knee at SNR ≈ 84, *outside* the observed population. So more SNR buys
nothing in the regime we can observe, and GW250114 at σ/prior = 0.822 is only just leaving the saturated
branch. **Scope: one event above our informativeness threshold, so the 1/SNR branch is unverified, not
refuted.** Artifact `35_sigma_vs_snr.json`. *(Original item follows.)*

**Added 2026-09-05 — P1 (asked by a sister): does σ(δ) scale as 1/SNR or saturate?** TheBridge's standing
`SISTER_REQUESTS.md` item for us. Data on disk (the 439-event δ re-measurement, 2026-08-07) can answer it
now: fit σ(δ) vs ringdown SNR across real events, test Fisher 1/SNR against a floor. Our G8 result already
found a species-4 crossover at SNR ≈ 124 from injections; this is the same question asked of the *real*
event population. Cheap, pre-registerable, and someone is waiting for it.

**Added 2026-09-06 — P1: RUN THE NPE ON GW231028. The cheapest shot at unparking v5 stacking.**
- **What:** fetch GW231028_153006 and run our existing start-time-marginalized NPE; read δ_σ/prior.
- **Why:** v5 δ-stacking is parked on one finding — *only GW250114 measures δ*. `arXiv:2509.08657` (verified
  at abstract level) reports a first-overtone detection in GW231028 at **BF ≈ 189, >7σ**, remnant 246 M☉,
  χ = 0.81. It is not in our 12-event sample. A second informative event unparks stacking at n = 2 and gives
  the saturation result above its second point above threshold.
- **Honest prior:** the remnant is ~4× heavier than GW250114, so the ringdown sits much lower in frequency,
  near a band edge where our NPE has never been tested. Their "decisive overtone evidence" is a different
  measurement from our δ and does not imply ours is informative. GWOSC availability unchecked.
- **Cost:** one fetch plus one NPE evaluation; no retraining. Pre-register the informativeness bar
  (δ_σ/prior < 0.88, the threshold already used) before looking.

**Added 2026-09-05 — P1: NARROW THE BAND. The next result that no audit prompted.**
- **What:** retrain `cnn_w64` on a spectrogram truncated at ~250 Hz instead of 1024 Hz, and measure sensitive
  distance at matched FAR against the existing model.
- **Why:** the CNN response probe (2026-09-02) measured that **97–100% of the network's sensitivity sits
  below 224 Hz**, with the centre at 111–121 Hz and band-power correlation falling to −0.04 by 717–1024 Hz.
  So roughly **three-quarters of the spectrogram is capacity the network learned to ignore** — and subsolar
  chirps genuinely deposit their SNR low, because they sweep slowly. Narrowing frees that capacity.
- **Honest prior:** this could easily buy nothing. The unused capacity may already be costing nothing, and a
  narrower input might simply train to the same operating point. The probe measured *where the sensitivity
  is*, not *whether the ignored region is harmful* — those are different claims and only the first is
  established.
- **Why it is the item and not another audit:** it is a question nobody's correction raised. It came from
  wanting to know what the detector actually responds to, and the answer suggested a change to the detector.
  Pre-register the bar before running, as with the SSL trend.
- **Cost:** one retrain plus a matched-FAR evaluation; the data is on disk. Waits for the box.

**Added 2026-09-05 — SSL data-wall trend, settled.** N4's trend was gated on a 2-seed run. Two
pre-registered re-runs (n=5 suggestive at 2.94σ, then n=20 fresh seeds) resolve it: **gap +0.209,
bootstrap p = 0.00000, 7.39σ**. The decomposition changed the claim: at 2,000 labels from-scratch clears
the 1%-FAR floor **0/20** and SSL **20/20**, so the gain is **floor-clearing, not distance**. Open follow-up
if anyone wants it: the label-efficiency *curve shape*, which needs more budgets, not more seeds.

---

## Known blockers carried forward (context for the above)
- **PBH subsolar:** template-bank density wall — subsolar needs ≤0.1% Mc spacing (~1,600+ templates);
  1,619 was our laptop ceiling. **⚠️ "Intractable locally" NO LONGER HOLDS (2026-08-15):** the field's
  ratio-filter de-chirping reports ~8× per-core speedup, i.e. ~13k templates on the same hardware — a real
  move down the density sweep toward the 0.72 oracle. Reclassified from *blocker* to **L1, a tracked
  long-horizon project**. Do not restate it as intractable. (pbh v2 PARKED; coincidence win +1.37× stands.)
  **Build C DONE (2026-06-20, L4 VM):** the "lower FAR needs more data" item is closed — coincidence is
  FAR-robust (graceful to 1/year; @1/day reproduces the +1.37×; @1/year still beats single-det floor ~1.2×).
  See [primordial_blackhole_search/RESULTS.md](primordial_blackhole_search/RESULTS.md).
- **Ringdown tone-count:** information-limited (see guardrail above).
