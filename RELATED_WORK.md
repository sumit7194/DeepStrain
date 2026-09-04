# Related work — the field around this repo

*A living literature record. Extend it; don't rewrite it. Each entry says what the paper claims, **what it
means for us**, and how well we verified it.*

**Verification marks** — because a search-result snippet is not a citation:
- **[A]** abstract fetched and read directly
- **[S]** search-result summary only — *treat the numbers as provisional; fetch the abstract before any
  load-bearing use* (see the `prior-art-rigor` standing rule)

**Last swept: 2026-08-15.** Literature ages: before citing anything here as "the current state", re-run the
sweep. Same rule as PLAN.md's parked list — a status is a measurement with a timestamp, not a fact.

---

## 1. Subsolar / PBH — `primordial_blackhole_search/`

### The collaboration's own O4a search — the direct comparison
**[A]** [arXiv:2602.12115](https://arxiv.org/abs/2602.12115) — *Search for Sub-Solar Mass Binaries in the First
Part of LIGO's Fourth Observing Run* (PRD). Companion **[S]** [arXiv:2605.05444](https://arxiv.org/abs/2605.05444)
(LVK, *Searches for Binary Mergers with Sub-solar Mass Components … O4a*).

- **25 million templates**; primary 0.1–2 M☉, secondary 0.1–1 M☉; tidal deformability up to 7×10⁵
- **No statistically significant candidates**
- Rate limit **< 2.5×10⁴ Gpc⁻³ yr⁻¹** at Mc = 0.2 M☉; **> 2× better than O1–O3 combined**
- **f_PBH < 0.5%** at 0.4 M☉ — a 1.8× tightening on O1–O3

**For us:** the honest yardstick for our subsolar arc, and it confirms the regime is null at far greater depth
than we can reach. Our contribution was never rate limits — it is the *minutes-long ML* corner and the
CNN-vs-realizable-bank comparison. Nothing here contradicts our results.

### The template-bank wall is being attacked — and it is our parked blocker
**[S]** A **"ratio filter" de-chirping framework** reports an **~8× per-core speedup** in template processing,
which is what made 25M templates affordable. Related: **[S]**
[arXiv:2511.12894](https://arxiv.org/html/2511.12894) — reconstructing matched-filter SNR time series from
*nearby* templates, aimed at long-duration low-mass signals (BNS, subsolar). **[S]** A dedicated O4 SSM
template-bank paper exists in PRD (`10.1103/c97v-bmj8`).

> **🔴 TESTED AND CLOSED 2026-08-15 — an honest NEGATIVE.** Primary source read
> ([arXiv:2601.18835](https://arxiv.org/abs/2601.18835), PRD `10.1103/k21q-wp8f`). The algebra is **exact** for
> our pipeline (untruncated kernel reproduces the matched filter to **1.000000**), but **measured speedup is
> 0.94× — marginally slower.** Mechanism: the method converts O(N log N) → O(N log K), so the gain ≈
> log N / log K; the published 8× assumes **K≈250 taps** (BNS) while **subsolar needs K≈16,385** because of the
> phase these inspirals accumulate. With N=16.7M that caps at **1.6×**. Memory doesn't rescue it — memory was
> never binding (bank_dense was already template-major); **compute time** is. **Criterion for when this WOULD
> pay: a signal class needing K ≲ 1,000 taps.** See RESULTS.md "L1 VERDICT". *An [S] number that did not
> survive contact — and a cost model of mine that had to be superseded for measuring a primitive at the wrong
> scale.*

**For us — this is a live lever on a wall we called intractable, and the wall is now weaker.** Follow-up A
measured it quantitatively: 1,619 templates at 0.1% Mc spacing was our laptop ceiling, real-bank MF 0.489 vs
CNN 0.472 (a tie), both far below the true-template oracle 0.72 ⇒ *bank mismatch dominates*. The field's answer
is not a smaller bank but **cheaper templates**. An 8× per-core speedup applied to our 1,619-template ceiling
is ~13,000 templates on the same hardware, i.e. a real move down the density sweep toward the 0.72 oracle —
**the one axis Follow-up A identified as the dominant loss and could not push.** Scoped as a long-horizon
project below; **the original "intractable" verdict no longer holds and must not be restated.**

### S251112cm — the subsolar candidate, still unresolved
**[A]** [arXiv:2603.17009](https://arxiv.org/html/2603.17009v1) (EM counterpart search) and **[A]**
[ApJ 10.3847/1538-4357/ae48f9](https://iopscience.iop.org/article/10.3847/1538-4357/ae48f9) (PBH interpretation).

| property | value |
|---|---|
| FAR | **1 per 6.2 years** |
| chirp mass | 0.1–0.87 M☉ |
| P(≥1 component < 1 M☉) | **> 99%** |
| luminosity distance | 93 ± 27 Mpc |
| status (2026-08-15) | **under investigation — neither confirmed nor retracted** |

Counterpart search vetted 456 candidates, found nothing (neither confirms nor refutes). PBH interpretation:
if astrophysical, **f_PBH > 0.04**, with a predicted rate ~0.8 yr⁻¹ matching one detection. Popular commentary
suggests a refined FAR nearer 1/4 yr — **secondary source, unverified, do not cite.**

**For us, two things.** (1) **Calibration:** a real LVK subsolar candidate sits at 1/6.2 yr, i.e. squarely
inside the ladder our deep-FAR work measured (1/month, 1/year, 1/decade). We are working at the right scale.
(2) **Data:** see below — we cannot touch it yet.

### Public data status — why we cannot analyse S251112cm
**[A]** [gwosc.org/data](https://gwosc.org/data/): public runs are O1, O2, O3a, O3b, **O4a**, **O4b**
(H1/L1/V1, Apr 2024 – Jan 2025). **O4c is NOT released.** Nov 2025 falls in O4c ⇒ **S251112cm strain is
unavailable.** Keep `scripts/o4c_release_watch.py` pointed at it. *(Supersedes any earlier session's
speculation about availability.)*

---

## 2. Ringdown — `ringdown_spectroscopy/`

### Orthonormal QNMs — the most actionable paper of this sweep
**[A]** [arXiv:2605.03576](https://arxiv.org/abs/2605.03576) — *Ringdown Analysis of GW250114 with Orthonormal
Modes*.

> QNMs are **not orthogonal**; including multiple QNMs induces correlations among them which "can hinder the
> robust identification of subdominant QNMs." Orthonormalizing, the ℓ=m=2 first overtone significance rises
> **82.5% → 99.9%**.

**For us — this may explain a negative we recorded as physics.** The 220 and 221 modes are only ~6 Hz apart:
near-parallel basis vectors, so their amplitudes trade off against each other and each is individually poorly
constrained. That is a candidate mechanism for **v4's tone-count negative (AUC ~0.61)**, which we attributed
to "too weak at this data/SNR scale", and for the parameter correlations throughout the δ work.

**Do not adopt on faith — the open question (pre-registered here):** `P(A₂₂₁ ≠ 0)` is **not basis-invariant**,
whereas a nested-model Bayes factor is. Orthonormalizing also **changes the implicit prior**: flat over
(A₂₂₀, A₂₂₁) is not flat over the rotated coordinates. So some of a 17-point significance jump could be a
change in the *question* rather than a gain in *information*. It may equally be real — their model is not
clean linear-Gaussian, and reparameterization genuinely improves conditioning and reduces degeneracy-induced
posterior leakage. **Decisive test we are equipped for:** inject overtones of *known* amplitude into real
noise and build **ROC curves in both bases**. Higher AUC ⇒ real information gain (and a possible rescue of the
parked tone-count arc). Equal AUC with a higher quoted significance ⇒ the number moved, the information did
not. Both outcomes are results.

> **✅ RESOLVED, same day — the second branch.** `27_orthonormal_roc.py` / `28_orthonormal_prior.py`:
> the premise is real (overlap **0.863**, modes 5.6 Hz apart) but orthonormal and properly-handled
> non-orthogonal are the **same detector** — max |ΔAUC| = **0.00000**, bit-for-bit, *and still 0.00000 when the
> basis is built at the wrong remnant*, because the Schur complement of the 220 block of HᵀH **is** the Gram
> matrix of the orthogonalized 221 directions. The only lever that moves a number is the prior: β = Rθ with R
> triangular, so "uninformative" means different things in the two bases, and the Savage–Dickey Bayes factor
> shifts **0.454 dex (2.8× odds) at ΔAUC 1.7e−6**. **Significance moves; information does not.**
> ⇒ **our v4 tone-count negative is NOT explained by non-orthogonality** — it stands as an information limit.
> **Scope:** linear/fixed-frequency, so this does *not* show their number is wrong; orthonormalization may
> still help MCMC conditioning and amplitude reporting. Untested route to a real gain: real fits span a
> *union* of bases as (M, χ) vary, not one fixed basis. Any significance quoted in a rotated basis must state
> its prior. Gated (47).

### Supporting ringdown context
- **[S]** [arXiv:2512.08098](https://arxiv.org/pdf/2512.08098) — *High-overtone ringdown fits: start time,
  no-hair tests, and correlations.* Same start-time systematic our **B1 sweep independently reproduced**
  (peak-start mass biased ~+10%, decaying as the start moves later), plus the correlation theme above.
- **[S]** [arXiv:2509.17315](https://arxiv.org/pdf/2509.17315) — inspiral-merger-informed templates and the
  *limitations of classical spectroscopy*.
- **[S]** [arXiv:2404.11373](https://arxiv.org/abs/2404.11373) — SBI of ringdowns in the time domain (PRD 110,
  083010); truncated sequential NPE, per-event rather than amortized. **Our angle stays distinct:** amortized
  with the start time marginalized *by construction*.
- **[S]** [arXiv:2507.11192](https://ar5iv.labs.arxiv.org/html/2507.11192) — review, *Recent Advances in SBI
  for GW Data Analysis*. Good place to re-check that our amortized-δ framing is still not standard.

### GWTC-5.0 — and a headline in our exact territory
**[S]** Released **2026-05-26** ([LIGO Caltech](https://www.ligo.caltech.edu/news/ligo20260526),
[GWOSC](https://gwosc.org/eventapi/html/GWTC-5.0/)): **161 new events, 390 confident total**, O4b data
(Apr 2024 – Jan 2025), all BBH, no new NS-containing events.

Highlighted: **"the first measurement of three gravitational-wave tones from a black hole."** That is directly
our tone-count / spectroscopy territory — **read before any further tone-count work.**

*Bookkeeping:* our own catalog query counted **439** public entries; 390 is the *confident* count. Broader vs
stricter counting, not a conflict — but state which one you mean.

---

## 3. Echoes — `echoes/`

### Our null replicates; their method is a rung above ours
**[A]** [arXiv:2512.24730](https://arxiv.org/abs/2512.24730) — *Model-agnostic search of gravitational wave
echoes in LVK data.*

- Targets long-lived QNMs from strong interior reflection, avoiding specific echo waveform models
- **Generalized phase-marginalized likelihood coherently combining each QNM across the detector network**,
  plus an optimized line-notching procedure
- Events: **GW150914 (O1), GW231226 (O4a), GW250114 (O4)**
- **No statistically significant evidence for postmerger echoes**; 90% upper limits on network SNR and average
  initial strain amplitude

**For us:** our E3 nulls (GW150914 / GW151012 / GW151226 / GW250114, both ML and comb, independent ±hour
background) are **consistent with the field's latest, including on GW250114.** Their coherent *network*
combination is the obvious next rung above our single-detector-ish comb on an ML-residual envelope, if echoes
are ever reopened.

- **[S]** [arXiv:2510.01001](https://arxiv.org/html/2510.01001v3) — *GW250114 reveals black hole horizon
  signatures* (context; unverified).

---

## 4. ML methods — adjacent, and one paper that is our audit's twin

### The robustness paper — cite this in the deep-FAR audit
**[A]** [arXiv:2509.05283](https://arxiv.org/abs/2509.05283) — *Robustness of Sensitivity Evaluations for
Gravitational Wave Detection Algorithms.*

Running the AresGW ML pipeline across multiple **month-long real-noise datasets** produced *"notable
performance variations, highlighting the challenges introduced by finite-duration datasets"*, specifically
**at low false alarm rates**, and calls for *"more rigorous statistical validation"* and better GW-specific
benchmarking standards.

**This is the independent statement of what our 2026-08-09 deep-FAR audit measured** — that a deep-FAR
threshold on a finite real-noise dataset is far less stable than its Poisson error bar implies. We quantified
the mechanism: our 1/decade threshold moved **±33–44%** under a leave-block-out jackknife because its 8
defining events traced to **2 distinct H1 windows in a single segment**. **Effect: our audit stops being
internal housekeeping and becomes a concrete, mechanism-level instance of a named open problem.**

### Self-supervised pretraining — independent convergence with our N4
**[S]** [GraviBERT, arXiv:2512.21390](https://arxiv.org/html/2512.21390) (IOP, Apr 2026) — multi-scale feature
extractor + transformer encoder, **BERT-style self-supervised pretraining then supervised fine-tuning**,
pitched explicitly as *"a pathway towards foundation-style models for gravitational-wave science"*, supporting
transfer to new detectors and waveform approximants.

**For us:** our **N4** reached the same conclusion from the other end — a masked-spectrogram autoencoder
pretrained on unlabeled noise gave **+0.124 val-AUC at 1000 labels**, gain scaling as 1/labels, and the AUC win
*translated to sensitive distance*. Independent convergence on the direction; a natural benchmark if we extend
the SSL work.

### Other ML context
- **[S]** [arXiv:2603.09386](https://arxiv.org/pdf/2603.09386) — *Deep Learning Search for GWs from Compact
  Binary Coalescence* (2026)
- **[S]** [arXiv:2505.08332](https://arxiv.org/html/2505.08332) — improving detection significance of GW
  transient searches with CNNs
- **[S]** [arXiv:2403.04350](https://arxiv.org/abs/2403.04350) — self-supervised extraction of non-Gaussian
  features (PRD 111, 063520)

### Glitches and background — supporting our audit's diagnosis
**[S]** Blip glitches resemble the final cycles of stellar-mass CBCs and occur roughly **every 30 min at LHO**
(~15 min at LLO) during O3. **[S]** Time-slide significance is known to be affected by *"correlations in the
set of background triggers, non-stationary noise, residual effects due to genuine GW transients, and finite
sample size"* (cf. [arXiv:1601.00130](https://arxiv.org/pdf/1601.00130)).

**For us:** independent support for the audit's conclusion that deep FAR is limited by **independent
loud-noise samples, not livetime** — and for why real searches use signal-consistency vetoes and DQ flags
rather than raw time-slides alone.

---

## Long-horizon projects — tracked, not skipped

*Standing rule for this section: **effort is not a reason to drop an item.** There is no deadline here and the
Mac runs unattended; the only filter is whether the work is correct and worth knowing. Long-running jobs are
normal — checkpoint them and let them run for days. An item leaves this list when it is **done** or
**measured to be impossible**, never because it looked big.*

**L1 — Cheap-template dense bank (de-chirping / ratio filter). 🔴 CLOSED 2026-08-15 — NEGATIVE, measured.**
Ratio filtering gives **0.94×** on our pipeline, not 8×: the gain scales as log N / log K and subsolar needs
**K≈16,385** taps vs the paper's ~250. The dense-bank wall stands, but is now *understood* rather than assumed,
with a quantitative reopening criterion (**K ≲ 1,000 taps**). The bank was deliberately not built (~162 h for no
speed gain). *(superseded scoping follows)*

*(original entry)* **L1 —** Follow-up A identified **bank mismatch as the
dominant loss** (real bank 0.489, CNN 0.472, oracle 0.72) and could not push the one axis that mattered,
because 1,619 templates was our laptop ceiling. An 8× per-core speedup ⇒ ~13,000 templates on the same
hardware, a genuine move down the density sweep toward the oracle. *Effort:* weeks — implement the de-chirping
filter representation, re-golden-test it against `pbh/bankmf.py`, then re-run `bank_dense.py` / `bank_vs_cnn.py`
at the finer spacing. *Buys:* the first real answer to "does a CNN still tie a matched filter once the bank is
adequate?" — which is the question the whole subsolar arc circled. *Prereq:* none.

**L2 — Deep background to 1/century, and a jackknife-stable 1/decade.** The audit showed the binding
constraint is **distinct loud-noise samples**, not livetime. Background = (62N−1)·N·4096 s, so:

| segments N | background | 1/century events | note |
|---|---|---|---|
| 100 (now) | 80.5 yr | 0.8 — not measurable | 1/decade rests on 8 events, ±33–44% |
| ~112 | 100 yr | 1 | measurable but a 1-event estimator: useless |
| **~353** | **~1000 yr** | **10** | 1/century on the same footing 1/decade has now |

Distinct glitch samples grow ∝ N, so the jackknife spread should fall roughly as 1/√N: **±33% → ~±18%** at
N=353. *Buys:* a deeper rung **and** a defensible error bar on the existing one. *Prereq:* none — `far_deep.py`
checkpoints per segment and purges strain. **The single best use of idle Mac time.**

> **LAUNCHED 2026-08-15, targeting the FULL 727-segment pool** (not the 353 first scoped) ⇒ **~4,253 yr, 42
> events at 1/century** — a real estimator rather than the 8 events behind the current ±33%.
>
> **Two things learned at launch, both worth keeping.**
> 1. **`far_deep.py` had no fetch retry.** A single transient SSL-handshake timeout on *one* detector (H1
>    fetched fine, L1 timed out) discarded a whole segment, and the loop would have burned all 627 pool
>    entries in minutes while scoring nothing. Fixed: per-segment exponential backoff + a consecutive-failure
>    detector that sleeps 30 min when GWOSC looks degraded. Validated end-to-end in a live outage. A failure
>    was never permanent (an unscored segment simply has no cache entry), but it would have wasted the run
>    and *looked* like progress.
> 2. **The pool is valid but only covers 2024-04-11 → 2024-07-01.** All 727 segments lie inside the public
>    O4b window, contiguous at one segment length — so fetch failures are GWOSC availability, not bad GPS.
>    But **O4b runs to 2025-01-28**, so ~7 further months were never enumerated: *727 is a limit of our
>    discovery pass, not of the observing run.* If more background is ever wanted, extend the discovery
>    before concluding the data is exhausted.
>
> **Cost check (measured, not assumed):** the O(N²) background kernel scales cleanly at 4.2× per doubling —
> 0.4 s at 100 segments, 7.5 s at 400, so **~25 s per pass at 727** and ~12 min for a full
> `far_background_validation.py` re-run. **The analysis side has no wall; GWOSC fetch is the sole bottleneck.**

**L3 — Orthonormal-mode adoption across the ringdown arc. ✅ ANSWERED 2026-08-15 — DEAD as a sensitivity
play.** Scripts `27_orthonormal_roc.py` / `28_orthonormal_prior.py` measured it: an orthonormal basis and a
properly-handled non-orthogonal fit are the **same detector** (max |ΔAUC| = 0.00000, and still 0.00000 when the
basis is built at the *wrong* remnant). The only thing that moves is the reported number, via the prior's Occam
factor: log₁₀BF shifts **0.454 dex (2.8× odds) at ΔAUC 1.7e−6** in the uninformative limit. **So the parked v4
tone-count negative is NOT explained by non-orthogonality**, and there is no retraining to do for sensitivity.
Residual (not pursued): orthonormalization may still help MCMC conditioning and amplitude reporting, and the
one untested route to a real gain is that real fits span a *union* of bases as (M, χ) vary rather than one
fixed basis. Superseded scoping follows for the record —

*(original entry)* **L3 —** The weekend ROC test (below) only decides
*whether* the basis change carries information. If it does, the follow-through is larger: rebuild the QNM basis
in `sbilib`, **retrain the NPE** in the orthonormal parameterization, re-run coverage/recalibration (v3's T=1.05
work), and re-examine the parked **v4 tone-count negative** and the **δ wall**. *Effort:* weeks. *Buys:*
possibly overturning our own parked negative — the highest-value outcome available in the ringdown arc.
*Prereq:* the ROC test result.

**L4 — Coherent network echo search. 🟡 FIRST RUNG DONE 2026-08-15 — coherent network combination HELPS
(1.12×, 3.2σ).** Measured on physically-injected echoes (H1/L1 geometry derived from the merger: −6.59 ms,
sign −1). **My prediction that the network axis would buy ~nothing was wrong.** Bigger finding: the injection
convention decides the answer — the convention every existing echo script uses (identical waveform, no delay,
independent carrier phase per detector) makes coherence *significantly worse*, and two such bugs were found
here. Remaining: the within-detector envelope and the paper's full phase-marginalized likelihood.
See echoes/notes/lab_notebook.md. *(original scoping follows)*

*(original entry)* **L4 —** [arXiv:2512.24730](https://arxiv.org/abs/2512.24730) combines each QNM
**coherently across the detector network** with a phase-marginalized likelihood plus line-notching; our comb
runs on an ML-residual envelope, essentially per-detector. *Effort:* weeks. *Buys:* the field-standard
sensitivity rung for echoes, and makes our nulls directly comparable to theirs rather than merely consistent.
*Prereq:* none, but the echoes arc is parked complete — this is what reopening it should mean.

**L5 — Three-tone spectroscopy. 🔴 CLOSED 2026-08-15 — information-limited, with a reopening number.**
⚠️ **The [S] claim in this file was WRONG**: GWTC-5.0 does not report "the first measurement of three tones".
Verified at [arXiv:2510.01001](https://arxiv.org/html/2510.01001v3): GW250114 has strong evidence for
(2,2,0)+(2,2,1), a *weak early-time* preference for (2,2,2), and (4,4,0) with **"SNR insufficient for
detection"** — the LVK **constrains** a third tone. Our own measurement (`29_third_tone_floor.py`): the two
candidates fail for **different** reasons — (2,2,2) is 86% absorbed by refitting 220+221 (**degeneracy**,
needs an unphysical A/A₂₂₀ = 1.45), while (3,3,0)/(4,4,0) are 61–81% orthogonal but intrinsically faint
(**weakness**). **Reopens at ~1.5× GW250114's ringdown SNR.** A second [S] entry that did not survive contact.
*(original scoping follows)*

*(original entry)* **L5 —** GWTC-5.0 reports **the first measurement of three tones from a black hole**.
Our machinery stops at two (220 + 221). *Effort:* read first (that measurement may itself be the answer), then
weeks if we extend `sbilib` to a third tone and re-derive the no-hair test with it. *Prereq:* read GWTC-5.0.

**L6 — Larger unlabeled pool for the SSL backbone. 🔴 CLOSED 2026-08-15 — NOT justified, measured.** The SSL
gain is fully achieved at **2,500** unlabeled specs (+0.108) and does not grow to 20,000 (slope −0.020 over
5k→20k, flat within a seed sd of 0.019). Cross-detector transfer is null (+0.0009 from 6,250 L1 specs). So
fetching a larger pool would have cost days for an effect bounded at ≲0.03 AUC. N4's caveat is answered: more
unlabeled noise does **not** help. See RESULTS.md "L6". *(original scoping follows)*

*(original entry)* **L6 —** N4's caveat was that the unlabeled pool *was* the labeled
set's own 20k noise spectrograms; GraviBERT-scale pretraining implies far more. *Effort:* days of fetching +
pretraining. *Buys:* tests whether N4's data-wall win keeps growing with pool size — the obvious open thread we
noted ourselves. *Prereq:* none.

**L7 — S251112cm.** Blocked on the **O4c public release** (not out as of 2026-08-15). `o4c_release_watch.py`
is the trigger. *Buys:* our pipeline pointed at a real subsolar candidate at 1/6.2 yr — precisely the regime
our FAR ladder now calibrates.

---

## Standing implications for the roadmap

Short items, in the order I would do them. The long-horizon items **L1–L7 above are part of this list, not an
appendix to it** — they are sequenced later only because some depend on the short ones, never because of size.

1. **Orthonormal-mode ROC test** *(recommended next)* — settle whether the 82.5% → 99.9% boost is information
   or reparameterization, by injecting known-amplitude overtones into real noise and comparing ROC/AUC in both
   bases. Machinery already in place (`sbilib.simulate_tonecount`, `.venv311` `ringdown` pinned). **Gates L3.**
2. **Cite [arXiv:2509.05283](https://arxiv.org/abs/2509.05283)** in the deep-FAR audit section of RESULTS.md —
   it independently names the problem our audit measured with a mechanism.
3. **Read GWTC-5.0's three-tone measurement** before further tone-count work. **Gates L5.**
4. **Launch L2** (deeper background) — unattended, no prerequisite, best use of idle Mac time; it can run
   underneath any of the above.
5. **L1, L4, L6** — independent of everything else; start whenever there is appetite.
6. **L7** waits on the O4c release; the watcher fires it.

---

## sGB scope analysis (2026-09-02) — asked by the user, cross-checked with `bridge` over four rounds

A theory programme proposed working at **first order in the sGB coupling** and **second order in spin**. The
question put to us: is that scope physically relevant, or an artefact of what is calculable? Answered from
the literature plus our own ringdown machinery. Every number below was recomputed independently by `bridge`;
seven claims were overturned across four rounds, none of them by argument.

**The observational bound.** Strongest current: **√α ≲ 0.26–0.30 km** (90% credible) from **GW230529**, a
neutron star merging with a lower-mass-gap object — 0.298 km, tightening to 0.260 km with higher-order EdGB
corrections; same analysis gives **ζ ≲ 0.024** [A]. Previous best from NSBH events (GW200105, GW200115,
GW190814) was √α ≲ 1.33 km [A]. **These come from the inspiral**, where scalar dipole radiation is a −1PN
effect accumulating over many cycles, and from *low-mass* systems, because the constraint scales as α/M².

- [arXiv:2405.13279](https://arxiv.org/pdf/2405.13279) — GW230529 EdGB constraints **[A]**
- [arXiv:2406.03568](https://arxiv.org/html/2406.03568) — Tests of GR with GW230529 **[A]**
- [arXiv:2201.02543](https://arxiv.org/abs/2201.02543) — NSBH EdGB bounds **[A]**

**First order in the coupling — fine, generously.** For a 68 M☉ remnant (GM/c² = 100.4 km) at √α ≤ 0.30 km:
ξ = α/M² ≈ **9 × 10⁻⁶**. Second-order terms are four to five orders below first. This truncation is not
where the programme is at risk.

**The QNM shift is linear in ξ², not ξ.** [arXiv:2412.09377](https://arxiv.org/html/2412.09377) defines
ξ = α/M² and expands the weak-coupling QNM correction **to quadratic order in α** **[A]**. We initially
quoted a gap of 3–4 orders by conflating ξ with the shift; `bridge` caught it. Corrected:

| remnant | ξ = α/M² | shift ~ ξ² | gap vs σ(δ) ≈ 0.14 |
|---|---|---|---|
| 68 M☉ | 8.9 × 10⁻⁶ | 8.0 × 10⁻¹¹ | **9.2 orders** |
| 30 M☉ | 4.6 × 10⁻⁵ | 2.1 × 10⁻⁹ | 7.8 orders |
| 5 M☉ | 1.7 × 10⁻³ | 2.7 × 10⁻⁶ | 4.7 orders |

We sit at ξ ~ 10⁻⁵, deep inside the regime where the quadratic scaling is exact (the same paper reports it
failing by ~55% only at large coupling), so this is a trustworthy perturbative number rather than an
extrapolation. **⇒ sGB is unreachable by ringdown at current sensitivity, by 7–9 orders. It is constrained
in the inspiral, and a ringdown/QNM programme works in the channel least sensitive to the theory.**

**Second order in spin — measured, not argued** *(numbers corrected 2026-09-04; see below)*. Exact Kerr 220
frequencies from the `qnm` package; Taylor coefficients extracted by a high-degree Chebyshev fit and truncated
low. Identical to six digits across extraction range and degree:

| | χ = 0.69 (real remnants) | χ = 0.90 (EMRI central objects) |
|---|---|---|
| O(χ²) truncation error | **6.4%** | **18.9%** |
| successive error ratio at 3rd order | 0.53 | 0.72 |

> **Correction, 2026-09-04.** These were first recorded as ~4% and ~16% with a quoted spread. Those came from
> `np.polyfit` **at the truncation order**, which is a least-squares fit on the interval, not a Taylor
> truncation — it understates the error, and the "spread" was fit noise. Found by checking against
> [arXiv:2207.11267](https://arxiv.org/abs/2207.11267) (Pierini & Gualtieri, *PRD* **106**, 104009), whose
> Eq. (44) puts the 1% crossings at ā ≈ 0.22 and 0.40 **[A]**; the corrected extractor reproduces them at
> 0.219 and 0.400, the old one gave 0.35 and 0.50. An analytic golden test on 1/(1−x) is what turned a
> disagreement into a bug rather than an argument.

Equal-mass non-spinning mergers give **χ_f ≈ 0.69 to ~1%** across NR codes
([astro-ph/0609172](https://arxiv.org/pdf/astro-ph/0609172), [arXiv:1305.5991](https://arxiv.org/abs/1305.5991))
**[A]**; our own GW250114 measurements agree (LVK 0.69, our coherent-package fit 0.730, our NPE 0.766).

**⇒ the answer splits by channel.** For **ringdown**, 6.4% truncation sits below our own σ(δ) ≈ 14% — *not*
the binding constraint, contrary to our first assessment, though the margin is 2.2× rather than the 3.2× first
recorded. For **EMRI**, 18.9% at χ ≈ 0.9 against phase accuracy over ~10⁵ cycles is disqualifying, and there
the remedy is non-perturbative in spin.

**What the field itself does about the spin truncation, verified 2026-09-04.** Two things worth having on
record before anyone plans an O(χ²) sGB calculation:
- The second-order EdGB QNM paper ([arXiv:2207.11267](https://arxiv.org/abs/2207.11267) **[A]**) calibrates
  its accuracy **on Kerr and extrapolates** — *"an indication that a second-order computation of QNMs may be
  accurate for ā ≲ 0.4 (ā ≲ 0.7 with Padé resummation) for EdGB gravity **as well**"*. That "as well" is the
  same proxy step our own open item flagged: the sGB correction's own spin series is **not** measured there.
  They also need **Padé resummation** to reach ā ≈ 0.7 at all, which is not what a rapidly converging series
  requires.
- The truncation is no longer the only option. **METRICS**
  ([arXiv:2406.11986](https://arxiv.org/abs/2406.11986), *PRD* **110**, 064019 **[A]**) computes sGB QNMs
  **non-perturbatively in spin** to a ≤ 0.85 (accuracy ≤10⁻⁵ for a ≤ 0.6, ≲10⁻² for 0.7 < a ≤ 0.85), with sGB
  corrections carried to ~40 orders in spin and no Padé needed for a ≥ 0.7. ⇒ **the coefficient-decay question
  is answerable by comparison rather than by fitting**, which matters because our own work shows fitted
  coefficients beyond n ≈ 3 are not recoverable.
- Separately, on the **coupling** axis: [arXiv:2412.09377](https://arxiv.org/abs/2412.09377) **[A]** finds
  weak-coupling approximations deviating by up to ~55% near the domain limit ξ ≈ 0.316. That is far above the
  coupling our observational bound allows, so it does not bite here — but it is the reason the coupling
  expansion cannot be waved through at arbitrary ξ.

**EMRI non-integrability — the observable, and our scope limit (verified 2026-09-05).** Asked whether a
structural non-existence result (no irreducible Killing tensor in a deformed-Kerr sGB metric ⇒ orbits not
integrable) is observable, and at what SNR. **Recorded because the honest answer is a refusal:** EMRIs are a
LISA source — millihertz band, 10⁴–10⁶ cycles accumulated over years **[S]**. We work 50–1024 Hz on LIGO
strain, hold no EMRI data, and have no instrument that could. Any SNR figure from us would be an analogy
from subsolar machinery that happens to fit superficially, which is the failure mode this repo spends most
of its time catching. **The question is well-posed; it is simply not ours.**

What the observable *is*, for whoever does hold the machinery: non-integrability shows up as **gravitational-wave
"glitches"** — abrupt frequency jumps as the orbit crosses **Birkhoff resonance islands** — and as a chaotic
signal developing dense inter-peak power in place of a discrete frequency comb
([arXiv:2103.05643](https://arxiv.org/abs/2103.05643) **[S]**;
[arXiv:2604.06053](https://arxiv.org/abs/2604.06053), *Probing Kerr Symmetry Breaking with LISA EMRIs*,
**[S]** — pointer only, abstract not read).

**What our own number does and does not say about such a proof.** If the metric is the second-order-in-spin
one — as every sGB QNM paper checked above is — then the object differs from the physical one by **18.9%** at
χ = 0.90 (measured, and externally validated). But the *direction* of a non-existence result makes it robust
to that: finding a hidden symmetry **present** in a truncation would be the fragile claim, since a symmetry
can appear by accident at finite order; finding it **absent** is stable, because restoring one by adding
higher-order terms requires fine-tuning nothing supplies. ⇒ **the truncation bounds what such a proof
licenses rather than threatening it** — exact for the truncated metric, strongly suggestive for the physical
one, and not a proof of the latter. **The general form, which is the transferable part: exactness of a
computation says nothing about fidelity of the object computed on.** An exact null has no error bar on
"does *this* metric admit the tensor" and none at all on "do real orbits fail to be integrable" — and the
second is the sentence that travels.

**What is NOT established, after three methods tried.** Whether the asymptotic error ratio tends to χ (radius
of convergence R = 1, the extremal singularity) or to something below it. **Only three Taylor coefficients
are recoverable** from numerically-computed Kerr frequencies at this precision — polyfit, high-order finite
differences, and Chebyshev all fail at the same orders. A degree sweep settles it: n=2 and n=3 drift by
0.000 and 0.002 across fit degrees 12–28, n=5 drifts by 1.06, n=6 by **124**. *A quantity three independent
methods cannot extract is not a hard extraction — it is absent.*

**Method warning worth more than the physics.** Each failure wore the costume of its own method: a wandering
|a₅|, a 10⁵ blow-up, a degree-dependent plateau. **The plateau was the most dangerous because it looked like
convergence** — a blow-up announces itself, a plateau recruits you. Consequently: **extracting series
coefficients from numerics is inference with an unquoted error bar; evaluating a closed form at points is
verification.** They are different operations and only the second is safe.

**Does an exact beyond-GR metric move our δ floor? No.** 7 of 8 events return the prior (σ/prior 0.93–0.99)
— an *information* limit, fixed by SNR and not by modelling. GW250114, our one informative event, sits at the
crossover: start-time systematic 0.175 peak-to-late against statistical σ ≈ 0.14–0.23. An exact ω_lmn(ε)
improves *interpretation* (constrain α rather than a phenomenological δ) but adds no SNR and cannot say when
linear ringdown begins. **It would not help.**

**Open item, filed:** run the coefficient-decay check on the **actual sGB spin expansion** — and on
**analytic coefficients if the theory provides them, never fitted**. This thread is the evidence for that
qualifier.
