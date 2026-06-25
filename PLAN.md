# DeepStrain — execution plan (knock-out tracker)

> Granular, do-able backlog mined from all sub-project docs (2026-06-23) + new cross-cutting
> angles. Complements [ROADMAP.md](ROADMAP.md) (higher-level moves) and the per-sub-project lab
> notebooks. **Discipline:** north-star — each item is real only once stress-tested + gated +
> documented. Tackle one at a time; mark `[x]` when its gate is green.
>
> **Status key:** 🟢 tractable now · 🟡 tractable, needs data/VM · 🔴 blocked by a known wall
> (see "Parked" — discuss before attempting). Skipping all 🔴 per the user's call.

---

## Tier 1 — quick wins (bounded improvements to existing gated results, all local)
- [x] **E1 · Echoes: ML scorer on the upper limits** ✅ **DONE — honest NEGATIVE (2026-06-25).** Two findings:
      **(gotcha)** the whitened-domain `11` harness reproduces the known v5 *convention artifact* (~13× fake
      tightening) — must NOT be read as a limit. **(result)** through the honest production path
      (`12_ul_production.py`, reuses `09`/`10`), at the predicted Δt **A90 comb=1.43σ vs ML=1.45σ → 0.98×: the ML
      scorer does NOT tighten the exclusion.** v5's ~1.2× edge is at the 50% point; it vanishes at the 90% level
      (ML's heavier-tailed background can't reliably reach 90% — NaN A90 at 4/12 spacings). ⇒ **corrects the
      ROADMAP "~1.2× tighter" note.** Gated in verify.sh; the gated comb UL (`11`) stands as the honest best.
- [x] **E2 · Echoes: independent background blocks** ✅ **DONE (2026-06-25).** `13_independent_bg.py` pools
      off-source blocks at ±0.5h…±16h from GW150914 (each own-PSD whitened) → **660-pair independent background
      (4× the 159 shared-block)**. The null HOLDS: A p=0.319 (vs 0.375), B p=0.977 (vs 1.000) → non-detection is
      robust to the stationarity/shared-block assumption. Gated. (O1 duty cycle killed many offsets — handled.)
- [ ] **R1 · Ringdown: per-parameter recalibration** 🟢 — v3 fit one global temperature T; fit per-parameter
      (M, χ, δ) temperatures. *Done =* per-param held-out coverage ∈ [0.85,0.95], gated.

## Tier 2 — medium, self-contained (local)
- [P] **R2 · Ringdown: explicit Bayesian tone-count model selection** 🔴 **ATTEMPTED → PARKED (2026-06-25).**
      Built the linear-Gaussian evidence (`14_bayes_tonecount.py`, amplitudes marginalized analytically,
      start-time marginalized). **Oracle diagnostic: NO 1-tone/2-tone separation at any σ_a** — the 220/221 are
      near-degenerate over the 0.04 s segment (the `06`/v4 wall). **But the published GW250114 analysis DOES
      detect the overtone** → my simplified machinery (white-noise likelihood, independent amplitudes, flat prior)
      isn't a *fair* test; the failure is an implementation limit, NOT the info limit. Won't claim a false
      negative. **Needs the proper FD coherent pipeline (`ringdown` pkg, Python 3.11, deferred).** Not gated.
- [ ] **E3 · Echoes: per-event scorers + per-event Δt + broaden on-source set** 🟢 — train a scorer per event,
      compute Δt from each event's catalog mass+spin, run on more O3 events. *Done =* a small table of honest
      per-event nulls/limits.
- [x] **R3 · Ringdown: IMR-waveform referee** ✅ **DONE (2026-06-25) — found a real systematic.** Injected
      realistic full-IMR ringdowns (IMRPhenomXAS via pbh's pycbc, NR-calibrated) into the no-hair NPE
      (`15_imr_referee.py`). The NPE is unbiased on its analytic-tone training family (control δ=+0.02) but
      carries a **δ≈−0.33 SYSTEMATIC on realistic ringdowns injected from the peak** — and the start-time sweep
      shows it **decays to ~0 by 6 ms post-peak** (−0.33→−0.01), proving it's the early-time
      merger-transition/overtone content the 220+221 model omits (NOT a convention artifact). This independently
      **quantifies the start-time systematic** at the heart of the no-hair controversy, and is a caveat on the
      09 GW250114 δ. Gated. (True SXS NR deferred — IMRPhenomXAS is NR-calibrated + needed no download.)

## Tier 3 — new angles (separate directions, the interesting ones)
- [x] **N1 · 🌟 Joint ringdown ↔ echo analysis** ✅ **DONE (2026-06-25).** `15_joint_ringdown_echo.py`: propagated
      GW250114's ringdown M,χ posterior through the verified Δt(M,χ) → echo Δt prior **0.357s [0.304,0.445]**, used
      it to condition the echo search on the SAME event. Conditioned window = 70/225 grid pts (**3.2× fewer
      trials**) → p<0.01 threshold −22% → **1.11× more sensitive** (A90 1.90→1.72σ, A50 1.60→1.44σ). On-source
      GW250114 **null** under both. **First echo search conditioned on the event's own ringdown mass.** Honest:
      the 3.2× trials cut gives only ~1.1× amplitude sensitivity (steep efficiency curve). Gated.
- [x] **N2 · H1×L1 consistency in echoes** ✅ **DONE — honest MIXED/modest (2026-06-25).** A *learned* head
      (Build C-2 style) would overfit the tiny echo data, so tested a robust training-free consistency statistic
      `combH+combL−λ|combH−combL|` (penalize per-Δt detector disagreement). `17_echo_consistency.py`, pre-chosen
      λ=0.5 (no best-λ selection bias), n=200, bootstrap ΔA90. **Significant for GW150914** (A90 2.05→1.97, ~4%,
      90%CI[−0.19,−0.02]) but **NOT for GW250114** (CI[−0.00,+0.05], P=0.08) ⇒ event-dependent, **not a robust
      universal win**. (The smoke's "~10% best-λ" was selection-bias-inflated — caught it.) The comb-sum is
      already largely Δt-consistency-aware. Gated (the honest "modest, not-universal" conclusion).
- [x] **N3 · Stacked multi-event echo search** ✅ **DONE (2026-06-25).** `16_stacked_echo.py`: stacked the comb
      statistic across 4 well-characterized events (GW150914/GW151226/GW151012/GW250114) at each event's
      formula-predicted Δt (z-scored per event, summed). **Population NULL** (stacked on-source z=−5.17 vs +5.05
      threshold, p=0.998 — no echo excess); combined upper limit **A90 1.43σ, 1.21× tighter than best single**
      (below √N=2× — heterogeneous events + GW151226 low-sensitivity + equal-weight stacking). Gated.
      (Full O3/O4 catalog deferred — per-event detector-frame remnant masses + fetches; the 4-event stack is the
      tractable core, no new data.)
- [x] **N4 · Self-supervised noise-embedding backbone** ✅ **DONE — a clean data-wall WIN (2026-06-26).**
      `ssl_pretrain.py`: masked-spectrogram autoencoder pretrains the SpectrogramCNN conv backbone on 20k
      UNLABELED noise spectrograms (MSE 1.05→0.75 — it learns the noise's t-f structure). `ssl_finetune.py`:
      fine-tune that backbone vs from-scratch at a reduced labeled budget (3 seeds). **SSL wins at every budget,
      gain ∝ 1/labels:** +0.124 AUC @1000 labels (0.539→0.663, ~10× seed scatter — highly significant), +0.021
      @4000 — textbook data-wall signature. Gated. **Honest caveats:** unlabeled pool = the labeled set's 20k
      noise windows (more O3 noise → likely more gain — a VM extension); metric is val AUC, not yet sensitive
      distance (the headline pbh metric — next); SSL *mitigates* the wall (scarce-label AUC 0.66 < full-data 0.79),
      doesn't break it. Landed in one session, not multi.
- [ ] **N5 · Triple-detector H1×L1×V1** 🟡 — add Virgo; the learned-consistency statistic extends to 3
      detectors. Medium payoff (Virgo less sensitive). 

---

## Parked — blocked by a known wall (discuss before attempting) 🔴
- **PBH dense template bank / true-waveform front end** — subsolar needs ≤0.1 % Mc spacing (~1,600+ templates);
  intractable without serious GPU/cloud compute. Blocks the real-MF detector + fine-timing coincidence.
- **PBH lower-FAR → 1/decade** — tractable but needs the VM back on + a fresh coincident-data fetch (~2 h, VM cost).
  *Not blocked, just deferred to a VM session.*
- **Ringdown real multi-event δ-stacking** — SNR information wall: only GW250114-class loudness measures δ (v6 mapped it).
- **Ringdown black-box tone-count** — parked honest-negative; guardrail: don't re-throw ML architectures at it.

## 🔑 Shared unblocker ✅ DONE (2026-06-25): the echo Δt(M,χ) formula — VERIFIED + caught a data bug
`14_echo_spacing.py`: the Kerr-tortoise echo spacing Δt = 2[r*(r_peak) − r*(r_mem)] from first principles
(Abedi 2017 Eq. 2; membrane at 1 ℓ_P, barrier at 3M). **Uncalibrated, it reproduces all 3 Abedi Table-I values
to <2%** (GW150914 0.297 vs 0.2925; GW151226 0.101 vs 0.1013; LVT151012 0.179 vs 0.1778). Float gotcha fixed
(δ~1e-85 s underflows in r₊+δ → use ln(δ) analytically). Gated. **BUG CAUGHT: the repo's hardcoded Δt were
wrong** — GW151226 was 0.0579 (→ 0.1013) and LVT151012 was 0.1013 (→ 0.1778, a shifted/mislabeled value); a
prior session's "correction" of GW151226 to 0.0579 was itself the error. Fixed in 11/run_event/13.
**⚠️ Follow-up: the GW151226 + LVT151012 prior results (upper limits, run_event) were computed at the WRONG Δt
→ re-run them.** Now unblocks N1 (propagate GW250114 M-posterior → Δt prior), E3, N3.

## Chosen execution order (driving it; we do them all)
**E1 ✅** → **E2** (harden the headline nulls) → **R2** (Bayesian tone-count, field's method vs our failed ML) →
**N2** (reuse learned coincidence) → **N1** (flagship, focused effort) → **E3** → **N3** → R3 → N4 → N5.
**R1 deprioritized to the tail** — low-value (v3's global T already calibrates each param to 0.90–0.92).
