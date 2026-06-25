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
- [ ] **R3 · Ringdown: SXS / NR-waveform injection referee** 🟡 — inject *full-merger* numerical-relativity
      waveforms (SXS catalog) instead of analytic tones; re-check the no-hair pipeline is unbiased. *Done =*
      δ recovered unbiased on NR injections (the strongest referee we can build).

## Tier 3 — new angles (separate directions, the interesting ones)
- [ ] **N1 · 🌟 Joint ringdown ↔ echo analysis** 🟢 — echo spacing Δt ≈ 8M·ln(M/ℓ_P) depends on the mass that
      *ringdown measures*. Use the ringdown M-posterior (tight for GW250114) to set a narrow, physical Δt prior
      for the echo search on the **same** event. Couples two sub-projects; GW250114 is the test case. *Done =*
      the M-conditioned echo search on GW250114, with its sensitivity vs the flat-Δt-prior version.
- [ ] **N2 · 🌟 Reuse the learned H1×L1 consistency statistic (Build C-2) in echoes** 🟢 — the "does H1 agree
      with L1" head is general, not subsolar-specific. Apply cross-detector consistency to the echo structure.
      *Done =* learned-consistency echo statistic vs the single-detector one, stress-tested for leakage.
- [ ] **N3 · Full O3/O4 catalog harvest** 🟡 — run the *validated* echo + ringdown pipelines on the whole event
      catalog with pre-registered per-event Δt / δ. *Done =* a population of honest nulls (or a candidate).
- [ ] **N4 · Self-supervised noise-embedding backbone** 🟡 — pretrain on unlabeled O3 noise, fine-tune for
      detection; attacks the "more data" wall shared by PBH + echoes. *Done =* fine-tuned > from-scratch at
      matched data.
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
