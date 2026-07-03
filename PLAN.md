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
- [x] **R1 · Ringdown: per-parameter recalibration** ✅ **DONE — gate-passing but does NOT beat global T
      (2026-06-26).** `17_recalibrate_perparam.py`: fit a separate T_M/T_chi/T_delta (one T-sweep suffices —
      widen() rescales each column independently, so param j's coverage depends only on T_j). Per-param T =
      1.10/1.05/1.05 → held-out coverage **M 0.94, χ 0.92, δ 0.90, each in [0.85,0.95]** (PLAN criterion met).
      BUT it does **NOT** beat v3's single global T=1.05 (per-param mean|cov−0.90| **0.020 vs global 0.011**) —
      the per-param fit slightly **overfits the n=600 calibration-set coverage noise** (T_M=1.10 overshoots M to
      0.94). ⇒ **confirms v3's global temperature was the right, sufficient choice; per-param tuning adds nothing.**
      GW250114 δ unchanged: −0.16 [−0.46,+0.33], Kerr-consistent. Gated (each-in-band + global≤per-param mad). The
      honest low-value result the PLAN anticipated.

## Tier 2 — medium, self-contained (local)
- [x] **R2 · Ringdown: explicit Bayesian tone-count model selection** ✅ **CLOSED (2026-07-02) — the proper
      pipeline DETECTS the overtone; our parked non-claim was right.** The Py3.11 wall fell (uv `.venv311`;
      ringdown 1.0.0 needs era-matched pins, frozen in `.venv311-pins.txt`). `20_extract_strain.py` +
      `21_ringdown_crosscheck.py` run the field-standard FD coherent pipeline on verified targets. **(a)**
      GW150914 validation in-band (M 77.5, χ 0.76). **(b) GW250114 220+221: A221 bounded away from zero
      (P = 0.000, A221/A220 = 1.02 at peak)** — matches arXiv:2509.08099, where our simplified white-noise
      machinery (`14`) saw nothing ⇒ implementation limit POSITIVELY demonstrated. GW150914 comes out marginal
      (P = 0.049), consistent with the contested literature. **(c) NPE referee: package M 74.8/χ 0.729 vs our
      09 NPE 76.0/0.762 — medians within 1.2 M☉/0.033, package CI nests inside NPE's ⇒ first independent
      field-standard cross-validation of the whole NPE arc.** NUTS x64, R̂ ≤ 1.004, ESS ≥ 950. Gated (a+b+c+
      convergence).
- [x] **E3 · Echoes: per-event scorers + per-event Δt + broaden on-source set** ✅ **DONE — all clean nulls
      (2026-07-02).** `19_per_event_ml.py`: a per-event autoencoder scorer (per detector, trained on the event's
      own off-source), the v2 ML network comb run at each event's formula-Δt, across the broadened set
      GW150914 / GW151012 / GW151226 / **GW250114** (Δt from its verified remnant M_f=68.1/χ=0.68 → 0.2952 s, a
      GW150914 twin). Background from the **independent ±hour blocks** (E2-style, own-PSD whitened, 660–1815 segs).
      **All four events are clean nulls under both the ML scorer and the comb** (ML p 0.13–0.99, comb p 0.23–0.97).
      Robustness payoff: a first pass on the tiny own-block background (n=59) threw up GW151012 ML p=0.033 —
      which **dissolved to 0.130 against the larger independent background**, a textbook small-sample artifact
      (and the comb never flagged it). The independent background also **rescued GW151226** (own-block was
      NaN-cropped to too few segments). Gated.
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
      **Sensitive-distance follow-up ✅ (translates to the headline metric).** `ssl_sensdist.py`: the AUC win
      reaches SENSITIVE DISTANCE too — at a defined (1%) FAR, SSL gives +0.278 distance-fraction @2000 labels
      (from-scratch non-functional at 0.000) shrinking to +0.01 @8000 (the data-wall signature again). At the
      *strict zero-FA* threshold both are 0 — a model-strength floor (needs near-full-data AUC ~0.79), not an SSL
      failure. So it's a real *detection* win, not an AUC artifact. Gated.
- [x] **N5 · Triple-detector H1×L1×V1** ✅ **DONE — honest NEGATIVE: Virgo does NOT help subsolar (2026-06-27).**
      `coinc_triple.py` extends the G1 double-coincidence to a 3rd detector (cnn_w64 on H1+L1+V1, 3-way time-slide
      matched-FAR background, injections projected onto all 3). Local H1∩L1 test segs are ALL Virgo gaps → discovered
      20 true H1∩L1∩V1 segments (intersecting 3 DATA flags), ran on 4 leakage-free ones. **(1)** Double H1×L1
      reproduces the win on fresh data (**1.33× over single**, validates G1/Build-C). **(2)** Triple H1×L1×V1 = **0.94×
      double — Virgo marginally HURTS.** Mechanism: V1's signal responsiveness is **+1.2 vs H1 +5.1 / L1 +7.4 (~19%)**
      — too insensitive at subsolar to carry signal, so summing its near-noise score + the higher 3-way threshold
      degrades the statistic. Rules out the learned-triple too (no V1 signal to weight → ≈double at best). **H1×L1
      double-coincidence remains the subsolar ceiling.** Gated. Eng note: per-segment checkpointing survived repeated
      power losses + service interruptions.

---

## Follow-up arc (added 2026-07-02) — the post-backlog extensions, tackled in order
> Direction discussion 2026-07-02: four extensions chosen; A first. GPU VM not available → **A runs on the
> M4 Mac** (10 cores / 16 GB / MPS), made tractable by three insights: the bank is effectively **1-D in Mc**
> (extrinsic params proven irrelevant by coinc_check), **512-s chunks** contain the full in-band chirp (the
> lightest Mc=0.17 chirp is ~360 s from 50 Hz), and **MPS-batched FD correlation** (hand-rolled torch MF,
> golden-tested against pycbc — Phase-1 style). Injection side may use a nearest-in-Mc template subset
> (bank is 1-D); the NOISE/FAR side always uses the full bank.

- [x] **A · pbh: REAL dense-bank matched filter on the Mac — the honest ML-vs-MF benchmark** ✅ **DONE — the
      CNN TIES a realizable dense bank; both bank-mismatch-limited (2026-07-03).** A1 golden-tested MPS MF
      (`pbh/bankmf.py`); A2/A2b: full-coherence is megatemplate-scale (FF collapses; matches LVK's real 3.45M
      O4 subsolar bank), but the n=8 semi-coherent statistic is tractable and its recovery-vs-spacing curve
      (`bank_semiff.py`) sets ~0.1%/1,619 templates. A3/A5: `bank_dense.py` (0.1% bank on 6 real segments, mid-
      segment checkpointed — survived 2 power losses + a restart) + `bank_vs_cnn.py` (cnn_w64 on IDENTICAL
      injections). **Result: real bank MF 0.489 vs CNN 0.472 = 1.03× (a TIE); density sweep 83→0.000 (reproduces
      bank_oracle) … 1619→0.489 (the wall, quantified); both << true-template oracle 0.72 ⇒ bank MISMATCH is the
      dominant loss, not learned-vs-MF.** Co-injection shrank an apparent ~10% win to a ~3% tie (prevented an
      overclaim). Gated. Stretch A6 (fine-timing coincidence + hybrid trigger→verify) deferred — the tie already
      answers the core question. **The learned CNN is not leaving meaningful sensitivity on the table vs any
      realizable detector.**
      The one question the whole v2 arc bumped into: our learned pipeline reaches ~45% of the *idealized* MF —
      but nobody (verified in the prior-art audit) has published learned-vs-REAL-MF for minutes-long subsolar
      signals. Steps: (A1) golden-test spike — torch-MPS FD matched filter vs pycbc on injected signals, SNR
      match <1%; (A2) build the ~1,650-template 0.1%-Mc bank (geomspace, full 512-s in-band templates,
      generated in batches); (A3) full-bank noise scan on the test/coinc segments → zero-FA + matched-FAR
      thresholds (the real-MF FAR floor); (A4) injection recovery (2,400 inj, nearest-40-in-Mc subset +
      spot-check vs full bank on a subsample) → real-MF sensitive distance; (A5) the benchmark table:
      cnn_w64 single / H1×L1 coinc / REAL MF / idealized MF, same segments, same FAR. Stretch: (A6) fine-timing
      coincidence + hybrid ML-trigger→MF-verify. *Done =* benchmark table gated; every claim vs the same
      noise + FAR convention.
- [~] **B · ringdown: ride the GW250114 toehold with the field-standard package** 🟢 (B1 done)
      - **B1 ✅ DONE (2026-07-03) — package refereess R3's start-time systematic.** `22_starttime_sweep.py`:
        GW250114 220+221 fit at 9 start offsets (0–16 t_Mf, t_Mf=0.335 ms). **The overtone is significant from
        the peak (P(A221≈0)=0.000) and damps away by ~5.4 ms (P→0.059)** — a real fast-damping τ221≈1.4 ms mode.
        **The peak-start mass is biased HIGH (74.7 vs the true 68.1 M☉, +10%) and drifts −8.8 M☉ as the start
        moves later** — the early-time systematic R3 found with our NPE, now independently reproduced by the
        coherent package (all rhat<1.01). Gated.
      - **B2 🅿️ PARKED-HONEST (2026-07-03) — tool inadequate, would be a false negative.** The nonlinear claim
        (arXiv:2601.05734) is **6 quadratic modes in the (4,4) multiplet** (220×22n coupling), BF 74 at 5 M_f,
        zero-amplitude excluded at 3σ, reconstructed at 6–10 M_f. Refereeing it needs (a) multi-multipole strain
        modeling [(2,2)+(4,4) simultaneously] and (b) frequency-LOCKING the quadratic mode to 2×f220 (=497 Hz,
        6.9% below the linear Kerr 440 at 534 Hz). Verified the vanilla `ringdown` package supports **neither**
        (only linear (2,2) Kerr QNMs; its "nonlinear/quadratic" source hits are the sampler nuisance params +
        peak-interpolation, unrelated). Wang & Ma used bespoke PyCBC-Inference templates + 30k live points. A
        vanilla (2,2)-QNM fit would null a genuinely-present-but-subtle (4,4) signal ⇒ **false negative; won't
        ship it** (R2 discipline). Come-back-later = the custom multi-multipole pipeline.
      - **B3 ✅ DONE (2026-07-03) — NPE loop closed.** `23_npe_package_loop.py` (synthesis of 09+21+22): (1) the
        NPE (M 76.0 [68.4,85.2], χ 0.762) agrees with the package (74.8 [70.4,79.0], 0.729) — gap 1.2 M☉, package
        CI nested in the NPE's ⇒ the amortized net does real, field-consistent inference; (2) the NPE median sits
        at ~0 t_Mf (the peak) in B1's start-time family ⇒ it weights the high-SNR early regime and **inherits the
        R3/B1 peak-start systematic** (+7.9 M☉ vs the true 68.1, matching the package's +6.6 peak bias). The NPE
        isn't bias-free from marginalizing t0 — it effectively infers from the peak. Gated.
      **B COMPLETE:** B1 ✅ + B3 ✅ gated; B2 honestly parked (tool-inadequate). Next: D (event watcher).
- [x] **D · event watcher: turn the stack into a standing instrument** ✅ **DONE (2026-07-03).** `watch_event.py`
      (repo root) orchestrates 3 stages across 3 venvs → a one-page markdown+JSON report: (1) ringdown remnant +
      overtone [package, venv311: `watch_ringdown.py`]; (2) no-hair δ + Kerr consistency [amortized NPE, .venv:
      `watch_npe.py`]; (3) echo Δt(M_f,χ_f) prediction + comb p-value [echoes venv: `watch_echo.py`]. Each stage
      isolated (one failing doesn't sink the others); event registry with verified sky-loc + remnant. **Reference
      run on GW250114 in 48 s reproduces ALL three sub-project headlines** — ringdown M 74.7+overtone (=21), NPE
      δ −0.15 Kerr-consistent (=09), echo Δt 295 ms p 0.33 null (=E3). Amortized NPE → seconds/event; ready for
      O4b/O5 GW250114-class events. Gated (the report reproduces the committed headlines). Artifacts:
      watch_GW250114_082203.{md,json}.
- [ ] **C · consolidate & release (capstone, last)** 🟢 — reproducibility pass (env pins everywhere, verify.sh
      as the CI gate), top-level honest summary (wins AND negatives AND benchmarks), Zenodo/GitHub release.
      *Done =* a stranger can reproduce the headline numbers from a fresh clone.

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
