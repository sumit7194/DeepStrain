# Lab notebook — ringdown spectroscopy

*Chronological record of what was run, what came out, and what it means.
Plain-language; every claim tied to a script + artifact in this repo.*

---

## 2026-06-10 — v1 built end-to-end

### 01 — data plumbing ✅
`01_fetch_data.py` fetches public strain, whitens, finds the per-detector peak.
- GW150914: H1 peak GPS 1126259462.4231 — matches the literature's .423. Chirp +
  ringdown clearly visible.
- **GW250114 strain is public** (`GW250114_082203-v2`): the loudest BBH ever (SNR≈80).
  Whitened peak amplitude ~7.2 vs noise rms ~0.35 — the ringdown decay is visible *by
  eye* for ~20 ms. Plots: `plots/01_*.png`.

### 02 — Kerr tone tables ✅ (validated)
`02_qnm_predictions.py`, via the `qnm` package. For a GW150914-like remnant
(M=68 M_sun detector-frame, chi=0.69):
- **220: f=251.0 Hz, tau=4.13 ms** — matches published GW150914 ringdown values.
- 221 overtone: f=245.4 Hz, tau=1.36 ms — nearly the same pitch, dies 3x faster.
  *That one line is the whole difficulty of the field.*

### 03 — classical one-tone fit ✅ (with honest failure)
`03_fit_220.py`: joint H1+L1 least squares (shared f/tau, per-detector amp/phase),
t0 = peak+3 ms.
- GW150914: f=251.3 Hz (right), tau=7.6 ms (way high) -> implied M=91, chi=0.93 (wrong).
- GW250114: f=238 Hz, tau=5.3 ms -> M=80, chi=0.81 (high).
- Diagnosis came from the referee (04), not from tuning on the events.

### 04 — injection harness ✅ (the referee; found two pipeline bugs)
`04_injections.py`: known Kerr 220 tones injected into real off-source noise, identical
pipeline.
- **Bug 1 (found & fixed):** injections were ~2/3 the real event's whitened loudness
  (calibrated against the real peak: A=1.5e-21 -> GW150914-like, 4.0e-21 in O1 noise /
  2.0e-21 in O4 noise -> GW250114-like).
- **Bug 2 (found & fixed):** fitting *bandpassed* data inflates tau — the zero-phase
  (filtfilt) bandpass smears the decay in time. Fix: fit on **whitened-only** data
  (noise is then white = exactly the least-squares assumption); bandpass for plots only.
- **Bug 3 (found & fixed):** blind random-restart frequency search latches onto
  broadband noise; FFT-peak-seeded initialization fixed it.
- Final calibration, GW150914 loudness: f unbiased but tau skewed +30% with huge
  scatter; **(M, chi) from one quiet event is untrustworthy — measured, not assumed.**
- Final calibration, **GW250114 loudness: f=249.6±8.3 Hz (true 251), tau=4.35±0.58 ms
  (true 4.13) -> M=69.8±6.1 (true 68), chi=0.69±0.13 (true 0.69). Pipeline validated.**

### 05 — the poisoned choice, reproduced ✅
`05_start_time.py`: scan ringdown start time t0 = peak+0..10 ms, one-tone fit.
- GW150914: inferred M wanders 67→105, chi 0.66→0.98 across t0; disintegrates ≥7 ms.
  *This is the Isi/Farr-vs-Cotesta controversy, reproduced on a laptop.*
- GW250114: early drift 0-2 ms (the overtone contaminating the one-tone model —
  visible!), stable plateau 3-8 ms averaging right at published (M≈68, chi≈0.69),
  noise takeover ≥9 ms.

### 06 — free two-tone fit: an honest negative result ✅
`06_no_hair.py`: fit 220+221 fully free at the peak; calibrate identically.
- **13/14 calibration injections railed at the search bounds for the overtone** —
  fully-free least squares CANNOT separate two tones 6 Hz apart when one dies in
  1.4 ms. The real-event tone-2 numbers are therefore meaningless, *and we know it
  because we calibrated.* This is precisely why the professionals use informed
  Bayesian methods. Negative result, fully documented.

### 07 — the no-hair test, parameterized ✅ (v1 headline)
`07_no_hair_locked.py`: the LVK-style test in simplified form — both tones LOCKED to
Kerr(M, chi), overtone frequency scaled by (1+delta); refit (M, chi, amps, phases) at
each delta on a grid; injections with true delta=0 calibrate the precision.
- Anchor (220-only @ peak+3ms on GW250114): **M=70.8, chi=0.68** — right at published.
- Real event: **delta_best = -0.16**.
- Calibration (O4 noise, matched loudness, n=12): delta_hat = +0.09 ± 0.36 — many
  injections rail at ±0.4: **σ(delta)≈0.3-0.4 is the ceiling of this classical
  least-squares method** at this loudness (LVK's full Bayesian gets tens of percent).
- **VERDICT: |delta| = 0.16 < 2σ -> the overtone sits where Kerr predicts, within the
  (large, honestly measured) precision of this method. No-hair: PASS.**

### 08 — start-time-marginalized SBI ✅ (the novelty prototype)
`08_sbi.py`: the angle the field hasn't done cleanly — the simulator randomizes the
ringdown start time (t0 ~ U(0,6) ms, never shown to the network), so the neural
posterior p(M, chi | segment) is **marginalized over the poisoned choice by
construction**. CNN embedding (2 det × 164 samples) → MAF flow (sbi NPE), 60k sims,
35 epochs on CPU. Posterior cached at `results/08_posterior.pt`.
- **Referee (i)** — coverage on 200 held-out sims: 0.76 vs ~0.81 expected (mildly
  overconfident; v2: more sims/calibration).
- **Referee (ii)** — the critical one — injections into REAL O4 noise (the network
  never saw real noise): **M = 71.1±6.9 (truth 68), chi = 0.71±0.12 (truth 0.69)** —
  unbiased. The whitened-Gaussian training approximation holds, and the start-time
  marginalization works.
- **Bug caught by referee (ii) en route:** first apply-run fed the network a segment
  starting 2 ms BEFORE the peak — which contains merger (frequency still sweeping),
  while the simulator only makes pure post-start tones. Posterior read (85, 0.85) —
  *model mismatch, diagnosed because injections (which match the simulator) stayed
  unbiased.* Lesson: **an NPE is only as good as its simulator's realism.**
- **GW250114 result** (segment starting at the peak): M = 78.8 [68.6, 89.5],
  chi = 0.79 [0.64, 0.89] (90%). **Covers the published values (~69-70, ~0.69)**;
  center pulled ~+10% high — residual near-peak content (higher modes etc.) the
  two-tone simulator omits. Honest v2 item, not hidden.
- **Lovely emergent detail:** the posterior cloud lies along a diagonal M-chi ridge —
  the network REDISCOVERED the physical degeneracy (heavier + faster-spinning rings
  at the same pitch) from data alone.

## 2026-06-12 — v2 PRE-REGISTRATION: the network IS the no-hair test

Design: extend 08's NPE from θ = (M, χ) to θ = (M, χ, δ), with δ the 07
deviation parameter (overtone frequency × (1+δ), τ untouched), prior
δ ~ U(−0.5, 0.5). Simulator otherwise IDENTICAL to 08 (whitened domain, two
tones, t0 ~ U(0,6) ms and per-detector amps/phases as hidden nuisances, unit
white noise; segment starts at the peak). N_TRAIN = 90k (08's 60k showed mild
overconfidence at 2 params; 3 params need more). The trained network outputs
the no-hair posterior directly, start time marginalized by construction — the
v1 novelty angle, completed.
Gates:
(R1) calibration: per-parameter 90% coverage in [0.85, 0.95] on 200 held-out
sims, including δ specifically (08's was 0.76 joint — v2 must not be worse).
(R2) real-O4-noise injections at GW250114 loudness: (a) δ_true = 0 (n=6) —
δ posterior median bias |b| < σ_NPE/2; report σ_NPE(δ) against the classical
ceiling σ ≈ 0.36 from 07 (hope: comparable or tighter); (b) δ_true = +0.3
(n=6) — the network must DETECT the violation (posterior median > 0 in ≥5/6,
covering 0.3).
(R3) GW250114: report δ posterior [median, 90% CI]; compare 07's classical
(δ_best = −0.16, 2σ ≈ 0.72) and check Kerr-consistency (δ = 0 inside 90%).
Known caveat carried forward: 08's +10% mass pull from near-peak content the
simulator omits — δ may inherit a related systematic; report, don't hide.

## 2026-06-12 — v2 FIRST-RUN RESULTS (gates) + fix round launched

90k sims, NPE over (M, χ, δ), referees per pre-registration:
- **R1 ~:** coverage M 0.83 / χ 0.86 / δ 0.88 — χ and δ in gate; M 0.02 below
  (binomial σ ≈ 0.021 at n=200 — marginal; consistent with v1's mild
  overconfidence). Triggers the one pre-registered fix round: N_TRAIN 90k→150k
  (running; first-run artifacts preserved as *_90k.*).
- **R2a ✓✓:** Kerr injections in real O4 noise: δ̂ = −0.021 ± 0.138 — unbiased,
  and the scatter is **~2.6× tighter than 07's classical ceiling (σ ≈ 0.36)**.
  The amortized network beats the grid-scan least squares at its own game.
- **R2b ✓ with honest shrinkage:** δ_true = +0.3 injections: 5/6 positive
  medians, CIs cover 0.3, but medians ≈ +0.09 — strong shrinkage toward the
  prior center; single-event δ at this loudness is weakly informative. The
  δ=0 vs δ=0.3 populations separate (−0.021±0.138 vs +0.086±0.059) —
  population-level detection, recorded as such.
- **R3 ✓ (the headline):** GW250114 amortized no-hair posterior:
  **δ = −0.13 [−0.42, +0.33] 90%, Kerr inside ✓** — and the point estimate
  lands on 07's classical δ_best = −0.16. Two independent methods (grid-scan
  least squares; start-time-marginalized NPE), one answer: the overtone sits
  where Kerr predicts. M = 75.1 [66.3, 84.5] — the known +10% near-peak pull,
  carried openly.

## 2026-06-12 — v2 FIX ROUND (150k) + CLOSURE

R1: M 0.88 ✓ (cured), δ 0.88 ✓, χ 0.84 (was 0.86) — the marginal miss moved
between parameters across runs: **stable mild overconfidence ~0.84–0.88,
unchanged by 90k→150k sims ⇒ not a sample-size problem; post-hoc recalibration
(SBC/temperature) is the v3 item.** R2a ✓ Kerr injections unbiased
(−0.05 ± 0.20). **R3 ✓✓: GW250114 δ = −0.16 [−0.45, +0.32] 90% — exactly the
classical point estimate (07: −0.16), Kerr inside 90%.** M = 76.0 [68.4,
85.2], χ = 0.76 (the +10% near-peak pull persists — v3 simulator realism).
**v2 CLOSED:** the amortized no-hair network works, agrees with the classical
method on the real event, beats it 2–2.6× on injection scatter, with honestly
quantified residual miscalibration.

## 2026-06-13 — v3 PRE-REGISTRATION: post-hoc recalibration

Design: the saved 150k posterior is mildly overconfident (coverage 0.84–0.88
vs 0.90, stable across sim counts ⇒ architectural). Fix: a single global
temperature T — widen samples about their median, s' = med + T·(s − med) —
fitted on 150 fresh sims to bring mean per-parameter 90% coverage to 0.90,
VALIDATED on 150 held-out sims. Gates: (T1) fitted T in a sane range
(1.0–1.6); (T2) held-out coverage for EACH of (M, χ, δ) in [0.85, 0.95]
(n=150 ⇒ binomial σ ≈ 2.4%, so this is ≈ ±2σ around target) with mean within
[0.88, 0.92]; (T3) GW250114 re-reported with the widened CI — δ Kerr-
consistency must be re-stated under honest calibration. Implementation note:
09 is a script (module-level flow), not importable — its simulator is
duplicated into 10_recalibrate.py with a keep-in-sync comment (recorded
debt: refactor 09 into lib+script later).

## 2026-06-13 — v3 RESULTS: calibrated and CLOSED

First round (n=300): T=1.00, held-out fail — **a 300-sim calibration cannot
resolve ~5% miscalibration (σ≈2.4%/param/half); lesson recorded.** Fix round
(n=1000, 600/400 split): **T1 ✓ T=1.05; T2 ✓✓ held-out coverage M 0.91 /
χ 0.92 / δ 0.90 (mean 0.911 — on target); T3 ✓ GW250114 recalibrated
δ = −0.16 [−0.46, +0.33], Kerr-consistent (essentially unchanged).** Side
finding: the fresh batch's RAW coverage (0.87–0.90) shows the original 200-sim
referee overstated the miscalibration — the noise floor cuts both ways.
**Ringdown arc closed: an amortized, start-time-marginalized,
calibration-certified no-hair test agreeing with the classical analysis on
the loudest event ever recorded.** v4 shelf: per-parameter/flow recalibration,
simulator realism for the +10% mass pull, tone-count selection, stacking.

### Field-status corrections worth remembering
- "SBI is underused in ringdown" (the original pitch) is stale: ≈4 papers 2023-2026
  (NPE-with-coverage on GW150914; time-domain SBI 2404.11373; CVAE 2506.17618).
  Our sharper angles: start-time amortization (08 = v1), neural tone-count model
  selection, hierarchical stacking.
- GW250114 official results: ≥2 QNMs incl. the 221 overtone, Kerr-consistent to tens
  of percent (arXiv:2509.08099); Hawking area law passed (arXiv:2509.08054).

### Gotchas (hard-won)
- Python 3.14 breaks the GW stack (pystan/numba); use 3.12. The `ringdown` package
  wants exactly 3.11 — deferred.
- GWOSC O4 strain near GW250114 has NaN stretches (e.g. +1500 s) — guard with
  `np.isfinite` before whitening.
- Zero-phase bandpass = time-smearing = tau bias. Whiten-only for fitting.
- The catalog GPS is rounded to 0.1 s; always re-find the peak per detector.

## 2026-06-13 — debt cleared: shared simulator extracted to sbilib.py
09/10 now import simulate/Embed/Kerr grids from scripts/sbilib.py (the
"keep in sync" duplication is gone). Pickle constraint handled: both scripts
keep a module-level `Embed` alias (load-bearing — saved posteriors resolve
__main__.Embed). Verified: compile-check, posterior load + sample, repo gate
ALL GREEN.

## 2026-06-15 — v4 PRE-REGISTRATION: amortized start-time-marginalized TONE-COUNT selection
(Pivot from pbh, which is PARKED COMPLETE.) New question, orthogonal to the v2/v3 no-hair test:
**how many QNM tones are in the data — 220 alone (1-tone) or 220+221 (2-tone)?** This is the live
Isi/Farr-vs-Cotesta GW150914 overtone fight, whose crux is START-TIME dependence — exactly what
this infra marginalizes by construction (our sharper angle, flagged in the v3 field-status note).

Design (neural ratio estimation as a binary classifier):
- Simulator: extend sbilib with `simulate_tonecount(mass, chi, n_tones, amp_frac, rng)` — 1-tone =
  220 only; 2-tone = 220 + Kerr-locked 221 (delta=0; tone-count tests PRESENCE, not deviation) with
  221/220 amplitude ratio `amp_frac`. Start time t0 ~ U(0,6ms) and amplitudes drawn per-sim →
  marginalized. Record the injected OVERTONE matched-filter SNR per sim (physical x-axis).
- Train a classifier (reuse `Embed` CNN + sigmoid head, BCE) on BALANCED 1-tone/2-tone sims, amp_frac
  ~ U(0.1,1.5). Output = amortized P(2-tone | data) = the tone-count Bayes-factor estimate.
- Script `11_tonecount.py`, mirroring 09 (train→referee→apply), heartbeating via rdlib.progress.

Pre-registered gates:
- **T1 (capacity):** held-out balanced AUC > 0.90 (the distinction is learnable).
- **T2 (sensitivity + specificity):** efficiency P(2t)>0.5 rises monotonically with overtone SNR;
  report the 50%-detection overtone SNR; specificity (1 − false-2-tone on real 1-tone) ≥ 0.90 at
  threshold 0.5. Measured in WHITE noise AND real O4-noise injections (the honest version).
- **T3 (calibration):** reliability diagram — binned P(2-tone) vs empirical fraction within tolerance
  (same honesty bar as v2/v3 coverage).
- **T4 (application):** GW250114 → expect 2-tone HIGH confidence (official ≥2 QNM incl. 221 →
  VALIDATES the method); GW150914 → report the honest start-time-marginalized verdict (novel).
Artifacts to produce: models/11_tonecount.pt, results/11_tonecount.json, plots/11_*.png.

## 2026-06-15 — v4 RESULTS (first cut): NEGATIVE — domain gap dominates; gates failed
Built sbilib.simulate_tonecount + 11_tonecount.py (ToneCounter = Embed + sigmoid head, BCE on 80k
balanced sims, 30 epochs, MPS). Pipeline runs end-to-end + dashboard-connected. Gates (all FAILED):
- **T1 AUC 0.716** (gate >0.90). Trains fine (loss 0.69→0.58) but caps ~0.72 — because the amp_frac
  prior includes faint overtones that are genuinely ~indistinguishable from 1-tone (physically real,
  not a bug). T2 sensitivity curve is sane: 50%-detection at **overtone SNR ≈ 6.2**.
- **T2 specificity 0.721** (gate ≥0.90) — over-calls 2-tone on clean 1-tone (the faint-overtone
  ambiguity again).
- **T3 ECE 0.120** (gate <0.10) — mildly miscalibrated.
- **T4 (the killer): GW250114 → P(2-tone) = 0.15** — WRONG (official result detected the 221), and
  **GW150914 → 0.28**. Validation FAILED.
- **The diagnostic contradiction:** real-O4-noise INJECTIONS (T2b) read P(2-tone) ≈ 0.96–1.00 for
  BOTH 1-tone and 2-tone, while the real EVENTS (T4) read 0.15–0.28. Same classifier, same "real
  whitened O4 data", opposite verdicts ⇒ the output is driven by **absolute amplitude scale + noise
  coloring**, not tone count. **The white-noise→real-data domain gap swamps the signal.** (Why did
  v2/v3 NPE transfer but this doesn't? Estimating the dominant mode's M/χ/δ is far more robust to
  noise statistics than presence-detecting a faint, fast-dying overtone — the harder, more fragile task.)
- **Verdict:** the idealized-white-noise classifier is untrustworthy on real data. Honest first-cut
  negative. **Path-forward options (for discussion):** (A) make it scale-invariant — RMS-normalize each
  segment (cheap, addresses absolute-scale sensitivity); (B) domain-matched training — train on REAL O4
  whitened noise + injected overtones, à la pbh (the proper fix); (C) reframe the deliverable as the
  *detectability threshold* (overtone-SNR sensitivity curve), which IS a clean result, rather than a
  per-event call; (D) restrict the amp_frac prior to detectable overtones. Gates NOT added to verify.sh
  (nothing green to lock). Artifacts: models/11_tonecount.pt, results/11_tonecount.json, plots/11_tonecount.png.
