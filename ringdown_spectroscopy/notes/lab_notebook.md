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

### 2026-06-15 — v4 fix A (RMS-normalization): PARTIAL — fixes calibration, NOT the transfer
Added per-detector unit-std normalization (norm_seg), applied consistently in training + both real-data
paths (scale-invariance). Full run (`--tag _norm`): T1 AUC 0.716→**0.728**, specificity 0.721→**0.767**,
**T3 ECE 0.120→0.057 (now PASSES the calibration gate)** — normalization genuinely helped the white-noise
side. BUT real-data transfer STILL broken: T2b injections **0.99/1.00** (still saturated to 2-tone),
T4 GW250114 **0.198** / GW150914 **0.341** (still 1-tone). ⇒ absolute scale was NOT the whole problem;
the **noise COLORING** is — real whitened O4 noise isn't white, so the white-noise-trained net still sees
real data as out-of-distribution. Indicated fix: **B = train on real O4 noise**. Artifact: results/11_tonecount_norm.json.

### 2026-06-15 — v4 fix B (real-O4-noise training): NEGATIVE — transfer STILL broken
Added sbilib.simulate_tonecount(noise=...) + disk-cached real-O4-noise pool (14 whitened chunks before
GW250114, reboot-safe) + make_batch_real; `--real-noise` injects overtones into real-noise windows.
Result (`_realnoise`): AUC 0.717, spec 0.761, ECE 0.073 — same as before; **real-data transfer STILL
broken** (T2b 1.00/1.00, GW250114 0.112). ⇒ noise coloring was NOT the (only) problem either.

### 2026-06-15 — v4 DIAGNOSTIC: the classifier learned a "loud ⇒ 2-tone" SNR shortcut
Probed P(2-tone) vs injected 220 amplitude (fix-B model, cached noise): flat-ish ~0.3–0.5 within the
training amp range (2–12), then **saturates to 1.0 for amp>16 regardless of tone count** (even pure
1-tone). norm_seg removes scale but NOT signal-to-noise ratio, so the net keyed on SNR. ROOT CONFOUND:
in training the 2-tone class adds the overtone energy on top of the same 220 ⇒ 2-tone has ~1.19×
the energy of 1-tone ⇒ the net learned SNR, not tone count.

### 2026-06-15 — v4 fix C' (SNR-matched classes): removes the shortcut, transfer STILL broken
Added `snr_match`: rescale the 2-tone to the pure-220 total energy (class energy ratio 1.19→1.02,
verified). Run (`_matched`, real-noise + SNR-match): AUC **0.72→0.63**, ECE 0.21 — the honest task
(detect spectral splitting at FIXED energy) is genuinely harder — but **transfer STILL broken**
(T2b 0.99/0.98, GW250114 0.20, GW150914 0.067).

### 2026-06-15 — v4 CONCLUSION (first arc): black-box ML tone-count does NOT transfer to real data
Four full attempts (first-cut / norm / real-noise / real-noise+SNR-match) + a diagnostic: the
sim→real gap is **multi-faceted** — (1) absolute scale [fixed by norm], (2) noise coloring [addressed
by real-noise training], (3) an SNR shortcut [removed by SNR-matching], and (4) **injection convention**
(training injects in the whitened domain; both real-data paths inject/observe raw THEN whiten, which
reshapes the ringdown — untested, needs per-example whitening = heavier redesign). Fixing 1–3
individually each helped a symptom but none restored a trustworthy real-event verdict: **GW250114 reads
1-tone (P≈0.1–0.2) in EVERY variant, contradicting the official 221 detection.** Why did v2/v3 NPE
transfer but this doesn't? Parameter-estimating the dominant mode is robust to these gaps; presence-
detecting a faint, fast-dying overtone is far more fragile.
**Salvageable / honest deliverable (option C):** the WHITE-NOISE sensitivity curve — overtone SNR≈6–7
for 50% detection — is a clean, simulation-only "detectability threshold" that does NOT depend on the
broken transfer. **Come-back-later lever:** injection-convention-matched training (raw-domain inject +
whiten per example), or abandon the black-box classifier for explicit Bayesian model selection with a
real noise model (what the field actually does). Artifacts: results/11_tonecount_{,_norm,_realnoise,_matched}.json.

### 2026-06-15 — v4 fix D (injection-convention-matched): TRANSFER PATHOLOGY GONE, but model overfits
Confirmed FIRST (cheap shape diagnostic): whitening RESHAPES the ringdown — raw vs gwpy-whitened shape
overlap only **0.48** ⇒ training on the raw shape (all prior variants) fed the net the wrong template.
Then validated a fast FD whitening (irfft(rfft(raw)/asd), full-band) reproduces gwpy's whiten(4,2) to
**shape-overlap 1.000** (band-limiting drops it to 0.43 — gwpy is full-band). Built it: sbilib.whiten_shape
+ simulate_tonecount_conv (whiten raw ringdowns through each chunk's real ASD, crop from the whitened
PEAK like T4, SNR-matched) + a per-chunk ASD noise pool (cached) + `--raw-conv`. No application changes
(T2b/T4 already full-band → now convention-matched to training).
**Result (`_conv`): the transfer pathology is GONE** — real-noise injections no longer saturate and are
correctly ordered (1t **0.60** < 2t **0.76**; every prior variant pinned both ~1.0), and **GW250114 reads
P(2-tone) = 0.687** (2-tone, consistent with the official 221 detection!); GW150914 0.458 (ambiguous —
fitting for the controversial event). **BUT the model OVERFITS:** train loss 0.35 vs held-out AUC **0.53**
(chance), ECE 0.26, injection spread ±0.34 ⇒ it removed the gross pathology but is NOT yet a trustworthy
discriminator. Cause: the small 14-chunk noise pool reused across 80k examples (the net memorizes noise).
**Next (to discuss):** more noise diversity (many more chunks) + regularization/early-stop, then re-check
whether GW250114→2-tone holds with a model that actually generalizes. The convention fix is the real
unlock; overfitting is now the bottleneck, not the transfer. Artifacts: results/11_tonecount_conv.json,
models/11_tonecount_conv.pt, data/o4_noise_pool_asd.npz.

### 2026-06-15 — v4 CLOSED: overfitting fixed → model is HONEST but WEAK; GW250114=2-tone was a mirage
Fixed the overfitting properly: 60-chunk noise pool (4× diversity), FRESH data regenerated each epoch
(`--fresh`, kills memorization of a fixed set), early-stopping on held-out AUC (keep best epoch, not the
over-trained final). Run (`_convbig`, 60 chunks + fresh + conv-match + SNR-match):
- **Overfitting GONE:** train loss flat ~0.67 (not 0.35), train≈held-out AUC (no gap), **ECE 0.006**
  (near-perfectly calibrated — best of the whole arc).
- **But the honest model is WEAK:** held-out **AUC 0.607**, specificity 0.63 — it only weakly separates
  1-tone from 2-tone. T2b real-noise injections now read **1t 0.79 ≈ 2t 0.77** (it CANNOT tell them apart
  in real noise). **GW250114 → P(2-tone) = 0.315 (1-tone-leaning), GW150914 → 0.469** — i.e. **fix D's
  exciting GW250114=0.69 was an OVERFITTING ARTIFACT; the trustworthy model reads it as ambiguous/1-tone.**
- **v4 VERDICT — honest NEGATIVE (now with a trustworthy model).** A black-box ML tone-count classifier,
  done correctly (real noise, convention-matched, SNR-matched, fresh data, early-stop, calibrated), only
  WEAKLY distinguishes 1 vs 2 tones (AUC ~0.61) and cannot make confident per-event calls at this
  data/SNR scale. The methodology chain itself is the contribution: scale→coloring→SNR-shortcut→injection-
  convention→overfitting were each diagnosed and fixed, and only then is the result believable.
  **Salvageable deliverable:** the calibrated detectability threshold — overtone SNR ≈ 5 for 50% detection.
  **Come-back-later levers:** far more data + a bigger/coherent model; a stacked multi-event statistic; or
  abandon the black-box for explicit Bayesian model selection with a real noise model (what the field uses).
  No verify.sh gate added (negative — nothing green to lock). Artifacts: results/11_tonecount_convbig.json,
  plots/11_tonecount_convbig.png. **v4 tone-count PARKED.**

## 2026-06-20 — v5: multi-event no-hair δ STACKING (roadmap P1) ✓✓
Combine the no-hair deviation δ across events under a common-δ assumption (Kerr deviation is universal):
precision-weighted Gaussian combination of the recalibrated per-event posteriors, σ_stack = (Σ1/σ_i²)^(-1/2).
Reuses the v2/v3 amortized NPE (09_posterior_150k.pt) + v3 temperature (T=1.05). `12_stacking.py`.
- **S2 (the headline) ✓✓ σ(δ) tightens as √N — measured matches ideal almost exactly:** N=1→0.275,
  2→0.191, 3→0.155, 5→0.122, **8→0.095** (ideal σ_single/√N = 0.097). σ_single≈0.274.
- **S1 ✓ unbiased** (δ=0 injections → stacked median −0.005 at N=8). **S3 ✓ coverage** in [0.80,1.0] for all N
  (in fact ~1.0 — slightly CONSERVATIVE/over-covering, so the σ values are if anything pessimistic; the
  √N scaling is the robust result).
- **S4 ✓ real events:** GW250114 δ=−0.126±0.225, GW150914 δ=−0.034±0.281 → **STACKED δ = −0.090 ± 0.176
  [−0.380, +0.199] 90%** — Kerr-consistent, **~1.3× tighter than the best single event** (0.225→0.176).
- ⇒ the amortized no-hair network combines across events exactly as theory predicts; more events directly
  sharpen the GR test (8 events would reach σ(δ)≈0.10 vs 0.24 single). Caveat: real-event posteriors inherit
  the NPE's GW250114-loudness domain (GW150914 fainter → broader, lower weight — honest). Gate added to
  verify.sh (S2 √N tightening). Artifacts: results/12_stacking.json, plots/12_stacking.png.

## 2026-06-20 — v5 STRESS-TEST (13_more_events.py): the real-event δ-stack is NOT achievable — correction
Per the robustness north star, before stacking MORE events I cross-checked the NPE on 8 candidate
events (recovered M roughly consistent with published det-frame remnants, allowing the known +10–15%
mass pull). **The decisive finding (info content = δ posterior width vs the δ PRIOR width 0.289):**
- **GW250114 INFORMATIVE** (δ_σ/prior 0.82; χ=0.77 away from prior mean) — the only one.
- **ALL 7 fainter events (GW150914, GW170814, GW170104, GW170809, GW170818, GW200129, GW190828) return
  ≈ the PRIOR** (δ_σ/prior 0.93–0.99; χ medians pulled to ~0.5, the prior mean). They "look Kerr-
  consistent" only because the prior is centred at 0 — it is NOT a measurement.
- **⇒ CORRECTION to the v5 real-event result.** The "GW250114+GW150914 → 1.3× tighter" (12) was a
  **Gaussian-approximation artifact**: GW150914's ≈-flat (prior-dominated) posterior, fit to a Gaussian
  σ=0.27, acted as a spurious second "measurement" in the precision-weighted stack. The GENUINE combined
  real-event constraint ≈ GW250114 alone (σ_δ ≈ 0.24). There is effectively ONE informative real event.
- **What still stands:** the stacking METHOD validation (S1/S2/S3 on informative INJECTIONS — √N
  tightening, unbiased, calibrated) is correct and valuable: it proves that *if* we had N informative
  events, σ(δ) would shrink as √N. We just don't, with current public data + this NPE.
- **Root cause = the per-event SNR information wall** (same flavour as v4 tone-count): only GW250114
  (SNR~80) is loud enough for the NPE to extract δ; GW150914 (~24) and O2 events (~10–20) cannot.
  To actually stack you need either more SNR-80-class events (rare) or an NPE that extracts δ at lower
  SNR (likely information-limited). **Real multi-event δ sharpening: parked, honestly.** verify.sh gate
  reduced to the √N METHOD assertion only (the "stack < singles" assertion removed — it locked in the
  artifact). Artifacts: results/13_more_events.json.

## 2026-06-20 — v6: the δ-MEASURABILITY THRESHOLD (14_delta_threshold.py) — the SNR wall, MAPPED ✓
The natural follow-up to 13's negative: *13 says only GW250114 measures δ; 14 quantifies WHY.* Inject
Kerr (δ=0) ringdowns across a sweep of loudness (220 amplitude a220 ∈ [2,12], the NPE's trained range),
run the v2/v3 NPE, and measure σ(δ) vs the **injected whitened ringdown SNR** (= √Σsig² in the 40 ms
window — the MF SNR for a known template in white noise). n=60 per loudness point.
- **Result (seed-robust over seeds 0/1/2):** σ(δ) shrinks **monotonically** with loudness but only
  reaches **σ/prior ≈ 0.87 (≈13% tighter than the prior) even at the top of the trained range**
  (ringdown SNR ≈ 40). Onset of any informativeness (σ/prior < 0.90) is at **ringdown SNR ≈ 37–38**.
  Below ringdown SNR ~30 the posterior **is** the prior (ratio ≥ 0.93).
- **The anchor:** GW250114 (real, from 13) reaches σ/prior ≈ 0.83 — i.e. it sits **at/just past the
  informative edge** of this curve. That is the whole story in one number: the single loudest public
  ringdown is *just barely* informative on δ, and nothing fainter is. The √N stacking method (12) is
  correct but starves for inputs because **essentially every event lands left of ringdown SNR ~37**.
- **Caveat (honest):** the NPE is trained on a220 ∈ [2,12]; I deliberately did NOT sweep louder
  (extrapolation would be untrustworthy), so the curve maps the wall *up to* GW250114-class loudness,
  not beyond. The "ringdown SNR" here is the post-peak in-band SNR (≈half the full-event SNR), which is
  why GW250114's full SNR~80 maps to a ringdown SNR in the ~40s. Artifacts: results/14_delta_threshold.json,
  plots/14_delta_threshold.png. Gated in verify.sh (monotone shrink + GW250114 at the edge).

## 2026-06-25 — R2 (PLAN.md): explicit Bayesian tone-count — ATTEMPTED, PARKED (simplified version not a fair test)
The v4 ML tone-count was a parked negative; the guardrail sanctions the non-ML route, explicit Bayesian model
selection. Built `14_bayes_tonecount.py`: the ringdown amplitudes are linear given (M,χ,t0), so the evidence
marginalizes them analytically (linear-Gaussian), and (M,χ,t0) — start time marginalized — are integrated on a
grid; logB_21 calibrated on 1-tone vs 2-tone injections.
- **It doesn't work, and the oracle says why.** The diagnostic (Bayes factor at the TRUE M,χ,t0, no grid) shows
  **NO separation between 1-tone and 2-tone at any σ_a ∈ [0.3, 8]** — in fact 1-tone injections often score
  *higher* logB than 2-tone. Cause: 220 (239 Hz, τ4.8 ms) and 221 (235 Hz, τ1.6 ms) are **near-degenerate over
  the 0.04 s segment** (4 Hz apart; only the damping differs), so the 2-tone model just flexibly fits the loud
  220, overfitting noise regardless of a real overtone. Same wall that beat the free LSQ (`06`) and the ML (v4).
- **But this is NOT the information limit — it's MY implementation.** The published GW250114 analysis
  (arXiv:2509.08099) reports a clean two-tone Kerr test, i.e. a proper Bayesian analysis DOES detect the overtone.
  My simplified machinery — time-domain, white-noise likelihood, independent per-detector amplitudes, flat σ_a,
  0.04 s segment, injections possibly fainter than the real event — is too crude to be a fair test. **So I will
  NOT claim "Bayesian model selection also fails" (that would be a false negative).**
- **Verdict: R2 PARKED — needs the proper frequency-domain coherent pipeline with physical priors (the `ringdown`
  package, Python 3.11, deferred env).** Script kept as the reproducible diagnostic, NOT gated. North-star call:
  diagnosed why the cheap version isn't fair and stopped, rather than shipping a misleading negative.

## 2026-06-25 — R3 (PLAN.md): IMR-waveform referee — the no-hair NPE has a START-TIME WAVEFORM SYSTEMATIC
The strongest referee: does the no-hair NPE (09, trained on analytic 220+(1+δ)·221 tones) stay unbiased on
REALISTIC full-merger ringdowns? Generated 3 NR-calibrated IMRPhenomXAS ringdowns via pbh's pycbc (42+42 s0.4,
40+40 s0.3, 45+38 s0.5 → f_ring 215–226 Hz; data/imr_ringdowns.npz), injected the post-peak shape (unit-peak,
scaled to GW250114 loudness) into the NPE's whitened/unit-noise convention, sampled δ, recalibrated (v3 T).
`15_imr_referee.py`, n=80/case.
- **Result:** the NPE is **unbiased on its own analytic family** (control δ=+0.02±0.13) but recovers
  **δ≈−0.33 to −0.35 on all 3 realistic IMR ringdowns injected from the peak** — a ≈1σ(δ) systematic. True δ=0
  (IMR is GR), so this is a model-incompleteness bias.
- **MECHANISM CONFIRMED (start-time sweep, matched loudness):** δ_hat = −0.327 (0 ms) → −0.290 (2) → −0.211 (4)
  → **−0.014 (6 ms)**. The bias decays monotonically to ~0 as the injection starts later ⇒ it is the **early-time
  merger-transition + higher-overtone content** (the first few ms after the peak) that the two-tone model cannot
  fit; by 6 ms the ringdown is clean 220+221 and the NPE is unbiased. NOT a convention artifact (that would
  persist at all offsets).
- **Significance:** R3 independently **quantifies the start-time systematic** — the exact methodological issue the
  whole project is about (Isi/Farr vs Cotesta). It's also a **caveat on the 09 GW250114 result** (δ=−0.16): that
  analysis crops from the peak, so it inherits some of this negative systematic; the "Kerr-consistent" verdict
  still holds (−0.16 sits well inside the broad σ(δ)≈0.36 posterior, and the systematic is comparable to one σ),
  but the waveform systematic belongs in the uncertainty budget. **Mitigation = start the fit later (≳6 ms
  post-peak) where the bias vanishes — at the cost of ringdown SNR (the v6 wall).** Gated. Caveat: IMRPhenomXAS is
  NR-calibrated (true SXS NR + coherent multi-detector deferred). Artifacts: results/15_imr_referee.json.

## 2026-06-25 — R3 capstone: the GW250114 no-hair δ vs ringdown START TIME (real data)
Applied the R3 lesson to the real event (`16_gw250114_starttime.py`): crop GW250114's whitened ringdown at
peak+offset for offset ∈ {0,2,4,6,8,12} ms and run the NPE at each.
- **δ vs start:** −0.161 (0ms) / +0.068 / −0.092 / −0.154 / −0.083 / **+0.014 (12ms)**; **every offset
  Kerr-consistent** (0 inside the 90% CI, all CIs ≈ ±0.45).
- **Two clean takeaways:** (1) the peak-cropped δ=−0.16 **reproduces the 09 headline** (validation); (2) the
  late-start (systematic-mitigated) δ=+0.01 — drift +0.175 toward Kerr, the **direction R3's injection sweep
  predicts**. So GW250114's apparent small negative δ is consistent with being partly the early-time waveform
  systematic, not new physics; the mitigated value is cleanly Kerr.
- **Honest caveat — don't overclaim the real-data drift.** Unlike the injection sweep (n=80, clean −0.33→0
  decay), the real event has ONE noise realization per offset and broad σ(δ)≈0.36 posteriors, so the per-offset
  values scatter (non-monotonic: +6ms gives −0.15) and all overlap zero. The **injection sweep is the decisive
  evidence** for the systematic; the real-event sweep is **consistent with it but noise-limited**. Net: the
  no-hair test passes at every start time, and understanding the systematic only *strengthens* the
  Kerr-consistency (apparent −0.16 → mitigated ≈0). Gated. Artifacts: results/16_gw250114_starttime.{json,png}.

## 2026-06-25 — SISTER-PROJECT requests (TheBridge SISTER_REQUESTS.md): A5 ✅ + A1 (running)
TheBridge (read-only consumer of the source repos) relayed two deepstrain asks; produced the artifacts here.
- **A5 (precise multi-event no-hair) ✅** — `18_export_tonefits.py`: ran the §06-style classical fit (joint
  H1+L1, shared f/τ, per-det amp/φ via rdlib.fit_modes) on all 8 §13 events; exported f_220, τ_220, f_221,
  τ_221, amplitudes + the 220 inversion (M,χ). **5/8 have a reliable (non-railed) 220** (GW250114, GW150914,
  GW170814, GW170104, GW190828); 3 rail at the f/τ bounds (GW200129, GW170809, GW170818 — the faint events
  §13 flagged) → `tone220_railed:true`. 221 flagged `tone221_reliable:false` (§06: free 2-tone can't split the
  ~6-Hz tones at this SNR). The bridge inverts the robust 220 → predicts 221 → per-event δ → stacks. Gated.
  Artifact: results/18_tonefits.json.
- **A1 (the §9 "most original": does amortization gap predict sim→real transfer?) ✅ DONE.**
  `19_amortization_transfer.py`: 5 no-hair NPE variants spanning N_TRAIN {5k,15k,40k,90k,150k} (clean common
  protocol, §09 architecture). Per variant: amortization_gap = mean|sim 90%-coverage − 0.90|; transfer =
  mean(real-O4-noise coverage) − mean(sim coverage). **Results:** amort_gap shrinks **monotonically** with
  training (0.095→0.072→0.058→0.035→0.028 — calibration improves as expected); transfer is **negative
  throughout** (−0.016…−0.142, real noise degrades inference). **gap vs transfer correlation = +0.04 ≈ 0 — the
  amortization gap does NOT predict sim→real transfer in this sample** (the bridge's §9 prediction is not borne
  out here; they're roughly decoupled).
  **Honest caveats for the bridge:** (1) only 5 points; (2) transfer is noise-limited — 60 real injections/variant
  → ±~0.06 per-param, and the 40k transfer (−0.142, the outlier driving most of the scatter) sits on a low
  real_cov [0.55,0.65,…] that may be a noisy realization; (3) single lever (N_TRAIN) — capacity/flow levers
  untried; (4) amortization_gap is the coverage-deviation proxy, not the C2ST. So read corr≈0 as "no detectable
  relationship at this resolution," not a strong null — more variants / more real injections / the C2ST proxy
  would sharpen it. Checkpoints 19_npe_n{N}.pt + results/19_amortization_transfer.json saved for the read-only
  correlation. Gated.
## R1 (2026-06-26): per-parameter temperature recalibration — gate-passing, but global T wins
v3 (`10`) fit one global temperature T=1.05. R1 (`17_recalibrate_perparam.py`) fits a separate T per
parameter. One T-sweep suffices: `widen()` rescales each column about its own median, so parameter j's
coverage depends only on T_j. Result (1000 calibration sims, 600 fit / 400 held-out):
- per-param T = **T_M 1.10, T_chi 1.05, T_delta 1.05**; held-out coverage **M 0.94, χ 0.92, δ 0.90** — each
  in [0.85,0.95] (the PLAN "done" criterion is met).
- **But per-param does NOT beat global:** mean|coverage−0.90| = **0.020 (per-param) vs 0.011 (global T=1.05)**.
  The per-param fit overfits the n=600 calibration-set coverage noise (≈2% per estimate) — T_M=1.10 pushes M to
  0.94, further from 0.90 than global's 0.91. Honest conclusion: **the single global temperature was already the
  right, sufficient choice; per-parameter tuning adds nothing** (exactly the low value the PLAN anticipated).
- GW250114 δ under its own T_delta=1.05: **−0.16 [−0.46,+0.33], Kerr-consistent** — unchanged from v3.
Gated (each-in-band AND global mad ≤ per-param mad). Artifact: results/17_recalibrate_perparam.json.

## 2026-07-02 — R2 v2 PRE-REGISTRATION: the proper tone-count test via the `ringdown` package
The Py3.11 wall fell (uv venv `.venv311`; working pins jax 0.4.35 / numpyro 0.15.3 / arviz 0.19.0 /
matplotlib 3.9.4, frozen in `.venv311-pins.txt` — ringdown 1.0.0's `jax~=0.4` pin is too loose, newest jax
breaks its `jaxlib.xla_extension` import). Targets verified before fitting (standing directive): GW150914 =
docs example (t0=1126259462.4083147, ra=1.95, dec=−1.27, psi=0.82); GW250114 = LVK max-L per arXiv:2601.05734
(t0=1420878141.2362, ra=2.35, dec=0.22, psi=1.37). Settings: duration 0.05 s, 4096→2048 Hz, 20 Hz HP, ACF from
128 s off-source, m∈[40,200], a_scale 1e-20. Pre-registered gates:
- **(a) validation:** GW150914 220+221 (M, χ) median lands in the known ballpark (M ~55–90, χ ~0.4–0.9;
  Isi/Farr 2019 got ~68, ~0.63). If this fails the stack is wrong — stop, don't interpret.
- **(b) the R2 question:** GW250114 220+221 — is A221 bounded away from zero (field statistic)? Published
  answer is YES (arXiv:2509.08099). Any outcome recorded honestly.
- **(c) NPE referee:** GW250114 (M, χ) vs our 09/10 NPE — consistency check of the whole arc.

## 2026-07-02 — R2 v2 RESULT: the proper pipeline DETECTS the GW250114 overtone; NPE arc cross-validated ✓
All three pre-registered gates pass (fits: NUTS x64, all R̂ ≤ 1.004, ESS ≥ 950; 21_ringdown_crosscheck.py,
results/21_ringdown_crosscheck.json + posterior npz):
- **(a) validation ✓** GW150914 220+221 at the docs target: M 77.5 [61.5, 94.5], χ 0.76 [0.38, 0.92] — in the
  pre-registered band (Isi/Farr-era analyses ~68/0.63; the docs example itself notes it doesn't exactly reproduce
  Isi 2019 — different polarization model/priors/conditioning). Stack validated.
- **(b) THE R2 ANSWER ✓ — GW250114 overtone: A221 = 3.6e-21 with q5 = 2.4e-21, P(A221 < 10% of median) = 0.000,
  A221/A220 = 1.02 at peak.** The field-standard statistic (amplitude bounded away from zero) FIRES, matching
  arXiv:2509.08099 ("bounded away from zero", ratio ~1 at peak). Adding the 221 also pulls M 84.6→74.8 (the
  early-time content the 220-only model absorbs into mass — the same start-time systematic R3 measured).
  **⇒ R2 CLOSED: our 14_bayes_tonecount null was an implementation limit, now POSITIVELY demonstrated —
  the proper FD coherent pipeline separates the tones on the same event where our simplified time-domain/
  white-noise machinery could not. Refusing to publish that false negative was correct.**
  Bonus context: GW150914's overtone comes out MARGINAL here (P = 0.049) — consistent with the contested
  literature (Isi/Farr claim vs Cotesta skepticism), a nice sanity check that the statistic isn't trigger-happy.
- **(c) NPE referee ✓ — first independent field-standard cross-validation of the whole arc:** package n=2
  M 74.8 [70.4, 79.0] / χ 0.729 [0.64, 0.80] vs our 09 NPE on-source M 76.0 [68.4, 85.2] / χ 0.762 [0.62, 0.86].
  Medians agree to 1.2 M☉ / 0.033; the package's tighter coherent-likelihood CI nests inside the NPE's (expected:
  FD coherent + antenna patterns vs our sky-independent per-detector amplitudes). The amortized NPE posterior is
  REAL, not an artifact of our simulator.
- Environment (the wall that kept R2 parked): `.venv311` via uv; ringdown 1.0.0's pins are too loose for 2026 —
  working set frozen in `.venv311-pins.txt` (jax 0.4.35 / numpyro 0.15.3 / arviz 0.19.0 / matplotlib 3.9.4 /
  scipy 1.14.1; newest jax breaks `jaxlib.xla_extension`, newest matplotlib breaks arviz 0.19, newest scipy's
  STFT breaks ringdown's pandas-Series data in Welch). Targets verified before fitting: GW150914 = docs example;
  GW250114 = LVK max-L via arXiv:2601.05734 (t0=1420878141.2362, ra=2.35, dec=0.22, psi=1.37).
- Honest caveats: package M sits ~1 M☉ below NPE median but 6-7 above the 05 start-time-plateau value (~68) —
  all peak-start fits carry the R3-measured early-time systematic; duration fixed at 0.05 s (docs value), not
  swept; GW150914 marginal-overtone number is duration/target-sensitive per the literature — we record, not claim.

## 2026-07-03 — B1 (PLAN.md): field-standard start-time sweep — package refereess R3's systematic
`22_starttime_sweep.py` (ringdown 1.0.0, .venv311): GW250114 220+221 at 9 start offsets 0–16 t_Mf
(t_Mf = M_f·T_sun = 68.1·4.925e-6 = 0.335 ms), duration 0.05 s, NUTS x64 (all rhat < 1.01).
| t0 [t_Mf] | ms | M | χ | A221/A220 | P(A221≈0) |
|---|---|---|---|---|---|
| 0 | 0.00 | 74.7 | 0.730 | 1.01 | 0.000 |
| 4 | 1.34 | 72.2 | 0.697 | 0.96 | 0.001 |
| 8 | 2.68 | 71.3 | 0.744 | 1.61 | 0.000 |
| 12 | 4.03 | 65.4 | 0.643 | 0.81 | 0.013 |
| 16 | 5.37 | 65.9 | 0.656 | 0.52 | 0.059 |
- **Overtone: significant from the peak (P=0.000), damps away by ~5.4 ms (P=0.059)** — matches τ221≈1.4 ms
  (by 16 t_Mf the overtone amplitude is ~e^-3.8 ≈ 2%). A real, fast-damping 221, as the published analysis found.
- **Peak-start mass biased HIGH: 74.7 vs the true detector-frame remnant 68.1 M☉ (+10%), drifting −8.8 M☉ to
  ~66 by late start; χ −0.074.** This is the R3 early-time systematic (peak-start fits carry merger/overtone
  content the 220+221 model omits) — R3 measured it with our NPE (δ≈−0.33 at peak → 0 by 6 ms); B1 **independently
  reproduces the same qualitative systematic with the field-standard coherent package** (a different tool, a
  different parameter). The late-start M (~66) sits near the 05 start-time plateau (~68), consistent.
- Caveat: as t0 increases the signal damps → CIs widen and the A221/A220 point estimate gets noisy (the 6–8 t_Mf
  bump 1.4–1.6 is that instability, not a physical rise); the clean, monotone signals are the overtone-significance
  decay and the mass drift. Gated. Artifact: results/22_starttime_sweep.json.

## 2026-07-03 — B2 PARKED-honest + B3 (NPE loop closed) → B complete
- **B2 (nonlinear quadratic QNM) PARKED — tool inadequate, R2 discipline.** The claim (arXiv:2601.05734) is 6
  quadratic modes in the (4,4) multiplet from 220×22n coupling (BF 74 at 5 M_f, 3σ, reconstructed 6–10 M_f). A
  fair referee needs (a) simultaneous (2,2)+(4,4) multipole strain modeling and (b) frequency-locking the
  quadratic mode to 2·f220 = 497 Hz (6.9% below the linear Kerr 440, 534 Hz). The vanilla ringdown package does
  NEITHER — it fits linear (2,2) Kerr QNMs; its "quadratic"/"nonlinear" source strings are peak-interpolation +
  the sampler's nuisance parameters, unrelated. Wang & Ma used custom PyCBC-Inference templates + dynesty 30k
  live points. A vanilla (2,2) fit would return ~null on a real-but-subtle (4,4) signal ⇒ a false negative, which
  the guardrail forbids. Parked with the requirement stated; not attempted, not claimed either way.
- **B3 (close the NPE loop) DONE.** `23_npe_package_loop.py` — pure synthesis of committed artifacts (09 NPE, 21
  package-fixed, 22 package-sweep), no new fits:
  (1) **Agreement:** NPE M 76.0 [68.4,85.2] / χ 0.762 vs package-peak M 74.8 [70.4,79.0] / χ 0.729 — median gap
      1.2 M☉ / 0.033, and the package 90% CI **nests inside** the NPE's ⇒ the amortized network does real,
      field-consistent inference (not a simulator artifact).
  (2) **Location:** the NPE median (76.0) sits at ~0 t_Mf (at/above the peak) in B1's start-time family ⇒ the NPE
      weights the earliest, highest-SNR configuration rather than averaging over t0, so it **inherits the R3/B1
      early-time systematic** — NPE mass bias +7.9 M☉ vs the true 68.1, comparable to the package's peak bias +6.6.
  Take-away: marginalizing the start time does NOT make the NPE bias-free; it effectively infers from the peak,
  carrying the same +10% systematic every peak-cropped analysis shares. Closes the NPE arc's real-data story:
  amortized + calibrated (v3) + field-cross-validated (R2/21) + now located in the systematic (B3). Gated.
  Artifact: results/23_npe_package_loop.json.

## 2026-07-24 — G8 PRE-REGISTRATION: the Fisher (Cramér-Rao) floor on δ (TheBridge falsification)
**Postulate G8:** "the 221 δ deficit is fundamental at current SNR — no pipeline beats the Cramér-Rao floor."
- **KILLED** iff NPE σ(δ) is significantly BELOW the Fisher floor at GW250114's ringdown SNR by a margin the
  prior cannot explain.
- **SURVIVES** iff NPE σ(δ) ≈ floor (efficient estimator at the limit).
**Pre-registered readings (both are results):**
- σ(δ)_NPE ≈ floor → G8 SURVIVES; headline UPGRADES "2.6× tighter than classical" → "efficient estimator at
  the information floor" (a stronger claim).
- σ(δ)_NPE < floor → the prior does that work (a Bayesian posterior legitimately can); G8 still stands, a
  pre-registered honesty footnote on the "2.6×" framing.
**Method (`24_fisher_floor.py`, .venv):** the no-hair model is 220 + (1+δ)-shifted 221 damped sinusoids +
WHITE unit-variance noise (sbilib.simulate) → the Fisher inner product is the plain Euclidean dot product and
SNR = √(Σh²). Fisher matrix over [M, χ, δ, t0] (shared) + per-detector [A220, φ220, A221, φ221] (2 detectors,
12 params) via central-difference Jacobians on the CONTINUOUS Kerr f_tau (smooth χ-derivative); σ_Fisher(δ) =
√[(F⁻¹)_δδ] (marginalized over all nuisances, matching the NPE which marginalizes them). NPE σ(δ) = posterior
std over N fixed-loudness GW250114 injections (δ=0), same SNR. Fiducial M/χ = GW250114 remnant; loudness = the
PEAK_AMP_RANGE mean, cross-checked against the published ringdown SNR. Report σ(δ)_NPE, σ_Fisher(δ), ratio,
verdict. Step-size convergence checked before trusting the matrix.

## 2026-07-24 — G8 RESULT: the NPE does NOT beat the Fisher floor on δ — G8 STANDS (prior-regularized)
`24_fisher_floor.py`, at GW250114's fiducial ringdown SNR **24.9** (sbilib GW250114-calibrated loudness;
~30% of the total-80 network SNR in quadrature — plausible):
| quantity | value |
|---|---|
| σ_Fisher(δ) data-only Cramér-Rao | **0.321** |
| σ_prior (δ ~ U[-0.5,0.5]) | 0.289 |
| σ_post_min (data+prior Bayesian floor) | 0.215 |
| σ_NPE (posterior width) | **0.263** |
| NPE point-estimate scatter | 0.126 |
| prior-shrinkage test: inject δ=0.4 → NPE median | **+0.055 (86% pulled to prior center)** |

**Verdict: G8 STANDS.** The pre-registered kill condition (NPE σ(δ) below the Fisher floor by a margin the
PRIOR cannot explain) is NOT met:
- **The data barely constrain δ at this SNR:** σ_Fisher(δ)=0.32 ≈ σ_prior — independently consistent with v6's
  "δ informative only at ringdown SNR ≳ 37" (GW250114 sits below that). The overtone simply carries little δ
  information at SNR ~25.
- **The NPE posterior width (0.263) is a proper Bayesian combination** sitting between the data-only floor (0.321)
  and the data+prior floor (0.215) — it does NOT beat the information limit; the sub-data-floor part is the prior.
- **The apparent point-estimate precision (0.126) is PRIOR REGULARIZATION, proven:** an injected δ=0.4 is pulled
  86% back to the prior center (median +0.055). At δ=0 (= the injected truth = the prior center) that same
  shrinkage makes estimates cluster near 0, looking precise — but it's the prior pulling toward a value that
  happens to be correct, not the data resolving δ.
- **Trust checks:** σ_Fisher step-convergence 0.8% across a 4× step range (the δ marginal is stable despite the
  nuisance-subspace degeneracy from the t0/phase near-degeneracy); the NPE's own calibrated posterior width
  independently corroborates σ_Fisher~0.3.

**Honesty footnote on the headline (pre-registered):** "σ(δ) 2.6× tighter than the classical fit" is TRUE and
remains our result — but it reflects efficient Bayesian estimation + prior regularization at the prior center,
NOT beating the Cramér-Rao information floor. No pipeline beats the floor on δ at current SNR; that deficit is
fundamental, exactly as G8 asserts. Gated. Artifact: results/24_fisher_floor.json.
