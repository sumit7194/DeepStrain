# Lab notebook — echoes

*Decisions, simplifications, and dead ends. Newest at the bottom.*

## 2026-06-10 — v1 pipeline design

**Goal of v1:** a complete, honest, end-to-end pipeline on ONE event (GW150914) —
fetch → inject → search → background → sensitivity — every piece validated, every
simplification recorded. Scaling to the full catalog and adding the ML scorer come
after this skeleton is trustworthy.

### Decisions

- **Shape-agnostic statistic (comb on envelope ACF).** We deliberately do NOT
  matched-filter an assumed echo waveform: nobody knows the waveform (it depends on
  unknown reflectivity), and template mismatch was a core weakness in the published
  debates. Our statistic only assumes *energy repeating at a fixed spacing*: take the
  whitened segment's energy envelope (|Hilbert|, 10 ms smoothing), autocorrelate,
  sum the ACF at lags {Δt, 2Δt, 3Δt}. Noise → teeth average ≈ 0; repeating energy →
  all teeth positive.
- **Network coherence for free:** a real echo has the SAME spacing in H1 and L1, so
  the network statistic sums per-detector comb scores at the same Δt before taking
  any maximum. Noise peaks in the two detectors don't line up in Δt; signals do.
- **Two pre-registered statistics:** (A) max over the whole Δt grid [0.05, 0.5] s —
  agnostic, pays a trials factor; (B) score AT the predicted spacing (0.2925 s for
  GW150914, from Abedi et al. Table I) — the sharp theory-driven test. Both fixed
  before looking at on-source data.
- **Background = identical statistic on event-free 3 s segment pairs** sliced from
  the same 512 s whitened blocks (8 s edge guard for whitening artifacts, ±8 s
  exclusion around the merger). p-value = rank of the on-source score among
  background scores, with the standard +1 correction.
- **Injection model:** damped sine-Gaussian pulse train — first pulse 0.1 s after
  segment start, spacing Δt, ringdown-like f0=250 Hz / tau=20 ms, per-bounce
  reflectivity damping gamma=0.7, sign flip per bounce, 6 echoes. Phenomenological,
  standard in the literature for sensitivity studies.

### Simplifications (v1) — to revisit

1. **Injection in the whitened domain.** Whitening is linear, so adding a whitened
   template to whitened noise ≈ adding raw strain and re-whitening; amplitudes are in
   whitened-noise σ units (≈ per-pulse SNR scale). A production pipeline injects into
   raw strain and re-runs the full conditioning. Fine for v1 relative sensitivity;
   redo before quoting physical strain amplitudes.
2. **Same injected waveform in both detectors.** Real signals differ by antenna
   pattern + arrival-time offset (≤10 ms H1–L1). The envelope statistic is insensitive
   to the time offset (≪ Δt) but the amplitude ratio matters for absolute claims.
3. **Background segments share the 512 s block with the event.** Independent but not
   far-separated; a production background uses different days. N≈160 pairs limits the
   smallest measurable p to ~1/160 ≈ 0.006.
4. **Stationarity assumed within the block** (one whitening PSD). O1 noise is not
   perfectly stationary; per-segment PSD or line tracking would harden this.

### Validation so far

- Comb statistic on synthetic white noise + injected train: recovers Δt = 0.290 s
  exactly at amp ≥ 3, degrades gracefully below — matches expectations.
- 01_fetch: GW150914 chirp clearly visible in whitened H1 & L1 (the toolchain
  reproduces the textbook waveform).

### Honest expectations

On-source GW150914 should come out CONSISTENT WITH NOISE (p ≫ 0.01) — Westerweck's
re-analysis is the prior. The deliverable of v1 is the *sensitivity curve*, i.e. the
statement "echo trains above amplitude X would have been seen; none were."

## 2026-06-10 — v1 RESULTS

**Sensitivity (GW150914 config, 60 injections/amplitude into real H1+L1 noise,
detection = p < 0.01 vs background):**

| first-pulse amplitude [whitened σ] | 0.5 | 1.0 | 1.5 | 2.0 | ≥2.5 |
|---|---|---|---|---|---|
| recovery fraction | 0% | 3% | 50% | **100%** | 100% |

→ The pipeline is blind below ~1σ, 50% efficient at 1.5σ, fully efficient at 2σ.

**On-source results (both statistics pre-registered):**

| event | A: max over Δt grid | B: at predicted Δt | verdict |
|---|---|---|---|
| GW150914 (Δt=0.2925 s) | p = 0.38–0.40 | p = 0.97–1.00 | null |
| GW151226 (Δt=0.0579 s) | p = 0.398 | p = 0.585 | null |

**The honest v1 sentence:** *GW150914-like echo trains with first-pulse amplitude
≥ 2 whitened-noise σ (per detector, 6 echoes, reflectivity 0.7) would have been
detected with ~100% probability at p < 0.01; the post-ringdown data of GW150914 and
GW151226 show no such structure (p ≈ 0.4–1.0). Consistent with Westerweck et al.*

**Bug found & fixed:** GWOSC blocks can contain NaN gaps when the science segment
starts/ends inside the requested span (GW151226 H1: first 133 s NaN). Fix: crop to
the longest contiguous finite run before whitening (`_longest_finite` in echolib).
Background population drops accordingly (159 → 117 pairs) — recorded, handled.

**Curious detail (recorded, not interpreted):** GW150914's statistic-B score
(−0.366) sits below the entire background distribution — an *anti*-correlation at
the predicted lag, i.e. a downward noise fluctuation. With N=159, being outside the
background on the low side is a ~1% tail either way; statistic B is one-sided by
design so this counts as p = 1.0. Worth re-checking with a larger background.

## 2026-06-12 — v2 PRE-REGISTRATION: the ML scorer through the identical harness

Design. A noise-trained anomaly scorer evaluated by the SAME injection +
background machinery as the v1 comb (that harness is the asset; the scorer is
the variable). Model: small 1-D conv autoencoder on 0.5-s windows (2048
samples) of whitened strain, trained ONLY on off-source noise. Score for a 3-s
segment: reconstruction-error envelope e(t) over sliding windows, then the
SAME network-coherent comb statistic as v1, applied to e(t) instead of the raw
envelope (periodicity stays the only shape assumption). CONTAMINATION RULE:
the 159 background pairs split 100 (scorer training) / 59 (background
distribution + p-values; floor rises to ~1/60) — scorer never trains on
segments that define significance, and never sees on-source data.
Pre-registered gates:
(V1) training stable; v2 background distribution well-behaved on the held-out
pool (no pathological tails vs v1's).
(V2) the head-to-head: injection-recovery curve of v2 vs v1 at identical
p < 0.01 thresholds; gate for "v2 wins" = 50%-recovery amplitude < 1.5σ.
PRE-REGISTERED HONEST ALTERNATIVE: on well-whitened near-stationary noise the
comb may already be near-optimal — a null ("NN adds nothing here; its value
would come from glitchy/nonstationary stretches") is a legitimate finding and
points v3 at glitch-rich segments rather than bigger models.
(V3) on-source rerun (GW150914, GW151226) with the v2 statistic — expect nulls;
report p-values alongside v1's.
Env note: torch being added to echoes/.venv (py3.14) — if wheels unavailable,
rebuild venv on python3.12 per repo standards.

## 2026-06-12 — v2 FIRST RESULT (provisional) + extended-run pre-registration

First head-to-head: **v2 (comb on AE residuals) recovered 50/50 injections at
EVERY amplitude tested, including 0.5σ where v1 was blind (0%)**; background
well-behaved (99th pct 0.093, N=59); on-source GW150914 null (p = 0.75/0.77).
Plausible mechanism: the AE suppresses the stationary noise floor so transient
pulses dominate its residual envelope (~3–4× effective gain). HELD AS
PROVISIONAL until two pre-registered checks pass:
(C1) the lower-amplitude curve (0.1–2.0σ) — find v2's actual 50% point; a
sensitivity claim without a measured floor is not a curve;
(C2) the SPECIFICITY control — inject pulse trains with IRREGULAR spacings
(same per-pulse energy, no fixed Δt) at 0.5σ and 1.0σ: statistic B (comb at
the predicted spacing) must NOT fire (recovery ≲ background rate ~1%). If it
fires, the pipeline is an energy detector wearing an echo costume, and the
claim dies. Architecture note recorded: the conv "AE" has latent size = input
size (no true bottleneck) — whatever it learned, the harness (held-out
background + injections) is the arbiter, but C2 is the test that the harness
itself can't fake.

## 2026-06-12 — v2 FINAL RESULTS: the ML scorer wins by ~13×

Extended run (low-amp curve + specificity control), all through the identical
v1 harness (59 held-out background pairs, p < 0.01 thresholds):

| amp [σ] | 0.1 | 0.2 | 0.3 | 0.5 | 0.75–2.0 |
|---|---|---|---|---|---|
| v2 recovery | 44% | 88% | 100% | 100% | 100% |
| v1 recovery (ref) | — | — | — | 0% | 3%@1.0, 50%@1.5, 100%@2.0 |

**Headline: v2's 50% point ≈ 0.11σ vs v1's 1.5σ — a ~13× amplitude-sensitivity
improvement** for the same injection family at the same false-alarm standard.
**C2 specificity ✓ (with a small honest caveat):** irregular-spacing trains of
equal energy fire at 6% (0.5σ) and 2% (1.0σ) vs 100% for periodic — the
detector requires periodicity at the predicted Δt; the 3/50 at 0.5σ is
marginally above the ~1% expectation (Poisson p ≈ 1.4%) — slight nonspecific
leakage at low amplitude, recorded. **V3 ✓:** on-source GW150914 remains null
(p = 0.75/0.77). Mechanism: the AE suppresses the stationary noise floor, so
transient pulses dominate its residual envelope; the comb then reads their
periodicity. Architecture note: the conv net has latent = input size (a learned
noise-suppression filter more than a bottleneck AE) — performance is
harness-validated either way.
Caveats inherited/new: sensitivity is family-specific (f0=250 Hz, τ=20 ms,
γ=0.7 — vary the family in v3); whitened-domain injection; background from the
same 512-s blocks; p-floor 1/60 on the eval pool.
**v3 queue:** waveform-family robustness sweep; physical-strain injections;
independent background blocks; GW151226 with per-event scorer; FAR scaling.

## 2026-06-12 — v3 PRE-REGISTRATION: waveform-family robustness of the 13×

The v2 sensitivity was measured on one family (f0=250 Hz, τ=20 ms, γ=0.7).
v3 sweep: saved scorers + existing background threshold, injections at amp
∈ {0.2, 0.5, 1.0}σ varying ONE family parameter at a time:
f0 ∈ {150, 250, 320} Hz (in-band), τ ∈ {10, 20, 50} ms, γ ∈ {0.5, 0.7, 0.9};
plus f0 = 450 Hz as a DELIBERATE out-of-band control (the 30–350 Hz bandpass
must kill it — physics, not pipeline failure; pre-registered expectation:
recovery collapses there and that's correct behavior).
Gate (W1): recovery at 0.5σ stays ≥ 80% across all IN-BAND variations (the
v2 advantage is family-robust, not tuned to one shape). Gate (W2): the
out-of-band control collapses (sanity that the band does its job).
n = 30 trials per (amp, config), eval-pool segments only.

## 2026-06-12 — v3 RESULTS: robust in-band; control invalid by design

**W1 ✓✓:** at 0.5σ, recovery 97–100% across ALL in-band variations (f0 150/320,
τ 10/50 ms, γ 0.5/0.9; baseline 100%); even at 0.2σ: 60–97%. **The 13×
advantage is family-robust, not tuned to one shape.**
**W2 INVALID-AS-DESIGNED (lesson recorded):** the f0=450 Hz "out-of-band"
control recovered 100% — because v1's standing simplification injects in the
WHITENED (post-bandpass) domain, so an out-of-band injection never meets the
filter; worse, post-filter noise has no power up there, so a 450 Hz transient
stands out MORE. A control must be implementable inside the simulation's
domain of validity — this one wasn't. Consequence: all v2/v3 sensitivity
claims are SCOPED to in-band morphologies, and **raw-strain injection is
promoted to the top of v4** (it converts the band claim from assumption to
measurement, and needs its own amplitude calibration — a pre-registered
iteration, not a patch).

## 2026-06-13 — v4 PRE-REGISTRATION: raw-strain injection (the production path)

Design: inject damped-sine-Gaussian echo trains into the RAW 512-s strain
blocks (eval-time region only, centers ≥ 308 s into the block = beyond the
scorer-training pairs), re-whiten 64-s slices through gwpy whiten(4,2) +
bandpass(30,350), crop the standard 3-s segment, score with the SAVED v2
scorers. Pulse times match v2's convention (first pulse at segment t = 0.1 s,
spacing 0.2925 s).
(X0) AMPLITUDE CALIBRATION: inject at reference raw amp 2e-21, measure the
whitened first-pulse amplitude (peak in a ±5 ms window minus a control-window
baseline, median over 8 trials/detector) → per-detector slope; raw amp for a
target whitened-σ equivalent = target/slope. Linearity spot-check at 2 ref
amps (ratio within 20%).
(X1) BACKGROUND CONSISTENCY: 30 noise-only trials through the 64-s-whitening
path; report the v2 statistic's 95th/99th pct vs the v2 threshold (0.093);
the spot check uses the NEW path's own 95th pct (n=30 → that's the honest
floor here; pre-registered).
(X2) SENSITIVITY SPOT CHECK: whitened-equivalents {0.2, 0.5, 1.0}σ, 30
trials each; gate: ≥ 80% recovery at 0.5σ — the 13× claim survives the
production injection path.
(X3) THE PROPER OUT-OF-BAND CONTROL: 450 Hz pulses injected in RAW strain at
the 1.0σ-equivalent raw amplitude — the 30–350 Hz bandpass must kill them;
gate: fire rate ≤ 10% (vs 100% in the invalid whitened-domain version).

## 2026-06-13 — v4 RESULTS: band-honesty measured ✓; sensitivity claims rescoped

**X3 ✓✓ (v4's mission):** 450 Hz raw-strain injections fire at 10% (~threshold
nominal) vs 100% in the invalid whitened-domain control — **the detector's
band-honesty is now a measurement, not an assumption.** X1 ✓ background
consistent across paths (99th: 0.082 vs 0.093). X0 ✓ calibration validated by
differencing (slope unchanged 1.98e21 — the first-run "bug" was immaterial;
fix round confirmed the calibration rather than changing it).
**X2 ✗ and it's REAL:** production-path recovery 57% @ 1.0σ-equiv, 3% @ 0.5σ
— the 50% point sits ≈ 1.0σ, far above the whitened-domain 0.11σ. Mechanism
(recorded as hypothesis): the bandpass+whitening reshapes raw pulses (filter
ringing spreads energy), so matched first-pulse PEAK ≠ matched detectability;
plus per-trial 64-s PSD noise and scorer distribution shift (trained on
512-s-whitened noise).
**Rescoped honest claims:** (1) the 13× was a SAME-CONVENTION comparison
(both methods, whitened-domain injections) — the relative claim is NOT
refuted but is UNVERIFIED in the production path; (2) absolute
production-path v2 sensitivity: 50% at ≈ 1.0 whitened-σ-equivalent (first-
pulse peak convention); (3) the completing experiment, queued at v5's top:
the comb baseline through the SAME raw path for a fair production-path
head-to-head. Process note: an untracked shell-& relaunch was killed and
relaunched tracked — background runs must go through the harness.

## 2026-06-13 — v5 PRE-REGISTRATION: the fair production-path head-to-head

Both statistics — v2 ML (comb on AE-residual envelope) and v1 comb (comb on
raw envelope) — computed on the SAME whitened segments, same raw-injection
trials, same path: shared backgrounds (n=30 noise trials, each stat's own 95th
pct threshold), shared injections at {0.5, 1.0, 1.5, 2.0, 3.0}σ-equiv × 25
trials (calibrated slope from v4). Gates: (Y1) both backgrounds well-behaved;
(Y2) BOTH 50% points measured through the identical path — the production-path
sensitivity ratio is then a MEASUREMENT (no gate on its value; pre-registered
expectation: ML still ahead but by less than 13×).

## 2026-06-13 — v5 RESULTS: the fair number is ~1.2×

Identical raw-injection path, shared trials, own 95th-pct thresholds (Y1 ✓:
ML 0.050, comb 0.136). Y2 ✓ both curves measured: ML 12%@0.5σ, 76%@1.0σ,
100%@1.5σ → 50% pt ≈ 0.85σ; comb 4%@0.5σ, 48%@1.0σ, 100%@1.5σ → ≈ 1.05σ.
**Production-path advantage ≈ 1.2–1.3× — the 13× was an artifact of the
whitened-domain convention** (unfiltered templates are maximally novel to a
noise-trained AE; filter-reshaped signals are not). FINAL echoes story: the
ML scorer is modestly but genuinely better through the real pipeline (76% vs
48% at 1.0σ), is band-honest (v4), family-robust (v3), specific to
periodicity (v2), and the on-source events remain null. The v2→v5 arc is a
case study in why production-path validation must precede sensitivity claims.

### v2 roadmap (next session)
1. The ML scorer: noise-trained model (autoencoder / next-sample predictor on
   whitened strain), scored through the IDENTICAL injection + background harness —
   compare its sensitivity curve against the comb baseline (06).
2. More background: pull separate event-free blocks (different days) → p floor
   below 1/160.
3. More events (O3/O4 catalog), per-event Δt computed from catalog mass+spin
   (currently hardcoded from Abedi Table I for the three O1 events).
4. Raw-strain injection (replace the whitened-domain v1 simplification).
5. Stack evidence across events (the predicted-Δt statistic combines naturally).

## 2026-06-20 — v6: UPPER LIMITS (roadmap P1 + P2) — non-detection → exclusion curve
Generalised 06's single-Δt sensitivity into a 2-D (amplitude × spacing Δt) efficiency map at HIGH N
(`11_upper_limits.py`). Per-Δt background threshold (comb-at-Δt over all off-source pairs, 99th pct =
p<0.01) → per-Δt amplitude A90 (90% recovery). Because the on-source data showed NO detection (05:
p_pred>0.01), echo trains louder than A90(Δt) would have been seen ≥90% of the time and were not ⇒
**we EXCLUDE first-pulse amplitude ≥ A90(Δt), at each spacing.**
- **GW150914 (N=300):** at predicted Δt=0.2925 s, **exclude amplitude ≥ A90 = 1.65σ** (A50 1.33σ);
  tightest A90 = 1.56σ at Δt=0.38 s. Curve smooth ~1.5–1.85σ across all spacings.
- **GW151226 (N=300):** at predicted Δt=**0.0579 s** (canonical on-source value), **exclude ≥ A90 =
  1.55σ** (A50 1.18σ) — also the tightest over the grid. [CORRECTED 2026-06-20: an earlier pass used a
  wrong Δt=0.105 s (a guess) and quoted 1.72σ; the north-star re-check caught it against the canonical
  0.0579 s from 05. GW150914's Δt=0.2925 s was correct throughout.]
- N=300 makes it decisive (binomial σ≈1.7% near 90%) — addresses P2 (underpowered-N concern).
- Units: whitened-noise σ (per-pulse SNR scale), reflectivity γ=0.7 fixed. The ML scorer (07) would
  tighten A90 ~1.2× (v5 head-to-head); comb used here = the statistic that produced the on-source null,
  so the limit is self-consistent. Artifacts: results/11_upper_limits_{GW150914,GW151226}.{npy,png}.
- **This converts the project's headline from "we found nothing" to "we exclude echoes above ~1.65σ."**
- **STRESS-TEST (north star, 2026-06-20) — PASSED.** (1) The single-Δt comb scoring (`score_at`) is
  provably identical to the validated 05/06 grid-then-index convention — `comb_on_env` computes each Δt
  independently and `_acf` normalizes per-segment, NO cross-Δt normalization. (2) Self-consistent: the
  exclusion, the per-Δt background, and the on-source null are all the same whitened-comb statistic.
  (3) No "13× artifact" — that was the v2 ML scorer's whitened number; the comb never had it. (4) The
  per-Δt threshold is a SMOOTH-tail 99th pct (top off-source scores 0.264/0.258/0.215…, no lone glitch;
  2.4σ above median) → not outlier-driven. (5) Cross-checks v1 (~100% @ 2σ). ⇒ the echo upper limit is
  sound and honestly framed (whitened-σ units, interpretable via the v4 raw-strain calibration slopes).
  Unlike the δ-stack, this fresh result holds up.

## 2026-06-20 — leg-8b SETTLED: the family-robustness "sensitivity reversal" is real signal, not noise
Re-ran 08 family-robustness at **N=300** (was N=30 → CIs overlapped, the leg-8b concern). `08
--n-trials 300 --tag _n300` adds binomial CIs + a two-proportion z-test vs baseline at the
discriminating 0.2σ. Result (baseline 82±4%):
- **In-band differences are REAL and physically sensible** (not a pathological reversal): f0=320 99%
  (+7.2σ) and gamma=0.9 97% (+6.2σ) genuinely EASIER; f0=150 73% (−2.7σ) and gamma=0.5 64% (−5.1σ)
  genuinely HARDER; tau=10/50ms consistent with baseline (noise). I.e. the ML scorer's sensitivity tracks
  waveform frequency/reflectivity in the expected direction — the N=30 "reversal" was just CI overlap.
- **The only true anomaly** — the f0=450 OUT-OF-BAND control reading 100% (+8.1σ, not noise) — is the
  KNOWN whitened-domain-injection invalidity: the valid v4 raw-injection test (09_raw_injection.json)
  correctly collapses it to **10%** (oob_rate 0.1). Convention artifact, already diagnosed + corrected.
- ⇒ leg-8b resolved: family-robustness holds at tight stats; no pathological reversal survives. The N=30
  version was underpowered, exactly as flagged. Original N=30 result preserved (08_family_robustness.json);
  high-N in 08_family_robustness_n300.{json,png}.

## 2026-06-25 — E1 (PLAN.md): does the ML scorer TIGHTEN the upper limit? Honest NO + an artifact trap caught
The ROADMAP guessed the v2 ML scorer would tighten the echo A90 exclusion by ~1.2× (the v5 production-path
advantage). Two findings, in order:
- **Artifact trap (caught before claiming).** The naive route — run the ML scorer through the whitened-domain
  upper-limit harness (`11`) — gives A90 ≈ 0.15σ, ~10× "tighter" than the comb. That is the **known v5
  whitened-domain convention artifact** (the same one that inflated the original "13×"), NOT a real limit.
  Reverted; `11` stays the clean gated comb UL.
- **Honest version** (`12_ul_production.py`, reuses the `09`/`10` raw-injection machinery): both statistics on the
  SAME re-whitened segments, p<0.01 over 45 off-source centers, n=120/cell. **At the predicted Δt: A90 comb=1.43σ
  vs ML=1.45σ → 0.98× — the ML scorer does NOT tighten the exclusion.** Across the grid the comb A90 is stable
  (1.3–1.6σ) while the ML A90 is noisier (NaN at 4/12 spacings: its heavier-tailed background pushes the p<0.01
  threshold up, so it can't reliably hit 90% efficiency).
- **Why:** v5's ~1.2× ML edge is at the **50% point** (low amplitude, where the AE-residual statistic helps); the
  upper limit is set at the **90% point**, in the steep/saturated part of the efficiency curve where the edge
  vanishes and the ML scorer's fatter noise tail actively hurts. ⇒ **ML doesn't help the exclusion; the gated comb
  UL stands.** Corrects the ROADMAP note. Caveat: 45 bg centers + n=120 is the production-path budget (coarser than
  `11`'s 159 whitened-domain centers); the predicted-Δt verdict is stable across n=12 and n=120 (both 0.98×).
  Gated in verify.sh. Artifacts: results/12_ul_production.{json,png}.
