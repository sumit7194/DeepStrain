# v1 Results — lab notebook (2026-06-10)

*A deep-learning trigger pipeline for subsolar-mass (primordial black hole candidate)
mergers in public LIGO data. Everything below is measured, not asserted; every
sensitivity number comes from injections into real O3a noise.*

---

## Headline numbers

| Quantity | Value |
|---|---|
| **Sensitive-distance fraction vs ideal matched filter** (zero-FA threshold) | **0.41 / 0.45 / 0.41** for chirp mass 0.17–0.35 / 0.35–0.55 / 0.55–0.88 M☉ |
| Detector SNR at 50% efficiency | ≈ 18.6 (ideal MF: 8 by definition) |
| Efficiency at SNR 18–21 / 21–24 | 0.56 / 0.95 |
| False alarms at chosen threshold | **0 in 6.80 h** of held-out real H1 noise (2868 windows) |
| End-to-end demo | SNR-22 event recovered in coincidence: network SNR 36.4, **dt error 0.02 ms** (−1.95 vs −1.93 ms true), per-detector timing error 0.1 ms, **18 glitch/noise peaks rejected by coincidence** |
| Template dephasing (the "why ML" plot) | ±0.01% chirp-mass error → −28% SNR; ±0.1% → −60%; ±1% → lost |

**Context:** the published state of the art for ML detection (MLGWSC-1) reached ~70% of
matched-filter sensitive distance on real noise for **≤20 s signals**. This v1 reaches
**41–45% on minutes-long subsolar signals** — the regime with no published ML search —
with a 1.17M-parameter CNN trained 40 min on a laptop. The 45→70% gap is the research
target for v2.

---

## What was built

```
noise (GWOSC, real O3a)          waveforms (PyCBC TaylorF2)
  24×4096 s H1 + 2×4096 s L1       masses 0.2–1.0 M☉, f from 50 Hz
  16 train / 2 val / 6 test          → antenna-projected, whitened
        └──────────────┬──────────────┘
                 dataset builder
        40k train / 2.5k val spectrograms
        (256 s windows, 128 log-f bins × 256 time bins, [50,1024] Hz)
                       │
              ┌────────┴────────┐
        SpectrogramCNN     ChunkTransformer
         1.17M params        0.82M params
         AUC 0.777 ✅         AUC 0.758
                       │
              evaluation harness
        FAR scan (6.8 h test noise) → zero-FA threshold
        1500 injections → efficiency vs SNR vs mass
                       │
                search pipeline
        ML sweep (cheap) → MF follow-up (precise)
        → H1×L1 peak-pair coincidence ≤15 ms
```

## The conventions (load-bearing)

- **Band-limited [50, 1024] Hz everywhere.** GWOSC 4 kHz data is anti-alias filtered,
  so the measured PSD near Nyquist is artificially tiny and the whitening weight
  `1/√S` explodes there. All SNRs (model *and* matched-filter baseline) use the same
  band — apples to apples.
- **Whitening normalized so `Σ h_w² = SNR²`** — verified against PyCBC `sigma()` to
  0.0% and by blind matched-filter recovery to 0.0 ms (tests/test_injection.py).
- **Threshold = zero false alarms on held-out real noise.** With 6.8 h of test data
  that bounds FAR ≲ 0.15/h — far looser than the production 1/month standard. Honest
  scaling to 1/month needs ~700 h scanned (days of GPU/CPU time, mechanical).
- Single-window scoring for efficiency numbers; the sweep gets multiple overlapping
  tries per signal, so deployed sensitivity is slightly better than quoted.

## Findings

1. **A plain CNN is a real (if modest) subsolar trigger.** 41–45% of ideal-MF
   sensitive distance at a zero-FA operating point, uniform across the subsolar mass
   range — encouragingly flat (no mass bin collapses).
2. **The chunked-transformer hypothesis lost round one** (0–41% vs CNN's 41–45%; worse
   noise-score tails, which is what kills you at strict thresholds). Honest negative.
   Suspected cause: per-chunk GAP destroys within-chunk track structure before
   attention sees it; 16 tokens is a trivially short sequence. v2: ViT-style 2-D
   patches, overlapping chunks, attention pooling.
3. **Overfitting was the first failure mode** (run 1: val AUC 0.72 collapsing to 0.67).
   Fix that worked: 2× noise diversity (8→16 segments) + augmentation (time shift,
   freq jitter, additive noise on log-power) + dropout 0.25 + weight decay 1e-3
   → stable AUC 0.777, eff@1e-3 0.42.
4. **The hierarchical design works end to end.** ML sweep flags ~30 hot windows around
   a real-loudness event in each detector; matched-filter follow-up localizes to
   0.1 ms; two-site coincidence recovers the true pairing at network SNR 36.4 while
   **rejecting 18 louder-or-comparable spurious peaks — including an H1 glitch that
   out-correlated the true signal (SNR 28.6 vs 22.4).** Coincidence is not decoration;
   it did real work in this demo.
5. **The template-dephasing measurement** (results/bank_mismatch.png): for a ~70 s
   subsolar inspiral, a chirp-mass error of *one part in ten thousand* already costs
   28% of SNR. This is the compute wall that makes classic subsolar template banks
   enormous — and the quantitative argument for a cheap learned trigger stage.

## Plots

- `results/dataset_montage.png` — what the model sees (tracks vs glitches)
- `results/efficiency_cnn.png` / `efficiency_transformer.png` — efficiency curves vs
  SNR per mass bin + MF-distance fraction bars
- `results/bank_mismatch.png` — SNR recovery vs chirp-mass mismatch
- `results/eval_cnn.json`, `eval_transformer.json`, `search_demo.json` — raw numbers

## Honest limitations of v1

- FAR demonstrated only to ~0.15/h (zero in 6.8 h); production claims need 1/month.
- L1 never seen in training (H1 only); the demo nonetheless triggered on L1 — good
  sign for domain transfer, but L1/V1 training data belongs in v2.
- Follow-up used the *true* template (validates plumbing, not a blind bank search).
  The dephasing curve shows exactly why a blind fine bank is expensive — that cost is
  the thing the ML stage must amortize, and v1 does not yet demonstrate the full blind
  hierarchy.
- Eval injections share segments with the FAR scan (disjoint from training, but the
  same 6.8 h of noise); a bigger held-out pool would decouple them.
- No spin, no eccentricity, no mass ratio beyond the component-mass grid.

## v2 directions (ranked)

1. **Close the gap (45% → 70%+):** deeper/wider CNN, curriculum on SNR, overlapping
   sweep windows with score aggregation across a track's full duration (cheap,
   physics-aware), multi-window coincident scoring in the statistic itself.
2. **Scale the FAR floor:** stream O3a bulk data through the trained model (GCP box),
   push the noise pool to hundreds of hours, re-measure at FAR 1/month equivalent.
3. **Fix the transformer fairly:** ViT-style patches + attention pooling, then rematch.
4. **Two-detector training** (H1+L1 channels or score-level combination).
5. **The under-covered corners:** eccentric and high-mass-ratio subsolar injections —
   where even a modest-sensitivity search is genuinely unexplored territory.

---

# v2 rung 1 — track-score aggregation (PRE-REGISTERED 2026-06-13, before running)

**Hypothesis.** The v1 statistic uses one window score per injection, but with the 8-s
sweep hop any signal lies inside ~32–78 overlapping 256-s windows. Aggregating the score
*series* along the track should recover part of the 45→70% gap and suppress isolated
glitch-like spikes. Honest expectation: adjacent windows share 248/256 s of data, so
noise scores are highly correlated and gains at this rung should be **modest**; the
independent-window version (shorter windows + retraining) is rung 2.

**Protocol change (production-path).** Inject the full whitened waveform into the
whitened *segment* (entire 512-s buffer inside the valid region — no truncation), sweep
the identical window grid as the FAR scan, recompute only signal-contaminated windows,
splice them into the cached noise score series, and evaluate aggregated statistics at
positions touching the contamination. This removes v1's asymmetry (noise threshold from
a sweep, injections scored in one pre-cut window).

**Statistics (all thresholds = zero-FA: max of the statistic over the identical
6-segment noise sweep, so trials factors price themselves in):**
- `max` — control; must land near v1's 0.41–0.45 under the new protocol.
- `boxcar_bank` — running mean over k consecutive windows, max over k ∈ {32,48,64,80,96}
  (track presence = signal duration + 256-s window ≈ 280–620 s ≈ 35–78 windows).
- `count_above` — longest consecutive run above the pooled-noise 99th percentile
  (consistency statistic; glitch-shaped complement to the boxcar).
- `boxcar_oracle` — k from the injection's true duration, per-k noise threshold,
  **diagnostic ceiling only** (pays no bank trials; never a headline).

**Decision rule (relative, no guessed absolute gates):**
(a) `boxcar_bank` clearly above `max` → iterate at rung 1.
(b) bank ≈ max but oracle clearly above → bank design problem, fix the bank.
(c) oracle ≈ max → the gap is within-window; rung 1 exhausted → rung 2 with evidence.
Secondary pre-registration: track statistics should *suppress* single-window
glitch-like spikes relative to `max` (purity check).

**Campaign:** 250 injections/segment × 6 test segments, SNR ∈ [4, 24] (total
band-limited SNR of the full signal — equals in-window SNR for fitting signals),
masses/sky from the v1 population, seed = SEED+888.
**Artifacts:** `results/track/` (per-segment caches, atomic + resumable),
`results/eval_cnn_track.json`, `efficiency_cnn_track.png`. v1 artifacts untouched;
`eval_cnn.json` remains the gated v1 record.

## v2 rung 1 — RESULTS (2026-06-13): clean negative, the gap is *within-window*

1500 injections (250 × 6 test segments), 6.79 h zero-FA noise. `max`-control threshold
reproduced v1 **bitwise** (2.7070367336273193); splice selftest exact (max|diff| 0.00e+00).

| sensitive-distance fraction | 0.17–0.35 | 0.35–0.55 | 0.55–0.88 |
|---|---|---|---|
| v1 (single pre-cut window) | 0.406 | 0.446 | 0.413 |
| `max` (sweep — this protocol) | 0.402 | 0.448 | 0.447 |
| `boxcar_bank` | 0.403 | 0.452 | 0.447 |
| `count_above` | 0.403 | 0.456 | 0.455 |
| `oracle` (true-duration k) | 0.402 | 0.447 | 0.455 |

**Decision → (c): `oracle ≈ max` ⇒ the gap lives inside the window; rung 1 exhausted.**
Track-shape aggregation does not clear the `max` control in any mass bin (≤ +0.01), and
crucially *neither does the oracle ceiling that knows the true duration*. The multiplicity
of overlapping windows carries almost no independent information — adjacent 256-s windows
share 248 s of data and the signal already fits inside one window, so ~32 overlapping
copies add nothing the best-aligned single window didn't already have. The pre-registered
"modest gains" outcome, sharpened: the gain is ~zero, and the oracle proves it is **not**
a bank-design problem (rules out branch (b)).

**Bycatch (real, modest).** The *protocol* — sweep + max over contaminated windows vs v1's
single randomly-placed window — lifted the high-mass `max` bin 0.413 → 0.447. That is a
**window-alignment gain**: v1's single-window scoring with a random merger position was
slightly pessimistic for short (high-mass) signals; sweeping lets the best-aligned window
win. v1's headline was thus a touch conservative at high mass; the honest within-protocol
baseline is ≈ 0.40 / 0.45 / 0.45 across the three bins.

**Secondary purity check: deferred** to rung 2 — suppression of single-window glitch
spikes needs the coincidence-demo glitch (`search.py`), not this single-detector
efficiency campaign.

**Verdict.** Rung 1 closes as a documented negative with a sharp pointer: the 45→70% gap
is **within the 256-s window**, so closing it needs **rung 2 — shorter windows (more
*independent* track views → genuine SNR² accumulation) + retraining**, exactly the
escalation pre-registered. Still local (retrain ≈ 40 min on the laptop); no VM. Artifacts:
`eval_cnn_track.json`, `efficiency_cnn_track.png`, `injections_cnn_track.parquet`.

# v2 rung 2 — shorter windows + accumulation (PRE-REGISTERED 2026-06-13, before building)

**Hypothesis.** Rung 1 failed because a 256-s window holds the whole signal and overlapping
windows are redundant. Shorter **non-overlapping** windows each see a distinct arc of the
chirp → independent evidence. The CNN emits a logit (BCEWithLogitsLoss), and independent
per-window log-odds **add**, so summing logits across the track is a genuine incoherent
SNR² accumulator — the thing rung 1's redundant windows could not provide.

**No architecture change.** `SpectrogramCNN` ends in `AdaptiveAvgPool2d(1)`, so it accepts
any `(128, T)` input. Rung 2 is the *same* model retrained on 64-s windows (input 128×63).

**Window = 64 s** (4× shorter than v1). Reasoned from durations (50 Hz cutoff): low-mass
0.17–0.35 M☉ (110–367 s) spans 2–6 windows, mid 1–2, high 1 — accumulation can pay where
signals are long, while per-window SNR stays inside the trained 5–30 range. W=32 (per-window
SNR near the floor) is a one-config-flip follow-up if 64 shows promise.

**Sweep non-overlapping** (hop ≈ window) so summed logits accumulate rather than double-count
(the rung-1 lesson). Training merger placement spans window boundaries so the model sees
partial-arc positives.

**Statistics** (thresholds = zero-FA over the identical short-window noise sweep):
- `max` — control; expected *worse* than v1 (a 64-s window sees less signal).
- `sum_track` — max over k ∈ {2,3,4,6,8} of summed consecutive logits, **per-k** noise
  thresholds (different k have different scale); the physical accumulator.
- `oracle` — sum over exactly the true-track windows (k from injected duration); ceiling.

**Decision rule (relative):**
(a) `sum_track` clears the v1/rung-1 best, especially low-mass → accumulation works, gap
closing → iterate (W=32, deeper combine).
(b) `oracle` clears but `sum_track` (blind bank) does not → bank/combine design issue.
(c) even `oracle` ≈ v1 `max` → per-window SNR too low for the CNN to extract; chunking loses
more than it accumulates → documented limit of the score-aggregation route → escalate to a
genuinely sequence-aware model (the ChunkTransformer's premise) or coherent methods.

**Build:** pipeline parameterized by window length (v1 shards/model untouched); 64-s shards
in `data/shards_w64/`, model `models/cnn_w64.pt`, eval `eval_cnn_w64_track.json`. All local,
background, atomic/resumable. Seed = SEED + 999 for the campaign.

## v2 rung 2 — RESULTS (2026-06-13): accumulation fails too; score aggregation exhausted

`cnn_w64` trained on 40k/2.5k 64-s shards → val AUC **0.793** (beats v1's 0.777 — a 63-bin
spectrogram concentrated on the near-merger arc is a *cleaner* per-window classifier, so the
test is fair, not weak-model-limited). Splice selftest exact (0.00e+00). 1500 injections,
64 non-overlapping windows/segment (384 noise windows), thresholds `max` 1.82 / `sum_track` 0.95.

| sensitive-distance fraction | 0.17–0.35 | 0.35–0.55 | 0.55–0.88 |
|---|---|---|---|
| `max` (control) | 0.407 | 0.457 | 0.476 |
| `sum_track` (√k-normalized accumulator) | 0.405 | 0.463 | 0.475 |
| `oracle` (sum over the true-track windows) | 0.406 | 0.464 | 0.475 |

**Decision → (c): `sum_track` ≈ `max` ≈ `oracle` (≤ +0.007 in every bin).** The oracle —
handed the exact number of windows the signal spans — *cannot* beat the single best window,
so independent per-window evidence does **not** accumulate. Physical read: subsolar-inspiral
SNR is not spread across a long track of comparable pieces; the early-inspiral windows sit
below the per-window detection floor, so summing them adds noise, not signal. Matched
filtering's edge is *phase coherence*, which summing per-window scores cannot recover.

**Confound flagged (no improvement claimed):** the rung-2 fractions read slightly above v1,
but non-overlapping 64-s windows give only 384 noise windows vs v1's 2868, so the zero-FA
threshold sits on a smaller sample — an easier, higher-FAR operating point. That inflates all
three rung-2 stats *equally*, so the within-rung `sum_track`-vs-`max` comparison is clean, but
a "shorter window helps" claim would need a FAR-matched comparison and is **not** made here.

**Verdict — score aggregation is exhausted (both rungs).** Window-score combination cannot
close the 45→70% gap: confirmed with overlapping windows (rung 1) and independent windows +
retraining (rung 2), each with an oracle ceiling that also fails. Closing the gap needs a
different mechanism — a sequence-aware model that integrates partial-track evidence with
learned weighting (the ChunkTransformer premise, done right) or phase-coherent stacking — not
score aggregation. Artifacts: `eval_cnn_w64_track_w64.json`, `efficiency_cnn_w64_track_w64.png`,
`models/cnn_w64.pt`.

# v2 rung 3 — semi-coherent learned bank (PRE-REGISTERED 2026-06-13, before building)

**Diagnosis driving the pivot.** All our detectors eat magnitude spectrograms, which discard
phase — the quantity matched filtering integrates coherently. MLGWSC-1 (arXiv:2209.11146)
states spectrograms are sub-optimal at low SNR; its ~70%-of-MF methods used phase-preserving
1-D whitened strain. Choice: NOT a monolithic 1-D ResNet port (option A — derivative, and
structurally capped at long signals); instead **option B, an MFCNN-style semi-coherent
hybrid** aimed at subsolar's specific pathology, motivated by v1's measured bank-dephasing
(±0.01% Mc → −28% SNR: full-length banks explode, but short *chunks* dephase gently, so a
small LEARNED bank of coherent chunks + learned cross-chunk combination might cover the mass
range). Why it can work where rung 2 failed: a CNN logit is not an SNR — but a per-chunk
matched-filter output is the optimal per-chunk statistic, and chi^2-combining those is the
textbook semi-coherent search. Chunked front-end also scales to minutes-long signals (rung 4).

**Stage 0 (THIS stage) — oracle ceiling before any training.** Per-chunk matched filtering
with the TRUE per-injection template (unit-norm quadrature/analytic chunks), S = max_t of
sum_i |rho_i(t+tau_i)|^2, on the identical 64-s window / zero-FA / 6-segment real-noise
convention as cnn_w64. Sweep n_chunks ∈ {1, 2, 4, 8, 16} (64 s → 4 s coherence): n=1 is the
fully-coherent in-window matched filter (upper anchor, expect ≈ ideal in the high-mass bin);
n=16 is closest to a learnable bank. Thresholds: zero-FA = max of S over the 6 noise segments
(continuous-t trials — MORE trials than the 384-window grid; convention declared, not mixed).
Template-independence of the noise distribution (unit-norm chunks) checked with 3 spaced Mc
templates. 1500 injections, same population/SNR ∈ [4,24], seed = SEED + 1111.

**Pre-registered gate:** the architecture class is viable iff some chunk config's oracle
clearly exceeds cnn_w64's same-protocol 0.407 / 0.457 / 0.476 (margin ≥ +0.05 in ≥1 mass
bin). Chunk length for the learned model = best measured ceiling-vs-bankability trade. If
even n=1 fails to clear cnn_w64, the 64-s window is the cap → rethink window, not the model.
(The n=1 and n=16 oracles also bracket option A's potential on this convention — priced
either way.) Expected semi-coherent loss vs n is MEASURED, not asserted from scaling lore.

**Artifacts:** `scripts/semicoherent_oracle.py`, `results/oracle_semicoherent.json` + png.
No model is built or trained in stage 0.

### Stage 0 AMENDED (2026-06-13, after the smoke run revealed a glitch problem)

The first smoke gave 0% everywhere. Diagnosis (not a bug — verified): the raw
matched-filter statistic is **glitch-dominated** on real O3a noise. Synthetic white
noise gives n=1 max S = 30.2 (= the chi^2_2n expectation ~33, so the statistic is
correctly scaled); real noise gives 717 (n=1) / 61450 (n=16) because the test segment
holds a **2310-sigma glitch**. A SNR-20 signal only reaches S~400, so a single glitch
sets the zero-FA bar above every recoverable signal. This is textbook — raw MF on real
noise is unusable without a **chi^2 signal-consistency veto**; our cnn_w64 is implicitly
glitch-robust, so a no-veto oracle is an unfair floor.

The chunk decomposition *is* the veto: a real chirp spreads SNR^2 across chunks in
proportion to per-chunk template energy p_i; a glitch concentrates it. So we add the
standard PyCBC-style reweighting — chi^2 = sum_i (rho_i^2 - p_i S)^2/(p_i S),
newSNR = rho / [(1+chi2_r^3)/2]^(1/6) (chi2_r>1) — which n=1 (one chunk) CANNOT do, so
n=1 stays glitch-vulnerable by construction; the veto is itself an argument for chunking.

**Revised stage 0 measures THREE ceilings per n_chunks, one run:**
- **clean** — raw S vs a *synthetic-noise* zero-FA threshold → pure phase-coherence ceiling.
- **vetoed** — newSNR vs a *real-noise* newSNR threshold → realistic, glitch-robust ceiling.
- **raw-real** — raw S vs real-noise S threshold → the glitch-limited reference (the gap
  clean→raw-real is the glitch tax; vetoed shows how much the chunk-consistency recovers).

**Gate (revised):** option B is viable if EITHER the clean ceiling clears cnn_w64's
0.407/0.457/0.476 by >= +0.05 (phase helps) OR the vetoed ceiling does (phase+robustness
helps); the best n picks the chunk length. If both hug cnn_w64, the 64-s window is the cap.

### Stage 0 RESULTS (2026-06-13): GATE CLEARED — first non-negative rung. n=8 is the spec.

1500 injections, all three ceilings (fraction of ideal-MF sensitive distance):

| n_chunks | clean | **vetoed (realistic)** | raw-real |
|---|---|---|---|
| 1 | 1.106 / 1.282 / 1.356 | 0 / 0 / 0 | 0 |
| 2 | 1.107 / 1.293 / 1.379 | 0 / 0 / 0 | 0 |
| 4 | 1.105 / 1.243 / 1.328 | 0 / 0 / 0 | 0 |
| **8** | 1.039 / 1.075 / 1.183 | **0.663 / 0.764 / 0.752** | 0 |
| 16 | 0.930 / 0.966 / 1.025 | 0.601 / 0.607 / 0.512 | 0 |
| *cnn_w64 gate* | — | *0.407 / 0.457 / 0.476* | — |

**Verdict: n=8 (8-s coherence) clears the gate by +0.25 to +0.30 in every mass bin** —
SNR50 ~11 vs the CNN's ~18, landing in the field's ~70% neighborhood. The chunk-count
sweet spot is itself a finding: n<=4 fail (too few chunks -> glitch veto too weak ->
glitch-limited like raw-real), n=16 over-chunks (coherence loss); **n=8 balances phase
coherence against glitch-robustness.** First rung that is NOT ruled out.

**HONEST CAVEATS (this is a ceiling, propped up by a lenient threshold — do not read as
the achievable number):**
1. **Clean > 1.0 is a tell, not a triumph** — nothing beats the ideal MF (the oracle IS
   MF). >1.0 means the zero-FA threshold is too lenient: set on only 6 short noise segments
   (tiny sample -> low bar). The SAME leniency inflates the vetoed 0.66-0.76; a realistic
   (hundreds-of-hours) noise pool has more glitches -> higher threshold -> lower fractions.
   Treat 0.66-0.76 as OPTIMISTIC; the realistic-FAR number is unmeasured and lower.
2. **Oracle = true templates.** A learned bank won't have the exact waveform (v1 measured
   +-0.01% Mc -> -28% SNR); the built model lands BELOW this ceiling, possibly well below.
3. cnn_w64's gate has its own small-noise caveat (384-window grid) and uses a different
   trials convention (continuous-t here), so the +0.25 margin is qualitative, not exact —
   but it is large enough to survive the convention slop.

**Decision:** build stage 1 — a LEARNED n=8 semi-coherent model with the chunk-consistency
veto — and measure how far below this ceiling it lands. The gap oracle->learned is the real
open question. Artifacts: `scripts/semicoherent_oracle.py`,
`results/oracle_semicoherent.json`, `results/oracle_semicoherent.png`.

# v2 rung 3 STAGE 1 — learned semi-coherent model (PRE-REGISTERED 2026-06-13, before building)

**Goal:** does a LEARNED n=8 model approach the stage-0 oracle ceiling (vetoed 0.66/0.76/0.75)?
The real win is beating cnn_w64 (0.41/0.46/0.48) with a phase-aware, strain-input model.

**Input:** 64-s WHITENED STRAIN window (262144 samples) — NOT a spectrogram (the rung-1/2/CNN
representation discarded phase; this keeps it).

**Architecture — `SemiCoherentNet` (~1M params):**
- 64-s window -> 8 chunks of 8 s (32768 samp). A SHARED per-chunk 1-D conv encoder (strided
  ResNet-style, SIGNED features so phase coherence builds through receptive-field depth, not a
  single giant matched-filter kernel) -> per-chunk embedding z_i and scalar score rho_i.
- Cross-chunk combiner: {rho_i} + a consistency feature (spread of rho_i across chunks = the
  LEARNED analog of stage-0's chi^2 veto, the glitch defense) -> MLP -> logit.
- Coherence length is emergent (receptive field), so the oracle->learned gap partly measures
  how much coherence the learned encoder captures vs the oracle's exact 8-s chunks.

**Data pipeline (storage-light, ~1.6 GB not ~22 GB):**
- Pre-generate a pool of ~3000 whitened waveforms once (last 64 s + full snr_ref + masses),
  whitened with a representative train PSD. SIMPLIFICATION (declared): the pool uses one
  representative PSD, so injected signals are slightly mismatched to each noise segment's exact
  PSD — fine for TRAINING (model learns morphology); EVAL uses the oracle's exact per-segment
  whitening for a fair comparison.
- On-the-fly injection Dataset: random whitened-noise 64-s window + (p=0.5) a random pooled
  waveform scaled to SNR ~ U(4,30) at a random merger position. Unlimited injection diversity.
- Train/val split by noise segment AND pool waveform (no leakage). Seed = SEED + 2222.

**Eval:** the trained model is a 64-s-window detector (rung 2 settled that per-window max is
right). Score test noise (zero-FA threshold) + a 1500-injection per-segment campaign using the
SAME proper per-segment whitening as the oracle/cnn_w64 -> mf_distance_fraction per mass bin.

**ML discipline (per the skill):** overfit a single batch to ~100% BEFORE full training
(capacity/expressivity check); LR 3-point sweep (3e-4/1e-3/3e-3); train to val-AUC plateau;
>=3 seeds for the headline. Gate cnn_w64's val-AUC 0.793 as a sanity floor.

**Decision rule (pre-registered):**
- WIN if learned mf_distance_fraction clears cnn_w64 (0.41/0.46/0.48) by >= +0.05 in >= 1 bin
  (a phase-aware learned detector beats the magnitude CNN — the real result).
- STRETCH: how close to the oracle 0.66/0.76/0.75 (the oracle->learned gap).
- NEGATIVE (also valuable): if it cannot beat cnn_w64, the learnable bank failed to realize the
  phase advantage the oracle proved exists -> the bottleneck is learnability, not the window.

**Artifacts (planned):** `scripts/build_waveform_pool.py`, `data/waveform_pool/`,
`pbh/models.py::SemiCoherentNet`, `models/semicoherent.pt`, `results/eval_semicoherent.json`.

### Stage 1 RESULTS (2026-06-13): NEGATIVE — learned design caps ~0.69 AUC, doesn't realize the ceiling

First full run (sweep winner lr=1e-3, 16 epochs) was **unstable**: val AUC peaked 0.687 at
epoch 0 then collapsed/thrashed to ~0.35 (below chance) while train loss kept dropping. Eval
of the best model: **mf_distance_fraction = 0.000 / 0.000 / 0.000** (vs cnn_w64 0.41/0.46/0.48,
oracle 0.66/0.76/0.75). At a zero-FA threshold you need ~0.78+ AUC for any non-zero distance.

**LR + clipping exhausted (8k/8–10-epoch probes), one stable config, hard ~0.69 ceiling:**

| config | behavior | best val AUC |
|---|---|---|
| lr 3e-4 | stable, flat plateau | **0.69** |
| lr 5e-4 | climbs then collapses (ep3) | 0.63 |
| lr 1e-3 | collapses (ep1) | 0.69 (transient) |
| lr 3e-4..3e-3 + grad-clip 1.0 | clipping does NOT stabilize | 0.55 / 0.69 (thrash) |

Collapses are sudden val cliffs to **below chance** while train loss is smooth → exploding-
gradient instability the high LRs can't escape and clipping doesn't fix; the only stable LR
(3e-4) converges flat at ~0.69. So the ~0.69 is an **architecture/representation ceiling**, not
an optimization one — and 0.69 < cnn_w64's 0.79, so this learned design lands at ~0 sensitive
distance. **Stage 0 proved the phase information is recoverable (oracle 0.66–0.76); this learned
semi-coherent architecture does not realize it.**

**Not yet closed — exhausting the hurdles before concluding:** (B) architecture attempt — a
learnable matched-filter front end (long quadrature kernels) to beat the 0.69 ceiling; (C) the
full lr=3e-4 / 20k-sample run for the definitive number on the current architecture. Artifacts:
`models/semicoherent*.pt`, `results/eval_semicoherent.json`, diagnostic probes `models/diag_*`.
