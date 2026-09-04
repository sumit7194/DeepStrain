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

## v2 ARC — PARKED COMPLETE (2026-06-15)

> One-screen summary of the whole v2 effort to close the 45→70% gap. Full per-rung detail,
> tables, and caveats are in the dated sections further down.

**The arc.** We attacked the 45→70% single-detector gap from every tractable angle and learned
exactly where the wall is:
- **Rungs 1–2 (score aggregation):** NEGATIVE. Combining scores across overlapping/long windows
  (boxcar, count-above, √k-summed accumulation) never beats the single-window `max`, and neither
  does the duration `oracle` ⇒ the gap lives *inside* the window; independent per-window evidence
  doesn't accumulate.
- **Rung 3 (phase/representation):** stage 0 oracle proved the phase information is physically
  recoverable (true-template chunked MF, 0.66–0.76 ≫ cnn_w64). But stage 1 — two learned-from-strain
  semi-coherent designs (V1 ResNet on strain, V2 learnable matched-filter front end) — both cap at
  ~0.69–0.71 AUC → **0 sensitive distance** at zero-FA. Robust across both architectures.
- **Second pass (diagnostics → pivot):** the glitch hypothesis was refuted but sharpened (fat noise
  tail, not one glitch). A coarse template bank scores 0.000, and the clean true-vs-bank diagnostic
  **quantified the blocking point** (below). Pivoted to multi-detector coincidence.
- **Path G (coincidence):** **POSITIVE — the win.** H1×L1 coincidence with a time-slide background,
  riding the learned model, gives **+1.3–1.5× sensitive distance (~2.3–3.3× volume)** over the
  single-detector ML search at matched FAR (1.48× high-mass). Every refinement lever then squeezed:
  better statistic (no gain, `sum` optimal), H1+L1 training (no gain, AUC↑ but coincidence flat).

**THE FINAL BLOCKING POINT (where we'd resume).** Subsolar matched filtering is *brutally* template-
density-limited: measured dephasing is +0.01% Mc → ~perfect, +0.1% → −30%, **+1% → dead**. Covering
the subsolar mass range at the required ≤0.1% spacing needs **~1,600+ templates** (mass alone; ×more
for spin), whose trials also inflate the noise floor. That density is intractable on a Mac Mini, and
it blocks BOTH (a) a real matched-filter-grade detector and (b) the fine (10 ms) timing coincidence
that would extend Path G. A *learned* detector sidesteps the density problem (it generalizes across
mass) but is then noise-floor-limited single-detector — which is exactly why coincidence is the only
lever that worked. **To go past +1.4× we'd need real compute (GPU/GCP): a dense coherent bank + lower
FAR, or true-waveform-supervised learning. That's the come-back-later thread.**

**Honest headline.** A single-detector *learned* subsolar search is noise-floor-limited at ~41–48% of
ideal-MF distance; requiring two-detector (H1×L1) coincidence recovers **~1.4× sensitive distance**,
and that is the ceiling for the learned approach at this data/compute scale. Null and positive results
both real, both measured on real O3a noise with injection-based sensitivity and time-slide backgrounds.

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

### Stage 1 RESULTS (2026-06-13→14): DEFINITIVE NEGATIVE — both learned designs cap ~0.69–0.71 AUC, zero sensitive distance

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

**Hurdles exhausted (2026-06-14) — B and C both run; the negative is now definitive.**
Pre-registered both before running: B should beat 0.69 if the bottleneck is the front-end
*representation*; C pins V1's plateau at the same budget V2 gets.

**(B) `SemiCoherentNetV2` — learnable matched-filter front end.** A bank of 64 quadrature
templates (`Conv1d(1,128,k=2048)`), squared+summed per pair into a phase-invariant
`|⟨d, template⟩|²` SNR map — the oracle's statistic, made learnable — then the same per-chunk
back end as V1. Capacity gate passed (memorizes a batch to 100%). Full run lr=3e-4 / 20k / 20
epochs: **stable and monotonic, clean plateau at val AUC 0.691** (best ep16; last 3 epochs
0.688/0.690/0.689 — saturated, no thrash). Eval: **0.000 / 0.000 / 0.000**.

**(C) `SemiCoherentNet` definitive — full lr=3e-4 / 20k / 20 epochs.** The "flat 0.69 plateau"
seen in the short probes was a **probe-length artifact**: at full budget V1 does not plateau —
it **overfits and goes unstable** (train loss falls smoothly 0.50→0.46 while val AUC oscillates
0.31↔0.62, sitting *below chance* on most late epochs). Best **0.706** (hit early @ ep3, never
recovered). Eval: **0.000 / 0.000 / 0.000**. (Survived a mid-run power loss; resumed from the
epoch-5 atomic checkpoint with zero work lost.)

| design | front end | behavior at full budget | best val AUC | sensitive distance (zero-FA) |
|---|---|---|---|---|
| V1 `SemiCoherentNet` | conv on raw strain | overfits, val thrashes 0.31–0.62 | 0.706 (unstable) | **0.000 / 0.000 / 0.000** |
| V2 `SemiCoherentNetV2` | learnable matched filter | monotonic, clean plateau | 0.691 (stable) | **0.000 / 0.000 / 0.000** |
| cnn_w64 (ref) | spectrogram | — | 0.793 | 0.41 / 0.46 / 0.48 |
| oracle ceiling | true templates | — | — | 0.66 / 0.76 / 0.75 |

**Conclusion — stage 1 CLOSED, definitive negative.** The ~0.69–0.71 AUC wall is **robust
across both natural learned realizations** of a semi-coherent detector — it is not an
architecture quirk or an optimization failure (V2 converges cleanly and still hits it). Adding
an explicit matched-filter front end did *not* help; it only made training better-behaved.
Stage 0 proved the phase information is physically recoverable (oracle 0.66–0.76 ≫ cnn_w64),
but **neither learned-from-strain design realizes it at this data/training scale**, both landing
below cnn_w64's 0.79 → zero sensitive distance at the zero-FA threshold. The 45→70% gap needs a
genuinely **coherent / fully-matched-filter** method (or far more data + true-waveform
supervision), not a better classifier on whitened strain. Artifacts: `models/semicoherent_v1def.pt`
(0.706), `models/semicoherent_v2.pt` (0.691), `results/eval_semicoherent_semicoherent_v1def.json`,
`results/eval_semicoherent_semicoherent_v2.json`, histories `models/*_history.json`.

**Open threads (for whenever this is revisited):** true-waveform-supervised front end (init the
V2 templates from a real subsolar bank instead of random); time-domain ResNet on strain at much
larger data scale; or accept stage-0's verdict that the gap is a *coherence* problem and port a
classical chunked matched filter as the detector with a learned veto on top.

### Stage 1 second pass (2026-06-14): threshold-robustness check — glitch hypothesis REFUTED, diagnosis sharpened
Prompted by an outside critique (the zero-FA threshold = single highest noise score = likely ONE
glitch → maybe 0.000 is a measurement artifact). Confirmed the threshold IS a lone outlier (V2 max
5.2σ above mean, V1 13.8σ). Re-thresholded the SAME injections/noise (scripts/threshold_robust_eval.py,
identical seeds) under a strict→loose ladder:

| policy (FAR over 6.7h) | V2 high-mass frac | V1 frac (all bins) |
|---|---|---|
| max (0 FA) / drop top-1 / drop top-2 | 0.000 | 0.000 |
| p99 (1% FA) | 0.348 | 0.000 |
| p95 (5% FA) | 0.348 (mid 0.361) | 0.000 |

### Path G milestone G0 (2026-06-14): coincidence plumbing check — PLUMBING SOUND, but it pivots the plan
Built scripts/coinc_check.py: inject one event into H1+L1 (v1 search.py geometry: per-detector antenna
+ light-travel delay), recover with the bank, check timing. Fetched 8 more L1 coincident segments
(10 total: 5 overlap H1 test, 5 overlap H1 train; ~1.3 GB). Findings via a clean true-vs-bank diagnostic:
- **Plumbing is correct:** the TRUE template recovers newSNR ~28–32 with **+0.2 ms** timing in both
  detectors — geometry, L1 whitening, windowing, statistic all sound.
- **Extrinsic params are irrelevant:** exact masses + WRONG (fiducial) sky/inclination still recovers
  27.6 (== true). The quadrature MF is orientation-invariant → a bank needs to cover MASS only (and F0
  was not confounded by sky/inclination).
- **Mass spacing is the whole game (dephasing curve, injected SNR 40):** +0.01% Mc → 27.9; +0.1% →
  18.9; **+1% → 6.2 (dead); +3% → 6.7 (dead).** A 32–64-template bank has ~3–6% Mc gaps → every
  injection is dephased to the noise floor. Covering Mc∈[0.17,0.87] at 0.1% spacing needs **~1,600
  templates** (Mc alone; ×more for q/spin) → intractable locally, and the trials would inflate the
  noise floor further. This is *why* F0 was flat-zero, now quantified.
- **Consequence — the pivot:** coincidence kills the NOISE floor, not the SIGNAL-recovery (bank-density)
  problem. You cannot coincide signals you cannot recover per-detector. So **G1 must ride on the LEARNED
  model (cnn_w64, 0.41–0.48 single-detector, AUC 0.79), not the bank** — the learned model is not
  bank-density-limited (it generalizes across mass), and its single-detector limitation IS the noise
  floor, which is exactly what H1×L1 coincidence + time-slides attack. Artifacts: scripts/coinc_check.py.

### Build C (2026-06-20, GPU VM): coincidence advantage HOLDS at realistic FAR — the win is FAR-robust
Roadmap follow-up to G1: the +1.37× was at a modest FAR (~1/6 h). On an L4 VM (`fetch_coinc.py` +
`coinc_far.py`), fetched **24 fresh H1×L1 coincident O3a segments** (26.9 h coincident livetime, none in
cnn_w64 training — no leakage), scored with cnn_w64, and built a **global time-slide background**:
**N−1 = 1511 distinct circular lags × 26.9 h = 1692 days (4.6 yr)** of honest background → sets the coincident
threshold down to **1/year**. 2400 coincident injections (parallel: 1 worker/segment over 8 cores, GPU
batch-score). *(An earlier write-up said "4480 days / 12.3 yr" — that used 4000 lags, but there are only N−1
distinct circular lags; lags beyond N−1 just repeat. Corrected via the honest-slides fix; the ratios below are
unchanged, only the background-livetime label.)*

| FAR | 0.17–0.35 | 0.35–0.55 | 0.55–0.88 |
|---|---|---|---|
| 1/6 h | 0.302 | 0.350 | 0.373 |
| 1/day | 0.295 | 0.332 | 0.358 |
| 1/week | 0.284 | 0.322 | 0.350 |
| 1/month | 0.276 | 0.316 | 0.325 |
| **1/year** | 0.267 | 0.300 | 0.303 |
| single-det floor (1/27 h) | 0.222 | 0.252 | 0.251 |

- **Graceful degradation:** tightening the FAR 4 orders of magnitude (1/6 h → 1/year) costs only ~15–20%
  of sensitive distance. The advantage does NOT collapse at low FAR.
- **Coincidence reaches FAR a single detector can't:** single-det floor = 1/T_real = 1/27 h (no slides
  possible); coincidence reaches 1/year via time-slides. **At 1/year, coincidence (0.303) still beats the
  single detector's BEST-achievable (0.251) by ~1.2×.**
- **Cross-check (validation):** coinc @1/day vs single-det floor = **1.33/1.32/1.43×** — matches the
  stress-tested local G1 +1.37–1.48× almost exactly. The realistic-data result reproduces the validated one.
- **Honest caveats:** global slides assume noise stationarity across the 24 segments/days (standard); the
  single-det floor is data-limited (more single-det data → lower floor, but the point is coincidence reaches
  far lower FAR from the SAME data); network-SNR axis (same convention as G1). Artifacts: results/coinc_far.{json,png}.

### Build C-2 (2026-06-20, GPU VM): a LEARNED coincidence statistic beats the sum baseline — significant + leakage-free
G2a had found *simple* coincidence statistics (min/prod/max+min) don't beat the plain `sum` of per-detector
scores — "sum is optimal." Build C-2 revisits that with a **learned** statistic: take the cnn_w64 penultimate
**256-d embeddings** of the H1 and L1 windows, form consistency features `[eH, eL, |eH−eL|, eH·eL]`, and train
a small head (`scripts/coinc_learned.py`, `CoincHead` 4·256→128→32→1) to separate real coincident injections
from time-slid (accidental) noise pairs. Intuition: the head learns whether H1 and L1 *agree* in morphology —
a real signal correlates across detectors, a glitch-coincidence does not. Evaluated against an **honest
time-slide background** (only the N−1 distinct non-zero circular lags — see the "honest-slides" note below)
across the FAR sweep, vs the `sum` baseline on the SAME embeddings.
- **The result (sensitive-distance fraction, high-mass, held-out-segments, honest slides):** sum → learned =
  0.370→0.390 (1/6h), 0.350→0.384 (1/day), 0.331→0.374 (1/week), 0.320→0.371 (1/month) — learned wins at
  **every honestly-supported FAR, all 3 mass bins**, the gain *growing* at stricter FAR. (Held-out-segments has
  only 504 eval-noise windows → honest background 0.51 yr → the sweep stops at 1/month; for the 1/year result see
  the lower-FAR row below.)
- **Stress-test 1 — LEAKAGE (the decisive one).** The head's training negatives are noise; the eval background is
  also noise → risk of memorizing noise realizations (the δ-stacking trap). Ran THREE modes: `leaky` (shared
  noise), `--holdout-noise` (head-neg and eval-bg are disjoint noise halves), and the gold-standard
  `--holdout-segments` (train on 16 segments, eval on **8 entirely unseen** segments — no noise *or* injection
  overlap). The gain is **stable across all three** (1/month high-mass: 0.369 / 0.369 / 0.371) ⇒ NOT memorization;
  it survives the strongest leakage test we can run.
- **Stress-test 2 — SIGNIFICANCE.** Bootstrap (B=500) over the 2000 held-out-segment eval injections, 90% CI on
  (learned − sum): the **high-mass gain (the headline) excludes zero at every honest FAR** (1/month Δ=+0.050
  [+0.024,+0.081], P=1.00), and mid-mass is significant too (1/month Δ=+0.018). The **light bin (0.17–0.35) is the
  weakest** — its gain is real by 1/month (Δ=+0.008) but only *marginal* at the loosest FAR (1/day 90% CI grazes
  zero, ≈[−0.001,…]). Honest: the advantage is strongest at high mass and tapers toward the light end.
- **Stress-test 3 — TRAINING STOCHASTICITY.** The bootstrap covers injection-sampling noise but not a *lucky
  head initialization*. Re-trained the head with 5 independent seeds (`--head-seed 0–4`; the split stays fixed,
  only init + negative-pair sampling + batch order vary): learned beats sum at **every seed, every mass bin, and
  every honestly-supported FAR** (≈±0.02 magnitude spread, sign never flips). Not a lucky init.
- **Lower FAR (the "1/year" result, honest + leakage-clean).** Held-out-segments runs out of background at 1/month;
  to reach 1/year *cleanly*, the `--holdout-noise` mode keeps the leakage-clean property (head never sees the
  eval-background noise) but pools 756 eval-noise windows → honest background **1.16 yr**, which supports 1/year.
  There learned still beats sum at **5/5 FARs incl. 1/year**: 1/year high-mass Δ=+0.048 [+0.030,+0.071],
  P(learned>sum)=1.00 (1/year is ~1 background event → thin but supported). The full `leaky` background (1512
  windows → 4.6 yr) reaches 1/year *robustly* and agrees: 1/year high-mass Δ=+0.032 [+0.018,+0.053] (leakage
  shown negligible above). **So the learned advantage holds, significantly, down to 1/year.**
- **honest-slides note (a rigor fix found while pushing to lower FAR).** The background was built as
  `sH+roll(sL,k)` for k=1..slides. With only N noise windows there are just **N−1 distinct circular lags**;
  slides>N−1 *repeats* lags (period N) and re-injects the zero-lag/on-source at k=N,2N,… → it **overcounts T_bg**
  (≈5–8× for the small held-out backgrounds) and inflates the reachable FAR. Fixed in `coinc_learned.py` and
  `coinc_far.py` (cap at N−1 distinct lags; the FAR sweep auto-drops any FAR with <1 expected background event).
  All numbers above are post-fix. The learned>sum conclusion is unchanged — only the optimistic FAR *labels* were.
- **Bottom line:** the learned H1×L1 consistency statistic adds a **significant +0.02–0.05 sensitive-distance
  fraction (≈+5–15%, growing with mass and with FAR strictness) on top of the `sum` coincidence** — which itself
  is +1.37× over single-detector (G1/Build C). First thing to *beat* sum for subsolar coincidence, leakage-free,
  significant, and stable to 1/year. **Honest caveats:** per-detector model is cnn_w64 (H1-trained, applied to both
  detectors); 1/year rests on a thin (clean) or leakage-caveated (robust) background; head trained at this data
  scale (more coincident data → 1/decade). Gated in verify.sh (cross-segment + bootstrap CI>0, honest FAR ≤1/month).
  Artifacts: results/coinc_learned_segments.json (+ _holdout for the clean 1/year, + leaky).
- **Follow-up — does a better base embedder COMPOUND? (honest no.)** G2b found the higher-AUC, H1+L1-trained
  `cnn_hl` (0.804 vs cnn_w64 0.793) did NOT help the `sum` statistic — the operating point is tail-separation,
  not AUC. Re-ran the learned coincidence on `cnn_hl` embeddings (`--weights cnn_hl`; first verified cnn_hl's
  training GPS times are **disjoint** from all 24 Build-C segments → leakage-free). Two findings: **(i)** the
  learned statistic helps on cnn_hl too — significant at 3/4 honest FARs (1/month high-mass Δ=+0.030
  [+0.013,+0.056]; 1/week marginal P=0.92), so the mechanism is *base-model-agnostic*; **(ii)** but it does NOT
  compound — learned-on-cnn_hl high-mass = 0.386/0.381/0.362 (1/6h/1/day/1/month) vs learned-on-cnn_w64
  0.390/0.384/0.371, i.e. **within the ±0.02 head-seed spread**. The higher-AUC base buys no clear extra distance
  — G2b's logic extends to the learned statistic. **Takeaway: the win is base-model-robust; the simpler,
  gate-critical cnn_w64 is sufficient — no need to ship cnn_hl.** Artifact: results/coinc_learned_segments_cnn_hl.json.

### Path G milestone G1 (2026-06-14): H1×L1 COINCIDENCE WORKS — first positive result (+1.3–1.5× distance)
After G0 forced the pivot (bank density-limited → ride coincidence on the LEARNED model), built
scripts/coinc_eval.py: cnn_w64 per-detector on 64-s windows, H1×L1 coincidence with a **time-slide
background** (pair H1 window i with L1 window i+lag, lag≠0 → 18,910 accidental coincidences over ~305
livetimes from just 5 coincident test segments). Coincident statistic = sH1+sL1; injected 1,500
coincident signals (proper antenna + light-travel delay) at network SNR 4–40. Compared single-H1 vs
coincidence **at matched false-alarm rate** (the fair comparison), network-SNR axis:

| mass bin | single-det SNR50 (frac) | coinc SNR50 (frac) | **distance gain** |
|---|---|---|---|
| 0.17–0.35 | 30.7 (0.261) | 23.2 (0.345) | **1.32×** |
| 0.35–0.55 | 27.3 (0.293) | 20.9 (0.382) | **1.30×** |
| 0.55–0.88 | 27.7 (0.289) | 18.7 (0.428) | **1.48×** |

**Requiring two-detector agreement improves sensitive distance ~1.3–1.5× over the single-detector ML
search (best 1.48× high-mass) → ~2.3–3.3× sensitive VOLUME.** First positive in the whole gap-closing
arc. Cross-check that validates the pipeline: single-det SNR50 27–31 (network) ÷√2 ≈ 19–22 per-detector,
matching v1's published per-detector SNR50 ~18.6. The lever (coincidence kills the noise floor) is real
and works for the learned detector — exactly G0's prediction.
**Honest caveats:** (1) coincidence is COARSE — window-level sum of logits, no matched-filter timing/phase
consistency (a finer coincidence should gain more → G2); (2) matched-FAR threshold is at ~1 FA/livetime
(~5.7 h), not a realistic 1/month — pushing lower needs more L1 data; (3) H1-trained model applied to L1
(transfer, not L1-optimized → H1+L1 training is upside); (4) network-SNR axis, internal single-vs-coinc
comparison (not directly the per-detector 0.41–0.48). Artifacts: scripts/coinc_eval.py,
results/coinc_eval.{json,png}, results/coinc_inj.parquet (raw scores → free re-binning).
**G2a follow-up (coinc_stat.py): the `sum` statistic is already (near-)optimal.** Tested sum / min /
prod-prob / max+min on the saved scores at matched FAR: sum 1.37× (mean), max+min 1.34×, prod-prob
1.24×, min 1.11×. `min` is worst (too strict — penalizes the antenna-weaker detector). ⇒ no free lift
from the combination rule. **Note on "finer (timing) coincidence":** true ~10 ms timing/phase
coincidence needs matched-filter arrival times → a dense bank → the SAME intractable wall as G0/F0, so
it is blocked for us.

**G2b (2026-06-15): H1+L1 training — NO improvement (clean negative).** Built a 64-s H1+L1 spectrogram
dataset (build_hl.py, self-contained + resumable; 16 H1-train + 5 L1-train-time segments, val on H1 val;
NO leakage — eval uses the 5 L1-test-time segments) → trained `cnn_hl` (same recipe as cnn_w64) → re-ran
G1 coincidence (coinc_eval.py --weights cnn_hl). Result: cnn_hl val AUC **0.804** (> cnn_w64's 0.793), but
coincidence sensitive distance **0.345 / 0.375 / 0.420 ≈ cnn_w64's 0.345 / 0.382 / 0.428** (flat, even
fractionally lower). ⇒ a higher global AUC did NOT translate to better coincidence: the operating-point
performance is set by **tail separation**, not AUC, and the H1→L1 transfer was already adequate so training
on L1 didn't sharpen the relevant tail. Artifacts: models/cnn_hl.pt, results/coinc_eval_cnn_hl.{json,png}.

**Path G CONCLUSION — every tractable lever squeezed; +1.37× is the honest ceiling.** Coincidence is the
win (+1.3–1.5× distance, ~2.3–3.3× volume over single-detector ML). Better statistic: no gain (sum optimal).
H1+L1 training: no gain (AUC↑ but coincidence flat). Finer timing coincidence: blocked by the bank-density
wall. Remaining is robustness only (lower FAR needs more coincident data). The arc's honest headline: a
single-detector learned subsolar search is noise-floor-limited; two-detector coincidence recovers ~1.4×
sensitive distance, and that is the ceiling for the learned approach at this data/compute scale.

### Path F milestone F0 (2026-06-14): bank-mismatch gate — NOT CLEARED, two-sided squeeze (clean negative)
Replaced the oracle's true template with a coarse equal-mass bank (mass-only grid, B up to 64),
same injections/seeds/n=8-vetoed statistic (scripts/bank_oracle.py). Result: **0.000 at every bank
size 3→64**, dead flat. Diagnosed against the true-template oracle (same injections):
- **Mismatch:** bank-max newSNR median **6.54** vs true-template **10.31** → the coarse bank recovers
  only **~69%** of the SNR (subsolar dephasing, as predicted by v1's −28%/0.01%-Mc).
- **Trials:** the bank's zero-FA noise floor is **~10.25 newSNR** — and it is NOT a lone glitch this
  time (top noise maxima 10.25/10.11/10.0/9.97/9.89…, a smooth populated tail across all 6 segments;
  median 7.62). 64 templates × a 4000-s scan give noise many chances to fake a chirp; the χ² veto
  isn't strong enough to push it below ~10.
- **The squeeze:** signal sits at ~6.5, noise floor at ~10 → no separating threshold exists, even at
  the median-segment threshold. Densifying can't escape it: more templates slightly raise recovery
  but raise the noise floor in lockstep (hence the flat-zero curve). A learned veto (F1) can only push
  noise *down*, but signal (6.5) is already *below* noise (10) in this statistic → F1 can't rescue it.
**Verdict:** the naive single-detector semi-coherent bank is mismatch+trials limited. The field reaches
~70% with what we are NOT using: a dense *coherent* bank (recovery >0.97, but huge → enormous trials)
AND **multi-detector coincidence** (kills the noise floor — random noise rarely coincides across H1×L1
with consistent timing). **Root cause across ALL our negatives (learned classifier, learned semi-coherent,
coarse bank) is the single-detector noise floor.** Artifacts: results/bank_oracle_B64.{json,png},
results/bank/. We have only 2 L1-coincident segments (thin) — a real coincidence study needs more L1 data.

---
### Stage 1 second pass threshold-robustness detail
**Dropping the top 1–2 outliers changes nothing** → the 0.000 is NOT a single-glitch artifact; the
original negative stands and is now more robust. **But the check sharpened the picture:** (1) V2 is
*not* null — at a loose 1–5% false-alarm rate it reaches ~0.35 sensitive distance in the mid/high
mass bins, i.e. it carries real signal information that only collapses at the strict tail; (2) V1 is
genuinely null at every operating point (consistent with its overfit/unstable training); (3) what
kills V2 is not one glitch but a **fat non-Gaussian noise tail** (~1–5% of windows score as high as
moderate signals) — exactly what a consistency veto (the oracle had one) or a sharper matched-filter
statistic would suppress. ⇒ the fix must **sharpen the signal/noise statistic or veto the tail**, not
tweak the threshold or the architecture. Points squarely at dense oracle supervision and/or a real-MF
front end. Artifacts: results/threshold_robust_semicoherent_{v2,v1def}.json.

### N4 (2026-06-26): self-supervised backbone — a clean data-wall win
The subsolar detector is labeled-data-limited; unlabeled real-noise spectrograms are abundant. `ssl_pretrain.py`
pretrains the SpectrogramCNN conv backbone (the exact 4 `_block`s, so weights transfer) as a **masked-spectrogram
autoencoder** on **20k UNLABELED noise spectrograms** — mask random t-f patches, reconstruct them. It learns
real structure: masked-recon MSE drops 1.05→0.75 over 30 epochs (the predictable part — PSD shape, lines,
glitches; the rest is irreducible noise).
- **The data-wall test** (`ssl_finetune.py`, input standardized to the SSL mu/sd for both models, 3 seeds): fine-tune
  the pretrained backbone vs from-scratch at a reduced labeled budget. **SSL wins at every budget, gain ∝ 1/labels:**
  +0.124 val-AUC @1000 labels (0.539±0.006 → 0.663±0.013, ~10× the seed scatter), +0.021 @4000. The gap shrinks as
  labels grow — exactly the data-wall signature (self-supervision matters most when labels are scarce).
- **Honest caveats:** (1) the unlabeled pool is the labeled set's own 20k noise windows — pretraining on *more* O3
  noise (a VM extension) should help further; (2) metric is val AUC, not yet the headline sensitive distance
  (8/SNR50) — the injection-recovery impact is the next step; (3) SSL **mitigates** the wall (scarce-label AUC 0.66
  is still below the full-data 0.79), it doesn't erase it. Gated. Artifacts: results/ssl_finetune.json,
  models/ssl_encoder.pt.

#### N4 sensitive-distance follow-up (2026-06-26): the AUC win translates to DISTANCE — at a defined FAR
val AUC ≠ sensitive distance (the headline metric, set by the score tails at the zero-FA threshold). `ssl_sensdist.py`
reads the efficiency-vs-SNR curve straight from the val shards (each injection's `in_window_snr` + `chirp_mass`),
at two operating points, for SSL-pretrained vs from-scratch at reduced budgets (2 seeds):
- **At the strict zero-FA threshold: 0 for BOTH at every budget (2k/4k/8k).** Reduced-budget models (AUC ≤0.74,
  SSL or not) can't reach 50% efficiency at that threshold → SNR50 undefined → distance 0. A **model-strength
  floor**: the zero-FA distance needs near-full-data strength (the full cnn_w64 reaches 0.41–0.45 at AUC 0.79),
  which the SSL gain alone doesn't bridge — *not* an SSL failure.
- **At a softer (1%) FAR the SSL win DOES translate to sensitive distance, with the same data-wall signature:**
  Δ(ssl−scratch) mean distance-fraction = **+0.278 @2000** (scratch non-functional at 0.000, SSL 0.278!),
  +0.184 @4000, +0.013 @8000 — a large gain at scarce labels shrinking as labels grow. So the AUC win is a real
  *detection* improvement, not an AUC artifact: SSL makes a reduced-budget model functional where from-scratch is
  not. Gated. Artifact: results/ssl_sensdist.json.

**Net N4 (honest):** self-supervised pretraining on unlabeled noise is a genuine data-wall win — it improves both
val AUC and sensitive distance at scarce labels (most at the scarcest), making sub-functional from-scratch models
functional. The strict zero-FA headline number specifically requires near-full-data model strength, so the win
shows up at a defined FAR rather than at zero-FA in this reduced-budget study; scaling the unlabeled pool (more O3
noise, a VM extension) and budget toward full data is the path to a zero-FA distance gain.

### N5 — triple-detector H1×L1×V1: Virgo does NOT help the subsolar coincidence search (2026-06-27)
Extended the G1/Build-C H1×L1 double-coincidence (+1.37×) to a 3rd detector. `coinc_triple.py`: cnn_w64
(H1-trained) scores 64-s windows in H1, L1 AND V1; triple statistic = sH1+sL1+sV1; 3-way time-slide
background (random (lagL,lagV) pairs, 8000 livetimes) → matched-FAR threshold; injections projected onto
all 3 detectors (pycbc antenna + delay) at network SNR √(snrH²+snrL²+snrV²). Ran on 4 fresh, leakage-free
H1∩L1∩V1 triple-coincident O3a segments (the original H1∩L1 test segments are ALL Virgo duty-cycle gaps —
discovered 20 true triple segments by intersecting the 3 DATA flags; GWOSC was degraded so a persistent
checkpointing fetcher accumulated the data over ~12 h).

Sensitive-distance fraction (8/SNR50, matched FAR), 2400 injections:

| mass bin | single | double (H1×L1) | triple (H1×L1×V1) | triple/double |
|---|---|---|---|---|
| 0.17–0.35 | 0.241 | 0.303 | 0.294 | 0.97× |
| 0.35–0.55 | 0.268 | 0.355 | 0.337 | 0.95× |
| 0.55–0.88 | 0.276 | 0.390 | 0.355 | 0.91× |
| **mean** | 0.261 | **0.349** | 0.329 | **0.94×** |

**Two findings:**
1. **Double H1×L1 reproduces the win on fresh data: 1.33× over single-det** (0.349/0.261) — independently
   validates the G1/Build-C +1.37× coincidence advantage on segments never used before.
2. **Adding Virgo does NOT help — it marginally HURTS (0.94×).** Honest negative, as the PLAN anticipated
   ("Virgo less sensitive"). **Mechanism (diagnostic):** per-detector signal responsiveness (mean score on
   loud netSNR>25 minus faint <10 injections) = **H1 +5.1, L1 +7.4, V1 +1.2** — V1 responds at only **~19%**
   of H1/L1. Virgo is too insensitive at subsolar masses to carry signal, so summing its near-noise score and
   requiring 3-way agreement (which raises the matched-FAR threshold via the larger 8000-livetime background)
   slightly degrades the statistic rather than improving it.

**This also rules out the learned-triple extension:** a learned consistency statistic (Build C-2 style) could
at best learn to *ignore* V1 — there is essentially no V1 subsolar signal to weight — so it would recover ≈ double,
not beat it. **Conclusion: the H1×L1 double-coincidence remains the ceiling for learned subsolar PBH search;
Virgo adds no sensitive distance at these masses.** Caveats: cnn_w64 is H1-trained applied to V1 (transfer) — a
V1-specific model might extract marginally more, but V1's fundamental insensitivity (higher PSD; the +1.2 response
even to LOUD signals) caps the upside; 4 segments (the double-vs-single cross-check validates the pipeline).
Gated. Engineering note: the eval is checkpointed per-segment (`coinc_triple_rows.parquet`) — it survived
repeated power losses + Anthropic service interruptions, resuming from the last finished segment each time.

### Follow-up A — the REAL matched-filter benchmark: CNN ties a realizable dense bank (2026-07-03)
The question the whole v2 arc bumped into — "how good is our ML detector vs a REAL matched filter?" — answered
on the Mac (GPU VM unavailable), made tractable by three facts: 64-s templates (the CNN window + the eval's
in-window-SNR convention), an effectively 1-D Mc bank (extrinsics proven irrelevant by coinc_check), and MPS-
batched FD correlation (`pbh/bankmf.py`, golden-tested against first principles — `bank_golden.py`: noiseless
self-correlation exact, phase-invariant, MPS≡CPU 9e-7).

**Two walls, one tractable path (A1/A2/A2b):**
- **Fully-coherent MF is intractable** — the coherent fitting factor collapses (FF 0.576 median at 0.3% spacing;
  templates 25% apart in Mc are near-orthogonal). The 64-s subsolar chirp carries ~1e4 cycles → a full-coherence
  bank is megatemplate-scale, consistent with LVK's real O4 subsolar bank of **3,452,006 templates**
  (arXiv:2412.10951). The parked "~1,650 templates" estimate was ~3 orders optimistic for full coherence.
- **The n=8 SEMI-coherent statistic is the tractable detector** (8-s chunk coherence → ~8× dephasing tolerance +
  chunk-|·| removes inter-chunk phase). `bank_semiff.py` measured its recovery vs spacing: 0.25%→0.858, 0.5%→0.566,
  1%→0.461, 2%→0.366 — a steep curve that **quantitatively explains bank_oracle's old 0.000** (its ~2.5%/64-template
  density gives recovery ~0.3 → under threshold), and sets the real requirement at ~0.1% (~1,619 templates, tractable).

**The benchmark (A3/A5) — `bank_dense.py` runs the semicoherent_oracle statistic at 0.1% density on the 6 real
test segments (template-major, per-segment + mid-segment atomic checkpointing — survived 2 power losses + a Claude
restart); `bank_vs_cnn.py` scores cnn_w64 on the IDENTICAL deterministic injections:**

| detector | 0.17–0.35 | 0.35–0.55 | 0.55–0.88 | mean | notes |
|---|---|---|---|---|---|
| coarse bank (83 tmpl, ~2.5%) | 0.000 | 0.000 | 0.000 | 0.000 | reproduces bank_oracle's parked 0.000 exactly |
| **real semi-coherent bank MF (0.1%, 1619 tmpl)** | 0.472 | 0.509 | 0.485 | **0.489** | the realizable detector |
| cnn_w64 (learned, SAME injections) | 0.436 | 0.505 | 0.477 | **0.472** | 1 forward pass |
| semi-coherent ORACLE (true template) | 0.663 | 0.764 | 0.752 | 0.720 | unrealizable ceiling |

**Headline (honest): the realizable dense-bank matched filter and the learned CNN are a statistical TIE**
(0.489 vs 0.472, **1.03×**; per-bin 1.08/1.01/1.02× — only the lowest-mass bin hints at an MF edge, within Monte
Carlo). A single CNN forward pass matches a 1,619-template matched-filter bank at ~1/1600 the compute. **The
dominant loss is template-bank MISMATCH, not learned-vs-MF**: both sit far below the true-template oracle (0.72).
And the **density sweep is the wall made quantitative** — 83→0.000, 326→0.12, 649→0.29, 1619→0.49: you need the
full ~1,600-template bank to reach the tie, validating A2b's tolerance curve end-to-end.

**North-star note:** the v1 gated CNN number (0.45) vs a different injection realization suggested a ~10% MF win;
the airtight co-injection (identical injections) shrank it to a ~3% tie — the co-injection prevented an overclaim.
Caveats: single-detector H1 (coincidence is the separate +1.37× axis); n=8 semi-coherent (not full-coherent, which
is intractable); 64-s window convention; the newSNR chi²-veto threshold is zero-FA over 6 segments. Gated.
Artifacts: bank_golden.json, bank_semiff.json, bank_dense.json, bank_vs_cnn.json; pbh/bankmf.py.

### N5 O4b RE-TEST (2026-08-06): the Virgo negative REPLICATES across detector generations
N5 (O3a, 2019) found Virgo does **not** help subsolar triple-coincidence. The obvious objection: that was
*O3a-era* Virgo. With O4b now public (Apr 2024–Jan 2025, H1+L1+V1) the objection is testable — and the answer
is a clean replication on **twice the segments, 5.5 years later, on a more sensitive network**.

**Prerequisite (`o4_transfer_scout.py`):** the O3a-trained `cnn_w64` transfers to O4b noise **unchanged
(0.97× sensitive distance)** — per-segment PSD whitening absorbs the era shift — so the comparison is not
confounded by model changes. O4b is **1.41× more sensitive** in-band, and its zero-FA threshold is much lower
(2.111 → 1.141; cleaner noise). Same statistic, same thresholds convention, same injection population; only
the era changes (`coinc_triple.py --segs o4b_triple_segs.json --tag _o4b`; the O3a artifact stays gated).

| | O3a (2019, 4 segs) | O4b (2024–25, 8 segs) |
|---|---|---|
| single → double → triple | 0.261 → 0.349 → 0.329 | 0.267 → 0.346 → 0.329 |
| **double / single** | **1.33×** | **1.30×** |
| **triple / double** | **0.94×** | **0.95×** |
| V1 signal responsiveness | +1.16 vs H1 +5.11 / L1 +7.37 = **19%** | +0.74 vs H1 +6.20 / L1 +5.75 = **12%** |

**Both headline ratios replicate to within 3%.** H1×L1 coincidence still buys ~1.3× sensitive distance;
adding Virgo still costs ~5%.

**The mechanism, now MEASURED rather than asserted (`o4_asd_compare.py`)** — median ASD in the subsolar band
[50, 300] Hz from our own cached strain:

| era | H1 | L1 | V1 | V1 / best LIGO |
|---|---|---|---|---|
| O3a | 5.18e-24 | 4.68e-24 | 1.33e-23 | **2.8×** louder |
| O4b | 3.63e-24 | 3.68e-24 | 1.17e-23 | **3.2×** louder |

**Virgo did improve (1.14×) — but LIGO improved faster (1.29×), so the gap WIDENED from 2.8× to 3.2×.**
Injections are scaled to a fixed *network* SNR and each detector's share is set by its own PSD, so a detector
3× louder contributes ~1/3 the amplitude SNR. That is exactly why V1's responsiveness fell 19% → 12% between
the two runs, and why summing its near-noise score plus paying the higher 3-way threshold still degrades the
statistic. **The negative is not an artifact of old Virgo data — it is structural at subsolar masses, and it
got slightly worse, not better.**

Bycatch: O4b has **177 clean H1∩L1∩V1 4096-s windows in a 30-day probe** vs 20 in six months of O3a (Virgo's
duty cycle is transformed) — and all 8 fetched segments were usable, vs 4–5 in O3a. So the re-test had strictly
better data in every respect except the one that matters. Gated (double reproduces + triple still no help +
cross-era replication within 5% + measured ASD gap). Artifacts: coinc_triple_o4b.json, o4_asd_compare.json,
o4_transfer_scout.json.

#### O4-3 STRESS-TEST (2026-08-07): the reach gain survives; the mass-dependence does not
Review pass over the O4-3 campaign. It carries a strong independent cross-check — **the measured 1.24× distance
gain matches the separately-measured 1.29× ASD improvement** (`o4_asd_compare.json`), two independent routes
agreeing. Three issues found and addressed:
- **(a) The eras were not FAR-matched.** O3a used 5 segments (5.7 h), O4b 8 (9.1 h); the zero-FA threshold is the
  max over noise windows, so it grows with livetime (thr_single O3a 2.111 vs O4b **3.233**, while the 3-segment
  scout gave O4b 1.141). O4b was held to a *stricter* bar — biasing **against** the claim. Re-ran at equal
  livetime (`--match-segs`, 5 each): **1.21 / 1.28 / 1.20×, mean 1.23× vs 1.24×.** Real concern, **immaterial
  effect** (~1%, within MC noise).
- **(b) No uncertainty.** `o4_reach_bootstrap.py` (B=1000, resamples the saved injection rows — no re-run):

  | mass bin | reach gain | 90% CI | significant? |
  |---|---|---|---|
  | 0.17–0.35 | 1.21× | [1.14, 1.29] | ✅ |
  | 0.35–0.55 | 1.28× | [1.21, 1.35] | ✅ |
  | 0.55–0.88 | 1.20× | [1.11, 1.30] | ✅ |

  Every bin's CI excludes 1 ⇒ **the gain is real everywhere**. But **all three CIs mutually overlap ⇒ the apparent
  mass-dependence (1.20 vs 1.29) is NOISE** — that claim is withdrawn.
- **(c) Volume uses (mean SNR_ref)³, not mean(SNR_ref³).** By Jensen's inequality this *underestimates* absolute
  volumes (isotropic sky/orientation ⇒ wide SNR_ref spread within a bin). The ratio largely cancels it, so the
  ~1.9× volume gain stands, but **absolute Mpc³ figures are lower bounds, not survey volumes.**

**Honest headline: O4b expands subsolar search reach by 1.23× in distance [1.11–1.35] and ~1.9× in surveyed
volume over O3a — significant in every mass bin, with no evidence of mass dependence.** Engineering: the campaign
now checkpoints per segment (a power loss destroyed a full 40-min all-or-nothing run) and persists injection rows
to parquet so re-analysis needs no re-run. Gated (42). Artifacts: o4_sensitive_distance_matched.json,
o4_reach_bootstrap_matched.json.

### Deep FAR: an 80-year background on O4b — 1/decade reached, zero-lag clean (2026-08-08)
Build C reached a 1/year false-alarm rate on 24 O3a segments (4.6 yr background). To support a *detection*
claim you need far more: the bar is "noise fakes this only once per N years", and N must be large. This is
pure compute, not a wall — global time-slides manufacture background livetime.

**Method (`far_deep.py`).** Shift L1's whole window-score stream against H1's; every surviving coincidence is
accidental by construction. With N_tot windows there are **N_tot − 1 distinct non-zero circular lags** (the
honest count — more repeats lags and re-injects zero-lag, the overcounting bug we fixed in Build C-2), each a
full copy of the livetime (**but NOT an independent sample — see the validation section below, which measures
exactly how far from independent they are**):

    background_time = (N_tot − 1) × total_livetime      [reproduces Build C's 1692 days exactly — verified first]

Both factors grow with segment count, so **background ∝ N_segments²**. 100 fresh O4b H1∩L1 segments
(leakage-free: cnn_w64 is O3a-trained) → 6,200 windows, 113.8 h real livetime, **6,199 lags → 80.5 years**
(**17× Build C**).

| FAR | threshold a signal must exceed |
|---|---|
| 1/month | 12.340 |
| 1/year | 14.112 |
| **1/decade** | **16.121** |
| 1/century | not measurable (0.01 × 80.5 yr < 1 expected event) |

**Zero-lag check (the real, unshifted H1×L1 data): loudest coincidence = 11.295 — below even the 1/month
threshold. A clean null: no subsolar candidate in 114 h of O4b, as expected for randomly-chosen noise stretches.**
The deepest *measurable* FAR here is ~1/80 yr; we quote 1/decade as the conservative ladder rung.
*(The follow-up audit below shows this number is not a near-miss at all: 11.295 is a one-sided H1 glitch, and
the null is far wider than the 8% margin suggests.)*

**Engineering (this ran unattended for ~4 h across a session restart).** The expensive step (fetch → whiten →
score) is checkpointed per segment as a tiny .npz of window scores, so a completed segment is never re-fetched;
writes are atomic; re-running resumes and extends. Raw strain is purged after scoring — **and the purge had to
clear astropy's download cache too**: `gwpy`'s `fetch_open_data(cache=True)` keeps a *second* copy (~0.25 GB per
segment, 5.3 GB accumulated) that our first purge missed and which would have exhausted disk near segment 80.
Caught mid-run, fixed, and disk then held flat at 19 GB for the remaining ~60 segments. Gated (45).
Artifact: far_deep.json; score cache: results/far_scores/ (100 segments, 160 KB — the reusable product).

### Deep-FAR AUDIT: the 80-year background stress-tested — thresholds are ±33–44%, but the null is 4/4 (2026-08-09)
A pass over the retained score cache asking "did we miss anything?" turned into a full audit of the deep-FAR
result. It found a real over-claim in our own reporting *and* a null considerably stronger than we had stated.
**Every test below can only weaken the headline — that is the point of running them.**

**Assumptions that HOLD.** `far_background_validation.py`:
- **Detector independence (V1).** Time-slides are only valid if H1 and L1 noise are independent; shared
  environmental noise would make the background understate the true accidental rate and thresholds too *low*.
  Zero-lag corr(sH, sL) = −0.0051 against a null of 0.0000 ± 0.0133 over 6,199 lags: **z = −0.38, p = 0.69.**
- **Shared data quality (V6).** Per-segment mean-score correlation H1 vs L1 = −0.075, permutation p = 0.45.
  No co-varying data quality, which is the physical mechanism that would have broken V1.

**What BROKE — "80.5 years" is livetime, not 80.5 years of independent statistics.** All N(N−1) slid pairs are
built from only 2N underlying window scores, and the extreme tail is dominated by the few loudest windows:

| rung | bg events setting it | distinct H1 windows behind them |
|---|---|---|
| 1/month | 965 | **4** |
| 1/year | 80 | **2** |
| 1/decade | 8 | **2** |

So the 1/decade threshold rests on 2 Hanford glitches slid against everything, not on 80 years of statistics.
The consequences, all measured:
- **Jackknife (V7), dropping each 10% block:** 1/month spread **44%**, 1/year **38%**, 1/decade **33%** of the
  quoted value — versus the ±sqrt(k) Poisson band of ~±2%, which was **an order of magnitude too optimistic.**
- **It is ONE segment.** 6 of the 8 loudest H1 windows in all 114 h live in segment 59 (gps 1397232640).
  Dropping it alone moves 1/decade **16.121 → 11.261**. Reported as a diagnostic, *not adopted* — post-hoc
  removal of the loudest segment would be tuning.
- **Not detector drift (V5 explained).** First/second-half thresholds differ 32–46%, but bulk noise is
  identical between halves (median −0.814 vs −0.769; p99 1.59 vs 2.48). The "non-stationarity" is entirely
  the placement of one glitch cluster, i.e. the *same* finding as V2, not a second problem.
- **Convergence (V4)** shows the same fingerprint: the ladder jumps at n=40→60 as segment 59 enters.

**The zero-lag "near miss" was never a coincidence.** `far_glitch_anatomy.py`: the loudest zero-lag event
(11.295, reported as 8% below the 1/month threshold) is **H1 +12.53, L1 −1.24** — precisely the single loudest
H1 window in the dataset with Livingston seeing nothing. Single-detector ceilings are max(H1) 12.53,
max(L1) 6.26, so **any background above ~12.5 physically requires both detectors**: the 1/year (14.11) and
1/decade (16.12) rungs are set by genuine two-sided accidental coincidences, while 1/month (12.34) is
glitch-reachable. One-sidedness must be read against loudness, not quoted as a single number:

| rank band | sum range | % one-sided |
|---|---|---|
| top 0–25 | 15.1–18.8 | **0%** |
| top 25–100 | 13.9–15.0 | 51% |
| top 100–500 | 12.7–13.8 | 90% |
| top 2000+ | 10.5–11.9 | 100% |

*(A first quick pass reported "96% of the loudest background is one-sided" — wrong, from sampling only every
37th lag, which makes the "top 100" a MID-tail population. Corrected before it reached any claim.)*

**Would a consistency statistic do better? Measured, and NO — an honest negative.** `far_min_vs_sum.py`.
This was previously recorded as untestable without the purged O4b strain; it is not — `o4_sensitive_distance_
rows_matched` and `coinc_triple_rows_o4b` already store **per-detector** scores for 4,800 O4b injections, so
both sides of the trade are measurable from retained artifacts. At **matched FAR**:

| statistic | bg half-to-half spread | sensitive distance vs sum (1/mo, 1/yr, 1/dec) |
|---|---|---|
| `sum` (incumbent) | 46% | 1.00× |
| `min` (consistency) | **25%** | 0.99× / 0.99× / **0.97×** |
| `veto` (reject if one det < 15% of other) | 40% | 1.04× / 1.00× / 0.99× |

`min` stabilises the background but **does not buy reach**. This extends G2a's "sum is optimal" — previously
established only at a 4.6-yr background — into the deep-FAR regime, where we had explicitly flagged it as
untested. **Keep `sum`.**

**And the result that survives all of it (V8).** Every test above attacks the *threshold*; what matters is the
*search*. Comparing each configuration's zero-lag maximum against the threshold derived from that same
configuration:

| config | statistic | 1/decade thr | zero-lag max | verdict |
|---|---|---|---|---|
| all 100 segments | sum | 16.121 | 11.295 | NULL |
| all 100 segments | min | 5.978 | 1.635 | NULL (3.7× below) |
| drop glitchiest seg 59 | sum | 11.261 | 5.546 | NULL (2.0× below) |
| drop glitchiest seg 59 | min | 5.201 | 1.635 | NULL (3.2× below) |

**Null in 4/4 configurations.** Removing the glitchy segment removes the loudest zero-lag event *and* the
background that nearly matched it — self-consistently, because they are the same glitches.

**Net correction to the record.** The 1/decade *reach* stands, and the *null is stronger* than we reported
(the "8% margin" was a glitch; the honest margin is 2–3.7×). What must change is the **precision**: quote
1/decade as **16.1 ± ~5 (33%)**, dominated by one glitchy segment, not ±0.3. The deep ladder is limited by the
number of independent loud-noise samples, not by livetime — which is why real LVK searches use signal-
consistency vetoes and data-quality flags rather than raw time-slides alone. Gated.
Artifacts: far_background_validation.json, far_glitch_anatomy.json, far_min_vs_sum.json.

**Related work (added 2026-08-15).** This is not an idiosyncrasy of our pipeline — it is a named open problem.
[arXiv:2509.05283](https://arxiv.org/abs/2509.05283), *Robustness of Sensitivity Evaluations for Gravitational
Wave Detection Algorithms*, runs the AresGW ML pipeline across multiple month-long real-noise datasets and
reports *"notable performance variations, highlighting the challenges introduced by finite-duration datasets"*
specifically **at low false alarm rates**, calling for *"more rigorous statistical validation"* and better
GW-specific benchmarking. Our audit is a concrete, mechanism-level instance of exactly that: we can point to
*which* 2 windows in *which* segment set the rung, and quantify the resulting instability by jackknife.
Supporting context: blip glitches (which resemble the final cycles of a CBC) occur roughly every 30 min at
LHO, and time-slide significance is long known to be affected by correlated background triggers,
non-stationary noise and finite sample size ([arXiv:1601.00130](https://arxiv.org/pdf/1601.00130)).
See [RELATED_WORK.md](../RELATED_WORK.md) for the full sweep and the L1–L7 long-horizon list.

### L1 ratio-filter dechirping: the FOUNDATION validated — the dense-bank wall is not a wall (2026-08-15)
Follow-up A left the subsolar arc at a wall it had measured precisely: **template-bank mismatch is the
dominant loss** (realizable 1,619-template bank 0.489 vs CNN 0.472 vs true-template oracle 0.720), and 1,619
was our ceiling because `bank_dense.py` could not hold the bank — its own header says *"B=1617 cannot hold all
analytic chunks in RAM (33 MB/template → 53 GB)"*. The 2026-08-15 literature sweep found the field's answer is
not a smaller bank but **cheaper templates**, so ROADMAP's "intractable locally" was retired and this became L1.

**Method (verified at the primary source, not from a search snippet).**
[arXiv:2601.18835](https://arxiv.org/abs/2601.18835) (PRD `10.1103/k21q-wp8f`), *Beyond FINDCHIRP*. Write a
target's analytic spectrum as a nearby reference's times a ratio, `A_t = A_r · R`. Our matched filter is a
cross-correlation, so with `c_x(t) = IFFT[D conj(A_x)]`,

    c_t = c_r (*) IFFT[conj(R)]        R = A_t / A_r

— the target's complex correlation series is the reference's convolved with **one short FIR**. New module
`pbh/ratiofilter.py`; the kernel is fit by **weighted least squares over frequency** (weight `|A_r|²`, so
accuracy is demanded only where the reference has power), whose normal equations are **Toeplitz**, making the
fit O(n log n + n_taps²) instead of O(n · n_taps²).

**The first golden test FAILED — and the failure was ours, not the method's.** `bank_ratio_golden.py` returned
match **0.814** at 1% Mc separation against a pre-registered **>0.999**. Rather than record L1 as dead,
`bank_ratio_diag.py` ran the control the golden test lacked:

| | result |
|---|---|
| **D1 untruncated kernel** | match **1.000000** ⇒ the algebra and the code are correct; the failure was purely truncation |
| **D2 taps for >0.999** | **1,025** @ 0.1% sep · **4,097** @ 0.5% · **16,385** @ 1.0% |

Subsolar needs **far more taps than the paper's ~250** because these inspirals accumulate enormous orbital
phase, so the same *fractional* Mc offset shifts the phase much further than in the BNS case the method was
tuned on. That is a property of our corner of parameter space, not a defect of theirs.

**A second self-correction.** `bank_ratio_diag.py`'s automated verdict printed *"no memory win for subsolar"* —
comparing taps to an arbitrary `L/16` cutoff that 16,385 **missed by one tap**. Wrong figure of merit. Total
bank memory is the right one and it points the **opposite** way, because the reference bank's size depends
only on the reference–target separation, **not on how dense the target bank is**.

**Cost model (`bank_ratio_costmodel.py`) — ⚠️ SUPERSEDED, WRONG, retained only as the record of the mistake.**
It reported waveform generation **441.8 ms** vs correlation **7.8 ms** and concluded generation was **56×** the
filtering. The correlation was timed on a single 262,144-sample *chunk*; the real pipeline correlates over the
**16.7M-sample segment**, so generation is actually **8%** of the cost. **Every number in the table below is
therefore wrong** — the corrected verdict is at the end of this section.

| ⚠️ SUPERSEDED — target spacing | templates | direct RAM | ratio RAM | RAM win | time/segment win |
|---|---|---|---|---|---|
| 0.1% (current) | 1,619 | 53 GB | 3.2 GB | 17× | 14× |
| 0.05% | 3,235 | 107 GB | 3.6 GB | 30× | 23× |
| **0.01%** | **16,166** | **533 GB** | **6.5 GB** | **82×** | **36×** |

**Does it hold across the whole bank? (`bank_ratio_mcscan.py`, pre-registered.)** Phase ∝ Mc^(−5/3) predicts
~2.3× more taps at the bank's low edge, which would have made the table above optimistic. Measured at fixed
0.5% separation: **4,097 taps suffices at every Mc from 0.18 to 0.85.** *Stated honestly:* the taps grid steps
by ×4, so this **bounds** the Mc dependence rather than disproving it — and the trend is plainly visible below
one grid step (match at 1,025 taps: 0.884 → 0.908 → 0.939 → 0.994 as Mc rises), exactly as the phase argument
predicts. 4,097 is an upper bound at every mass tested, which is what the cost model needs.

**⚠️ THE COST MODEL ABOVE IS WRONG AND IS SUPERSEDED — see the verdict below.** It timed the matched filter on
one 262,144-sample *chunk* (7.8 ms) and concluded generation (442 ms) was **56×** the filtering. But
`bank_dense.py`'s expensive step is `so.segment_stats`, which correlates each template's 8 chunks across the
**entire ~16.7M-sample segment** — 64× more data per correlation. The 82× RAM / 36× time figures do not survive
that correction. Left in place, struck through, because the mistake is the instructive part: *a primitive
measured at the wrong scale.*

### L1 VERDICT: ratio filtering does NOT help subsolar — an honest negative with a mechanism (2026-08-15)
Priced against the **real** pipeline (`bank_ratio_realcost.py`, on the actual 16.7M-sample segment):

| | measured |
|---|---|
| direct: `segment_stats` 5.2 s + generation 0.45 s | **5.6 s** / template / segment |
| ratio: 8 × `oaconvolve` at 16,385 taps | **6.0 s** |
| **speedup** | **0.94× — marginally SLOWER** |
| generation share of the direct cost | **8%**, not the 98% the superseded model assumed |

**The mechanism, which is the real result.** Ratio filtering converts O(N log N) into O(N log K), so the gain is
about `log N / log K`. The published **8× assumes K ≈ 250 taps** (BNS). Subsolar needs **K ≈ 16,385** — measured,
not guessed (`bank_ratio_regime.py`: 8,193 taps → 2.4% statistic error, 16,385 → 0.89%, clearing the 1% bar).
With N = 16.7M that gives a **1.6× theoretical ceiling**, and we measure 0.94×. **The benefit is inversely tied
to the kernel length a signal class demands, and subsolar demands the longest kernels** — precisely because
these inspirals accumulate enormous orbital phase. Not a defect of their method; our regime sits outside where
it pays.

**Why the memory win doesn't rescue it.** Kernels really are ~31× smaller than stored analytic chunks, and that
part is real. But **memory was never the binding constraint**: `bank_dense.py` had already worked around it by
going template-major (generate → score → free). The binding constraint is **compute time**, which ratio
filtering does not reduce. Projected wall-clock for 6 segments is essentially unchanged: 0.1% spacing 15.2 h
direct vs 16.2 h ratio; 0.01% spacing 151.9 h vs 161.8 h.

**What survives, and it is not nothing.**
- The algebra is **exact** — an untruncated kernel reproduces the matched filter to **1.000000**, at every
  separation and even when the basis is built at the wrong remnant.
- The statistic is faithfully reproducible at 16,385 taps: noise-regime error is **unbiased jitter**
  (median bias +0.17%, 57% positive ⇒ the threshold is safe), signal-regime error 0.89%.
- **The dense-bank wall stands, but is now understood** rather than assumed. Follow-up A parked it as
  "intractable locally"; the sweep reopened it as L1; this closes it again with a measured reason and a
  quantitative criterion for when it *would* pay: a signal class needing **K ≲ 1,000 taps**.

**Three of my own errors were caught inside this one item**, which is the honest record: (1) the first golden
test failed at 0.814 because I truncated the kernel, not because the method failed; (2) an automated verdict
declared "no memory win" on an arbitrary `L/16` cutoff missed by one tap; (3) the cost model measured
correlation at chunk scale instead of segment scale, inflating a 0.94× into a claimed 36×. Each was found by
testing my own implementation before believing the result. Gated (48) — **as a negative**, so it cannot be
quietly re-inflated.
Artifacts: bank_ratio_{golden,diag,mcscan,chunked,regime,realcost}.json;
bank_ratio_costmodel.json retained but **superseded**.

**Not built, deliberately.** The hierarchical bank and a `bank_vs_cnn` re-run at 0.01% spacing would cost ~162 h
for no speed advantage, so *does a CNN still tie a matched filter once the bank is adequate?* remains open —
and reopening it needs a genuinely cheaper filter, not this one.

### L6: the SSL win SATURATES by 2,500 unlabeled spectrograms — N4's open caveat answered (2026-08-15)
N4 showed masked-spectrogram pretraining beats from-scratch at scarce labels (+0.124 val AUC at 1,000 labels)
and that the win translates to sensitive distance. Its recorded caveat was that the unlabeled pool *was* the
labeled set's own 20k noise, so *"more unlabeled O3 noise would likely give more"* — a hypothesis, never
measured. GraviBERT ([arXiv:2512.21390](https://arxiv.org/html/2512.21390)) makes the same bet at far larger
scale. The obvious move was to fetch a bigger pool; the cheaper move was to ask whether that would help at all.

**Method (`ssl_poolscale.py`).** Pretrain on 2,500 / 5,000 / 10,000 / 20,000 unlabeled noise spectrograms,
fine-tune each at N4's biggest-gain point (1,000 labels), 3 seeds, and read the gain against pool size.
Pre-registered bar, fixed before running: **gain(20k) − gain(5k) > 0.02 AUC ⇒ still climbing, fetch more.**
`SpecMAE`, `random_mask` and the fine-tune/eval loops are **imported from the N4 scripts** rather than
reimplemented — an earlier draft rewrote the autoencoder with a different channel ladder (1→16→32→64→128
instead of 1→32→64→128→256), which would have silently produced a curve not comparable to N4 at all.

| pool | SSL AUC | gain vs scratch |
|---|---|---|
| scratch | 0.5498 | — |
| **2,500** | 0.6575 | **+0.1076** |
| 5,000 | 0.6466 | +0.0968 |
| 10,000 | 0.6558 | +0.1060 |
| 20,000 | 0.6266 | +0.0768 |

**Result: the gain is fully achieved at 2,500 specs — 8× less unlabeled data than N4 used — and does not grow
thereafter.** Slope over 5k→20k is **−0.020**, robustly failing the pre-registered "still climbing" bar.

**Read honestly, the curve is FLAT, not declining.** Within-pool seed scatter is **sd 0.019**; the spread of
gains across all four pools is **0.031**, i.e. about two standard errors. So the apparent drop at 20k is *not*
significant, and this measurement **bounds** the pool-scaling effect rather than proving it is exactly zero:
we can exclude a gain larger than ~0.03 AUC between 5k and 20k, not a smaller one. Either way, fetching more
noise is not justified — which was the decision this existed to make.

**Cross-detector transfer: null.** Adding the 6,250 available **L1** noise specs to the 20,000 H1 ones changed
the result by **+0.0009** — twenty times smaller than the seed scatter. Unlabeled noise from a *different*
interferometer does not help an H1-applied model here, so the pool cannot be cheaply grown across detectors
either. (Inventory checked rather than assumed: `shards_w64_hl` looks like a 26,250-spec pool but 20,000 of
those are the *same 16 H1 segments* as `shards_w64` — duplicates; only its 6,250 L1 specs are new. Leakage
check: 0 pool segments in H1 val or test.)

**Caveat on comparability with N4.** Our absolute gain at 20k (+0.077) is below N4's (+0.124) because we
re-draw the labeled subset per seed where N4 held it fixed — widening error bars (our scratch sd 0.024 vs
N4's 0.006). That is paired across pools, so the *scaling* comparison is fair, but the absolute numbers are
not a reproduction of N4's.

**Consequence.** **L6b (fetching a larger unlabeled pool) is not justified** — it would have cost days of
GWOSC fetching and disk, in competition with the running L2 job, for an effect this bounds at ≲0.03 AUC.
N4's headline stands and is now better understood: the win is real, cheap, and **saturates almost
immediately**. Gated (49). Artifact: results/ssl_poolscale.json.

### Confound check: is the learned coincidence gain a per-segment constant? — no, and the control that couldn't have told us (2026-08-15)
**Where the warning came from.** The `tabula` sibling planted a per-realization nuisance channel — a
calibration-offset stand-in with **zero dynamical meaning** — and their invariant engine ranked it *more
conserved than the genuine invariant*, passing out-of-sample validation **completely**. The generalisable
mechanism: **held-out validation catches overfitting, not confounding**, because a nuisance constant
generalises flawlessly precisely by being constant.

**Why that pointed straight at Build C-2.** Our learned coincidence head trains on `[eH, eL, |eH−eL|, eH·eL]`
to separate real coincident injections from time-slid noise pairs, and we called `--holdout-segments`
(train 16 segments, evaluate on 8 unseen) the **gold-standard** leakage control. But the two classes are built
differently by construction:

```python
Xpos = pair_feats(iH[tr], iL[tr])                    # positives: an injection's H1 & L1 -> SAME segment
a = noise_tr[rng.integers(0, len(noise_tr), ...)]    # H1 noise, uniform over the pool
b = noise_tr[rng.integers(0, len(noise_tr), ...)]    # L1 noise, INDEPENDENT -> usually a DIFFERENT segment
```

Positives are same-segment pairs; negatives are overwhelmingly cross-segment. **Any** per-segment constant in
the embeddings — a PSD residual, a calibration-like offset — separates those classes with no gravitational-wave
content at all, and `--holdout-segments` **cannot detect it**, because the same-segment-vs-cross structure
persists in the held-out segments. Our strongest leakage control is blind to this entire class of confound.

**The cheap decisive test (`coinc_confound.py`), run before any expensive re-run.** Ask whether the cheating
channel *exists*, using pure noise and no injections: label a pair 1 if its H1 and L1 windows come from the
same segment, 0 otherwise, and train the identical `CoincHead` on the identical features. There is no signal
anywhere in that data, so any separation is a per-segment constant and nothing else.

| test | result |
|---|---|
| **C1** same-segment vs cross-segment, pure noise, 3 seeds | **AUC 0.530** (0.519 / 0.529 / 0.542) |
| **C2** between/within-segment embedding variance, H1 | median **0.025**, p90 0.050, max 0.082 |
| **C2** between/within-segment embedding variance, L1 | median **0.097**, p90 0.146, max 0.172 |
| dims with between > within variance | **0 of 256**, both detectors |

**Verdict: no usable channel.** A head cannot identify same-segment pairs from noise embeddings, and not one
of 256 dimensions carries more between-segment than within-segment variance. **Build C-2's +0.02–0.05
sensitive-distance gain is not explained by a per-segment constant**, and the `--holdout-segments` control —
though blind to this class of confound *in principle* — was not hiding one *in fact*.

**Scope, stated honestly.** This bounds the concern rather than eliminating it. The test ran on the **5 O3a
segments** in `shards_w64_hl` that carry both detectors, not on Build C's own 24 segments (whose embedding
cache lived on the retired L4 VM). AUC 0.530 is *slightly* above chance, so a weak channel exists; it is far
too weak to manufacture the observed gain, but the complete answer would re-run `coinc_learned.py` with
**same-segment negatives** on Build C's segments — which needs 24 fetches and is deferred while the L2
deep-background job has GWOSC. Gated (52) as an assertion that the channel stays absent.
Artifact: results/coinc_confound.json. Credit: mechanism from `tabula` via TheBridge Round 12.

### L2 DEEP FAR: 4,120 years of background — 1/century reached, null 4/4, precision fixed (2026-08-19)

**The audit's central complaint, answered by data.** The 2026-08-09 audit found the 80.5-yr ladder was
**±33–44%** and that 1/decade rested on **8 background events tracing to just 2 distinct H1 windows**. The
prescribed fix was more independent loud-noise samples, and background grows as **N_segments²**, so this ran
the same pipeline out to **727 O4b segments** (from 100). It survived **three power losses** — every recovery
verified **0 corrupt of N** thanks to per-segment atomic checkpointing.

| | published (2026-08-08) | **now** |
|---|---|---|
| segments | 100 | **727** |
| windows | 6,200 | **45,074** |
| analyzed livetime | 113.8 h | **801.3 h** |
| distinct lags | 6,199 | **45,073** |
| background | 80.5 yr | **4,119.9 yr** (51×) |
| deepest rung | 1/decade | **1/century** |

| FAR | published thr | **new thr** | jackknife spread |
|---|---|---|---|
| 1/month | 12.340 | **11.246** | 12% |
| 1/year | 14.112 | **12.799** | 11% |
| 1/decade | 16.121 | **14.532** | 10% |
| 1/century | not measurable | **16.394** | 10% |

**PRECISION FIXED — the headline result.** ⚠️ **PARTIALLY CORRECTED 2026-08-20 — see "the jackknife understates by 4.2×" below: the RATIO here stands, but every absolute spread on this page is ~4× too small, and the *mechanism* claim (effective-N controls the error) is refuted.** Leave-block-out jackknife spread fell from **33–44% → 10–12%**
(sd ≈ 0.41 at every rung), a 3–4× tightening. The old 1/decade value (16.1 ± ~5) contains the new one
(14.532), so **the audit's error bar was honest and the extra data resolved it**. Assumptions re-verified at
scale: H1⊥L1 independence **p=0.647**, shared per-segment data quality **p=0.836**.

**THRESHOLDS FELL, and that is a finding, not a bug.** At fixed FAR the threshold is a fixed *quantile* of the
coincidence distribution (background-years and pair-count both scale as N², so the fraction is invariant).
The drop of 1.1–1.6 therefore means the 100-segment tail was **glitch-inflated** — precisely what the audit
diagnosed when it traced 6 of the 8 loudest H1 windows to one segment.

**THE NULL IS STRONGER AND BETTER UNDERSTOOD.** Zero-lag max is **11.295, verified unchanged** at 7.3× the
data — same segment (1397232640), same window, H1 **+12.53** with L1 at **−1.24**. Checked directly against
the cache rather than trusting the artifact, because an unchanged value across a 7× data increase is exactly
the shape of a stale read. It is not one: nothing louder exists in 801 h.

*Pre-registered before the numbers landed:* our zero-lag search covers 0.0914 yr, so at 1/month we **expect
1.1 background events** — an event above the 1/month threshold is the *median* noise outcome, not a
detection. Measured: the loudest zero-lag event has **FAR = 11.5/yr**, i.e. **1.05 expected**. Textbook null.
It exceeds the 1/month threshold (11.295 > 11.246) and this is meaningless; **the claim-capable rungs are
1/year and deeper**, where zero-lag sits far below.

**Single-detector ceilings now put 3 of 4 rungs beyond glitch reach** (max H1 **12.53**, max L1 **6.37**):
1/month 11.25 is glitch-reachable; **1/year, 1/decade and 1/century all require genuine two-sided
coincidences.** One-sidedness vs loudness reproduces the audit's pattern at 7× the data: **0%** one-sided in
the top 25, 6% in the top 100–500, **97%** beyond rank 2000. The loudest zero-lag event is explicitly
one-sided and dies under `min` (2.59 vs a 4.95 threshold). **Null holds in 4/4 configurations**
(all-segs/drop-59 × sum/min), each against its own matched threshold.

**`sum` still wins, four rungs deeper.** `min` costs 3–4% of sensitive distance at every FAR (0.96–0.98×)
while halving background instability (worst half-to-half 15% vs sum's 34%); `veto` is 0.99–1.00×. Extends
G2a and the 80-yr audit's verdict into the 4,000-yr regime. **Keep `sum`.**

**WHAT DID *NOT* IMPROVE — the honest half.**
1. **Effective sample size is still the binding limit.** 1/decade's 425 background events come from **8
   distinct H1 windows**; 1/century's 43 events from **3**. That "8" is the *same count* the published
   1/decade had. We multiplied background-years by 51 and barely moved the number of independent loud
   Hanford glitches that set the deep tail. **The audit's diagnosis survives its own fix.**
2. **Convergence is not demonstrated.** 1/decade reads 17.058 (n=60), 16.779 (n=80), **14.532 (n=727)** —
   still drifting down at 9× more data, because the tail is set by rare glitches whose density is still
   being sampled.
3. **Non-stationarity persists**: first half 1/decade 14.999 vs second half 11.243 (**25% apart**, 34% at
   1/month) — the glitch cluster's placement, not bulk noise drift.
4. **Poisson bands remain an order of magnitude too narrow** (1/decade ±0.04 vs jackknife ±0.41). Never quote √k.

**THREE TOOLING BUGS FOUND WHILE THE RUN WAS IN FLIGHT** — all found by pre-flighting the analysis at the new
scale rather than waiting for the end, and all of the same species: *a constant sized against yesterday's data
volume becomes a silent truncation when the data grows.*
- **Silent rung-drop (×2 scripts).** The background keeps only the loudest `keep` lag values, but a rung's
  event count grows as N². At 80.5 yr, 1/month needed ~966 events (cap 20,000 = ample); at 4,120 yr it needs
  **49,440** — so `far_background_validation.py` **and** `far_min_vs_sum.py` would have silently returned a
  report *missing 1/month*, including from the jackknife, the very test that decides this item. `ladder()`
  dropped it with **no message at all**. Fixed: caps 20,000→400,000 / 5,000→200,000, 1/century added, and an
  unresolvable rung now prints `!! RAISE keep, not a data limit` instead of vanishing.
- **Livetime overcount (×3 scripts).** Background time used `n_segments × 4096 s`, but the 8-s whitening crop
  leaves **62 of 64 windows**, so 3.23% of the quoted time was never searched — inflating background-years and
  nudging thresholds *anti*-conservative. Same species as the earlier honest-slides lag overcount. Fixed to
  `n_windows × 64 s` in `far_deep.py`, `far_background_validation.py`, `far_min_vs_sum.py`; all numbers above
  use the corrected **801.3 h / 4,119.9 yr**.

**Net.** Reach is one rung deeper (**1/century**), precision is **3–4× tighter and now honestly quoted**
(14.5 ± 1.5 rather than 16.1 ± 5), the **null is 4/4** with 3 of 4 rungs above the single-detector ceiling,
and the remaining limit is named and measured: **independent loud-noise samples, not livetime** — which is why
real LVK searches lean on signal-consistency vetoes and DQ flags rather than raw time-slides.
Artifacts: far_deep.json, far_background_validation.json, far_glitch_anatomy.json, far_min_vs_sum.json.

### The 45,073 zero-lag measurements we had been discarding — independence verified to the 99.98th percentile, and a structural limit found (2026-08-19)

**The gap this fills.** Every deep-FAR result so far compared exactly *one* number — the loudest zero-lag
coincidence — against a threshold, discarding the other 45,073 zero-lag measurements sitting in the cache.
Two questions live in the discarded part, neither answerable from a maximum: **(Q1)** is H1⊥L1, the
assumption the entire 4,120-yr ladder rests on, true *where the ladder uses it*? We had verified it only via
the **bulk** correlation (r = −0.0022, p = 0.65) — the one place it is easiest to satisfy. **(Q2)** is there
a **sub-threshold population**? A real subsolar population too faint to produce any loud event would appear
only as a diffuse statistical excess across the whole distribution.

**The null is assumption-free.** Under H1⊥L1, the zero-lag sum `H_i + L_i` and the slid sum `H_i + L_j` are
draws from the same distribution — so the time-slide background *is* the independence null, built from the
same windows and the same noise. No model of either detector's score distribution is required.

**GOLDEN TEST FIRST.** On synthetic data the instrument is silent under true independence (|z| ≤ 0.66) and
detects a planted 1% shared-noise component at **z = +6.35** — and, informatively, *only at the deepest
rung*: correlated noise affecting a small fraction of windows is invisible in the bulk. Exactly the regime
we had never examined.

**RESULT — CLEAN, in bulk and tail.**

| expected | threshold | observed | ratio | z |
|---|---|---|---|---|
| 9,998.6 | −0.900 | 10,010 | 1.00 | +0.11 |
| 999.5 | 1.095 | 997 | 1.00 | −0.08 |
| 100.0 | 3.734 | 103 | 1.03 | +0.30 |
| 10.0 | 7.456 | 8 | 0.80 | −0.63 |

**KS(zero-lag, background) = 0.00214** against a 0.00640 critical value (**verified 2026-08-22 to be a measurement, not our own grid resolution** — see the clip-band check below) — the two CDFs agree to ~0.2%
*everywhere*, not merely above some threshold. Joint exceedances at the 90th percentile: **451 observed vs
450.7 expected** (perm p = 1.000); 99th: 4 vs 4.5. Median shift +0.0018. **No correlated noise, no
sub-threshold population.**

**Independence is now verified to the 99.98th percentile** (score 7.456), with ±3.2% precision at the 97.8th
and ±9.9% at the 99.77th — versus "the bulk" previously.

**THE STRUCTURAL FINDING (the part worth keeping).** Three of our four rungs sit **beyond all zero-lag data**:

| rung | threshold | zero-lag events at/above |
|---|---|---|
| 1/month | 11.246 | 1 |
| 1/year | 12.799 | **0 — beyond all data** |
| 1/decade | 14.532 | **0** |
| 1/century | 16.394 | **0** |

Verification reaches score ~7.5; the 1/century threshold is 16.4, itself above the zero-lag maximum of
11.295. **And this gap cannot be closed by sliding**: time-slides manufacture more *pairs* but zero new
zero-lag samples. Testing independence at amplitude 14.5 requires observing until a real event of that
amplitude occurs — about a decade at 1/decade FAR. We have 0.0914 yr.

⇒ **Every factor of depth gained in a time-slide background moves the threshold further past the regime
where the independence it assumes can be empirically verified.** L2 showed livetime does not buy independent
*samples*; this shows it does not buy *verification reach* either — and unlike the first limit, no amount of
computation substitutes for real observing time. This is a general property of time-slide backgrounds, not
of our pipeline.

**Honest scope.** The constraining rungs are the well-populated ones (±1% at 10,000 expected, ±3.2% at
1,000); the 10-expected rung carries ±35% and the 99.9th-percentile dependence test has expected 0.0 and is
uninformative. So: independence is **verified** to ~99.98th percentile, **assumed** where the claims live,
and that assumption is **untestable with this much observing time** by construction. A one-sidedness column
was guarded after the fact — `min < 0.15·max` inverts meaning once the louder detector is itself negative,
which it is at the shallow rungs, so it is now computed only where max > 0 and reports its sub-sample size.
Artifact: far_zerolag_population.json.

### The jackknife understates our error bars by 4.2× — and effective-N does NOT control them (2026-08-20)

**Why this exists.** L2 reported threshold precision improving from ±33–44% (100 segments) to ±10–12% (727)
and I wrote it up as *"the audit's central complaint answered by data"*. Both numbers came from one
estimator — a leave-10%-block-out jackknife — and neither had ever been checked against the actual sampling
distribution. Measuring the scaling law meant to quantify L2's effective-N story instead exposed the
estimator.

**(1) THE JACKKNIFE UNDERSTATES THE TRUE SAMPLING SPREAD BY 4.2×** (`far_estimator_bias.py`; both estimators
run on the *same* subsets at the *same* n, so no sample-size difference can explain the gap; L2's jackknife
reproduced exactly — contiguous 10% blocks, `np.std` ddof=0):

| n | between-subset σ | jackknife σ | bias |
|---|---|---|---|
| 100 | 2.03 / 1.97 / 2.02 | 0.485 / 0.503 / 0.484 | 4.18 / 3.91 / 4.17× |
| 320 | 2.44 / 2.25 / 2.27 | 0.570 / 0.540 / 0.561 | 4.28 / 4.16 / 4.05× |

**The mechanism is specific**: the jackknife drops 10% of segments, but the deep tail is set by a handful of
glitchy segments — with 727 in play, dropping 72 almost never removes *the* dominant glitch, so the estimator
looks calm regardless of the true uncertainty.

**The bias is STABLE in n (growth 1.02×), which is what saves the L2 ratio.** Both endpoints are understated
by the same factor, so 33–44% → 10–12% is a real *relative* improvement. **What must change is every absolute
number: quote 1/decade as 14.53 ± ~1.7, not ± 0.41.** (The committed "±1.5" is coincidentally close but was
derived by treating a jackknife *range* as an error bar; the corrected derivation is jackknife σ × 4.16.
Caveat: the bias factor is measured at n=100 and n=320 and extrapolated to 727.)

**(2) EFFECTIVE-N GROWS AS L2 SAID — BUT DOES NOT CONTROL THE ERROR** (`far_effective_n.py`, 40 subsets per
size, finite-population corrected). The pre-registered test was a *collapse*: if the loud windows are the
effective independent draws, σ·√N_eff must be constant across every size **and every rung**. A power-law fit
cannot distinguish "N_eff controls it" from "n controls it"; the collapse can.

| quantity | predicted | measured |
|---|---|---|
| N_eff ~ n^β | β ≈ 1.0 | **β = 0.927** ✓ |
| σ ~ n^−α | α ≈ 0.5 | **α = 0.10**; per-rung −0.016 [−0.114,+0.070] / 0.052 [−0.030,+0.120] / 0.267 [0.173,0.347] ✗ |
| σ·√N_eff | constant | **62% scatter, rising 5.7 → 16.7** ✗ |

⇒ **more independent loud windows do not buy threshold precision.** L2's "effective sample size is the
binding limit" is right that something binds and **wrong about what**: N_eff and σ are decoupled. α is firmly
below the naive 0.5 at every rung and consistent with zero at two of three — so **more data barely improves
the absolute precision of a fixed-FAR threshold**, and the relative improvement seen in L2 is substantially
the threshold *mean* rising (1/month 7.07 → 10.87 across n=20→320), i.e. the known non-convergence, rather
than the error shrinking.

**(3) A HEAVY-TAIL HYPOTHESIS OF MINE, RAISED AND REFUTED** (`far_sigma_convergence.py`, 200 independent
draws). Three signs suggested the sampling distribution might have infinite variance — two runs disagreeing
1.5×, σ rising systematically with rep count, and the threshold being an extreme of a glitch-dominated
distribution. Had it been true, **no ± would have been quotable at all** — not the original ±33–44%, not
L2's ±10–12%, not the ±1.7 above. Direct test says **no**: Hill tail index **19.7–195** (infinite variance
needs < 2), σ flat from m=80 (late-half slope +0.03). σ ≈ 1.85 at n=160, near-identical across rungs. The
1.5× disagreement was ordinary small-sample SD noise at 10–20 reps. **A ±σ is a legitimate summary.**

**TWO OF MY OWN INSTRUMENTS RETURNED FALSE VERDICTS TONIGHT, both caught by reading the numbers rather than
the verdict line.** (a) The effective-N "planning number" computed 2^(1/α) with α≈0, printing 10⁵⁵× — now
suppressed unless α is resolvably > 0.15, with the reason stated. (b) The heavy-tail rule compared σ's growth
over the *whole* range (dominated by ordinary small-m SD bias, present for any distribution) against the
quantile range's drift over the *late half* — apples to oranges, and it declared "heavy-tailed" for all three
rungs. Compared like-for-like, σ's late slope (+0.03) is *flatter* than the IQ range's (+0.06), the opposite
of the signature. Rule rewritten to use the Hill index as the primary discriminator, with the failure kept in
the docstring as a warning. Same species as the silent rung-drops: **a confident automated verdict is the
easiest thing to not check.**

**Net.** The 4,120-yr background, the 1/century reach and the null 4/4 are untouched — this concerns the
error bar, not the threshold or the search result. What changes: absolute spreads ×4.2, the mechanism claim
retracted, and "more data fixes the precision" replaced by **more data barely moves it**.
Artifacts: far_estimator_bias.json, far_effective_n.json, far_sigma_convergence.json.

### Are we quoting the wrong number? Threshold-at-fixed-FAR vs FAR-of-the-observed-event — a pre-registered refutation (2026-08-21)

**The idea, and why it looked promising.** Two results said the deep-FAR ladder saturates: the threshold at a
fixed FAR barely improves with data (σ ~ n^−0.10 absolute), and 3 of 4 rungs sit beyond any zero-lag sample.
Both concern an *extreme quantile* estimated by extrapolating into a sparse, glitch-dominated tail. But a
null result is a statement about the event we **observed**, and the natural statistic — the FAR of the
loudest zero-lag event — is a **count** in a region holding ~47,500 background events. Interpolation where
the data is thick, rather than extrapolation where it is thin.

**REFUTED, as pre-registered.** `far_null_precision.py`, 40 subsets per size, FPC-corrected, both summaries
computed on the *same* subsets so no seed or sample-size difference can explain a gap:

| summary | relative spread ~ n^−α |
|---|---|
| threshold 1/month / 1/year / 1/decade | 0.214 / 0.252 / 0.308 (mean **0.258**) |
| **FAR at the observed 11.295** | **0.283** |

The FAR of our loudest event converges at essentially the **same rate** as the threshold. ⇒ **no choice of
summary statistic escapes the glitch limit; the uncertainty is a property of the deep tail, not of how we
describe it.**

**THE STRUCTURE OF THE REFUTATION IS THE USEFUL PART.** Sweeping the probe value S₀ shows convergence is
governed by how densely the background populates that amplitude:

| S₀ | mean background count | α |
|---|---|---|
| 7.5 | 22,634 | **0.969** |
| 9.5 | 7,036 | 0.389 |
| 11.295 (observed) | 2,145 | 0.283 |
| 13.0 | 138 | **0.019** |

The interpolation/extrapolation intuition was right about the **mechanism** and wrong about **where our event
sits**: 11.295 is already in the sparse regime. A FAR *is* well determined — but only at amplitudes where
background events are plentiful, which is precisely not where a loudest-event claim lives. The caveat
pre-registered in the docstring turned out to be the entire result.

**MECHANISM CONFIRMED BY AN INVARIANT THAT CANNOT BE VIOLATED** (`far_jacobian_check.py`; the check, and the
principle of seeking a quantity the machinery cannot violate even when wrong, came from `ansatz` via the
cross-session coordination channel). If S → FAR is a deterministic monotone map then the FAR spread is
*forced*: σ(ln FAR) = |d(ln FAR)/dS| · σ(S). Predicting the FAR spread from the **measured** threshold spread,
with the derivative measured from our own probe counts at each n:

| n | d(lnC)/dS | σ(thr) | predicted (dex) | measured (dex) | ratio |
|---|---|---|---|---|---|
| 40 | 1.024 | 2.953 | 1.314 | 1.417 | 1.08 |
| 160 | 1.124 | 2.028 | 0.990 | 1.107 | 1.12 |
| 320 | 1.080 | 2.276 | 1.067 | 1.460 | 1.37 |

**Tracks within 8–37% across an 8× range in n** ⇒ the two summaries carry the same information, established
by a relation that must hold rather than by argument. Our ±1.7 on the threshold simply **is** a ×/÷17 on the
rate, because the count falls 293 → 17 between 11.3 and 13.0.

**An unexplained residual, kept as such.** Measured runs ~20% above predicted and the excess *grows* with n
(1.08 → 1.37). A pure counting-noise explanation predicts the opposite trend, so the extra term is not
identified. Recorded, not rationalised.

**Note on comparing α's across our own scripts.** `far_effective_n` measured α on the **absolute** spread
(0.10); this measures the **relative** spread (0.258). They differ because the threshold *mean* rises with n
— the known non-convergence — so relative precision improves faster than absolute. Same data, two different
quantities; do not quote one as the other.
Artifacts: far_null_precision.json, far_jacobian_check.json.

### `sum` survives a real objection: equalising the 2× noise-tail asymmetry HURTS (2026-08-22)

**Where the objection came from.** `bridge` (TheBridge/G3) reported a statistic whose *gain* varied 2.1× along
a comparison axis — manufacturing structure with no physics in it — and, the part that transfers, **16× more
data bought essentially nothing**, because a systematic gain variation does not average down at any N.
Checking our own exposure found the reassuring answer on their axis and an unexamined one next to it:

| | H1 | L1 | ratio |
|---|---|---|---|
| gain (planted SNR → score), O4b | 0.4735 | 0.4613 | **0.97×** |
| noise q99.9 | 5.402 | 3.417 | 1.58× |
| noise max | 12.532 | 6.370 | **1.97×** |
| Hill index (top 1%) | 3.84 | 5.57 | H1 genuinely heavier |

For an unweighted `sum`, **the detector with the heavier noise tail dominates the false-alarm rate regardless
of its gain.** G2a and the L2 audit both tested `sum` against `min`/`veto` and kept `sum`, but neither tested
a statistic that *equalises the tails* — so the ~2× asymmetry had never actually been challenged.

**The candidate.** `coinc_tailnorm.py`: map each detector's score to its own noise tail-probability
`u_d(s) = −log₁₀ P(S_d ≥ s)` before summing, so each contributes at equal **rarity** rather than equal
**score**. In-sample it looked like a win: **+5.2% / +5.1% / 0.0%** sensitive distance at 1/month / 1/year /
1/decade.

**IT DOES NOT SURVIVE (`coinc_tailnorm_stress.py`).** Fitting the map on a random half of *segments* and
building the background on the other half — segment-level so within-segment correlation cannot leak:

| FAR | in-sample | held-out | bootstrap 90% CI | P(>1) |
|---|---|---|---|---|
| 1/month | 1.052× | **0.983×** | [0.962, 0.997] | 0.02 |
| 1/year | 1.051× | **0.962×** | [0.932, 0.981] | 0.00 |
| 1/decade | 1.000× | **0.946×** | [0.903, 0.972] | 0.00 |

The gain was **entirely in-sample**: the map was flattening the specific noise realisation whose own
time-slides then formed the background, lowering the matched-FAR threshold without buying discrimination.
Held out it is *significantly worse* than `sum` — every CI excludes 1 on the low side. Note this is the
**confounding** direction, not overfitting: a map fitted to a realisation generalises badly to a fresh one
precisely because it encoded that realisation. Same species as the tabula warning.

**A MECHANISM I PROPOSED AND REFUTED IN THE SAME RUN.** I predicted the 1/decade zero came from *censoring* —
the empirical map floors at `u_max = log₁₀(N) = 4.65` past the loudest observed noise window, so deep rungs
would degenerate to a constant, which would have tied neatly to the zero-lag population finding that our deep
rungs sit beyond all observed data. Measured: **0.0% of the background sits at the ceiling at any rung**
(thresholds 5.26 / 6.10 / 6.80 against a ceiling of 8.71). The zero was just the artefact fading where the
background thins. A tidy mechanism, and fiction.

**Net: `sum` stands — and is now vindicated *against the specific objection* rather than merely untested.**
The honest reading is stronger than "no change": equalising the tails actively **costs** 2–5% of reach, so
the tail asymmetry is carrying real information that a rarity transform discards. Artifacts:
coinc_tailnorm.json, coinc_tailnorm_stress.json. Objection and the gain-axis framing: `bridge` via the
cross-session channel.

### Clip-band check: the KS = 0.00214 is physical, not our binning floor (2026-08-22)

**Technique borrowed from `bridge`** via the cross-session channel: *before quoting a small residual, sweep
the numerical floor that could be producing it and check the residual does not track it.* They swept a clip
floor five decades and obtained a band 2254× smaller than the spread they were reporting, which is what
licensed calling it physical rather than a precision limit.

**Why ours needed it.** `far_zerolag_population` reports KS(zero-lag, background) = 0.00214 vs a 0.00640
critical value, and we wrote that the two CDFs "agree to ~0.2% EVERYWHERE". That number is load-bearing: it
is the evidence for H1⊥L1 across the whole distribution, and the 4,120-yr ladder rests on that independence.
But both CDFs were built by **quantising scores into 40,000 bins**, and a KS computed between two binned CDFs
cannot resolve differences finer than the grid. The published claim did not distinguish "the CDFs agree" from
"our grid cannot tell them apart".

**Method** (`far_ks_clipband.py`): accumulate the background once at the finest grid — the expensive part is
the 45,073-lag sweep, not the histogram — then coarsen by summing adjacent bins to obtain every coarser grid
for free, and recompute KS at each.

| bins | 2,500 → 640,000 |
|---|---|
| KS variation across the sweep | **1.033×** |
| d(log KS) / d(log bin width) | **−0.0063** |

**Flat across a 256× range of bin counts** ⇒ **PHYSICAL**. The 0.00214 is a measurement; the claim stands as
written. Had KS tracked the bin width we would have had to re-quote it as an upper bound.
Artifact: far_ks_clipband.json.

**Process note, recorded because it is the actual lesson.** We would not have run this check on our own — it
came from another project's methodology, applied to a number we had already published and were relying on.
The cost was minutes. The general form: **any small residual quoted from a discretised computation needs its
discretisation swept before it counts as a measurement.**

### The deep-FAR tail is NOT glitch-driven — the veto programme dies before it starts (2026-09-02)

**The premise being tested.** Every limit the deep-FAR arc hit traces to a handful of loud H1 windows (8 set
1/decade, 3 set 1/century), and 51× more background barely moved those counts. We had called them "glitchy
segments" throughout, and the plan was to identify the glitch family so they could be vetoed on principled
grounds rather than the post-hoc removal we refused as tuning.

**The instrument, validated on knowns before use.** `glitch_anatomy_morphology.py` measures the largest
*connected* excess cluster in a whitened window: per-row median normalisation, a trials-aware threshold (so
that <0.01 false pixels are expected across ~250,000 time-frequency pixels), then connected-component
labelling. Smoke-tested on synthetics — a 10 ms broadband burst returns **0.055 s / 112 Hz** (published blips:
~10 ms, ~100 Hz) and a 4 s low-frequency tone returns **4.04 s / 64 Hz peak** (published scattered light:
~4 s arches, 8–64 Hz). Pure noise returns *no-excess*.

**THE RESULT, and it inverts the premise.** On the one deep-FAR segment obtainable before GWOSC degraded, all
three tail-setting windows contain **no supra-threshold cluster at all** — max-excess 17.9–22.4, kurtosis ~0,
i.e. **0.88× an ordinary noise window** from the same segment. Meanwhile an ordinary window in that segment
holding an enormous transient (**max-excess 4708, kurtosis +2404**) scores **−0.484**: below average, ignored.
Window extraction was verified by re-scoring — all three reproduce their cached scores to 3 decimals.

**Confirmed at n = 1,860** (`glitch_score_correlation.py`, 30 cached O3a segments — no fetching required,
the data was already on disk):

| measure | value |
|---|---|
| Spearman(score, max_excess) | **+0.091** |
| top-18 by score ∩ top-18 by excess | **0 windows** |
| fraction of top scorers with any excess | **0.11** (all windows: 0.12) |
| median score of the highest-excess windows | **−0.900** (all windows: −0.976) |

⇒ **the detector does not respond to transient excess.** Loud glitches score like ordinary noise, and the
loudest-scoring windows are no more transient-rich than average. **There is nothing to veto**, and the whole
signal-consistency programme — which we had queued as the obvious next move — is dead on evidence rather than
left plausible-but-untested.

**IT RETRO-EXPLAINS TWO RESULTS WE ALREADY HAD AND DID NOT UNDERSTAND.** G2a and the L2 audit both found
`min` and `veto` cost reach and never bought any (0.96–0.99×); the tail-normalisation test found equalising
the 2× noise-tail asymmetry actively **hurt** by 2–5% held-out. Both are exactly what you expect if the deep
tail is not glitch-driven: a consistency cut can only remove glitches, and there are none to remove. Three
independent measurements, one mechanism.

**A CORRECTION TO OUR OWN LANGUAGE.** RESULTS.md, CLAUDE.md and several commit messages describe these as
"glitchy segments" and "independent loud-noise samples", and attribute the effective-N limit to glitches.
That language is now wrong. The windows setting the deep tail are **not transients**; they are something
distributed in the noise that the CNN responds to and that leaves no localised time-frequency signature. The
effective-N limit is real and unchanged — what changes is the mechanism we ascribed to it.

**SCOPE, stated precisely.** The n=1,860 test is O3a (in-domain for cnn_w64, which was trained on it) and
establishes the *mechanism*: this detector does not track transients. Whether the specific O4b tail windows
are glitch-free is established for **3 of 8** so far; the remaining five are behind a GWOSC outage running at
~1 kB/s. The two questions are separate and only the first is settled. **Open and now much more interesting:
what DOES the CNN respond to?** — since it is demonstrably not transient power.
Artifacts: glitch_morphology checkpoints, glitch_score_correlation.json.

### What the CNN actually responds to: band-limited noise power at ~110 Hz — and it unifies the arc (2026-09-02)

**The question, sharpened by the previous result.** Having established that the detector does *not* respond
to transient excess (Spearman +0.091 over 1,860 windows; a max-excess-4708 transient scores −0.484), the
obvious follow-up is what it *does* respond to. `cnn_response_probe.py` asks two ways, because a correlation
can be confounded and an occlusion map cannot say what a feature means.

**(A) Which part of the spectrum tracks the score** (band-power vs score, over noise windows):

| band (Hz) | Spearman |
|---|---|
| 50–71 | +0.210 |
| 73–104 | +0.351 |
| **107–153** | **+0.395** |
| 157–224 | +0.197 |
| 229–327 | +0.123 |
| 490–700 | +0.040 |
| 717–1024 | −0.038 |

**(B) Occlusion** — replace one band with the window's own median, re-score, record the drop — run on three
populations: the top-scoring noise (what triggers a false alarm), injected subsolar signals (what the
detector is meant to use), and median noise (the null).

| | top-scoring noise | median noise | injections |
|---|---|---|---|
| peak band | 107–153 Hz | 73–104 Hz | 73–104 Hz |
| sensitivity centre | 121.0 Hz | 118.4 Hz | **111.4 Hz** |
| fraction of sensitivity below 224 Hz | **0.996** | 1.000 | **0.972** |
| profile correlation vs injections | **+0.916** | +0.932 | — |

**THE ANSWER: band-limited power in ~73–224 Hz, centred near 110–120 Hz.** And false alarms use the *same*
region as real signals — the profiles correlate at **+0.916** and the sensitivity centres differ by 10 Hz.

**THE DETECTOR IS HONESTLY FOOLED, AND THAT UNIFIES FOUR EARLIER RESULTS.** A high-scoring noise window is
not a glitch or an artefact: it is noise with elevated power exactly where the detector watches, which is
exactly where a subsolar chirp deposits its SNR (these binaries sweep slowly at low frequency, so that is
where the accumulated signal lives). Signal and false alarm are *the same feature*. Consequences we had each
measured separately and never connected:

| previously measured | now explained |
|---|---|
| `min`/`veto` cost 1–4% reach and never bought any (G2a, L2) | nothing separate to cut — the trigger *is* the signal feature |
| tail-normalising the 2× noise-tail asymmetry HURT 2–5% held-out | same |
| the tail-setting windows contain no glitch morphology | there are no glitches; it is band-limited noise power |
| **H1×L1 coincidence buys 1.37×** while every single-detector cut fails | the fluctuation is **independent between detectors**, so it dies under coincidence and survives every within-detector test |

That last row is the one we had never had a mechanism for. Coincidence works *because* the trigger is
ordinary noise rather than an instrumental artefact — the opposite of the glitch picture we had assumed.

**AN ACTIONABLE OBSERVATION.** Above ~250 Hz the model contributes essentially nothing (occlusion drops go
negative; band correlation −0.04 at 717–1024 Hz), yet the analysis band runs to 1024 Hz. Roughly
three-quarters of the spectrogram is capacity the network learned to ignore. Whether narrowing the band buys
anything or is merely cosmetic is a separate, testable question — logged, not claimed.

**A SIXTH FALSE VERDICT FROM MY OWN TOOLING, and the same species as the other five.** The script first
declared "false alarms key on a DIFFERENT band → the noise trigger is a separate feature" — from an `argmax`
over two **adjacent** bands, while the two profiles correlated at 0.92 and their sensitivity centres differed
by 10 Hz. An argmax over adjacent bins is a coin flip when the peak is broad. Rewritten to compare profile
*shape*, with the failure kept in the code. Caught, as all six were, by reading the numbers rather than the
conclusion line.

**Scope.** O3a data (in-domain for cnn_w64, which was trained on it); 20 segments, 1,240 noise windows, 40
injections. The injections are louder than the noise windows (base score 25.5 vs 1.5), so the occlusion
*magnitudes* are not comparable across populations — only the normalised *shapes* are, which is what the
claim rests on. Artifact: cnn_response_probe.json.

---

## PRE-REGISTRATION (2026-09-04, written before the seeds=5 result exists)

A re-run of `ssl_sensdist.py` at `--seeds 5` is in flight (PID 23054). The bar is written down first, because
the whole reason it is running is that the committed 2-seed artifact could not support a scatter-based claim,
and a bar chosen after seeing five seeds would be worth no more than the one it replaces.

**What is being asked.** Does the SSL data-wall *trend* — the gain being larger at scarce labels — clear the
seed scatter? The committed run reports **+0.278** sensitive-distance fraction at 2,000 labels falling to
**+0.01** at 8,000, but with `seeds: 2` (one degree of freedom) that separation was never testable. The gate
currently guards it with a fixed 0.05 margin, which is a judgement wearing a number.

**Declared in advance.**
1. **Statistic.** `gap = delta@2000 − delta@8000`, where `delta = mean(ssl) − mean(scratch)` per budget, and
   `SE(gap) = sqrt(SE(delta@2000)² + SE(delta@8000)²)` with `SE(delta) = sqrt(sd_ssl² + sd_scratch²)/sqrt(5)`.
   Same form as the N4 AUC gate already uses, so it is not a statistic chosen to fit the answer.
2. **Bar.** The trend is **resolved** if `gap > 3·SE(gap)`. Between 2σ and 3σ it is **suggestive, not
   resolved**, and gets reported that way rather than rounded up.
3. **If it does not clear.** The fixed 0.05 gate margin is replaced by an explicit statement that the trend is
   not resolved at this seed count — *not* quietly retained. A margin that survives because nobody re-measured
   it is precisely the failure this day has been about.
4. **The headline is also on the table.** If `delta@2000` at five seeds lands materially below +0.278, the
   published number is corrected to the five-seed value. Two seeds is the weaker measurement, and that it
   happens to be the one already committed carries no weight.
5. **What this cannot settle.** Five seeds is still small. Clearing 3σ says the trend is real at this scale;
   it says nothing about the *shape* of the label-efficiency curve, which needs more budgets, not more seeds
   — the same direction-versus-shape distinction settled on the ringdown side this morning.

**Prediction, recorded so it can be wrong.** I expect the trend to clear comfortably: the AUC version of the
same trend measures 10σ, and the distance gap (0.278 vs 0.01) is proportionally larger than the AUC gap. I do
**not** expect `delta@2000` to survive at 0.278 — a two-seed mean of a quantity this noisy is likelier high
than exact, and the zero-FA floor beside it is already known to be threshold-sensitive.

### RESULT against that pre-registration (2026-09-04): **2.94σ — SUGGESTIVE, not resolved. My prediction was wrong.**

| budget | scratch per seed (5) | ssl per seed (5) | Δ | SE |
|---|---|---|---|---|
| 2,000 | 0, 0, 0, 0, 0 | 0.223, 0.334, 0.324, 0.107, 0.212 | **+0.240** | 0.042 |
| 4,000 | 0, 0.104, 0.211, 0, 0.105 | 0.118, 0.353, 0.205, 0.232, 0.207 | +0.139 | 0.055 |
| 8,000 | 0.205, 0.235, 0.229, 0.221, 0 | 0.225, 0.241, 0.257, 0.214, 0.248 | +0.059 | 0.045 |

`gap = +0.181 ± 0.062` ⇒ **2.94σ**. The declared bar was 3σ. **It missed, and it is reported as suggestive
rather than rounded up** — which is the entire reason the bar was written down first, because 2.94 against a
bar set afterwards would have been "essentially three sigma."

**I predicted it would clear comfortably and it did not.** The reasoning was that the AUC version of the same
trend measures 10σ and the distance gap is proportionally larger. What that missed is that the distance metric
is far noisier per seed than AUC — the scatter, not the gap, is what changed.

**The headline moves and the direction was predicted.** `Δ@2000` is **+0.240 ± 0.042** at five seeds against
the committed **+0.278** at two. They agree inside 1 SE, so this is a sharpening, not a retraction — but the
five-seed value is the one to quote, and the two-seed number was high, as recorded in advance.

**The metric is CENSORED and this is the honest limit on the 2.94.** A model below the 1%-FAR detection floor
scores exactly 0.0, and **8 of 30 per-seed values are exactly zero** (all five scratch runs at 2,000; two at
4,000; one at 8,000). The per-budget distribution is therefore a mixture — "below floor" and "functional at
≈0.22" — not a Gaussian, so a normal-theory SE is the wrong error model and **2.94σ is approximate**. The
zero among the 8,000 scratch seeds is the one that matters: dropping it would *raise* the significance. It has
not been dropped, because a post-hoc exclusion that strengthens your own result is not available under a
pre-registration, and the reason to write one is exactly to make that unavailable at the moment it is tempting.

**What would settle it:** more seeds (the scatter is the binding term, not the gap), or an error model that
respects the censoring. Not more budgets — those address the curve's *shape*, which this was never going to
resolve. **Gate:** the old fixed 0.05 margin is **removed**, not retained; the seeds=5 artifact is gated at
the pre-registered band, so a later re-run cannot quietly promote 2.94 to "resolved" without writing it up as
a new result. Artifact: `ssl_sensdist_seeds5.json`.

### PRE-REGISTRATION II (2026-09-04, before the n=20 run) — settling the 2.94σ

The 5-seed result landed at 2.94σ against a 3σ bar. Running more seeds *because* it fell just short is the
textbook setup for optional-stopping bias, so the design below removes every degree of freedom I would
otherwise have while the numbers come in.

**1. Fresh seeds, not more of the same ones.** `--seed-offset 5 --seeds 20` ⇒ seed values 5–24, none of which
the 5-seed run drew. **Primary analysis uses the new 20 alone.** The n=5 run selected this hypothesis for
follow-up, so pooling it in would reuse the data that did the selecting. The pooled n=25 number is reported
as **secondary**, clearly labelled.

**2. Fixed n, declared now: 20.** From the pilot (`gap 0.181 ± 0.062` at n=5), SE scales as 1/√n, so n=20
gives SE ≈ 0.031 — about 5.8σ if the gap holds, and still ~3.2σ if the true gap is only 0.10, i.e. if the
pilot was a lucky-high draw. **I will not extend the run if it lands near a bar.** If it lands at 2.9 again,
that is the answer.

**3. Primary test is censoring-robust, because the normal-theory SE is known to be the wrong model here.**
8 of 30 pilot values were exactly 0.0 (below the 1%-FAR floor), making each budget a mixture. Primary:
**bootstrap over seeds, 10,000 resamples within budget, p = fraction of draws with gap ≤ 0.** The
normal-theory σ is reported beside it for comparability with the 2.94, not as the decision statistic.

**4. Bar.** **Resolved** if bootstrap p < 0.0027 (the two-sided 3σ equivalent, matching the pilot's bar).
**Suggestive** if 0.0027 ≤ p < 0.05. **Not resolved** if p ≥ 0.05. Reported as it falls.

**5. Decomposition, declared in advance because it changes what the result MEANS.** The censoring means the
estimand mixes two effects: how often a model clears the detection floor at all, and how far it gets once it
does. Both are reported per budget — `P(clear floor)` and `mean | cleared` — regardless of the headline. If
the trend is carried entirely by the floor-clearing rate, then "SSL buys sensitive distance" is the wrong
description of it and the write-up says so.

**6. What this still cannot settle.** The shape of the label-efficiency curve, which needs more budgets, not
more seeds. And the pilot's lone zero among the 8,000-scratch seeds is *not* revisited here — that stays a
separate declared question about the floor, decided on its own evidence.

**Prediction, recorded so it can be wrong (the last one was).** I expect p < 0.0027 and a gap near +0.15
rather than +0.181 — the pilot was selected for being large enough to notice. I expect the decomposition to
show the effect is **mostly floor-clearing** at 2,000 labels, where all five scratch seeds scored exactly
zero, and mostly magnitude at 8,000.

### RESULT II (2026-09-04): **RESOLVED at 7.39σ — and the effect is not what the headline said it was**

Primary analysis, fresh seeds 5–24 (n=20, none drawn by the pilot), through the pre-registered script
unmodified:

| | gap | bootstrap p(gap ≤ 0) | 99% CI | normal-theory |
|---|---|---|---|---|
| **primary, n=20 fresh** | **+0.2086** | **0.00000** (0/10,000) | [+0.134, +0.278] | 7.39σ |
| secondary, pooled n=25 | +0.2031 | 0.00000 | — | — |

**Verdict: RESOLVED**, against the bar declared before the run (p < 0.0027). The 2.94σ is settled — the
data-wall trend is real. And the pilot was **not** a lucky-high draw as I predicted: the fresh-seed gap is
*larger* (+0.209 vs +0.181), so the winner's-curse correction I expected went the other way.

**THE DECOMPOSITION IS THE FINDING, AND IT CORRECTS MY OWN READING FROM THE PILOT.**

| budget | scratch P(clear) | scratch mean \| cleared | ssl P(clear) | ssl mean \| cleared |
|---|---|---|---|---|
| 2,000 | **0.00** | n/a | **1.00** | 0.267 |
| 4,000 | 0.65 | 0.148 | 0.95 | 0.256 |
| 8,000 | 0.95 | 0.192 | 1.00 | 0.241 |

At **2,000 labels the effect is purely floor-clearing**: from-scratch reaches the 1%-FAR floor in **0 of 20**
runs, SSL in **20 of 20**. There "SSL buys sensitive distance" is the wrong description — SSL is the
difference between a model that functions at all and one that does not.

But at 4,000 and 8,000 **both components are real**, and this is where the n=5 pilot misled me. I reported a
few hours ago that mean-given-cleared was "nearly flat everywhere (0.223–0.240)". At n=20 it is not: scratch
gets **0.148 / 0.192** where SSL gets **0.256 / 0.241**. Five seeds hid a genuine magnitude effect of
+0.05–0.11. **A decomposition run on the underpowered pilot was itself underpowered** — the same lesson as
the headline, one level down, and I stated it as a finding rather than as a preview. The transferable form,
sharpened by `bridge`: **caveating one number does not caveat the numbers computed from it.** The headline
was explicitly flagged as underpowered, and then the same five seeds were split three ways without
re-deriving that the power survived the split. It did not.

**A COMMITTED N4 CLAIM DOES NOT SURVIVE 20 SEEDS.** The gated statement "at the strict zero-FA threshold the
reduced-budget distance is 0 for **both**, a model-strength floor" came from 2 seeds. At 20: scratch is still
0.0000 at every budget, but **SSL reaches 0.0260 at 4,000 and 0.0109 at 8,000**. So the floor is not a
property of reduced-budget models in general — from-scratch never clears zero-FA and SSL occasionally does.
The claim needs the qualifier; the direction of the original conclusion (zero-FA needs near-full-data
strength) stands.

**Prediction scorecard, since one was recorded.** Resolved as predicted ✓. Gap *larger* than the pilot, not
smaller ✗. Floor-clearing dominant at 2,000 ✓. Magnitude-dominant at 8,000 ✓ (scratch clears 0.95 there, so
the remaining gap is magnitude). Artifacts: `ssl_sensdist_seeds20.json`, `ssl_trend_test_seeds20.json`.
