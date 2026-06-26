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
