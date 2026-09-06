# BlackHole — project memory

> **Repo split (2026-06-13):** these three LIGO-data projects moved here from the
> `SpaceTime/` repo to live with the black-hole physics notes. SpaceTime keeps the
> neural-network / curvature work. `conjecture_machine/` (symbolic GR) is a separate
> standalone repo at the Github root. The concept docs `dimensional_ladder.md`,
> `emergent_dimension.md`, `3plus1_vs_2plus1.md` live in BOTH repos (shared). User
> will `git init` BlackHole separately.

## What this is
The user's black-hole physics thread, made real on public LIGO/Virgo data. Three
deep-learning / data-analysis sub-projects searching real gravitational-wave strain,
plus the conceptual notes that started it (a Brian Cox talk → the information paradox,
holography, entropy). It is a real-data project: **null results are results**, and
nothing is claimed without sensitivity-via-injection and a background that defines
significance.

## Who I'm working with
The user is **not a physicist** and is explicitly relying on me to carry the technical
correctness. He is a **computer engineer** — CS framings (bits, encoding, hashing) land
well. Treat the responsibility seriously; don't simplify into vagueness.

## Standing directive (keep in memory)
**Do real research every time. Verify load-bearing claims with web search before
asserting them — never recite physics from memory and hope.** Especially numbers,
formulas, and dimension-dependent facts. Cite sources in the docs. If something can't be
verified, say so plainly rather than bluffing.

## Model tiering (user directive, 2026-09-05)
**Use the cheapest model that can do the step correctly; reserve Fable for the hardest thinking.**
- **Sonnet** — fetching, reading long docs, summarising, extracting lists, routine grep/inventory passes,
  first-draft boilerplate. Anything where the answer is *in the files* and the job is to collect it.
- **Opus** — judgement calls of moderate depth: surveying what a sibling repo can actually compute, triaging
  a literature sweep, classifying results, drafting a pre-registration from an existing template.
- **Fable (this model)** — synthesis and the genuinely hard steps only: forming a new hypothesis, deciding
  what a measurement licenses, catching a proxy error, resolving a disagreement between two oracles,
  anything where being subtly wrong is expensive and nobody downstream can check it.
- Mechanism: the Agent tool with `model: sonnet|opus`. Launch independent reads in parallel; keep Fable's own
  context for the synthesis. The reports come back as data — verify anything load-bearing before it becomes
  a claim, same as with a sister session's relay.
- This overrides the older "don't use subagents unless asked" note for read/summarise work; it does not
  license delegating a claim's *verification* to a cheaper model when the claim is going into a gate.

## Working style that's landing well
- Explain intuitively first (plain-language CS framing), then the precise statement.
- End sections with **open threads** so the user can pick the next direction.
- Keep docs as **living documents** — extend, don't rewrite from scratch.
- The user flagged some responses got "very very technical" — lead with the analogy,
  put math second and clearly optional. Reflect his idea back accurately before extending.

## Context the user brought in
Arrived from a deep black-hole chat (Brian Cox talk notes): the information paradox,
complementarity, holographic principle, Bekenstein–Hawking entropy **S = A/4**, the M²
entropy law, Planck-area tiles, and spaghettification geometry (stretch-one / squeeze-two,
traceless; the singularity is a *time* not a *place* inside).

## Docs in this repo
- `Black_Hole_Notes.md` — the conceptual black-hole notes.
- `dimensional_ladder.md` *(shared with ../SpaceTime)* — scaling laws across dimensions,
  black-hole horizons across the ladder (perimeter→area→volume law; the holography/entropy
  tie-in), 4+1 extrapolation, the bridge to curvature/gravity.
- `emergent_dimension.md` *(shared with ../SpaceTime)* — "is the extra dimension real?",
  holographic principle, AdS/CFT, ties S=A/4 back to the black-hole chat.
- `3plus1_vs_2plus1.md` *(shared with ../SpaceTime)* — our world vs Flatland.
- `neural_network_holography_experiment.md` — the Hashimoto depth=dimension /
  weights=metric experiment, holography-framed and accessible.
- `make_lightcone_diagrams.py`, `time_orientation.py`, `real_blackholes.py`,
  `nn_spacetime.py` — the figure generators (+ their PNGs: penrose, collapse, many_holes,
  light cones, tipping cones, time orientation).
- `paradox.txt` — the information-paradox notes.

## Sub-project: `primordial_blackhole_search/`
Deep-learning search for **subsolar-mass black hole mergers** (primordial black hole
candidates — below ~1 M☉ no star can make a black hole) in public LIGO strain.
Verified context: LVK O3a/O3b subsolar searches = null (arXiv:2106.08979, MNRAS 524);
MLGWSC-1 = the honest ML-vs-matched-filter benchmark (~70% on real noise, ≤20 s signals
only — minutes-long subsolar signals are the open gap). See its README.md for phases.
- Environment: uv-managed Python 3.12, PyCBC 2.10, GWpy 4.0, torch 2.12 (MPS).
- Conventions: band-limited [50, 1024] Hz everywhere (GWOSC 4k anti-alias filter makes
  PSD near Nyquist unusable); whitening normalized so Σh_w² = SNR²; all sensitivity
  claims from injections into REAL O3a noise; threshold = zero false alarms on held-out
  test segments. GOTCHAS hit: PSD grid must match data length (interpolate to fs/len);
  multiprocessing.Pool deadlocks on macOS → use xargs -P over --job indices.
- Phase 0 ✅ spike (fetch+inject+recover, SNR 81.8). Phase 1 ✅ golden test (SNR match
  0.0% vs pycbc sigma; 0.0 ms timing). Models: SpectrogramCNN 1.17M / ChunkTransformer
  0.82M params. Data: 24×4096 s H1 (16 train / 2 val / 6 test) + 2 L1 coincident.
- **🅿️ v2 ARC PARKED COMPLETE (2026-06-15).** Closing the 45→70% gap: rungs 1–2 (score
  aggregation) negative; rung 3 (learned semi-coherent, V1+V2) negative — both cap ~0.69–0.71 AUC;
  pivoted to **Path G coincidence = the win: H1×L1 gives +1.3–1.5× sensitive distance (~2.3–3.3×
  volume) over single-detector ML at matched FAR.** All refinement levers then squeezed (statistic,
  H1+L1 training — no further gain). **Final blocking point:** subsolar matched filtering needs ≤0.1%
  Mc template spacing (~1,600+ templates) → intractable locally; that wall blocks both a real-MF
  detector and fine 10-ms timing coincidence. Come-back-later = GPU/GCP dense bank + lower FAR, or
  true-waveform supervision. One-screen summary at top of RESULTS.md ("v2 ARC — PARKED COMPLETE").
- **v1 COMPLETE (2026-06-10), all numbers in RESULTS.md.** Headline: CNN reaches
  **41–45% of ideal-MF sensitive distance at zero-FA threshold** (6.8 h real test
  noise), SNR_50≈18.6, flat across subsolar masses. Transformer = honest negative
  (loses to CNN; heavier noise tails). Overfitting fixed via 2× noise + augmentation
  (AUC 0.725→0.777). End-to-end demo: SNR-22 event recovered in H1×L1 coincidence
  (network SNR 36.4, dt err 0.02 ms) with 18 spurious peaks incl. a LOUDER glitch
  (28.6 vs 22.4) rejected by the 15 ms test. Bank-dephasing measured: ±0.01% chirp-mass
  error → −28% SNR (the "why ML trigger" argument, results/bank_mismatch.png).
- v2 ranked: close 45→70% gap (score aggregation across track); FAR→1/month (scale
  noise pool on GCP); ViT-style transformer rematch; H1+L1 training; eccentric corner.
- **v2 rung 1 DONE (2026-06-13): clean negative — the gap is within-window.** Track-score
  aggregation (pbh/aggregate.py boxcar_bank/count_above, sweep+splice protocol in
  scripts/track_eval.py) does not beat the per-window `max` control (≤+0.01 in any mass
  bin), AND neither does the duration-`oracle` ceiling ⇒ overlapping 256-s windows carry
  no independent info; the 45→70% gap lives inside the window. Bycatch: the sweep protocol
  lifted v1's high-mass bin 0.413→0.447 (window-alignment gain; v1 single-window was a bit
  pessimistic). Next: **rung 2 = shorter windows + retraining** (still local). Artifacts
  results/eval_cnn_track.json; pre-registration + table in RESULTS.md.
- **v2 rung 2 DONE (2026-06-13): accumulation fails too — score aggregation exhausted.**
  Retrained `cnn_w64` on 64-s non-overlapping windows (val AUC 0.793 > v1's 0.777; pipeline
  parameterized by window length, v1 untouched) + `sum_track` √k-normalized accumulator
  (pbh/aggregate.py). `sum_track` ≈ `max` ≈ `oracle` (≤+0.007) ⇒ independent per-window
  evidence does NOT accumulate (subsolar SNR isn't track-distributed; early windows below
  the per-window floor). Confound noted: 384 non-overlap noise windows vs v1's 2868 inflates
  all rung-2 stats equally → no FAR-matched "shorter helps" claim. **Both aggregation rungs
  negative; the 45→70% gap needs a sequence-aware/coherent method, not score combination.**
  Full table in RESULTS.md; artifacts eval_cnn_w64_track_w64.json.
- **v2 rung 3 stage 0 DONE (2026-06-13): GATE CLEARED — first non-negative rung.** After
  rungs 1&2 killed score-aggregation, diagnosed the gap as a *representation* problem
  (magnitude spectrograms discard phase). Chose option B (semi-coherent learned bank, MFCNN
  style) over a plain time-domain ResNet port. Stage 0 = oracle ceiling (true-template
  chunked matched filter, scripts/semicoherent_oracle.py): n=8 chunks (8-s coherence) gives
  vetoed fractions **0.66/0.76/0.75 vs cnn_w64 0.41/0.46/0.48** (SNR50 ~11 vs ~18). Sweet
  spot found: n<=4 glitch-limited (chunk-consistency chi^2 veto too weak), n=16 over-chunked.
  **Heavily caveated: it's an ORACLE (true templates) with a LENIENT threshold (6-segment
  noise sample; clean ceiling >1.0 proves the bar is soft) -> the number is optimistic; a
  learned model lands below.** Next = stage 1: build/train the n=8 learned model, measure
  the oracle->learned gap. Full table + caveats in RESULTS.md.
- **v2 rung 3 stage 1 DONE (2026-06-14): DEFINITIVE NEGATIVE — both learned designs cap ~0.69–0.71
  AUC, 0 sensitive distance.** Built SemiCoherentNet (per-chunk 1-D ResNet on whitened strain +
  consistency combiner, 1.24M) + on-the-fly strain-injection dataset, train/eval. Exhausted A/B/C
  before closing: **(A)** documented; **(B) SemiCoherentNetV2** = learnable matched-filter front end
  (64 quadrature templates -> phase-invariant |<d,template>|^2 map, the oracle's statistic learned)
  trained **stable + monotonic, clean plateau val AUC 0.691**, eval **0.000/0.000/0.000**; **(C)**
  definitive V1 full lr=3e-4/20k/20ep showed the earlier "0.69 plateau" was a SHORT-PROBE ARTIFACT —
  at full budget V1 **overfits/goes unstable** (train loss falls 0.50->0.46 while val AUC thrashes
  0.31<->0.62, below chance late), best 0.706, eval **0.000/0.000/0.000**. ⇒ the ~0.69–0.71 wall is
  robust across BOTH natural realizations (not arch quirk, not optimization — V2 converges cleanly
  and still hits it); explicit matched-filter front end only stabilized training, didn't raise the
  ceiling. Stage 0's phase info is real (oracle 0.66–0.76) but neither learned-from-strain design
  realizes it < cnn_w64 0.79. **45→70% gap needs a genuinely coherent/fully-MF method (or true-
  waveform supervision + far more data), not a better strain classifier.** Robust infra survived
  THREE power losses (atomic ckpt/--resume, nohup-detached). Open threads + full A/B/C table in
  RESULTS.md; artifacts semicoherent_v1def.pt/semicoherent_v2.pt + eval_semicoherent_*.json.
- **v2 second pass (2026-06-14): cross-field brainstorm → diagnostics → PIVOT to coincidence →
  FIRST POSITIVE.** Triaged 3 external models' ideas; the convergent diagnosis (weak supervision /
  noise floor) held. (B-threshold) glitch-robust re-threshold REFUTED the single-glitch hypothesis but
  sharpened it: V2 weakly real, dies to a fat noise tail (threshold_robust_eval.py). (F0) bank-mismatch
  gate (bank_oracle.py): coarse bank = 0.000, two-sided squeeze — and coinc_check.py's true-vs-bank
  diagnostic QUANTIFIED it: subsolar needs ≤0.1% Mc template spacing (+1% Mc → SNR dead), ~1,600+
  templates → intractable; extrinsic params irrelevant (orientation-invariant MF). (G0) pivot: coincidence
  kills the NOISE floor not SIGNAL recovery → ride it on the LEARNED model, not the bank. Fetched 8 more
  L1 coincident segments (10 total, 5 overlap H1 test). **(G1) coinc_eval.py: cnn_w64 per-detector +
  H1×L1 time-slide coincidence → +1.3–1.5× sensitive distance over single-det at matched FAR (1.48×
  high-mass; ~2.3–3.3× volume), first positive of the arc.** Pipeline cross-checks v1's SNR50. Caveats:
  coarse window-level coincidence, modest FAR (~1/6h), H1→L1 transfer. Next: **G2 = finer coincidence
  (timing/phase consistency)**. Full tables + caveats in RESULTS.md.
- **v2 path G CLOSED (2026-06-15): +1.37× coincidence is the honest ceiling — every lever squeezed.**
  (G2a) better coincidence statistic: no gain (sum already optimal; min/prod/max+min all ≤). (G2b) H1+L1
  training (build_hl.py + cnn_hl, no eval leakage): val AUC 0.804 > cnn_w64 0.793 but coincidence FLAT
  (0.345/0.375/0.420 ≈ 0.345/0.382/0.428) — higher AUC doesn't help because the operating point is set by
  tail separation, not AUC. (timing) finer 10-ms coincidence is BLOCKED by the same bank-density wall.
  **Headline: single-detector learned subsolar search is noise-floor-limited; H1×L1 coincidence recovers
  ~1.4× distance (~2.5× volume), and that is the ceiling for the learned approach at this scale.** Robust
  infra survived ANOTHER reboot (build_hl resumable; everything finished pre-reboot, only /tmp logs lost).
  Artifacts: cnn_hl.pt, coinc_eval_cnn_hl.json. Remaining = robustness only (lower FAR needs more L1 data).
- **Build C DONE (2026-06-20, L4 GPU VM): coincidence advantage is FAR-ROBUST.** Fetched 24 fresh H1×L1
  coincident O3a segments (26.9 h, no train leakage; fetch_coinc.py), global time-slide background (4000
  N−1=1511 distinct lags → 4.6 yr honest [an earlier "12.3 yr" overcounted by using 4000>N−1 lags; fixed] →
  reach 1/year), 2400 parallel injections (coinc_far.py, 1 worker/seg × 8
  cores, GPU batch-score). Result: coincidence degrades only GRACEFULLY with FAR (1/6h→1/year loses ~15–20%);
  **coinc @1/day = 1.33/1.32/1.43× over single-det floor (reproduces local G1 +1.37×), and even @1/year
  (single-det can't reach it) coinc still beats the single-det floor by ~1.2×.** Gated in verify.sh.
  Workspace ~/deepstrain on alphaludo-l4 (separate from other VM projects). Artifacts: results/coinc_far.{json,png}.
- **Build C-2 DONE (2026-06-20, L4 GPU VM): a LEARNED coincidence statistic BEATS sum — significant + leakage-free.**
  Overturns G2a's "sum is optimal" (that was for *simple* scalar combos). `coinc_learned.py`: cnn_w64 256-d
  penultimate **embeddings** of H1+L1 windows → consistency features `[eH, eL, |eH−eL|, eH·eL]` → small head
  (CoincHead) trained to separate real coincident injections from time-slid noise pairs (it learns whether H1
  *agrees* with L1). vs `sum` on the same embeddings, **HONEST** distinct-lag time-slide bg (see honest-slides
  below). **Learned wins at every honestly-supported FAR, all 3 mass bins, gain grows at stricter FAR**
  (held-out-segments 1/month high-mass 0.320→0.371). **Stress-tests (north star): (1) LEAKAGE** — 3 modes
  (leaky / `--holdout-noise` / gold-standard `--holdout-segments` = train 16 segs, eval 8 UNSEEN segs); gain
  stable across all three (1/month hi 0.369/0.369/0.371) ⇒ not memorization. **(2) SIGNIFICANCE** — bootstrap
  B=500 over 2000 eval inj, **every honest FAR × every mass-bin 90% CI excludes zero, P=1.00** (1/month hi
  +0.050[+0.024,+0.081]). **(3) STOCHASTICITY** — 5 head seeds (--head-seed 0–4, split fixed): learned > sum at
  every seed/bin/honest-FAR ⇒ not a lucky init. **(4) LOWER FAR** — held-out-segments runs out of bg at 1/month;
  the leakage-clean `--holdout-noise` (756 bg windows → honest 1.16 yr) reaches **1/year**: learned still > sum,
  1/year hi Δ+0.048[+0.030,+0.071] P=1.00 (thin); full leaky bg (4.6 yr) agrees robustly (Δ+0.032[+0.018,+0.053]).
  **honest-slides FIX (found while pushing FAR):** bg `sH+roll(sL,k)` has only N−1 distinct circular lags; slides>N−1
  repeats lags + re-injects zero-lag → overcounted T_bg ~5–8× (inflated the reachable FAR). Capped at N−1 in
  coinc_learned.py + coinc_far.py; FAR sweep auto-drops FARs with <1 bg event. Conclusion unchanged, only labels.
  **Net: learned adds a significant +0.02–0.05 sensitive-distance fraction (≈+5–15%) on top of sum's +1.37× over
  single-det, stable to 1/year — first thing to beat sum for subsolar coincidence, leakage-free.** Caveats: cnn_w64
  H1-trained applied to both; 1/year thin(clean)/caveated(robust); this data scale (→1/decade = more data). Gated
  (cross-segment + bootstrap CI>0, honest FAR≤1/month). Segment-tagged cache. Artifacts: coinc_learned_segments.json
  (+ _holdout = clean 1/year, + leaky).
  **Follow-up — base-model COMPOUNDING = honest no.** Ran the learned head on the higher-AUC H1+L1-trained
  `cnn_hl` (--weights cnn_hl; verified leakage-free: cnn_hl train GPS disjoint from all 24 Build-C segs). The
  learned statistic helps on cnn_hl too (sig 3/4 honest FARs, 1/month hi Δ+0.030) so it's base-model-agnostic — BUT
  no compounding: learned-cnn_hl ≈ learned-cnn_w64 within the ±0.02 head-seed spread (G2b's tail-not-AUC logic
  holds). ⇒ the simpler gate-critical cnn_w64 suffices; don't need cnn_hl. Artifact: coinc_learned_segments_cnn_hl.json.
- **N4 DONE (2026-06-26): self-supervised backbone is a data-wall WIN.** `ssl_pretrain.py` (masked-spectrogram
  autoencoder pretrains the SpectrogramCNN conv backbone on 20k UNLABELED noise specs; MSE 1.05→0.75) +
  `ssl_finetune.py` (fine-tune vs from-scratch at reduced labels, 3 seeds, input standardized to SSL mu/sd).
  **SSL wins at every budget, gain ∝ 1/labels: +0.124 val-AUC @1000 labels (0.539→0.663, ~10× seed scatter),
  +0.021 @4000** — the data-wall signature. Caveats: unlabeled pool = labeled set's 20k noise (more O3 noise →
  likely more, a VM extension); mitigates not breaks the wall (0.66<0.79). **Sens-distance follow-up (ssl_sensdist.py):
  the AUC win TRANSLATES to sensitive distance** — at a defined (1%) FAR, SSL +0.278 distance-fraction @2000 labels
  (from-scratch non-functional) → +0.01 @8000 (data-wall signature). At the strict zero-FA threshold both are 0 — a
  model-strength floor (needs ~full-data AUC), not an SSL failure. A real detection win. Gated.
  Artifacts: results/ssl_finetune.json, results/ssl_sensdist.json, models/ssl_encoder.pt.
  **⚠️ SHARPENED 2026-09-04 by a pre-registered 5-seed re-run.** The original used `seeds=2` (one degree of
  freedom), so its data-wall *trend* was never testable — and the gate had guarded it with a fixed 0.05
  margin, a judgement wearing a number. Re-run at 5 seeds against a bar declared **before** the result:
  **gap +0.181 ± 0.062 = 2.94σ ⇒ SUGGESTIVE, NOT resolved** at the declared 3σ. Reported as such rather than
  rounded up — the whole point of writing the bar first, since 2.94 against a bar set afterwards is
  "essentially three sigma". **My prediction that it would clear comfortably was WRONG**: the distance metric
  is far noisier per seed than AUC, so the scatter changed, not the gap. **Headline sharpens: Δ@2000 =
  +0.240 ± 0.042** (5 seeds) vs the committed +0.278 (2 seeds) — agreeing inside 1 SE, so a sharpening not a
  retraction, and the 2-seed value was high as predicted. **CENSORED METRIC caveat:** 8 of 30 per-seed values
  are exactly 0 (below the 1%-FAR floor), so the per-budget distribution is a mixture and a normal-theory SE
  is the wrong error model ⇒ 2.94 is approximate. The zero among the 8,000-scratch seeds would *raise*
  significance if dropped; it was not dropped, because a post-hoc exclusion favouring your own result is
  exactly what a pre-registration exists to forbid. **What would settle it: more seeds** (scatter binds, not
  the gap), not more budgets. Gated at the pre-registered band so a re-run cannot quietly promote it.
  Artifact: results/ssl_sensdist_seeds5.json.
  **✅ SETTLED 2026-09-04 by a second pre-registered run at n=20 — RESOLVED at 7.39σ, and the effect is not
  what the headline said.** Fresh seeds **5–24** (none drawn by the pilot, so the primary is new data rather
  than a re-analysis of the run that selected the hypothesis), fixed n declared in advance, analysis code
  written and committed BEFORE the result existed. **gap +0.2086, bootstrap p = 0.00000 (0/10,000), 99% CI
  [+0.134, +0.278], 7.39σ**; pooled n=25 secondary agrees (+0.2031). **The pilot was NOT a lucky-high draw as
  I predicted — the fresh gap is LARGER** (+0.209 vs +0.181). **THE DECOMPOSITION IS THE REAL RESULT:** at
  2,000 labels from-scratch clears the 1%-FAR floor **0/20** and SSL **20/20** ⇒ there the gain is
  **floor-clearing, not distance** — "SSL buys sensitive distance" is the wrong description, SSL is the
  difference between a model that functions at all and one that does not. **AND IT CORRECTS MY OWN PILOT
  READING:** I had reported mean-given-cleared "nearly flat everywhere (0.223–0.240)" from n=5; at n=20 it is
  **0.148/0.192 (scratch) vs 0.256/0.241 (SSL)** — a genuine +0.05–0.11 magnitude effect that five seeds hid.
  **A decomposition run on an underpowered pilot is itself underpowered**, and I stated it as a finding rather
  than a preview. **AND A COMMITTED N4 CLAIM DIES:** "at zero-FA the distance is 0 for BOTH" was a 2-seed
  statement — at 20 seeds scratch is still 0.0000 everywhere but **SSL reaches 0.0260 @4,000 and 0.0109
  @8,000**, so the floor is not a property of reduced-budget models in general. Gated (61).
  Artifacts: results/ssl_sensdist_seeds20.json, results/ssl_trend_test_seeds20.json.
- **N5 DONE (2026-06-27): triple-detector H1×L1×V1 — honest NEGATIVE, Virgo does NOT help subsolar.**
  `coinc_triple.py` extends the G1 double-coincidence to a 3rd detector (cnn_w64 on H1+L1+V1, 3-way time-slide
  matched-FAR background w/ 8000 livetimes, injections projected onto all 3 via pycbc antenna+delay). Local H1∩L1
  test segs are ALL Virgo duty-cycle gaps (0/5 clean V1) → discovered 20 true H1∩L1∩V1 segments (intersect 3 DATA
  flags), 4 leakage-free fetched by a persistent checkpointing fetcher (GWOSC degraded ~12 h). **(1)** Double H1×L1
  reproduces the win on fresh data (**1.33× over single** — validates G1/Build-C +1.37×). **(2)** Triple = **0.94×
  double — Virgo marginally HURTS.** Mechanism (diagnostic): V1 signal responsiveness (loud−faint mean score)
  **+1.2 vs H1 +5.1 / L1 +7.4 = ~19%** → too insensitive at subsolar to carry signal; summing its near-noise score
  + the higher 3-way threshold degrades the sum. Rules out the learned-triple (no V1 signal to weight → ≈double at
  best). **H1×L1 double-coincidence is the subsolar ceiling.** Gated. Per-segment checkpoint (coinc_triple_rows.parquet)
  survived repeated power losses + service interruptions. Artifact: results/coinc_triple.json.
- **Deep FAR DONE (2026-08-08): 80.5-year background on O4b — 1/decade reached on the Mac, no VM.** `far_deep.py`:
  global time-slides, **background = (N_windows−1) distinct lags × total livetime** (verified: reproduces Build C's
  1692 days exactly), so background **∝ N_segments²**. 100 fresh O4b H1∩L1 segments (leakage-free; cnn_w64 is
  O3a-trained) → 6,200 windows, 113.8 h → **6,199 lags → 80.5 yr = 17× Build C**. Ladder: 1/month 12.340,
  1/year 14.112, **1/decade 16.121** (1/century not measurable: 0.01×80.5<1 event). **Zero-lag = 11.295, below
  even 1/month ⇒ clean null, no subsolar candidate in 114 h of O4b.** Engineering: per-segment atomic score-cache
  checkpoints (never re-fetch a done segment), raw strain purged — **and the purge must clear astropy's download
  cache too** (`gwpy` keeps a 2nd copy, ~0.25 GB/segment, 5.3 GB accumulated; would have exhausted disk ~segment
  80 — caught mid-run, fixed, disk then flat at 19 GB). Gated (45). Artifacts: far_deep.json, results/far_scores/.
- **CNN RESPONSE PROBE DONE (2026-09-02): it responds to BAND-LIMITED NOISE POWER at ~110 Hz — and that one
  mechanism unifies four separate results.** Follow-up to the glitch finding (detector ignores transients).
  `cnn_response_probe.py`, two probes: **(A)** band-power vs score over noise windows peaks **+0.395 at
  107–153 Hz**, falls to **−0.04 at 717–1024 Hz**; **(B)** occlusion (replace a band with the window's median,
  re-score) on top-scoring noise / median noise / injections. **ANSWER: ~73–224 Hz, sensitivity centre
  111–121 Hz, and 97–100% of ALL sensitivity sits below 224 Hz.** **False alarms use the SAME region as real
  signals — profile correlation +0.916, centres 121 vs 111 Hz** ⇒ **the detector is HONESTLY FOOLED**: a
  high-scoring noise window is not a glitch but noise with elevated power exactly where a subsolar chirp
  deposits its SNR (they sweep slowly at low f). **UNIFIES FOUR PRIOR RESULTS:** min/veto cost reach and never
  bought any; tail-norm HURT 2–5%; no glitch morphology in the tail windows; and **H1×L1 coincidence buys
  1.37× while every single-detector cut fails** — all follow if signal and false alarm are the SAME feature
  and the fluctuation is INDEPENDENT BETWEEN DETECTORS. That last one we had never had a mechanism for.
  **Actionable:** >250 Hz contributes ~nothing yet the band runs to 1024 Hz ⇒ ~¾ of the spectrogram is
  ignored capacity; narrowing is untested, logged not claimed. **SIXTH false verdict from my own tooling:**
  the script declared "different band ⇒ separate feature" from an **argmax over two ADJACENT bands** whose
  profiles correlate 0.92 — rewritten to compare profile SHAPE, failure kept in the code. **Scope:** O3a
  (in-domain), 1,240 noise windows + 40 injections; injections are louder (base 25.5 vs 1.5) so only the
  NORMALISED shapes are comparable, which is what the claim uses. Gated. Artifact: cnn_response_probe.json.
- **GLITCH MORPHOLOGY DONE (2026-09-02): the deep-FAR tail is NOT glitch-driven — the veto programme dies
  before it starts, and two old results are retro-explained.** We had called the tail-setting windows "glitchy
  segments" throughout and planned to identify the family so they could be vetoed on principled grounds.
  `glitch_anatomy_morphology.py` measures the largest CONNECTED excess cluster (per-row median normalisation,
  trials-aware threshold so <0.01 false pixels across ~250k, connected-component labelling), **validated on
  synthetics first**: a 10 ms burst → 0.055 s/112 Hz (published blips ~10 ms/~100 Hz), a 4 s tone → 4.04 s/
  64 Hz (published scattered light ~4 s/8–64 Hz), pure noise → no-excess. **RESULT INVERTS THE PREMISE:** all
  3 obtainable tail-setting windows have **NO supra-threshold cluster** (max-excess 17.9–22.4, kurtosis ~0 =
  **0.88× an ordinary window**), while an ordinary window holding a huge transient (**max-excess 4708,
  kurtosis +2404**) scores **−0.484**, i.e. ignored. Extraction verified by re-scoring (all 3 reproduce cached
  scores to 3 dp). **CONFIRMED AT n=1,860** (`glitch_score_correlation.py`, 30 cached O3a segments, no fetch
  needed — the data was on disk): **Spearman(score, max_excess) = +0.091**, top-18-by-score ∩ top-18-by-excess
  = **0 windows**, top scorers with any excess **0.11** vs 0.12 for all, median score of highest-excess windows
  −0.900 vs −0.976. ⇒ **the CNN does not respond to transient excess; there is nothing to veto.**
  **RETRO-EXPLAINS TWO PRIOR RESULTS:** G2a/L2 found `min`/`veto` cost reach and never bought any (0.96–0.99×),
  and tail-normalisation actively HURT 2–5% held-out — both are exactly what a non-glitch-driven tail predicts,
  since a consistency cut can only remove glitches and there are none. Three measurements, one mechanism.
  **CORRECTS OUR OWN LANGUAGE:** "glitchy segments" / glitch-attributed effective-N is WRONG across several
  entries — the effective-N limit stands, the mechanism we ascribed to it does not. **Scope:** n=1,860 is O3a
  (in-domain) and settles the MECHANISM; whether the specific O4b tail windows are glitch-free is settled for
  **3 of 8**, rest behind a GWOSC outage at ~1 kB/s. **Now open and more interesting: what DOES the CNN respond
  to?** Artifacts: glitch_score_correlation.json, glitch_morph_ckpt/.
- **TAIL-NORM TEST DONE (2026-08-22): `sum` survives a REAL objection — equalising the 2x noise-tail
  asymmetry HURTS.** From `bridge` (TheBridge/G3): a statistic whose GAIN varies along a comparison axis
  manufactures structure, and 16x more data bought them ~nothing (systematic variation doesn't average down).
  Our gain axis is fine (**L1/H1 0.97x O4b, 1.10x O3a**) — but measuring the NOISE distributions found a
  bigger, unexamined asymmetry: **q99.9 ratio 1.58x, max ratio 1.97x (H1 12.53 vs L1 6.37), Hill 3.84 vs
  5.57 ⇒ H1's tail is genuinely HEAVIER.** For unweighted `sum` the heavier-tailed detector sets the FAR
  regardless of gain, and G2a/L2 had only ever tested `sum` vs `min`/`veto`, never vs a tail-EQUALISING
  statistic ⇒ the objection was live. `coinc_tailnorm.py` (u_d = −log10 P(S_d ≥ s), sum of rarities) looked
  like **+5.2%/+5.1%/0.0%** in-sample. **`coinc_tailnorm_stress.py` KILLS IT:** fit the map on half the
  SEGMENTS, background on the other half ⇒ **0.983/0.962/0.946x, every bootstrap 90% CI excluding 1 on the
  LOW side (P(>1)=0.02/0.00/0.00)** ⇒ the gain was the map flattening the very realisation whose slides made
  the background — **confounding, not overfitting** (tabula species). **MY OWN MECHANISM FOR THE 1/decade
  ZERO WAS ALSO REFUTED:** I predicted censoring at the map's ceiling u_max=log10(N)=4.65, tying it to the
  zero-lag "deep rungs sit beyond all data" finding; measured **0.0% censored at every rung** (thr 5.26/6.10/
  6.80 vs ceiling 8.71) — the zero was just the artefact fading. **Net: keep `sum`, now vindicated against
  the specific objection rather than untested — and equalising tails COSTS 2–5% reach, so the asymmetry
  carries real information a rarity transform discards.** Gated. Artifacts: coinc_tailnorm{,_stress}.json.
- **ZERO-LAG POPULATION TEST DONE (2026-08-19): independence verified to the 99.98th pctile — and a
  STRUCTURAL limit of time-slides found.** Every deep-FAR result had compared ONE number (the loudest zero-lag
  coincidence) and discarded the other **45,073 zero-lag measurements** in the cache. `far_zerolag_population.py`
  uses them: under H1⊥L1 the zero-lag sum and the slid sum are the SAME distribution, so **the time-slide
  background IS the independence null** — assumption-free, same windows, same noise. **Golden test first:**
  silent under true independence (|z|≤0.66), detects a planted 1% shared-noise component at **z=+6.35**, and
  only at the DEEPEST rung ⇒ correlated noise in a small fraction of windows is invisible in the bulk.
  **RESULT CLEAN:** excess ladder ratios 1.00/1.00/1.03/0.80 (max |z| 1.00 vs bar 3), tail dependence 451 obs
  vs 450.7 exp at the 90th pctile (p=1.000), **KS 0.00214 vs crit 0.00640** ⇒ the zero-lag and background CDFs
  agree to ~0.2% EVERYWHERE. No correlated noise, **no sub-threshold population**. Independence now verified to
  the **99.98th pctile** (±3.2% at the 97.8th) vs "the bulk" before — the assumption the committed 4,120-yr
  ladder depends on, tested near where it is used. **THE STRUCTURAL FINDING: 3 of 4 rungs sit BEYOND ALL
  zero-lag data** (1/year 12.799, 1/decade 14.532, 1/century 16.394 all have **0** zero-lag events at/above;
  zero-lag max is 11.295, verification reaches ~7.5). **The gap CANNOT be closed by sliding** — slides make
  more PAIRS but zero new zero-lag samples; testing independence at 14.5 needs ~a decade of real observing at
  that FAR and we have 0.0914 yr. ⇒ **every factor of depth gained in a time-slide background moves the
  threshold further past the regime where the independence it assumes can be verified.** L2 showed livetime
  doesn't buy independent SAMPLES; this shows it doesn't buy VERIFICATION REACH either, and no computation
  substitutes for observing time. General to time-slide searches, not our pipeline. Scope: constraining rungs
  are the populated ones (±1% @10k, ±3.2% @1k); @10-expected is ±35% and the 99.9th dependence test is
  uninformative (expected 0.0). Gated. Artifact: far_zerolag_population.json.
- **ESTIMATOR AUDIT DONE (2026-08-20): our jackknife understates error bars 4.2× — and effective-N does NOT
  control them.** Measuring the scaling law meant to quantify L2's effective-N story instead audited the
  estimator behind L2's headline. **(1) BIAS 4.2×** (`far_estimator_bias.py`, both estimators on the SAME
  subsets at the SAME n, L2's jackknife reproduced exactly — contiguous 10% blocks, ddof=0): between-subset σ
  2.03/2.44 vs jackknife 0.485/0.570 at n=100/320 ⇒ **4.18×/4.28×**. Mechanism: the jackknife drops 10% of
  segments, but with 727 in play that almost never removes THE dominant glitch, so it looks calm regardless of
  the truth. **Bias is STABLE in n (growth 1.02×) ⇒ L2's RATIO stands** (both endpoints equally understated),
  **but every absolute spread is ~4× too small: quote 1/decade 14.53 ± ~1.7, NOT ± 0.41.** (The committed
  "±1.5" is coincidentally close but was a jackknife RANGE treated as an error bar; bias factor measured at
  n=100/320 and extrapolated to 727.) **(2) COLLAPSE TEST REFUTES THE MECHANISM** (`far_effective_n.py`, 40
  subsets/size, FPC-corrected): pre-registered that σ·√N_eff must be constant across sizes AND rungs if N_eff
  is the controlling variable — a power-law fit can't separate "N_eff controls it" from "n controls it", the
  collapse can. **β = 0.927 ⇒ N_eff grows linearly EXACTLY as L2 said**, but **α = 0.10** (per-rung −0.016
  [−0.114,+0.070] / 0.052 [−0.030,+0.120] / 0.267 [0.173,0.347], all far below the naive 0.5, two consistent
  with ZERO) and **σ·√N_eff scatters 62%, rising 5.7→16.7** ⇒ **more independent loud windows do NOT buy
  precision; N_eff and σ are decoupled.** L2's relative improvement is substantially the threshold MEAN rising
  (1/month 7.07→10.87 over n=20→320 = the known non-convergence), not the error shrinking. **(3) MY
  HEAVY-TAIL HYPOTHESIS REFUTED** (`far_sigma_convergence.py`, 200 draws): three signs suggested infinite
  variance (two runs disagreeing 1.5×, σ rising with rep count), which would have made **no ± quotable at
  all**; direct test says NO — **Hill index 19.7–195** (needs <2), σ flat from m=80 (late slope +0.03),
  σ≈1.85 at n=160. The 1.5× was ordinary small-sample SD noise. **±σ is legitimate.** **TWO OF MY OWN
  INSTRUMENTS RETURNED FALSE VERDICTS, both caught by reading numbers not verdict lines:** the planning number
  computed 2^(1/α) at α≈0 → **10⁵⁵×** (now suppressed unless α>0.15, with reason); the heavy-tail rule
  compared σ's growth over the WHOLE range (ordinary small-m SD bias, present for any distribution) against
  the quantile range's LATE-half drift — apples to oranges, false-positive on all 3 rungs (like-for-like, σ's
  late slope +0.03 is FLATTER than IQ's +0.06). Rule rewritten around the Hill index, failure kept in the
  docstring. **Untouched: 4,120-yr background, 1/century reach, null 4/4** — this is the error bar, not the
  threshold or the search result. Gated. Artifacts: far_estimator_bias.json, far_effective_n.json,
  far_sigma_convergence.json.
- **L2 DEEP FAR DONE (2026-08-19): 4,120-yr background — 1/century reached, precision FIXED, null 4/4.**
  The 08-09 audit's prescription was "more independent loud-noise samples"; background grows as N_segments², so
  `far_deep.py` was run out to **727 O4b segments** (from 100) — **45,074 windows, 801.3 h, 45,073 lags →
  4,119.9 yr (51×)**. Survived **THREE power losses**, every recovery verified **0 corrupt of N** (per-segment
  atomic checkpoints). **HEADLINE (⚠️ CORRECTED 2026-08-20, see next entry): jackknife spread 33–44% → 10–12%** (sd ≈0.41 all rungs) = the audit's central
  complaint answered by data, and the old 16.1±~5 CONTAINS the new 14.532 ⇒ our error bar was honest.
  Ladder: 1/month **11.246**, 1/year **12.799**, 1/decade **14.532**, 1/century **16.394** (new rung).
  **Thresholds FELL 1.1–1.6 and that's a finding:** at fixed FAR the threshold is a fixed QUANTILE (bg-yr and
  pair-count both ∝N², fraction invariant), so the drop proves the 100-segment tail was glitch-inflated —
  exactly the audit's diagnosis. **Zero-lag max 11.295 VERIFIED UNCHANGED** at 7.3× data (same seg 1397232640,
  H1 +12.53 / L1 −1.24) — checked against the cache, not the artifact, because an unchanged value across a 7×
  increase is the shape of a stale read. **Pre-registered before the numbers landed:** 0.0914 yr of zero-lag ⇒
  1/month expects **1.1** background events, so exceeding it is the MEDIAN noise outcome; measured FAR of the
  loudest zero-lag = **11.5/yr (1.05 expected)** ⇒ textbook null, and **claim-capable rungs are 1/year+**.
  Single-det ceilings (maxH1 12.53 / maxL1 6.37) now put **3 of 4 rungs beyond glitch reach**; one-sidedness vs
  loudness reproduces the audit (0% top-25 → 97% beyond rank 2000); **null 4/4**. **`sum` still wins** (min
  0.96–0.98×, halves bg instability 15% vs 34%; veto 0.99–1.00×) ⇒ extends G2a into the 4,000-yr regime.
  **WHAT DIDN'T IMPROVE (the honest half): effective-N is STILL binding** — 1/decade's 425 events come from **8
  distinct H1 windows**, 1/century's 43 from **3**; that "8" is the SAME count the published 1/decade had ⇒ 51×
  the background barely moved the independent-glitch count. **Not converged** (1/decade 17.06@n=60 → 16.78@n=80
  → 14.53@n=727, still drifting). Halves still 25–34% apart. Poisson still 10× too narrow (±0.04 vs ±0.41).
  **THREE TOOLING BUGS caught by pre-flighting at the new scale, all one species — a cap sized for yesterday's
  data volume becomes a SILENT TRUNCATION when data grows:** (1)+(2) **silent rung-drop in TWO scripts** —
  `keep` caps (20k/5k) are exceeded by 1/month's 49,440 events at 4,120 yr, so far_background_validation.py AND
  far_min_vs_sum.py would have returned reports **missing 1/month including from the jackknife**, and `ladder()`
  dropped it with **no message**; fixed (400k/200k, +1/century, loud `!! RAISE keep, not a data limit`).
  (3) **livetime overcount in THREE scripts** — used `n_segments×4096 s` but the 8-s whiten crop leaves **62 of
  64** windows ⇒ 3.23% of quoted time never searched, inflating bg-yr and nudging thresholds ANTI-conservative
  (same species as the honest-slides lag overcount); fixed to `n_windows×64 s`, all numbers use 801.3 h.
  **Net: one rung deeper, 3–4× tighter and honestly quoted (14.5±1.5 not 16.1±5), null 4/4 — and the remaining
  limit is NAMED: independent loud-noise samples, not livetime.** Artifacts: far_deep.json,
  far_background_validation.json, far_glitch_anatomy.json, far_min_vs_sum.json.
- **Deep-FAR AUDIT DONE (2026-08-09): thresholds are ±33–44%, NOT ±0.3 — but the null is 4/4 and WIDER than reported.**
  A "did we miss anything?" pass over the retained score cache (user declined deletion — correct call) became a full
  stress-test of our own headline. **Assumptions HOLD:** H1⊥L1 independence z=−0.38 **p=0.69** (if this failed the whole
  ladder would be biased low), per-segment data-quality corr −0.075 p=0.45. **What BROKE:** "80.5 yr" is *livetime*, not
  independent statistics — all N(N−1) pairs come from 2N window scores, and **1/decade's 8 events trace to just 2 distinct
  H1 windows** (1/year 2, 1/month 4). Jackknife (drop each 10% block) spread **33–44%** vs the ±√k Poisson band's ~±2% =
  **an order of magnitude over-precise**. **It is ONE segment:** 6 of the 8 loudest H1 windows are in seg 59 (gps
  1397232640); dropping it moves 1/decade **16.121→11.261** (diagnostic only — post-hoc removal would be tuning). V5's
  "non-stationarity" is the *same* finding, not a second problem: bulk noise is identical between halves (median −0.814
  vs −0.769), only the glitch cluster's placement differs. **Zero-lag was never a near-miss:** 11.295 = H1 +12.53 /
  L1 −1.24, i.e. the single loudest H1 window with L1 seeing nothing; single-det ceilings (maxH1 12.53, maxL1 6.26) mean
  **any bg >12.5 REQUIRES both detectors ⇒ 1/year + 1/decade rungs are genuine two-sided coincidences, 1/month is
  glitch-reachable.** One-sidedness must be read vs loudness (top-25 **0%** one-sided → top-2000+ 100%); a first quick
  pass claiming "96%" was wrong (sampled every 37th lag ⇒ mid-tail population) and was caught before it reached a claim.
  **min/veto tested and REFUTED on the signal side** — previously recorded as untestable without the purged strain, but
  `o4_sensitive_distance_rows_matched` + `coinc_triple_rows_o4b` already store **per-detector** scores for 4,800 O4b
  injections: at matched FAR `min` = 0.97–0.99× sum's sensitive distance (`veto` 0.99–1.04×) though it *does* halve the
  background instability (25% vs 46%) ⇒ **extends G2a's "sum is optimal" into the deep regime where we'd flagged it
  untested; KEEP sum.** **V8 = the result that survives:** null in **4/4** configs (all-segs/drop-59 × sum/min), each vs
  its OWN threshold, margins **2.0–3.7×** — removing the glitchy segment removes the loudest zero-lag AND the background
  that nearly matched it, self-consistently, because they are the same glitches. **Net: reach stands, null is STRONGER,
  precision must be re-quoted as 16.1 ± ~5.** Deep FAR is limited by independent loud-noise samples, not livetime — which
  is why real LVK searches use signal-consistency vetoes + DQ flags, not raw time-slides alone. Gated (46).
  Artifacts: far_background_validation.json, far_glitch_anatomy.json, far_min_vs_sum.json.
- **N5 O4b RE-TEST DONE (2026-08-06): the Virgo negative REPLICATES across detector generations.** O4b is now
  public (Apr 2024–Jan 2025, H1+L1+V1). **Prereq (`o4_transfer_scout.py`): cnn_w64 transfers to O4b UNCHANGED
  (0.97×)** — per-segment PSD whitening absorbs the era shift, so no retraining and no confound; O4b is 1.41×
  more sensitive in-band, zero-FA threshold 2.111→1.141. `coinc_triple.py` parameterized (`--segs/--tag`), O3a
  artifact untouched. **Result on 8 fresh O4b triple segments: double/single 1.30× (vs O3a 1.33×), triple/double
  0.95× (vs 0.94×) — both replicate to within 3%; Virgo still does NOT help.** **Mechanism MEASURED
  (`o4_asd_compare.py`), not asserted:** median ASD [50,300] Hz — O3a V1 2.8× louder than best LIGO, O4b V1
  **3.2×** louder. Virgo improved 1.14× but LIGO improved 1.29×, so **the gap WIDENED**; V1 responsiveness fell
  19%→12% of H1/L1. Bycatch: O4b has 177 triple-coincident windows in a 30-day probe (vs 20 in six months of
  O3a) and 8/8 segments usable (vs 4–5) — strictly better data in every respect except the one that matters.
  Gated (40). Artifacts: coinc_triple_o4b.json, o4_asd_compare.json, o4_transfer_scout.json.
- **Follow-up A DONE (2026-07-03): the REAL matched-filter benchmark — CNN TIES a realizable dense bank.**
  On the Mac (GPU VM down), `pbh/bankmf.py` (golden-tested MPS FD matched filter, `bank_golden.py`). Full-coherent
  MF is intractable (coherent FF collapses; matches LVK's real 3,452,006-template O4 subsolar bank, arXiv:2412.10951)
  — but the n=8 SEMI-coherent statistic is tractable: `bank_semiff.py` measured recovery vs Mc spacing (0.25%→0.86,
  2%→0.37), quantitatively explaining bank_oracle's old 0.000 and setting ~0.1%/1,619 templates. `bank_dense.py`
  (0.1% bank, 6 real test segs, template-major + mid-segment atomic checkpoint — survived 2 power losses + a Claude
  restart) + `bank_vs_cnn.py` (cnn_w64 on IDENTICAL injections): **real bank MF 0.489 vs CNN 0.472 = 1.03× — a
  statistical TIE** (a CNN forward pass matches a 1,619-template MF bank). Density sweep 83→0.000 (reproduces
  bank_oracle) … 1619→0.489 = the wall quantified; both far below the true-template oracle (0.72) ⇒ **template-bank
  MISMATCH is the dominant loss, not learned-vs-MF.** Co-injection shrank an apparent ~10% win to ~3% (prevented an
  overclaim). Gated. Artifacts: bank_{golden,semiff,dense,vs_cnn}.json.
- **CONFOUND CHECK DONE (2026-08-15): the learned-coincidence gain is NOT a per-segment constant — and our
  "gold-standard" control could never have told us.** From TheBridge Round 12: `tabula` planted a per-realization
  nuisance channel (calibration-offset stand-in, zero dynamical meaning) and their engine ranked it **more
  conserved than the genuine invariant**, passing out-of-sample validation completely ⇒ **held-out validation
  catches OVERFITTING, not CONFOUNDING**, because a nuisance constant generalises flawlessly by being constant.
  **This points at Build C-2:** its positives are an injection's H1+L1 (**same segment**) while its negatives are
  two independent draws from the noise pool (**usually different segments**) — so any per-segment constant in the
  256-d embeddings separates the classes with zero GW content, and `--holdout-segments` **cannot see it** (the
  structure persists in held-out segments). `coinc_confound.py` asks the cheap decisive question first — does the
  channel EXIST? — on **pure noise, no injections**: label pairs same-segment vs cross-segment, train the identical
  CoincHead on identical features. **Result: AUC 0.530** (0.519/0.529/0.542, 3 seeds) = **no usable channel**;
  between/within-segment embedding variance median **0.025 H1 / 0.097 L1**, **0 of 256 dims** with between >
  within. ⇒ **Build C-2's +0.02–0.05 stands.** **Scope:** bounds rather than eliminates — ran on the 5 O3a segments
  in shards_w64_hl carrying both detectors, not Build C's 24 (embedding cache died with the L4 VM); a complete
  answer re-runs coinc_learned with **same-segment negatives**, deferred while L2 has GWOSC. Gated (52).
  Artifact: results/coinc_confound.json.
- **L6 DONE (2026-08-15): the SSL win SATURATES by 2,500 specs — N4's open caveat answered, and it's a NO.**
  N4 left "more unlabeled O3 noise would likely give more" as a hypothesis. `ssl_poolscale.py` measured it from
  data already on disk (deliberately, rather than fetching in competition with the running L2 job): pretrain on
  2.5k/5k/10k/20k unlabeled specs, fine-tune at 1,000 labels, 3 seeds. **Gain is FULLY achieved at 2,500 specs
  (+0.1076) — 8× less data than N4 used — and does not grow** (5k +0.0968, 10k +0.1060, 20k +0.0768); slope
  5k→20k **−0.020**, robustly failing the pre-registered ">+0.02 ⇒ fetch more" bar. **Honest reading: FLAT, not
  declining** — within-pool seed sd **0.019** vs pool-to-pool spread **0.031** (~2 SE), so the 20k dip isn't
  significant and this **bounds** the effect (excludes >~0.03 AUC from 5k→20k) rather than proving zero.
  **Cross-detector NULL:** adding the 6,250 available **L1** specs to 20k H1 gives **+0.0009**, 20× below seed
  scatter ⇒ the pool can't be grown across detectors either. Inventory checked not assumed: shards_w64_hl's
  26,250 includes 20,000 that are the SAME 16 H1 segments (duplicates); only 6,250 L1 are new; 0 pool segments
  in H1 val/test. **Comparability caveat:** our 20k gain (+0.077) < N4's (+0.124) because we re-draw the labeled
  subset per seed where N4 fixed it (our scratch sd 0.024 vs N4's 0.006) — paired across pools so the scaling
  comparison is fair, but not a reproduction of N4's absolutes. **Process note: an early draft REWROTE SpecMAE
  with a different channel ladder** (1→16→32→64→128 vs 1→32→64→128→256), which would have silently produced a
  curve incomparable to N4; now imports SpecMAE/random_mask/train_model from the N4 scripts (and reuses the
  repo's own AUC helper instead of adding sklearn). **⇒ L6b (fetch a bigger pool) is NOT justified.** N4's
  headline stands and is better understood: real, cheap, saturates almost immediately. Gated (49).
  Artifact: results/ssl_poolscale.json.
- **L1 RATIO-FILTER ⚠️ RE-SCOPED 2026-09-06 — the measurements stand, the GENERALISATION does not, and our
  own cost model is what refutes it.** A production O4a subsolar search ([arXiv:2602.12115](https://arxiv.org/abs/2602.12115),
  Kacanja/Soni/Akyüz/Nitz, PRD) built a **25-million-template** subsolar bank, primary 0.1–2 M☉, and states
  the ratio-filter de-chirping framework is what made it feasible — *"an eightfold improvement in template
  processing speed per core"*. That is the regime we closed. **THE RECONCILIATION, and it is not what I first
  guessed:** my first hypothesis was that they use much shorter waveforms (they fix duration to 512 s by
  raising f_low); **measured and REJECTED** — our own f_low = 50 Hz already gives 128 s at 1.0 M☉ and 1024 s
  at 0.2 M☉, comparable, so duration is not the difference. **The actual answer is that our cost model was the
  wrong model:** log N / log K gives 2.4× at K=1024 and 3.0× at K=256, and **reaching 8× would need K = 8
  taps** — i.e. *a model that cannot reproduce the published number at any kernel length was used to conclude
  the published number does not apply to us.* The method paper says plainly what the gain is: it *"transforms
  the operation from a memory-bound FFT into a cache-efficient, compute-bound FIR convolution"* — a
  **memory-hierarchy** effect, invisible to a numpy wall-clock timing where both paths are memory-bound, and
  not represented in an operation-count model at all. ⇒ our 0.94× is a true statement about **our numpy
  implementation**; "the method does not pay for subsolar" does not follow. **The recorded reopening criterion
  ("a signal class needing K ≲ 1,000 taps") is WITHDRAWN — it was derived from the refuted model.** What would
  settle it is an implementation question we never tested: does a cache-blocked compiled FIR beat FFT
  correlation at our N and K? **If it does, the dense-bank wall — the largest blocker in the subsolar arc —
  comes down, and *"does a CNN still tie a matched filter once the bank is adequate?"* becomes answerable.**
  Gate re-scoped, assertions kept so the measurements cannot drift. — *original entry follows* —
- **L1 RATIO-FILTER DONE (2026-08-15): HONEST NEGATIVE — 0.94x, not the published 8x. Dense bank stays blocked,
  now for an UNDERSTOOD reason.** Verified the method at the PRIMARY source (**arXiv:2601.18835**, PRD
  10.1103/k21q-wp8f, *Beyond FINDCHIRP*): with A_t = A_r·R our cross-correlation gives **c_t = c_r (*) IFFT[conj(R)]**,
  one short FIR per target. New `pbh/ratiofilter.py` (kernel by weighted LSQ over f, weight |A_r|²; normal
  equations **Toeplitz** ⇒ O(n log n + taps²)). **Algebra is EXACT** — untruncated kernel reproduces the matched
  filter to **1.000000**, at every separation and even at a WRONG remnant. **But it does not pay here.**
  **THE MECHANISM (the real result):** the method turns O(N log N) into O(N log K), so the gain ≈ log N / log K.
  The published 8x assumes **K≈250 taps** (BNS). **Subsolar needs K≈16,385** (measured, `bank_ratio_regime.py`:
  8,193 taps → 2.4% statistic error; 16,385 → 0.89%, clearing the 1% bar) because these inspirals accumulate
  enormous orbital phase. With N=16.7M that is a **1.6x ceiling**; **measured 0.94x (marginally SLOWER)**.
  ⇒ **the benefit is inversely tied to the kernel length a signal class demands, and subsolar demands the
  longest.** **Memory doesn't rescue it:** kernels are ~31x smaller than stored analytic chunks, but memory was
  never binding — bank_dense already went template-major to work around it; **compute time** binds, and ratio
  filtering doesn't cut it (6 segs @0.01%: 151.9 h direct vs 161.8 h ratio). **Statistic IS faithfully
  reproducible** at 16,385 taps: noise error is **unbiased jitter** (median bias +0.17%, 57% positive ⇒
  threshold safe), signal error 0.89%. **THREE of my own errors caught inside this item:** (1) first golden test
  failed 0.814 — my truncation, not the method; (2) an auto-verdict said "no memory win" on an arbitrary L/16
  cutoff **missed by one tap**; (3) the cost model timed correlation at CHUNK scale (262k) not SEGMENT scale
  (16.7M), inflating 0.94x into a claimed **36x** — `bank_ratio_costmodel.json` is retained but **SUPERSEDED**.
  **Criterion for when it WOULD pay: a signal class needing K ≲ 1,000 taps.** Gated (48) **as a negative**, so it
  cannot be quietly re-inflated. **Bank deliberately NOT built** (~162 h for no speed gain) ⇒ *does a CNN still
  tie a matched filter once the bank is adequate?* stays OPEN, and needs a genuinely cheaper filter.
  Artifacts: bank_ratio_{golden,diag,mcscan,chunked,regime,realcost}.json.
- **Resource accounting on this box (measured 2026-09-04): RSS under-reports an MPS job by ~1.3x, and a
  SINGLE `free` sample is not a state.** Metal allocations live in unified memory and do not fully land in
  RSS, so `ps` under-counts a torch/MPS job — but by **1.33x** (footprint 1.00 GB against RSS 754 MB),
  measured as the difference between two **medians of 8 samples**. ⚠️ **An earlier entry here said 2.19x;
  that was one instantaneous `free` read before a kill against one after.** `free` oscillates by >1 GB on
  this box under a symbolic-algebra workload, so a difference of two single reads is *noisier* than one read,
  not more robust — and I wrote it up as the better instrument while making exactly that error. Note the
  direction: it made the box look tighter than it is, which flattered a decision I had already taken.
  ⇒ **size a run by delta-in-`free` sampled as a median over ~15 s at each end.**
  ⚠️ **AND A SECOND CORRECTION, measured 2026-09-04 while launching the n=20 run: `free` is NOT a pressure
  signal on this box.** With a job running, `free` sits pinned at **0.07–0.12 GB** — sampled ten times over
  40 s — while **swap stays at 0.00 M and the compressor is FLAT at 2.53 GB** and `available` is 5.8 GB.
  That is macOS working correctly: it does not leave memory idle, so low `free` is the normal state under
  load, not a warning. The earlier rule here ("schedule against `free` rather than `available`") would fire
  on every run and mean nothing. **The real pressure signals are SWAP GROWTH and COMPRESSOR GROWTH**, which
  is what `blackhole_watchdog_9f21.sh` in the coordination dir watches (kills our own run by exact PID at
  swap >512 MB or compressor +1 GB over its launch baseline, so a sister project's long job never has to ask).
  Same error as the 2.19x: a rule generalised from one reading of a quantity nobody had watched move.
- **Fleet convention (ansatz, 2026-09-04): no pre-emptive kills on another project's account.** Their runs
  are checkpointed per level; the ask is *"tell me when there is an ACTUAL overlap and I will schedule around
  it."* A resource question with no time axis — *is there room* — is the wrong shape; *is there room WHEN I
  need it* is answerable only by whoever holds the resource, which is why it must be asked rather than
  inferred from a memory reading.
- **Dashboard:** `python3 dashboard.py` (repo root, stdlib only) serves a live run monitor
  over `*/results/progress/*.json` for all three sub-projects; pbh gained `pbh/progress.py`
  (same heartbeat convention as echolib/rdlib). Writes `.dashboard.pid` on start; **stop it
  ONLY by precise PID — `kill "$(cat .dashboard.pid)"` (or by port via lsof). NEVER
  `pkill -f dashboard.py`** — that matches other repos' dashboards in other sessions. build_dataset.py + train.py also heartbeat
  (per-segment / intra-epoch loss) so build and train show live, not just eval.

## Sub-project: `echoes/`
A real-data project searching LIGO public strain for post-merger **gravitational-wave
echoes** (quantum structure at horizons → `bang…beep…beep…` at predicted spacing
`Δt ≈ 8M·log(M/ℓ_P)`, ~0.29 s for GW150914). See `echoes/README.md` for the verified
physics, the Abedi-vs-Westerweck background-estimation controversy, pipeline design,
and ground rules (sensitivity-via-injections before searching; background defines
significance; pre-registered per-event Δt; null results are results).
- Environment: `echoes/.venv` (Python 3.14, gwpy 4.0.1, gwosc 0.8.2).
- **v1 COMPLETE** (2026-06-10): full pipeline working — echolib.py (shared lib),
  01 fetch ✅, 03 injection framework ✅, 04 on-source comb search ✅, 05 background
  + p-values ✅, 06 sensitivity curve ✅, run_event.py (catalog scaling) ✅.
- **v1 results:** sensitivity = blind <1σ, 50% @1.5σ, 100% @≥2σ (p<0.01, real H1+L1
  noise). On-source GW150914 p=0.38/0.97 and GW151226 p=0.40/0.59 (statistics A/B,
  pre-registered) — clean nulls, consistent with Westerweck. Full table + the
  honest v1 sentence in `echoes/notes/lab_notebook.md`.
- Gotcha fixed: GWOSC blocks can have NaN gaps (GW151226 H1 first 133 s) —
  `_longest_finite` crops to the longest valid run before whitening.
- **v2 COMPLETE (2026-06-12): the ML scorer wins ~13×.** Conv-net noise model
  (trained on 100 pairs, judged on 59 held-out) + comb on its residual envelope:
  50% recovery at ≈0.11σ vs v1's 1.5σ, identical p<0.01 harness. Specificity
  control ✓ (irregular spacing fires 6%/2% vs 100% periodic; small low-amp
  leakage caveat). On-source GW150914 still null (p=0.75). Process note: first
  run looked too good (100% at every amp) → extended to 0.1σ + added the
  irregular control BEFORE claiming. Table + caveats in notes/lab_notebook.md;
  plot results/07_ml_vs_comb.png; scorers saved results/07_scorer_{H1,L1}.pt.
- **v3 (2026-06-12):** 13× is family-robust in-band (97–100% @ 0.5σ across
  f0/τ/γ); out-of-band control proved INVALID in the whitened domain (lesson).
- **v4 (2026-06-13):** raw-strain injection. **X3 ✓✓ band-honesty measured**
  (450 Hz: 10% vs invalid version's 100%); X0 calibration validated by
  differencing; X1 backgrounds consistent. **X2: production-path 50% point
  ≈ 1.0σ (pulse reshaping by filter chain) ⇒ the 13× is a same-convention
  claim — not refuted, unverified in production path.**
- **v5 (2026-06-13) FINAL: the fair head-to-head — production-path advantage
  ≈ 1.2× (ML 50% pt ≈ 0.85σ vs comb ≈ 1.05σ; 76% vs 48% @ 1.0σ). The 13× was
  a whitened-domain-convention artifact.** Echoes story complete: modest real
  ML edge + band-honest + family-robust + periodicity-specific + on-source
  nulls. Later: independent background blocks, per-event scorers, FAR scaling.
- **L4 FIRST RUNG ⚠️ RETRACTED 2026-08-22 (was: "coherent NETWORK combination HELPS — 1.12× at 3.2σ"; a paired bootstrap resampling trials AND background gives sd 4.00× larger ⇒ **0.96σ**, gain CI [0.93, 1.48] spanning 1 ⇒ UNDERPOWERED. The analytic bar omitted three terms: a slope denominator from two adjacent grid points, `hypot()` on two PAIRED statistics, and zero uncertainty on a 95th percentile from n_bg=60. **My original "network axis buys ~nothing" prediction was right after all** — the run that disagreed with me was the one needing scrutiny, inverting the lesson I recorded. **The injection-convention finding STANDS** (physical 1.12× vs identical 0.92×), as does the measured geometry. Gated as a retraction. Artifact: 24_l4_significance_stress.json.) — original entry follows — and TWO injection
  bugs found on the way.** From the sweep: arXiv:2512.24730 searches echoes with a phase-marginalized likelihood
  that combines QNMs **coherently across the network**; our statistic is **doubly incoherent** (envelope kills
  phase *within* a detector; scores are *summed* across detectors). `20_coherent_network.py` tests the network
  axis only. **Geometry MEASURED not recited:** cross-correlating the whitened merger gives **−6.59 ms, sign −1**
  vs GW150914's published ~6.9 ms + known anti-alignment (golden test; script refuses to proceed if it fails).
  **MY PRE-REGISTERED PREDICTION WAS WRONG** — I argued the network axis buys ~nothing because coherent and
  incoherent *power* summation give the same network SNR; measured **1.12× at 3.2σ**.
  **TWO BUGS IN OUR OWN INJECTIONS, both found by disbelieving a plausible result:** (1) no inter-detector delay
  or antenna response — same waveform, same GPS time, same polarity in both detectors; (2) `raw_train` draws a
  **fresh random carrier phase per call**, so the two detectors got *different* waveforms (correlation **0.366**).
  ⇒ the injected "echo" was never a common network signal, so any coherent statistic failed by construction.
  **Existing results UNAFFECTED** (the incoherent envelope statistic never uses relative timing or phase — v1,
  v5, ULs stand), but the injection set could not have tested coherent methods. **THE CONVENTION DECIDES THE
  ANSWER:** physical injections (measured delay + polarity + shared phase) → **1.12×, 3.2σ**; the existing
  identical-injection convention → **0.92×, significantly WORSE (2.2σ)**. **Process: 4 runs, 3 retractions** —
  v1 "no gain (0.67×)" agreed with my prediction and was wrong (statistic cancelled the injection); v2 fixed
  delay+sign, still lost (phase differed); v3 gave 1.21× at only **1.4σ** on a grid too coarse to resolve it;
  v4 (finer grid, 120 trials, **significance-gated verdict**) gave 1.12× at 3.2σ — the gain shrank as statistics
  improved, as it should. Verdict logic now refuses "helps" below 2σ. **Scope: network axis only** — the
  within-detector envelope and the paper's full phase-marginalized likelihood remain untested. Gated (50).
  Artifact: echoes/results/20_coherent_network.json.
- **E3 DONE (2026-07-02): per-event ML scorers across the broadened set — all clean nulls.**
  `19_per_event_ml.py`: per-event autoencoder scorer + v2 ML network comb at each event's formula-Δt, for
  GW150914 / GW151012 / GW151226 / **GW250114** (Δt from its verified remnant M_f=68.1/χ=0.68 → 0.2952 s).
  First pass on the tiny own-block background (n=59) threw up GW151012 ML p=0.033 + skipped NaN-cropped
  GW151226 → swapped to the **independent ±hour background** (E2-style own-PSD whitened, n_bg 660–1815):
  GW151012's 0.033 **washed to 0.130** (small-sample artifact; comb never flagged it) + GW151226 rescued.
  **All four events clean nulls under both statistics** (ML p 0.13–0.99). Gated. Artifact: results/19_per_event_ml.json.

## Sub-project: `ringdown_spectroscopy/`
Black-hole spectroscopy on public LIGO data: fit the post-merger ringdown tones (QNMs)
and test the no-hair theorem (each tone must imply the same mass & spin). Verified
status: GW250114 (Jan 2025, SNR~80, data PUBLIC on GWOSC) gave the first clean two-tone
Kerr test (arXiv:2509.08099); the GW150914 overtone start-time controversy (Isi/Farr vs
Cotesta) is the methodological opening; SBI-for-ringdown exists but is young (~4 papers)
— our angle must be sharper than "apply SBI" (candidates: amortize start-time, neural
tone-count model selection, hierarchical stacking). See `ringdown_spectroscopy/README.md`.
- Environment: `ringdown_spectroscopy/.venv` (Python 3.12: gwpy, gwosc, qnm, torch, sbi).
  NOTE: the `ringdown` package needs Python 3.11 exactly — deferred.
- **v1 COMPLETE** (2026-06-10): scripts 01-08 working; results + gotchas in
  `ringdown_spectroscopy/notes/lab_notebook.md`.
- **v1 headline results:** (a) injection referee validated the pipeline at GW250114
  loudness (M=69.8±6.1 vs truth 68; χ=0.69±0.13 vs 0.69) after catching 3 real bugs
  (zero-phase bandpass smears τ — fit whitened-only; injection loudness must be
  calibrated to the event; FFT-seed the restarts); (b) start-time "poisoned choice"
  reproduced on both events (05), GW250114 plateau lands on published (68, 0.69);
  (c) free two-tone fit PROVEN impossible at this SNR by calibration — honest negative
  (06); (d) parameterized no-hair test on GW250114: δ=−0.16, calibrated 2σ=0.72 ⇒
  **overtone consistent with Kerr** (07); (e) SBI/NPE prototype with the start time
  marginalized BY CONSTRUCTION — the novelty angle — trained + coverage-checked +
  real-noise-injection-checked, applied to GW250114 (08).
- Key numbers: 220 @ (68 M☉, χ=0.69) = 251.0 Hz / 4.13 ms; 221 = 245.4 Hz / 1.36 ms
  (3× faster death, ~6 Hz apart — THE difficulty of the field in one line).
- **v2 (2026-06-12) first run:** NPE over (M, χ, δ), start-time marginalized —
  the network IS the no-hair test. R2a ✓✓ Kerr injections in real O4 noise
  unbiased, σ(δ)≈0.14 = **2.6× tighter than classical 0.36**; R2b ✓ violations
  detected at population level (honest shrinkage: medians ~+0.09 for true
  +0.3); **R3 ✓ GW250114 δ = −0.13 [−0.42, +0.33] 90%, Kerr-consistent,
  landing on 07's classical −0.16.** Fix round (150k): M cured (0.88), χ
  slipped to 0.84 — **stable mild overconfidence ~0.84–0.88 across runs, not
  sample-size-curable; post-hoc recalibration = v3 item. v2 CLOSED. Final R3:
  GW250114 δ = −0.16 [−0.45, +0.32] — exactly the classical point estimate,
  Kerr-consistent.** 90k artifacts preserved (*_90k). Scripts:
  09_sbi_nohair.py; rdlib gained progress() + heartbeat().
- **v3 (2026-06-13) CLOSED ✓:** post-hoc temperature recalibration. n=300
  failed (noise-limited — lesson: can't resolve 5% miscalibration with
  σ≈2.4%); n=1000 fix round: T=1.05, held-out coverage 0.91/0.92/0.90 (mean
  0.911 — on target), GW250114 δ = −0.16 [−0.46,+0.33] Kerr ✓ unchanged.
  Ringdown arc complete: amortized + calibration-certified no-hair test.
  Scripts: 10_recalibrate.py (Embed-class pickle gotcha: posterior pickled
  from 09's __main__ needs the class redefined in the loading script).
- v4 shelf: per-param/flow recalibration; simulator realism (+10% mass pull);
  tone-count selection; stacking; SXS injections.
- **v5 δ STACKING ⚠️ RE-SCOPED 2026-08-22: the ARITHMETIC is right, the VALIDATION was prior-dominated.**
  `30_stacking_audit.py`: S2 compares `1/√(Σ1/σᵢ²)` against `σ_single/√N` — the SAME formula, so it tests
  inverse-variance arithmetic, not the method. S3's coverage 1.00 at every N was trivial: the δ prior is
  BoxUniform(−0.5,+0.5), centred EXACTLY on the injected δ_true=0, so returning the prior scores a perfect
  interval — and the gate bar 0.80≤cov≤1.0 ADMITS maximum over-coverage. The validation injections are
  **prior-dominated: σ_single/prior = 0.919**, while the project's own faint-event gate calls a REAL event
  uninformative at >0.88. **Off-centre, stacking makes it WORSE:** at δ_true=0.3 coverage collapses
  **0.95 (N=1) → 0.00 (N=8)** because the interval shrinks by √8 while the 82% shrinkage persists ⇒ √N
  tightening is NOT √N improvement. Intervals 2.5× wider than the estimate's actual scatter. **Scope: the
  VALIDATION protocol, not the NPE at all loudnesses** (09/R2a σ(δ)≈0.14 at GW250114 loudness is real); the
  v5 stress-test conclusion is UNTOUCHED and reinforced. Same artefact v5 caught in real events, sitting
  undetected in the injection validation — **a check evaluated at the prior's centre cannot detect a
  prior-returning estimator.** Gated. Artifact: 30_stacking_audit.json.) — original entry follows —
- **v5 δ STACKING (2026-06-20): METHOD validated ✓, but real multi-event stack NOT achievable (stress-test
  correction).** 12_stacking.py validated the common-δ stacking METHOD: σ(δ) tightens as **√N** on
  informative injections (N=8 → 0.095 vs ideal 0.097, unbiased, calibrated) — solid. BUT the
  stress-test (13_more_events.py, per the robustness north star) cross-checked the NPE on 8 real events
  and found **only GW250114 measures δ** (δ_σ/prior 0.82); all 7 fainter events (GW150914, GW170814, …)
  **return ≈ the prior** (δ_σ/prior 0.93–0.99, χ pulled to ~0.5). ⇒ the v5 "GW250114+GW150914 → 1.3×
  tighter" was a **Gaussian-approx-of-prior artifact** (GW150914 ≈ flat posterior fit as a fake σ=0.27
  measurement); genuine combined constraint ≈ GW250114 alone. Root cause = per-event SNR information wall
  (only GW250114 SNR~80 is loud enough). **Real multi-event δ sharpening parked, honestly.** verify.sh
  gate = √N METHOD + the stress-test (only GW250114 informative). Artifacts: results/13_more_events.json.
- **v4 tone-count selection PARKED (2026-06-15): honest NEGATIVE.** Amortized,
  start-time-marginalized 1-tone vs 2-tone classifier (sbilib.simulate_tonecount + 11_tonecount.py).
  First cut didn't transfer to real data; chased it through a full diagnostic chain — fixed scale
  (norm), noise coloring (real-O4 training), an SNR shortcut (SNR-matched classes), the injection
  convention (whitening reshapes the ringdown — raw-vs-whitened shape overlap 0.48; FD whitening matches
  gwpy to 1.000), and overfitting (60-chunk pool + fresh-per-epoch + early-stop). Only with ALL fixed is
  the model trustworthy — and then it's **honest but WEAK: held-out AUC ~0.61, ECE 0.006 (well-calibrated),
  but it can't confidently call tone count on real events** (GW250114 P(2-tone)=0.32; the earlier 0.69 was
  an overfitting mirage). ⇒ black-box ML tone-count is too weak at this data/SNR scale. Salvage: calibrated
  detectability threshold (overtone SNR≈5 for 50%). Six-attempt table + diagnostics in notes/lab_notebook.md.
  Come-back-later: more data / coherent model, multi-event stacking, or explicit Bayesian model selection.
- **R2 v2 CLOSED (2026-07-02): the PROPER pipeline detects the GW250114 overtone — tone-count resolved.**
  The Py3.11 wall fell: `.venv311` (uv) runs `ringdown` 1.0.0 (Isi/Farr FD coherent pipeline); its pins are
  too loose for 2026 — working set frozen in `.venv311-pins.txt` (jax 0.4.35 / numpyro 0.15.3 / arviz 0.19.0 /
  matplotlib 3.9.4 / scipy 1.14.1). `20_extract_strain.py` (3.12 venv, gwpy→npz) + `21_ringdown_crosscheck.py`
  (3.11): verified targets (GW150914 = docs example; GW250114 = LVK max-L via arXiv:2601.05734). **(a)** GW150914
  validation in-band (M 77.5, χ 0.76). **(b) GW250114 220+221: A221 bounded away from zero (P=0.000,
  A221/A220=1.02 at peak)** = the published result, where our simplified `14` machinery saw nothing ⇒ the parked
  "implementation limit" call POSITIVELY demonstrated; GW150914 overtone marginal (P=0.049, contested-literature-
  consistent). **(c) NPE referee: package M 74.8 [70.4,79.0] / χ 0.729 vs our 09 NPE 76.0 [68.4,85.2] / 0.762 —
  first independent field-standard cross-validation of the NPE arc (package CI nests inside NPE's).** NUTS x64,
  R̂≤1.004, ESS≥950. Gated. Caveat: all peak-start fits carry the R3 early-time systematic; duration fixed 0.05 s.
- **SPIN TRUNCATION ⚠️ CORRECTED 2026-09-04 — the numbers were 4.43%/16.15%, they are 6.36%/18.86%; both
  conclusions survive and are now externally validated. See the cross-check entry immediately below; the
  original entry follows unedited except for this header.** — **SPIN TRUNCATION MEASURED (2026-09-02):
  O(χ²) is ~4% at real remnants, ~16% at EMRI spins — and the answer SPLITS BY CHANNEL.** Asked by the user as a scope question for a scalar-Gauss-Bonnet programme
  working to 2nd order in spin. `31_spin_truncation.py` measures it against exact Kerr 220 from `qnm` instead
  of arguing it. **BOTH standard arguments are wrong, one of them ours:** (a) *"first neglected term is
  χ³/χ² = χ = 0.69 so the error is ~70%"* — **a term ratio is not an error**, it assumes O(1) coefficients and
  they decay; measured error is **4.4%**, wrong by >10×, and it was OUR claim. (b) *"therefore no finite order
  is controlled"* — same unchecked premise amplified; error falls 4.43→2.13→1.07→0.29% over orders 2–6, so the
  series converges. (b) was `bridge` correcting (a) and **inheriting its premise** → PROTOCOL §24.
  **MEASURED (spread over fit ranges 0.15–0.40): χ=0.69 → 4.43% [3.44,5.35]; χ=0.90 → 16.15% [14.66,17.47].**
  χ=0.69 is the universal remnant spin (~1% across NR codes; our GW250114: LVK 0.69 / package 0.730 / NPE
  0.766). ⇒ **RINGDOWN: NOT the binding constraint** (4.4% < our σ(δ)≈0.14 — we withdrew our own
  "disqualifying" call); **EMRI: disqualifying** (16% against phase accuracy over ~10⁵ cycles; remedy is
  non-perturbative in spin, not more terms). **NOT ESTABLISHED — and the test that showed it:** a fit-DEGREE
  sweep (12–28) gives drift 0.000/0.002 at n=2,3 but **1.06 at n=5 and 124 at n=6** ⇒ **only 3 coefficients
  are recoverable** from numerical Kerr frequencies at this precision, so no statement about the asymptotic
  limit is supported. We had reported 0.669/0.731/0.775/0.801/0.812 as a decelerating sequence — **two of
  those five values do not exist.** **THREE METHODS, ONE FAILURE, THREE COSTUMES:** polyfit (|a₅| wandering),
  finite differences (a₆=807312, discarded), Chebyshev (degree-dependent above n=4) — all die at the same
  orders ⇒ a statement about the DATA not the tools. **"A blow-up announces itself; a plateau recruits you"**
  — the blow-up cost minutes, the plateau cost two rounds. **GENERAL FORM: extracting series coefficients from
  numerics is inference with an unquoted error bar; evaluating a closed form at points is verification.**
  Gated (incl. an honesty guard asserting n=5 stays unstable, so nobody later quotes a limit from this data).
  Artifact: 31_spin_truncation.json.
- **SPIN-TRUNCATION CROSS-CHECK DONE (2026-09-04): an outside number found a METHOD ERROR in our own script
  — 4.43% → 6.36%, and the corrected value reproduces a published calibration exactly.** Prompted by
  `bridge` relaying the open item on the sGB spin expansion. Before touching the sGB series, checked 31
  against something we did not tune to: **Pierini & Gualtieri, PRD 106, 104009 (2022) / arXiv:2207.11267
  Sec. IV A** — the second-order-in-spin EdGB QNM paper, i.e. the exact class of calculation the theory
  programme proposes — whose Eq. (44) calibrates against exact Kerr and reports the Taylor expansion within
  1% for **ā ≲ 0.22 at first order, ā ≲ 0.40 at second**. **First run said DISAGREE (ours 0.35/0.50), and the
  tempting explanation was WRONG:** that their number bundles the slow-rotation method's own error (truncated
  angular system, direct integration) on top of series truncation, so theirs is legitimately larger — a
  physically sensible, unfalsifiable story for what was **our bug**. **THE BUG:** 31 obtained coefficients
  with `np.polyfit` **at the truncation order** over [0,hi]; a degree-n least-squares fit is the best n-th
  degree approximation *on that interval* and absorbs higher-order behaviour into its low coefficients, so
  evaluating it outside understates the truncation error. **Fit HIGH in Chebyshev, truncate LOW** returns the
  series' own coefficients. **CORRECTED: χ=0.69 → 6.364%, χ=0.90 → 18.857%** (was 4.43/16.15); the reported
  **"spread [3.44,5.35]% was never a systematic — it was fit noise**, the correct calculation is identical to
  six digits across every extraction range and degree. **VALIDATED: our crossings 0.219 / 0.400 vs published
  0.22 / 0.40**, on the earliest of the four curves their single 1% line actually reports (comparing only
  (022) real is what made the first run say DISAGREE). **THE GOLDEN TEST IS WHY THIS WAS FOUND, and it is
  `tabula`'s precondition relayed the same morning — *a sweep is evidence only once the identical sweep has
  been shown to land correctly on a case whose answer is fixed in advance*:** on 1/(1−x), whose order-n
  relative error is exactly x^(n+1), the low-degree fit gives crossings 0.193/0.345 where truth is
  0.1000/0.2154 — **it fails a case it must pass.** Without the analytic control the disagreement with the
  paper would have been argued about instead of fixed. **CONCLUSIONS SURVIVE, margin shrinks:** ringdown
  6.4% still below σ(δ)≈0.14 (margin 3.2×→2.2×), EMRI 18.9% still disqualifying. **31's own two halves had
  disagreed methodologically all along** — its coefficient-stability section already used the correct
  Chebyshev route while its headline used polyfit. **Gate:** the 3× channel-split ratio bar was calibrated to
  the superseded numbers and tripped at 2.96; loosened to 2× **on the record**, since a ratio bar calibrated
  to a corrected number measures the old number — `chi069 < 0.14` is the assertion that carries the finding.
  **⚠️ WHAT THE CONVERGENCE REFUTATION DOES NOT LICENSE (relayed from `ansatz` via `bridge`, 2026-09-05,
  and it applies to us unchanged):** our "the error falls 6.36→3.38→1.87→1.06→0.79% over orders 2–6, so the
  series converges" refutes the *no-finite-order-is-controlled* claim **only for the analytic part**. A
  contribution non-perturbative in χ is **invisible to an order-by-order method by construction** — it is not
  fine-tuned, it simply cannot appear at any order — so our six-order sequence would look **exactly as it
  does** whether or not one is present. This also weakens my own robustness argument made the same day (that
  restoring a hidden symmetry would require fine-tuning): that reasoning silently assumes analyticity. The
  numbers are unchanged; what they license is narrower. **STILL OPEN, and now correctly scoped:** this
  validates the *Kerr* truncation; the sGB correction's own
  spin series is untouched, and the field's own O(χ²) sGB paper **extrapolates its accuracy from Kerr** ("for
  EdGB gravity **as well**") rather than measuring it — the same proxy step our open item flagged. The
  decisive numbers now exist non-perturbatively: **METRICS, arXiv:2406.11986 / PRD 110, 064019**, sGB QNMs to
  a ≤ 0.85 with ~40 orders in spin. Gated (59). Artifact: 32_spin_truncation_crosscheck.json.
- **ERROR-RATIO STABILITY MEASURED (2026-09-04, same day, prompted by `bridge` asking whether their
  observation survived the corrected extractor): the RISE survives, and we had judged it with the WRONG
  INSTRUMENT.** The observation was that the successive error ratio climbs with order (convergence rate
  degrading toward χ); we had dismissed any ratio involving a₄ because the *coefficient* ratio |a₅/a₄| drifts
  0.048 under fit degree. Measured directly instead of inferred: at χ=0.90 the ratios are **0.7235 → 0.7485 →
  0.7652** (superseded extractor gave 0.696/0.722/0.741 — nearly identical increments, so the rise was never
  an artefact of the bug), with **drift 0.0000 / 0.0012 / 0.0720** over extraction range × degree. ⇒ **2→3 and
  3→4 are DATA; only 4→5 is fit-dominated.** The 3→4 error ratio is **40× steadier than the coefficient ratio
  that supposedly controls it**, because the order-n error depends on the **whole neglected tail**, not on one
  coefficient — a fit that misplaces a₅ typically misplaces a₆ compensatingly. **So this is the first thing in
  the thread that got STRONGER under a better instrument, not weaker** — a two-point rise on stable numbers,
  where we had been treating it as one point. **Terminal form, sharpened by `bridge` against their own
  observation: the DIRECTION is resolved at 21:1 (χ=0.90; 11:1 at χ=0.69) — the gap between the two solid
  ratios against how much they move — the TREND SHAPE is not, and the LIMIT is not.** Two points separate
  rising from falling; rising-toward-a-limit, rising-linearly and rising-then-turning-over are
  indistinguishable from two. 31's "asymptotic limit NOT measurable" guard stands exactly as written. Gated (59; the cross-check's
  block gained assertions rather than a new one) incl. an assertion that the
  rise must persist or be recorded as dead rather than quietly dropped.
- **L5 THIRD-TONE FLOOR DONE (2026-08-15): a third tone is out of reach here — and the two candidates fail for
  OPPOSITE reasons.** The prereq ("read GWTC-5.0 first") **corrected our own [S] note**: we had recorded "GWTC-5.0
  reports the first measurement of three tones"; verified at **arXiv:2510.01001**, GW250114 has strong evidence
  for **(2,2,0)+(2,2,1)** only, a **weak early-time preference** for (2,2,2) at t≲5 M_f, and (4,4,0) with *"SNR
  insufficient for detection"* — the LVK **constrains** a third tone, doesn't detect one. So `29_third_tone_floor.py`
  turns "undetectable" into a number instead of building a chase. **Framework** (reuses 27's linear/quadrature
  setup): a third mode is measurable only via its component **orthogonal to span{220,221}**. At GW250114's remnant
  (ρ_rd 24.9, our convention): **(2,2,2) orth frac 0.139 ⇒ needs A/A220 = 1.45** (an overtone louder than the
  fundamental — unphysical) = **DEGENERACY-limited**; **(3,3,0)/(4,4,0) orth 0.609/0.813** (well separated at
  394/534 Hz) but need A/A220 ≈ 0.15–0.20 = **WEAKNESS-limited**. **Reproduces the LVK pattern from our own
  machinery, with a mechanism for each.** **Reopening criterion: (4,4,0) at a 10% amplitude ratio needs ρ_rd ≈ 38
  = ~1.5× GW250114** (quote the RATIO — 24.9 is our whitened convention). **A hypothesis of mine was REFUTED:** I
  predicted (2,2,2) would be most separable in a SHORT window; the orthogonal fraction instead RISES with window
  length and saturates by ~10 ms (0.003@2ms → 0.139@20ms) — in a short window all three modes look alike and are
  *more* degenerate. The script's auto-written conclusion asserted the opposite and had to be corrected. **The
  variable that matters is START TIME:** (2,2,2)'s measurable SNR falls **7.4× between t_s=0 and 2 ms** while the
  long-lived multipoles barely move — *that* is the mechanism behind LVK's early-time-only (2,2,2) hint, and it
  ties to the R3/B1 early-time systematic. **⇒ L5 closes as information-limited**, consistent with v4's tone-count
  negative (AUC~0.61) and 27/28's finding that the wall is an information limit, now with a quantitative
  reopening number. Caveats: linear/fixed-frequency; NR amplitude ratios deliberately NOT recited (result is
  stated as the ratio REQUIRED). Gated (51). Artifact: 29_third_tone_floor.json.
- **ORTHONORMAL-QNM TEST DONE (2026-08-15): honest NEGATIVE on a published method — the basis carries NO
  detection information.** From the literature sweep: arXiv:2605.03576 orthonormalizes the QNM basis and reports
  GW250114's overtone significance **82.5%→99.9%**, blaming non-orthogonality for "hindering identification of
  subdominant QNMs" — a candidate explanation for our parked **v4 tone-count negative**, so queued as L3.
  **Pre-registered prediction (27's docstring, before running):** with frequencies fixed + whitened noise the
  model is LINEAR in amplitudes, so overtone detection is the nested GLRT = power in V2⊖V1, which Gram-Schmidt
  returns — and the SAME quantity appears in the non-orthogonal fit as A221 weighted by the FULL 2×2 covariance
  block (Schur complement of the 220 block of HᵀH = Gram matrix of the orthogonalized 221 directions) ⇒ ROC must
  be IDENTICAL. **Premise is real** (251.0 Hz/4.13 ms vs 245.5 Hz/1.36 ms, 5.6 Hz apart, overlap **0.863**).
  **(A) `27_orthonormal_roc.py`, 40k trials/class paired: max |ΔAUC| = 0.00000** bit-for-bit (monotone
  transforms ⇒ identical ranks). Only ignoring the covariance costs anything: **ΔAUC ≤ 0.0076**.
  **(B1) `28_orthonormal_prior.py` mismatch:** basis built at WRONG remnant (ΔM ±5/±8, Δχ ±0.06–0.08) ⇒ still
  **0.00000** ⇒ it's the Schur algebra, not a lucky fiducial; "helps under mismatch" hatch CLOSED.
  **(B2) the one lever that DOES move a number:** β=Rθ with R triangular (NOT orthogonal), so "uninformative"
  priors in the two bases are DIFFERENT physical priors. Savage–Dickey on identical data, uninformative limit
  (sd 300): **log₁₀BF shifts −0.454 (2.8× odds) at ΔAUC 1.7e−6**; the AUC gap collapses monotonically with
  prior breadth (−0.051→0.000) while the BF shift persists ⇒ **significance moves, information does not** (pure
  Occam factor). Separate honest finding: at a TIGHT mis-specified prior the bases do differ (0.5645 vs 0.6157)
  but the ordering isn't stable across scales — that's the prior as a modelling choice, not decorrelation.
  **⇒ L3 as a sensitivity play is DEAD, and our v4 tone-count AUC~0.61 wall is NOT explained by
  non-orthogonality — it stands as an information limit.** **Scope stated honestly:** linear/fixed-frequency
  case, so this does NOT show their number is wrong — orthonormalization can still help MCMC conditioning and
  amplitude reporting; what it shows is that decorrelation per se carries no detection information, and that
  **any significance quoted in a rotated basis must state its prior.** Open thread: real fits span a UNION of
  bases as (M,χ) vary — B1 covers any single wrong basis, not a union. Gated (47).
  Artifacts: 27_orthonormal_roc.json, 28_orthonormal_prior.json.
- **B (follow-up) DONE (2026-07-03): package start-time referee + NPE loop closed; nonlinear-QNM honestly parked.**
  **B1** (`22_starttime_sweep.py`): GW250114 220+221 across 9 start offsets (0–16 t_Mf) — the overtone is
  significant from the peak (P(A221≈0)=0.000) and damps by ~5.4 ms (→0.059), and the peak-start mass is biased
  HIGH (74.7 vs true 68.1, +10%), drifting −8.8 M☉ later ⇒ the R3 early-time systematic **independently
  reproduced by the coherent package**. **B3** (`23_npe_package_loop.py`, synthesis): the NPE (76.0/[68.4,85.2])
  agrees with the package (74.8/[70.4,79.0], CI nested) AND sits at ~0 t_Mf (the peak) in the sweep ⇒ the NPE
  **inherits the peak systematic** (+7.9 M☉), not bias-free from marginalizing t0. **B2** (nonlinear (4,4) quadratic
  mode, arXiv:2601.05734) PARKED — the vanilla package fits only linear (2,2) QNMs; a fair test needs multi-multipole
  + 2·f220 frequency-locking (Wang & Ma's custom pipeline), so a vanilla fit would be a false negative (R2 discipline).
  Gated (35). Artifacts: results/22_starttime_sweep.json, 23_npe_package_loop.json.

## Status & ground rules
- **All three arcs are PARKED COMPLETE** (FOCUS DIRECTIVE in ../SpaceTime: curvature
  only until mined out). Green gates, shelf lists in each lab notebook. Revisit when
  the curvature project is done.
- **Regression gate: `./verify.sh`** — asserts echoes (07) + ringdown (09/10) + pbh
  (CNN sensitivity, eval_cnn) headline artifacts against saved results. Run after any
  change here; a result isn't real until the gate is green. (The `.venv` folders were
  decoupled from SpaceTime on 2026-06-13 — all internal paths rewritten to BlackHole;
  gate green, activation works, no SpaceTime references remain.)
- **Engineering standards:** `.claude/skills/ai-coding-standards/SKILL.md` governs all
  code work (search-before-write, smallest diff, verify-before-done with fresh output,
  no narration comments, dependency restraint, decisions recorded in sub-project
  README/lab notebook, status blocks here updated when milestones land).

## Documentation taxonomy (mirror of SpaceTime's)
- `ROADMAP.md` (this root) — forward-looking next moves + guardrails (P1/P2 + carried blockers).
- `RELATED_WORK.md` (this root) — living literature record: what the field has published around our three
  arcs, what it means for us, and the **L1–L7 long-horizon project list**. Entries are marked [A] (abstract
  read) vs [S] (search snippet — verify before load-bearing use). Swept 2026-08-15; re-sweep before citing
  it as current. **Standing rule recorded there: effort is never a reason to drop an item** — long-running
  jobs are normal, an item leaves only when done or measured impossible.
- `JOURNAL.md` (this root) — dated activity log, one entry per session, newest first.
- `<sub>/notes/lab_notebook.md` — raw per-subproject record: pre-registrations, results,
  gotchas, corrections.
- `<sub>/README.md` — methods + decisions (ADR equivalent).
- `CLAUDE.md` (this file) — machine memory / status blocks.
