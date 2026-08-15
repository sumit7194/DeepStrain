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
