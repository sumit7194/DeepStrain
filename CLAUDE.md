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
- `JOURNAL.md` (this root) — dated activity log, one entry per session, newest first.
- `<sub>/notes/lab_notebook.md` — raw per-subproject record: pre-registrations, results,
  gotchas, corrections.
- `<sub>/README.md` — methods + decisions (ADR equivalent).
- `CLAUDE.md` (this file) — machine memory / status blocks.
