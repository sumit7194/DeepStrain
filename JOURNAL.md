# Journal — activity log (BlackHole: LIGO-data projects)

*One entry per working session, newest first. Lab-notebook-level detail stays in each
sub-project's `notes/lab_notebook.md`.*

> **Repo split 2026-06-13:** these three projects (echoes, ringdown_spectroscopy,
> primordial_blackhole_search) moved here from `../SpaceTime/`. The full pre-split
> narrative — including the interleaved night-shift sessions that wove black-hole and
> curvature work together — is archived in `../SpaceTime/JOURNAL.md`. This journal
> carries black-hole work forward from the split.

---

## 2026-06-14 — pbh v2 rung 3 stage 1 CLOSED: definitive negative (A/B/C exhausted)
- Finished the "be sure of the hurdles" pass before concluding. **(B) SemiCoherentNetV2** —
  learnable matched-filter front end (64 quadrature templates → phase-invariant |⟨d,t⟩|² map,
  the oracle's statistic, learned). Capacity gate passed; full run lr=3e-4/20k/20ep was **stable,
  monotonic, clean plateau val AUC 0.691** (no thrash) — eval **0.000/0.000/0.000**. **(C)**
  definitive original-arch run revealed the earlier "flat 0.69 plateau" was a **short-probe
  artifact**: at full budget V1 **overfits/destabilizes** (train loss 0.50→0.46 smooth while val
  AUC oscillates 0.31↔0.62, below chance on most late epochs), best 0.706, eval **0.000/0.000/0.000**.
- **Verdict:** the ~0.69–0.71 AUC wall is robust across BOTH natural learned realizations — not an
  architecture quirk, not optimization (V2 converges cleanly and still hits it). The explicit
  matched-filter front end only made training better-behaved; it did not raise the ceiling. Stage 0
  proved the phase info is recoverable (oracle 0.66–0.76 ≫ cnn_w64 0.41–0.48), but neither
  learned-from-strain design realizes it (<cnn_w64's 0.79 → 0 sensitive distance at zero-FA).
  **The 45→70% gap needs a coherent / fully-matched-filter method (or true-waveform supervision at
  much larger scale), not a better classifier on whitened strain.** Open threads banked in RESULTS.md.
- **Infra:** survived a THIRD power loss mid-C — resumed from the epoch-5 atomic checkpoint with zero
  work lost (dashboard relaunch gotcha noted: root dashboard runs under system `python3`, not `.venv`).

## 2026-06-13 — pbh v2 rung 3 stage 1 (learned semi-coherent model): NEGATIVE so far
- Built the learned realization: SemiCoherentNet (per-chunk 1-D ResNet on whitened strain +
  consistency combiner, 1.24M), on-the-fly strain-injection dataset from a 2500-waveform pool,
  train/eval + self-healing overnight runner. Overfit gate passed (capacity OK).
- First full run (sweep winner lr=1e-3, 16 epochs) UNSTABLE: val AUC peaked 0.687 @ep0 then
  collapsed/thrashed to ~0.35; eval **0.000/0.000/0.000** vs cnn_w64 0.41/0.46/0.48, oracle
  0.66/0.76/0.75. Exhaustive LR + grad-clip probing: only lr=3e-4 stable (flat ~0.69), all
  higher LRs collapse (below-chance cliffs = exploding gradients), clipping doesn't fix.
  ⇒ ~0.69 is an ARCHITECTURE ceiling < cnn_w64's 0.79 ⇒ ~0 sensitive distance. Stage-0 phase
  info real but this learned design can't realize it.
- **Infra win:** survived TWO power losses + repeated session-kill of background tasks. Lessons:
  long runs must be nohup-detached (not harness background tasks) [[nohup-long-running]];
  per-epoch atomic checkpoint + --resume; ps RSS undercounts MPS memory (use Activity Monitor).
- Not closing yet (be sure of the hurdles): (B) matched-filter front-end architecture,
  (C) full lr=3e-4/20k definitive run. Table in pbh RESULTS.md.

## 2026-06-13 — pbh v2 rung 3 stage 0 (semi-coherent oracle): GATE CLEARED (first non-negative)
- After rungs 1&2 ruled out score aggregation, diagnosed the 45->70 gap as a *representation*
  problem (magnitude spectrograms discard phase; MLGWSC ~70% used time-domain). Chose option B
  (semi-coherent learned bank, MFCNN-style) over a plain 1-D ResNet port — it attacks subsolar's
  bank-dephasing pathology and scales to long signals.
- Stage 0 = oracle ceiling before building. First smoke gave 0% -> diagnosed as REAL-noise glitch
  domination (2310-sigma glitch; not a bug — synthetic noise gives the correct chi^2_2n ~33).
  Added the chunk-consistency chi^2 veto (PyCBC newSNR); n=1 can't veto by construction.
- Result (1500 inj, 3 ceilings): **n=8 chunks vetoed = 0.66/0.76/0.75 vs cnn_w64 0.41/0.46/0.48**
  (SNR50 ~11 vs ~18). Sweet spot n=8 (coherence vs glitch-robustness). First rung not ruled out.
- **Honest caveats banked:** oracle = true templates (learned model lands below); threshold is
  lenient (6-segment noise; clean ceiling >1.0 proves it) so the absolute number is OPTIMISTIC.
  Decision: build stage 1 (learned n=8 model), measure the oracle->learned gap.
- Process: never `pkill -f` (killed another session's dashboard); dashboard now writes
  `.dashboard.pid`, stop via `kill "$(cat .dashboard.pid)"`. Memory saved.

## 2026-06-13 — pbh v2 rung 2 (shorter windows + accumulation): also negative
- **Score aggregation exhausted.** Parameterized the whole pbh pipeline by window length
  (config/spectrogram/sweep/build_dataset/train/track_eval), v1 path byte-identical + gate
  green throughout. Built 64-s shards (40k/2.5k, 128×63), retrained `cnn_w64` (val AUC 0.793
  > v1 0.777), added `sum_track` (√k-normalized summed logits) to pbh/aggregate.py.
- **Result:** `sum_track` ≈ `max` ≈ `oracle` (≤+0.007) — even the duration-oracle can't beat
  the single best 64-s window ⇒ independent per-window evidence does not accumulate. Decision
  (c). Confound flagged: 384 non-overlap noise windows vs v1's 2868 → no FAR-matched "shorter
  helps" claim. Both aggregation rungs negative; gap needs a sequence-aware/coherent method.
  Pre-registered before building; table in pbh RESULTS.md. cnn_w64.pt + eval_cnn_w64_track_w64.json.
- **Dashboard fix:** build_dataset.py (per-segment) + train.py (intra-epoch loss sparkline)
  now heartbeat, so build/train are visible live, not just eval.

## 2026-06-13 — pbh v2 rung 1 (track aggregation) + repo decoupling + dashboard
- **Decoupled from SpaceTime:** all three `.venv` folders had internal paths rewritten
  `SpaceTime/`→`BlackHole/` (pyvenv.cfg, activate, ~400 console shebangs); skill re-adapted
  to BlackHole; code had no runtime coupling. Gate green through the rewritten venvs.
- **verify.sh** gained a pbh assertion (eval_cnn `mf_distance_fraction`) + now a track
  assertion (max-control reproduces v1 threshold).
- **pbh v2 rung 1 — track-score aggregation = clean negative.** New `pbh/aggregate.py`
  (max / boxcar_bank / count_above) + `pbh/sweep.py` (shared window grid, refactored out of
  evaluate.py) + `scripts/track_eval.py` (full-signal segment injection → spliced sweep →
  masked stats, atomic+resumable). 1500 injections: no statistic beats the per-window `max`
  control (≤+0.01), and the duration-`oracle` ceiling doesn't either ⇒ the 45→70% gap is
  *within the 256-s window*. Bycatch: the sweep protocol lifted v1's high-mass bin
  0.413→0.447 (alignment gain). Pre-registered before running; full table in pbh RESULTS.md.
  Next is rung 2 (shorter windows + retrain, still local).
- **Dashboard:** root `dashboard.py` (stdlib HTTP, live UI) over `*/results/progress/*.json`;
  pbh gained `pbh/progress.py`. Watches all three sub-projects.

## 2026-06-13 — repo split: black-hole projects moved out of SpaceTime
- echoes/, ringdown_spectroscopy/, primordial_blackhole_search/ moved from SpaceTime
  to this BlackHole folder, joining the existing black-hole concept notes. New CLAUDE.md,
  verify.sh (echoes 07 + ringdown 09/10 asserts), and a copy of the ai-coding-standards
  skill set up here. The three shared concept docs (dimensional_ladder, emergent_dimension,
  3plus1_vs_2plus1) are copied into both repos. Both repos will be `git init`'d separately.
- **venvs decoupled from SpaceTime (2026-06-13):** all internal absolute paths in the
  three `.venv` folders (pyvenv.cfg, activate scripts, console-script shebangs) rewritten
  `SpaceTime/`→`BlackHole/`. Zero SpaceTime references remain, the gate passes all-green
  through them, the science stacks (torch, gwpy, pycbc) import, and `source activate` now
  resolves to the BlackHole path. The projects are independent of SpaceTime.

---

## State at split — the three completed arcs (detail in each notes/lab_notebook.md)

**echoes/ — v5 COMPLETE (2026-06-13).** Post-merger GW-echo comb search on real LIGO
strain. The honest final number: production-path ML advantage ≈ **1.2×** over the plain
comb (ML 50% point ≈ 0.85σ vs comb ≈ 1.05σ). The earlier "13×" was a whitened-domain
convention artifact, caught and corrected. Story: modest real ML edge + band-honest +
family-robust + periodicity-specific + on-source nulls (GW150914 p=0.75). Shelf:
independent background blocks, per-event scorers, FAR scaling.

**ringdown_spectroscopy/ — v3 COMPLETE (2026-06-13).** No-hair test via amortized SBI
(NPE over M, χ, δ with start-time marginalized by construction — the network IS the
test). σ(δ) ≈ 2.6× tighter than classical; v3 added post-hoc temperature recalibration
(T=1.05, held-out coverage mean 0.911). GW250114 δ = −0.16 [−0.46, +0.33], Kerr-consistent,
landing exactly on the classical point estimate. Shelf: per-param recalibration, simulator
realism, tone-count selection, stacking, SXS injections.

**primordial_blackhole_search/ — v1 COMPLETE (2026-06-10).** Deep-learning search for
subsolar-mass mergers (primordial BH candidates) in public LIGO strain. CNN reaches
41–45% of ideal matched-filter sensitive distance at a zero-false-alarm threshold (6.8 h
real noise), flat across subsolar masses; transformer an honest negative. End-to-end
H1×L1 coincidence demo recovers an SNR-22 event and rejects a louder glitch. Shelf:
close the 45→70% gap, FAR→1/month, ViT rematch, H1+L1 training, eccentric corner.
