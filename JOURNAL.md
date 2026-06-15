# Journal — activity log (BlackHole: LIGO-data projects)

*One entry per working session, newest first. Lab-notebook-level detail stays in each
sub-project's `notes/lab_notebook.md`.*

> **Repo split 2026-06-13:** these three projects (echoes, ringdown_spectroscopy,
> primordial_blackhole_search) moved here from `../SpaceTime/`. The full pre-split
> narrative — including the interleaved night-shift sessions that wove black-hole and
> curvature work together — is archived in `../SpaceTime/JOURNAL.md`. This journal
> carries black-hole work forward from the split.

---

## 2026-06-15 — ringdown v4 tone-count: PARKED, honest negative (6 attempts, full diagnostic chain)
- Pivoted from pbh (parked) to a new ringdown thread: an amortized, start-time-marginalized AI to count
  QNM tones (1 = 220 only, 2 = 220+221 overtone) — addressing the live GW150914 overtone controversy,
  whose crux is start-time dependence (which the SBI infra marginalizes by construction).
- First cut didn't transfer to real data. Chased it down a clean diagnostic chain: (A) scale → norm;
  (B) noise coloring → train on real O4 noise; DIAGNOSTIC → caught a "loud⇒2-tone" SNR shortcut;
  (C') SNR-matched classes → removed it; DIAGNOSTIC → whitening reshapes the ringdown (raw-vs-whitened
  shape overlap 0.48; built a fast FD whitening matching gwpy to 1.000); (D) injection-convention-matched
  training → transfer pathology GONE (GW250114 read 2-tone) BUT model overfit the 14-chunk pool;
  finally 60 chunks + fresh-per-epoch + early-stop → overfitting fixed.
- **Verdict (honest NEGATIVE with a now-trustworthy model):** calibrated (ECE 0.006) but WEAK — held-out
  AUC ~0.61, can't confidently call tone count on real events (GW250114 P(2-tone)=0.32; the earlier 0.69
  was an overfitting mirage). Black-box ML tone-count is too weak at this data/SNR scale. Salvage: a
  calibrated detectability threshold (overtone SNR≈5 for 50% detection). The diagnostic chain itself is
  the contribution. Six-attempt table in ringdown notes/lab_notebook.md; survived 3 power losses (all
  artifacts disk-cached/reboot-safe). Come-back-later: more data/coherent model, multi-event stacking,
  or explicit Bayesian model selection with a real noise model.

## 2026-06-15 — pbh path G CLOSED: +1.37× coincidence is the ceiling (G2a/G2b negatives)
- (G2a, coinc_stat.py) better coincidence statistic: no gain — `sum` already optimal (min/prod-prob/
  max+min all ≤ it). (G2b, build_hl.py + cnn_hl) H1+L1 training: built a 64-s H1+L1 spectrogram set
  (self-contained, resumable, no eval leakage), trained cnn_hl → val AUC 0.804 (> cnn_w64 0.793) but
  coincidence FLAT (0.345/0.375/0.420 ≈ cnn_w64's 0.345/0.382/0.428). Higher AUC didn't carry to the
  operating point (tail-separation-limited, not AUC-limited). Finer 10-ms timing coincidence stays
  blocked by the bank-density wall.
- **Path G headline:** single-detector learned subsolar search is noise-floor-limited; H1×L1 coincidence
  recovers ~1.4× sensitive distance (~2.5× volume) — the honest ceiling for the learned approach at this
  data/compute scale. Remaining work is robustness only (lower FAR ← more coincident data).
- **Infra:** survived ANOTHER reboot (~11:56). build_hl.py resumable; build+train+eval all finished
  pre-reboot (cnn_hl.pt 05:24, coinc_eval_cnn_hl.json 05:52), only /tmp logs lost. Note: the g2b_chain.sh
  orchestrator does NOT survive a reboot (a nohup bash dies with it) — fine here since work completed first,
  but truly reboot-proof automation would need launchd/cron. Saved a documentation-discipline memory.

## 2026-06-14 — pbh v2 second pass: cross-field brainstorm → coincidence PIVOT → FIRST POSITIVE (+1.3–1.5×)
- Took 3 external models' brainstorms, triaged them; convergent diagnosis (weak supervision / noise floor)
  held. Ran the cheap diagnostics first: glitch-robust re-threshold (threshold_robust_eval.py) REFUTED the
  single-glitch hypothesis but sharpened it (V2 weakly real, dies to a fat noise tail).
- F0 bank-mismatch gate (bank_oracle.py + coinc_check.py): a coarse template bank gives 0.000, and the
  clean true-vs-bank diagnostic QUANTIFIED why — subsolar needs ≤0.1% Mc template spacing (+1% Mc → SNR
  dead), ~1,600+ templates → intractable locally; extrinsic params (sky/inclination) are irrelevant (the
  quadrature MF is orientation-invariant). This is why F0 was flat-zero.
- **The pivot (G0):** coincidence kills the NOISE floor, not the SIGNAL-recovery (bank-density) problem →
  ride it on the LEARNED model (cnn_w64), not the broken bank. Fetched 8 more L1 coincident segments
  (10 total; 5 overlap H1 test).
- **G1 — FIRST POSITIVE (coinc_eval.py):** cnn_w64 per-detector + H1×L1 coincidence with a TIME-SLIDE
  background (18,910 accidentals from 5 segments). At matched FAR, two-detector agreement gives
  **+1.3–1.5× sensitive distance** over the single-detector ML search (1.48× high-mass → ~2.3–3.3× volume).
  Pipeline cross-checks v1's per-detector SNR50 (~18.6). After a long run of honest negatives, the
  coincidence lever finally moved the number. Caveats: coarse window-level coincidence, ~1/6h FAR, H1→L1
  transfer. Next: G2 = finer coincidence (timing/phase). Full tables in RESULTS.md.

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
