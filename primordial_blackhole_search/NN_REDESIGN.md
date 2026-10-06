# Rethinking the networks — brainstorm + plan (2026-10-06)

> Living document. Origin: a first-principles brainstorm with the user ("we were always short on data — can
> maths or architecture get more out of it?"), cross-checked against the `tabula-geometrica` session's own
> NN-architecture brainstorm (discussion shared, work kept separate). **Nothing here is built yet.**
> Status per item is at the end. Detailed numbers live in [RESULTS.md](RESULTS.md); this file is the map.

---

## 1. The hard limits (what no trick can do)

Plain version: a signal's detectability is a fixed amount of energy weighed against the noise. Fourier,
wavelets, any reversible change of view only *moves* that energy around; "amplifying" multiplies the
noise by the same factor. So there is no transform that makes a quiet signal louder.

- **Parseval:** SNR² = ∫ |h(f)|² / S_n(f) df is preserved by every unitary transform.
- **Data-processing inequality:** no function of the data carries more information about the signal than the data.
- **Neyman–Pearson:** for a known waveform in Gaussian noise the matched filter is the optimal detector.
- **Trials:** the number of *distinguishable* signal shapes sets the false-alarm penalty, whatever algorithm searches them.

Only three honest levers remain: **lose less** of the SNR we have, **lower S_n**, or **collect more signal**
(time, detectors, events). Every idea below is one of these. The optimality theorems all assume Gaussian,
stationary noise and a known waveform family — real gains live where those assumptions fail.

## 2. What we actually have (model cards)

**Shared subsolar input pipeline:**
- **Data:** GWOSC O3a strain at 4096 Hz, whitened with ONE PSD per 4096-s segment.
- **Picture:** spectrogram with 1-s STFT and 0.5-s hop, 128 log-spaced bins over 50–1024 Hz, `log1p` of power. A 64-s window becomes a 128×63 image.
- **Detector:** H1 only.
- **Training data:** 40k examples from 16 H1 segments (about 18 h of noise). TaylorF2 injections, m₁, m₂ ∈ [0.2, 1.0] M☉, SNR 5–30.
- **Training recipe:** BCE, AdamW, 16 epochs.

| Network | Input | Architecture | Result |
|---|---|---|---|
| SpectrogramCNN `cnn_w64` | 128×63 spectrogram, H1 | 4 conv blocks 32→256, GAP, 1.17M | AUC 0.79, distance fraction 0.41–0.48 |
| `cnn_hl` | + L1 windows as extra single-channel examples | same | AUC 0.80, coincidence flat |
| ChunkTransformer | spectrogram → 16 chunk tokens | 4-layer transformer, 0.82M | lost to CNN (AUC 0.76) |
| SemiCoherentNet V1 | raw whitened strain (262,144 samples) | 8 chunks × 1-D ResNet + combiner, 1.24M | AUC 0.71, distance 0 |
| SemiCoherentNet V2 | raw strain | learnable MF front end (64 quadrature pairs), 0.68M | AUC 0.69, distance 0 |
| SpecMAE (SSL) | 20k unlabelled noise specs, 60% masked | masked autoencoder → CNN backbone | big win at low labels; saturates by 2.5k specs |
| CoincHead | 256-d CNN embeddings of H1 and L1 | [eH, eL, \|eH−eL\|, eH·eL] → MLP | +5–15% over `sum` |

Other results that bear on the redesign:
- Score-sum coincidence gives ×1.37; adding V1 gives 0.94×.
- Adequate MF bank vs CNN on identical injections: 0.514 vs 0.484.
- The ringdown NPE is the only network that takes both detectors as input channels. It is trained only on white Gaussian noise.
- The echoes autoencoder was trained on about 5 minutes of noise per detector.

## 3. The diagnosis

**Before any subsolar network sees data, the pipeline throws away three things:**

1. **Phase.** A magnitude spectrogram is irreversible. Incoherent power summation obeys the textbook
   **N^(1/4) sensitivity law** ([LIGO-T2100266](https://dcc.ligo.org/LIGO-T2100266/public)). That is why the CNN,
   the transformer and the score aggregators all land at the same place: different architectures on the same lossy input.
   Unlike re-encoding a Ludo board, which changes learnability but not information, our encoding destroyed information.
2. **The second detector.** Every network is single-detector. Fusion happens only at score/embedding level, though
   the signal is phase-coherent between detectors and our measured false alarms are not.
3. **Most of the signal.** Leading-order chirp durations above 50 Hz (computed 2026-10-06):

   | Binary | Mc (M☉) | Duration above 50 Hz | Frequency 64 s before merger |
   |---|---|---|---|
   | 0.2+0.2 | 0.174 | ~351 s | 95 Hz |
   | 0.4+0.4 | 0.348 | ~111 s | 61 Hz |
   | 0.6+0.6 | 0.522 | ~56 s | 48 Hz |
   | 1.0+1.0 | 0.871 | ~24 s | — |

   The 64-s window holds 32–64 s, with the merger in the second half. **No network has ever seen a whole light chirp.**
   The rung 1/2 negatives were for adding up separate window scores, not for a model that sees the whole track.
4. **Noise variety** (a fourth loss, of data rather than information). Training uses 18 h of H1 noise. 58 h are on disk
   (30 H1, 15 L1, 6 V1 files) and thousands of hours are public.

**Measured shortcut** ([cnn_response_probe.json](results/cnn_response_probe.json)):
- The CNN responds to band-limited noise **power** around 110 Hz.
- That is the same region real signals use (profile correlation 0.92), so it is "honestly fooled".
- The false alarms are short-term power drift, not glitches. Spearman against transient excess is +0.09 at n=1,860.

**Yardstick correction.** The "1.0 = ideal MF" in [pbh/metrics.py](pbh/metrics.py) is a hard-coded SNR 8 with no
false-alarm penalty, so it cannot be reached. The three tiers, fixed from now on:

| Tier | Meaning | Distance fraction |
|---|---|---|
| Oracle | true template known (cheats), semi-coherent n=8, real zero-FA threshold | 0.66–0.76 |
| Realizable best | adequate 3,235-template bank | 0.514 |
| Model | `cnn_w64` on identical injections | 0.484 |

Rearranging layers on the same input can buy at most about 6%. **Real gains need the input itself to change.**
That is the user's architecture/embedding instinct, aimed at information rather than capacity.

## 4. Directions

| # | Idea | Attacks | Prior art (searches 2026-10-06; "not found" means not found, not proof of absence) | Cost |
|---|---|---|---|---|
| 1 | **Counterfactual decoys as hard negatives** (see §5) | the 110-Hz power shortcut | time-reversed *templates* for background estimation ([2009.03025](https://arxiv.org/abs/2009.03025v1)); as NN training negatives: not found | cheap |
| 2 | **Complex chunk tokens.** One token per time chunk = complex matches against a coarse template set; a transformer learns the inter-chunk phase drift (coherence without millions of templates) | phase loss | not found in this form | medium |
| 3 | **Early two-detector fusion.** Cross-spectrum H1 × conj(L1) as an input channel | detector separation | late fusion exists ([2108.10715](https://arxiv.org/abs/2108.10715v2)); early: not found | medium |
| 4 | **Whole-signal context.** A long-sequence model over the full minutes-long track | window truncation | not checked yet | medium–high |
| 5 | **Proposer + exact verifier** (AlphaZero pattern). The NN proposes a chirp neighbourhood; a fully coherent MF verifies locally | bank cost / coherence | heavy BBH patch localisation, ~93% ([2609.28031](https://arxiv.org/abs/2609.28031)); subsolar: not found | high |
| 6 | **Learn the noise, not the signal.** A generative noise model gives a likelihood-ratio detector, which reduces exactly to the MF in Gaussian noise | non-Gaussian / non-stationary noise | parameter estimation ([2410.19956](https://arxiv.org/html/2410.19956v1)); ranking density ([2604.26581](https://arxiv.org/pdf/2604.26581)); as core detector: not found | high |
| 7 | **Physics as geometry.** Re-plot against f^(−8/3), where every leading-order chirp is a straight line (slope encodes Mc, intercept is merger time). Plus **PCEN**, the per-band automatic gain control from far-field speech | template sharing; power drift | no GW-NN prior art found for either | cheap |
| 8 | **Scale real noise** from 18 h to hundreds of hours | noise variety | standard | data-bound (GWOSC throughput, disk) |
| — | **Test-time adaptation / fast weights** (from tabula: TTT [2407.04620](https://arxiv.org/abs/2407.04620), Titans [2501.00663](https://arxiv.org/abs/2501.00663)) for adapting to *local* noise | power drift | not checked for GW | medium |
| — | **Conformal abstention** for the ringdown NPE (from tabula; SBI overconfidence [2110.06581](https://arxiv.org/abs/2110.06581)) | NPE over-confidence, prior dominance | standard in ML | cheap |

Notes from the cross-session exchange:
- **#7 alone is NOT coherence.** A Hough/Radon search on a power image is still incoherent and still pays N^(1/4).
  Straightening buys template sharing; coherence needs phase in the input. #7 composes with #2/#3 (straighten, then feed complex values).
- **#5:** calibrate the verifier's acceptance band on injections spread across the proposer's error radius and beyond, not at its centre.
- **#6:** test it off-centre. The hard axis for us is **power-drift segments**, with glitch-heavy segments second.
- Further reading from tabula, not yet verified by us:
  - prediction ≠ learning the law (Vafa [2507.06952](https://arxiv.org/abs/2507.06952));
  - PFN / TabPFN synthetic-prior training ([2112.10510](https://arxiv.org/abs/2112.10510));
  - frozen probes over fine-tuning (PhyIP [2602.12218](https://arxiv.org/abs/2602.12218)).

## 5. The first experiment: what has the current CNN learned? (draft — freeze before running)

**Question.** Does `cnn_w64` respond to the physics of a chirp (its sweep law), or only to band power?

**Decoys.** Two families. Each is matched to real chirps on band power, amplitude envelope and in-window
position. The tabula critique: a naive time reversal also flips the envelope, since a real chirp grows into
merger (∝ f^(2/3)) and the reversed one decays. Our windows also always put the merger in the second half,
which leaks position.

- **(a) Time-reversed sweep** with the original envelope re-imposed and the position matched.
- **(b) Wrong sweep law.** An upward sweep with the wrong frequency law (e.g. linear in f, or a wrong power index), with the same envelope and band power.

**Reading the outcome.**
- Rejects both → learned the sweep law.
- Rejects only (a) → envelope or position cue.
- Rejects neither → pure power shortcut.

**Probe ladder** (frozen weights; fine-tuning corrupts what is measured):
1. Linear probe for Mc from the 256-d embedding.
2. Nonlinear probe.
3. Behavioural test: score response to Mc at fixed band power. The decoy families serve as this test.

A failed linear probe alone proves nothing (tabula's Phase C: information stored nonlinearly).

**Draft prediction, written before any run.** Given the occlusion result (97–100% of sensitivity below 224 Hz,
false alarms and signals sharing one profile), decoys at matched SNR score close to real chirps: decoy
detection ≥ 70% of the real-chirp rate for both families. If true, hard-negative training (#1) is justified.
If false, the CNN already knows more physics than the occlusion probe suggested — also a finding.

**Gate before training anything new** (borrowed back from tabula's M0.5):
- Measure the oracle ceiling *of the candidate input encoding* before training on it.
- Measure a nuisance-only probe that must score at chance (position, duration, energy).

## 6. Plan, in order

0. **Rebuild the environment** from `requirements-lock.txt` (venvs deleted 2026-10-03) and re-run `./verify.sh` green.
1. **§5 decoy test and probe ladder** on the existing `cnn_w64`. Cheap; tells us what the current model learned. Pre-registered.
2. **If the shortcut is confirmed:** retrain with decoys as hard negatives (#1). Measure on the three-tier yardstick plus the power-drift off-centre set.
3. **Change the input** (#2 complex chunk tokens, #3 cross-spectrum early fusion), each preceded by the input-oracle gate.
4. **Bigger swings:** #5 proposer + verifier, #6 learned noise model, #4 whole-signal context. Each needs its own prior-art sweep first.
5. **In parallel, cheap:** #7 straightened input and PCEN as input-encoding ablations; conformal sets on the ringdown NPE.

**Guardrails (unchanged):**
- Injections into real noise; zero-FA / time-slide backgrounds.
- Pre-register before results.
- Prior-art search before any "first" claim.
- Effort is not a filter.

## 7. Scope notes for the other arcs

- **Ringdown and echoes are limited by the information in one event**, not by architecture. Only GW250114 is
  loud enough for δ, and the third tone needs about 1.5× louder. Architecture work does not move that.
- **The modelling gaps worth fixing** are the ringdown NPE's idealised-noise training and peak start (the +10%
  mass systematic), and its calibration (conformal sets).

## 8. Status

| Item | Status |
|---|---|
| Brainstorm + diagnosis | ✅ done 2026-10-06 |
| Exchange with `tabula-geometrica` | ✅ two rounds; they adopted the encoding audit, the three-tier yardstick, decoy worlds and nuisance-matching (in SpaceTime `curvature/notes/cartographer_design.md`) |
| §5 decoy experiment | ⏳ not started — waiting on the user's go and their own ideas (Ludo input-encoding experience) |
| Everything else | 💤 proposed |
