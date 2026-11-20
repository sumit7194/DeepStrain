# Gemini "Neon High" on the same task: record and comparison (2026-11-20)

Run in Sumit's Chrome (gemini.google.com, model picker "Neon High"), at his request, to compare with the Opus report
(`reports/Neural network architectures and learning.md`). Two messages were sent.

## Message 1: synthesise the 7 notes files (same brief as the Opus report writer)

- **Input:** the seven notes files uploaded as attachments (215 KB) and a condensed version of the report-writer brief.
- **Output:** about 61,000 characters, then **it ended in a Gemini "An error occurred / Try again"** partway through
  "Thread 4". It completed:
  - the one-page map;
  - all of section 2 (architectures);
  - all of section 3 (paradigms, backprop alternatives, optimisation, adaptation);
  - 3½ of the unifying threads.

  It did **not** reach recurring lessons, the hypothesis seeds, or gaps.
- **Citation hygiene (checked in-page):**
  - 554 distinct arXiv ids, 553 of them exactly in our notes.
  - The one "extra" (`1512.0338`) is `1512.03385` cut off by the error. **No invented citations.**
  - It kept 32 `[unverified]` tags and the snippet/background tags, and it never upgraded a claim.
- **Style:** dense and faithful. It is close to a re-typeset of the notes with a CS/systems analogy for each family (e.g. CNN = sliding stencil, SSM = fixed-size register, attention = content-addressed cache, MoE = routed co-processors), and a per-paragraph source-file attribution UI.

## Message 2: our problem statement (who we are, what we have, what we want)

- **Context loss:** after the error, Gemini continued in a new conversation id. The message arrived **without the attachments or the earlier reply**. Neon said so itself, answered from its own knowledge plus web search, and labelled everything `[CONJECTURE]`. That was honest handling, but the ideas below are **not grounded in our notes**.
- **External 2025–2026 papers it cited (none verified by us; check before any use):**
  - arXiv:2603.01737 "LRao detector" (learned locally-most-powerful detector in non-Gaussian noise)
  - arXiv:2605.22791 "Gated DeltaNet-2" (decoupled erase/write gates; fp32 decay accumulation)
  - arXiv:2602.21204 "TTT-KVB" (ICML 2026: TTT with key-value binding is a form of linear attention; gradient ascent in the inner loop works as well)
  - arXiv:2605.12466 "Solve the Loop" (fixed-point attractor architecture)
  - arXiv:2512.12399 Yamamoto et al. (CNN over template-bank SNR maps for GW; PRD 2026)
  - "LaCT" (ICLR 2026) and "TTT-E2E" (Dec 2025), given without ids

### Its four candidate ideas (all [CONJECTURE])

1. **Unsupervised noise-score front-end + phase-coherent linear integrator.**
   - Mechanism: learn the noise score −∇log p_noise from unlabelled background (denoising score matching). This is the Rao / locally-most-powerful weak-signal detector, which replaces linear whitening in non-Gaussian noise. Feed it to an O(T) phase-preserving recurrent or semi-coherent aggregator.
   - Must beat: LRao (2603.01737) and Huber/Tukey soft-clipping + whitening + matched filter.
   - Kill test: synthetic AR(2) + Student-t + 15σ glitches; it must clearly beat whitening **and** static soft-clipping at FPR 1e-4.
2. **Physically grounded online TTT for non-stationary background tracking.**
   - Mechanism: fast weights updated at test time with a robust next-step-prediction loss (Huber/Cauchy), so the detector sees adaptively whitened residuals.
   - Must beat: TTT-KVB / LaCT / TTT-E2E and classical RLS/NLMS.
   - Kill test: if a tuned RLS/NLMS whitens real non-stationary strain as well at matched latency, use RLS.
3. **Complex-oscillatory recurrence with decoupled erase/write and bounded writes.**
   - Mechanism: complex diagonal transitions for phase tracking, Gated DeltaNet-2-style decoupled gates, and tanh/Huber-bounded writes in fp32. The aim is to stop one glitch from contaminating minutes of accumulated phase.
   - Kill test: a 2×2×2 ablation against "3-line static input clipping + vanilla LRU/Gated DeltaNet".
4. **Compute-conditional coarse scan → fixed-point phase refinement.**
   - Mechanism: a cheap linear-time scanner flags the top-K windows; a weight-tied DEQ/attractor loop (phantom gradients) aligns cross-chunk phase, only on candidates.
   - Must beat: semi-coherent + Viterbi/Hough track linking, and an unrolled feed-forward network at matched FLOPs.
   - Kill test: a 32-chunk phase-linking toy; dead if gains plateau by K = 1–2 or it loses to semi-coherent + Viterbi.

### "Already done / known to fail" (its list, [CONJECTURE])

- Naive end-to-end CNN on whitened strain for long faint signals.
- A CNN over stacked matched-filter SNR maps (Yamamoto 2512.12399; it also discards phase).
- Generic TTT-KVB as online memory.
- Softmax attention over raw ultra-low-SNR waveforms: attention weights collapse to uniform when every key is noise.
- Diffusion/flow ELBO as a detection statistic: likelihood is dominated by local noise texture.
- Unregularised neural ODEs on noisy streams.

### "Where you are most likely fooling yourselves" (its list)

1. **Statistical vs architectural ceiling.** In stationary Gaussian noise the coherent matched filter is Neyman–Pearson optimal. A network can only win by (A) recovering the cross-chunk phase coherence that the semi-coherent bank drops, or (B) exploiting non-Gaussian / non-stationary noise. Pre-register a **phase-randomised Gaussianised twin of real noise** (same PSD, no glitches) to tell (A) from (B).
2. AUC instead of extreme-tail FPR.
3. Background-segment leakage and memorisation.
4. "General method on a laptop" leads to toy scale. Instead, target one mathematical regime (long-horizon, low-SNR, non-Gaussian streams) and show transfer across 2–3 domains.
5. Under-tuned classical baselines (Huber pre-whitening, RLS, Viterbi).
6. MPS fp16/bf16 phase-accumulation drift and complex-op fallbacks: check parity against CPU fp64.

## Assessment (Opus, for Sumit)

- **Synthesis task:** both models kept citations honest. Neon's first message was a faithful compression but **did not finish** (Gemini error), so on that task the complete deliverable is ours. It has 62-method maturity tables, 8 threads, 9 lessons, 14 hypothesis seeds and gaps.
- **Ideas task:** Neon's answer is sharp, but it is about **our GW problem** more than "general NN methods". That is the natural reading of the problem statement we sent. Several points line up with things this repo has already measured:
  - (1A/1B), the Gaussianised twin: our deep-FAR glitch work found the CNN's noise tail is **not glitch-driven** (glitch_score_correlation, 2026-09-02). That predicts lever (B), non-Gaussianity, is small for the CNN on LongChirp, and puts the weight on lever (A), phase coherence. The twin test would check this directly.
  - Ideas 1 and 3 assume glitches are the binding problem. Our evidence says they are not for this detector, though they might be for a phase-coherent one.
  - Idea 4 targets exactly the gap LongChirp exposes: Y1/Y0 ≈ 1/3, lost to semi-coherent chunking and template mismatch. It is the most aligned with our data.
  - "Fooling yourselves" items 2, 3 and 5 are already standing practice here: zero-FA/FAR thresholds, wall-clock-disjoint segments, and the "adequate bank" baseline. Item 6 is a real risk; we found an MPS float32 path agrees with float64 only to ~1e-5.
- **Next step if pursued:** verify Neon's external ids first (prior-art rigour rule). Then put Ideas 1–4 next to the 14 seeds in our report and pick by cheapest decisive test.
