# Prior-art check: explicit anti-collapse regularizers for local learning rules

Date: 2026-11-20. Status: prior-art check only; nothing was built.

**How citations were checked.** Every arXiv id below was checked against the arXiv API (title, first author, date) and tagged **[A]**. Journal papers were checked through Crossref or Europe PMC and tagged **[C]**. Conference pages fetched directly are tagged **[P]**. A paper whose text I read beyond the abstract (HTML or PDF) also carries **[read]**. The arXiv API rate-limited me (HTTP 429) twice, and I retried both times until the request went through, so no citation stayed unverified for that reason. Anything I could not verify is tagged [unverified]. Statements marked *Inference:* are my reasoning, not something a paper reports.

---

## 1. Verdict

The broad idea is **already largely done**. People have added anti-collapse, diversity, entropy, decorrelation or whitening terms to local learning rules several times:

- **Forward-Forward (FF) with an entropy regularizer.** Pardi et al., 2023: a Rényi entropy term was added to FF's layer loss specifically to keep deep layers from losing information. At 3 layers it preserved the accuracy of the deepest layer.
- **FF with a decorrelation goodness.** Shah & Tripathi, 2025: FF trained with goodness ‖h‖² − λ‖Cov(h)‖²_F.
- **FF with a whitened goodness.** Giannitrapani, 2026: on Imagenette this removed most of FF's accuracy loss with depth (3→7 layers).
- **Block-local supervised training with VICReg terms.** DeInfoReg, 2025: VICReg variance and covariance terms on each gradient-isolated block, alongside cross-entropy.
- **Information-preserving local losses.** InfoPro, 2021, and predsim, 2019, both target the collapse of information in greedy local layers.
- **Local self-supervised rules with decorrelation.** Halvagal & Zenke, 2023: dimensionality collapses to 1 when the decorrelation term is removed.

The Price of Locality paper (arXiv:2609.33240) exists and says what was claimed. It also tests two rank-raising interventions of its own: relaxing locality, and making the weight updates full-rank (Muon-like). The second recovers about 10 pp of FF accuracy but leaves a gap of about 21 pp.

**What is still open is narrower.** No paper I found takes the kernel/error-signal effective rank that Price of Locality defines and tests causally whether an explicit representation-level isotropy term (VICReg variance/covariance or SIGReg) changes the *slope* of the local-vs-backprop gap as depth grows, against the existing rank-raising fixes as controls, for both:

- FF-type rules, where the error signal *is* the representation, and
- local cross-entropy rules (Mono-Forward, greedy layer-wise), where the error-signal rank is plausibly capped near the number of classes (*inference*).

One 2026 paper is a warning sign for this test. It found that repairs raising deep-layer "health" by 5–45× moved FF accuracy by less than 1 pp, which is exactly the dissociation the experiment has to rule out. The worthwhile contribution is therefore a **mechanism test with a pre-registered dissociation criterion**, not a new regularizer.

---

## 2. Closest prior work

### 2a. The diagnosis paper the idea rests on

**Wu, Liu, Chen, "The Price of Locality: Why Forward-Forward Underperforms Backpropagation?", arXiv:2609.33240 (Sep 2026; comment says accepted at NeurIPS 2026) [A][read].**

It **exists**, and the claims in the brief match it. Its abstract describes two failure modes:

- an optimization floor: each layer satisfies the Polyak–Łojasiewicz condition, but concurrent updates make each layer's input distribution drift, which produces an error floor;
- representational collapse: the pairwise similarity kernel contracts exponentially toward rank one with depth.

Effective rank is defined as erank(M) = (tr M)²/tr(M²).

Details that matter for the proposal (read in the HTML):

- **Theorem 3.1 is a random-product result.** It assumes an i.i.d. weight sequence and a negative top Lyapunov exponent of the layer-product map. The argument that the FF objective "cannot effectively exploit" residuals or normalization comes on top of that.
  *Inference:* the theorem guarantees contraction for the forward map in a random-weight regime. It does not show that *learning* under FF pushes the kernel toward rank one.
- **The capacity argument is about the error signal.** "Effective learning capacity" sums the effective ranks of the per-layer *error signals* Γ. For norm goodness the error signal is proportional to the representation, so the two coincide. For label- or target-dependent goodnesses, the paper says the signal keeps "the residual diversity present in" the goodness-gradient family.
- **Intervention 1, locality relaxation** (CNN12, CIFAR-10, 3 seeds):

  | Configuration (blocks × layers) | Accuracy | Mean erank(Γ) |
  |---|---|---|
  | 12×1 (strictly local) | 53.0% | 2.20 |
  | 6×2 | 58.1% | — |
  | 4×3 | 67.6% | — |
  | 3×4 | 72.2% | — |
  | 2×6 | 76.2% | — |
  | 1×12 | 78.7% | 7.98 |
  | Backprop | 85.7% | 31.7 |

- **Intervention 2, update spectral flattening** (CNN3, CIFAR-10). Every convolutional update's singular values are interpolated toward a flat, Muon-like spectrum:

  | Method | Accuracy, p = 0 | Accuracy, p = 1 | Update erank |
  |---|---|---|---|
  | FF | 47.4% | 57.7% | about 3.4 → 4.4 |
  | Backprop | 77.6% | 79.0% | — |

  The paper's own reading: low-rank update geometry is "a contributing bottleneck", but "spectral flattening alone does not address the locality issues."
- **It does not propose a representation-level anti-collapse loss.** It explicitly leaves "translating the diagnosis into a new local learning algorithm" to future work.
- It also argues that predictive coding without the inference loop is an instance of local supervised learning and "inherits the pathologies".

*Inference:* the paper's measured intervention is on **update rank**, with **error-signal rank** as the diagnostic. A representation-level regularizer is a different lever. That distinction is where the open space is.

### 2b. Anti-collapse, entropy, decorrelation or whitening terms already added to FF

1. **Pardi, Tortorella, Micheli, "Entropy Based Regularization Improves Performance in the Forward-Forward Algorithm", ESANN 2023, pp. 393–398 [P][read, full PDF].**
   - **What it does.** It changes the FF layer loss to L_BCE − T·H₂(h). H₂ is the Rényi quadratic entropy of the layer's batch activations, estimated with a Parzen window. The stated aims are to "avoid redundancy in each layer representation due to high correlations among unit activations" and to make each layer carry information forward.
   - **Results** (3-layer MLPs, double-moon, noisy double-moon and MNIST): gains everywhere. The accuracy of the deepest layer is preserved where vanilla FF loses it. Double-moon, 10 units, layer-3 goodness accuracy: 63.3% → 99.5%. MNIST, 20 units, layer 3: 87.5% → 92.4%.
   - **How it differs from the idea.** It is an entropy (spread) regularizer, not VICReg, Barlow or SIGReg. It uses only 3 layers on toy and vector data, no depth sweep, no rank measurement, and no CNNs.
   - **Closest existing instance of the idea for FF.**

2. **Shah & Tripathi, "In Search of Goodness: Large Scale Benchmarking of Goodness Functions for the Forward-Forward Algorithm", arXiv:2511.18567 (Nov 2025) [A][read via fetch].**
   - **What it does.** Among 21 goodness functions it includes `decorrelation_local`, G(h) = ‖h‖² − λ‖Cov(h)‖²_F, applied per layer, and `whitened_energy_local`. Network: 4 fully connected layers of 2000 units.
   - **Results for decorrelation vs baseline:**

     | Dataset | Baseline | Decorrelation goodness |
     |---|---|---|
     | CIFAR-10 | 39.59% | 41.46% |
     | Fashion-MNIST | 84.87% | 85.56% |
     | STL-10 | 36.64% | 36.89% |
     | MNIST | 98.05% | 97.95% |

   - **Caveat.** Whitened energy reproduces the baseline numbers *exactly* on all four datasets, and several rows share identical values, so the implementation is suspect.
   - **How it differs.** Fixed depth, no rank diagnostics, MLP only.

3. **Giannitrapani, "Gauge-Fixing the Forward-Forward Objective: A Whitened Goodness Derived from a Likelihood-Ratio Account", arXiv:2607.12501 (Jul 2026) [A][read via fetch].**
   - **What it does.** It replaces FF goodness with G(z) = zᵀΣ̂₋⁻¹z/C, the inverse of a running covariance of the negative samples, which makes the goodness scale-invariant (whitened). The stated motivation is a "gauge": FF can lower its loss by inflating the scale of its weights.
   - **Setup.** CNNs, 64×64 inputs, depths 3, 5 and 7, 13 seeds per cell.
   - **Results:**

     | Dataset and method | Depth 3 | Depth 5 | Depth 7 |
     |---|---|---|---|
     | Imagenette, plain FF | 42.7% | 39.2% | 35.7% |
     | Imagenette, whitened | 44.3% | 43.8% | 43.0% |
     | Imagenette, backprop | 52.7% | 52.5% | 51.3% |

     On CIFAR-10 it closes 45–61% of the gap to backprop.
   - **How it differs.** The whitening is *inside the goodness*, not a regularizer on the representation's spread, and its motivation is scale, not rank. No kernel-rank measurement across depth was found in the parts I read; there is a per-layer participation ratio on plainly trained networks only.
   - *Inference:* this is the most important empirical prior. It already *flattens* FF's accuracy-vs-depth curve at 3–7 layers, so a new regularizer has to beat this.

4. **Jiang, Al-Hashimi, Xu, "Covariance-Aware Goodness for Scalable Forward-Forward Learning", arXiv:2605.04346 (May 2026) [A].**
   - **What it does.** Second-order (cross-channel and spatial covariance) goodness, plus a Feature Alignment Layer for misaligned representations in deep local networks. It roughly doubles viable FF depth to VGG-16. Results: ImageNet-100 73.0%, Tiny-ImageNet 50.3%.
   - **How it differs.** Covariance is used as a *feature* for goodness, not a penalty; collapse is not discussed in the rank sense.

5. **Sun et al., "DeeperForward: Enhanced Forward-Forward Training for Deeper and Better Performance", ICLR 2025 [P].**
   - **What it does.** Layer normalization plus mean goodness, aimed at "deactivated neurons" and goodness leaking to the next layer. It reaches a 17-layer CNN.
   - **How it differs.** A normalization-based design, not an explicit distributional regularizer.

6. Related, not collapse terms:
   - **Dooms et al., "The Trifecta", arXiv:2311.18130 [A]**: normalization choices for deeper FF.
   - **Lorberbom et al., "Layer Collaboration in the Forward-Forward Algorithm", arXiv:2305.12393 [A]**.
   - **Chen et al., "Self-Contrastive Forward-Forward algorithm", Nature Communications 2025, DOI 10.1038/s41467-025-61037-0 [C]; arXiv:2409.11593 [A]**.
   - **Yao, Chen, Kuo, "HCL-FF", arXiv:2605.24797 [A]**: adds a supervised contrastive term to FF.
   - None of these add an explicit variance/covariance or isotropy penalty, as far as their abstracts show. I did not read their full texts.

### 2c. Block-local supervised training with VICReg-type terms

7. **Huang, Lin, Chen, "DeInfoReg: A Decoupled Learning Framework for Better Training Throughput", arXiv:2506.18193 (Jun 2025) [A][read via fetch].**
   - **What it does.** Each gradient-isolated block minimizes L_LL + α·L_CE with α = 0.001. L_LL = variance (VICReg hinge on per-dimension standard deviation) + invariance (MSE between the batch label-similarity matrix and the embedding cosine-similarity matrix, a predsim-like term) + covariance (off-diagonal penalty).
   - **Depth sweep** (LSTM on AGNews, 4–14 modules): DeInfoReg stays flat at about 92%, while backprop collapses from 91% to 25%. That collapse is a backprop *vanishing-gradient* failure, not a local-learning gap.
   - **Ablation** (VGG on CIFAR-100): all three terms 74.27% vs invariance alone 74.02%. The variance and covariance terms are worth about 0.2–0.4 pp, single-seed and untested for significance.
   - **How it differs.** This *is* "VICReg variance/covariance on a supervised local rule". But:
     - it has no rank or collapse measurement;
     - it does not compare against FF, Mono-Forward, DGL, InfoPro or AugLocal;
     - the supervised signal enters through a similarity-matching invariance term, not plain cross-entropy;
     - there is no CNN depth sweep against backprop.
   - **This is the strongest "already done" item for the supervised local cross-entropy branch.**

8. **Wang, Ni, Song, Yang, Huang, "Revisiting Locally Supervised Learning: an Alternative to End-to-end Training" (InfoPro), arXiv:2101.10832, ICLR 2021 [A].**
   - Reports that greedy local cross-entropy "tends to collapse task-relevant information at early layers". The fix (InfoPro) reduces to reconstruction plus cross-entropy or contrastive loss.
   - **How it differs.** The collapse here is *information* collapse in the early layers, fixed by reconstruction rather than a distributional regularizer. This is the standard baseline for "anti-collapse in supervised local learning".

9. **Nøkland & Eidnes, "Training Neural Networks with Local Error Signals" (predsim), arXiv:1901.06656, ICML 2019 [A].**
   - Each layer's loss combines cross-entropy with *similarity matching* (the layer's cosine-similarity matrix vs the label-similarity matrix).
   - *Inference:* similarity matching directly shapes the very kernel Σ that Price of Locality analyses, toward the label kernel of rank C, so it is a supervised kernel regularizer already in wide use.

10. **Siddiqui, Krueger, LeCun, Deny, "Blockwise Self-Supervised Learning at Scale", arXiv:2302.01647 [A][read].**
    - 70.48% vs 71.57% on ImageNet ResNet-50 as quoted in the brief. Also tested VICReg and SimCLR blockwise, with consistent results.
    - **Key finding:** *supervised* blockwise training makes early blocks better and later blocks worse, which they read as supervised local losses discarding information early.
    - *Inference:* this is direct evidence that the depth gap of supervised local rules is related to early-layer information loss, and that an SSL anti-collapse loss mitigates it. Arguably the strongest support for the idea, and also evidence that it is not new as a qualitative claim.

11. Other supervised local baselines:
    - **Belilovsky et al., "Decoupled Greedy Learning of CNNs", arXiv:1901.08164 [A]**.
    - **Belilovsky et al., "Greedy Layerwise Learning Can Scale to ImageNet", arXiv:1812.11446 [A]**.
    - **Ma et al., "AugLocal", arXiv:2402.17318, ICLR 2024 [A]**: about backprop-level accuracy at up to 55 local layers.
    - **Su et al., "Momentum Auxiliary Network", arXiv:2407.05623 [A]**.
    - **Yu et al., "ContSup", arXiv:2312.07636 [A]**: reports that accuracy degrades as the number of gradient-isolated modules grows, and adds context supply between modules.
    - **Patel, Eickenberg, Belilovsky, "Local Learning with Neuron Groups", arXiv:2301.07635 [A]**: search snippets describe a diversity penalty between groups; I did not confirm it in the text [snippet].
    - **Gong, Li, Abdulla, "Mono-Forward", arXiv:2501.09238 [A][read via fetch]**: no regularizers; no accuracy-vs-depth table (main CNN results at 4 layers). Appendix G says FF shows "progressively weaker class structure in deeper layers" while Mono-Forward keeps it.
      *Inference:* the collapse premise may be **weaker or absent for Mono-Forward**, so the proposal must first *measure* collapse there before claiming to fix it.

### 2d. Local self-supervised rules with anti-collapse terms (the idea's source domain)

12. **Halvagal & Zenke, "The combination of Hebbian and predictive plasticity learns invariant object representations in deep sensory networks", Nature Neuroscience 2023, DOI 10.1038/s41593-023-01460-y [C].**
    - A layer-local rule built from a predictive (invariance) term, a Hebbian (variance) term and a decorrelation term, i.e. a local VICReg. The search snippet reports that removing the Hebbian or decorrelation term collapses representation dimensionality to 1 [snippet; full text not read].
    - **How it differs.** Self-supervised, not a supervised local rule.

13. Other local self-supervised rules:
    - **Datta et al., "Deep representation learning using layer-wise VICReg losses", Scientific Reports 15 (2025), DOI 10.1038/s41598-025-08504-2 [C; abstract via Europe PMC]**: layer-wise VICReg with two views, then a few-label probe.
    - **Wiest, "Temporal Regularized Learning (TeReL)", arXiv:2611.17019 (Nov 2026) [A]**: VICReg made local in space and time.
    - **Andrews, "Local Synaptic Rules Can Implement a SIGReg Gradient Without Backpropagation", arXiv:2607.21622 (Jul 2026) [A]**: STDP⁺ plus homeostatic plasticity implements the exact SIGReg gradient, unsupervised, 2 layers.
    - **Mansur & Zylberberg, "Increasing Width Allows Greedy Layer-wise Training to Rival End-to-End Backpropagation in Self-Supervised Learning", arXiv:2610.00753 (Sep 2026) [A]**: Barlow Twins greedy training; width closes the gap.
    - **Löwe et al., Greedy InfoMax, arXiv:1905.11786 [A]**; **Xiong et al., LoCo, arXiv:2008.01342 [A]**.
    - **Ren, Kornblith, Liao, Hinton, "Scaling Forward Gradient With Local Losses", arXiv:2210.03310 [A]**.
    - **Journé et al., "Hebbian Deep Learning Without Feedback" (SoftHebb), arXiv:2209.11883 [A]**: soft winner-take-all competition is its implicit anti-collapse mechanism; accuracy *increases* with depth up to 5 layers. I did not read the full text.

### 2e. Decorrelation-based learning more broadly

14. **Offergeld & Ahmad, "Correlations Are Ruining Your Gradient Descent", arXiv:2407.10780 (v5; v1 lists Ahmad alone) [A].**
    - The input-correlation factor of natural gradient can be moved into the network as a decorrelating transform, with a Hebbian/local recurrent form. Decorrelation makes approximate-gradient methods that "previously failed catastrophically" work.
    - **How it differs.** Decorrelation acts on layer *inputs* as preconditioning, aimed at optimization conditioning, not at rank collapse in FF or local supervised training. Not tested on FF or local cross-entropy, as far as the abstract shows.

15. **Dalm, Offergeld, Ahmad, van Gerven, "Efficient Deep Learning with Decorrelated Backpropagation", arXiv:2405.02385 [A].** Also, by overlapping authors: **"Node Perturbation Can Effectively Train Multi-Layer Neural Networks", arXiv:2310.00965 [A]** (input decorrelation makes node perturbation work), and **Carrigg et al., "Decorrelation Speeds Up Vision Transformers", arXiv:2510.14657 [A]**.
    - *Inference:* the obvious untested cross is input decorrelation (Dalm-style) inside FF or Mono-Forward. I found no paper combining them.

16. Related regularizers outside local learning:
    - **Zhu et al., "Variance-Covariance Regularization Improves Representation Learning" (VCReg), arXiv:2306.13292 [A]**: VICReg's variance and covariance terms on intermediate representations, supervised, end-to-end backprop.
    - **Akbar, "Weak-SIGReg: Covariance Regularization for Stable Deep Learning", ICLR 2026 workshop (Geometry-grounded Representation Learning) [P]**: sketched SIGReg as a supervised stabilizer; collapsed ViT on CIFAR-100 goes from 20.7% to 72.0%; end-to-end.
    - **Daneshmand et al., arXiv:2003.01652 [A]**: batch normalization provably avoids rank collapse at random initialization (cited by Price of Locality).
    - Foundational sources: **Bardes et al., VICReg, arXiv:2105.04906 [A]**; **Zbontar et al., Barlow Twins, arXiv:2103.03230 [A]**; **Balestriero & LeCun, LeJEPA/SIGReg, arXiv:2511.08544 [A]**; **Jing et al., arXiv:2110.09348 [A]** (dimensional collapse).

### 2f. Evidence that may cut against the idea

17. **Yousefiramandi, "Cumulative-Goodness Free-Riding in Forward-Forward Networks: Real, Repairable, but Not Accuracy-Dominant", arXiv:2605.06240 (May 2026) [A].**
    - Repairs raised deeper-layer separation up to 5× (CIFAR-100) and 45× (8-block CIFAR-10), yet moved accuracy by under 1 pp. "Layer health likewise does not track accuracy."
    - *Inference:* this is the exact failure mode the proposal must design against. Raising deep-layer rank could likewise fail to move accuracy, because readout or summed-goodness scoring is invariant to how goodness is redistributed across layers.

18. **Hofmann & Mäder, "Force without transmission: a depth-induced rank collapse that no loss on the representation reopens", arXiv:2610.09958 (Oct 2026) [A; abstract via fetch].**
    - End-to-end small transformers: no added representation loss repaired rank collapse, because the gradient path to the relevant weights was cut. Restoring the skip connection did repair it.
    - *Inference:* in local learning the per-layer term reaches its own weights directly, so this obstruction is structurally weaker. It does warn that a regularizer placed only at the top cannot fix collapse lower down.

19. **Chen, "Synthetic Benchmarks Overstate Forward-Forward Scaling", arXiv:2606.06539 [A]**: FF's ceiling on real data; a deeply supervised backprop baseline beats the best FF variant by 2.4/5.9 pp on CIFAR-10/100.
    **Moon et al., "Greedy Local Learning for Language Model Pretraining", arXiv:2610.04867 [A]**: the gap to end-to-end training more than doubles from K = 2 to K = 4 blocks.

### 2g. Predictive coding and NoProp

20. Predictive coding:
    - **Innocenti, Achour, Buckley, "μPC: Scaling Predictive Coding to 100+ Layer Networks", arXiv:2505.13124 [A]**: depth scaling fixed by parameterization (Depth-μP), not by anti-collapse.
    - **Beumer et al., "Natural Gradient Learning in Predictive Coding through Lateral Circuits", arXiv:2611.16666 (Nov 2026) [A]**: a lateral circuit learns activity correlations, i.e. local whitening for natural gradient. Results are on reconstruction tasks.
    - I found **no** predictive-coding paper that adds a VICReg or SIGReg term to fix depth collapse.

21. NoProp:
    - **Li, Teh, Pascanu, "NoProp", arXiv:2503.24322 [A]**; **Shing, Koyama, Akiba, "DiffusionBlocks", arXiv:2506.14202, ICLR 2026 [A]**.
    - I found no anti-collapse work on NoProp.
    - *Inference:* NoProp fixes each block's representation to a noised label embedding by construction, so "kernel collapse across depth" is not well defined in the same way. Including NoProp in the proposal would need a different premise.

---

## 3. Pieces that are partly done

| Piece | Status | Evidence |
|---|---|---|
| Anti-collapse or entropy term on FF layers | **Done at small scale** (3–4 layer MLPs) | Pardi 2023; Shah & Tripathi 2025 |
| Whitening inside FF goodness flattens accuracy vs depth | **Done at 3–7 conv layers** | Giannitrapani 2026 |
| VICReg variance/covariance on supervised gradient-isolated blocks | **Done, but no rank diagnostics and tiny ablation gains** (0.2–0.4 pp) | DeInfoReg 2025 |
| Information collapse in greedy supervised layers, plus a fix | **Done** (reconstruction / similarity matching) | InfoPro 2021; predsim 2019; ContSup 2023 |
| SSL anti-collapse loss per block closes most of the backprop gap | **Done** (self-supervised setting) | Siddiqui 2023; Mansur & Zylberberg 2026; Halvagal & Zenke 2023 |
| Rank-collapse theory for FF and a measured rank intervention | **Done for update rank and locality** | Price of Locality 2026 (Muon-like flattening: FF +10 pp, about 21 pp gap remains) |
| SIGReg as a supervised stabilizer | **Done end-to-end** (workshop) | Weak-SIGReg, ICLR 2026 workshop |
| SIGReg implemented by local plasticity | **Done, unsupervised, 2 layers** | Andrews 2026 |
| Input decorrelation (Dalm/Ahmad) inside FF or Mono-Forward | **Not found** | — |
| Kernel-rank depth sweep for local cross-entropy / Mono-Forward | **Not found** (Mono-Forward reports class structure is *kept* at depth) | — |
| Causal test: does a representation-level isotropy term change the *depth slope* of the gap, measured with Price of Locality's erank, vs the existing rank fixes | **Not found** | — |

---

## 4. The narrowest version that is genuinely open

**Question.** Is representation rank collapse a *cause* of the depth-growing gap in supervised local rules, or an epiphenomenon? Test this by comparing, on Price of Locality's own diagnostics, an explicit per-layer isotropy regularizer against the rank-raising interventions that already exist.

Three things make it new, given what I found:

1. **It tests a mechanism, not a method.** Prior regularizers (Pardi, decorrelation goodness, DeInfoReg) report accuracy only. None measures kernel or error-signal effective rank versus depth, and none checks whether the rank gain *mediates* the accuracy gain.
2. **It covers two branches separately.**
   - **FF-type rules (norm goodness):** the error signal is proportional to the representation, so raising representation rank should mechanically raise erank(Γ).
   - **Local cross-entropy rules (Mono-Forward, greedy layer-wise):** *inference:* the per-layer error signal comes from a C-way softmax head, so its rank is capped near C − 1 whatever the representation's rank is.

   **Prediction (to pre-register):** a representation regularizer helps FF more than Mono-Forward. If it helps Mono-Forward equally, the Price of Locality capacity story is not the mechanism.
3. **It includes a dissociation control.** This answers the 2605.06240 objection that "layer health does not track accuracy".

**Baselines to beat.** Each is an existing method that already raises rank or flattens the depth curve:

- for FF: whitened goodness (Giannitrapani 2026);
- the Muon-like update flattening at p = 1 (Price of Locality, Table 3);
- for the local cross-entropy branch: DeInfoReg's variance/covariance terms and InfoPro;
- the upper reference at matched depth: AugLocal or end-to-end backprop.

**Cheapest decisive experiment.**

- **Setup.** CIFAR-10, Price of Locality's CNN family (its CNN12, strictly-local 53.0% vs backprop 85.7%, is the natural anchor). Depth L ∈ {3, 6, 9, 12}, 3 seeds.
- **Arms:**
  1. FF (paper recipe);
  2. FF + VICReg variance/covariance on each layer's output;
  3. FF + SIGReg (sketched, per layer);
  4. FF with whitened goodness;
  5. FF + update flattening (p = 1);
  6. local cross-entropy (Mono-Forward-style), with and without variance/covariance.

  That is about 6 arms × 4 depths × 3 seeds ≈ 72 short runs, feasible on one GPU. Start with an MLP version to debug.
- **Measure per layer:** test accuracy and linear-probe accuracy, kernel erank of Σ^(ℓ), and error-signal erank(Γ). Use the same definitions as Price of Locality.
- **Pre-registered decision rule:**
  - **(a) Mechanism supported:** the regularizer raises deep-layer kernel erank *and* reduces the slope of (backprop accuracy − local accuracy) against L by a declared margin relative to plain FF, with the effect at least that of the whitened-goodness baseline.
  - **(b) Rank is epiphenomenal:** erank rises but the slope does not move (the 2605.06240 pattern).
  - **(c) No rank effect:** the regularizer cannot raise erank under FF's dynamics, a regime the "force without transmission" obstruction would predict.

  Report also whatever gap is left once rank is matched, because that remainder estimates the separate inter-layer-drift floor that Price of Locality identifies.

**Known interaction to handle (*inference*).** FF goodness depends on activation norms, and the VICReg variance hinge also acts on scale. The Giannitrapani "gauge" result shows that scale shortcuts can dominate. So apply the regularizer *after* the inter-layer normalization, and log the goodness scale over training.

---

## 5. Gaps in this search

- **Full texts not read** (claims rest on abstracts or snippets):
  - Halvagal & Zenke: the "collapse to 1" claim is from a search snippet.
  - Layer-wise VICReg (Datta et al.): abstract only.
  - Local Learning with Neuron Groups: the diversity penalty is a snippet.
  - SCFF, HCL-FF, Trifecta, Layer Collaboration, SoftHebb: abstracts only. Any of them could contain a variance or decorrelation term I missed.
- **Partly read:** Mono-Forward appendices after J and the last 26k characters of Giannitrapani were not read. The Price of Locality tables were read from LaTeX alt-text extracted from the HTML.
- **Partly verified:** DeInfoReg's numbers come from a fetch-model summary of its HTML. Spot-check its Tables 5 and 9 before citing them.
- **Search coverage:** no direct OpenReview or Google Scholar search (no access), and ICLR 2026 / NeurIPS 2026 submissions were reached only through web search. Non-English literature and theses (for example the TU Delft 2026 Mono-Forward thesis, snippet only) were not checked.
- **Not found, so absence is not established:** anti-collapse work on NoProp or DiffusionBlocks; decorrelated-input (Dalm/Ahmad) FF; supervised-local SIGReg. Any of these may exist under other names ("feature diversity", "redundancy", "orthogonality", "isotropy", "spectral regularization").
- **Rate limits:** the arXiv API returned HTTP 429 twice; both requests succeeded on retry.

Environment: web research only, no code run · git 7f0b1a9 (working tree at time of writing).
