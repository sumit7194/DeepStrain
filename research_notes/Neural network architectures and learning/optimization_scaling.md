# Neural network optimization, normalization, regularization, parameterization, and the science of training

Method note: every arXiv id below was fetched from its abstract page (arxiv.org/abs/ID) in this session, and the stated title/authors/date/claims come from that fetch. Where a claim comes from my background knowledge and was NOT checked in a fetched page it is marked [unverified]. Claims in the "Cited Findings" sections are what the cited page says; claims about what the authors "report" are the authors' own, not independently replicated by me. Dates are first-submission dates unless noted.

## Which newer optimizers (Muon, SOAP, Shampoo, schedule-free, Lion, Sophia) have independently verified wins over AdamW, at what scale, and what do careful benchmarks say?

### Takeaway
AdamW remains the production default. Under rigorous per-optimizer tuning, matrix-preconditioned optimizers (Muon, SOAP, Kron/PSGD) win roughly 1.3-1.4x at 0.1-0.5B parameters, but the measured speedup decays to about 1.1x at 1.2B, and Lion/Sophia claims largely do not survive careful re-tuning. The two strongest independent signals are the AlgoPerf competition (Distributed Shampoo and Schedule-Free AdamW won their tracks) and Kimi K2 (the one large public frontier-scale run using a Muon variant).

### Cited Findings

Foundations and plain-language mechanisms
- Adam (Kingma and Ba, arXiv:1412.6980, 2014): per-parameter adaptive step from running estimates of first and second gradient moments; invariant to diagonal gradient rescaling. Plain terms: SGD with momentum, but each weight's step is divided by a running RMS of its own gradient. — [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)
- AdamW (Loshchilov and Hutter, arXiv:1711.05101, 2017): L2 regularization and weight decay are not equivalent under adaptive methods; decoupling decay from the gradient step makes the best decay factor largely independent of learning rate and improves Adam's generalization. Adopted in TensorFlow and PyTorch. — [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)
- Shampoo (Gupta, Koren, Singer, arXiv:1802.09568, 2018): keeps one preconditioning matrix per tensor dimension (Kronecker-style structure) rather than one huge matrix; convex-stochastic convergence guarantees; per-step cost claimed comparable to Adam. — [arXiv:1802.09568](https://arxiv.org/abs/1802.09568)
- K-FAC (Martens and Grosse, arXiv:1503.05671, 2015): approximates the Fisher information matrix with layer-wise Kronecker products of two small matrices to approximate natural gradient descent; costs several times SGD per update; unlike Hessian-free methods it tolerates stochastic gradients because storage/inversion cost does not depend on the amount of data used to estimate curvature. — [arXiv:1503.05671](https://arxiv.org/abs/1503.05671)
- Lion (Chen et al., arXiv:2302.06675, 2023): found by program search; keeps only momentum (less memory than Adam) and uses sign-based updates so every parameter gets the same update magnitude; needs a smaller learning rate than Adam; authors report gains growing with batch size and note some settings with small or insignificant gains. Deployed in Google search-ads CTR models (authors' statement). — [arXiv:2302.06675](https://arxiv.org/abs/2302.06675)
- Sophia (Liu, Li, Hall, Liang, Ma, arXiv:2305.14342, 2023): diagonal-Hessian estimate refreshed every few steps, update = momentum / Hessian-estimate with element-wise clipping; authors report about 2x fewer steps than Adam on GPT models of 125M-1.5B. — [arXiv:2305.14342](https://arxiv.org/abs/2305.14342)
- SOAP (Vyas et al., arXiv:2409.11321, 2024): shows Shampoo (power 1/2) equals Adafactor run in Shampoo's eigenbasis; SOAP runs Adam in that eigenbasis, updating second moments continually in the slowly changing basis; one extra hyperparameter (preconditioning frequency). Authors report, at 360M and 660M with large batch, over 40% fewer iterations and over 35% less wall-clock than AdamW and about 20% better than Shampoo. — [arXiv:2409.11321](https://arxiv.org/abs/2409.11321)
- Schedule-Free (Defazio et al., "The Road Less Scheduled", arXiv:2405.15682, 2024): removes the learning-rate schedule by unifying scheduling with iterate averaging; no extra hyperparameters over momentum optimizers; Schedule-Free AdamW was core of a winning entry in the AlgoPerf 2024 self-tuning track. — [arXiv:2405.15682](https://arxiv.org/abs/2405.15682)
- Muon (Keller Jordan et al., blog, 2024; no arXiv paper for the original): applies an SGD-momentum update to each 2D hidden-layer weight, then approximately orthogonalizes it (replace G = U S V^T by U V^T) with 5 Newton-Schulz iterations in bfloat16 using tuned quintic coefficients (3.4445, -4.7750, 2.0315). Embeddings, output head and scalars/vectors should stay on AdamW; Q, K, V work better orthogonalized separately. Author-reported results: NanoGPT speedrun about 1.35x faster, 1.5B model reaching GPT-2-XL-level HellaSwag in about 10 vs 13.3 8xH100-hours. Author states open questions: scaling beyond 20B/1T tokens, distributing Newton-Schulz, fine-tuning/RL. — [Jordan blog](https://kellerjordan.github.io/posts/muon/)
- Theory framing: Bernstein and Newhouse (arXiv:2409.20325, 2024) argue Adam, Shampoo and Prodigy (EMAs off) are steepest descent under particular norms; Muon-style orthogonalized updates are the spectral-norm member of that family (the Muon link is my reading, [unverified] in the abstract). — [arXiv:2409.20325](https://arxiv.org/abs/2409.20325)

Scale evidence for Muon
- "Muon is Scalable for LLM Training" (Liu et al., Moonshot, arXiv:2502.16982, Feb 2025): adding weight decay and per-parameter update-scale matching lets Muon run without hyperparameter retuning; authors' scaling-law experiments indicate about 2x compute efficiency vs AdamW (compute-optimal); Moonlight 3B/16B-total MoE trained on 5.7T tokens. — [arXiv:2502.16982](https://arxiv.org/abs/2502.16982)
- Kimi K2 (Kimi Team, arXiv:2507.20534, Jul 2025): 1T total / 32B active MoE, 15.5T tokens, trained with MuonClip (Muon plus QK-clip to fix instability). — [arXiv:2507.20534](https://arxiv.org/abs/2507.20534)
- "Practical Efficiency of Muon for Pretraining" (Essential AI, arXiv:2505.02222, May 2025): Muon expands the compute-time Pareto frontier over AdamW, keeps data efficiency at batch sizes well past critical batch size, combined with muP via a "telescoping" tuning algorithm; models up to 4B. — [arXiv:2505.02222](https://arxiv.org/abs/2505.02222)

Independent / careful benchmarks
- AlgoPerf benchmark (Dahl, Schneider et al., arXiv:2306.07179, 2023, rev. Jun 2025): time-to-result benchmark on fixed hardware with workload variants to expose non-robust methods and separate external-tuning vs self-tuning rule sets. — [arXiv:2306.07179](https://arxiv.org/abs/2306.07179)
- AlgoPerf competition analysis (Kasimbeg et al., arXiv:2502.15015, Feb 2025, ICLR 2025): 18 submissions from 10 teams; external-tuning winner Distributed Shampoo (credited to non-diagonal preconditioning beating Adam even in wall-clock); self-tuning winner Schedule-Free AdamW; top entries robust to workload changes. — [arXiv:2502.15015](https://arxiv.org/abs/2502.15015)
- "Fantastic Pretraining Optimizers and Where to Find Them" (Wen, Hall, Ma, Liang, arXiv:2509.02046, Sep 2025; v3 Oct 2026): ten optimizers, four scales 0.1B-1.2B, 1-8x Chinchilla data ratios, trained on DCLM/StarCoder/ProofPile. Findings read from the full text: (i) many claimed 1.4-2x speedups come from unequal tuning (e.g. tuning AdamW's learning rate alone gave about 2x on a 130M baseline; Lion's optimal weight decay about 0.6 vs AdamW's about 0.1, so blind hyperparameter transfer is unfair); (ii) scalar optimizers (AdamW variants, Lion, Mars, Adam-mini) all within <1.2x of AdamW after tuning; (iii) matrix-based optimizers (Kron, Muon, Soap) about 1.3x for models under 520M, but the speedup falls to about 1.1x at 1.2B and no longer produces downstream gains; their fitted scaling law predicts Muon slightly worse than AdamW at 7B and 1x Chinchilla; (iv) Muon best at 1-4x Chinchilla but Soap/Kron overtake at 8x and above; (v) rankings can flip during training due to LR decay, so compare only final checkpoints; (vi) Sophia gave no significant speedup under 0.5B once the data loader was fully randomized. — [arXiv:2509.02046](https://arxiv.org/abs/2509.02046), full text read via the PDF
- "Benchmarking Optimizers for LLM Pretraining" (Semenov, Pagliardini, Jaggi, arXiv:2509.01440, Sep 2025): standardized comparison across model size, batch size and duration with careful tuning, code released; the abstract names no winner (I did not read the body, so its ranking is a gap). — [arXiv:2509.01440](https://arxiv.org/abs/2509.01440)

Production practice (verified)
- DeepSeek-V3 (671B total / 37B active, 14.8T tokens) pretrained with AdamW, beta1 = 0.9, beta2 = 0.95, weight decay 0.1, gradient clip 1.0, batch-size ramp 3072 to 15360; LR schedule: 2K-step linear warmup to 2.2e-4, constant until 10T tokens, cosine decay to 2.2e-5 over 4.3T tokens, then two constant stages (2.2e-5, then 7.3e-6). That is a warmup-stable-decay-like shape with a cosine tail. — [arXiv:2412.19437](https://arxiv.org/abs/2412.19437), hyperparameters read from the PDF text
- Wen et al. state AdamW remains dominant and that DeepSeek, Qwen and Llama 3 had not adopted the new optimizers, with Kimi K2 the notable exception — [arXiv:2509.02046](https://arxiv.org/abs/2509.02046)
- Warmup-Stable-Decay (WSD) schedule proposed in MiniCPM (Hu et al., arXiv:2404.06395, Apr 2024): stable phase then short decay; supports continual training and cheap data/model scaling-law studies without a pre-fixed horizon. — [arXiv:2404.06395](https://arxiv.org/abs/2404.06395)
- Llama 3 (arXiv:2407.21783) exists and is the stated AdamW-based baseline family per Wen et al.; I did not read its training section myself. — [arXiv:2407.21783](https://arxiv.org/abs/2407.21783)

### Inferences
- The consistent pattern is "tuned-baseline effect": gains shrink when the baseline gets equal tuning budget and final-checkpoint evaluation. Original Muon/SOAP/Sophia numbers (1.4-2x) and the independent 1.1-1.4x are both likely right for their regimes; the disagreement is protocol, not fraud.
- The decay with scale (1.4x to 1.1x) is only measured to 1.2B by Wen et al., while the positive large-scale evidence (Moonlight 16B-total, Kimi K2 1T-total) comes from the optimizer's developers without a matched tuned-AdamW run at that scale in what I read. So "does Muon win at frontier scale" is unresolved: one sees a shrinking edge, the other a deployed choice.
- Schedule-free and Shampoo wins come from a different protocol (AlgoPerf: diverse workloads, not LLM-only), so they transfer to LLM pretraining only loosely.
- Practical hedge: matrix-optimizer cost matters (extra FLOPs, distribution of Newton-Schulz/eigendecompositions); a 1.1x step speedup can be net zero in wall-clock.

### Gaps
- GLM-4.5 (arXiv:2508.06471) reportedly used Muon [unverified: the abstract page I fetched does not state the optimizer; I did not read the body]. Same for any other 2025-26 Chinese lab Muon adoption beyond Moonshot.
- I did not fetch Distributed Shampoo / Meta's production Shampoo paper (arXiv id not verified), so claims about Shampoo usage at Meta are [unverified].
- No verified source on Google/OpenAI/Anthropic/Meta current optimizers beyond Llama 3/DeepSeek; closed labs are assumed to use AdamW-family [unverified].
- Body of Semenov et al. (2509.01440) not read, so its specific rankings are missing.
- Sophia and Lion large-scale production adoption: none found (Lion only the author-stated ads-model use).

## Normalization, regularization, initialization, and parameterization (muP, hyperparameter transfer)

### Takeaway
Pre-norm RMSNorm Transformers are the standard LLM recipe; BatchNorm's benefit is optimization smoothing rather than the originally claimed covariate-shift fix; normalization-free Transformers (DyT) are a 2025 result. muP gives zero-shot learning-rate transfer across width, and depth-wise transfer (CompleteP) is an active 2025 line, though other parameterizations can also transfer.

### Cited Findings
Normalization
- BatchNorm (Ioffe and Szegedy, arXiv:1502.03167, 2015): normalizes layer inputs per mini-batch; allows much higher learning rates and less careful initialization; same ImageNet accuracy in 14x fewer steps; acts as a regularizer. — [arXiv:1502.03167](https://arxiv.org/abs/1502.03167)
- "How Does Batch Normalization Help Optimization?" (Santurkar et al., arXiv:1805.11604, 2018): the internal-covariate-shift story is not what drives BN's success; its main effect is a significantly smoother optimization landscape (more predictive, stable gradients). — [arXiv:1805.11604](https://arxiv.org/abs/1805.11604)
- LayerNorm (Ba, Kiros, Hinton, arXiv:1607.06450, 2016): normalization over features within each example, independent of batch size. Details beyond title/date not fetched. — [arXiv:1607.06450](https://arxiv.org/abs/1607.06450)
- RMSNorm (Zhang and Sennrich, arXiv:1910.07467, 2019): drops re-centering, scales by root mean square; comparable to LayerNorm, reduces running time 7%-64% depending on model. — [arXiv:1910.07467](https://arxiv.org/abs/1910.07467)
- Pre-LN vs Post-LN (Xiong et al., arXiv:2002.04745, 2020, ICML): with Post-LN, gradients near the output are large at initialization, so warmup is needed; Pre-LN has well-behaved gradients at initialization and trains without warmup with comparable results. — [arXiv:2002.04745](https://arxiv.org/abs/2002.04745)
- NFNets (Brock et al., arXiv:2102.06171, 2021): normalizer-free ResNets stabilized by adaptive gradient clipping; match EfficientNet-B7 accuracy up to 8.7x faster; 86.5% ImageNet top-1. — [arXiv:2102.06171](https://arxiv.org/abs/2102.06171)
- "Transformers without Normalization" (Zhu, Chen, He, LeCun, Liu, arXiv:2503.10622, 2025, CVPR): replace norm layers by Dynamic Tanh tanh(alpha x); matches or exceeds normalized Transformers across vision and language, mostly without retuning. — [arXiv:2503.10622](https://arxiv.org/abs/2503.10622)

Regularization
- Dropout (Hinton et al., arXiv:1207.0580, 2012): randomly omit half of feature detectors per case to prevent co-adaptation. — [arXiv:1207.0580](https://arxiv.org/abs/1207.0580)
- Weight decay: see AdamW above (arXiv:1711.05101). Wen et al. observe parameter norms track the learning-rate decay when weight decay is on (a shared phenomenon across optimizers) — [arXiv:2509.02046](https://arxiv.org/abs/2509.02046)
- SAM (Foret et al., arXiv:2010.01412, 2020): min-max objective that minimizes loss within a neighborhood (loss value and sharpness together); gains on CIFAR, ImageNet, fine-tuning; robustness to label noise. Costs roughly two gradient computations per step [unverified from abstract]. — [arXiv:2010.01412](https://arxiv.org/abs/2010.01412)
- Rethinking generalization (Zhang et al., arXiv:1611.03530, 2016): standard CNNs fit random labels and noise inputs, largely unchanged by explicit regularization, so capacity/regularizer-based explanations are insufficient. — [arXiv:1611.03530](https://arxiv.org/abs/1611.03530)
- Label smoothing: I could not verify its origin via the paper I fetched (Szegedy et al., arXiv:1512.00567, "Rethinking the Inception Architecture"; the abstract page does not mention label smoothing though the paper is commonly cited as the source) — [arXiv:1512.00567](https://arxiv.org/abs/1512.00567) [attribution unverified].

Parameterization and transfer
- muP / muTransfer (Yang et al., "Tensor Programs V", arXiv:2203.03466, 2022): under Maximal Update Parametrization many optimal hyperparameters stay stable with width; tune a small proxy and transfer zero-shot. Reported: 13M proxy beats published BERT-large at tuning cost equal to one BERT-large pretrain; 40M proxy beats published GPT-3 6.7B at about 7% of its pretraining cost. Code microsoft/mup. — [arXiv:2203.03466](https://arxiv.org/abs/2203.03466)
- "Scaling Exponents Across Parameterizations and Optimizers" (Everett et al., arXiv:2407.05872, 2024, ICML): tens of thousands of models up to 26.8B, three optimizers, four parameterizations; hyperparameter transfer is achievable under parameterizations other than muP; a per-layer LR prescription for standard parameterization outperforms muP; Adam epsilon must be scaled to avoid gradient underflow; proposes Adam-atan2 (no epsilon). — [arXiv:2407.05872](https://arxiv.org/abs/2407.05872)
- CompleteP (Dey et al., arXiv:2505.01618, 2025, NeurIPS 2025): some parameterizations lose depth-wise LR transfer or stay in a "lazy" regime; CompleteP claims both depth-wise transfer and non-lazy learning in all layers, 12-34% compute-efficiency improvements over prior state of the art. — [arXiv:2505.01618](https://arxiv.org/abs/2505.01618)
- muP combined with Muon in Essential AI's work (above) — [arXiv:2505.02222](https://arxiv.org/abs/2505.02222)
- Warmup: Porian et al. find warmup duration is one of three factors behind scaling-law discrepancies (see next section) — [arXiv:2406.19146](https://arxiv.org/abs/2406.19146)

### Inferences
- Taken together, Pre-LN, RMSNorm and weight decay are the "boring defaults" because they are cheap and stable; the research frontier is whether normalization can be dropped (DyT) and whether parameterization (muP, CompleteP) can remove per-scale tuning.
- Since Wen et al. show optimal hyperparameters differ per optimizer and Everett et al. show many parameterizations transfer, "muP is required for transfer" is too strong; "some principled per-layer scaling plus correct epsilon/beta2 handling is required" fits the evidence better.
- Dropout is largely absent from modern single-epoch LLM pretraining; data-constrained multi-epoch regimes may revive regularization needs (see scaling section) [inference, not sourced].

### Gaps
- Current LLM norm placement beyond Pre-LN (QK-norm, sandwich norm, Post-norm revivals) not researched and unverified.
- Data augmentation: no specific paper fetched; claims about augmentation are not covered.
- Dropout/label-smoothing usage rates in 2025-26 production models: no source.
- SAM adoption at scale and cost: not verified; a known critique literature exists but I did not fetch it.
- Whether DyT is used in any production LLM: no evidence found.

## What do scaling laws predict, where have they been revised, and what are their limits?

### Takeaway
Kaplan 2020 found smooth power laws and favored big models on modest data; Hoffmann 2022 (Chinchilla) revised to scale parameters and tokens equally; the Kaplan-Chinchilla gap was later attributed to implementation details (last-layer FLOPs, warmup, optimizer tuning). Later work revises Chinchilla for inference cost (train smaller for longer), for finite data (repetition up to about 4 epochs is nearly free), and shows the fitted parametric forms can be poorly estimated.

### Cited Findings
- Kaplan et al. (arXiv:2001.08361, Jan 2020): cross-entropy loss is a power law in model size, data and compute across more than seven orders of magnitude; architectural details like width/depth matter little; larger models are more sample-efficient so compute-optimal training uses very large models and stops well short of convergence. — [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)
- Hoffmann et al., Chinchilla (arXiv:2203.15556, Mar 2022): 400+ models (70M to 16B+ params, 5-500B tokens); model size and tokens should scale in equal proportion; 70B Chinchilla trained on about 4x the data of 280B Gopher at equal compute beat Gopher, GPT-3, Jurassic-1 and MT-NLG on downstream tasks; 67.5% MMLU. The "about 20 tokens per parameter" rule of thumb is not stated in the abstract [unverified here]. — [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)
- Porian et al. (arXiv:2406.19146, Jun 2024, NeurIPS 2024 Spotlight): Kaplan and Hoffmann predict different compute-optimal model sizes; reproducing Kaplan, the gap is attributed to last-layer computational cost, warmup duration and scale-dependent optimizer tuning; after fixing these the results agree closely with Chinchilla; careful LR decay is not essential for the law (contra a Hoffmann et al. hypothesis); tuning AdamW beta2 matters at low batch size. — [arXiv:2406.19146](https://arxiv.org/abs/2406.19146)
- Besiroglu et al., "Chinchilla Scaling: A replication attempt" (arXiv:2404.10102, Apr 2024): Hoffmann's third estimation method (parametric fit to data reconstructed from plots) is inconsistent with their first two methods and does not fit the extracted data; the reported confidence intervals are implausibly narrow (would need over 600,000 experiments vs fewer than 500 run); their rederivation is compatible with methods one and two. — [arXiv:2404.10102](https://arxiv.org/abs/2404.10102)
- Sardana et al., "Beyond Chinchilla-Optimal" (arXiv:2401.00448, Dec 2023, ICML 2024): including inference cost, models expecting about 1B requests should be smaller and trained longer than Chinchilla-optimal; 47 models show quality still improving up to 10,000 tokens per parameter; laws fit at typical ratios overestimate gains at extreme ratios. — [arXiv:2401.00448](https://arxiv.org/abs/2401.00448)
- Muennighoff et al., data-constrained scaling (arXiv:2305.16264, May 2023, rev. Jun 2025): about 400 runs up to 900B tokens / 9B params; up to 4 epochs of repeated data cost negligible loss at fixed compute; further repetition gives value of added compute falling to zero; proposes a compute-optimal law with diminishing value of repeated tokens and excess parameters; mitigations: code data, dropping quality filters. — [arXiv:2305.16264](https://arxiv.org/abs/2305.16264)
- MiniCPM (arXiv:2404.06395) reports a compute-optimal data-to-model ratio well above Chinchilla's, using WSD to study scaling cheaply — [arXiv:2404.06395](https://arxiv.org/abs/2404.06395)
- Optimizer-dependence: Wen et al. show optimizer speedups shrink with scale and that rankings flip with data-to-model ratio, so scaling-law constants are optimizer- and tuning-dependent — [arXiv:2509.02046](https://arxiv.org/abs/2509.02046)
- Hyperparameter scaling laws: Porian et al. derive laws for optimal LR and batch size — [arXiv:2406.19146](https://arxiv.org/abs/2406.19146)

### Inferences
- "Scaling law" is a family of fitted empirical regressions whose constants depend on tokenizer, data mix, learning-rate schedule, optimizer tuning, and parametrization; the Kaplan vs Chinchilla episode is the clearest case of an apparent scientific disagreement that was an experimental-protocol artifact.
- Chinchilla-optimal is compute-optimal for training only; production labs over-train smaller models for inference economics (supported by Sardana et al.; the specific token counts of Llama 3 etc. I did not verify).
- Extrapolation risk: Besiroglu et al. and Sardana et al. both show fits are fragile at the edges (narrow CIs from few runs; extreme token ratios).
- Scaling laws predict pretraining loss reliably; mapping loss to downstream capability, or to post-training and RL-era gains, is a separate and weaker claim [inference, not sourced].

### Gaps
- Scaling laws for downstream/emergent capabilities, MoE (e.g. sparsity-aware laws), test-time compute, multimodal, and over-training laws (e.g. arXiv:2403.08540, recalled but NOT verified) not researched.
- Specific 2025-26 revisions beyond the above (e.g. data-quality-aware laws, "scaling wall" debates) not searched; any such claim is [unverified].
- No verification of the "20 tokens/param" figure or of actual token/param ratios of 2025 frontier models.

## What is understood vs mysterious about generalization (double descent, grokking, edge of stability, lottery tickets, NTK vs feature learning, loss landscape, implicit regularization)?

### Takeaway
Descriptive phenomena are solid (double descent, grokking, edge of stability, lottery tickets), and some have mechanisms in toy/algorithmic settings (modular-addition circuits, weight-norm LU picture, kernel regimes). What is not established is a predictive theory for generalization of realistic networks at scale; the NTK/lazy regime is understood mathematically but is not where useful deep learning happens.

### Cited Findings
Generalization puzzle and double descent
- Zhang et al. (arXiv:1611.03530): CNNs trained by SGD fit random labels and noise, largely regardless of explicit regularization, so classical complexity arguments fail. — [arXiv:1611.03530](https://arxiv.org/abs/1611.03530)
- Belkin et al. (arXiv:1812.11118, 2018): proposes the double-descent curve unifying the classical U-shape with improvement past the interpolation threshold; evidence across models and datasets and a proposed mechanism. — [arXiv:1812.11118](https://arxiv.org/abs/1812.11118)
- Nakkiran et al., "Deep Double Descent" (arXiv:1912.02292, Dec 2019): double descent occurs for model size and for training epochs; defines "effective model complexity"; regimes where more data (up to 4x) hurts test performance. — [arXiv:1912.02292](https://arxiv.org/abs/1912.02292)

Grokking
- Power et al. (arXiv:2201.02177, Jan 2022): on small algorithmic datasets networks can jump from chance to perfect generalization long after fitting the training set; smaller datasets need more optimization. — [arXiv:2201.02177](https://arxiv.org/abs/2201.02177)
- Nanda et al. (arXiv:2301.05217, Jan 2023): small transformers on modular addition learn a Fourier/trig-identity algorithm; training has memorization, circuit formation and cleanup phases; grokking is gradual amplification of structured mechanisms, not a sudden switch. — [arXiv:2301.05217](https://arxiv.org/abs/2301.05217)
- Liu, Michaud, Tegmark, "Omnigrok" (arXiv:2210.01117, Oct 2022): attributes grokking to a train/test loss mismatch as a function of weight norm (the "LU mechanism"); explains dependence on dataset size and weight decay; induces grokking on image, language and molecule tasks and removes it on algorithmic data. — [arXiv:2210.01117](https://arxiv.org/abs/2210.01117)

Optimization dynamics and landscape
- Edge of Stability (Cohen et al., arXiv:2103.00065, 2021, ICLR): full-batch gradient descent typically drives the top Hessian eigenvalue to just above 2/step-size; training loss is non-monotonic short-term but decreases long-term; conflicts with classical optimization assumptions. — [arXiv:2103.00065](https://arxiv.org/abs/2103.00065)
- Loss landscape visualization (Li et al., arXiv:1712.09913, 2017): filter-normalized plots; skip connections give easier-to-train landscapes; batch size, LR and optimizer shape the minimizers found; reasons for the link to generalization "not well understood". — [arXiv:1712.09913](https://arxiv.org/abs/1712.09913)
- BN smooths the landscape rather than reducing covariate shift — [arXiv:1805.11604](https://arxiv.org/abs/1805.11604)
- SAM operationalizes the flat-minima idea — [arXiv:2010.01412](https://arxiv.org/abs/2010.01412)

Lottery tickets
- Frankle and Carbin (arXiv:1803.03635, 2018, ICLR 2019): dense random-init nets contain subnetworks ("winning tickets", 10-20% of size on MNIST/CIFAR-10 FC and conv nets) that train in isolation to comparable accuracy in similar iterations, owing to their initialization. — [arXiv:1803.03635](https://arxiv.org/abs/1803.03635)

NTK vs feature learning
- NTK (Jacot et al., arXiv:1806.07572, 2018): infinite-width nets are Gaussian processes at init; under gradient descent the function evolves by a kernel gradient with a kernel that becomes constant in the infinite-width limit; positive-definiteness result on the sphere; least-squares training is linear, fastest along top kernel components. — [arXiv:1806.07572](https://arxiv.org/abs/1806.07572)
- Chizat, Oyallon, Bach, "On Lazy Training" (arXiv:1812.07956, 2018, NeurIPS 2019): lazy training (near-linearized, kernel-like) arises from a scaling choice, not only overparameterization; deep CNNs trained lazily performed worse, so lazy training is unlikely to explain deep learning's success on hard tasks. — [arXiv:1812.07956](https://arxiv.org/abs/1812.07956)
- Feature-learning parameterization: muP maximizes feature learning (Yang et al., arXiv:2203.03466); CompleteP adds a non-lazy requirement for depth — [arXiv:2203.03466](https://arxiv.org/abs/2203.03466), [arXiv:2505.01618](https://arxiv.org/abs/2505.01618)

Implicit regularization
- Wen et al. and Porian et al. document that hyperparameters, data order randomization and schedule interact with outcomes (optimization choices change what is found) — [arXiv:2509.02046](https://arxiv.org/abs/2509.02046), [arXiv:2406.19146](https://arxiv.org/abs/2406.19146). Li et al. (landscape) state the optimizer/LR/batch-size effect on the minimizers found — [arXiv:1712.09913](https://arxiv.org/abs/1712.09913)

### Inferences
- Understood well: why training is stable with BN/Pre-LN (gradient scale, landscape smoothness); infinite-width limits (NTK, muP); mechanistic grokking on modular arithmetic.
- Partly understood: double descent (clean in linear/kernel models; mechanism for deep nets is conjectural), edge of stability (empirical regularity; theory exists for simplified settings, [unverified] from my fetches), grokking beyond algorithmic data.
- Mysterious: why SGD-found solutions of real overparameterized nets generalize, why scaling laws have the exponents they do, and what lottery-ticket sparsity says about optimization vs architecture.
- The EoS result and Adam-type/matrix optimizers: the 2/eta stability threshold is for GD; how it generalizes to preconditioned optimizers (Adam, Muon) is the kind of question the optimizer benchmarks leave open [inference].

### Gaps
- Not verified: theory papers behind implicit regularization toward max-margin/low-rank, the Hessian-spectrum and loss-of-plasticity literature, critiques of SAM, double descent in LLMs, grokking in non-algorithmic LLM settings, and the later lottery-ticket "stabilization" results.
- No 2025-26 survey fetched; coverage of the newest theory is thin.

## Open problems

### Takeaway
The biggest open items are (1) whether any optimizer beats well-tuned AdamW at frontier scale beyond small percentages, (2) reliable, protocol-independent scaling-law prediction, (3) a predictive theory of generalization/feature learning for real networks, and (4) principled hyperparameter transfer across width, depth, data and optimizer.

### Cited Findings
- Muon's own author lists open questions: scaling beyond 20B parameters / 1T tokens, distributing Newton-Schulz, fine-tuning and RL — [Jordan blog](https://kellerjordan.github.io/posts/muon/)
- Optimizer comparisons: need rigorous tuning, multiple scales and data ratios, final-checkpoint evaluation; the speedup-vs-scale trend beyond 1.2B is extrapolated by a fitted law — [arXiv:2509.02046](https://arxiv.org/abs/2509.02046)
- Benchmark protocol problems (stopping criteria, workload sensitivity, tuning fairness) motivate AlgoPerf — [arXiv:2306.07179](https://arxiv.org/abs/2306.07179)
- Depth-wise hyperparameter transfer without lazy learning (CompleteP) is recent, so its generalization across architectures is unsettled — [arXiv:2505.01618](https://arxiv.org/abs/2505.01618)
- Scaling-law estimation: replication shows fit methodology can mislead — [arXiv:2404.10102](https://arxiv.org/abs/2404.10102); data exhaustion and repetition limits — [arXiv:2305.16264](https://arxiv.org/abs/2305.16264)
- Loss-landscape/generalization link not understood — [arXiv:1712.09913](https://arxiv.org/abs/1712.09913)
- EoS conflicts with classical optimization assumptions — [arXiv:2103.00065](https://arxiv.org/abs/2103.00065)

### Inferences
- Concrete open questions: (a) matched-tuning AdamW vs Muon/SOAP at 7B-1T with fully randomized data loading; (b) whether matrix preconditioners' benefit is a small-scale/short-horizon artifact (as Wen et al. suggest) or a real efficiency gain at MoE scale (as Moonshot reports); (c) optimizer-specific scaling laws; (d) a unified theory linking steepest-descent-under-a-norm optimizers, muP and EoS; (e) schedule-free / WSD versus cosine at long horizons; (f) why edge-of-stability-like and sharpness dynamics matter for stochastic, preconditioned training.
- For a practitioner: the safe defaults supported by verified sources are AdamW (beta2 about 0.95, weight decay about 0.1 as in DeepSeek-V3), Pre-LN/RMSNorm, warmup + long stable phase + decay, tune LR at small scale with a parameterization that transfers, and treat any claimed optimizer speedup above about 1.2x as needing a matched-tuning check.

### Gaps
- No verified source for closed-lab practice; no verified evidence on Muon at scale from independent (non-developer) groups other than Essential AI up to 4B and Wen et al. up to 1.2B.
- Missing coverage of second-order methods in practice beyond K-FAC/Shampoo abstracts (e.g. natural gradient in RL/science settings), and of newer 2026 optimizers; a search for 2026 work was not done, so the picture may be incomplete as of Nov 2026.
