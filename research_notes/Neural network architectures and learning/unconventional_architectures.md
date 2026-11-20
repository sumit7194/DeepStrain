# Unconventional and alternative neural-network architectures

Method note: ids/titles/authors/dates below were checked against the arXiv API feed (export.arxiv.org, two batch queries) or individual abs pages, unless tagged [unverified]. Mechanism descriptions ("plain terms") restate the paper abstracts plus standard background; where a claim goes beyond the abstract it is tagged [background]. Maturity labels (production / established research / frontier / speculative) are my classification (Inference), not sourced facts. Third query batch hit arXiv HTTP 429, so a few candidate ids were dropped rather than cited unchecked.

## Q1. Which of these have demonstrated advantages at scale, and which remain niche — and why?

### Takeaway
Only sparse MoE is unambiguously production-scale at the frontier (DeepSeek-V3, Kimi K2, Mixtral, gpt-oss, Qwen3, LFM2-MoE). Everything else is either (a) absorbed into mainstream models under another name (neural ODE -> flow matching/SDE diffusion; fast weights -> linear attention/test-time memory; Hopfield -> attention interpretation; liquid-style gating -> hybrid conv/recurrent LMs), (b) niche but real (neural CDE/ODE-RNN for irregular time series, NCAs, neuromorphic hardware, neuro-symbolic theorem proving), or (c) faded (capsules).

### Cited Findings

#### 1. Mixture-of-experts (MoE) — PRODUCTION
- Plain terms: a layer has many parallel feed-forward "experts"; a learned router sends each token to only a few, so parameter count grows without proportional compute per token. Precise: sparsely-gated top-k MoE layer, up to thousands of experts, >1000x capacity — Shazeer et al. 2017, [arXiv:1701.06538](https://arxiv.org/abs/1701.06538).
- GShard (2020): annotation APIs + compiler support to shard a 600B MoE translation model across TPUs — [arXiv:2006.16668](https://arxiv.org/abs/2006.16668).
- Switch Transformer (2021): route each token to ONE expert (top-1), simplifying routing and enabling trillion-parameter training — [arXiv:2101.03961](https://arxiv.org/abs/2101.03961).
- Expert-choice routing (2022): experts pick tokens instead of tokens picking experts; balances load, >2x faster pretraining convergence — [arXiv:2202.09368](https://arxiv.org/abs/2202.09368).
- DeepSeekMoE (2024): fine-grained expert segmentation + shared experts for better specialization — [arXiv:2401.06066](https://arxiv.org/abs/2401.06066). Scaling laws for fine-grained MoE add a granularity parameter and find common expert-sizing practice suboptimal — [arXiv:2402.07871](https://arxiv.org/abs/2402.07871).
- Auxiliary-loss-free load balancing: dynamic routing bias instead of an auxiliary loss — [arXiv:2408.15664](https://arxiv.org/abs/2408.15664).
- Deployed scale: Mixtral 8x7B routes each token to 2 of 8 experts, 13B active params, matches/beats Llama 2 70B — [arXiv:2401.04088](https://arxiv.org/abs/2401.04088); DeepSeek-V3 671B total / 37B active, 14.8T tokens — [arXiv:2412.19437](https://arxiv.org/abs/2412.19437); Kimi K2 1T-parameter MoE — [arXiv:2507.20534](https://arxiv.org/abs/2507.20534); gpt-oss-120b/20b are MoE open-weight models — [arXiv:2508.10925](https://arxiv.org/abs/2508.10925); Qwen3 family [arXiv:2505.09388](https://arxiv.org/abs/2505.09388) (MoE variants: the abstract I retrieved did not state this; [unverified] from the abstract alone); LFM2 includes an 8.3B-total/1.5B-active MoE — [arXiv:2511.23404](https://arxiv.org/abs/2511.23404).
- Strengths: capacity/FLOP decoupling. Weaknesses [background]: memory footprint (all experts resident), load balancing, routing instability, communication cost across devices. Why it won: fits the existing transformer stack and GPU/TPU sharding with a drop-in FFN replacement.

#### 2. Implicit-depth models — neural ODE / SDE / DEQ — ESTABLISHED RESEARCH (ODE/SDE idea mainstream via generative models; DEQ niche)
- Neural ODE (2018): depth defined continuously via a black-box ODE solver; constant memory, adaptive computation — [arXiv:1806.07366](https://arxiv.org/abs/1806.07366). Link to ResNets: a residual block x+f(x) is an Euler step of dx/dt=f [background; ResNet arXiv:1512.03385 verified as Deep Residual Learning, He, Dec 2015, [link](https://arxiv.org/abs/1512.03385)].
- Limits: plain neural ODEs cannot represent some functions (flows can't cross); Augmented Neural ODEs fix it — [arXiv:1904.01681](https://arxiv.org/abs/1904.01681).
- Time series niche: Latent ODEs/ODE-RNNs for irregular sampling — [arXiv:1907.03907](https://arxiv.org/abs/1907.03907); Neural CDEs adjust trajectory as new observations arrive — [arXiv:2005.08926](https://arxiv.org/abs/2005.08926). Comprehensive treatment: Kidger, On Neural Differential Equations — [arXiv:2202.02435](https://arxiv.org/abs/2202.02435).
- Neural SDE: adjoint method extended to SDEs for memory-efficient gradients — [arXiv:2001.01328](https://arxiv.org/abs/2001.01328); SDEs as infinite-dimensional GANs (neural CDE discriminator) — [arXiv:2102.03657](https://arxiv.org/abs/2102.03657).
- The idea that "came back" at scale: score-based generative modeling through SDEs unifies score matching and diffusion — [arXiv:2011.13456](https://arxiv.org/abs/2011.13456); flow matching trains continuous normalizing flows (ODE-defined) simulation-free — [arXiv:2210.02747](https://arxiv.org/abs/2210.02747).
- DEQ (2019): find the fixed point of a weight-tied layer directly; equivalent to infinite depth with constant memory — [arXiv:1909.01377](https://arxiv.org/abs/1909.01377). Monotone-operator equilibrium nets guarantee a unique equilibrium and stable solver convergence — [arXiv:2006.08591](https://arxiv.org/abs/2006.08591). Training instability addressed by Jacobian regularization (ResNet-101-like speed/perf) — [arXiv:2106.14342](https://arxiv.org/abs/2106.14342).
- Strengths: memory O(1) in depth, adaptive compute. Weaknesses [background]: solver cost at inference, backprop through implicit function (Jacobian conditioning), stability — the reason regularization papers exist.
- Adoption evidence: I found no evidence of DEQs in a production frontier model [unverified/absence].

#### 3. Capsule networks — FADED (largely abandoned)
- Idea: neurons grouped into vectors (capsules) encoding entity pose; higher capsules receive votes from lower ones via iterative routing-by-agreement — Sabour/Frosst/Hinton, [arXiv:1710.09829](https://arxiv.org/abs/1710.09829) (better on overlapping digits). Unsupervised follow-up: Stacked Capsule Autoencoders, part-whole geometry — [arXiv:1906.06818](https://arxiv.org/abs/1906.06818). (Matrix capsules with EM routing, ICLR 2018: not on arXiv; not verified here.)
- Why it faded: AAAI 2023 analysis argues no work scaled CapsNets beyond small datasets, the parse-tree idea is not actually present in CapsNets, and a vanishing-gradient problem starves many capsules; best capsule ImageNet top-1 cited as 60.07% (Ahmed & Torresani 2019 — secondary citation, [unverified] directly) — [Mitterreiter et al., arXiv:2301.01583](https://arxiv.org/abs/2301.01583). Deep capsule nets show vanishing activations; research mostly refined routing on toy benchmarks — [arXiv:2305.11178](https://arxiv.org/abs/2305.11178) (via search snippet).
- Surviving legacy [Inference]: Hinton's part-whole/equivariance motivation; attention/ViT and data augmentation achieved the generalization capsules promised without iterative routing cost.

#### 4. Hypernetworks and fast weights — ESTABLISHED RESEARCH; fast-weight idea now mainstream as linear attention / test-time memory
- HyperNetworks: one network generates weights of another (non-shared LSTM weights, fewer params) — Ha et al., [arXiv:1609.09106](https://arxiv.org/abs/1609.09106).
- Fast weights for attending to the recent past — Ba et al., [arXiv:1610.06258](https://arxiv.org/abs/1610.06258). Linear transformers are equivalent to 1990s fast weight programmers; delta-rule update improves them — Schlag et al., [arXiv:2102.11174](https://arxiv.org/abs/2102.11174).
- Modern descendants: Gated DeltaNet (gating + delta rule, beats Mamba2/DeltaNet) — [arXiv:2412.06464](https://arxiv.org/abs/2412.06464); TTT layers (hidden state is a model updated at test time) — [arXiv:2407.04620](https://arxiv.org/abs/2407.04620); Titans (neural long-term memory beside attention, >2M-token contexts) — [arXiv:2501.00663](https://arxiv.org/abs/2501.00663); xLSTM (exponential gating, new memory structures) — [arXiv:2405.04517](https://arxiv.org/abs/2405.04517).
- Hypernetwork in practice 2025: Text-to-LoRA generates LoRA adapters from a task description in one pass — Charakorn et al., [arXiv:2506.06105](https://arxiv.org/abs/2506.06105).
- Maturity [Inference]: fast-weight/delta-rule mechanisms are frontier-to-emerging in LLM architectures; pure hypernetworks niche (adapter generation, conditioning).

#### 5. Energy-based models, modern Hopfield, dense associative memory — ESTABLISHED RESEARCH (theory) / frontier (as architecture)
- Dense associative memory: higher-order (rectified polynomial) interactions interpolate between feature matching and prototype regimes, linked to feedforward nets — Krotov & Hopfield, [arXiv:1606.01164](https://arxiv.org/abs/1606.01164). Exponential interaction functions give exponential storage capacity — Demircigil et al., [arXiv:1702.01929](https://arxiv.org/abs/1702.01929). Biologically plausible two-body-interaction formulation with hidden neurons — [arXiv:2008.06996](https://arxiv.org/abs/2008.06996).
- "Hopfield Networks is All You Need": continuous modern Hopfield net whose update rule equals transformer attention; usable as a layer — Ramsauer et al., [arXiv:2008.02217](https://arxiv.org/abs/2008.02217).
- Energy Transformer: attention layers that descend an engineered energy function — [arXiv:2302.07253](https://arxiv.org/abs/2302.07253). Hopfield-Fenchel-Young networks generalize energy functions via Fenchel-Young losses — [arXiv:2411.08590](https://arxiv.org/abs/2411.08590).
- Practical weakness [background]: energy-based training/sampling is costlier than feedforward; attention already gets retrieval for free. Adoption: used as an interpretation and design lens; no flagship production model built as an explicit Hopfield network found [unverified/absence].

#### 6. Spiking neural networks and neuromorphic hardware — FRONTIER (algorithms) / niche research hardware
- Surrogate-gradient training makes backprop possible through non-differentiable spikes (tutorial) — Neftci et al., [arXiv:1901.09948](https://arxiv.org/abs/1901.09948). SEW ResNet allows directly trained SNNs >100 layers — [arXiv:2102.04159](https://arxiv.org/abs/2102.04159). Spikformer: spiking self-attention without softmax/multiplication — [arXiv:2209.15425](https://arxiv.org/abs/2209.15425); Spikformer V2: first SNN result >80% ImageNet — [arXiv:2401.02020](https://arxiv.org/abs/2401.02020). SpikeGPT: spiking language model, linear-complexity attention — [arXiv:2302.13939](https://arxiv.org/abs/2302.13939).
- Large-model attempt: SpikingBrain-7B (linear) and 76B-A12B hybrid-linear MoE, trained on MetaX GPUs; reports >100x TTFT speedup at 4M tokens and 69.15% spiking sparsity — [arXiv:2509.05276](https://arxiv.org/abs/2509.05276), summary via [search](https://arxiv.org/abs/2509.05276v3). Follow-up SpikingBrain2.0 (5B, hybrid sparse attention, multiple hardware platforms) — [arXiv:2604.22575](https://arxiv.org/abs/2604.22575). Caveat [Inference]: these run on GPUs with spike-inspired encodings; efficiency on actual neuromorphic silicon is not demonstrated in these abstracts.
- Energy claims need care: an analytical hardware-independent estimate found tested SNNs 6-8x more efficient than feedforward counterparts — [arXiv:2210.13107](https://arxiv.org/abs/2210.13107) (title: Lemaire et al.; the paper's point is that efficiency depends on assumptions).
- Hardware: Intel Hala Point = 1,152 Loihi 2 chips, 1.15B neurons, 128B synapses, 140,544 cores, max 2,600 W, claims up to 20 petaops and >15 TOPS/W on conventional DNNs; a research prototype, not commercially available — [Intel newsroom](https://newsroom.intel.com/artificial-intelligence/intel-builds-worlds-largest-neuromorphic-system-to-enable-more-sustainable-ai) (headline figures are vendor claims). Nature review "Neuromorphic computing at scale" (Kudithipudi et al., Nature 637, 801-812, Jan 2025): scaling from proof-of-concept to real deployment faces challenges at every stack level and no one-size-fits-all solution — [search summary/press](https://www.eurekalert.org/news-releases/1071576).
- Original Loihi paper (Davies et al., IEEE Micro 2018): not re-checked [unverified].

#### 7. Liquid time-constant / closed-form continuous-time networks — NICHE research, with a commercial spin-out
- LTC: continuous-time recurrent nets with input-dependent time constants; stable, strong on time series — Hasani et al., [arXiv:2006.04439](https://arxiv.org/abs/2006.04439). CfC: closed-form approximation of the LTC dynamics avoids ODE solvers — [arXiv:2106.13898](https://arxiv.org/abs/2106.13898); published Nature Machine Intelligence 4(11), 992-1003 (2022) — [via search](https://arxiv.org/abs/2106.13898).
- Application: drone navigation out-of-distribution with liquid networks (Chahine et al.; venue not confirmed in my sources [unverified]) — [MIT CAP PDF](https://cap.csail.mit.edu/sites/default/files/research-pdfs/Robust%20flight%20navigation%20out%20of%20distribution%20with%20liquid%20neural%20networks.pdf).
- Commercial lineage: LFM2 (Liquid AI) is a hybrid of gated short convolutions + few GQA blocks found by hardware-in-the-loop search, 350M-8.3B, up to 2x faster CPU prefill/decode — [arXiv:2511.23404](https://arxiv.org/abs/2511.23404). Caveat [Inference]: the abstract describes conv+attention hybrids; it does not claim LTC/CfC cells inside, so "liquid" here is brand/lineage, not the ODE-RNN mechanism.
- Related continuous-time sequence line: Oscillatory SSM (LinOSS), forced harmonic oscillators, beats Mamba/LRU on long sequences — [arXiv:2410.03943](https://arxiv.org/abs/2410.03943).

#### 8. Neural cellular automata (NCA) — FRONTIER / niche (art, texture, morphogenesis, small reasoning)
- Growing NCA: differentiable CA learns a persistent, robust rule to grow a target from one cell — Mordvintsev et al., Distill 2020, doi:10.23915/distill.00023 (author order varies across catalogs) — [Distill thread](https://distill.pub/2020/selforg/), [DOI record](https://mlanthology.org/distill/2020/mordvintsev2020distill-growing).
- 2025: NCAs trained per-task as iterative rules for ARC-AGI grid tasks, framed as a model-of-computation study, not leaderboard chasing — Xu & Miikkulainen, [arXiv:2506.15746](https://arxiv.org/abs/2506.15746) (via search; ALIFE 2025).
- Strengths [background]: tiny parameter counts, local/parallel, self-repair. Weakness: training long unrolls is unstable; no large-scale benchmark wins found.

#### 9. Neuro-symbolic architectures — ESTABLISHED/FRONTIER; strongest results are LLM + formal verifier loops
- Differentiable logic: dILP "Learning Explanatory Rules from Noisy Data" — [arXiv:1711.04574](https://arxiv.org/abs/1711.04574); DeepProbLog (neural predicates in probabilistic logic programs) — [arXiv:1805.10872](https://arxiv.org/abs/1805.10872); Logic Tensor Networks — [arXiv:2012.13635](https://arxiv.org/abs/2012.13635); Scallop (Datalog-based neurosymbolic programming) — [arXiv:2304.04812](https://arxiv.org/abs/2304.04812); Differentiable Logic Gate Networks, trainable logic gates with very fast inference — [arXiv:2210.08277](https://arxiv.org/abs/2210.08277); convolutional version 86.29% CIFAR-10 — [arXiv:2411.04732](https://arxiv.org/abs/2411.04732).
- Program synthesis hybrid: DreamCoder (wake-sleep, library learning) — [arXiv:2006.08381](https://arxiv.org/abs/2006.08381).
- Neural theorem proving: AlphaGeometry (neural LM guiding a symbolic deduction engine, 100M synthetic theorems; 25/30 olympiad problems vs 10 prior best), Nature 625, 476-482 (2024), doi:10.1038/s41586-023-06747-5 — [Nature PDF](https://www.nature.com/articles/s41586-023-06747-5.pdf), [summary](https://research.google/pubs/solving-olympiad-geometry-without-human-demonstrations/); AlphaGeometry2 84% solve rate over 25 years of geometry problems — [arXiv:2502.03544](https://arxiv.org/abs/2502.03544) (id from search snippet; not abs-checked [unverified]). AlphaProof (RL in Lean, test-time RL; Nature, Nov 2025, doi:10.1038/s41586-025-09833-y; with AG2 earned 28/42 at IMO 2024) — [Google Research](https://research.google/pubs/olympiad-level-formal-mathematical-reasoning-with-reinforcement-learning/), [Nature table](https://www.nature.com/articles/s41586-025-09833-y/tables/1). Lean Copilot (LLM proof assistance inside Lean) — [arXiv:2404.12534](https://arxiv.org/abs/2404.12534); DeepSeek-Prover-V2 (subgoal decomposition + RL, Lean 4) — [arXiv:2504.21801](https://arxiv.org/abs/2504.21801).
- Why symbolic part works [Inference]: a formal checker gives a hard, noise-free reward/verification signal; the weak spot of "pure" differentiable-logic systems is scaling beyond benchmark-sized knowledge bases (no large-scale win found).

### Inferences
- Scale advantage correlates with fit to existing GPU/transformer infrastructure and with a clean training signal: MoE (drop-in FFN) wins; formal-proof RL (verifier reward) wins; capsules (iterative routing, no parallel-friendly form) lost; SNNs/neuromorphic lack mature hardware+software ecosystem so gains stay in research prototypes.
- "Niche" ideas often succeed by dissolving into mainstream methods (flow matching, linear attention/delta rule, Hopfield as attention lens), so asking whether the original architecture won is the wrong test.

### Gaps
- No verified evidence of DEQs, explicit Hopfield-layer models, or NCAs in any production system.
- Mixtral/Switch real-world serving cost comparisons not collected; Qwen3 MoE detail not read from abstract.
- Loihi 2018 paper and Matrix-capsules (EM routing) not verified; Ahmed & Torresani ImageNet number is second-hand.
- Mechanism descriptions beyond abstracts are background, not re-verified against full texts.

## Q2. What theoretical links connect them?

### Takeaway
The recurring pattern: iterate/descend a state to a fixed point or integrate it through time, and that single structure is what ties ODEs, DEQs, Hopfield nets, attention, fast weights and liquid RNNs together.

### Cited Findings
- Neural ODE <-> ResNet: continuous-depth limit; ODE solver defines depth — [arXiv:1806.07366](https://arxiv.org/abs/1806.07366); ResNet [arXiv:1512.03385](https://arxiv.org/abs/1512.03385).
- DEQ <-> implicit/fixed-point layers: infinitely deep weight-tied net = fixed point — [arXiv:1909.01377](https://arxiv.org/abs/1909.01377); uniqueness via monotone operators — [arXiv:2006.08591](https://arxiv.org/abs/2006.08591). Recurrent-depth LMs iterate a block in latent space for test-time compute — Geiping et al., [arXiv:2502.05171](https://arxiv.org/abs/2502.05171); recursive tiny networks beat HRM on ARC-AGI — [arXiv:2510.04871](https://arxiv.org/abs/2510.04871); HRM (27M params) — [arXiv:2506.21734](https://arxiv.org/abs/2506.21734).
- Hopfield <-> attention: modern Hopfield update equals transformer attention — [arXiv:2008.02217](https://arxiv.org/abs/2008.02217); Energy Transformer [arXiv:2302.07253](https://arxiv.org/abs/2302.07253); 2025: Hopfield hidden states added to self-attention reduce rank collapse — [arXiv:2511.20698](https://arxiv.org/abs/2511.20698). Linear attention as an iterated (classical) Hopfield network — informal note by Millidge, [beren.io](https://beren.io/2024-03-03-Linear-Attention-as-Iterated-Hopfield-Networks/) (blog, lower authority).
- Fast weights <-> linear attention: formal equivalence — [arXiv:2102.11174](https://arxiv.org/abs/2102.11174); delta-rule/gated variants [arXiv:2412.06464](https://arxiv.org/abs/2412.06464); TTT treats the hidden state as a model [arXiv:2407.04620](https://arxiv.org/abs/2407.04620); hypernetworks generate weights [arXiv:1609.09106](https://arxiv.org/abs/1609.09106).
- Liquid nets <-> ODE-RNNs: LTC are continuous-time RNNs with input-dependent time constants, CfC their closed-form approximation — [arXiv:2006.04439](https://arxiv.org/abs/2006.04439), [arXiv:2106.13898](https://arxiv.org/abs/2106.13898); ODE-RNN/neural CDE for irregular data [arXiv:1907.03907](https://arxiv.org/abs/1907.03907), [arXiv:2005.08926](https://arxiv.org/abs/2005.08926); SSMs (S4 [arXiv:2111.00396](https://arxiv.org/abs/2111.00396), Mamba selective SSM [arXiv:2312.00752](https://arxiv.org/abs/2312.00752), LinOSS [arXiv:2410.03943](https://arxiv.org/abs/2410.03943)) are continuous-time state-space lineage (S4/Mamba's discretized-ODE view is [background]).
- Diffusion/flows <-> neural ODE/SDE: [arXiv:2011.13456](https://arxiv.org/abs/2011.13456), [arXiv:2210.02747](https://arxiv.org/abs/2210.02747).
- Dense AM <-> feedforward nets: [arXiv:1606.01164](https://arxiv.org/abs/1606.01164).
- MoE <-> conditional computation: routing as sparse selection, with expert-choice reversing the selection direction — [arXiv:2202.09368](https://arxiv.org/abs/2202.09368).
- NCA <-> recurrence: an NCA is a weight-tied local recurrent update iterated to a stable pattern (cf. DEQ/recursive reasoning) [Inference]; applied to ARC in [arXiv:2506.15746](https://arxiv.org/abs/2506.15746).

### Inferences
- Unifying view: Hopfield/DEQ = fixed point of an update; neural ODE/liquid/SSM = integrate an update through time; fast weights/linear attention/TTT = update a state by a local learning rule per token; MoE/hypernetworks = input-dependent selection or generation of parameters. Several modern recurrent LMs combine two or more of these.
- Hopfield-as-attention explains attention as one-step retrieval, so extra Hopfield iterations are the DEQ-like depth direction [Inference, not shown in a single paper I read].

### Gaps
- Equivalence conditions (e.g., attention = Hopfield only with specific beta/identity projections; Ramsauer noted beta=1/sqrt(D) case per search summary) not checked against the full paper.
- No formal result found linking MoE routing to the others.

## Q3. Current (2024-2026) frontier results and open problems

### Takeaway
2024-26 frontier: open-weight MoE giants and fine-grained/aux-loss-free routing; recurrent/latent-depth reasoning (HRM, TRM, recurrent depth); test-time-memory recurrent layers (TTT, Titans, Gated DeltaNet); first billion-parameter spiking-inspired LLMs; formal-proof RL (AlphaProof, DeepSeek-Prover-V2). Open problems center on scalability, stability, and honest efficiency accounting.

### Cited Findings
- MoE: DeepSeek-V3 [arXiv:2412.19437](https://arxiv.org/abs/2412.19437), Kimi K2 [arXiv:2507.20534](https://arxiv.org/abs/2507.20534), gpt-oss [arXiv:2508.10925](https://arxiv.org/abs/2508.10925), aux-loss-free balancing [arXiv:2408.15664](https://arxiv.org/abs/2408.15664), granularity scaling laws [arXiv:2402.07871](https://arxiv.org/abs/2402.07871).
- Latent/recurrent depth: Geiping et al. [arXiv:2502.05171](https://arxiv.org/abs/2502.05171); HRM [arXiv:2506.21734](https://arxiv.org/abs/2506.21734) (claims strong Sudoku/maze/ARC with ~1,000 examples); TRM 7M params outperforms HRM and several LLMs on ARC-AGI per its abstract [arXiv:2510.04871](https://arxiv.org/abs/2510.04871); Continuous Thought Machines (neuron-level temporal processing, synchronization as representation) [arXiv:2505.05522](https://arxiv.org/abs/2505.05522).
- Memory/fast-weight LMs: TTT [arXiv:2407.04620](https://arxiv.org/abs/2407.04620), Titans [arXiv:2501.00663](https://arxiv.org/abs/2501.00663), Gated DeltaNet [arXiv:2412.06464](https://arxiv.org/abs/2412.06464), xLSTM [arXiv:2405.04517](https://arxiv.org/abs/2405.04517).
- Spiking: SpikingBrain [arXiv:2509.05276](https://arxiv.org/abs/2509.05276), SpikingBrain2.0 (Apr 2026) [arXiv:2604.22575](https://arxiv.org/abs/2604.22575); Spikformer V2 [arXiv:2401.02020](https://arxiv.org/abs/2401.02020); Hala Point [Intel](https://newsroom.intel.com/artificial-intelligence/intel-builds-worlds-largest-neuromorphic-system-to-enable-more-sustainable-ai); Nature 2025 review [Kudithipudi et al.](https://www.eurekalert.org/news-releases/1071576).
- Hopfield: NeurIPS 2025 paper on Hopfield hidden states in transformers [arXiv:2511.20698](https://arxiv.org/abs/2511.20698); Hopfield-Fenchel-Young [arXiv:2411.08590](https://arxiv.org/abs/2411.08590).
- Liquid/edge: LFM2 [arXiv:2511.23404](https://arxiv.org/abs/2511.23404).
- Hypernetworks: Text-to-LoRA [arXiv:2506.06105](https://arxiv.org/abs/2506.06105).
- Neuro-symbolic/theorem proving: AlphaProof (Nature 2025) [Google Research](https://research.google/pubs/olympiad-level-formal-mathematical-reasoning-with-reinforcement-learning/); DeepSeek-Prover-V2 [arXiv:2504.21801](https://arxiv.org/abs/2504.21801); Lean Copilot [arXiv:2404.12534](https://arxiv.org/abs/2404.12534); convolutional logic-gate nets [arXiv:2411.04732](https://arxiv.org/abs/2411.04732).
- NCA: ARC-AGI NCAs [arXiv:2506.15746](https://arxiv.org/abs/2506.15746).

### Inferences — open problems (my synthesis)
- MoE: routing stability, memory/communication cost, expert specialization measurement.
- Implicit/recurrent depth: stable training and cost of variable depth at LLM scale; whether HRM/TRM ARC results generalize beyond tiny, task-specific training (these are small-data, narrow-task results).
- Fast-weight/memory layers: whether test-time memory beats plain attention at long contexts under fair comparison.
- SNNs: demonstrating real wall-clock/energy wins on actual neuromorphic hardware for LLM-scale models, not GPU-simulated sparsity; software ecosystem.
- Capsules: no one has shown a dynamic parse-tree benefit at scale.
- Neuro-symbolic: automatic formalization and extending verifier-driven success beyond mathematics/geometry.
- Liquid/CfC: independent large-scale validation beyond robotics/time-series demos.

### Gaps
- No systematic 2025-26 survey of NCAs, DEQs, or capsules was retrieved; recent activity in those areas may be missed.
- ARC-AGI claims for HRM/TRM are abstract-level; independent replication status not checked.
- Real neuromorphic-hardware benchmark results (Loihi 2 on LLM-class workloads) not found.
