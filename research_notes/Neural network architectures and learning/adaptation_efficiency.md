# How neural networks adapt, compress and compute at inference

Verification legend. Every arXiv id below had its abstract page (or the arXiv API record) read on 2026-11-20 unless tagged. **[snippet]** = id and title confirmed only from a search-result listing, not the abstract page. **[unverified]** = not checked; do not rely on it. Mechanism descriptions marked **[background]** come from general knowledge, not from the cited abstract. Plain-language analogies are mine (inferences), not claims of the papers.

## Q0. Core ideas, mechanisms and key papers per family (reference catalogue)

### Takeaway
Eight families. Meta-learning and test-time training both treat "learning" as something done inside or at inference. Distillation, pruning, quantization and PEFT all exploit the same fact: trained networks are heavily redundant in parameter count and precision. Merging exploits the fact that fine-tunes of one pretrained model stay in one loss basin.

### Cited Findings

**Meta-learning**
- MAML: train initial parameters so a few gradient steps on a small new-task dataset give good performance (second-order unless approximated) — [Finn 2017, 1703.03400](https://arxiv.org/abs/1703.03400)
- Prototypical networks: classify new classes by distance to per-class prototype embeddings — [Snell 2017, 1703.05175](https://arxiv.org/abs/1703.05175)
- Learned optimizers: LSTM optimizer trained by gradient descent outperforms hand-designed ones on similar tasks — [Andrychowicz 2016, 1606.04474](https://arxiv.org/abs/1606.04474); scaled up as VeLO, a small network optimizer meta-trained at large scale, no hyperparameter tuning, open-sourced — [Metz 2022, 2211.09760](https://arxiv.org/abs/2211.09760)
- In-context learning (ICL) as learning: Transformers trained from scratch in-context learn linear and more complex function classes, matching task-specific algorithms — [Garg 2022, 2208.01066](https://arxiv.org/abs/2208.01066); trained self-attention Transformers implement gradient descent in the forward pass on regression tasks — [von Oswald 2022, 2212.07677](https://arxiv.org/abs/2212.07677)
- Instruction tuning over many tasks improves zero-shot generalization to unseen tasks (a practical multitask meta-learning flavour) — [FLAN, Wei 2021, 2109.01652](https://arxiv.org/abs/2109.01652)

**Continual learning**
- EWC: slow learning on weights important to earlier tasks (a quadratic penalty weighted by Fisher information **[background]**) — [Kirkpatrick 2016, 1612.00796](https://arxiv.org/abs/1612.00796)
- Replay: keep a small buffer of old data and rehearse; valued for simplicity and stability even with small memory — summarised in the LLM CL survey [Shi 2024, 2404.16789](https://arxiv.org/abs/2404.16789) **[snippet]**; companion survey grouping methods into replay / regularization / dynamic architecture — [Wu 2024, 2402.01364](https://arxiv.org/abs/2402.01364) **[snippet]**
- Loss of plasticity (separate from forgetting): see Q2.

**Test-time training / adaptation (TTT/TTA)**
- TENT: adapt a model at test time by minimizing prediction entropy on unlabeled test batches — [Wang 2020, 2006.10726](https://arxiv.org/abs/2006.10726)
- TTT layers: the RNN hidden state is itself a small model updated by self-supervised learning at test time; keeps improving with longer context unlike Mamba — [Sun 2024, 2407.04620](https://arxiv.org/abs/2407.04620)
- Titans: attention as short-term memory plus a learned neural long-term memory updated at test time; contexts beyond 2M tokens — [Behrouz 2024, 2501.00663](https://arxiv.org/abs/2501.00663)
- Linear attention = fast weight programmers: linearized self-attention is equivalent to 1990s fast-weight programmers; delta-rule update improves it — [Schlag 2021, 2102.11174](https://arxiv.org/abs/2102.11174)
- Mamba (selective state space, linear-time baseline that TTT layers are compared against) — [Gu 2023, 2312.00752](https://arxiv.org/abs/2312.00752)
- TTT for few-shot LLM tasks: temporary per-task weight updates give up to 6x ARC accuracy over fine-tuned baselines, 53.0% on ARC public validation with an 8B LM (61.9% ensembled with program synthesis), BIG-Bench Hard 10-shot 50.5% to 57.8% — [Akyürek 2024, 2411.07279](https://arxiv.org/abs/2411.07279)
- Self-Adapting LMs (SEAL): model writes its own finetuning data/"self-edits", RL rewards edits that improve the updated model — [Zweiger 2025, 2506.10943](https://arxiv.org/abs/2506.10943)
- Test-Time Learning for LLMs: perplexity-minimizing updates on unlabeled test data with LoRA, claims ≥20% over base on domain tasks — [2505.20633](https://arxiv.org/abs/2505.20633) **[snippet]**; In-Place TTT repurposes MLP output projections as fast weights — [2604.06169](https://arxiv.org/abs/2604.06169) **[snippet]**
- Zeroth-order MeZO fine-tunes with forward passes only, near-inference memory — [Malladi 2023, 2305.17333](https://arxiv.org/abs/2305.17333)

**Test-time compute**
- Chain-of-thought prompting — [Wei 2022, 2201.11903](https://arxiv.org/abs/2201.11903); self-consistency (sample many paths, keep the most consistent answer) — [Wang 2022, 2203.11171](https://arxiv.org/abs/2203.11171); Tree of Thoughts (explore/evaluate/backtrack) — [Yao 2023, 2305.10601](https://arxiv.org/abs/2305.10601); LATS (MCTS with LM value functions) — [Zhou 2023, 2310.04406](https://arxiv.org/abs/2310.04406)
- Verifiers: process supervision beats outcome supervision on MATH; PRM800K released — [Lightman 2023, 2305.20050](https://arxiv.org/abs/2305.20050)
- Adaptive computation: PonderNet learns per-input number of steps — [Banino 2021, 2107.05407](https://arxiv.org/abs/2107.05407); Universal Transformers (recurrent in depth, dynamic halting) — [Dehghani 2018, 1807.03819](https://arxiv.org/abs/1807.03819); Mixture-of-Depths (top-k routing spends compute unevenly across tokens) — [Raposo 2024, 2404.02258](https://arxiv.org/abs/2404.02258); Quiet-STaR (learn internal rationales per token) — [Zelikman 2024, 2403.09629](https://arxiv.org/abs/2403.09629)
- Recurrent depth: iterate a recurrent block in latent space instead of emitting more tokens — [Geiping 2025, 2502.05171](https://arxiv.org/abs/2502.05171)
- RL-trained reasoners: pure RL elicits strong math/code/STEM reasoning without human-labeled trajectories — [DeepSeek-R1, 2501.12948](https://arxiv.org/abs/2501.12948); s1: 1,000-example SFT plus "budget forcing" lets s1-32B beat o1-preview by up to 27% on MATH/AIME24 — [Muennighoff 2025, 2501.19393](https://arxiv.org/abs/2501.19393)

**Distillation**
- Hinton: compress an ensemble/teacher into a single model via soft targets — [2015, 1503.02531](https://arxiv.org/abs/1503.02531)
- Prune-then-distill for LLMs: pruning Nemotron-4 15B with <3% of original data gives 8B and 4B models with up to 40x fewer training tokens and up to 16% MMLU gain over training from scratch — [Minitron, Muralidharan 2024, 2407.14679](https://arxiv.org/abs/2407.14679)
- Distillation from a stronger teacher can add reasoning patterns that RL does not — [Yue 2025, 2504.13837](https://arxiv.org/abs/2504.13837)

**Sparsity / pruning**
- Lottery ticket: dense nets contain subnetworks <10–20% of size that train to comparable accuracy from their original init — [Frankle 2018, 1803.03635](https://arxiv.org/abs/1803.03635)
- Movement pruning beats magnitude pruning at high sparsity; with distillation keeps minimal loss at 3% of parameters — [Sanh 2020, 2005.07683](https://arxiv.org/abs/2005.07683)
- SparseGPT: one-shot, no retraining, 60% unstructured sparsity on OPT-175B/BLOOM-176B with negligible perplexity rise, under 4.5 h — [Frantar 2023, 2301.00774](https://arxiv.org/abs/2301.00774)
- Wanda: prune by |weight| x input activation, no retraining, beats magnitude pruning on LLaMA — [Sun 2023, 2306.11695](https://arxiv.org/abs/2306.11695)
- Structured/depth pruning: performance shows minimal degradation until up to half of the layers are removed in some open-weight models — [Gromov, 2403.17887](https://arxiv.org/abs/2403.17887)

**Quantization**
- LLM.int8(): mixed-precision int8 matmul, halves inference memory up to 175B with no degradation — [Dettmers 2022, 2208.07339](https://arxiv.org/abs/2208.07339)
- GPTQ: one-shot PTQ to 3–4 bits for 175B-parameter models in ~4 GPU-hours, ~3.25x (A100) to 4.5x (A6000) speedup over FP16 — [Frantar 2022, 2210.17323](https://arxiv.org/abs/2210.17323)
- AWQ: protecting ~1% salient weights enables 4-bit weight-only quantization — [Lin 2023, 2306.00978](https://arxiv.org/abs/2306.00978)
- 4-bit scaling laws: >35,000 zero-shot experiments, 3–8 bit, 19M–176B params; 4-bit almost universally optimal for total model bits vs accuracy — [Dettmers & Zettlemoyer, 2212.09720](https://arxiv.org/abs/2212.09720)
- BitNet: 1-bit weights trained from scratch (BitLinear) — [Wang 2023, 2310.11453](https://arxiv.org/abs/2310.11453); BitNet b1.58: ternary {-1,0,1} weights claimed to match FP16 Transformers in perplexity and task performance — [Ma 2024, 2402.17764](https://arxiv.org/abs/2402.17764)
- QAT vs PTQ: no abstract-verified QAT-specific primary paper was fetched here; see Gaps.

**PEFT**
- Adapters: within 0.4% of full fine-tuning on GLUE with 3.6% extra parameters per task — [Houlsby 2019, 1902.00751](https://arxiv.org/abs/1902.00751)
- LoRA: low-rank updates to frozen weights, 10,000x fewer trainable parameters and 3x less GPU memory than full GPT-3 175B fine-tuning — [Hu 2021, 2106.09685](https://arxiv.org/abs/2106.09685)
- QLoRA: 4-bit frozen base plus LoRA, 65B model on one 48 GB GPU — [Dettmers 2023, 2305.14314](https://arxiv.org/abs/2305.14314)

**Merging / averaging**
- Model soups: averaging weights of fine-tunes with different hyperparameters improves accuracy with no extra inference cost; ViT-G 90.94% ImageNet top-1 — [Wortsman 2022, 2203.05482](https://arxiv.org/abs/2203.05482)
- Task arithmetic: task vectors (fine-tuned minus base weights) can be negated or added to steer behaviour — [Ilharco 2022, 2212.04089](https://arxiv.org/abs/2212.04089)
- TIES-Merging: trim small updates, resolve sign conflicts, merge only aligned parameters — [Yadav 2023, 2306.01708](https://arxiv.org/abs/2306.01708)

### Inferences
- **Mechanism lines linking the families [background + inference]:** (1) ICL-as-gradient-descent (von Oswald) + fast-weight equivalence of linear attention (Schlag) + TTT layers/Titans form one chain: a recurrent state that is itself updated by a learning rule. TTT layers make the state an explicit model trained by self-supervised loss at each token. (2) MAML optimises for "good after a few gradient steps", which is exactly what test-time training (Akyürek) does at inference, so TTT is meta-learning without the outer loop unless trained for it. (3) Distillation, pruning and quantization are three ways of removing redundancy (teacher capacity, weights, bits) and are routinely stacked: Minitron is prune+distill; QLoRA is quantize+PEFT; movement pruning paired with distillation is the prune+distill pattern. (4) LoRA, task arithmetic, soups and TIES all assume updates live in a low-dimensional/linear region around the pretrained weights.
- **Analogy for a computer engineer:** quantization = lossy codec on weights; pruning = dropping rows from a sparse matrix; distillation = training a small model on the big model's outputs; LoRA = a patch file applied to a frozen binary; merging = three-way merge of patches; TTT = JIT-style learning during execution.

### Gaps
- No primary QAT paper, no HQQ/AQLM/QuIP-type 2-bit PTQ paper, no OBS/OBD classic pruning, no Reptile/MANN/learned-optimizer-at-LLM-scale source checked. GPTQ/AWQ/SmoothQuant details beyond abstract claims not read.
- Mixed-id items not checked and therefore excluded: ShortGPT (2403.03853), Rethinking the Value of Network Pruning (1810.05270), Shrink-and-Perturb (1910.08475), Model-merging survey (2408.07666), Simple scalable continual pretraining (2403.08763), quantization-of-reasoning-models (2504.04823), Llama-3 quantization study (2404.14047) — all **[unverified]**; the arXiv API was rate-limited (HTTP 429) during the final batch.

## Q1. What has been learned (2023–2026) about trading training compute for test-time compute?

### Takeaway
Test-time compute is a real second scaling axis: adaptive allocation of inference compute can let a smaller model beat a ~14x larger one on suitable problems, and tiny amounts of reasoning-specific SFT or RL unlock long chains. But returns are task-, difficulty- and verifier-dependent, flatten or reverse with long chains, and RL on verifiable rewards appears to sharpen sampling rather than add capability beyond the base model.

### Cited Findings
- Snell et al.: compute-optimal, per-prompt adaptive allocation beats a best-of-N baseline by >4x efficiency and lets a smaller model outperform one 14x larger (in FLOPs-matched settings) — [2408.03314](https://arxiv.org/abs/2408.03314)
- Repeated sampling: solution coverage grows with the number of samples, raising success on verifiable tasks such as SWE-bench Lite (coverage is only converted into accuracy if a verifier can pick the right sample) — [Brown 2024, 2407.21787](https://arxiv.org/abs/2407.21787)
- Inference scaling laws: spending more inference compute with search strategies can be more efficient than scaling model size; smaller models can match larger ones — [Wu 2024, 2408.00724](https://arxiv.org/abs/2408.00724)
- Process reward models beat outcome supervision on MATH — [2305.20050](https://arxiv.org/abs/2305.20050)
- Training for long thinking is cheap: s1 uses 1,000 examples and budget forcing (appending "Wait" or truncating) and reports beating o1-preview by up to 27% on MATH and AIME24 — [2501.19393](https://arxiv.org/abs/2501.19393)
- Pure RL (R1) elicits strong reasoning without human-labeled trajectories — [2501.12948](https://arxiv.org/abs/2501.12948)
- Counter-evidence on what RL adds: RLVR models beat base at k=1 but base models achieve higher pass@k at large k; coverage and perplexity analyses suggest RLVR gains stay within the base model's range; six RLVR algorithms perform similarly and fall short of the base model's potential; distillation can introduce new reasoning patterns — [Yue 2025, 2504.13837](https://arxiv.org/abs/2504.13837)
- Behaviours such as verification and backtracking explain why some models improve under RL; priming a model with them closes the gap — [Gandhi 2025, 2503.01307](https://arxiv.org/abs/2503.01307)
- Longer chains can hurt: "Inverse Scaling in Test-Time Compute" builds tasks (counting with distractors, regression with spurious features, deduction, advanced-AI-risk) where longer reasoning lowers accuracy; five failure modes identified (distractibility in Claude models, framing over-fit in o-series, drift to spurious correlations, loss of focus on deduction, amplified concerning behaviours); TMLR Dec 2025 — [Gema 2025, 2507.14417](https://arxiv.org/abs/2507.14417)
- Diminishing returns of "think longer": initial gains from extended thinking then fade; parallel sampling with majority voting proposed as alternative — [2506.04210](https://arxiv.org/abs/2506.04210) **[snippet]**; marginal utility falls at higher budgets (ACL 2026 Findings) — [2604.10739](https://arxiv.org/abs/2604.10739) **[snippet]**; models should match reasoning depth to compute budget — [2502.18080](https://arxiv.org/abs/2502.18080) **[snippet]**
- Latent-space alternative: a recurrent-depth model scales test-time compute by iterating a block rather than emitting tokens — [2502.05171](https://arxiv.org/abs/2502.05171); earlier adaptive-depth lineage in PonderNet, Universal Transformers, Mixture-of-Depths (links in Q0)
- Training-side trade: Chinchilla compute-optimal scaling (model size and tokens scale equally) is the baseline against which test-time trade-offs are framed — [Hoffmann 2022, 2203.15556](https://arxiv.org/abs/2203.15556)
- TTT gives a different test-time trade: spend inference FLOPs on gradient steps for the specific task (Akyürek numbers in Q0) — [2411.07279](https://arxiv.org/abs/2411.07279)

### Inferences
- Three distinct things share the name "test-time compute": (a) sampling and selection (best-of-N, self-consistency, verifier search) which needs a verifier; (b) longer serial chains which depend on the model having been trained to use them; (c) test-time weight updates (TTT). Evidence for (a) is strongest where answers are checkable; (b) shows inverse-scaling regimes; (c) is the least mature at LLM scale.
- Yue et al. plus s1 together suggest much "reasoning" is elicited latent ability; this implies compute is being moved from training to inference only up to the base model's coverage ceiling. Distillation (training on a stronger model's traces) is the verified route beyond it.
- The trade-off is not free: inference cost is paid per query, so it favours low-volume/hard queries; Snell's compute-optimal framing is the right lens. [inference; deployment economics not verified in sources]

### Gaps
- No verified head-to-head of training FLOPs vs inference FLOPs at matched total cost for frontier models; proprietary systems (o1 etc.) have no arXiv paper to cite.
- Reliability of reported "14x" depends on which problem difficulties and FLOPs accounting Snell used; full text not read.
- 2025–2026 papers tagged [snippet] need abstract-level verification before being quoted for numbers.

## Q2. How severe is loss of plasticity, and which continual-learning methods actually work at scale?

### Takeaway
Loss of plasticity is severe in long task sequences: on a continual ImageNet benchmark, standard backprop drops from 89% to 77% accuracy, about the level of a linear network, across many architectures and optimizers. Cheap fixes work in these settings (L2 plus weight perturbation; continual backprop's selective reinitialization), but evidence is mostly on small-to-mid benchmarks; at LLM scale the verified evidence concerns forgetting under fine-tuning, and no method has been shown to solve it in general.

### Cited Findings
- Dohare et al.: repurposing MNIST/ImageNet as task sequences, ImageNet binary classification accuracy fell from 89% on an early task to 77% by the 2000th task, roughly a linear network; effect seen across architectures, optimizers, activations, batch norm and dropout; L2 regularization substantially eases it, especially with weight perturbation; continual backprop (reinitialize a small fraction of less-used units after each example) appears to maintain plasticity indefinitely — [arXiv 2306.13812](https://arxiv.org/abs/2306.13812)
- The Nature version (vol. 632, pp. 768–774, Aug 2024, DOI 10.1038/s41586-024-07711-7) concludes the problem is not inherent to ANNs and that sustained learning needs a random non-gradient component — [search-result summary of the Nature paper](https://www.amii.ca/research/loss-of-plasticity-in-deep-continual-learning); journal page was behind an auth redirect so full text not read
- Continual Backprop precursor — [Dohare 2021, 2108.06325](https://arxiv.org/abs/2108.06325)
- Lyle et al.: plasticity loss is linked to loss-landscape curvature rather than saturated units; design choices that preserve it identified — [2303.01486](https://arxiv.org/abs/2303.01486)
- Dormant neurons in deep RL: agents accumulate inactive neurons that reduce expressivity; ReDo recycles them and improves performance; ICML 2023 oral — [Sokar 2023, 2302.12902](https://arxiv.org/abs/2302.12902)
- Forgetting at LLM scale: LoRA fine-tuning of Llama 2 7B chat still forgets; forgetting is strongly inversely linear with fine-tuning performance, grows as a shifted power law in trainable parameters and update steps, and cannot be avoided by early stopping — [Kalajdzievski 2024, 2401.05605](https://arxiv.org/abs/2401.05605)
- LoRA learns less on the target domain but forgets less of the base than full fine-tuning — [Biderman 2024, 2405.09673](https://arxiv.org/abs/2405.09673)
- Surveys: replay, regularization and architecture expansion are the three standard families; replay valued for simplicity and stability — [2404.16789](https://arxiv.org/abs/2404.16789), [2402.01364](https://arxiv.org/abs/2402.01364) (both **[snippet]**)
- EWC original claim: slowing learning on important weights lets sequential tasks be learned — [1612.00796](https://arxiv.org/abs/1612.00796)
- Weight-space route to avoiding forgetting: merging fine-tunes (soups, task arithmetic, TIES) — links in Q0
- Self-directed weight updates for LLMs (SEAL) frame continual adaptation as an RL problem — [2506.10943](https://arxiv.org/abs/2506.10943)

### Inferences
- Two separate failure modes: **forgetting** (old tasks degrade) and **plasticity loss** (new tasks become harder to learn). Methods aimed at one do not obviously fix the other; EWC addresses forgetting but adds constraint, which plausibly worsens plasticity [inference, untested in sources].
- The convergent practical picture: replay and low-rank/PEFT updates are what is used at LLM scale; EWC-style penalties are mostly benchmark-scale. Pretrained LLMs fine-tuned a few times mainly face forgetting, not plasticity loss, which needs very long non-stationary streams. [inference]
- Plasticity loss and the TTT/fast-weight line connect: a model that must keep learning at test time needs plasticity, so reinitialization/perturbation ideas may be relevant to long-context TTT. [speculative]

### Gaps
- No verified evidence on loss of plasticity in LLM pretraining/long fine-tuning streams, or continual backprop at LLM scale.
- Shrink-and-Perturb (1910.08475) and simple continual-pretraining recipes (2403.08763) not verified; replay-at-scale numbers not obtained.
- Nature paper's quantitative RL results not read (paywall/redirect).

## Q3. How far can compression (sparsity, low-bit) go without loss, and what does that imply about network redundancy?

### Takeaway
Large LLMs tolerate roughly 4-bit weights and ~50–60% unstructured sparsity almost for free post-training, and ternary (1.58-bit) models are claimed to match FP16 when trained that way; but there is a ceiling, and recent scaling-law work shows heavily-trained models are harder to quantize. Redundancy is real but shrinks as training data per parameter grows.

### Cited Findings
- 4-bit is almost universally optimal for total model bits vs zero-shot accuracy across 3–8 bits, 19M–176B params; only small blocks and data-type choice improved the trade-off — [2212.09720](https://arxiv.org/abs/2212.09720)
- GPTQ 3–4 bit on 175B-parameter models in ~4 GPU-hours — [2210.17323](https://arxiv.org/abs/2210.17323); int8 with no degradation to 175B — [2208.07339](https://arxiv.org/abs/2208.07339); ~1% salient weights matter for 4-bit — [2306.00978](https://arxiv.org/abs/2306.00978)
- 60% unstructured sparsity on OPT-175B/BLOOM-176B with negligible perplexity increase, one-shot — [2301.00774](https://arxiv.org/abs/2301.00774); Wanda no-retrain activation-aware pruning — [2306.11695](https://arxiv.org/abs/2306.11695)
- Retraining-era result: subnetworks <10–20% of size reach comparable accuracy from original init (small models) — [1803.03635](https://arxiv.org/abs/1803.03635); movement pruning + distillation retains accuracy at 3% of parameters — [2005.07683](https://arxiv.org/abs/2005.07683)
- Layer removal: minimal degradation until up to half the layers are removed in some open models — [2403.17887](https://arxiv.org/abs/2403.17887); pruning plus distillation gave 8B/4B models with 40x fewer tokens — [2407.14679](https://arxiv.org/abs/2407.14679)
- Ternary: BitNet b1.58 reports parity with FP16 in perplexity and task performance when trained at ternary — [2402.17764](https://arxiv.org/abs/2402.17764) (claims of the authors; independent replication not checked)
- Limits: post-training-quantization degradation grows with training data; extra pretraining can become "actively harmful"; training larger models in lower precision may be compute-optimal (>465 runs, up to 1.7B params / 26B tokens) — [Kumar 2024, 2411.04330](https://arxiv.org/abs/2411.04330)
- Same direction: over 1,500 quantized checkpoints; smaller models trained on much data show significant quantization-induced degradation; projections that low-bit quantization of models trained on >100T tokens "may NOT be desirable" (authors flag work in progress) — [Ouyang 2024, 2411.17691](https://arxiv.org/abs/2411.17691)
- Pruning/quantization interplay: Minitron pairs pruning with distillation recovery; QLoRA pairs 4-bit base with LoRA recovery (see Q0)

### Inferences
- Redundancy has two layers: (a) over-parameterization in count and depth (pruning, layer removal) and (b) over-precision in each weight (quantization). The scaling-law papers imply redundancy is not a constant of the architecture but a function of how undertrained a model is relative to its size: Chinchilla-style long training consumes the slack. [inference from 2411.04330, 2411.17691, 2203.15556]
- Trade-off structure: unstructured sparsity gives the best accuracy-per-parameter but little hardware speedup; structured/depth pruning and low-bit give speed but cost accuracy, so recovery (distillation or LoRA) is the usual companion. Since both quantization and pruning are mostly evaluated on perplexity and zero-shot accuracy, harder skills (long reasoning) may degrade earlier [unverified; the one study I intended to cite on this, 2504.04823, was not checked].
- The tension with test-time compute: heavily compressed small models plus large inference budgets (Snell, Wu) are an alternate frontier to a big model run once.

### Gaps
- No verified head-to-head of sparsity vs quantization at equal bits (e.g. 2307.02973) — id not verified.
- No verified numbers on sub-4-bit PTQ (2-bit) methods, nor independent replication of BitNet b1.58 at scale.
- Hardware realism (N:M sparsity kernels, actual speedups) not sourced.

## Q4. Open problems and surprising results

### Takeaway
The surprises cluster around "more is not monotone": longer thinking can hurt, more pretraining can hurt quantization, RL may not add capability, LoRA forgets less because it learns less, and a continual learner can silently drift to linear-network performance.

### Cited Findings
- Inverse scaling in test-time compute (Gema et al., TMLR) — [2507.14417](https://arxiv.org/abs/2507.14417)
- RLVR narrows rather than expands: base models win at large pass@k — [2504.13837](https://arxiv.org/abs/2504.13837)
- Tiny data suffices to unlock long reasoning (1,000 examples, s1) — [2501.19393](https://arxiv.org/abs/2501.19393)
- More pretraining data can be actively harmful after PTQ — [2411.04330](https://arxiv.org/abs/2411.04330)
- Plasticity collapse to linear-network level, fixed by reinitializing a few units — [2306.13812](https://arxiv.org/abs/2306.13812)
- LoRA tradeoff: learns less, forgets less — [2405.09673](https://arxiv.org/abs/2405.09673); forgetting not avoidable by early stopping — [2401.05605](https://arxiv.org/abs/2401.05605)
- Half the layers removable with little degradation — [2403.17887](https://arxiv.org/abs/2403.17887)
- A Transformer's forward pass can implement gradient descent on in-context regression — [2212.07677](https://arxiv.org/abs/2212.07677); linear attention is a fast-weight programmer — [2102.11174](https://arxiv.org/abs/2102.11174)
- TTT layers keep improving with context length where Mamba does not — [2407.04620](https://arxiv.org/abs/2407.04620)
- Weight-space linearity: averaging fine-tunes improves accuracy with no inference cost — [2203.05482](https://arxiv.org/abs/2203.05482)

### Inferences (candidate open problems)
- Does a verifier-free test-time compute method exist for open-ended tasks? Sampling gains rely on checkable answers (coverage vs selection gap in 2407.21787).
- Can test-time weight updates (TTT, Titans, SEAL) be made stable and safe in deployment given measured forgetting, and do they need the plasticity fixes from Q2?
- Is the quantization ceiling a training-data effect or an architecture effect, and does quantization-aware training move it? (QAT not sourced here.)
- Do merging, LoRA and sparsity compose, and why does weight-space averaging work (shared basin)? Mechanistic account not sourced.
- Unifying theory of fast weights / ICL / TTT / meta-learning: connections cited above are individually verified; a single formal equivalence across all is **[unverified]** and may not hold.

### Gaps
- No verified source for the claim that most 2025 frontier reasoning models rely on any particular test-time method; proprietary.
- Many 2025–2026 items are snippet-verified only; they should be abstract-checked before the report quotes their numbers.
