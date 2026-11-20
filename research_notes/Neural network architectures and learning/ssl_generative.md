# Learning paradigms: self-supervised learning and generative modelling

Verification legend. Every arXiv id below was checked on 2026-11-20 against the arXiv API (id resolves, title and first author match). Claims quoted from an abstract I read are untagged. Claims tagged **[unverified]** come from background knowledge: the id is real, but I did not re-read the text for that specific claim. Links are https://arxiv.org/abs/<id>. Not read in this session: full papers (abstracts only), and no benchmark tables were re-checked, so quantitative SOTA claims beyond the abstract numbers are [unverified].

Plain-language frame (CS analogy): supervised learning = labelled lookup table. SSL = learn a compression/encoding from the data's own structure (views, masks, next token) so a cheap linear "head" can do the task later. Generative modelling = learn the data distribution itself so you can sample from it or use it as a prior.

## Q1. What prevents representation collapse in each SSL family, and what theory exists?

### Takeaway
Each family blocks the trivial "constant output" solution by a different mechanism: contrastive uses negatives (repulsion), non-contrastive self-distillation uses asymmetry (predictor, stop-gradient, EMA teacher, centering/sharpening), covariance methods use explicit variance/decorrelation penalties, masked modelling uses the reconstruction target itself (the target is data, so it cannot be constant), and JEPA relies on asymmetry plus (in LeJEPA) an explicit distribution regulariser. Theory exists but is partial: contrastive and non-contrastive are provably close cousins (spectral-embedding view), the asymmetry-based methods have only linearised dynamics theory, and the first "provable, heuristic-free" JEPA recipe (LeJEPA) appeared only in Nov 2025.

### Cited Findings

**Supervised vs self-supervised (framing)**
- A Cookbook of Self-Supervised Learning is the practical survey of recipes and families (Balestriero et al., 2023) — [2304.12210](https://arxiv.org/abs/2304.12210)
- The Platonic Representation Hypothesis (Huh et al., 2024): representations from different models/modalities are argued to converge toward a shared statistical model of reality (relevant to why SSL features transfer) — [2405.07987](https://arxiv.org/abs/2405.07987)

**Contrastive family (mechanism: pull positives together, push negatives apart)**
- CPC / InfoNCE (2018): predict future latents with a contrastive loss that lower-bounds mutual information — [1807.03748](https://arxiv.org/abs/1807.03748). The InfoNCE form is `L = -E log[ exp(s(x,x+)/tau) / sum_j exp(s(x,x_j)/tau) ]` [unverified: formula from background].
- SimCLR (2020): augmentations + large batch as the source of negatives, nonlinear projection head — [2002.05709](https://arxiv.org/abs/2002.05709)
- MoCo (2019): momentum-encoder queue gives a large, consistent dictionary of negatives without huge batches — [1911.05722](https://arxiv.org/abs/1911.05722)
- SwAV (2020): contrast cluster assignments rather than instances (clustering route, no pairwise negatives) — [2006.09882](https://arxiv.org/abs/2006.09882)
- CLIP (2021): contrastive image-text pairs; the "supervision" is natural-language captions — [2103.00020](https://arxiv.org/abs/2103.00020)
- Theory, alignment/uniformity: the contrastive loss asymptotically optimises closeness of positives and uniformity of the normalised features on the hypersphere; directly optimising those two metrics gives comparable or better downstream accuracy — [2005.10242](https://arxiv.org/abs/2005.10242)
- Theory, spectral: spectral contrastive loss gives provable downstream guarantees under an augmentation-graph assumption (HaoChen et al., 2021) — [2106.04156](https://arxiv.org/abs/2106.04156)
- Contrastive vs mutual information: the MI-maximisation story is incomplete. Tschannen et al. argue the success of InfoNCE-type methods depends strongly on the encoder/critic parametrisation and the (tight or loose) bound, not only on MI itself — [1907.13625](https://arxiv.org/abs/1907.13625) [unverified: summary of their finding from background; abstract not re-read]. Variational MI bounds have a high-variance / low-bias tradeoff, and InfoNCE is capped at log(batch size) — [1905.06922](https://arxiv.org/abs/1905.06922) [unverified: cap statement].
- Dimensional collapse: even contrastive methods can collapse to a low-rank subspace of the embedding space, not just to a constant — [2110.09348](https://arxiv.org/abs/2110.09348) [unverified: from title/background].

**Non-contrastive / self-distillation (no negatives; collapse prevented by asymmetry)**
- BYOL (2020): online network predicts a slow moving-average target network's representation of another view; 74.3% ImageNet linear top-1 with ResNet-50, no negative pairs — [2006.07733](https://arxiv.org/abs/2006.07733)
- SimSiam (2020): Siamese net with predictor and stop-gradient, no negatives, no momentum encoder, no large batches; stop-gradient is the key ingredient — [2011.10566](https://arxiv.org/abs/2011.10566) [unverified: details beyond title]
- Why they do not collapse (theory): Tian et al. study the learning dynamics of BYOL/SimSiam-style setups with a predictor plus stop-gradient and derive DirectPred, which sets the linear predictor from the statistics of its inputs without gradient training; it matches 2-layer nonlinear predictors and beats a linear predictor by 2.5% at 300 epochs on ImageNet — [2102.06810](https://arxiv.org/abs/2102.06810). Their analysis ties non-collapse to predictor alignment with the input correlation matrix and weight decay [unverified: mechanism detail].
- DINO (2021): student matches an EMA teacher's output distribution across views; collapse avoided by centering plus sharpening of the teacher outputs [unverified: mechanism from background]; ViT features show emergent object segmentation — [2104.14294](https://arxiv.org/abs/2104.14294)
- DINOv2 (2023): scaled, curated-data, general-purpose frozen features — [2304.07193](https://arxiv.org/abs/2304.07193). DINOv3 (2025): scales data and model size and adds Gram anchoring to fix degradation of dense feature maps over long training — [2508.10104](https://arxiv.org/abs/2508.10104)

**Covariance / redundancy-reduction (explicit anti-collapse regularisers)**
- Barlow Twins (2021): push the cross-correlation matrix of twin embeddings toward identity (invariance + redundancy reduction) — [2103.03230](https://arxiv.org/abs/2103.03230)
- VICReg (2021): explicit variance (hinge on per-dimension std), invariance, covariance (decorrelation) terms; needs no negatives, no momentum, no stop-gradient — [2105.04906](https://arxiv.org/abs/2105.04906) [unverified: term details]
- Unification: contrastive and covariance-based non-contrastive criteria can be related algebraically and are equivalent under limited assumptions; the low performance of SimCLR is investigated in light of this — [2206.02574](https://arxiv.org/abs/2206.02574). Balestriero & LeCun show contrastive and non-contrastive methods recover global and local spectral embedding methods — [2205.11508](https://arxiv.org/abs/2205.11508).

**Masked modelling (reconstruction target is data, so no constant solution)**
- BERT (2018): masked language modelling, bidirectional pretraining — [1810.04805](https://arxiv.org/abs/1810.04805)
- MAE (2021): mask a high fraction of image patches, asymmetric encoder-decoder, reconstruct pixels — [2111.06377](https://arxiv.org/abs/2111.06377) [unverified: 75% mask ratio]
- MAE vs JEPA tradeoff: JEPA is motivated as prioritising abstract features over unpredictable pixel detail. Littwin et al. analyse deep linear self-distillation (JEPA-style) vs MAE and show an implicit bias toward influential features, which can make JEPA pick up "slow"/low-noise features and potentially miss others — [2407.03475](https://arxiv.org/abs/2407.03475) [unverified: finding summarised from the abstract's first lines only]

**Predictive-architecture (JEPA)**
- I-JEPA (2023): predict representations of masked target blocks from a context block, in latent space, no hand-crafted augmentations — [2301.08243](https://arxiv.org/abs/2301.08243) [unverified: details]
- V-JEPA (2024): feature prediction as the sole objective for video — [2404.08471](https://arxiv.org/abs/2404.08471). V-JEPA 2 (2025): pretrained on over 1 million hours of internet video; 77.3 top-1 on Something-Something v2, 39.7 recall@5 Epic-Kitchens-100 action anticipation; with a little robot-interaction data it supports planning — [2506.09985](https://arxiv.org/abs/2506.09985)
- LeJEPA (Nov 2025): identifies the isotropic Gaussian as the optimal embedding distribution for minimising downstream prediction risk, and enforces it with Sketched Isotropic Gaussian Regularization (SIGReg); one trade-off hyperparameter, linear time/memory, stable across hyperparameters — [2511.08544](https://arxiv.org/abs/2511.08544). This is the cleanest answer to "what prevents collapse in JEPA": an explicit distributional constraint rather than EMA/stop-gradient heuristics.

### Inferences
- Collapse-prevention taxonomy: (a) repulsion (negatives), (b) asymmetry (predictor/stop-grad/EMA/centering), (c) explicit statistics (variance + covariance, SIGReg), (d) data-valued target (MAE/BERT). Trend 2020 to 2025 is from (b) toward (c): replacing heuristics with a stated target distribution.
- The contrastive-vs-non-contrastive divide is smaller than the folklore: two papers (2206.02574, 2205.11508) show algebraic/spectral equivalences, so differences are largely optimisation and normalisation details.
- "Contrastive ~ mutual information" is best treated as a motivating story, not an explanation (1907.13625); alignment+uniformity and spectral-graph views explain more.
- For the repo's own work (a CNN on whitened spectrograms; the N4 masked-spectrogram autoencoder pretraining in CLAUDE.md): MAE-style pretraining is the family with the weakest collapse risk, which fits the reported "SSL helps at low label count" result.

### Gaps
- No theory I could verify that covers EMA+predictor collapse-avoidance beyond linearised/simplified dynamics (2102.06810 is a simplified-setting analysis).
- I did not read BYOL's follow-up on the BatchNorm hypothesis, or SimSiam's/VICReg's full text; details tagged [unverified].
- No verified source for the claim that JEPA "outperforms" reconstruction at scale in general; evidence is benchmark-specific and I did not re-check numbers.

## Q2. How do diffusion, flow matching and consistency models relate mathematically, and what is the state of the art (2025-2026)?

### Takeaway
Diffusion, score matching, and flow matching are the same family seen from different parametrisations: a time-indexed path of distributions from noise to data, learned by regressing a vector field/score/noise/data prediction; the probability-flow ODE is the deterministic flow whose marginals equal the diffusion's. Consistency models and mean-flow methods learn the integral of that ODE (the flow map) to enable one- or few-step sampling. Through 2025 the field moved to flow-matching/rectified-flow transformers (SD3), few-step flow-map models (sCM, MeanFlow), and diffusion for discrete text (MDLM, LLaDA).

### Cited Findings

**Generative family overview (core idea, objective, trade-offs)**
- Autoregressive: exact likelihood by chain rule `p(x)=prod p(x_i|x_<i)`; PixelRNN (2016) [1601.06759](https://arxiv.org/abs/1601.06759); GPT-3 (2020) shows few-shot scaling for text [2005.14165](https://arxiv.org/abs/2005.14165). VAR (next-scale prediction) [2404.02905](https://arxiv.org/abs/2404.02905); MAR drops vector quantisation by using a per-token diffusion loss [2406.11838](https://arxiv.org/abs/2406.11838). Strengths: exact likelihood, scales; weakness: sequential sampling cost, ordering choice.
- VAE (2013): maximise ELBO, `E_q[log p(x|z)] - KL(q(z|x)||p(z))` [1312.6114](https://arxiv.org/abs/1312.6114) [unverified: formula]. VQ-VAE (2017): discrete latent codebook [1711.00937](https://arxiv.org/abs/1711.00937). VQGAN: codebook + adversarial/perceptual loss + transformer prior, "Taming Transformers" [2012.09841](https://arxiv.org/abs/2012.09841). Weakness: blurry samples with plain VAE; codebook collapse [unverified].
- GAN (2014): minimax game between generator and discriminator [1406.2661](https://arxiv.org/abs/1406.2661); WGAN (2017) addresses training instability with a Wasserstein objective [1701.07875](https://arxiv.org/abs/1701.07875); StyleGAN (2018) [1812.04948](https://arxiv.org/abs/1812.04948). Weaknesses: mode collapse, no likelihood, training instability [unverified: general knowledge]. Diffusion "beats GANs" on ImageNet synthesis (2021) [2105.05233](https://arxiv.org/abs/2105.05233).
- Normalizing flows: invertible maps with tractable change-of-variables likelihood. NICE [1410.8516](https://arxiv.org/abs/1410.8516), RealNVP [1605.08803](https://arxiv.org/abs/1605.08803), Glow [1807.03039](https://arxiv.org/abs/1807.03039); continuous-time: Neural ODE [1806.07366](https://arxiv.org/abs/1806.07366), FFJORD [1810.01367](https://arxiv.org/abs/1810.01367). Weakness: architectural constraint from invertibility; sample quality trailed diffusion [unverified]. Note: the BlackHole repo's cited DINGO (Dax et al., 2021) uses NPE with flows for gravitational-wave posteriors — [2106.12594](https://arxiv.org/abs/2106.12594).

**Diffusion and score matching**
- Origin: diffusion probabilistic models from nonequilibrium thermodynamics (2015) — [1503.03585](https://arxiv.org/abs/1503.03585).
- Score-based: NCSN, generate by estimating gradients of the data distribution (annealed Langevin) — [1907.05600](https://arxiv.org/abs/1907.05600).
- DDPM (2020): fixed forward noising, network predicts the noise, simple weighted-ELBO loss — [2006.11239](https://arxiv.org/abs/2006.11239). DDIM: deterministic non-Markovian sampler, same training objective — [2010.02502](https://arxiv.org/abs/2010.02502). Improved DDPM (learned variances, cosine schedule) — [2102.09672](https://arxiv.org/abs/2102.09672).
- Unification via SDEs (Song et al., 2020): forward SDE perturbs data to noise; reverse-time SDE needs the score; a "probability flow ODE" shares the same marginals; DDPM and NCSN are two discretisations (VP and VE SDEs) — [2011.13456](https://arxiv.org/abs/2011.13456) [unverified: VP/VE labelling from background; the paper title and the SDE framing are verified]. Likelihood training for score-based models: [2101.09258](https://arxiv.org/abs/2101.09258). Schrodinger-bridge view: [2106.01357](https://arxiv.org/abs/2106.01357).
- Design-space clarification (EDM): separates preconditioning, noise schedule, sampler choices — [2206.00364](https://arxiv.org/abs/2206.00364).
- Guidance: classifier-free guidance combines conditional and unconditional score estimates at sampling time — [2207.12598](https://arxiv.org/abs/2207.12598) [unverified: mechanism summary].
- Latent diffusion (2021): run diffusion in a pretrained autoencoder's latent space — [2112.10752](https://arxiv.org/abs/2112.10752). DiT: transformer backbone — [2212.09748](https://arxiv.org/abs/2212.09748).
- Tutorials/surveys: Luo, "Understanding Diffusion Models: A Unified Perspective" (ELBO, score, SDE views) — [2208.11970](https://arxiv.org/abs/2208.11970); Yang et al. comprehensive survey — [2209.00796](https://arxiv.org/abs/2209.00796).
- Theory of sampling: polynomial-time convergence guarantees from L2-accurate scores with minimal data assumptions — [2209.11215](https://arxiv.org/abs/2209.11215).

**Flow matching / rectified flow / stochastic interpolants (three near-simultaneous 2022 formulations)**
- Flow Matching (Lipman et al.): regress a vector field defined by conditional probability paths; includes diffusion paths as a special case and also non-diffusion paths such as optimal-transport displacement — [2210.02747](https://arxiv.org/abs/2210.02747) [unverified: the special-case claim as phrased from background; consistent with the Guide below].
- Rectified flow (Liu et al.): learn an ODE along straight-line couplings between noise and data; repeating ("reflow") straightens paths for fewer-step sampling — [2209.03003](https://arxiv.org/abs/2209.03003) [unverified: reflow detail]. InstaFlow turns Stable Diffusion into a one-step model via rectified flow — [2309.06380](https://arxiv.org/abs/2309.06380).
- Stochastic interpolants (Albergo & Vanden-Eijnden): flows built with interpolants between two densities — [2209.15571](https://arxiv.org/abs/2209.15571).
- Unifying exposition: Flow Matching Guide and Code (Lipman et al., 2024) — [2412.06264](https://arxiv.org/abs/2412.06264).
- Production use: Stable Diffusion 3 (MMDiT with rectified flow, Esser et al., 2024) — [2403.03206](https://arxiv.org/abs/2403.03206).
- Mathematical relation (summary, my synthesis from the above): for a Gaussian path `x_t = a_t x_1 + s_t eps`, the learnable quantities (score, noise-prediction eps, data-prediction x, velocity v) are linear reparametrisations of one another, so "diffusion with ODE sampler" and "Gaussian-path flow matching" differ only in loss weighting, schedule, and network parametrisation. [unverified: stated from the Guide/EDM framing, not re-derived here]

**Consistency models and few-step sampling**
- Consistency Models (Song et al., 2023): learn a function mapping any point on a probability-flow-ODE trajectory to its endpoint, via distillation or direct training — [2303.01469](https://arxiv.org/abs/2303.01469) [unverified: mechanism from background; id/title verified].
- Improved consistency training — [2310.14189](https://arxiv.org/abs/2310.14189); Consistency Models Made Easy (ECT) — [2406.14548](https://arxiv.org/abs/2406.14548).
- sCM (Lu & Song, 2024): a simplified theoretical framework that unifies previous parametrisations of diffusion models and CMs, identifies the root causes of continuous-time instability, scales to 1.5B parameters on ImageNet 512x512 — [2410.11081](https://arxiv.org/abs/2410.11081).
- Flow map matching: a mathematical framework for consistency models using stochastic interpolants — [2406.07507](https://arxiv.org/abs/2406.07507). MeanFlow: models the average velocity rather than the instantaneous velocity, with an identity linking the two; FID 3.43 at 1 NFE on ImageNet 256x256 trained from scratch, no distillation or curriculum — [2505.13447](https://arxiv.org/abs/2505.13447).
- Distillation: progressive distillation [2202.00512](https://arxiv.org/abs/2202.00512); guided-model distillation [2210.03142](https://arxiv.org/abs/2210.03142); Adversarial Diffusion Distillation [2311.17042](https://arxiv.org/abs/2311.17042); Distribution Matching Distillation (one-step) [2311.18828](https://arxiv.org/abs/2311.18828) and DMD2 [2405.14867](https://arxiv.org/abs/2405.14867).

**Discrete diffusion for text**
- D3PM: structured diffusion on discrete state spaces — [2107.03006](https://arxiv.org/abs/2107.03006).
- SEDD: score-entropy objective estimating ratios of the data distribution — [2310.16834](https://arxiv.org/abs/2310.16834).
- MDLM: simple masked discrete diffusion with a Rao-Blackwellised objective that is a mixture of classical masked-LM losses; supports semi-autoregressive arbitrary-length generation — [2406.07524](https://arxiv.org/abs/2406.07524).
- LLaDA (2025): 8B masked-diffusion LM trained from scratch, competitive with LLaMA3 8B in in-context learning, instruction following after SFT — [2502.09992](https://arxiv.org/abs/2502.09992).

**2025 state of the art signals (representation- and simplicity-driven)**
- REPA: aligning diffusion transformer hidden states to pretrained SSL features speeds training — [2410.06940](https://arxiv.org/abs/2410.06940) [unverified: summary from title]. RAE: replace the VAE with pretrained representation encoders (DINO, SigLIP, MAE) plus trained decoders, giving semantically rich latent spaces — [2510.11690](https://arxiv.org/abs/2510.11690). This is a direct bridge between Q1 (SSL features) and Q2 (generation).
- Noise conditioning may not be necessary: most denoising generative models degrade gracefully without it, some improve; a noise-unconditional model reaches FID 2.23 on CIFAR-10 — [2502.13129](https://arxiv.org/abs/2502.13129).
- "Back to Basics" (JiT): argues networks should predict clean data rather than noise or velocity, by the manifold assumption; large-patch pixel Transformers with no tokenizer, pretraining, or extra loss — [2511.13720](https://arxiv.org/abs/2511.13720).

### Inferences
- Mathematical hierarchy: (stochastic-interpolant/flow-matching paths) contains (Gaussian-path diffusion) as a case; consistency/MeanFlow/flow-map methods are amortised integrators of the same ODE. The SOTA question is therefore mostly about parametrisation, weighting, and distillation, not a different model class.
- A sign of convergence: sCM's abstract says it "unifies previous parameterizations of diffusion models and CMs" and flow-map matching frames CMs via stochastic interpolants — the two communities' notations are merging.
- 2025 trend: few-step from-scratch models (MeanFlow at FID 3.43 at 1 NFE) are closing the gap with multi-step ones without distillation; discrete diffusion LMs are now viable at 8B scale but I found no verified head-to-head against frontier AR models.
- Trade-off summary: AR/flow = exact likelihood; VAE = ELBO bound + fast sampling + blur; GAN = sharp + fast, no likelihood + mode collapse; diffusion/flow-matching = best sample quality and stable training, costly sampling unless distilled.

### Gaps
- I did not verify any 2026 papers; "state of the art 2025-2026" is covered only to Nov 2025 via arXiv ids that resolved. Anything claiming 2026 SOTA is absent here.
- I did not verify FID numbers beyond those quoted in abstracts, nor compare against closed-source systems.
- A single canonical reference for "diffusion = flow matching with Gaussian paths" beyond the Guide (2412.06264) and EDM I did not read in full; the equivalence summary is my synthesis.
- Not covered: Tweedie-formula derivation sources, and DDPM-to-score equivalence original proofs (Vincent denoising score matching 2011 has no arXiv id I checked).

## Q3. Where do generative models serve as priors or inference engines (inverse problems), and what are the reliability concerns?

### Takeaway
Diffusion models are widely used as learned priors for posterior sampling in inverse problems (medical imaging, astronomical imaging), but all practical algorithms are approximations, and there is a proven computational-hardness result for exact posterior sampling. Reliability concerns documented in the literature: hallucination via mode interpolation, memorisation of training data, and recursive-training collapse.

### Cited Findings
- Medical imaging: unconditional score-based models used as priors for conditional sampling at inference (CT, MRI), no retraining per measurement operator — [2111.08005](https://arxiv.org/abs/2111.08005).
- Diffusion Posterior Sampling (DPS) for general noisy inverse problems, including nonlinear ones — [2209.14687](https://arxiv.org/abs/2209.14687). DDRM for linear restoration — [2201.11793](https://arxiv.org/abs/2201.11793). Latent-diffusion priors with a provable guarantee for linear inverse problems (PSLD-related) — [2307.00619](https://arxiv.org/abs/2307.00619). Flow-matching analogue: PnP-Flow — [2410.02423](https://arxiv.org/abs/2410.02423). Provable probabilistic imaging with score priors — [2310.10835](https://arxiv.org/abs/2310.10835). Astronomical/black-hole-style imaging with score-based priors — [2304.11751](https://arxiv.org/abs/2304.11751) [unverified: application detail; title verified].
- Hardness: under the assumption that one-way functions exist, there are instances where every posterior-sampling algorithm takes superpolynomial time even though unconditional sampling is provably fast; exponential-time rejection sampling is essentially optimal under a stronger assumption — [2402.12727](https://arxiv.org/abs/2402.12727). This means DPS-style methods cannot be guaranteed correct in general; they are heuristic approximations.
- Hallucination mechanism: diffusion models smoothly interpolate between nearby data modes, producing samples outside the training support ("mode interpolation"), linked to a discontinuous loss landscape in the decoder — [2406.09358](https://arxiv.org/abs/2406.09358).
- Memorisation: diffusion models can regurgitate training images — [2301.13188](https://arxiv.org/abs/2301.13188) [unverified: finding from title/background]. Counterpoint: two networks trained on disjoint subsets learn nearly the same score when the training set is large enough, i.e. strong generalisation with inductive biases aligned to the data density (geometry-adaptive harmonic representations) — [2310.02557](https://arxiv.org/abs/2310.02557). Statistical-physics view: a "speciation" transition then a "collapse" (memorisation) transition during reverse diffusion, with timing computable from the data's correlation matrix — [2402.18491](https://arxiv.org/abs/2402.18491).
- Training on generated data: model collapse, where tails of the original distribution disappear; shown for VAEs, GMMs and LLMs — [2305.17493](https://arxiv.org/abs/2305.17493).
- Neural posterior estimation with flows is already used for GW inference (DINGO) — [2106.12594](https://arxiv.org/abs/2106.12594).

### Inferences
- Reliability checklist for using a generative prior in inverse problems: (1) check posterior calibration with simulation-based coverage (as the repo's ringdown NPE work does), not just visual quality; (2) test prior support (hallucination risk, 2406.09358); (3) test memorisation vs generalisation in the data regime (2310.02557, 2402.18491); (4) treat approximate posterior samplers as heuristics given the hardness result (2402.12727).
- For physics/scientific use, a prior that "interpolates modes" can fabricate plausible but non-physical structure; this is a concrete reason to prefer amortised NPE with coverage checks over image-quality benchmarks.

### Gaps
- I did not find or verify a systematic benchmark of hallucination rates of DPS-type solvers on scientific data.
- Applications to gravitational-wave data with diffusion priors specifically: not searched; only the flow-based DINGO is cited. Gap.
- Not verified whether flow-matching priors share the same hardness/hallucination profile (expected by the unification, but untested here).

## Q4. Open problems and surprising results

### Takeaway
The most surprising 2024-2025 results are that many "essential" components are optional (noise conditioning, a VAE, SSL heuristics) and that theory for SSL collapse and diffusion generalisation is still partial. Core open problems: provable SSL objectives at scale, why diffusion generalises rather than memorises, reliable posterior sampling, and one-step quality parity.

### Cited Findings
- Noise conditioning is not indispensable for denoising generative models — [2502.13129](https://arxiv.org/abs/2502.13129).
- Predicting clean data beats predicting noise/velocity in high dimensions under the manifold assumption; tokenizer-free pixel Transformers work — [2511.13720](https://arxiv.org/abs/2511.13720).
- The VAE in latent diffusion can be replaced by pretrained SSL encoders (RAE) — [2510.11690](https://arxiv.org/abs/2510.11690).
- JEPA now has a "provable and scalable" objective without heuristics — [2511.08544](https://arxiv.org/abs/2511.08544).
- Diffusion LMs competitive with an 8B AR baseline — [2502.09992](https://arxiv.org/abs/2502.09992).
- One-step generation from scratch at FID 3.43 (ImageNet 256) — [2505.13447](https://arxiv.org/abs/2505.13447).
- Generalisation vs memorisation of diffusion models is characterised by regime (data amount, dimension) — [2310.02557](https://arxiv.org/abs/2310.02557), [2402.18491](https://arxiv.org/abs/2402.18491).
- Posterior sampling hardness — [2402.12727](https://arxiv.org/abs/2402.12727); mode-interpolation hallucination — [2406.09358](https://arxiv.org/abs/2406.09358); recursive-training collapse — [2305.17493](https://arxiv.org/abs/2305.17493).

### Inferences
- Open problems I would list for a report: (1) a theory of non-collapse for asymmetry-based SSL beyond linear dynamics; (2) whether SIGReg-style Gaussian constraints scale and match DINOv3-class results; (3) why trained diffusion models generalise (inductive bias vs data scaling); (4) calibrated, provably-correct posterior sampling with generative priors (hardness suggests only restricted classes); (5) discrete diffusion vs autoregression at frontier scale; (6) unifying generative training and representation learning (REPA/RAE point this way); (7) data contamination by generated content (model collapse).
- Surprising: simplification trend (JiT, noise-unconditional, MeanFlow) suggests much of the published complexity was historical convention.

### Gaps
- No verified evidence on how well LeJEPA scales against DINOv3/V-JEPA 2 at the largest sizes; it is a single Nov 2025 paper, not yet independently replicated as far as I could find. [unverified] whether it has been replicated.
- Any 2026 results are not covered.
- I did not search for contradicting views on the Platonic Representation Hypothesis or critiques of SSL benchmark practice.
