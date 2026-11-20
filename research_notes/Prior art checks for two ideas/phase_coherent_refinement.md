# Prior-art check: "cheap scan, then phase-coherent refinement" for long faint chirps

*Checked 2026-11-20. Every arXiv id below was resolved against its arxiv.org/abs page (title, first author, date
read from the page metadata); the two non-arXiv items were resolved via Crossref. The arXiv API itself rate-limited
("Rate exceeded"), so the abs pages were used instead. **[A]** = claim read in the abstract (or in the paper's HTML,
where stated). **[inf]** = my inference, not something the paper says. Nothing here was read past the abstract unless
marked.*

---

## 1. Verdict (plain language)

**The architecture is not new; one narrow comparison inside it is.** "Scan cheaply with a semi-coherent or coarse
statistic, flag candidates, then go fully coherent only on those" is the standard design of continuous-wave
searches. It has a 25-year history: Brady & Creighton 1998, Einstein@Home's multi-stage follow-ups, Shaltev & Prix's
coherent follow-up, and Ashton & Prix's MCMC "coherence ladder". It is also already used for subsolar-mass inspirals,
the very signal class we care about: Soni & Nitz 2024 run a two-stage coarse→fine matched-filter search for subsolar
binaries. The same "widen the peak semi-coherently, then raise coherence step by step" idea is the published plan for
LISA stellar-mass inspirals, which are the closest physical analogue of a minutes-to-years chirp.

The machine-learning versions of each piece also exist:
- a CNN as the cheap first stage that hands a narrowed region to matched filtering (Verma et al. 2026; Modafferi
  et al. 2023);
- a network that reaches coherent sensitivity once the parameter region is narrow (Joshi & Prix 2023, for continuous
  waves);
- a neural follow-up stage on real O3 continuous-wave candidates (Di Cesare et al. 2026);
- learned iterative refinement around a trigger for long inspirals. Dingo-BNS (2024) heterodynes with a trigger
  chirp mass and conditions a network on a ±0.005 M☉ prior. Note that it is parameter estimation, not detection.

**What I could not find:** a head-to-head of a *learned iterative refiner* (weight-tied, deep-equilibrium-style or
recurrent) against the *classical coherence-ladder follow-up*, at matched compute and matched false-alarm rate, for
minutes-long inspirals. That is the open piece.

Two caveats shrink it before any work starts:
1. **The sensitivity ceiling is fixed by theory [inf].** In Gaussian noise, no learned stage-2 can beat a properly
   marginalised coherent likelihood ratio on sensitivity. The only real room is *compute to converge* (fewer
   likelihood evaluations on a very spiky likelihood surface) and *robustness to non-Gaussian noise*.
2. **"Coherent at semi-coherent cost" now has a classical competitor that did not exist two years ago:** Kacanja et
   al. 2026 ran a fully coherent 25-million-template subsolar search on O4a using ratio-filter dechirping.

## 2. Closest prior work

### 2a. Hierarchical semi-coherent → coherent (classical)

| Work | What it does [A] | How it differs from the idea |
|---|---|---|
| **Brady & Creighton 1998**, *Searching for periodic sources with LIGO. II: Hierarchical searches*, [gr-qc/9812014](https://arxiv.org/abs/gr-qc/9812014) | Stacked power spectra (incoherent) are 2–4× more sensitive than coherent at fixed cost for all-sky continuous-wave searches. A two-stage hierarchy adds another 20–60%. | Founding statement of the trade-off. Classical; not chirps. |
| **Prix & Shaltev 2012**, *Optimal StackSlide method at fixed computing cost*, [1201.4321](https://arxiv.org/abs/1201.4321) | Analytic optimum of semi-coherent parameters at fixed cost. Coarse- and fine-grid mismatch contribute equally. | The cost model our stage-1/stage-2 split would need. Classical. |
| **Shaltev & Prix 2013**, *Fully coherent follow-up of continuous gravitational-wave candidates*, [1303.2471](https://arxiv.org/abs/1303.2471) | Direct two-stage transition from semi-coherent candidates to a fully coherent F-statistic, using a grid-less MADS optimiser. Monte-Carlo detection power and cost. | **Exactly the "scan then coherent refinement" design, classical.** Continuous waves, not inspirals. |
| **Ashton & Prix 2018**, *Hierarchical multi-stage MCMC follow-up of CW candidates*, [1802.05450](https://arxiv.org/abs/1802.05450) | MCMC on the F-statistic with coherence time raised stage by stage until fully coherent. Reported "close to the theoretical optimal performance". | **The coherence ladder: the classical baseline a learned refiner must beat.** |
| **Papa et al. 2016**, Einstein@Home S6 hierarchical follow-up, [1608.08928](https://arxiv.org/abs/1608.08928) | 16 million sub-threshold candidates followed up in three stages on real data. | Production scale. Shows the dominant cost is candidate *count*. |
| **Dergachev 2010**, *loosely coherent* searches, [1003.2178](https://arxiv.org/abs/1003.2178) | A statistic that "bridges the gap between semi-coherent and purely coherent". Partial phase coherence is controlled by the accepted signal family. | **The classical version of "partially recover cross-chunk phase".** A knob between our n=8 and n=1. |
| **Covas et al. 2024**, [2404.18608](https://arxiv.org/abs/2404.18608); **Martins et al. 2025**, [2508.18204](https://arxiv.org/abs/2508.18204) | MCMC or nested-sampling follow-up of CW candidates through bilby. Recovers the maximum-posterior point over much larger regions than thought. The Bayesian framework handles candidates from deep surveys. | The current classical follow-up. Marginalised (Bayesian) rather than maximised, which matters for the Neyman–Pearson ceiling (§4). |
| **Gadre, Mitra & Dhurandhar 2018**, hierarchical aligned-spin CBC search in PyCBC, [1807.06803](https://arxiv.org/abs/1807.06803) | Coarse bank (stage 1) then fine bank around triggers (stage 2). About 20× cheaper in Gaussian noise; "a factor of few" expected in real data. | Compact binaries, but **both stages are fully coherent per template**: coarse→fine *in parameters*, not semi→fully coherent *in phase*. |
| **Soni, Dhurandhar & Mitra 2023**, *Statistical significance in hierarchical search*, [2309.00019](https://arxiv.org/abs/2309.00019) | Background estimation for two-stage searches, using time-shifted triggers and extrapolated tails. Sensitive volume×time matches PyCBC at inverse false-alarm rate 10 yr, at ~13× speed-up. | **The false-alarm pitfall any cascade inherits**: running stage 2 only on zero-lag triggers biases the background. |
| **Soni & Nitz 2024**, *Hierarchical searches for subsolar-mass binaries and the 3G era*, [2409.11317](https://arxiv.org/abs/2409.11317) | Two-stage hierarchical search **for subsolar-mass binaries**. Filtering from 35 Hz gives +6% SNR and +10–20% detection volume vs two-detector PyCBC, at 2.5× lower cost. | **Closest compact-binary prior art.** Again coarse→fine bank, coherent in both stages, classical. |
| **Dhurkunde et al. 2021**, reduced-basis hierarchical matched filtering, [2110.13115](https://arxiv.org/abs/2110.13115) | Hierarchy in a reduced basis. ~10× speed-up at SNR ≥ 6, no loss of sensitivity; names subsolar as a target. | Classical hierarchy in a different basis. |
| **Kacanja, Soni, Akyüz & Nitz 2026**, O4a subsolar search, [2602.12115](https://arxiv.org/abs/2602.12115); **Nitz et al. 2026**, *Beyond FINDCHIRP* (ratio-filter dechirping), [2601.18835](https://arxiv.org/abs/2601.18835) | Fully coherent 25-million-template subsolar search, made "computationally feasible" by ratio-filter dechirping. Null result. | **Competing route**: make the coherent bank cheap rather than avoid it. A learned cascade now has to beat this on cost. |
| **Hanna et al. 2024**, LVK O4 subsolar template bank, [2412.10951](https://arxiv.org/abs/2412.10951) | Archive bank of 3,452,006 templates; f_low = 45 Hz; maximum duration 128 s. | Shows the scale of the bank a cascade would replace. |
| **Bandopadhyay & Moore 2023**, [2305.18048](https://arxiv.org/abs/2305.18048); **Bandopadhyay et al. 2024**, [2408.13170](https://arxiv.org/abs/2408.13170); **Fu et al. 2024**, [2407.10797](https://arxiv.org/abs/2407.10797) | LISA stellar-mass BBH (years-long chirps). The semi-coherent likelihood "widens the peak, congealing secondary peaks". Coherence is then raised progressively (particle swarm, GPU). Fu et al.: hierarchical semi-coherent F-statistic → **fully coherent final stage**, with chirp-mass error ≲0.01% at SNR 11–14. | **Closest physics analogue for chirps.** The scan→coherence-ladder cascade for long inspirals is published. Classical only. |
| **Miller et al. 2024**, [2402.19468](https://arxiv.org/abs/2402.19468) & [2407.17052](https://arxiv.org/abs/2407.17052); **LVK 2025** O4a, [2511.19911](https://arxiv.org/abs/2511.19911) | Planetary-mass PBH inspirals searched with the generalised frequency-Hough (semi-coherent track finder), followed by a candidate follow-up procedure. | Semi-coherent scan + follow-up for even longer, lighter chirps. |
| **Alestas et al. 2024**, Viterbi for planetary-mass BHs, [2401.02314](https://arxiv.org/abs/2401.02314) | Hidden-Markov/Viterbi tracking for the subsolar-to-planetary range. "Agnostic and computationally efficient." | A cheap stage-1 alternative to a CNN. |

### 2b. ML versions

| Work | What it does [A] | How it differs |
|---|---|---|
| **Verma et al. 2026**, *Neural Network Guided Parameter Space Constraints for GW Searches from BBHs*, [2609.28031](https://arxiv.org/abs/2609.28031) | A CNN detects, then picks which bank patch the signal is in (93.2% accuracy in the chirp-mass–duration representation), so matched filtering runs only there. | **ML stage-1 → matched-filter stage-2 for compact binaries.** The refinement is classical and not iterative; BBH, short signals; classification accuracy only, no matched-false-alarm sensitivity. |
| **Modafferi et al. 2023**, CNN search for long-duration transients from glitching pulsars, [2303.16720](https://arxiv.org/abs/2303.16720) | Combines matched filtering with a CNN. Reaches similar sensitivity to the standard method at practical false-alarm probabilities, much faster (search snippet: CNN pre-screens ~1.15×10⁷ templates; 10⁻³ go to the expensive stage) [snippet]. | ML scan + expensive follow-up on a long-duration signal; not an inspiral, not iterative. |
| **Di Cesare et al. 2026**, *Alternative neural-network follow-up for all-sky FrequencyHough*, [2609.06211](https://arxiv.org/abs/2609.06211) | NN classifier inserted after FrequencyHough candidates. Improves sensitivity "without increasing computational cost"; tested on real O3 data. | **A learned follow-up stage on real candidates.** A one-shot classifier, not an iterative coherent refiner; CW, not chirps. |
| **Joshi & Prix 2023**, *Novel neural-network architecture for CWs*, [2305.01057](https://arxiv.org/abs/2305.01057) | Earlier CNNs failed beyond ~1 day. Their design attains **matched-filtering sensitivity for a targeted search** over 10 days, two detectors. Output maps monotonically onto the F-statistic. | **Key evidence for the premise [inf]**: once the region is narrow, a network *can* reach coherent sensitivity. But it only ties the coherent statistic; it does not beat it. |
| **Dreissigacker et al. 2019**, [1904.13291](https://arxiv.org/abs/1904.13291); **Dreissigacker & Prix 2020**, [2005.04140](https://arxiv.org/abs/2005.04140) | Against a fully coherent WEAVE search: −5% depth at T=10⁵ s / 20 Hz, but **−66% at T=10⁶ s / 1000 Hz**. Two-detector all-sky: −7% to −51%. | Learned detectors lose coherence as duration × parameter-space volume grows: the same wall we hit. |
| **Joshi & Prix 2024**, [2408.07070](https://arxiv.org/abs/2408.07070); **Joshi & Prix 2025**, ViT, [2509.10912](https://arxiv.org/abs/2509.10912); **Cheung 2025**, [2509.19838](https://arxiv.org/abs/2509.19838) | Wide-parameter DNNs improve on the earlier ones. ViT "essentially matched-filter sensitivity" on targeted searches, 78–88% vs 90% all-sky. Attention U-Net sensitivity ∝ T^0.28, "similar to semi-coherent". | **Wide-area learned detectors scale like semi-coherent; narrow-area ones reach coherent [inf].** That is the empirical case for a cascade, and it is already visible in CW work. |
| **Beheshtipour & Papa 2020**, [2001.03116](https://arxiv.org/abs/2001.03116), [2012.04381](https://arxiv.org/abs/2012.04381) | Deep-learning clustering of CW candidates to cut follow-up cost. | ML between the stages, not as the refiner. |
| **Yan et al. 2022**, *Hierarchical Detection Network*, [2207.11583](https://arxiv.org/abs/2207.11583); **Yan et al. 2021**, *Generalized approach to matched filtering using NNs*, [2104.03961](https://arxiv.org/abs/2104.03961) | Matched filtering is exactly a neural network. Trainable hierarchical matched filtering gives a 79% efficiency gain at equal error rate. With a prior, the network "can approach the statistically optimal performance" and beats max-over-templates matched filtering above a 5×10⁻³% false-positive rate. | **Learned hierarchical matched filtering exists.** Feed-forward, BBH-scale. Their outperformance comes from *marginalising over a prior*, which a classical Bayesian follow-up also does. |
| **Wang et al. 2020** (MFCNN), [1909.13442](https://arxiv.org/abs/1909.13442); **Ma et al. 2023**, template-prediction model bank, [2309.02030](https://arxiv.org/abs/2309.02030) | Matched-filter-layer CNN on real O1 data; deep-learning "model bank" predicts the template, then matched filtering. | Learned template estimation feeding matched filtering. Not iterative, not long-duration. (Our rung-3 V2 was MFCNN-style and capped at AUC ~0.69.) |
| **Beveridge et al. 2023**, [2308.08429](https://arxiv.org/abs/2308.08429), and **2025**, [2512.04516](https://arxiv.org/abs/2512.04516); **McLeod et al. 2024** (BNS), [2409.06266](https://arxiv.org/abs/2409.06266); **Yamamoto et al. 2025**, [2512.12399](https://arxiv.org/abs/2512.12399) | Deep learning on matched-filter SNR time series or SNR maps (BBH, BNS); comparable to standard pipelines. | The reverse order (matched filter first, ML second). |
| **Andrés-Carcasona et al. 2023**, *Searches for mass-asymmetric CBC events using neural networks in O3*, PRD 107, 082003, [doi:10.1103/PhysRevD.107.082003](https://doi.org/10.1103/PhysRevD.107.082003) (Crossref-verified; arXiv id not checked) | CNN search of O3 for asymmetric-mass binaries, including subsolar secondaries [snippet]. | The only ML search touching the subsolar regime I found. No coherent follow-up stage seen in snippets. |
| **Dax et al. 2024**, *Dingo-BNS*, [2407.09602](https://arxiv.org/abs/2407.09602) | Full BNS inference in ~1 s. From the HTML: **heterodyning with an approximate chirp mass**, a network **prior-conditioned on narrow (±0.005 M☉) chirp-mass windows**, multibanding (60–650× compression), signals up to 4096 s. The authors *chose not to* use iterative GNPE ("increases inference time"). | **Closest to "learned stage-2 refinement around a trigger for long inspirals".** But it is parameter estimation, not detection at matched false-alarm rate; BNS masses, not subsolar; non-iterative by choice. |
| **Dax et al. 2021**, GNPE, [2111.13139](https://arxiv.org/abs/2111.13139); DINGO, [2106.12594](https://arxiv.org/abs/2106.12594); neural importance sampling, [2210.05686](https://arxiv.org/abs/2210.05686) | GNPE is an **iterative, Gibbs-style learned fixed-point refinement** of a pose parameter (time shift). Importance sampling corrects the network against the exact likelihood (~10% efficiency). | The iterative-learned-refinement mechanism already exists in GW, for parameter estimation. Importance sampling is the "trust but verify" pattern a learned stage-2 should copy. |

### 2c. Outside GW

| Work | What it does [A] | Relevance |
|---|---|---|
| **Ransom 2026**, *A Coherent Harmonic Summing Pulsar Search Code*, [2610.11749](https://arxiv.org/abs/2610.11749) | Restores the harmonic **phases** that standard incoherent summing discards. At matched false-alarm rate: 76% recovered vs 64% (riptide FFA) vs 42% (PRESTO accelsearch), and faster than FFA. | **Clean classical demonstration that recovering dropped phase pays at matched false-alarm rate**, at near-incoherent cost. The pulsar analogue of the idea, and it is classical. |
| **Morello et al. 2020**, riptide FFA, [2004.03701](https://arxiv.org/abs/2004.03701) | Folding approaches optimal periodic sensitivity, beating FFT + incoherent harmonic summing. | Same lesson. |
| **Borhani-Darian et al. 2023**, *Deep Learning of GNSS Acquisition*, Sensors 23, 1566, [doi:10.3390/s23031566](https://doi.org/10.3390/s23031566) (Crossref); **Gómez-Casco et al. 2018**, [1810.04085](https://arxiv.org/abs/1810.04085) | CNN on the correlation function, with non-coherent integration fused across steps [snippet]. Optimal (GLRT/Bayesian) post-detection integration for weak GNSS signals under bit/phase uncertainty. | GNSS fixes coherent length and integrates non-coherently. Learned versions classify; they do not restore coherence. |
| **Fishel et al. 2024**, *Learn to Track-Before-Detect via Neural Dynamic Programming*, ICASSP, [doi:10.1109/ICASSP48485.2024.10448128](https://doi.org/10.1109/ICASSP48485.2024.10448128) (Crossref) | Viterbi augmented with a DNN for weak targets [snippet]. | Learned versions of the Viterbi track-linking stage exist in radar. |
| **Bai et al. 2019**, DEQ, [1909.01377](https://arxiv.org/abs/1909.01377); **Geng et al. 2021**, phantom gradients, [2111.05177](https://arxiv.org/abs/2111.05177); **Geng et al. 2023**, TorchDEQ, [2310.18605](https://arxiv.org/abs/2310.18605) | Implicit fixed-point networks and cheap training gradients. | The proposed stage-2 machinery; mature and off the shelf. |
| **Gilton et al. 2021**, DEQ for inverse problems, [2102.07944](https://arxiv.org/abs/2102.07944); **Zhang et al. 2025**, *DEQReg*, [2507.00582](https://arxiv.org/abs/2507.00582); **Samuel et al. 2017**, *Deep MIMO Detection*, [1706.01151](https://arxiv.org/abs/1706.01151) | DEQs and unrolled networks replacing classical iterative solvers (imaging, registration, MIMO detection). DEQReg is framed explicitly as "bridging classical and learning-based iterative registration". | Learned-vs-classical iterative refinement is an established comparison genre. Registration (iterative alignment) is the nearest analogue of iterative template-parameter refinement. |

## 3. Partly-done pieces

1. **Cascade architecture**: done, classically, for CW (since 1998), subsolar CBC (coarse→fine bank) and LISA
   chirps (semi→fully coherent ladder).
2. **ML stage-1 with matched-filter stage-2**: done for BBH (Verma 2026), long transients (Modafferi 2023) and as a
   CW follow-up (Di Cesare 2026). Not done for subsolar chirps that I found.
3. **Learned network reaching coherent sensitivity on a narrow region**: done for CW (Joshi & Prix 2023, ViT 2025).
   Not for chirps.
4. **Learned iterative refinement around a trigger for long inspirals**: done as parameter estimation
   (Dingo-BNS: trigger chirp mass → heterodyne → prior-conditioned network; GNPE as the iterative form). Not as a
   detection/ranking stage at matched false-alarm rate, and not for subsolar.
5. **Phase-restoring follow-up beats incoherent at matched false-alarm rate**: shown classically in pulsars
   (Ransom 2026) and quantified for CW (Brady–Creighton, Prix–Shaltev).
6. **Background estimation through a cascade**: solved for classical two-stage CBC (Soni, Dhurandhar, Mitra 2023).
   It needs redoing for any learned stage-2.
7. **Our own repo has already measured part of the decomposition** (CLAUDE.md status blocks, not re-verified here):
   - Semi-coherent n=8 with the *true* template ≈ 0.66–0.76 of ideal; with an adequate *bank* ≈ 0.49 ⇒ "bank
     mismatch is the dominant loss".
   - A learnable matched-filter front end (rung 3 V2) capped at AUC ~0.69.
   - LongChirp Y1/Y0 ≈ 57.5/155 Mpc.

   [inf] These ratios use different normalisations: matched-false-alarm distance fraction vs fixed-SNR oracle. They
   must be put on one footing before the "1/3" is split between "lost phase" and "lost parameters".

## 4. The narrowest open version, the baseline, and the cheapest decisive test

**Ceiling first [inf].** In stationary Gaussian noise with a known waveform family, the optimal stage-2 statistic is
the coherent likelihood ratio marginalised over the candidate's local prior (Neyman–Pearson). A classical Bayesian
coherent follow-up approximates it (Ashton & Prix: "close to the theoretical optimal"; Covas/Martins). So a learned
refiner **cannot beat classical follow-up on detection power in Gaussian noise**. It can only:
- **(a)** reach the same coherent peak in fewer likelihood evaluations (wall-clock) on a very multimodal surface; or
- **(b)** hold its false-alarm rate better in real non-Gaussian noise.

Pitching it as "learned beats classical at matched compute" is only meaningful as (a) or (b), and at a *fixed
compute budget*. Under a fixed budget, classical follow-up may fail to converge and lose candidates. That is where a
learned refiner could genuinely win.

**Narrowest open version.** For subsolar-mass inspirals in real O3a H1×L1 noise (minutes-long, fully coherent needs
≲0.01–0.1% chirp-mass accuracy), compare, **at matched compute per candidate and matched false-alarm rate (1/day,
1/week on LongChirp)**:
- (i) the classical coherence-ladder follow-up: n = 8 → 4 → 2 → 1 chunks, local optimiser (Nelder–Mead/MADS or
  particle swarm) per rung, as in Ashton & Prix / Bandopadhyay & Moore / Fu et al.;
- (ii) a weight-tied learned refiner that maps (whitened data, current template) → parameter update, iterated to a
  fixed point, with the final statistic **always the exact coherent matched-filter SNR / newSNR at the refined
  parameters** (learned proposal, exact scoring, as in Dingo's importance sampling — so false-alarm control never
  rests on a network output);
- both seeded by the same stage-1 top-K, with background run **through the whole cascade on time-slides**
  (Soni et al. 2023).

The claim to test is "same recovered coherent SNR in N× fewer likelihood evaluations" (or "more candidates
converged at a fixed budget"), not "more sensitive than coherent".

**The baselines it must beat:**
1. **Within the cascade**: the classical coherence ladder (above) at the same evaluation budget.
2. **Against the whole pipeline**: a fully coherent dense bank run cheaply via ratio-filter dechirping (the
   Kacanja/Nitz 2026 route; our corrected 2.46–3.45× local speed-up). If that is affordable, a cascade only wins on
   cost, so the comparison must be in core-hours at equal sensitive distance.
3. **Against the existing classical hierarchy**: Soni & Nitz's coarse→fine two-stage search (2.5× cheaper,
   +10–20% volume).

**Cheapest decisive test — no training, run before anything is built.** An *oracle cascade* on LongChirp's existing
1,344 injections and slide background:
1. Stage-1 recall: the fraction of injections whose true window is in the CNN's / semi-coherent bank's top-K, as a
   function of K (equivalently, of stage-1 false-alarm rate, say 10–1000 per day).
2. Stage-2 upper bound: score the top-K with the **true-template fully coherent SNR** for injections and with the
   **best-of-local-fine-grid coherent SNR** for background slides. Using the true template for background would be
   wrong.
3. Read off sensitive distance at 1/day and 1/week, and compare with Y1 (57.5/51.7 Mpc).

**Decision rule [inf]:**
- If even this oracle cascade gains < ~10% over Y1, the idea is dead for both learned and classical refiners,
  because the loss sits in stage-1 recall, not in stage 2.
- If it gains a lot, run the classical ladder (step (i)) under a fixed evaluation budget. **Only the gap between the
  classical ladder and the oracle cascade is available for a learned refiner to win**, and its size decides whether
  building a deep-equilibrium refiner is worth it.

This also splits the "1/3 of ideal" into phase loss vs parameter loss, which no paper does for subsolar.

## 5. Gaps in this search

- **Abstracts only.** Most papers read at abstract level. Not read in full: Soni & Nitz 2024 (stage-2 details,
  whether a semi-coherent stage was considered), Shaltev & Prix 2013 and Ashton & Prix 2018 (cost/sensitivity
  numbers), Verma et al. 2026 (no false-alarm-matched sensitivity seen in the abstract), Di Cesare et al. 2026
  (abstract truncated before results).
- **Search coverage.** No ADS full-text or IEEE Xplore search. The radar/sonar learned coherent-integration
  literature and learned GNSS high-sensitivity acquisition are under-covered; results there were snippet-level. The
  Fishel 2024 and Borhani-Darian 2023 contents are snippet-level (only their metadata was Crossref-checked).
- **arXiv id not confirmed** for the Andrés-Carcasona PRD 107, 082003 paper. My guess, 2302.00666, resolved to a
  different paper (Ravichandran et al.), so I cite the DOI only.
- **Unverified hit, not cited above.** An ET conference abstract (indico.dfa.unipd.it, Aug–Sep 2026) describing a
  CNN vs SFT-based matched filter at 25 min / 2 h / 6 h BNS durations. It is the closest "ML for hours-long
  inspirals vs semi-coherent MF" item, but only a talk abstract with no materials.
- **Possible misses.** Thesis/LVK-internal work on ML-first-stage subsolar searches, Chinese-group ML work on
  long-duration 3G or space-detector chirps (e.g. TianQin/Taiji learned semi-coherent pipelines), and any 2026
  learned-optimiser follow-up for CW (the Einstein@Home groups are the likeliest authors).
- **arXiv API rate limit.** It rate-limited once; the abs-page fallback succeeded for all ids. No id is tagged
  [unverified] because every cited id resolved.
