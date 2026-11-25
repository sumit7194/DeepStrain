# Step T: independent sealed prediction (Fable)

*[model: Fable 5.2] · SEALED · written 2026-11-26 on branch `fable/2026-11-26-stepT-prediction` (from main at
d01cb8d), in answer to `reports/stepT_blind_prediction_brief.md`. Nothing in this file changes after its commit.*

## 0. Blinding statement

- **Blind, as far as I know.** I did not open §7 of `reports/stepT_registration.md` or of the DRAFT, the 2026-11-25 →
  11-26 `JOURNAL.md` entry, the commit messages of `8ce04cf` / `d4c3ca4`, or any Bridge ↔ DeepStrain message about
  step T. I ran no `git log` on main; the only commit subject I saw there is the brief's own (d01cb8d).
- **What I did read that contains an older forecast:** the frozen `PREREG_coherent2.md` §5 ("at one per minute:
  SURVIVES 30%") and §9 ("I expect G0 clause 1 to fail", at one per hour). Both are in a file the brief allows, and
  both predate the redesign.
- **No step-T output exists** in anything I opened.
- **One listed input was missing:** `results/coh2t/S0_H1_1249001472.json` is not on main at d01cb8d. I take S0 = PASS
  from the brief.
- I wrote the redesign (c) round myself, so I am not independent of the engine. I am independent of the statistic,
  the gates and the driver, none of which I built.

## 1. The predictions (these are what is graded)

**1. R_tune** (chosen configuration, 1/day): point **1.76**, 80% interval **[1.20, 1.87]**.

**2. P(G0 passes) = 0.12.**
- P(clause 1: R_tune ≥ 1.15 and CI_lo > 1.00) = **0.80**.
- G0 is low because it needs eleven conditions together, not because I expect the cascade to be weak.

**3. Per-gate pass probabilities** (the gate's recorded outcome for the chosen configuration, if it is evaluated):

| Gate | P(pass) |
|---|---|
| S0 | 1.00 |
| K1 (twin vs independent Monte Carlo, and off-grid χ²) | **0.40** |
| K2 (linearity) | 0.85 |
| N1 (float32 vs float64) | 0.98 |
| F1 nulls | 0.97 |
| F1 wrong-template | 0.93 |
| F2 (i) single sine-Gaussians | 0.95 |
| F2 (ii) glitch trains | 0.88 |
| F2 (iii) power drift, primary statistic | 0.84 |
| P1, per-q median ρ̂/ρ_opt ≥ 0.90 | 0.78 |
| P1, ≥ 90% of ρ_opt ≥ 15 pass the twin threshold | 0.88 |
| G0 clause 3 (cost ≤ 120 h) | 0.90 |

**4. Configuration T will choose** (probabilities sum to 1):

| | p = 4 | p = 8 | p = 16 |
|---|---|---|---|
| L_max = 128 | 0.09 | 0.21 | **0.25** |
| L_max = 384 | 0.08 | 0.17 | 0.20 |

## 2. Supporting numbers I also commit to (not in the brief's grading; they show where I would be wrong)

| Quantity | My value |
|---|---|
| Gaussian-twin 1/day threshold on ρ̂_net | 7.2, range 6.9–7.6 |
| Real-background 1/day threshold on ρ̂_net (χ² + v) | 8.2, 80% interval 7.3–11 |
| Real / twin threshold ratio | 1.12, 80% interval 1.0–1.5 |
| Median ρ̂/ρ_opt over caught injections | p = 16: 0.95; p = 8: 0.92; p = 4: 0.89 |
| Caught injections lost to anything but the threshold (glitches in the span, box misses) | 3–10% |
| K1 ratio, twin over an ideal white-noise Monte Carlo, at survival 10⁻² | 1.15–1.30 |
| Plain-ρ ablation, R | below 0.3 (as in test 2) |

## 3. Reasoning

### 3.1 R_tune depends almost only on the real threshold

From stored tuning files (`injections.csv`, `results/oracle_snr.csv`, `results/y1_bins/fg_*.json`) I rebuilt the
caught set and the two anchors: 218 caught at one per minute (109.25 Mpc), 111 found by Y1 alone (57.47 Mpc). Both
match `results/idealised_cascade.json`. Then I cut the caught set at ρ_opt ≥ c:

| c | 8 | 9 | 10 | 11 | 12 | 14 | 16 | 18 | 20 | 22 |
|---|---|---|---|---|---|---|---|---|---|---|
| R | 1.89 | 1.88 | 1.84 | 1.78 | 1.69 | 1.49 | 1.29 | 1.21 | 1.11 | 0.97 |

- **Y1 alone behaves like a cut near ρ_opt ≈ 21.** The cascade clears 1.15 if its effective cut is below ~19.
- The ceiling is 1.90. Only 2 of 218 caught injections have ρ_opt < 9, and 22 have ρ_opt < 12.
- A soft version (noise scatter, recovery 0.88–0.95, 0–30% random losses) gives R = 1.6–1.88 for any threshold
  up to 10.5, and 1.2–1.4 at a threshold of 14–16.

### 3.2 Why I expect the real threshold to stay near the Gaussian one

The plain coherent statistic failed in test 2 on transients below the gate. For the registered statistic:

- **One loud transient lands in one frequency bin of a chirp.** Then χ²_r ≈ ρ²/2 for any p, and ρ̂ falls to about 1.6
  per detector whatever ρ was. With k of p bins hit equally, ρ̂ is capped at 1.12·√(k(2p−2)/(p−k)): for p = 16
  that is 2.3 (k = 2), 3.5 (k = 4), 6.1 (k = 8).
- **Noise whose variance is inflated by s² (any slow pattern)** raises ρ by s and χ²_r by s². newSNR then gives at
  most 1.12 × the Gaussian value, before v is applied.
- So the registered statistic bounds both mechanisms I can name. What it does not bound is a disturbance that puts
  balanced, phase-alignable power into most bins. I give that, plus anything I have not thought of, about 30%.
- From memory, not checked at the source: PyCBC's reweighted SNR for long templates has a near-Gaussian background,
  and LVK subsolar searches report loudest candidates at network SNR ~8.5–9.5 at about one per year.
- **Against this:** Y1's own real threshold (16.78) is about 1.46 × what its 1/min threshold (10.65) implies for
  Gaussian tails, despite a consistency test. (My rough estimate, assuming Y1's statistic has 32 degrees of freedom.) Y1's test is on chunk powers, which is weaker than a phase-coherent
  frequency χ², but it is a warning from the same data.

### 3.3 Scenario mix behind the R_tune numbers

| Scenario | Weight | R_tune |
|---|---|---|
| Statistic works: real threshold ≤ 10 | 0.68 | 1.72–1.87 |
| Partial: threshold 10–15, or ~30% of signals lost | 0.18 | 1.25–1.70 |
| Bad: heavy residual tail, broad signal loss, or a defect | 0.14 | 0.5–1.25 |

Raw, this gives P(R ≥ 1.15) ≈ 0.88. I lower it to 0.80 because this repo's idealised-to-real steps have
disappointed more often than not (test 2: 109 Mpc predicted, 7.8 measured).

### 3.4 Why K1 is my most likely failure

- The per-candidate maximum sits deep in the tail (ρ² ≈ 43–53 at survival 10⁻²), where survival changes by about
  0.47 per unit of ρ².
- Test 2's twin ran 1.06× (at 10⁻³) and 1.17× (at 10⁻⁵) above the ideal χ²₄. That is a variance excess of 0.7–1.2%,
  which I attribute to whitening with an estimated PSD.
- The same excess at the per-candidate maximum is a factor 1.15–1.35 in survival: at or beyond the ±15% bar.
- v's own estimation noise (about 1.7% over 8 s, 50–480 Hz) adds roughly +8% if the Monte Carlo omits v.
- The cost plan has two passes and no third. A subsampled Monte Carlo of 5,000 candidates has ±14% sampling error at
  survival 10⁻² on its own.
- K1 passes comfortably only if the Monte Carlo goes through the identical whitening chain with a full-size sample.
  I give that reading about 45%.
- The off-grid χ² clause is tight at p = 16: bins resolve nearly all of the mismatch, so the mean sits close to
  Allen's bound, and a 1% noise-variance excess can push it over.

### 3.5 P1 and the choice of p

With the grids' median fitting factor 0.990 (`G_grids.json`) and χ² excess ≈ 2·m·ρ² resolved by the bins, the
newSNR factor at the median caught injection (ρ_opt 22.6) is:

| p | newSNR factor (median) | × fitting factor |
|---|---|---|
| 4 | 0.90 | 0.89 |
| 8 | 0.93 | 0.92 |
| 16 | 0.96 | 0.95 |

- **p = 4 sits on the 0.90 bar**, so I expect it to fail P1 more often than not, and it is the weakest against
  balanced glitch trains. That is why p = 4 gets 0.17 in total.
- At L_max = 128, templates with chirp mass below ~0.32 lose a few percent of SNR below their start frequency. That
  touches about half of the q < 0.5 stratum and pulls its median down by 1–3%.
- Because T selects subject to the gates, the chosen configuration passes P1 and F2 more often than a random one.

### 3.6 L_max, and ties

- Distance is close to its ceiling in my main scenario, so configurations differ by a handful of injections.
- L_max = 128: shorter spans, fewer transients in them, fewer templates. L_max = 384: full SNR for light systems
  (31% of the caught volume weight has chirp mass < 0.32). I lean 128 at 0.55.
- Identical found sets for p = 8 and p = 16 are plausible. A tie would then be broken by an order I do not know,
  probably the first listed, so I moved weight from p = 16 to p = 8.

### 3.7 The remaining gates

- **F1 nulls:** Binomial(218, 1/1440) gives x* = 2. Nulls sit at random times, where the pass rate is below
  1/1440.
- **F1 wrong-template:** a loud signal can leak into a +5% box only where two chirp tracks touch, which is one or
  two bins, so χ² caps it.
- **F2 (iii):** the worst case is a drift that covers the whole span but that v misses. ρ̂ is then at most 1.12 × a
  Gaussian box maximum, which I put at a 1–2% pass rate against a 2% bar.
- **K2:** test 2's linearity held at 0.22% on ρ². The risk is the injection placed on a gated glitch.
- **Cost:** the model says 18–22 h. Even 3× over is inside 120 h. The risk is a contended box inflating T's
  measured wall time, or a fall back to CPU.

## 4. Optional, advisory only (this does not change the registration)

**Do the three F2 families represent what failed in test 2?** Partly. (i) and (ii) cover impulsive transients, and
(iii) covers broadband drift, which is what v is built for. Two cases are not covered:

1. **A disturbance that is away from the merger time, or band-limited.** v is read at the peak time over 8 s and
   50–480 Hz. A 128–384-s template collects roughly half its SNR between 50 and 100 Hz, tens to hundreds of seconds
   earlier. A decoy I would add: a noise-variance increase confined to 50–100 Hz, lasting 10–100 s, ending at least
   16 s before the merger time, reported with and without v. It tests whether χ² alone carries that case.
2. **Glitch trains placed on the template's own track with equal SNR per bin.** Random sine-Gaussians mostly miss a
   track that is ±1.5% wide in frequency, so after scaling the family tends to collapse to one dominant glitch,
   which is the easy case. On-track, balanced trains are the adversarial case, and they separate p = 4 from p = 16.

## 5. Compute record

- Reading, plus three short numpy/scipy runs on stored CSV and JSON files (seconds each, one process, CPU only,
  footprint well under 1 GB). No MPS, no strain, no fresh data.
- Scratch script kept outside the repo; it re-derives the table in §3.1 from the three files named there.

Environment: Mac (macOS 27.2, arm64) · `primordial_blackhole_search/requirements-lock.txt` · Python 3.12.13, numpy
2.4.6, scipy 1.17.1 · git d01cb8d (branch point).
