# Step T: clarifications of the registered text (dated 2026-11-26, before any step-T number exists)

*Deepstrain-personal [model: Claude Opus 5.5], on the Bridge's instruction (Bridge - claude personal, 2026-11-26).*

Scope:
- Clarifies `reports/stepT_registration.md` (amendment T1 to `PREREG_coherent2.md`, d4c3ca4) where its wording is
  ambiguous.
- **The registered text governs.** These readings fix how the code applies it.
- Every one applies identically to background, twin, decoys, nulls and foreground.
- Committed before the driver's first run. No step-T number (no stage-2 score on any candidate) exists at this commit.
- Implemented in `primordial_blackhole_search/benchmark/coh2t_drive.py`.

## C1. "Top K = 5 (template, lag) peaks of |z|²" (registration §3)

- **Reading:** take each template's own best lag first. The K = 5 points are then the 5 templates with the largest
  per-template peak of |z|², each at that best lag. This is done per candidate, for H1 (see C2).
- **Why:** adjacent lags of one template have almost the same |z|². Without de-duplication, the top 5 would be five
  neighbouring samples of a single template.
- **Status:** accepted by the Bridge.

## C2. The network value ρ̂²_net = ρ̂_H² + max over |δ| ≤ 10 ms of ρ̂_L² (registration §3)

**Reading.** The formula is H-anchored, as written:
- the anchor points are C1's K points, chosen in H1 by |z_H|²;
- at each, ρ_H and Allen's χ² (p = 4, 8, 16) are computed at that H1 (template, lag);
- the L1 term uses the **same template**. For that template, ρ_L and χ² are computed at **every** L1 sample within
  ±10 ms (±41 samples at 4,096 Hz) of the H1 lag;
- PyCBC then applies v and newSNR at each point's own GPS time, in its own process. The max over δ is taken of the
  resulting ρ̂_L²;
- **candidate score** = max over the K anchors of ρ̂_H² + max_δ ρ̂_L².

**Why the same template.** The registration does not say. The stage-2 statistic's direct predecessor, the
real-background cascade, paired the detectors on the same template:
- `realbg_cascade.survival`: `net = Hz[r0:r0 + 8] + _lmax3(Lz[r0:r0 + 8])`;
- rows are templates, and `_lmax3` takes the L1 maximum over adjacent bins within each row.

Neither the verified box (`coh2c_verify`) nor gate E3 used two detectors, so neither fixes this point. Y1 itself
combines per-detector maxima, which may sit on different templates. That is not followed here, because stage 2 is
coherent per template.

**Withdrawn alternative.** An earlier draft of the driver paired in both directions and let L1 take any template. The
Bridge rejected it: it is a superset of the registered formula and raises both background and foreground. It was
never run.

**Consequences, stated so nothing is a surprise later:**
- L1's own loudest templates do not enter the score unless they coincide with an H1 anchor.
- For an anchor within 10 ms of the ±1-s search edge, the L1 lags can extend up to 10 ms beyond it, as the formula
  implies.

## C3. Box centre (PREREG §3.3, "around each candidate's Y1 best template")

- **Background:** the box is centred on the **H1** template at the candidate's Y1 bin.
- **Why:** this is Y1's own foreground convention (`y1_bank.network_trigger` returns the H1 tid). It is also the one
  the stage-1 bias, and hence the 2.20% half-width, was measured with.
- **Status:** accepted by the Bridge.

## C4. Edge rule

- **Rule:** a candidate is scored only if its centre sample lies in [392 s, 4,088 s] of its segment, i.e.
  [(384 + 8) s, 4,096 − 8 s]. This is the real-background cascade's rule (`realbg_cascade.EDGE_LO/EDGE_HI`).
- **Excluded candidates** are counted per kind, and the background livetime is scaled by the valid fraction.
- **Foreground:** every tuning injection lies at 400–3,340 s, so every caught injection is valid.
- **Status:** accepted by the Bridge.

## C5. Foreground centre

- **Rule:** a caught injection's box and ±1-s window are centred on its **Y1 trigger** (the time and tid in
  `results/y1_bins/fg_*.json`), not on its true merger. This is exactly as for background candidates.
- **Why:** the foreground stays as blind to the truth as the background.
- **Status:** accepted by the Bridge.

## Ablations (registration §4), in the same form

- **Plain ρ:** max over all templates and H1 lags of |z_H|² + max over |δ| ≤ 10 ms of |z_L|², on the same template.
  It is computed exactly over the whole box.
- **χ² only:** the primary statistic with v set to 1.

Environment: Mac (macOS 27.2, arm64) · text only, no computation · git 3ddf5a5 (parent).

## C6–C9 (2026-11-26, later the same day): twin, nulls and decoys

Written with their code (`coh2t_drive.py`) before any of these kinds ran.

**Rulings (Bridge - claude personal, 2026-11-26):**
- C6, C7 and C8 are ACCEPTED, with the additions marked below.
- C9 was accepted with a cheap check required. The check then failed, so C9 is open again; see C9 and the run notes.

### C6. Decoy amplitude calibration (registration §4, "each decoy family scaled to the median plain-coherent ρ of caught tuning signals")
- The fg pass also scores each caught injection's signal **alone** (zero noise, the noise's own float scale) and keeps
  that signal-only plain network value.
- **Target:** the median of the signal-only plain network value over the valid caught injections, per L_max.
  `calib` writes it once and refuses to overwrite it.
- Each sine-Gaussian or glitch-train decoy is scaled, as a whole, so that its own decoy-only plain network value equals
  the target. Then it is scored in real noise.
- **Why signal-only on both sides:** it compares like with like. A value measured in noise would mix in each window's
  own noise maximum.
- **Power drift is not scaled.** It is multiplicative, and its strength is the registered 10–30 % range itself.
- **Status:** ACCEPTED.

### C7. Decoy families: parameters, drawn once in the plan with this build's seed 202611261
- **Placement:** in **both** detectors, at a valid background candidate time, with parameters drawn independently per
  detector.
  - It is the harder case for an H-anchored network value.
  - The registration does not say whether decoys go in one detector or both.
- **(i) single sine-Gaussian:**
  - f₀ log-uniform in [50, 400] Hz;
  - Q uniform in [4, 20];
  - random phase;
  - centre uniform within the template span before the candidate, t = i_c − u·L_max.
- **(ii) glitch train:** 3–10 sine-Gaussians per detector, each as in (i), with relative amplitudes uniform in
  [0.5, 1], at independent uniform times within the span.
- **(iii) power drift:** the noise multiplied by 1 + a·bump(t).
  - a is uniform in [0.10, 0.30];
  - bump is a Hann window, duration uniform in [10, 100] s, centred uniformly within the span;
  - the decoy is the change of the gated raw strain, x·g·a·bump, then whitened.
- **Gating and whitening:** every decoy is added to the raw strain and gated with the **noise's own** gate window, then
  double-whitened with the noise's PSD.
  - This is exactly the linearity route used for injections, so decoys and foreground are treated the same.
  - A consequence, stated now: a decoy loud enough that PyCBC autogating would have caught it is not gated here.
    That makes the test harder, not easier.
- **Status:** ACCEPTED. The parameter ranges are **my choice, not registered text**, except the 3–10 count, the
  10–30 % strength and the 10–100 s duration. Why each:
  - **f₀ 50–400 Hz:** 50 Hz is the stage-2 band's lower edge. The template SNR of these light chirps sits mostly below
    a few hundred hertz: our CNN response probe put 97–100 % of its sensitivity below 224 Hz. 400 Hz covers that with
    margin.
  - **Q 4–20:** short, blip-like transients (low Q; from memory, blips have Q of a few) up to longer-ringing ones. Q 20
    at 50 Hz lasts about 0.1 s, still short against a 128–384-s template.
  - **Train relative amplitudes 0.5–1:** no single glitch dominates. That is the harder case for χ², because the power
    is spread across bands.
  - **Times uniform within the template span:** that is where a coherent template integrates.
  - **Hann bump for the drift:** a smooth, slow modulation, so the decoy is a power drift rather than a transient.

### C8. F1 nulls
- **(a) null:** the caught injections' own windows (C5: Y1 trigger time and tid), scored with the signal left out.
- **(b) wrong:** the same windows with the signal in, but the box centred at **Y1's tid Mc × 1.05**, the nearest bank
  template. The registration says "a box offset to chirp mass +5 %"; the offset is applied to the box, which is
  centred on Y1's template (C3), not on the true mass.
- **Bar, as registered:** passes ≤ x*, with x* = 2 for n = 218 under Binomial(n, 1/1440).
  - Where the number comes from: registration §5 F1 defines x* as the smallest count with P(X > x*) ≤ 0.01 for
    X ~ Binomial(n, f/r₁), with f/r₁ = (1/day)/(1/minute) = 1/1440.
  - For n = 218 valid caught injections, P(X > 1) = 0.0103 (just above 0.01) and P(X > 2) = 0.00051, so x* = 2
    (scipy.stats.binom.sf, computed 2026-11-26).
  - The number is not in the registration; the rule is.
- **Status:** ACCEPTED.

### C9. Gaussian twin (K1), drawn on the snippet grid
- **Rule:** each valid background candidate is scored again on Gaussian noise drawn directly on the 512-s snippet grid:
  Y_k = w_k·√(NS/2)·(a + i b), with a, b ~ N(0, 1), seeded by (202611261, candidate, detector).
  - w is the gated noise's whitening filter on the snippet grid (`coh2c_local.prep`).
  - This gives E|Y_k|² = NS·w_k², the same as a 512-s cut of the real series. That is exactly realbg's twin spectrum,
    there drawn on the full segment, restated on the snippet grid.
- **Why the snippet grid is equivalent:**
  - The draw is circularly stationary on 512 s.
  - The correlation length of double-whitened noise is set by the 16-s inverse-spectrum truncation.
  - The data a template uses lie ≥ 50 s from the snippet's ends.
  - So the samples used have the same Gaussian statistics as a cut of a full-length draw.
- **Status: the required check FAILED twice.** C9 was then amended to option (i) on the Bridge's ruling. See the
  run notes, 2026-11-26 12:26–12:28.

**C9 amendment (2026-11-26, post-check, before any step-T number exists): the twin is a cut of a full-length draw.**
- **What was wrong.** My equivalence argument above fails at narrow spectral lines. Whitening puts deep notches
  there, and the real pipeline's 512-s rectangular cut leaks sidelobe power into them.
  - At 476–537 Hz: a cut of a full draw shows 3.42×, real data 3.53×, the snippet-grid draw 1.00.
  - At 963–1,024 Hz: 2.11×, 2.04× and 1.00.
  - A 2 × 512-s draw cut to 512 s carries most of the leakage, but it still failed the same pre-set bar.
- **Both FAIL runs stay on record.**
- **Rejected:** keeping a padded draw and designing a better-posed check. The Bridge ruled that would be a post-hoc
  rescue of a failed registered check.
- **New construction, option (i):**
  - each twin snippet is realbg_cascade's twin built on the **full 4,096-s grid**: Gaussian, the gated series' own
    PSD, double-whitened;
  - it uses an independent seed per candidate and detector, (202611261, candidate, detector);
  - it is cut to 512 s exactly as the real series is cut;
  - so it carries the same notch leakage by construction.
- **How the PSD is rebuilt exactly.** The full-resolution PSD comes from the PSD estimate's own 16-s
  inverse-spectrum-truncation kernel: 2 × 32,768 samples per series, about 0.5 MB. The kernel step asserts, bit for
  bit, that it reproduces `lowmem_psd.estimate_psd` on prep's gated input, and that its snippet-grid filter equals the
  one prep stored.
- **Cost:** about 0.6 s CPU per candidate and detector, roughly 12 h of child CPU per L_max pass, on the 1 extra core
  the Bridge accepted. Twin snippets are transient: about 8 MB each, deleted after their box.
- **Post-hoc context only, never a gate:** the matched-filter-weighted noise power (weight f^(−7/3)·w², the |z|²
  variance of a TaylorF2 template). Measured on run 2: twin 0.998, cut of a full draw 1.004, real 0.996.
- **Replacement check, registered before it runs (`coh2t_drive.py twinexact`):** an exact identity. For 4 twin
  candidates (H1 and L1, tuning segment 0, different seeds and cut positions):
  - the driver's twin snippet must be bit-identical to a reference built independently from the data:
    `lowmem_psd.estimate_psd` of prep's gated series, `realbg_cascade.whiten_filter`, realbg's twin lines written out
    again, the same seed and the same cut;
  - this tests that the code does what (i) says. K1-(A) tests the statistics.
- **K1 design (ruled):** option (C).
  - (A) is the registered gate: the same grid, independent Gaussian noise, ~5,000 maxima, ±15 % at 10⁻².
  - (B), numpy brute force at 10⁻¹, is an extra engine cross-check only. It runs when cores free up and does not
    block T.
  - The off-grid twin Allen bound is written before (A) runs.
  - **Open, raised with the Bridge:** under (i), K1-(A)'s "independent Gaussian noise" would come from the same
    generator as the twin. It would then test only seed independence. An independent generator would keep it
    informative, for example pycbc.noise.noise_from_psd: time-domain coloured noise, run through prep's whitening path
    at random valid times.

## C10 (2026-11-26): K1-(A) with an independent noise generator

Registered before any K1 code existed. The generator choice and the sample size are both ruled by the Bridge (below).

**Generator.** `pycbc.noise.noise_from_psd`, a different algorithm from the twin's, run in the child process so pycbc
stays out of the engine. The series then goes through prep's own whitening path:
- gated? no, it is synthetic;
- double-whitened with the real segment's filter w, so the twin and the MC share their whitening and differ only in
  how the Gaussian noise is generated.

**(3) Conventions, read at source: pycbc 2.10.0, `pycbc/noise/gaussian.py`, lines 78–131.**
- Seeding: `lal.gsl_rng("ranlux", seed)` with an integer seed. Ours is seed = 7,000,000 + 2·i + d, for segment index
  i and detector d (H1 = 0, L1 = 1). These are distinct from every other seed in this build.
- Segmenting: segment length N = 1/(Δt·Δf), stride N/2, stitched by `lalsimulation.SimNoise`. How SimNoise colours
  and stitches internally is from memory, not read.
- The PSD's DC and last bins are set to 0.
- There is **no** low-frequency cutoff inside the function.
- **Our inputs:**
  - Δt = 1/4,096 s; length 2²⁴ samples (one 4,096-s series per detector and segment).
  - Δf = 1/16 Hz, the PSD estimate's own resolution (16-s Welch segments). The PSD passed is the segment's exact
    pipeline PSD (rebuilt from its kernel) sampled every 256th bin.
  - Below F_LOWER − 10 = 40 Hz the PSD is extended flat at its 40-Hz value. There the truncated PSD is an artefact of
    the inverse-spectrum truncation, and prep's whitening filter is zero, so in-band statistics are unaffected and
    low-frequency leakage stays small.

**(1) Pre-check, before any K1 maximum.**
- For segment 0, H1 and L1: the Welch estimate of the generated series (`lowmem_psd.estimate_psd`, the pipeline's own
  median Welch with 511 segments of 16 s) divided by the target PSD, averaged over each of 16 equal bands in
  [50, 1,024] Hz.
- **Bar:** |band mean − 1| ≤ 1.5 % in every band.
- **Why that tolerance.**
  - A median-of-511 estimate has a relative scatter of about 1.44/√511 ≈ 6.4 % per frequency bin.
  - Each 61-Hz band holds about 975 bins at 1/16 Hz, roughly 650 independent ones after Hann-window correlation.
  - So a band mean scatters by about 0.25 %, and 1.5 % is about 6σ.
  - The bar catches convention errors (factors of 2, Δf, one-sided against two-sided), which give ≥ 30 %, not noise.
- The ratio is of two PSDs, both large at the lines, so unlike the failed C9 bar it is well conditioned in notches.

**(2) Comparison, as registered (frozen §4 K1; registration §5).**
- The per-candidate-maximum survival of the twin, with ρ̂²_net as the primary statistic, against the pycbc-noise MC,
  through the same boxes.
- **Bars:** ±15 % at survival 10⁻², and within a factor of 2 at the deepest level the MC populates.
- **Design:**
  - MC samples are drawn segment-major: the 2 series of a segment are generated, scored, then deleted (128 MB
    transient).
  - Each sample takes the box and detector pairing of a randomly chosen twin candidate (seeded), at a random valid time
    in [392, 4,088] s of that segment, so the box mix matches the twin's.
  - Matched-filter output decorrelates within milliseconds, so samples from one series at different times are treated
    as independent.

**OPEN, the sample size. My earlier "~5,000 maxima" was WRONG.**

| MC maxima | Events at 10⁻² | sd of the twin/MC ratio | P(fails ±15 % with no real difference) |
|---|---|---|---|
| 5,000 | 50 | 0.151 | **0.32** |
| 35,482 (= twin) | 355 | 0.075 | 0.046 |
| 100,000 | 1,000 | 0.062 | 0.015 |

- The sd combines the MC with the twin's own 355 events.
- With 5,000 maxima the registered bar would be a coin flip. A sample size is needed before (A) runs.

**RULED (Bridge - claude personal, 2026-11-26, pre-result): N_MC = 100,000 per L_max, fixed. The ±15 % at 10⁻² bar is
unchanged.**
- **Stated plainly:** even if the twin and the MC are identical in truth, this gate fails by chance about **1.5 %** of
  the time (Poisson approximation, sd 0.062 of the ratio).
- **No sequential or two-stage reading:** no "35k, then extend if it fails". A fixed N keeps the error rates exactly as
  stated.

**Independence of the 100,000 maxima.**
- **The problem:** cut from one pycbc series per (det, segment), the samples would sit about 0.9 s apart, so their
  ±1-s windows would overlap and the maxima would be correlated.
- **N_REAL = 8 independent realisations of 12,500 samples each.** Within a realisation, each sample takes a 4-s time
  slot that no other sample uses in the same H1 series or the same L1 series. There are 924 slots per segment, and
  about 521 are used per segment.
- **Seeds:** lal ranlux seed = 7,000,000 + 1,000·r + 2·segment + detector. Realisation 99 is reserved for the
  pre-check and never used by K1.

**Disk.** One realisation is 48 series, 3.1 GB, so realisations run one at a time: generate, run both L_max, delete.
That only fits after the main passes. At that point the real noise series (3.1 GB) must be off the disk; they can be
regenerated by `prepseries` in about 1 h. The K1 phase needs only prep's small filter files. **Deleting them is a
decision for the Bridge** before K1 starts.

## C11 (2026-11-26): the off-grid Allen bound for K1's second clause — PROPOSED, before any code or run

The frozen text says: "off-grid injections in the twin: the χ² mean must stay at or below Allen's mismatch upper bound
for their measured mismatch". The bound is **derived here** for our exact χ², so it does not rest on a recollection of
Allen (2005). Comparing it with Allen's paper at source is not done; the derivation stands on its own.

**Setting.**
- Bands j = 1…p split the frequency range so that the template T (normalised, |T|² = 1) has |T_j|² = 1/p in each.
- The complex inner product is ⟨a, b⟩ = Σ_f a b*·weight; T_j is T restricted to band j; ⟨T, T_j⟩ = 1/p.
- At the peak's time and phase, z_j = s_j + n_j and z = Σ_j z_j.
- Our χ² = p·Σ_j |z_j − z/p|².
- In Gaussian noise E|n|² = 2 (|z|² ~ χ²₂), E|n_j|² = 2/p, and the noise part of χ² has mean 2(p − 1), i.e. 2p − 2 degrees
  of freedom.

**Derivation.**
1. Write the normalised signal as ĥ = α T + r, with r ⊥ T and m = |α| its match to T, so |r|² = 1 − m².
2. With ρ the signal's optimal |z| (an exactly matched template), s_j = ρ (α/p + ⟨r, T_j⟩), and s = ρ α.
3. Hence d_j ≡ s_j − s/p = ρ ⟨r, T_j⟩.
4. Cauchy–Schwarz within each band: |⟨r, T_j⟩| ≤ |r_j| |T_j| = |r_j|/√p.
5. Summing over bands: Σ_j |d_j|² ≤ ρ² (1 − m²)/p.
6. The cross terms of signal and noise have zero mean, so

   **E[χ²] = 2(p − 1) + p·Σ_j |d_j|²  ≤  2(p − 1) + ρ² (1 − m²)  ≤  2(p − 1) + 2 ε ρ²,** with ε = 1 − m.

**Test, run only after the Bridge accepts this.**
- **Injections:** 200 per L_max, per detector, into twin noise. Each sits at a random valid twin candidate's time,
  offset uniformly within ±0.5 s, at an (Mc, q) drawn uniformly inside that candidate's box, so it is off-grid by
  construction. The signal is TaylorF2, generated in the child.
- **Amplitude:** set so the noise-free optimal |z|, with the exact template evaluated by the engine, is ρ, with ρ
  uniform in [8, 25].
- **Measured, per injection:** m, the largest noise-free |z| over the grid divided by ρ; and χ² (p = 4, 8, 16) at that
  grid template's peak in the noisy data.
- **Bar, per p:** mean_i χ²_i ≤ mean_i [2(p − 1) + ρ_i² (1 − m_i²)] + 3·SE, with SE = sd_i(χ²_i − bound_i)/√200.
- **Why the 3·SE allowance:** at zero mismatch the bound equals the noise mean, so a plain "at or below" would fail
  about half the time by chance alone. A true failure (χ² above the bound) grows with ρ² and is not hidden by 3·SE.

