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
regenerated by `prepseries` in about 1 h. The K1 phase needs only prep's small filter files.
- **RULED (the Bridge): deletion approved, with a condition.**
  - `hashseries` records the sha256 of every series file, along with its filter and rows files, in
    `results/coh2t/series_sha256.json`, and that record is committed.
  - `dropseries` re-hashes every series against the record before deleting anything.
  - Any later `prepseries` must reproduce every hash exactly before its output is used. A mismatch stops the step and
    is reported.

## C11 (2026-11-26): the off-grid Allen bound for K1's second clause — derivation and design APPROVED

**Non-author read (Bridge - claude personal, 2026-11-26): the derivation is CORRECT and the design APPROVED.** Record:
`TheBridge/fleet/reads/deepstrain_C11_read_2026-11-26.md`.
- Each step was checked: the noise mean 2(p − 1); s_j and d_j = ρ⟨r, T_j⟩ with Σ d_j = 0; the per-band Schwarz step;
  the bound ≤ ρ²(1 − m²) ≤ 2ερ².
- **The subtlety it names:** the bound holds at a fixed time and phase, while the test measures at the noisy peak.
  Control (a), measured the same way, catches any peak-selection bias, so it is **kept exactly as coded**.

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
  construction.
- **How the signal is built (as coded, 2026-11-26).** It uses the engine's own verified phase model (gate E1:
  5·10⁻¹⁰ rad against pycbc): its nearest reference's TaylorF2 amplitude and band, times exp(i·sign·ΔΨ) from
  lalsimulation's coefficients, double-whitened like the data. There is no pycbc in the engine and no disk.
  - Consequence: the signal shares its reference's start frequency, and so its band edges, with the templates. Allen's
    tighter bound (eqs. 75 / 100) then applies and is reported, for information only.
  - The registered comparison stays the general bound.
- **Amplitude:** set so the noise-free optimal |z|, with the exact template evaluated by the engine, is ρ, with ρ
  uniform in [8, 25].
- **Measured, per injection:** m, the largest noise-free |z| over the grid divided by ρ; and χ² (p = 4, 8, 16) at that
  grid template's peak in the noisy data.
- **Bar, per p:** mean_i χ²_i ≤ mean_i [2(p − 1) + ρ_i² (1 − m_i²)] + 3·SE, with SE = sd_i(χ²_i − bound_i)/√200.
- **Why the 3·SE allowance:** at zero mismatch the bound equals the noise mean, so a plain "at or below" would fail
  about half the time by chance alone. A true failure (χ² above the bound) grows with ρ² and is not hidden by 3·SE.
  The bar is ACCEPTED by the Bridge.

**Relation to Allen (2005), gr-qc/0405045, read at source** (arXiv HTML of v2; the PRD typesetting was not checked, so
equation numbers may differ there):
- **Two-phase case, Sec. VII, eqs. (96)–(99):** ⟨χ²⟩ = 2p − 2 + κ|⟨z⟩|², with 0 ≤ κ ≤ 1/cos²θ − 1 ≤ 2ε. Here
  cos θ is the fitting factor, ε = 1 − cos θ, and ⟨z⟩ is the expected filter output, with ⟨|z|²⟩ = 2 + (D/d)²
  (eq. 83). That is the same noise normalisation as ours.
- With |⟨z⟩| = ρ·m, so cos θ = m:

  κ|⟨z⟩|² ≤ ρ² m² (1/m² − 1) = ρ² (1 − m²).

  **Our bound is identical to Allen's general bound**, reached by the same per-band Schwarz inequality (his
  eqs. 63–64).
- **Allen's tighter bound is not used.** His eqs. (75) and (100) give κ ≤ (p − 1)(1/cos θ − 1)², about (p − 1)ε²,
  when the template and signal share band edges and ε ≤ 2/p.
  - **Correction, after coding:** as built, the injections do share their reference's start frequency (see
    above), so the tighter bound's assumption holds. The general bound stays the registered bar because it is what
    the frozen text names, "Allen's mismatch upper bound". The tighter one is reported alongside, for information
    only.

**Controls, added before coding (the Bridge, 2026-11-26).** Both use the same injection machinery as the real test,
the same 200 injections per L_max per detector, in twin noise, with ρ uniform in [8, 25].
- **(a) ON-grid control, which must sit at the noise mean.**
  - Each injection is exactly a grid template of its box, using its reference's start frequency (D2), so m = 1 and
    κ = 0.
  - The ON-grid case is evaluated at that template and its own peak lag.
  - **Bar, per p:** |mean χ² − 2(p − 1)| ≤ 3·SE.
  - This tests the χ² normalisation, the band edges and the noise level at once.
- **(b) MUST-FAIL control, which must EXCEED the grid's nominal bound.**
  - The injections sit just outside the box, at Mc = Mc*·exp(+0.022 + δ). δ is chosen per injection from a noise-free
    scan (δ = 0.0005, 0.001, …), taking the first value whose best-grid match is ≤ 0.9. As coded, a match below 0.6
    is not refined, only recorded; each injection's m is kept. The match is far worse than the grid's 0.97 minimal
    match.
  - **Bar:** mean χ² > mean [2(p − 1) + ρ²(1 − 0.97²)] + 3·SE, for at least one p, and for p = 16, the most resolving.
  - The bound is mathematically valid at any measured m, so a must-fail against the measured-m bound is impossible by
    construction. What (b) shows is that the test has the power to see χ² above the bound the grid promises when the
    real mismatch exceeds it.
  - The real test is also reported against this nominal bound, for information.

## C10 addendum (2026-11-26 21:20, before the control runs): k1precheck FAILED; a perfect-generator control

**What failed.** k1precheck, run 21:07 on segment 0, failed its ±1.5 % bar. Every other band is within about 0.7 %.

| Band | H1 (est/target) | L1 (est/target) |
|---|---|---|
| 293–354 Hz | 1.068 | 1.021 |
| 476–537 Hz | 11.07 | 1.28 |
| 963–1,024 Hz | 11.14 | 1.57 |

**Rule (the Bridge):** the bar is not changed on an untested hypothesis.

**Hypothesis, untested.** The bar may be ill-posed at strong lines. The target PSD already carries the Welch window's
leakage around each line, and re-estimating generated noise with the same window smears that leakage a second time.
Then ANY correct generator would fail there.

**Control (`k1perfect`), fixed now.**
- A perfect generator: a Gaussian process drawn exactly from the target PSD on the full 4,096-s circle,
  X_k = √(N·f_s·S_k/4)·(a + i b), then irfft. This is the same exact stationary construction as the twin, and it uses
  no pycbc.
- It is run through the same `lowmem_psd.estimate_psd` and the same 16 bands, for segment 0, H1 and L1, with seed
  (202611261, 3, detector).

**Decision rule, per detector, on the two notch bands (476–537 and 963–1,024 Hz):**
- **ILL-POSED**, if both of these hold:
  - the perfect generator also fails ±1.5 % in BOTH notch bands;
  - in each, its excess is "the same" as pycbc's, meaning 2/3 ≤ (r_perfect − 1)/(r_pycbc − 1) ≤ 3/2.

  Then a redefined check is registered, with its exclusion or windowing fixed in advance, BEFORE k1precheck reruns.
- **GENERATOR AT FAULT**, if the perfect generator passes all 16 bands.
- **INCONCLUSIVE**, in every other case: reported, with no rerun and no change.

**The 293–354 Hz band** is reported the same way, but it is not part of the decision.

**Line heights** (explaining H1 at about 11× against L1 at about 1.3× in the same bands):
- For each detector and band: the target PSD's peak-to-median ratio and the fraction of band bins above 10× the
  band median.
- Prediction under the double-smear hypothesis: the excess grows with line strength, so H1's lines in those bands
  must be much stronger than L1's.
- **If the line heights are similar, the hypothesis is weakened**, whatever the decision above says.

### k1perfect result (run 21:21, code d58ffe5): INCONCLUSIVE for both detectors, by the registered rule

| Detector, band | Perfect generator | pycbc | Ratio of excesses (perfect/pycbc) | Line height (peak/median) |
|---|---|---|---|---|
| H1, 476–537 Hz | 28.8 | 11.07 | 2.76 | 1.3·10⁹ |
| H1, 963–1,024 Hz | 52.2 | 11.14 | 5.05 | 1.1·10⁹ |
| L1, 476–537 Hz | 1.223 | 1.285 | 0.78 (inside 2/3–3/2) | 1.6·10⁷ |
| L1, 963–1,024 Hz | 1.018 | 1.572 | 0.03 | 33 |

- In every other band the perfect generator is within 0.6 %.
- **H1:** the perfect generator fails in both notch bands, but its excess is not "the same" as pycbc's.
- **L1:** the second band fails the "same" test.
- **Per the rule: reported, no rerun, no change.** k1precheck's FAIL stands, and K1-(A) is blocked until the Bridge rules.

**Descriptive reading, not a verdict.**
1. **Where the lines are huge** (H1's two bands, L1's 476–537 Hz), even an exact Gaussian with the target PSD fails
   the bar, and it fails more where the lines are taller: H1 at 10⁹ against L1 at 10⁷. That fits the double-smear
   hypothesis. **There the bar cannot be met by a correct generator.**
2. **L1 at 963–1,024 Hz has no strong line** (peak/median 33). The perfect generator nearly passes there (1.018),
   while pycbc gives **1.57**. That is a pycbc-generator-specific deviation near the band top, which the double-smear
   does not explain. It is a real open question about the K1-(A) generator.

## C12 (2026-11-26, before production): production closure tolerance

**What the record says.** The 1e-4 closure bar is registered **only for the golden** (`coh2t_drive.golden`'s
docstring, committed before the golden runs). No production tolerance was registered.

**What was observed first.** Disclosed, because it was seen before this was written: the fixed 2-box pilot gave
closure up to **1.5e-4** relative to max(|z|, 1), on box 1410 bg.

**Proposal: a correctness gate, stopping at the box boundary, of closure ≤ 1e-3 relative to max(|z|, 1),** at every
anchor and in both slots.

**Reasoning: the effect on any decision.**
- Closure measures the gap between the χ² band sum and the engine's z, so it bounds the error in ρ by 1e-3 × max(|z|, 1).
- **Noise scale:** |z| has σ ≈ 1 per quadrature (E|n|² = 2), so 1e-3 is 0.1 % of the noise scale.
- **At the threshold:** at a 1/day-like ρ̂_net of about 9, the error is at most 0.009 in ρ, or 0.2 % in ρ².
- **Against the statistics:** in K1 a 1.2 % variance excess moves survival at 10⁻² by about 28 %. Survival changes by
  about 0.47 per unit of ρ², so 0.2 % of ρ² ≈ 80 is about 0.16 units, a survival shift of about 7 %. That is
  comparable to K1's statistical sd.
- **The pilot's 1.5e-4 is 7× smaller,** about 1 %, well below every statistical error in play.
- Hence 1e-3 is the loosest gate that stays clearly below the decision-relevant scale, and production values are
  expected around 1e-4.

**RULED (Bridge - claude personal, 2026-11-26): closure ≤ 3e-4, not 1e-3**, as a correctness gate.
- At 1e-3 the worst-case deterministic survival shift (about 7 % at 10⁻²) is comparable to K1's statistical sd. A
  deterministic bias should not be allowed to be that big.
- 3e-4 is 2× the observed 1.5e-4 and keeps the worst case near 2 %, about a third of the sd or less.
- **Disclosure:** the 1.5e-4 was seen before this tolerance was set.
- **As coded:** each box's `closure_max` is logged in its output, and a breach stops the run once that box has been
  written (`CLOSURE_MAX = 3e-4`).

### k1perfect, the Bridge's reading (2026-11-26)

**(a) A design flaw in the K1 pre-check clause.** The ±1.5 % band-mean bar on Welch re-estimates is ill-posed at
strong lines: the exact generator fails it harder than pycbc does (28.8 and 52.2 against 11.1 at H1). This is recorded
as a flaw in the clause, in the same class as the voided V21/V22 control clauses, **not as a pycbc result.**

**(b) The real finding.** pycbc departs from an exact generator, measured as pycbc / perfect per band:
- L1, 963–1,024 Hz: 1.572/1.018 ≈ 1.54;
- H1, 476–537 Hz: 11.07/28.8 ≈ 0.38;
- H1, 963–1,024 Hz: 11.14/52.2 ≈ 0.21.

Any redefined bar must be argued from the Welch estimator's own variance and fixed before it is applied. It cannot be
tuned to these numbers, which have all been seen.

## C13 (2026-11-26, before running): investigating the pycbc deviation (b)

**Variants (`coh2t_drive.py k1diag`).** CPU, one at a time, segment 0, H1 and L1, the same 16 bands, each reported as
est/target and as variant/perfect.
- **V0:** pycbc at Δf = 1/16 Hz, seed of realisation 99. Must reproduce k1precheck **exactly**, as a reproduction
  check.
- **V1:** pycbc at Δf = 1/64 Hz (PSD sampled every 64th bin).
- **V2:** pycbc at Δf = 1/8 Hz (every 512th bin).
- **V3:** pycbc at Δf = 1/16 Hz, with the PSD above 1,024 Hz replaced by its value at 1,024 Hz.
- **V4:** the perfect generator, but fed the 1/16-Hz-sampled PSD, linearly interpolated back to the full grid. This
  introduces only the PSD-sampling factor, with no pycbc.
- **V5:** the perfect generator, which must reproduce k1perfect **exactly**.

**The three deviations, stated before running:**
- **D1:** L1, 963–1,024 Hz, pycbc/perfect = 1.54.
- **D2:** H1, 476–537 Hz, 0.38.
- **D3:** H1, 963–1,024 Hz, 0.21.

**What counts as "explained", fixed now.** A factor is the explanation of a deviation D only if both hold:
1. Changing only that factor in pycbc (V1, V2 for PSD sampling; V3 for the PSD above 1,024 Hz) moves pycbc/perfect
   in that band at least 75 % of the way to 1, i.e. |r_variant/r_perfect − 1| ≤ 0.25·|D − 1|.
2. For PSD sampling, also V4: introducing only that factor into the perfect generator reproduces D's sign with at
   least 50 % of its size.

**Other outcomes.**
- Whether a deviation moves monotonically with Δf across V2, V0 and V1 is reported, as "moves with Δf: yes/no".
- **If no factor meets the rule, D is recorded as UNEXPLAINED.** No bar is redefined either way until the Bridge rules.
- **Sample rate is NOT varied.** At f_s = 2,048 Hz the band top becomes Nyquist, which makes D1's band meaningless,
  and the data cannot carry a PSD above 2,048 Hz for a higher rate.
- **Cost:** 6 variants × 2 detectors × about 8 s, about 2 min, peak ≤ 2.2 GB.

## C14 (2026-11-26, before anything runs): what in step T depends on K1-(A)

**G0 depends on K1-(A), but only through K1's first clause** (twin survival against the K1-(A) Monte Carlo). Every
other part of step T is independent of K1-(A).

| Part | Depends on K1-(A)? | State |
|---|---|---|
| Main passes, L_max 128 and 384 (bg, twin, fg, null, wrong) | no | ready |
| Decoy calibration (calib) and the three decoy families | no | ready |
| K1 second clause (C11: off-grid Allen bound with the ON-grid and MUST-FAIL controls), in twin noise, no pycbc | no | ready |
| F1, F2, P1 (P1's twin threshold is the twin's, not K1-(A)'s) | no | ready |
| G0 clause 1, R_tune with the segment-block bootstrap | no | ready |
| G0 clause 3, cost from T's own wall time | no | ready |
| K1 first clause | **YES** | blocked until (b) and the precheck redefinition are ruled |

**NOT BUILT YET, stated plainly** (each a pre-G0 build step):
- **the reducer:** PyCBC v and newSNR at every stored point, the H-anchored network score, thresholds, R_tune and its
  bootstrap, and the gate evaluations;
- **K2:** the linearity shortcut against really injected strain, 15 injections;
- **N1:** float32 against float64 for a 384-s template through the identical chain.

Their code comes before G0. Their runs are short, except K2's injection preparation.

### C13 result (k1diag, run 21:26, code 80f68ca)

**Reproductions:** both exact. V0 equals k1precheck and V5 equals k1perfect, bit for bit.

| Deviation | Registered | Δf 1/8 → 1/16 → 1/64 | PSD flat above 1,024 Hz | Perfect fed the 1/16 PSD | Verdict (rule) |
|---|---|---|---|---|---|
| D1, L1 963–1,024 Hz | 1.54 | 1.055 → 1.544 → 1.000 | 1.004 | 1.007 | **EXPLAINED: PSD above 1,024 Hz** |
| D2, H1 476–537 Hz | 0.38 | 0.077 → 0.384 → 0.115 | 0.530 | 9.45 | **UNEXPLAINED** |
| D3, H1 963–1,024 Hz | 0.21 | 6.60 → 0.213 → 0.069 | 1.192 | 24.6 | **EXPLAINED: PSD above 1,024 Hz, marginal** |

All values are pycbc/perfect.

**Notes on each.**
- **D1** (L1 band top, no strong line) is clean. Flattening the PSD above 1,024 Hz removes it (1.004), and so does
  Δf = 1/64 Hz (1.000). Δf = 1/8 Hz nearly does (1.055). It appears only at Δf = 1/16 Hz, so it does **not** move
  monotonically with Δf, and the mechanism is not fully understood. The PSD-sampling explanation fails its second leg
  (V4 = 1.007).
- **D2** is unexplained. **D3** passes the rule only barely: |1.192 − 1| = 0.192 against the 0.25·|0.213 − 1| = 0.197
  limit.
- At H1's strong-line bands, the band mean of the Welch re-estimate jumps by orders of magnitude under tiny generator
  changes. Examples: the perfect generator fed the sampled PSD gives 272 and 1,285; Δf = 1/8 Hz gives 344 at
  963–1,024. **That supports (a): the metric is unusable at strong lines,** whatever the generator.

### C13, recorded per the Bridge (2026-11-26)

**D2 (H1, 476–537 Hz) is UNEXPLAINED.** It stays recorded that way in the K1 notes. D3's explanation is marginal, and
D1's cure is not monotone in Δf.

## C15 (2026-11-26, pre-result): P1's ρ_opt

**The ambiguity.** PREREG_coherent2 and the step-T registration say "median ρ̂/ρ_opt ≥ 0.90", but neither defines
ρ_opt.
- **Reading A:** the full-signal optimal SNR (Y0's oracle).
- **Reading B:** the optimal SNR the engine's templates can reach, i.e. the norm of the L_max-truncated true template.

**RULED (the Bridge, pre-result): reading B.**
- P1 measures how well the engine recovers what its templates can see.
- ρ_opt = the norm of the L_max-truncated true template, evaluated with the same reference start frequency as the
  templates (D2), in each detector, combined over H1 and L1 as the network value is.
- **Precedent:** `realbg_cascade` line 132 used the truncated true template's norm, with the full-length oracle as a
  cross-check.

**Kept visible.** ρ̂/ρ_opt,full (reading A) is reported per q stratum alongside, so the truncation loss stays on the
record. That loss belongs to the L_max comparison (the configuration choice and cost), not to P1.

**Made pre-result:** no P1 number exists.

## C16 (2026-11-26, written BEFORE check (i) runs): the K1-(A) generator configuration, with an a-priori reason

**Disclosure first.** C13's numbers, including that V3 cures D1, were seen before this was written. The reasons below
rest on conventions recorded independently of C13:

- **Δf = 1/16 Hz.**
  - This is the PSD estimate's own resolution: 16-s median-Welch segments (`C.PSD_SEG_SEC = 16`), and an inverse-
    spectrum truncation to 16 s.
  - The pipeline PSD carries no information finer than 1/16 Hz.
  - A finer Δf would interpolate a smoothed estimate, and a coarser one would discard resolution the pipeline uses.
  - So this choice is fixed by the estimator, not by C13.
- **PSD above F_HIGH = 1,024 Hz set flat at its 1,024-Hz value.**
  - Since the project began, the band is limited to [50, 1,024] Hz "everywhere". The reason is that "the GWOSC 4k
    anti-alias filter makes the PSD near Nyquist unusable": project conventions in CLAUDE.md, recorded long before
    step T.
  - Every whitening filter in this pipeline (`realbg_cascade.whiten_filter`, and hence the twin and the real series)
    is exactly **zero above F_HIGH**. The pipeline assigns that region no weight and no meaning.
  - Feeding a generator the anti-alias-shaped, unusable PSD above F_HIGH injects a region the convention says is not
    trustworthy.
  - Flat at the band edge is the neutral extension, the same rule already used below 40 Hz (C10).

**The configuration is therefore C13's V3: pycbc, Δf = 1/16 Hz, PSD flat below 40 Hz and above 1,024 Hz.**
- **Check (i) is the BINDING gate on it.** If (i) fails, that is the result.
- **Sensitivity readout, not a gate:** (i) is also run on the original configuration (Δf 1/16 Hz, the PSD above
  1,024 Hz as estimated).

## C17 (2026-11-26, design ACCEPTED, n sized for power before any code or run): check (i), noise-only |z|² against the exact generator

**What is compared.** The per-template mean of noise-only |z|² over all ±1-s lags, which is the quantity K1's
survival comparison depends on.
- **The K1-(A) generator, C16's configuration:** pycbc, Δf 1/16 Hz, PSD flat below 40 Hz and above 1,024 Hz. This
  comparison is the gate.
- **Against the exact generator:** a Gaussian drawn exactly from the same PSD on the full 4,096-s circle.
- Both go through prep's whitening path (×w², the real segment's own filter) and the identical engine.

**Templates.** 21 per L_max: 7 from each of 3 boxes (the bg boxes nearest Mc 0.20, 0.35 and 0.60), evenly spaced in
the box's grid order, at L_max 128 and 384. Both detectors, segment 0.

**Sample.**
- **144** independent 4,096-s series per (generator, detector), seeded. This replaces the first draft's 64; see the
  power calculation below.
- From each, 31 snippets with centres 128 s apart in [392, 4,088] s, so about 4,464 snippets per (generator, detector).
- Inside a series the ±1-s windows are far apart, so the |z| values are nearly independent. The correlation from
  overlapping data is handled by the bootstrap below.

**Decision: on the POOLED ratio only.**
- R = mean over templates, lags and snippets of |z|² for the generator, divided by the same for the exact generator,
  per L_max and per detector, so 4 pooled ratios.
- **Bar:** |R − 1| ≤ 3·SE_pooled in all 4.
- **SE_pooled** comes from a bootstrap over SERIES (B = 2,000), resampling each generator's 64 series independently.
  That keeps the cross-template, cross-lag and in-series correlations, as the Bridge's fix 1 requires.
- Per-template ratios are reported descriptively, never gated: 20 tests at 3σ would give about a 5 % chance of a
  spurious fail.

**Power, the must-fire control (fix 3).**
- **The deviation that matters for K1, derived for the per-candidate max:** χ²₄ trials, N_eff 10⁶, so x ≈ 43.1 at
  survival 10⁻².
- A noise-variance excess δ shifts survival at 10⁻² as follows (computed, scipy):

  | δ | 0.1 % | 0.2 % | 0.3 % | 0.5 % | 1.0 % |
  |---|---|---|---|---|---|
  | Survival factor | ×1.021 | ×1.042 | ×1.063 | ×1.107 | ×1.225 |

- K1's own statistical sd of the survival ratio at 10⁻² is 6.2 % (N_MC = 100,000, C10). **A deviation matters if it
  alone shifts survival by one K1 sd, so δ* = 0.3 %.**
- **Planted fault:** the exact generator with its PSD multiplied by (1 + δ*) over 50–300 Hz, where the templates' σ²
  sits, then whitened with the UNCHANGED filter. It must FAIL the pooled bar against an independent exact draw, in
  all 4 (L_max, detector) cells.
- **Null control:** two independent exact-generator draws must PASS in all 4 cells.

**Order.**
1. The null control and the must-fire control run first. They involve no pycbc.
2. **If the planted fault does not fail in every cell,** n is too small. I raise n (more series), re-register the new
   n, and repeat step 1 before anything else.
3. Only then the gate: C16's configuration against the exact generator.
4. Then the sensitivity readout: pycbc with the original PSD above 1,024 Hz. Reported, not a gate.

**n set from an a-priori SE estimate, before any run.**
- |z|² behaves like χ²₂ (relative sd 1). The autocorrelation of whitened chirp z is about 1/(150 Hz), about 27
  samples, so each 2-s window holds about 300 independent lags: a relative SE of about 5.8 % per template per snippet.
- With about 3–10 effectively independent templates over 3 boxes, that is about 2–3 % per snippet.
- For the ratio of two generators with n snippets each: SE ≈ √2 × 2.5 %/√n. That is ≈ 0.08 % at n ≈ 1,984, so
  3·SE ≈ 0.24 % < δ*·f_w ≈ 0.28 %, where f_w ≈ 0.95 is the fraction of template σ² in 50–300 Hz.
- That margin is thin, which is why the must-fire control decides whether n is enough.

**Cost, an estimate.**
- CPU, the series: 64 × 2 detectors × 5 generators-or-controls, about 2–6 s each, about 35 min in all.
- MPS, the engine: one rfft per snippet (~60 ms, shared by all boxes and both L_max), plus 3 boxes × 2 L_max × about
  15 ms of reference correlations. About 0.15 s per snippet, so 1,984 × 2 detectors × 5 ≈ **50 min**.
- Peak ≤ 2.6 GB, under the guard.

**Power calculation (the Bridge's change, before any code or run).**
- The observed planted shift is ≈ N(δ*·f_w, SE), with δ*·f_w ≈ 0.28 %. A cell fires if the shift exceeds 3·SE, which
  happens with probability Φ(δ*·f_w/SE − 3).
- **Requirement:** per-cell power ≥ 0.987, so the joint power over 4 cells is ≥ 95 %. That needs
  δ*·f_w ≥ (3 + 2.23)·SE, i.e. **SE ≤ 0.054 %.**
- **With the a-priori SE**, which scales as 1/√n_series:

  | Series | SE | Per-cell power | Joint power (4 cells) |
  |---|---|---|---|
  | 64 | 0.080 % | 0.69 | ≈ 0.23 |
  | 144 | 0.053 % | ≥ 0.987 | ≥ 0.95 |

- **SE is re-estimated from the null control's own bootstrap.** If the measured SE is larger than 0.054 %, n is raised
  and re-registered before the must-fire result counts.
- **Cost, re-estimated:** about 80 min CPU for the series, about 2 h MPS for the engine, peak ≤ 2.6 GB.
- **Order:** null and must-fire controls (no pycbc) → the binding gate on C16's configuration → the sensitivity
  readout on the original configuration.

