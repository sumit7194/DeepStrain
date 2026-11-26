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
  - For n = 218 valid caught injections, P(X > 1) = 0.0383 and P(X > 2) = 0.00051, so x* = 2.
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
- **Status: the required check FAILED** (run notes, 2026-11-26 12:26–12:28).
  - A draw made directly on the snippet grid lacks the 512-s cut's leakage into whitening notches at narrow lines.
    The equivalence argument above is wrong there.
  - A 2 × 512-s draw cut to 512 s reproduces most of the leakage, but still fails the pre-set band bar.
  - The twin's final form waits for the Bridge's ruling.
- **K1 design (ruled):** option (C).
  - (A) is the registered gate: the same grid, independent Gaussian noise, ~5,000 maxima, ±15 % at 10⁻².
  - (B), numpy brute force at 10⁻¹, is an extra engine cross-check only. It runs when cores free up and does not
    block T.
  - The off-grid twin Allen bound is written before (A) runs.
