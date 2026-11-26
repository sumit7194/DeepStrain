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
