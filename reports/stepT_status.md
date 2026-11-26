# Step T: where it stands

*Deepstrain-personal [Claude Opus 5.5], 2026-11-27. For Sumit. Technical record: `reports/stepT_clarifications.md`
and `reports/stepT_runnotes.md`.*

**What step T is for.** Our benchmark's best real search (Y1) finds subsolar black-hole mergers out to about 57 Mpc.
A careful "second look" at its most promising candidates might reach further. Step T measures that on the old 24 data
stretches, which have already been seen.
- **Pass mark (G0):** at least 15 % more reach than Y1, with the uncertainty range staying above "no gain"; every
  sanity check passing; and the fresh run fitting in 120 h.
- **Only if G0 passes** do we spend about a week on 24 fresh, unseen stretches. That run is the real result.

**Done and passing.**
- Redesign (c) makes the second look about 30× cheaper. Fable built it; DeepStrain verified it.
- The step-T driver, the reducer, and the K2 and N1 checks are built.
- N1 (numerical precision): **PASS**.
- The goldens on CPU and GPU, all fields: **PASS**.
- A GPU NaN bug in PyTorch 2.12 was found and fixed, and the bad pilot is marked invalid.
- The out-of-population boxes (C19) and the grid-spacing rule (C20) are fixed, and Fable's 12 verified grids are
  reproduced exactly.
- The twin (artificial noise) is exact by construction (C9, bit-for-bit check).
- The noise check (i) is validated: its null control and planted fault both behave.

**Unblocked (2026-11-27).** One of the sanity checks (K1) compares our twin noise with an independently generated
noise.
- pycbc's generator **failed** that comparison: 0.6 % too loud in H1.
- A home-built filter generator (FIR, C21) **passed** it in all four cells, within 0.04 %.
- So K1 can be evaluated, with FIR as the independent generator.

**Running (from 2026-11-27):** the main measurement, staged on Sumit's go. The L128 stage runs first, about
2–2.5 days plus about 1 day for K1. After it, we stop and ask Sumit before L384.

| # | Amendment or clarification (all made before any step-T number) | Status |
|---|---|---|
| C1–C5 | top-5 anchor reading; network formula on the same template; box centre; edge rule; foreground centre | ruled |
| C6–C8 | decoy loudness; decoy parameters; must-fail controls | ruled |
| C9 | twin = cut of a full 4,096-s draw (my first version was wrong; caught by its check) | ruled |
| C10 | K1 independent noise: pycbc, 100,000 samples, 8 independent realisations | ruled; generator FAILED |
| C11 | the χ² mismatch bound (= Allen 2005), with on-grid and must-fail controls | ruled; not yet run |
| C12 | numerical-consistency gate, ≤ 3·10⁻⁴ | ruled |
| C13 | diagnosis of pycbc's deviation | done; partly unexplained |
| C14 | what depends on K1 | recorded |
| C15 | P1's ideal SNR = what the templates can see | ruled |
| C16 | pycbc's configuration, from rules set in advance | ruled; failed check (i) |
| C17 | check (i) design; n raised to 216 by its own rule | ruled; done |
| C18 | K1 depth bar (20 events); K2 per injection | ruled |
| C19 | 15 out-of-population boxes (the Bridge's first rule was wrong at the heavy edge) | ruled; checked |
| C20 | grid spacing follows Fable's verified rule, with a stricter stop | ruled; 12/12 reproduced |
| C21 | FIR noise generator for K1 | ruled; **PASSED** check (i) |
| C22 | K1 run with the FIR generator, seeds independent of every validation draw | ruled; runs in the L128 stage |
