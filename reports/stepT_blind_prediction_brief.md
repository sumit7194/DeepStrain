# Brief: an independent, sealed, blind prediction of step T (for Deepstrain-personal / Fable)

*DeepStrain, 2026-11-26, for the Bridge's Fable window. Short, self-contained. Compute-light: reading and arithmetic
only.*

## 1. The question

**Step T** is the tuning-set test registered in `reports/stepT_registration.md` (amendment T1 to
`primordial_blackhole_search/benchmark/PREREG_coherent2.md`). Before it runs, predict:

1. **R_tune:** the cascade's sensitive chirp distance at 1/day over Y1-alone's, on the tuning set, for the
   configuration T will choose. Give a point value and an **80% interval**.
2. **P(G0 passes):** G0 = R_tune ≥ 1.15 and the segment-block bootstrap CI lower bound > 1.00, plus all gates, plus
   fresh cost ≤ 120 h.
3. **Per-gate pass probabilities:**
   - S0 (already PASS, so give 1.0);
   - K1, K2, N1;
   - F1 nulls, and F1 wrong-template;
   - F2 (i) single sine-Gaussians, (ii) glitch trains, (iii) power drift;
   - P1, as two parts: per-q median ≥ 0.90, and ≥ 90% of ρ_opt ≥ 15 pass the twin threshold;
   - G0 clause 3 (cost).
4. **Which configuration T will choose:** L_max ∈ {128, 384} × p ∈ {4, 8, 16}, as probabilities.
5. **Optional, advisory only:** do the three F2 decoy families represent what made the plain coherent statistic fail
   in test 2? Is there a decoy you would add? This does **not** change the registration. Anything adopted would need a
   separate amendment, approved before T runs.

Your prediction is graded **beside DeepStrain's own sealed prediction**, which you must not see (§3).

## 2. What you may use (all on main; no new measurement is needed)

- **The registered design:** `reports/stepT_registration_REDACTED.md` (the registration with §7, the prediction,
  removed); `PREREG_coherent2.md` (frozen text, §9–§12).
- **The redesign and its verification:**
  - `reports/coh2c_handover.md`, `reports/coh2c_verification.md`;
  - `results/coh2c/*.json`, `results/coh2c_verify/*.json`, `results/coh2t/S0_H1_1249001472.json` (all under
    `primordial_blackhole_search/benchmark/`).
- **Earlier tests on the same tuning set**, in `benchmark/RESULTS.md`:
  - the real-background cascade, NO VERDICT. The gated plain-coherent tail at candidate times: survival 10⁻³ at
    ρ² ≈ 1,254, against 33 for the Gaussian twin; slope −0.0011 vs −0.5; exploratory 7.8 / 6.1 Mpc;
  - the idealised cascade: at one per minute ≈ 109 Mpc, 1.89× Y1; at one per hour 1.169×;
  - the Y1 numbers: 57.5 / 51.7 Mpc; newSNR thresholds 16.78 / 17.87.
- `results/idealised_cascade.json`, `results/realbg_cascade.json`, `benchmark/README.md`.
- Your own knowledge of PyCBC-style χ² + v_s searches. **Label it "from memory"** unless you quote it at the source.

## 3. Exclusions (blinding): do NOT open or read

- **§7 of `reports/stepT_registration.md`**, and **§7 of `reports/stepT_registration_DRAFT.md`**. Use the REDACTED
  copy instead.
- **The 2026-11-25 → 11-26 entry of `JOURNAL.md`** (it quotes the prediction).
- **Commit messages** of `8ce04cf` and `d4c3ca4` (or `git log -p` on `reports/`). Use `git log --oneline` with care.
- **Bridge ↔ DeepStrain messages** about step T.
- **Any output of step T itself.** None exists yet. If one appears while you work, stop and say so.
- **The 24 fresh validation segments.** They are not on disk; do not fetch them.

If you see the prediction by accident, **say so in your sealed file**. The grade is then recorded as non-blind; that
is not a failure.

## 4. Sealing and grading (fixed now)

**Sealing:**
- Write `reports/stepT_blind_prediction_fable.md` on a new branch `fable/2026-11-26-stepT-prediction` from current main,
  in a **separate worktree** (never the shared tree).
- Commit it tagged `[model: Fable 5.2] SEALED`, and give the hash to the Bridge. Nothing in it changes after the
  commit.

**Grading**, applied identically to both predictors once G0 is evaluated:
- **R_tune:** HIT if the measured value lies in the 80% interval; also report |point − measured|.
- **Probabilities (G0 and every gate):** Brier score.
- **Configuration choice:** log score of the probability given to the chosen configuration.
- A gate that is not evaluated (because an earlier failure stops T) is graded "not evaluated" and left out of the
  Brier score.

## 5. House rules

- **Compute:** reading files and arithmetic (numpy on existing JSONs) only. **No MPS. No strain processing.** Seconds
  of CPU. One process. Memory declared as footprint (top MEM), well under 1 GB.
- **Literature:** quoted at the source, or marked "from memory".
- **Interpreter:** `primordial_blackhole_search/.venv/bin/python`. Record versions if you run anything
  (`python tools/envstamp.py`).
- **Start by 05:00; done by 05:30** (the window).
