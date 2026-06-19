# DeepStrain — Roadmap (forward-looking)

> Complements [CLAUDE.md](CLAUDE.md) (current status per sub-project) and
> [JOURNAL.md](JOURNAL.md) (dated history). This file captures the **next
> high-leverage moves** and the standing guardrails. Each item says *what*,
> *why it's high-leverage*, *which sub-project / where the detail lives*, and
> *status*. Added 2026-06-20 from a cross-project review ("legs" analysis).

All three arcs are currently PARKED with honest results (two wins, one modest,
one honest negative). These items are the cleanest ways to **strengthen what we
already have on the same data**, plus the one guardrail on a parked thread.

---

## P1 — Echo non-detections → real UPPER LIMITS  *(highest leverage)*
- **What:** add an **injection-efficiency curve** to the echo comb search — i.e.
  measure, per echo spacing Δt (≡ λ), the amplitude you *would have detected*.
  That converts the current honest "non-detection" into a quantitative
  **"we exclude λ above X"** exclusion.
- **Why high-leverage:** same data, but a genuinely stronger and more publishable
  result — an exclusion is a constraint, a non-detection is not. Flagged honestly
  in the leg-8 review.
- **Where:** `echoes/`. Detail + the v1/v5 sensitivity machinery to extend:
  [echoes/notes/lab_notebook.md](echoes/notes/lab_notebook.md). (v1 already has a
  sensitivity-curve harness `06`; this generalizes it to a per-Δt efficiency → λ map.)
- **Status:** 🔲 NEW — not started. **Top priority.**

## P1 — Multi-event no-hair δ STACKING
- **What:** combine the no-hair deviation δ across **multiple events**, not just the
  single spine event (GW250114). Single-event σ(δ) ≈ 0.24; stacking is the clean way
  to sharpen it.
- **Why high-leverage:** the one place more data **directly tightens a real GR test**.
  Already noted as a v2 direction; the amortized SBI network is built and calibrated,
  so this is mostly a hierarchical-combination layer on top of proven infra.
- **Where:** `ringdown_spectroscopy/` (no-hair arc, v2/v3 — COMPLETE & calibrated).
  Detail: [ringdown_spectroscopy/notes/lab_notebook.md](ringdown_spectroscopy/notes/lab_notebook.md).
- **Status:** 🔲 NEW — shelved direction, now prioritized.

## P2 — Higher-N injection campaigns where claims are UNDERPOWERED
- **What:** re-run the underpowered claims at **N ≈ 300–500** injections. Specifically
  the leg-8b "sensitivity reversal" rested on **N = 25** (confidence intervals overlap,
  so it's not yet real or refuted).
- **Why:** cheap and **decisive** — settles whether the effect is real instead of leaving
  it ambiguous. Low effort, high clarity.
- **Where:** `echoes/`. Detail: [echoes/notes/lab_notebook.md](echoes/notes/lab_notebook.md).
- **Status:** 🔲 NEW — not started.

## GUARDRAIL — Do NOT throw more ML at the tone-count gap
- **What:** keep the ringdown **v4 tone-count thread PARKED**. Do not iterate more
  classifier architectures on the same data.
- **Why:** the gap is **information-limited, not legibility-limited** — independently
  confirmed (leg 2 and leg 7, and our own six-attempt diagnostic chain ending in a
  calibrated-but-weak AUC ~0.61). The real lever is **more SNR / a coherent multi-
  detector model**, not a fancier net on the same parked data.
- **Where:** `ringdown_spectroscopy/` v4 (PARKED — honest negative). Detail + the full
  six-attempt table: [ringdown_spectroscopy/notes/lab_notebook.md](ringdown_spectroscopy/notes/lab_notebook.md).
- **Status:** 🅿️ PARKED intentionally. Revisit only with more data / a coherent model /
  multi-event stacking / explicit Bayesian model selection.

---

## Known blockers carried forward (context for the above)
- **PBH subsolar:** template-bank density wall — subsolar needs ≤0.1% Mc spacing
  (~1,600+ templates) → intractable locally; blocks a real-MF detector and finer-timing
  coincidence. Come-back-later = GPU/cloud dense bank + lower FAR. (pbh v2 PARKED;
  coincidence win +1.37× stands.) See [primordial_blackhole_search/RESULTS.md](primordial_blackhole_search/RESULTS.md).
- **Ringdown tone-count:** information-limited (see guardrail above).
