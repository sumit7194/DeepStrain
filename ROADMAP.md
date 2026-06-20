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
- **Status:** ✅ **DONE (2026-06-20, v6).** `scripts/11_upper_limits.py`: per-Δt exclusion
  curve at N=300. **GW150914: exclude amplitude ≥ A90=1.65σ** at predicted Δt (A50 1.33σ);
  GW151226: ≥1.55σ at its canonical Δt=0.0579 s. Smooth across all spacings; stress-tested
  (statistic verified, threshold not glitch-driven). (γ=0.7 fixed; ML scorer would tighten ~1.2×.)

## P1 — Multi-event no-hair δ STACKING
- **What:** combine the no-hair deviation δ across **multiple events**, not just the
  single spine event (GW250114). Single-event σ(δ) ≈ 0.24; stacking is the clean way
  to sharpen it.
- **Why high-leverage:** the one place more data **directly tightens a real GR test**.
  Already noted as a v2 direction; the amortized SBI network is built and calibrated,
  so this is mostly a hierarchical-combination layer on top of proven infra.
- **Where:** `ringdown_spectroscopy/` (no-hair arc, v2/v3 — COMPLETE & calibrated).
  Detail: [ringdown_spectroscopy/notes/lab_notebook.md](ringdown_spectroscopy/notes/lab_notebook.md).
- **Status:** ⚠️ **METHOD ✅ / real payoff ❌ PARKED (2026-06-20, v5 + stress-test).** `12_stacking.py`
  validated the stacking METHOD — σ(δ) tightens as **√N** on informative injections (N=8 → 0.095 vs
  ideal 0.097, unbiased, calibrated), gated. **BUT** the north-star stress-test (`13_more_events.py`)
  showed **only GW250114 actually measures δ**; all 7 fainter public events return ≈ the prior. So the
  v5 "GW250114+GW150914 → 1.3× tighter" was a Gaussian-approx-of-prior **artifact** (corrected) — there
  is effectively ONE informative real event. **Real multi-event sharpening is blocked by the per-event
  SNR information wall** (only SNR~80-class events measure δ). Come-back-later = more very-loud events, or
  an NPE that extracts δ at lower SNR (likely information-limited, like tone-count).
  **v6 (2026-06-20) MAPPED THE WALL:** `14_delta_threshold.py` swept injected ringdown loudness and
  measured σ(δ) vs SNR — δ only becomes informative (σ/prior < 0.90) at **ringdown SNR ≳ 37**, and even
  at the top of the NPE's trained loudness it's just **~13% tighter than the prior**; GW250114 (real,
  σ/prior 0.83) sits right at that edge. So the stacking starvation is now quantitative, not anecdotal:
  every public event lands at-or-below the informative threshold. Seed-robust, gated.

## P2 — Higher-N injection campaigns where claims are UNDERPOWERED
- **What:** re-run the underpowered claims at **N ≈ 300–500** injections. Specifically
  the leg-8b "sensitivity reversal" rested on **N = 25** (confidence intervals overlap,
  so it's not yet real or refuted).
- **Why:** cheap and **decisive** — settles whether the effect is real instead of leaving
  it ambiguous. Low effort, high clarity.
- **Where:** `echoes/`. Detail: [echoes/notes/lab_notebook.md](echoes/notes/lab_notebook.md).
- **Status:** ✅ **DONE (2026-06-20).** (a) Upper limits run at **N=300**. (b) The specific leg-8b
  "sensitivity reversal" SETTLED: re-ran `08 --n-trials 300` → the in-band family differences are REAL
  & physically sensible (f0=320/γ=0.9 genuinely easier, +6–7σ; f0=150/γ=0.5 harder), NOT a pathology —
  the N=30 overlap was just underpower. The one true anomaly (out-of-band control not collapsing) is the
  known whitened-domain artifact (valid v4 raw test = 10%). No pathological reversal survives.

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
  coincidence. Come-back-later = GPU/cloud dense bank. (pbh v2 PARKED; coincidence win +1.37× stands.)
  **Build C DONE (2026-06-20, L4 VM):** the "lower FAR needs more data" item is closed — coincidence is
  FAR-robust (graceful to 1/year; @1/day reproduces the +1.37×; @1/year still beats single-det floor ~1.2×).
  See [primordial_blackhole_search/RESULTS.md](primordial_blackhole_search/RESULTS.md).
- **Ringdown tone-count:** information-limited (see guardrail above).
