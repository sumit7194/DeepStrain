# Ringdown Spectroscopy — do black holes really have "no hair"?

*Real-data project: fingerprint newly-merged black holes from the few milliseconds of
"ringing" after merger, and test GR's austere claim that the bell has only two knobs —
mass and spin.*

---

## The physics (one paragraph)

Strike a bell and it rings at its natural tones; the tones fingerprint the bell. A
just-merged black hole is a distorted bell: it sheds its distortion by ringing in
gravitational waves (quasinormal modes, "QNMs"), then settles smooth. The **no-hair
theorem** says every tone's pitch (frequency) and fade-out (damping time) are completely
fixed by just **mass and spin**. So: measure tone #1 → infer (mass, spin). Independently
measure tone #2 → infer (mass, spin). **If they disagree, the object is not Einstein's
black hole.** This is spectroscopy — a chemist fingerprinting an element from its light —
applied to spacetime itself.

## Verified status of the field (checked 2026-06, sources below)

- **GW250114** (Jan 14, 2025): loudest BBH merger ever (network SNR ≈ 80). LVK found
  **≥2 quasinormal modes** (incl. the 221 overtone) with the spectroscopic pattern
  **consistent with Kerr** to tens of percent, plus a pass of Hawking's area theorem.
  ✅ Strain data is **public on GWOSC** (`GW250114_082203-v2`) — our primary test bed.
- **The start-time controversy is real and unresolved in method**: whether GW150914's
  overtone was detected hinges on when "pure ringing" begins (Isi & Farr 2019 vs Cotesta
  et al. 2022 vs the bug-fix rebuttal vs "low evidence when marginalizing over time").
  Fit too early → contaminated by merger; too late → tone sunk in noise. **Robustness to
  this choice is the research opportunity.**
- **SBI for ringdown exists and is now growing** (literature-checked 2026-06-23): NPE-with-coverage
  on GW150914 ([2305.18528](https://arxiv.org/abs/2305.18528)), time-domain SBI
  ([2404.11373](https://arxiv.org/abs/2404.11373)), a CVAE variant ([2506.17618](https://arxiv.org/abs/2506.17618)),
  plus 2025 additions (mode-content [2510.13954](https://arxiv.org/abs/2510.13954), spin-precession
  [2512.05193](https://arxiv.org/abs/2512.05193)). NOT virgin ground — "apply SBI" is taken.
  **Where our angle survives the prior art:** the two relevant ingredients exist *separately* —
  amortized SBI for ringdown is at a **fixed** start time (2305.18528, peak amplitude), and start-time
  *marginalization* exists but in a **classical** frequency-domain analysis ([2312.14118](https://arxiv.org/abs/2312.14118)).
  Joining them — **marginalizing the start time *inside* an amortized SBI model so the no-hair test is
  start-time-agnostic by construction** (`08`/`09`) — is the niche we found unclaimed. (The honest
  contribution is the *construction* + calibrated sensitivity characterization, not a new physics
  result; our δ lands on the classical Kerr-consistent value, and v6 maps why only GW250114 is informative.)
  Other candidate angles, both since explored and PARKED as honest negatives: **(b)** neural tone-count
  model-selection (v4, information-limited); **(c)** hierarchical δ-stacking (v5, starves — only GW250114
  measures δ).

## Ground rules (same discipline as `../echoes/`)

1. **Injections before claims.** Public numerical-relativity waveforms (SXS catalog) with
   known mass/spin, injected into *real* detector noise, are the referee. A method that
   can't recover truth there has no business fitting real events.
2. **Reproduce before innovating.** First reproduce the known GW150914 + GW250114
   results with standard damped-sinusoid fits; only then build the ML.
3. **Negative/confirmatory results are results.** "Sharper confirmation of Kerr" is
   citable science; we do not need a tension to win.

## Pipeline / milestones (v1 COMPLETE — see `notes/lab_notebook.md` for results)

| # | Milestone | Status |
|---|---|---|
| 1 | **Data plumbing** (`01`): fetch + whiten GW150914 & GW250114, find true peaks | ✅ |
| 2 | **Tone tables** (`02`): Kerr QNM (f, τ) via `qnm`; validated vs published (251 Hz / 4.13 ms) | ✅ |
| 3 | **Classical baseline** (`03`): joint-H1+L1 one-tone fit → (M, χ); f right, τ biased — diagnosed by 04 | ✅ |
| 4 | **Injection harness** (`04`): the referee. Found 3 pipeline bugs (loudness mismatch, bandpass τ-bias, restart latching). Final: **unbiased at GW250114 loudness** (M=69.8±6.1 vs 68; χ=0.69±0.13 vs 0.69) | ✅ |
| 5 | **Start-time scan** (`05`): the "poisoned choice" reproduced on both events; GW250114 plateau lands on published (M≈68, χ≈0.69) | ✅ |
| 6 | **Free two-tone fit** (`06`): honest negative — 13/14 injections rail; free LSQ cannot split tones 6 Hz apart. Documented | ✅ |
| 7 | **No-hair test** (`07`): parameterized (LVK-style) — overtone slides by δ; **GW250114: δ=−0.16, calibrated σ(δ)≈0.36 ⇒ consistent with Kerr** | ✅ |
| 8 | **SBI prototype** (`08`): NPE with the **start time marginalized by construction** (the novelty angle); coverage-checked, real-noise-injection-checked, applied to GW250114 | ✅ |

**v2 directions:** sharper δ via Bayesian likelihood; neural model selection for tone
count; hierarchical stacking across events; extend the NPE to (M, χ, δ) to make IT the
no-hair test; SXS waveform injections (full merger, not just analytic tones).

## Environment

- `.venv` — Python 3.12: `gwpy 4.x`, `gwosc`, `qnm` (+ `torch`/`sbi` when we reach #6).
- ⚠️ The `ringdown` package (Isi & Farr) pins Python `>=3.11,<3.12` — install it in a
  dedicated 3.11 venv *later* if we want it for cross-checks (PyPI version is stale and
  pulls a broken `pystan`). `pyRing` lives on git.ligo.org. Neither blocks milestones 1–4.

## Scripts

- `scripts/01_fetch_data.py <EVENT>` — fetch public strain, whiten, plot full chirp +
  zoomed post-peak ringdown window. Writes plots to `plots/`.
- `scripts/02_qnm_predictions.py <M_solar> <chi>` — predicted Kerr tone table
  (220, 221, 330, 440): frequency in Hz and damping time in ms for a remnant of given
  detector-frame mass and spin.

## Sources (verified)

- [LVK, "Black Hole Spectroscopy and Tests of General Relativity with GW250114," arXiv:2509.08099](https://arxiv.org/abs/2509.08099) · [companion: area law + Kerr nature, arXiv:2509.08054](https://arxiv.org/pdf/2509.08054)
- [Isi & Farr's overtone claim and the dispute](https://link.aps.org/doi/10.1103/PhysRevLett.129.111102) · [Cotesta et al. reanalysis, arXiv:2202.02941](https://arxiv.org/abs/2202.02941) · [Isi & Farr comment (bug fix), arXiv:2310.13869](https://arxiv.org/abs/2310.13869) · [time-marginalized "low evidence," arXiv:2312.14118](https://arxiv.org/html/2312.14118v1)
- SBI prior art: [NPE with exact coverage (GW150914 ringdown)](https://www.researchgate.net/publication/371162989) · [time-domain SBI, arXiv:2404.11373](https://arxiv.org/html/2404.11373) · [CVAE spectroscopy, arXiv:2506.17618](https://arxiv.org/html/2506.17618v1)
- Tools/data: [GWOSC open data](https://gwosc.org) · [`qnm` package (Stein)](https://github.com/duetosymmetry/qnm) · [SXS waveform catalog](https://data.black-holes.org) · [`ringdown` package](https://github.com/maxisi/ringdown) · [pyRing](https://git.ligo.org/lscsoft/pyring)
