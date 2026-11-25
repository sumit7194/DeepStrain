# Redesign (c), sealed design and predictions: the local heterodyned refiner

*[model: Fable 5.2] · UNVERIFIED · written 2026-11-26 on branch `fable/2026-11-26-coh2c`, in answer to
`reports/coh2_redesign_brief.md` v1.1 (a8c3d44). **Sealed before any measurement**: at this commit I have read the
brief, `PREREG_coherent2.md`, `coh2_cost.py`, `realbg_cascade.py` and the two stored tuning results. Nothing in this
design has been run or timed.*

## 1. The idea in one paragraph

The frozen stage 2 costs 3.36 ms per template because each template is correlated against a 512-s snippet
(a 2²¹-sample inverse FFT), although the search only reads the answer in a **±1-s window** around the candidate. That
is about 250 times more output than is used. The redesign computes the same number a cheaper way:

1. Correlate the data **once** with a nearby *reference* template (one full-length inverse FFT).
2. Keep only a few seconds of that correlation around the candidate, shift it to baseband and decimate it to the
   signal band (50–1024 Hz needs only 1024 complex samples per second, not 4096 real ones).
3. Get every other template in the box from that short piece, by multiplying its spectrum by the **phase difference**
   between the template and the reference and doing a short inverse FFT (thousands of samples, not millions).

Nothing about the statistics changes: same box, same grid, same per-candidate maximum, same trials, same procedure
for background, twin, decoys and foreground. Only the arithmetic changes. So every frozen gate and the frozen
stage-1-rate rule carry over word for word, and no "coarse stage" loss exists to be measured.

## 2. Why it should be exact

- Whitened-domain correlation: z(t; θ) ∝ Σ_f Y(f) H_θ*(f) e^{2πift}, with Y the double-whitened data and
  H_θ = A(f) e^{iΨ_θ(f)} a TaylorF2 template (Newtonian amplitude, so A is the same for every template that starts at
  the same frequency).
- With a reference θ₀: Y H_θ* = (Y H₀*) · e^{−iΔΨ(f)}, ΔΨ = Ψ_θ − Ψ₀. So z(·; θ) = x₀ ∗ k, where x₀ is the reference
  correlation and k is an all-pass kernel whose time spread is the spread of the group delay
  Δτ(f) = −(1/2π) dΔΨ/df over the band.
- For templates near the reference, Δτ is small (my estimate: ≤ 1.5 s for a reference lattice of ~0.25% in chirp
  mass). So z in the ±1-s window depends on x₀ only inside about ±2.5 s. A window of ±4 s holds it with a taper to
  spare.
- x₀ is band-limited to 50–1024 Hz by construction, so shifting it by ~537 Hz and keeping every 4th sample loses
  nothing before the taper.

**Lineage (checked at the source today, abstract level only):**
- Nitz, Kacanja & Soni, "Beyond FINDCHIRP… Ratio-Filter Dechirping", arXiv:2601.18835, Phys. Rev. D 113, 082003
  (2026): a reference template "to remove common orbital phase evolution", turning the FFT into "a cache-efficient,
  compute-bound Finite Impulse Response (FIR) convolution". The identity z = x₀ ∗ k is theirs, and this repo already
  verified it (`pbh/ratiofilter.py`).
- Zackay, Dai & Venumadhav, "Relative Binning…", arXiv:1806.08792 (2018): "precomputing frequency-binned overlaps of
  the best-fit waveform with the data". Same heterodyne idea, for parameter estimation.
- Cornish, "Fast Fisher Matrices and Lazy Likelihoods", arXiv:1007.4820: I cite it *from memory* as the heterodyned
  likelihood. Its abstract does not use that word, so treat the attribution as unverified.
- **What is added here** is only the combination for a *follow-up* search: truncate the reference correlation to the
  candidate's window and decimate to the band. I have not searched for prior use of exactly this; no novelty is
  claimed.

## 3. Declared choices (fixed now)

- **Primary configuration:** window ±4 s, 1024 complex samples per second, M = 8192 samples per template. Alternates
  timed as well: M = 4096 (±2 s) and M = 16384 (±4 s at 2048 Hz).
- **Reference lattice rule:** references are spaced so that the largest |Δτ| over the band, between any box template
  and the reference it uses, is ≤ 1.5 s (computed numerically, not from my estimate above). The lattice may be
  two-dimensional (chirp mass and mass ratio) if one dimension cannot meet the rule.
- **Template start frequency:** a template uses its *reference's* start frequency, max(50 Hz, f(L_max before
  merger)), so that amplitudes cancel exactly. The frozen text sets it per template; across one lattice cell the
  difference is < 0.2% in frequency. **This is a deviation DeepStrain must accept or reject.**
- **Phase differences:** from lalsimulation's own TaylorF2 phasing coefficients, evaluated in float64 on the CPU per
  template (a handful of numbers), then assembled on the GPU from a few stored basis functions of frequency. The PN
  coefficients are not written from memory. If this does not match `pycbc.waveform.get_fd_waveform` (gate E1), the
  fallback is the complex ratio of two pycbc waveforms on the coarse frequency grid.
- **Brute-force yardstick for exactness:** the `coh2_cost.py` definition: `get_fd_waveform` TaylorF2 at
  df = 1/512 Hz, a 2²¹-sample complex inverse FFT, |z|² read at the same lags.
- **Timing protocol:** MPS, warm median (7 repeats, median of the last 5), per template, batched. The stored 3.36 ms
  is the reference number. I also re-time the baseline function in the same session, before and after (a paired
  control, because the box is loaded: load average ~20 on 10 cores when this was written).
- **Data:** tuning set only (the 24 old test segments, their injections, stored Y1 products). No fresh data.
- **Memory:** every run declared < 2 GB peak RSS.

## 4. Predictions and grading rules

| # | Quantity | Prediction | Graded as |
|---|---|---|---|
| **P1** | Per-template time of the local engine at M = 8192 (phase assembly + exponential + multiply + inverse FFT + \|z\|² + max), MPS, warm median | **25 µs** (about 130× faster than 3.36 ms); 80% interval **10–70 µs** | HIT if inside the interval. MISS-fast or MISS-slow otherwise, reported either way. |
| **P2a** | Gate E1: lalsimulation-coefficient phase vs `get_fd_waveform`, largest phase error over 50–1024 Hz for 20 templates across the mass range | ≤ **10⁻³ rad** | PASS/FAIL. On FAIL the fallback is used and said so. |
| **P2b** | Gate E2: local vs brute-force complex series over the ±1-s window at shared lags, ~200 random box templates, on real tuning strain (noise only, and noise plus a caught injection) | normalised overlap ≥ **0.9999** (median) and ≥ **0.999** (worst); peak \|z\|² ratio within **10⁻³** (median) and **5·10⁻³** (worst) | PASS only if all four hold at the primary configuration with the declared lattice rule. My confidence: 80%. |
| **P2c** | Time-resolution loss: mean drop in the per-template maximum of \|z\|² at 1024 samples/s, against 4096 | **1.0%** (interval 0.3–2.5%); at 2048 samples/s ≤ 0.4% | HIT if inside. If the measured mean is > 1%, the primary configuration moves to 2048 samples/s and P3 is graded there. |
| **P3** | **Stage-2 total at one per minute** (39,321 candidates; background + Gaussian twin + 600 decoys + foreground, the brief's 2.02× multiplier), with the stored N_tmpl table unchanged, **plus the costs the old model left out**: reference correlations, window cut and FFT, phase-coefficient generation, χ² at the top 5 peaks | **≤ 20 h** at both L_max 128 and 384 (≥ 29× and ≥ 35× faster than 584 h and 703 h); 80% interval **3–40 h** | **Design target: PASS if ≤ 96 h (≥ 6.1×).** Also stated against 120 h. If > 96 h, the redesign as specified FAILS and the handover says so. |
| **P4** | Peak RSS | < 2 GB in every run; ~1.3–1.6 GB while re-timing the baseline (its own batch dominates), < 1 GB for the local engine alone | PASS/FAIL |
| **P5** | Loss of headroom against the frozen design | Only the P2c time-resolution loss: ≤ 1% in ρ², about ≤ 0.5% in distance | Follows from P2b + P2c. **The real-noise headroom (≥ 1.15× Y1) is NOT measured in this round**: it needs the χ² + v_s statistic, which is DeepStrain's tuning step T. |
| **P6** (pilot, only if the round allows) | Box-maximum plain coherent ρ̂ / ρ_opt for caught tuning injections, with a *constructed* grid, per q stratum | median **≥ 0.90** in both strata (expected 0.95–1.02, since the box maximum adds noise upward) | PASS/FAIL per stratum. Also reports the constructed grid's N_tmpl against the covering estimate; my guess is the constructed count is **1.0–2.5×** the estimate. |

## 5. What would make me wrong

- Small batched FFTs on MPS may run far below the per-sample throughput of one large FFT (P1 slow).
- The phase-difference kernel may spread more than I estimated at low chirp mass with L_max = 384 s, forcing a denser
  reference lattice or a longer window (P2b fails at the primary configuration, or P3 grows).
- The costs the old model left out may dominate once correlation is cheap. P3 counts them for that reason.
- The stored N_tmpl is a covering estimate. A constructed grid on the curved chirp-mass / mass-ratio ridge may need
  more templates (P6).

## 6. What this round does not settle

Cost is only half of the deadlock. Whether the χ² + v_s statistic keeps ≥ 1.15× in real noise at one per minute is
still unknown, and is not answered here. If P3 passes, the frozen rule itself selects one per minute, and the tuning
step T can run as originally planned.

Environment: Mac (macOS 27.2, arm64) · `primordial_blackhole_search/requirements-lock.txt` · Python 3.12.13, numpy
2.4.6, torch 2.12.0 (MPS), pycbc 2.10.0, lal 7.7.1 · git a8c3d44 (branch point).
