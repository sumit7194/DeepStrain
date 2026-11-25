# Verification of the redesign (c) round (DeepStrain, 2026-11-26)

*[model: Claude Opus 5.5], verifying `fable/2026-11-26-coh2c` at 4f01c22 (`reports/coh2c_handover.md`).*
- Run from a separate worktree, never the shared tree. Tuning data only; no fresh data was fetched or opened.
- Fable's result files were re-generated during the checks and then restored, so they stay as committed.
- My outputs: `primordial_blackhole_search/benchmark/results/coh2c_verify/`; script: `benchmark/coh2c_verify.py`.

## Bottom line

**The cost claim survives independent checking, and the engine is exact on whole constructed grids, not only on
single templates.**
- Stage 2 at one per minute is ~18–22 h at full time resolution, against the ≤ 96 h target.
- Real-noise headroom (step T) is still untested, so **G0 is not passed**, exactly as the handover says.
- **Verification status (the Bridge, 2026-11-26): VERIFIED except checklist item 5.** The MPS timing re-run is still
  pending a quiet box. Fable's 59 µs per template at full rate is therefore the round's own measurement, not
  independently reproduced. It must be re-timed on a quiet core before step T is costed for its slot.

## 1. The checklist (§10 of the handover)

| # | Check | Result |
|---|---|---|
| 1 | memory reader | **PASS**: reader equals top at 17 / 318 / 918 MB |
| 2 | phase model E1 | **PASS**, identical: worst 4.67·10⁻¹⁰ rad, sign −1; fast vs first implementation 0.0 |
| 3 | E2 on the **CPU** path (never run by Fable) | **PASS 44/44**: worst 1 − overlap 3.0·10⁻¹⁰, worst peak error 2.0·10⁻⁵ (MPS 2.2·10⁻⁵). The numpy brute-force side is bit-identical to Fable's run (0.0 over 1,320 templates). Footprint 0.81 GB. |
| 4 | E3 χ² on the CPU | **PASS**: fine-route band error ≤ 3.5·10⁻⁶, χ² ≤ 9.8·10⁻⁶. Footprint 0.94 GB (MPS run: 2.1 GB). The CPU takes 22.7 ms for 5 templates at p = 16, where MPS takes 9.3. |
| 5 | MPS timing | **NOT re-run.** The box was contended (load ~22 on 10 cores) and no quiet core was free, so Fable's 59 µs at full rate stands as Fable's measurement. CPU spot check at full rate (M = 32,768, contended): ~225 µs per template on 1 thread, ~380 µs at batch 128. |
| 6 | cost | **PASS**, reproduced exactly: golden 584.1 / 693.8 / 703.3 h; full rate with constructed grids **17.8 h (L_max 128), 22.3 h (384)** |
| 7 | grids | **PASS**, the §4.4 table reproduced identically |
| 8 | far-\|z\|² false lead | **PASS**, all 12 rows reproduced identically; the lead is closed |

## 2. The checks Fable's tests could not make

**V1. Whole-box brute force.** Each box is a constructed grid around a real candidate. Every grid template is evaluated
by the local engine (full rate, ±4 s) and by an independent numpy brute force: a pycbc template, the one-sided
product, and a full 2²¹ inverse FFT. The box maximum is taken over templates and ±1 s of lags.

| Box | Templates | References | Box-max error | Same argmax | Worst per template |
|---|---|---|---|---|---|
| H1 noise candidate, Mc 0.70, L 128 (CPU) | 429 | 1 | 5.2·10⁻⁶ | yes | 1.5·10⁻⁵ |
| the same box on **MPS** | 429 | 1 | 3.6·10⁻⁶ | yes | 1.7·10⁻⁵ |
| H1 noise, Mc 0.50, L 128 | 2,241 | 2 | 1.2·10⁻⁷ | yes | 1.3·10⁻⁵ |
| H1 noise, Mc 0.26, L 128 (600 of 4,417, random) | 600 | 4 | 9.0·10⁻⁷ | yes | 1.0·10⁻⁵ |
| H1 caught injection 676 (Mc 0.57), L 128 | 1,209 | 2 | 6.0·10⁻⁷ | yes | 8.9·10⁻⁶ |
| H1 caught injection 471 (Mc 0.20, lowest caught), L 128 | 2,070 | 3 | 2.9·10⁻⁷ | yes | 4.1·10⁻⁵ |

- **CPU vs MPS on the same box:** box maximum 1.6·10⁻⁶ apart.
- **V2, L1:** a noise box at Mc 0.70 on L1 segment 1249001472: box-max error 2.6·10⁻⁷, same argmax.
- **V3, reference assignment over real grids:** every grid template is within the rule (1.5 s). The worst is
  **1.451 s**, in the Mc 0.20 box, so the margin is thin at the lowest chirp masses.
- **V4, the seal:** every result file's environment stamp is at 09a68e0 or a descendant, so predictions came first. The
  §3 grades match the JSON files (P1 14.05 µs; P3 rows; E2/E3 pass flags).

## 3. Cost-model overhead terms, re-derived

What is counted:
- **Per template:** phase assembly (dcoef · basis), cos/sin, multiply, inverse FFT, and |z|² max over ±1 s. All of
  these are inside the timed `Engine.z` step (`tlocal`).
- **Per candidate and detector:** n_ref reference correlations (2²¹ inverse FFT plus window), one snippet read and
  FFT, and χ² at 5 peaks.
- **Per distinct box, once:** coefficients, reference templates, and grid construction.
- **Passes:** background + Gaussian twin (2×), plus 900 extra looks.

What is missing or approximate:
- **(a) The H1 + L1 combination** (max over adjacent L1 bins, sum): per template, element-wise. I estimate +5–20% of
  the template term, as Fable did.
- **(b) The 900 extra looks** under-count at one per minute: about 600 decoys plus roughly 218 caught + 218 null + 218
  wrong-template + K2. That is about 1,250, so +0.9% of the background term.
- **(c) Conservative in our favour:** stage-1 candidates are clustered at 1 s, not the frozen ≥ 2 s, so there are fewer
  candidates in practice.

**Nothing else scales per template.** Even doubling the full-rate total gives ~45 h, against the 96-h rule.

**The exactness argument** (non-author read):
- z_θ = IFFT[(Y H₀*) e^{iΔΨ}], normalised by the reference's σ, is an identity under two conditions:
  - templates and reference share the amplitude shape (TaylorF2's Newtonian amplitude, so the ratio is a constant
    that cancels in σ) and the same band (hence D2);
  - nothing reaches the ±1-s read-out from outside the ±4-s window. The group-delay spread is bounded by the
    lattice rule (≤ 1.5 s), so the circular convolution cannot wrap.
- The band-edge truncation of the window's tails is the one approximation. It is measured, not derived: 3·10⁻¹⁰.
- **The argument is sound.** V1 now confirms it on whole grids.

## 4. Decisions

### D2, start frequency: recommend ACCEPT, with a dated amendment

- **Proposed amendment text:** "templates that share a reference use the reference's start frequency
  f_start(Mc_ref, L_max)".
- **Measured effects:**
  - **Signals, at the box edge (worst case):** ρ² changes by **±1.4%** (±0.7% in ρ). It is zero at the reference and
    symmetric in sign. Analytic, from the tuning-segment PSD, Mc 0.18–0.32 at L_max 128.
  - **Box maximum on the lowest-mass caught injection** (id 471, 2,063 of 2,070 templates affected): **−0.02%**.
  - **Noise box maximum** (Mc 0.26): −0.9%; per-template median +0.14%, worst 3.3%. Fable's E2 reported up to 7% per
    template.
  - **Template durations:** stay within 0.965–1.036 × L_max (≤ 132.6 s at 128) and fit the snippet.
- **Why it is safe:**
  - every template is still a normalised matched filter, so its Gaussian-noise statistic is still χ²₂ and
    calibration is unchanged;
  - it applies identically to background and foreground, so it cannot create a gain or loss in the cascade ratio
    beyond these sub-percent template-level changes.
- The box-centre variant is untested and need not be adopted.

### Also recommended (amendments, for the Bridge and Sumit)

- **D1, full rate (±4 s, 4,096 samples/s).** Agree: no deviation from the frozen lags.
- **D3, the stage-1 rate.** Under the frozen §5 rule (≤ 96 h ⇒ one per minute), one per minute is now selected. §9
  and §10's "one per hour forced" need a dated amendment citing P3.
- **D4, thin boxes.** Accept the over-placed grids (they are cheap).
- **D5 (new, not in the handover): the box half-width.**
  - The engine and grids use the pooled p99, **2.15%**. The frozen §3 text says "stratified by q" (2.20% for q < 0.5,
    1.90% for q ≥ 0.5).
  - Recommend an amendment to a uniform **2.20%** (the larger stratum). That is conservative, costs ~+2%, and keeps a
    single box shape.
- **Memory and device.**
  - MPS footprint for one box was **1,859 MB**, including brute-force pycbc templates in the same process: at the 2-GB
    cap.
  - The CPU path stays ≤ 0.96 GB, but production on the CPU would likely exceed 96 h. The reference correlations
    (2²¹ FFTs) are ~40× slower there, and templates are ~4× slower.
  - **Production should run on MPS with pycbc out of the engine process**, and its memory declared as footprint.
  - The frozen "< 2 GB" needs to be stated as footprint, with this measured margin.

## 5. What is still open

- **Step T** (real-noise headroom ≥ 1.15× at one per minute) is the next real gate. Its registration is to be drafted
  and sent to the Bridge before anything runs.
- **Not built yet:** v_s, the H1 + L1 combination, the twin, decoys, and the driver loop.
- **MPS timing** should be re-checked on a quiet core before T is costed for its slot.

Environment: Mac (macOS 27.2, arm64) · primordial_blackhole_search/requirements-lock.txt · Python 3.12.13, numpy 2.4.6,
torch 2.12.0, pycbc 2.10.0, lal 7.7.1 · each verification JSON carries its own stamp · verifying 4f01c22.
