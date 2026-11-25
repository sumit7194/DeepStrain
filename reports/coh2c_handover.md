# Redesign (c) handover: the local heterodyned refiner

*[model: Fable 5.2] · **UNVERIFIED** · 2026-11-26 · branch `fable/2026-11-26-coh2c` · answers
`reports/coh2_redesign_brief.md` v1.1 · sealed design and predictions: `reports/coh2c_sealed_predictions.md` (09a68e0).
Tuning set only; no fresh data was fetched, opened or needed. `PREREG_coherent2.md`, every `RESULTS.md` and every
`README.md` are untouched.*

## 1. Bottom line

**The cost half of the deadlock is solved on paper and in measurement. The headroom half is NOT tested.**

- **Cost:** stage 2 at one candidate per minute drops from 584–703 h to **about 18–22 h** at full time resolution
  (constructed grids, everything I could count included). The target was ≤ 96 h.
- **How:** nothing statistical changed. Same box, same grid, same per-candidate maximum, same trials. Each template is
  now computed only in a short window around the candidate instead of over the whole 512-s snippet.
- **Exactness:** the new arithmetic agrees with the frozen brute-force definition to **3 parts in 10¹⁰** on real
  tuning strain (44 test cases, 1,320 templates).
- **⚠ NOT TESTED: whether the χ² + v_s statistic keeps ≥ 1.15× Y1 in real noise at one per minute.** That is tuning
  step T. A cheap refiner does not pass G0; it only makes the one-per-minute rate, where the headroom is, affordable.

**In plain terms.** The old code asked each template "where in these 512 seconds do you match?" and then read only
the 2 seconds it cared about. The new code asks one nearby reference template that full question once, keeps a few
seconds of the answer, and derives every other template's answer from that piece with a small correction. It is like
decoding a file once and then applying small diffs, instead of decoding it again for each of 3,000 near-identical
keys.

## 2. The design

1. **Reference correlation.** For a candidate, correlate the double-whitened snippet with a reference template
   θ₀ near the box (one 2²¹-sample inverse FFT, as before).
2. **Window.** Keep ±4 s of that correlation around the candidate, with a 0.5-s taper.
3. **Per template.** Multiply the window's spectrum by e^{iΔΨ(f)}, where ΔΨ is the phase difference between the
   template and the reference, and inverse-FFT (32,768 samples at full rate, not 2,097,152). Read |z|² in ±1 s.
4. **Phase differences** come from lalsimulation's own TaylorF2 coefficients (float64, ~4 µs per template on the
   CPU) and are assembled on the GPU from ten stored basis functions of frequency.
5. **Reference lattice.** A box needs 1–9 references, chosen so that no template's group delay differs from its
   reference's by more than 1.5 s.
6. **χ² at the top peaks:** band sums at the snippet's full frequency resolution, in the same heterodyned form
   (exact; §4.3).

Code: `primordial_blackhole_search/benchmark/coh2c_local.py` (engine, gates, timing, cost),
`coh2c_grid.py` (grids), `footprint.py` (memory reader), `run_coh2c.sh` (launcher for this round).

## 3. Sealed predictions, graded

| # | Sealed prediction | Measured | Grade |
|---|---|---|---|
| P1 | 25 µs per template at M = 8192 (80% interval 10–70 µs) | **14.1 µs** (239× the stored 3.36 ms) | **HIT** |
| P2a | Phase model within 10⁻³ rad of pycbc | 4.7·10⁻¹⁰ rad | **PASS** |
| P2b | Local vs brute force: overlap ≥ 0.9999 median, ≥ 0.999 worst; peak within 10⁻³ / 5·10⁻³ | 1 − overlap ≤ 3.0·10⁻¹⁰; peak error ≤ 2.2·10⁻⁵ | **PASS** |
| P2c | Decimating to 1,024 samples/s costs 1.0% of ρ² on average (0.3–2.5%); ≤ 0.4% at 2,048 | signal templates: **2.4–6.1%** at 1,024 and 0.4–1.4% at 2,048; noise peaks: 0.8–2.4% and 0.1–0.7% | **MISS** (too optimistic). By the sealed rule the graded setting moved to 2,048 samples/s. |
| P3 | Stage-2 total at one per minute ≤ 20 h (80% interval 3–40 h); design target ≤ 96 h | at the sealed-rule setting: **9.1–11.3 h** (covering estimate), **14.4–17.5 h** (constructed grids) | **HIT**; target **PASS** |
| P4 | Peak memory < 2 GB in every run | three runs went over (§6) | **FAIL** |
| P5 | Headroom loss against the frozen design is the time-resolution loss only | zero at full rate (§5, D1) | holds by construction at full rate; **real-noise headroom not measured** |
| P6 | Constructed grid needs 1.0–2.5× the brief's estimate; injection recovery median ≥ 0.90 | **0.67–0.94×** for chirp mass 0.26–0.70; **5–11×** for the thin boxes at 0.18 and 0.86 | grid ratio **MISS on both sides**; recovery half **NOT RUN** |

## 4. Results

### 4.1 Exactness on real tuning strain (gates E1, E2)

- **E1** (`results/coh2c/E1_phase_model.json`): 20 template pairs over the mass range, phase spans up to 1,576 rad;
  worst error 4.7·10⁻¹⁰ rad.
- **E2** (`results/coh2c/E2_local_vs_brute.json`): H1, seven data/box cases:
  - a real stage-1 noise candidate with its own Y1 box (chirp mass 0.390);
  - forced boxes on that noise at chirp mass 0.18, 0.26 and 0.86 (the longest templates and both mass edges);
  - three caught injections: id 676 (chirp mass 0.57), id 471 (0.20, the lowest caught), id 473 (q = 0.25).
- Each case at L_max 128 and 384 s and up to four window settings: 44 runs, 1,320 templates, each compared with a
  pycbc template through the frozen 2²¹-sample definition.
- **Worst 1 − overlap 3.0·10⁻¹⁰; worst peak error 2.2·10⁻⁵.** All 44 pass, including ±2-s windows.

### 4.2 Speed (P1; `results/coh2c/P1_timing.json`)

MPS, warm median, each engine in its own short child, baseline and local engine alternated.

| Setting | Samples per template | Per template | Against the stored 3.36 ms |
|---|---|---|---|
| ±2 s, 1,024 samples/s | 4,096 | 9.2 µs | 365× |
| ±4 s, 1,024 samples/s (sealed primary) | 8,192 | 14.1 µs | 239× |
| ±4 s, 2,048 samples/s (sealed-rule setting) | 16,384 | 36.6 µs | 92× |
| ±2 s, full rate | 16,384 | 34.4 µs | 98× |
| **±4 s, full rate (recommended)** | 32,768 | **59.0 µs** | **57×** |

- **Fresh baseline, same session:** 4.56 / 3.89 / 3.21 / 3.00 ms per template at batch 2 / 4 / 8 / 16 (median of 4
  rounds each). Against batch 16 the sealed primary is 214× and full rate is 51×.
- **Batch 64 (the stored protocol) was not repeated.** It needs several GB of unified memory (§6). The trend with
  batch size suggests it would come in at or below 3.0 ms tonight, so ratios against the stored 3.36 ms may flatter
  the new engine by ~10%.

### 4.3 χ² in the same form (gate E3; `results/coh2c/E3_chi2.json`)

- **Exact route:** band sums over the snippet's own frequency bins. Against brute force, the worst band error is
  3.1·10⁻⁶ and the worst χ² error is 1.2·10⁻⁵ (p = 4, 8, 16; five cases). 9.35 ms for 5 templates at p = 16.
- **Cheap route** (masks on the window's coarse bins): χ² off by 4–8%. **Do not use it.**

### 4.4 Constructed grids (`coh2c_grid.py`; `results/coh2c/G_grids.json`)

Real hexagonal grids for the frozen box, with a 400-point fitting-factor Monte Carlo at full-rate lags.

| L_max | Chirp mass | Templates built | Brief's estimate | Ratio | Worst fitting factor |
|---|---|---|---|---|---|
| 128 | 0.18 | 1,091 | 212 | 5.15 | 0.971 |
| 128 | 0.26 | 4,417 | 5,440 | 0.81 | 0.973 |
| 128 / 384 | 0.35 | 8,100 | 12,067 | 0.67 | 0.972 |
| 128 / 384 | 0.50 | 2,241 | 2,497 | 0.90 | 0.972 |
| 128 / 384 | 0.70 | 429 | 457 | 0.94 | 0.976 |
| 128 / 384 | 0.86 | 151 | 14 | 11.0 | 0.975 |
| 384 | 0.18 | 3,715 (lattice shrunk to 0.7) | 465 | 7.98 | 0.989 |
| 384 | 0.26 | 5,484 | 7,669 | 0.72 | 0.972 |

- **Thin boxes** (0.18, 0.70, 0.86) are narrower than one lattice cell, so every point is an edge point and my
  edge handling over-places them, by about 3×. A one-dimensional chain would do better. They are cheap in absolute
  terms.
- **Candidate-weighted template total** against the estimate: **0.89×** (L_max 128), **1.03×** (384).
- The 0.18 / 384 box fails at the nominal spacing (6% of points below 0.97); it passes at 0.7.

### 4.5 Cost at one per minute (P3; `results/coh2c/P3_cost.json`)

Golden check first: with 3.36 ms and no overheads, the code reproduces the stored 584.1 / 693.8 / 703.3 h exactly.

| Setting | L_max 128 | L_max 384 | Speed-up |
|---|---|---|---|
| Old model (correlation only) | 584 h | 703 h | — |
| Sealed-rule setting, covering estimate | 9.1 h | 11.3 h | 64× / 62× |
| Sealed-rule setting, **constructed grids** | 14.4 h | 17.5 h | 41× / 40× |
| **Full rate, constructed grids (recommended)** | **17.8 h** | **22.3 h** | **33× / 32×** |
| Full rate, covering estimate × 2.5 (stress case) | 28.3 h | 34.5 h | 21× / 20× |

What the full-rate, L_max 384 figure is made of: templates 12.7 h, reference correlations 0.8 h, snippet read and
FFT 0.3 h, exact χ² at 5 peaks 0.4 h, one-off tables 2.1 h, grid building ~6 h (2,718 distinct boxes at ≤ 8 s).

**Counted now, left out before:** reference correlations, snippet handling, χ², coefficient and reference-template
tables, grid construction.

**Still not counted:** v_s; the H1 + L1 combination (a max-filter and a sum per template: my guess is +10–20% of the
template time); the Gaussian twin's noise generation; job start-up per segment.

**Conservative in two places:** the one-off term uses the old 125-µs coefficient time (it is 4 µs now) and pycbc for
reference templates (0.56 s each; an analytic version would be ~20× faster).

## 5. Decisions for DeepStrain (and, where marked, Sumit)

- **D1, time resolution: I recommend full rate (±4 s, 4,096 samples/s).** It reads exactly the lags the frozen
  design reads, so there is no loss and no deviation. It costs ~5 h more than 2,048 samples/s. The sealed 1,024
  setting should be dropped.
- **D2, template start frequency (a deviation from the frozen text).** The frozen rule sets L per template. The
  engine needs every template that shares a reference to start at the same frequency.
  - As tested, a template takes its *reference's* start frequency. Against the frozen rule this changes noise-peak
    |z|² by up to 7% (median 0.3–2%), in L_max 128 boxes at low chirp mass only. It is a change of band edge, not a
    loss, and it applies identically to background and foreground.
  - I suggest one start frequency per *box* (the box centre's), which is simpler to state. Not tested.
  - It matters only where the cap bites: chirp mass below ~0.32 at L_max 128. At L_max 384 nothing in the
    population is capped.
- **D3, the stage-1 rate.** The frozen §5 rule picks one per minute when stage 2 is ≤ 96 h. With these numbers it
  does. §9 and §10 record "one per hour FORCED", so this needs a dated amendment. That is your edit, not mine.
- **D4, thin boxes.** Either accept the over-placed grids (small) or place a one-dimensional chain.
- **Production loop.** The cost model assumes a **box-major** loop: build a box's grid and references once, then
  run every candidate of that box (real, twin, decoys, foreground) before moving on. That needs random access to
  all 48 whitened series (memory-mapped float32, 64 MB each, ~3 GB of disk), which `load_ww` already does per file.

## 6. Memory record (footprint = top's MEM; RSS hides MPS memory)

| Run | Peak | How known | Note |
|---|---|---|---|
| E2 run 1 (two boxes) | 2,332 MB **RSS** | measured (RSS) | **over 2 GB.** True footprint higher, not measured. |
| prep child, two injections | 2,131 MB **RSS** | measured (RSS) | **over 2 GB.** pycbc waveform memory (gwastro/pycbc#5554). |
| prep child, no injection / one injection | 1,173 / 1,435 MB RSS | measured (RSS) | CPU only, so footprint ≈ RSS |
| E2 extended (44 cases) | 821 MB RSS; **~1.9 GB footprint** | **inferred** (RSS + the ~1.1 GB MPS floor) | not measured |
| timing attempt 1 (pid 87972) | **5,837 MB footprint** at 36 MB RSS | measured by the Bridge with top | **critical pressure; stopped by PID.** It was inside the OLD baseline's 64 × 2²¹ batch. |
| timing, child runs | ≤ **1,783 MB** | measured (footprint) | largest = baseline batch 16; local children ≤ 1,307 MB |
| cost model | 346 MB | measured (footprint) | |
| grids | 404 MB | measured (footprint) | CPU only |
| E3 | **2,155 MB** | measured (footprint) | **over 2 GB by 0.16 GB**; launched without a measured prediction (my error) |
| far-|z|² probe | not recorded | — | CPU only, numpy + pycbc, no torch |

- **A single 2²¹-sample FFT on MPS holds ~1.1 GB of driver memory** whatever the batch (baseline batch 2:
  1,333 MB). The local engine's short FFTs show the same floor once a batch passes ~4 million samples.
- **The stored T-cost run** (1.31 GB by RSS) very probably took 5–6 GB when it was measured.
- Result files record the peak just before the environment stamp is taken, so they can read ~20–150 MB low.
- **Expected production footprint:** ~1.3 GB MPS floor plus ~0.4 GB of Python, if pycbc is kept out of the engine
  process. That estimate is not measured.

## 7. What went wrong, and what I corrected

- **Memory, three times** (§6). The cause each time was trusting RSS or launching unmeasured. Fixed with
  `footprint.py`, child processes, and a rule in the timing driver that skips any run predicted over 1,900 MB.
- **P2c decimation loss:** predicted 1%, measured 2.4–6.1% on signals. My bandwidth estimate was too low.
- **Grid run 1** read matches at 1,024 lags per second and reported 15–17% of points below 0.97. That was the test,
  not the grids. Stopped; log kept as `G_run1_coarse_lags_STOPPED.log`.
- **A false lead, closed.** E2 printed a far-from-candidate mean |z|² of 2.6–2.8 for the longest templates, which
  looked like excess noise at L_max 384. `coh2c_farz2_probe.py` shows 2.01–2.04 on valid lags. The excess came from
  lags where the template wraps around the snippet into the glitchy candidate time. **There is no L_max finding
  here.**
- **E2 run 1** covered only two mid/high-mass boxes, so the cap on template length was never exercised. The
  extended run fixed that.

## 8. Not done, and open risks

- **Real-noise headroom ≥ 1.15×: not tested.** The idealised gain at one per minute is 1.89×, so real losses may
  take up to ~39% of it. The plain coherent statistic's tail was catastrophic in the earlier test; χ² and v_s are
  untried here.
- **No end-to-end run.** The engine has not been run over a constructed grid on real candidates. E2 used random box
  templates; the grid check used noise-free matches. The injection-recovery half of P6 was not run.
- **Not built:** v_s, the H1 + L1 combination, the Gaussian twin, the decoys, the driver loop.
- **CPU path** (`COH2C_DEV=cpu`) is code only. I did not run it.
- **L1 was never used.** Every data test is H1.
- **One PSD** (tuning segment 0, H1) shaped the grids, as in `coh2_cost.py`.
- **Literature.** I did not search for prior use of a windowed, decimated ratio filter for follow-up. No novelty
  is claimed. The attribution of "heterodyned likelihood" to arXiv:1007.4820 is from memory.

## 9. Commits on `fable/2026-11-26-coh2c`

| Commit | What |
|---|---|
| 09a68e0 | sealed design and predictions, before any measurement |
| c1f197e | engine; gates E1 and E2 |
| **104c69c** | **the OFFICE session's commit, not part of this round** (shared working tree; the same change is on main) |
| ec5dd31 | timing (P1), cost model (P3), footprint reader |
| a019eba | constructed grids, gate E3, cost with both, the closed false lead |
| (this commit) | this handover |

## 10. Checklist for DeepStrain

Run from `primordial_blackhole_search/`. `benchmark/run_coh2c.sh <name> <script> …` launches detached at normal
priority with the pause-and-alert guard; logs go to `benchmark/state/<name>.log`. The cached whitened series are in
`benchmark/state/coh2c/` (git-ignored, on disk, 216 MB). If they are missing, the gates rebuild them through prep
children, and **a prep with two injections peaked at 2.13 GB RSS**.

| # | Verify | Command | Expect | Memory to declare |
|---|---|---|---|---|
| 1 | The memory reader | `.venv/bin/python benchmark/footprint.py` | `footprint reader PASS`; reader within 1 MB of top at 17 / 317 / 918 MB | < 1 GB |
| 2 | Phase model | `… benchmark/coh2c_local.py e1` | `E1 PASS: worst phase error 4.67e-10 rad`, sign −1; fast coefficients identical to the first implementation (0.0e+00) | ~0.5 GB, CPU |
| 3 | Exactness | `COH2C_DEV=cpu … coh2c_local.py e2` (or on MPS) | 44 lines, all PASS; on MPS worst 1 − overlap 3.0·10⁻¹⁰, worst peak error 2.2·10⁻⁵. **CPU numbers are not known: I never ran that path.** | CPU ~1 GB (estimate); MPS ~1.9 GB (inferred) |
| 4 | χ² | `COH2C_DEV=cpu … coh2c_local.py e3` | fine route PASS in 5 cases, band error ≤ 3.1·10⁻⁶, χ² error ≤ 1.2·10⁻⁵; coarse χ² off by 4–8% | MPS: **2,155 MB measured** |
| 5 | Timing | `… coh2c_local.py time` (MPS, quiet core) | M = 8192: ~14 µs; M = 32768: ~59 µs; baseline 4.6 / 3.9 / 3.2 / 3.0 ms at batch 2 / 4 / 8 / 16; five 2²⁴-sample local runs SKIPPED | largest child 1,783 MB |
| 6 | Cost | `… coh2c_local.py cost` | `golden … PASS [17.5, 20.6, 20.9, 584.1, 693.8, 703.3]`; full rate, constructed: 17.8 h (L_max 128), 22.3 h (384) | 0.35 GB |
| 7 | Grids | `… benchmark/coh2c_grid.py` | the table in §4.4; ~62 s | 0.4 GB, CPU |
| 8 | False lead | `… benchmark/coh2c_farz2_probe.py` | `mean_z2_nowrap` 1.98–2.09 in all 12 rows; `mean_z2_wrap` raised only in `shift0` rows (5.4, 7.0, 3.0), 1.98–2.10 elsewhere | CPU, < 1 GB (estimate) |

**Checks my own tests cannot make, in order of value:**

1. **Independent brute force of a whole box.** Pick one real candidate, take its constructed grid, and compute the
   box maximum with pycbc templates and plain 2²¹ FFTs. Compare with the engine's box maximum. (E2 does this per
   template, with random templates, never the grid.)
2. **L1.** Repeat gate E2 on one L1 segment.
3. **Reference assignment over a real grid.** Assert that every grid template's group-delay difference to its
   assigned reference is ≤ 1.5 s (`assign` returns it).
4. **The seal.** Confirm 09a68e0 predates every result file's environment stamp, and that the grades in §3 match
   the JSON files, not this text.
5. **Then step T**, at one per minute, with whichever of D1–D4 you adopt, recorded as a dated amendment.

Environment: Mac (macOS 27.2, arm64) · `primordial_blackhole_search/requirements-lock.txt` · Python 3.12.13, numpy
2.4.6, torch 2.12.0 (MPS), pycbc 2.10.0, lal 7.7.1 · every result file carries its own stamp · git a019eba.
