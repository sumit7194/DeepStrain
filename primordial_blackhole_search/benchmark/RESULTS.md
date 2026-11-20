# LongChirp benchmark: yardstick results

Design and rules are in [PLAN.md](PLAN.md). Data: 24 O3a H1×L1 test segments (4096 s each), 1,344 TaylorF2
injections (component masses 0.2–1.0 M☉, chirp distance d_c uniform on [5, 500] Mpc), background = zero lag + 23
cross-segment slides = 27.3 days. 1/month is NOT reachable with this background (needs 30.4 d). Chirp distance =
(3V/4π)^(1/3) with V the MLGWSC-1-style sensitive chirp-volume. Found = a trigger within ±var of the H1 merger.

## Headline (2026-11-20)

| yardstick | rule | 1/day: chirp distance (found) | 1/week: chirp distance (found) |
|---|---|---|---|
| Y0 oracle (optimal network SNR, exact timing) | SNR ≥ 8 / 10 / 12 | 155 / 124 / 104 Mpc | (not a trigger list) |
| **Y1** adequate semi-coherent bank (3,235 templates) + H1×L1 | own var = 1 s | **57.5 Mpc (8.3%)** | **51.7 Mpc (7.5%)** |
| Y1 | fixed ±1 s | 57.5 Mpc (8.3%) — same rule | 51.7 Mpc (7.5%) |
| **Y2** cnn_w64 + H1×L1 sum | own var = 20.2 s (dev-calibrated) | **48.4 Mpc (7.1%)** | **46.2 Mpc (6.7%)** |
| Y2 | fixed ±1 s | 25.5 Mpc (0.8%) | 25.4 Mpc (0.7%) |

Chance-match bound (a noise trigger within ±var of an injection, 2·var·FAR): Y2 own 5e-4 / 7e-5; Y1 and fixed ±1 s
< 3e-5. Thresholds: Y1 network newSNR 16.78 / 17.87; Y2 summed score 7.88 / 9.77.

**Y1 / Y2 distance ratio** (paired bootstrap over injections, 2,000 resamples, 90% CI; results/y1_vs_y2_bootstrap.json):

| FAR | Y2 at its own var | both at ±1 s |
|---|---|---|
| 1/day | 1.19 [1.10, 1.29] (111 vs 96 found, 93 by both) | 2.25 [1.86, 3.10] |
| 1/week | 1.12 [1.05, 1.20] (101 vs 90, 88 by both) | 2.04 [1.69, 2.78] |

The CI resamples injections only; it does not include the uncertainty of the background threshold (one
background realisation). The bank beats the CNN by ~12–19% in distance even when the CNN is allowed its own
20-s timing; under a tight timing rule the CNN is not competitive because it cannot localise a merger to the second.
Our earlier window-based comparison had the bank ~6% ahead (RESULTS.md 2026-09-23); continuous data, coincidence and
the per-detector trials factor make the gap larger here. Both baselines reach only ~1/3 of the ideal (Y0) distance:
the room a better method could take.

## Idealised two-stage cascade (2026-11-20): ROOM, an upper bound only

Pre-registered in `idealised_cascade.py` (commit c9f323a, before any run); result `results/idealised_cascade.json`.
- **Stage 1 (real):** the Y1/Y2 submissions keep candidates at one per hour or one per minute.
- **Stage 2 (idealised):** each caught injection scores its true-template optimal network SNR. The background pays a
  Gaussian chi-square(4) trials penalty for N_eff = 1e2–1e8 independent trials per candidate.

**Verdict by the committed rule: ROOM.**

| stage 1 | candidates | caught | 1/day (gain vs Y1 57.47) | 1/week (gain vs Y1 51.68) | stage-2 threshold rho* (N_eff 1e2 → 1e8) |
|---|---|---|---|---|---|
| Y1, 1/hour | 655 | 9.7% | 67.2 Mpc (+17%) | 67.2 Mpc (+30%) | 4.5 → 7.1 (1/day) |
| Y1, 1/minute | 39,321 | 16.2% | 109.3 → 108.5 Mpc (+90% → +89%) | 109.3 → 108.5 Mpc (+111% → +110%) | 5.4 → 7.6 |
| Y2, 1/hour | 655 | 8.5% | 58.8 Mpc (+2%) | 58.8 Mpc (+14%) | as Y1 |
| Y2, 1/minute | **degenerate, excluded** | 53.9% | (179 → 136 Mpc) | (166 → 133 Mpc) | |

**How to read it:**
- **Y2 at one per minute is degenerate.** Y2 has fewer than one clustered trigger per minute, so its stage-1 threshold
  is −∞ and selects nothing. With var = 20.2 s, about 49% of injections have a trigger nearby by chance, so its
  "caught" 54% is mostly coincidence. The pre-registration did not anticipate this case. The verdict does not depend
  on it: Y1 alone (±1 s, chance coincidence ≈ 3% at one per minute) meets the ROOM criterion at every N_eff.
- **Where the gain comes from.** Y1's own 1/day threshold (network newSNR 16.8) is set by searching 3,235 templates
  over all 27 days. A second look restricted to a few thousand candidates pays a far smaller trials penalty. The
  injections Y1 catches at one per minute (stage-1 statistic > 10.65) are loud in true SNR. So the cascade's distance
  is set almost entirely by stage-1 recall: it is flat in N_eff and equals the recall ceiling.
- **Why it is only an upper bound.** Three idealisations, each optimistic:
  - the stage-2 background is Gaussian, whereas real noise has heavier tails, which raises rho*;
  - every caught injection is refined to its true template, which is exactly the hard part, given coherent subsolar
    template density;
  - each injection scores its expected optimal SNR, ignoring noise fluctuation.

  ROOM licenses a real test, not a claim.
- **The decisive next measurement.** Run the coherent statistic on REAL background around Y1's one-per-minute
  candidates (local coherent template neighbourhood, real noise), and check whether its tail stays below the caught
  injections' SNRs.

**Why it matters (digest-first).** The question: on minutes-long subsolar chirps, is the matched filter's shortfall
caused by the first stage missing signals, or by the final statistic and its trials penalty? This test says, with the
caveats above, that it is mostly the final statistic. Y1 already ranks about twice as many injections above a
one-per-minute line as pass its own 1/day threshold. A careful coherent re-examination of only those candidates could,
at best, roughly double Y1's reach on this benchmark.

That matters to anyone building ML or fast matched-filter searches for long signals: it points effort at the
second-look step rather than at a better first-stage network. For the benchmark, it means LongChirp's yardsticks leave
real headroom. The test does NOT show that a learned refiner beats a classical coherent follow-up; the prior-art check
says the classical cascade is established (CW hierarchical follow-up; Soni & Nitz 2024 for subsolar). It does not
measure real-noise stage-2 tails, and it is not a detection result.

## Caveats

- **Y1 foreground shortcut.** Foreground uses only the ±48 chirp-mass-neighbour templates of each injection; the
  background uses all 3,235. This can only LOWER a foreground statistic, so Y1 is conservative. Measured on dev
  (results/y1_golden.json, y1_golden_confirm.json): identical on every loud on-time recovery (6/6); +0.2–0.7 lower
  near threshold.
- **Y1 timing.** Templates are cut at t_c: pycbc's time-domain TaylorF2 carries 5.000 s after coalescence, which the
  first Y1 version missed (golden caught it). All 111 Y1 triggers above the 1/day threshold are within 78 ms of the
  merger.
- **Y2 timing** was calibrated on the dev segments only (offset 5.7 s, var 20.2 s, results/y2_calibration.json).
- Equal-mass templates and an n = 8 semi-coherent statistic: Y1 is the adequate bank as previously defined, not a
  fully coherent search (that needs millions of templates at subsolar masses).

## Compute and memory record (honest numbers)

- Y1 background: 24 segments × 3,235 templates, 28.3 h (2026-11-18 20:19 → 11-20 00:35), 68–70 min per segment,
  peak 2.56 GB in-process (2.48 GB by the 5-s watchdog). Declared 3.6 GB.
- Y1 foreground: declared 3.6 GB, **exceeded**: peak 4.06 GB in-process while building injection windows; the
  watchdog stopped the run three times (4.38, 4.24 and 4.94 GB group RSS). Cause measured: memory retained by
  pycbc's TaylorF2 waveform generation, which grows whenever a waveform of a new (larger) length appears and is not
  returned after gc (30 random-mass TD waveforms: 0.2 → 2.0 GB; 30 near-identical ones: flat at 0.6 GB; also seen calling lalsimulation's
  FD TaylorF2 directly, 0.04 → 0.82 GB, while numpy's FFT alone on the same lengths stays flat; standalone repro benchmark/pycbc_leak_repro.py). Not shown to be unbounded — it tracks
  the set of lengths seen — but it reached 2–3 GB in our injection builds. Fix: windows built in short-lived child
  processes of 4 segments each (peak 2.7–3.2 GB), run by a driver that imports nothing heavy
  (y1_fgdrive.py); filtering in its own process. Arithmetic unchanged. Total foreground ≈ 4.5 h.
- Y2: peak 3.59 GB (5-s watchdog) vs 2.27 GB steady in a repro; the same leak is the likely cause (Y2 also generates
  injections in-process).
- Y0: peak 3.53 GB (pycbc PSD estimate +1.6 GB per segment; replaced by lowmem_psd.py for Y1/Y2, identical to 5e-10).

## Artifacts

results/oracle_snr.csv (Y0); submissions/y2/ + results/y2_eval_{own,fixed1}.json (Y2); results/y1_bins/ (Y1's
per-segment binned background maxima and per-injection foreground triggers — the 28-h product; `y1_bank.py submit`
regenerates submissions/y1/ from it after copying it to state/y1/) + results/y1_eval_{own,fixed1}.json (Y1);
results/y1_vs_y2_bootstrap.json.

Environment: Mac (macOS 27.2, arm64, MPS) · primordial_blackhole_search/requirements-lock.txt · Python 3.12.13,
numpy 2.4.6, scipy 1.17.1, torch 2.12.0, pycbc 2.10.0, gwpy 4.0.1, lal 7.7.1 · git (this commit)
