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
  watchdog stopped the run three times (4.38, 4.24 and 4.94 GB group RSS). Cause measured: pycbc's TaylorF2
  waveform generation leaks memory with every distinct-length waveform (30 random-mass waveforms: 0.2 → 2.0 GB;
  numpy and default FFT backends alike; not a cache that plateaus). Fix: windows built in short-lived child
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
