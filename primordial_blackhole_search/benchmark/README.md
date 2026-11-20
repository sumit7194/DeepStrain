# LongChirp: a benchmark for searches of long subsolar-mass inspirals in real LIGO noise

**In one line:** a fixed set of real LIGO O3a data with known simulated signals hidden in it, a fixed way of
measuring false alarms, and a scoring script — so that a new search method (machine learning or otherwise) can be
compared like-for-like against a matched filter and a CNN on signals that last **minutes**, not seconds.

**Why it exists.** The MLGWSC-1 mock data challenge ([Schäfer et al., arXiv:2209.11146](https://arxiv.org/abs/2209.11146))
is the standard ML-versus-matched-filter comparison, but its signals last at most ~20 s. Subsolar-mass binaries
(component masses below 1 M☉, the primordial-black-hole regime) spend minutes in the LIGO band, which is where
learned searches are least tested. LongChirp uses MLGWSC-1's submission format and scoring conventions on that
regime, on real detector noise.

**Why it matters, and for whom.** It answers one question: *on minutes-long subsolar-mass chirps in real noise, how
far does a new search method reach at a fixed false-alarm rate, compared with a matched filter and a CNN scored the
same way?* It is for groups developing ML or fast matched-filter searches that want a fixed, reproducible
comparison before running on full observing runs.

**What it does NOT do.**
- It is not a search for real signals and claims no detection.
- It does not measure astrophysical rate limits.
- It does not reach the false-alarm rates of a production search: 1/week at best, not 1/year.
- It does not test spinning or eccentric binaries, Virgo, or any noise era other than O3a.
- Its matched-filter yardstick is not the field's best (see caveat 5).

Design, charter and every decision with its reason: [PLAN.md](PLAN.md). All numbers: [RESULTS.md](RESULTS.md).

**Authorship and AI use.** Author: Sumit Pal, independent, non-academic work by a software engineer. Developed with
substantial assistance from Anthropic's Claude AI models (several versions), which wrote most of the code and ran
most of the analyses under the author's direction. The author chose the questions, directed and reviewed the
work, and takes responsibility for its content.

---

## What is in it

| | |
|---|---|
| **Data** | 24 test + 8 dev segments of O3a H1 and L1 strain, 4096 s each, 4 kHz, from GWOSC. Test: 2019-08-05 → 2019-10-01; dev: 2019-06-30 → 2019-08-04. Every segment has the DATA quality bit set throughout in both detectors, and none is within 64 s of a GWTC-2.1 confident or marginal event. GPS list: [manifest.json](manifest.json). |
| **Injections** | [injections.csv](injections.csv): 1,344 non-spinning TaylorF2 binaries (8 realizations × 168), component masses log-uniform on [0.2, 1.0] M☉, isotropic sky and orientation, from 50 Hz. Chirp distance d_c uniform on [5, 500] Mpc (MLGWSC-1 convention: d = d_c (M_c / M_c,BNS)^(5/6)). Merger times ≥ 480 s apart, so no two signals overlap. Added to raw strain by `injections.py` (antenna pattern + inter-detector delay via pycbc, one shared waveform phase, exact sub-sample placement). |
| **Background** | Zero lag + 23 slides: H1 of test segment *i* paired with L1 of test segment (*i*+*k*) mod 24. Pairing across segments avoids the wrap-around discontinuity of circular shifts. 24 × 24 × 4096 s = **27.3 days**. |
| **Scoring** | [evaluate.py](evaluate.py). A trigger is `(time, stat, var)`, MLGWSC-1's format: `time` is the H1 GPS time, `var` the trigger's own time tolerance, 0 < var ≤ 60 s. An injection is **found** if a trigger with stat above the threshold lies within ±var of its H1 merger time (MLGWSC-1's rule). Threshold at a false-alarm rate (FAR) = the loudest background trigger allowed at that rate. Output: found fraction, sensitive chirp-volume V and chirp distance (3V/4π)^(1/3) at FAR 1/day and 1/week, plus a **chance-match bound** 2·var·FAR (the probability that a noise trigger lands within ±var of an injection by luck). `--fixed-var S` re-scores any submission with one tolerance for all triggers. |

**Why a per-trigger tolerance, capped at 60 s.** A long-window detector cannot always place a merger to the second.
A fixed ±1 s rule would then measure timing, not detection. MLGWSC-1 leaves `var` uncapped. Here injections are
480 s apart, so 60 s keeps every match unambiguous, and the chance-match bound is printed so a wide tolerance cannot
hide its cost. Report your result at your own `var` **and** at `--fixed-var 1.0`.

## Results: three yardsticks (2026-11-20)

| yardstick | rule | 1/day | 1/week |
|---|---|---|---|
| **Y0** ideal: optimal network SNR of each injection, exact timing | SNR ≥ 8 / 10 / 12 | 155 / 124 / 104 Mpc | (not a trigger list) |
| **Y1** semi-coherent matched-filter bank (3,235 equal-mass templates, 8 chunks, newSNR) + H1×L1 | var = 1 s (= the tight rule) | **57.5 Mpc** (8.3% found) | **51.7 Mpc** (7.5%) |
| **Y2** CNN (`cnn_w64`, 64-s spectrogram windows, 1-s hop) + H1×L1 sum | own var = 20.2 s | **48.4 Mpc** (7.1%) | **46.2 Mpc** (6.7%) |
| Y2 | fixed ±1 s | 25.5 Mpc (0.8%) | 25.4 Mpc (0.7%) |

Chirp distances; found fractions in brackets. Chance-match bound: Y2 at its own var 5×10⁻⁴ (1/day) and 7×10⁻⁵
(1/week); every ±1 s entry below 3×10⁻⁵.

**Y1 / Y2 distance ratio**, paired bootstrap over injections (2,000 resamples, 90% CI):

| FAR | Y2 at its own var | both at ±1 s |
|---|---|---|
| 1/day | **1.19** [1.10, 1.29] | 2.25 [1.86, 3.10] |
| 1/week | **1.12** [1.05, 1.20] | 2.04 [1.69, 2.78] |

**Reading.** The matched-filter bank beats the CNN by 12–19% in distance even when the CNN is given its own 20-s
timing tolerance. Under a tight timing rule the CNN is not competitive, because it cannot localise a merger to the
second. Both reach only about **one third of the ideal distance**: that is the room a better method can take.
**A good submission should beat Y1.**

### How each yardstick was scored

- **Y0** ([oracle_snr.py](oracle_snr.py)): each injection's optimal network SNR with the true template and its own
  segment's PSD, band [40, 1024] Hz. A ceiling, not a pipeline: it has no background and no threshold, so it is
  quoted at fixed SNR thresholds.
- **Y1** ([y1_bank.py](y1_bank.py)): the bank previously shown adequate in this project (0.05% chirp-mass spacing;
  doubling the density changes the result by ≤3%). For background, every template is run over every whole H1 and
  L1 series and the running maximum is kept in 62.5-ms bins. For foreground, only windows of ±2 s around each
  injection are filtered. Network statistic √(n_H1² + n_L1²); triggers clustered at 1 s; time = template end =
  merger, so var = 1 s. Dev-only golden tests check four things:
  - GPU against a float64 reference: within 3.5×10⁻⁵;
  - windowed against whole-series filtering: within 3×10⁻⁶;
  - template whitening against the repo's reference function: exact;
  - recovery of loud dev injections within 59 ms at ≥ 0.77 of their optimal SNR, on held-out dev segments under a
    rule fixed before the test.
- **Y2** ([y2_cnn.py](y2_cnn.py)): the project's CNN, slid over each segment in 1-s steps and summed across
  detectors at the same time index. Triggers are clustered at 64 s. Trigger time and tolerance come from the
  **dev segments only**: the median offset of its score peak from the merger (5.7 s) and the largest deviation plus
  2 s (20.2 s).

## Caveats (read these before quoting a number)

1. **The bootstrap CI does not include threshold uncertainty.** It resamples injections only. The thresholds come
   from one background realisation. With 27.3 days of background, the 1/day threshold sits on the 28th-loudest
   background trigger and the 1/week threshold on the 4th, so the 1/week numbers rest on very few background
   events.
2. **1/month is not reachable.** 27.3 days of background supports 0.90 events at 12/yr; 1/month needs ≥ 30.4 days.
   Every distinct cross-segment pairing is already used, so going deeper needs more segments or a second slide
   family. Not done.
3. **Y1's foreground is conservative.** The foreground uses only the ±48 chirp-mass neighbours of each injection's
   own template; the background uses all 3,235. The shortcut can only **lower** a foreground statistic against a
   full-bank background. On dev data it was identical on every loud, on-time recovery (6/6) and 0.2–0.7 lower near
   threshold. Y1 is therefore a slight underestimate.
4. **Y2's timing was calibrated on dev data only** (results/y2_calibration.json), from 32 dev injections that beat
   every dev noise peak. A different calibration choice would move Y2's own-var result; the ±1 s column does not
   depend on it.
5. **Y1 is not a fully coherent search.** It uses equal-mass templates and an 8-chunk semi-coherent statistic. A
   fully coherent subsolar bank needs millions of templates (LVK's O4 subsolar bank has 3,452,006;
   [arXiv:2412.10951](https://arxiv.org/abs/2412.10951)). Y1 is the strongest matched filter we could make adequate
   on a laptop GPU, not the field's best.
6. **One noise era, two detectors.** O3a H1 and L1 only. Virgo is not used (in this project's earlier tests it did
   not help at these masses).
7. **Waveform convention.** pycbc's time-domain TaylorF2 returns 5.000 s of samples after coalescence. Injections
   here are placed by their true merger time; if you build templates with pycbc, cut them at t_c.
8. **Memory (practical).** pycbc's PSD estimate adds ~1.6 GB per 4096-s segment; `lowmem_psd.py` is a numpy
   equivalent, identical to 5×10⁻¹⁰. pycbc's waveform generation keeps memory per new waveform length and does not
   return it (standalone reproduction: `pycbc_leak_repro.py`), so build many injections in short-lived processes.
   Exact peaks of our own runs are in RESULTS.md.
9. **Validated range and what is not checked.** The numbers hold for this population (non-spinning TaylorF2,
   0.2–1.0 M☉, d_c 5–500 Mpc), O3a H1×L1, at FAR 1/day and 1/week. They were produced on one machine (macOS arm64,
   Apple GPU). Not checked: results on Linux or other GPUs (expect agreement to float32 precision, not bitwise);
   sensitivity to the choice of the 24 test segments beyond the bootstrap; and any signal model other than the one
   injected.

## How to use it

```bash
# 1. environment (Python 3.12.13; exact package set)
cd primordial_blackhole_search
uv venv --python 3.12.13 .venv && uv pip install --python .venv/bin/python -r requirements-lock.txt

# 2. data: ~8 GB downloaded one file at a time, resumable; ~4 GB kept on disk as float32 .npy
.venv/bin/python benchmark/fetch.py fetch        # do NOT run `discover`: it would rewrite the fixed manifest

# 3. check the evaluator and the injection path
.venv/bin/python benchmark/golden_evaluate.py
.venv/bin/python benchmark/golden_injection.py 1249001472
```

Then produce a submission directory with 32 CSV files, each with columns `time,stat,var`:

- `bg_k{k}.csv` for k = 0 … 23: run your search on H1 of `test[i]` and L1 of `test[(i+k) % 24]` for every i,
  with no injections, and report `time` on the H1 segment's clock;
- `fg_r{r}.csv` for r = 0 … 7: run on `test[i]` H1 and L1 at zero lag with realization *r* of injections.csv added.
  Use `injections.detector_signal(row, ifo)` and `injections.add_signal(...)` so the signals are exactly the
  benchmark's.

You don't need to cluster; the evaluator clusters at 1 s. Then:

```bash
.venv/bin/python benchmark/evaluate.py path/to/submission --out my_result.json
.venv/bin/python benchmark/evaluate.py path/to/submission --fixed-var 1.0 --out my_result_tight.json
```

Rules that keep comparisons fair: tune nothing on the test segments. Use the 8 dev segments, which are disjoint
from the test segments and from the baselines' training data, for any calibration. Report both rules.

Yardstick submissions: `submissions/y2/` is committed. Y1's (47 MB) is regenerated from `results/y1_bins/` with
`y1_bank.py submit` after copying that folder to `state/y1/`.

## Data provenance and integrity

Strain is GWOSC's O3a 4 kHz release (`O3a_4KHZ_R1`), one file per segment: segments are aligned to GWOSC's 4096-s
file boundaries. GWOSC publishes no checksum, so each file was checked three ways.
- At download (`fetch.py`): it must open as HDF5, have the expected GPS start, and have the DATA bit set
  throughout. sha256 of the raw file and of the saved array are recorded in manifest.json.
- After download (`verify_published.py`): the saved arrays reproduce the minimum, maximum, mean and standard
  deviation that GWOSC's API publishes for each file. Worst relative error 5.6×10⁻⁸ (float32 storage); mean
  within 10⁻¹¹ standard deviations. Result in results/verify_published.json.
- Re-checking raw byte sizes was not possible afterwards, because raw files are deleted after conversion.

## Licence, citation, acknowledgement

- **Code and benchmark definition** (scripts, injections.csv, manifest.json): MIT ([../../LICENSE](../../LICENSE)).
- **Strain data** are not redistributed here; they are fetched from GWOSC, whose data are licensed under
  **Creative Commons Attribution 4.0 International** ([gwosc.org/acknowledgement](https://gwosc.org/acknowledgement/)).
- If you use LongChirp, please cite this repository ([../../CITATION.cff](../../CITATION.cff)). Also cite the GWOSC
  O3 data release: R. Abbott et al. (LIGO Scientific Collaboration, Virgo Collaboration and KAGRA Collaboration),
  "Open data from the third observing run of LIGO, Virgo, KAGRA and GEO", ApJS 267, 29 (2023). And include GWOSC's
  acknowledgement, reproduced verbatim as GWOSC requires:

> This research has made use of data or software obtained from the Gravitational Wave Open Science Center
> (gwosc.org), a service of the LIGO Scientific Collaboration, the Virgo Collaboration, and KAGRA. This material is
> based upon work supported by NSF's LIGO Laboratory which is a major facility fully funded by the National Science
> Foundation, as well as the Science and Technology Facilities Council (STFC) of the United Kingdom, the
> Max-Planck-Society (MPS), and the State of Niedersachsen/Germany for support of the construction of Advanced LIGO
> and construction and operation of the GEO600 detector. Additional support for Advanced LIGO was provided by the
> Australian Research Council. Virgo is funded, through the European Gravitational Observatory (EGO), by the French
> Centre National de Recherche Scientifique (CNRS), the Italian Istituto Nazionale di Fisica Nucleare (INFN) and the
> Dutch Nikhef, with contributions by institutions from Belgium, Germany, Greece, Hungary, Ireland, Japan, Monaco,
> Poland, Portugal, Spain. KAGRA is supported by Ministry of Education, Culture, Sports, Science and Technology
> (MEXT), Japan Society for the Promotion of Science (JSPS) in Japan; National Research Foundation (NRF) and
> Ministry of Science and ICT (MSIT) in Korea; Academia Sinica (AS) and National Science and Technology Council
> (NSTC) in Taiwan.

(Text as retrieved from gwosc.org/acknowledgement on 2026-11-20.)

Environment of the published numbers: Mac (macOS 27.2, arm64, MPS) · primordial_blackhole_search/requirements-lock.txt
· Python 3.12.13, numpy 2.4.6, scipy 1.17.1, torch 2.12.0, pycbc 2.10.0, gwpy 4.0.1, lal 7.7.1.
