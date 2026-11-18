# LongChirp benchmark — charter and plan (item 2)

*Started 2026-11-17 on Sumit's go (relayed by The Bridge). Local only: no GitHub release, DOI or announcement
without Sumit's direct word.*

## Charter
- **Who benefits:** ML gravitational-wave groups. The MLGWSC-1 challenge found ML searches "struggle to identify long
  duration signals" and recommended future challenges with longer signals, naming subsolar black holes
  ([arXiv:2209.11146](https://arxiv.org/abs/2209.11146)). Our prior-art sweep (RESULTS.md, 2026-11-17) found no open
  benchmark for that.
- **Deliverable:** an open, reproducible benchmark for minutes-long subsolar chirps in **real** O3a H1+L1 noise:
  - a GPS manifest plus a fetch script (the strain stays on GWOSC; nothing is redistributed);
  - a deterministic injection generator, with the parameter table committed;
  - an evaluator that turns a participant's trigger list into sensitive distance and volume vs false-alarm rate (FAR);
  - three honest yardsticks, and golden tests for every step.
- **Done when:**
  1. Manifest, fetch, injection and evaluator are committed, and the golden tests pass.
  2. The data is fetched and every array matches its recorded sha256.
  3. Yardstick Y0 is computed. Y1 and Y2 are scored after the macOS 27.2 torch regression passes.
  4. A README lets an outsider reproduce everything from a clean checkout.
- **Time box:** 14 calendar days (to 2026-12-01), with a review at day 7. If GWOSC blocks fetching for more than 3
  days, stop and report rather than waiting.
- **Not in scope:** new detectors or new experiments on our own models. Any result that suggests one goes to Sumit
  first (see the contribution-first rule).

## Benchmark design
**Signals:**
- non-spinning binary inspirals, approximant TaylorF2;
- component masses log-uniform on [0.2, 1.0] M☉;
- isotropic sky position and orientation;
- signal from 50 Hz up, so 24 s (1+1 M☉) to about 6 min (0.2+0.2 M☉) in band.

This is the population of every result in this repo, so the yardsticks are measured, not borrowed.

**Data:** O3a, both H1 and L1 in the GWOSC DATA flag (which passes CAT1 data-quality vetoes). Segments are 4096 s
long at 4096 Hz. Excluded:
- any segment within 64 s of a GWTC-2.1 event (confident or marginal);
- every segment used to train or validate our baseline models (`data/manifest.json`).

Splits:
- **test:** 24 coincident segments, about 27 h, drawn from GPS ≥ 1249000000 (Aug 2019 onward). This keeps them far
  from our models' training data.
- **dev:** 8 coincident segments, drawn from before that, that participants may train or tune on.
- Participants may use any other public data, except the test GPS ranges.

**Storage:** strain is kept as float32 `.npy` (64 MB per detector-segment, about 4.1 GB in total). The download is
about 3 GB, one file at a time. Every file's sha256 is recorded in the manifest.

**Foreground:**
- K = 8 injection realizations of the test set;
- in each segment, mergers at least 480 s apart (longer than the longest signal) and 400 s or more from the segment
  edges, so about 7 per segment, about 170 per realization and about 1,350 in all;
- chirp distance d_c uniform in [5, 500] Mpc, d = d_c (Mc/Mc_BNS)^(5/6), volume weighting as in MLGWSC-1. (The cap
  was first 300 Mpc; measured on real strain, SNR 8 is still reached at d_c = 272 Mpc, so it was raised to 500.)

Injections are projected with antenna patterns and the inter-detector delay (pycbc `Detector`). Both detectors share
the waveform phase (the echoes L4 bug: never draw it per detector).

**Background:**
- the same test noise with no injections, at zero lag;
- plus S = 23 cross-segment slides: H1 of test[i] paired with L1 of test[(i+k) mod 24], k = 1…23. (Built this way
  instead of the planned circular 600-s shifts, which put a wrap-around discontinuity inside every slide.)

Each slide is one more background "observing run" that a pipeline processes like real data. With S = 23 that is
about 26 days (27.3 with zero lag) of background, which reaches 1/day and 1/week but NOT 1/month (that needs >= 30.4 days; 27.3 d supports 0.90 events — corrected 2026-11-18 after the Y2 run; reaching it needs more segments or a second slide family). **Known limit, recorded in this repo:** time slides add
background livetime, but not independent noise samples or a way to verify the independence they assume
(L2, ESTIMATOR AUDIT, ZERO-LAG POPULATION entries in RESULTS.md). The README will say so.

**Submissions:** a trigger list per file: (GPS time, ranking statistic, var), MLGWSC-1's format. An injection is
"found" if a trigger lies within ±var of its merger time in H1 (MLGWSC-1's rule, |t − t_c| ≤ var), with var capped
at 60 s. (Changed 2026-11-18 from a fixed ±1 s: the cnn_w64 probe showed a long-window learned detector localises a
merger only to tens of seconds, so ±1 s would score timing rather than detection. Injections are ≥ 480 s apart, so
60 s stays unambiguous, and the evaluator prints the chance-match bound 2·var·FAR beside every found fraction.)

**Evaluator outputs:**
- FAR(stat) from the background;
- sensitive volume and distance vs FAR on the foreground;
- an efficiency-vs-optimal-network-SNR curve (with SNR50) per mass bin, so results are comparable with this repo's
  past numbers.

## Yardsticks (all three reported, all with their caveats)
| | What | Status |
|---|---|---|
| **Y0** | **Oracle reference:** the optimal network SNR of each injection, i.e. the true template, which no search has. Reported as SNR thresholds 8 and 10. A ceiling, not a pipeline. | numpy only, can be computed now |
| **Y1** | **Adequate semi-coherent bank:** our 3,235-template, n = 8 statistic (bank_dense, 0.05 % Mc spacing; shown adequate 2026-09-23), per detector + H1×L1 coincidence. Not fully coherent: a fully coherent bank at subsolar masses needs millions of templates (LVK's O4 bank: 3,452,006; arXiv:2412.10951). | torch (MPS): **counts only after the 27.2 regression passes** |
| **Y2** | **Learned baseline:** `cnn_w64` on spectrograms + H1×L1 coincidence (sum), as committed. | torch: same gate |

Honest framing:
- On our own past numbers, Y1 beats Y2 by about 6 % in distance.
- Both reach about 40–50 % of the ideal matched-filter distance in single-detector, zero-false-alarm terms.
- A good submission should beat Y1. That is what makes the benchmark worth having.

## Engineering rules (Bridge, 2026-11-17)
- **CPU and memory:** at most 2 cores (OMP/VECLIB threads = 2) and nice 15. A watchdog terminates our own process
  tree if its RSS exceeds about 4 GB, by verified PID only. Never SIGSTOP, never `pkill`.
- **Disk:** stop if free disk falls below 5 GB. Downloads total at most 8 GB. Raw GWOSC files and astropy's download
  cache are purged after each file is converted.
- **GWOSC:** one file at a time, with backoff on errors (10, 30 then 60 min). A file is skipped after 3 failures. The
  whole job pauses after 3 consecutive failed files, and reports. It also yields while the 27.2 regression fetch is
  running, so we never make two concurrent GWOSC transfers.
- **Pause and resume:** every long step checks `benchmark/state/PAUSE` between units. State is saved atomically
  after each unit (at most about 30 min apart). `benchmark/stop.sh` kills only the PID in the PID file, after
  checking its command line.
- **Torch gate:** nothing torch-scored counts as a result until the 27.2 regression passes.
