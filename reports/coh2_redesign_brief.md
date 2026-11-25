# Redesign brief v1.1: an affordable consistency-tested coherent second look (LongChirp)

*DeepStrain, 2026-11-25; v1.1 2026-11-26 (core-reservation addendum the same day) adds the House rules and Where things live sections, and re-checks every
number against `results/coh2/T_cost.json` and `T1_mc_bias.json` (all unchanged). Self-contained. All numbers are measured on the tuning set (the 24 original LongChirp test
segments) unless marked. Full record: `primordial_blackhole_search/benchmark/PREREG_coherent2.md` (frozen 715a0b3;
§9–10).*

## The problem in one paragraph

- **Stage 1:** Y1 is a 3,235-template semi-coherent bank, using only each template's last 64 s, with newSNR and
  H1×L1 coincidence. Alone it reaches **57.5 Mpc** at a false-alarm rate of 1/day.
- **The cascade:** keep Y1's candidates at a loose rate r₁, then examine only those with a *coherent*, χ²-tested,
  PSD-variation-normalised statistic over a real local template search.
- **Idealised headroom** (Gaussian noise, true template every time) at 1/day:
  - one per minute (39,321 background candidates over 27.3 days): **≈ 109 Mpc, 1.89× Y1**;
  - one per hour (655 candidates): **67.2 Mpc, 1.169×**.
- **The gate:** a go/no-go gate on the tuning set needs **≥ 1.15× with a bootstrap CI lower bound > 1.00**, and
  stage-2 cost **≤ 120 h**. One per minute is also capped at ≤ 96 h by the frozen rule.
- **The deadlock:** one per hour fits the cost but has no headroom; one per minute has the headroom but costs
  584–703 h.

## Measured cost

- **Search box per candidate:** chirp mass ± the 99th percentile of stage-1's chirp-mass bias (**≈ 2%**; the 90th
  percentile is only 0.3–0.7%); mass ratio over the population range (both components 0.2–1.0 M☉); merger time ±1 s.
- **Grid:** coherent (ln Mc, q) at min match 0.97, f_low 50 Hz. Template counts per candidate (L_max = 128 s / 384 s):

  | chirp mass | 0.18 | 0.26 | 0.35 | 0.50 | 0.70 | 0.86 |
  |---|---|---|---|---|---|---|
  | L_max 128 s | 212 | 5,440 | 12,067 | 2,497 | 457 | 14 |
  | L_max 384 s | 465 | 7,669 | 12,067 | 2,497 | 457 | 14 |

  The median over real candidates is 2,654–3,565 at one per minute.
- **Per-template correlation:** **3.36 ms** (Mac MPS, 2²¹-sample complex correlation, warm median).
- **Stage-2 total at one per minute:** **584 h** (L_max 128) to **703 h** (384). That is background plus a Gaussian
  twin, 600 decoys and the foreground; the background alone is 289–348 h.

## What (c) must achieve

1. **Cost:** cut stage-2 cost at one per minute by **≥ 6.1×** (to ≤ 96 h), or ≥ 4.9× against the 120-h G0 cap. Peak
   memory must stay under 2 GB, on one shared Mac.
2. **Keep the headroom:** still reach **≥ 1.15× Y1 on the tuning set** with real losses included. So the redesign may
   give up at most ~40% of the idealised 1.89× gain.
3. **Keep the statistics honest:**
   - every refinement step is applied **identically to background candidates** (the hierarchical-background pitfall:
     Soni, Dhurandhar & Mitra, arXiv:2309.00019);
   - trials are counted by construction;
   - thresholds are measured directly at 1/day;
   - all frozen gates stay: Gaussian-twin Monte Carlo, linearity, binomial null plus wrong-template null, glitch-train
     and drift decoys, and per-q recovery.
4. **No peeking:** design and tuning use only the 24 tuning segments. The 24 fresh validation segments stay unopened
   until the design is frozen.

## Levers already measured in this repo (starting points, not conclusions)

- **Ratio-filter de-chirping** (arXiv:2601.18835): measured here at **2.46–3.45×** faster than direct FFT correlation
  at subsolar kernel lengths (RESULTS.md, 2026-09-21).
- **Box width:** p90 (0.3–0.7%) instead of p99 (~2%) would shrink the chirp-mass axis about 3–6×. Cost: the stage-1
  mis-centred tail gets dropped, and it changes a frozen rule.
- **Coarse→fine:** a coarse coherent grid (e.g. min match 0.8–0.9) or shorter templates first, with the fine grid only
  around the top few coarse peaks. The trials and losses of the coarse step must be measured, not assumed.
- **Mc–q degeneracy:** the box is mostly an elongated ridge, so a grid aligned to the metric's eigen-directions may
  need far fewer points than a rectangular box.
- **Mass dependence:** chirp masses 0.26–0.5 dominate the cost (2,500–12,000 templates). Above 0.7 it is nearly free.
- **Timing:** Y1's loud triggers are within 78 ms of the merger, so the ±1-s window may be narrowable. Check it on the
  tuning-set foreground first.

## What is not known yet

- **The cost model counts correlation time only** (3.36 ms per template per detector). χ² at the top peaks, template
  generation, and the per-candidate snippet FFT are not counted. N_tmpl comes from a hexagonal covering estimate on a
  locally fitted (ln Mc, q) metric, not from a constructed grid. Both are estimates, to be checked before relying on
  them.

- The real-noise tail of the χ² + v_s statistic at candidate times. The plain coherent statistic's tail was
  catastrophic: survival 10⁻³ at ρ² ≈ 1,254, against 33 for a Gaussian twin.
- How much a coarse stage loses on real signals.
- Whether any design keeps ≥ 1.15× after both.

## Where things live (all paths relative to `primordial_blackhole_search/`)

- **Tuning data, the only data you may use:**
  - `benchmark/data/{H1,L1}_<gps>.npy`: float32 strain, 24 tuning (old test) + 8 dev segments, all on disk;
  - `benchmark/injections.csv`: 1,344 injections;
  - `benchmark/results/y1_bins/`: Y1 per-segment bins with the winning template indices, and per-injection
    foreground triggers;
  - `benchmark/state/realbg/caught.json`: one-per-minute stage-1 candidates and caught injections;
  - `benchmark/submissions/y1/`: Y1 trigger lists.
  - `state/` and `submissions/` are git-ignored but are on disk in this working tree.
- **Reusable code:**
  - `benchmark/realbg_cascade.py`: whitening, gating, coherent |z|², linearity shortcut;
  - `benchmark/coh2_cost.py`: metric and timing;
  - `benchmark/y1_bank.py`: stage 1;
  - `pbh/ratiofilter.py`: the ratio filter;
  - `benchmark/lowmem_psd.py`.
- **Launching:** `benchmark/run.sh <name> <script> …` starts a job detached at nice 15 with `benchmark/job_guard.sh`
  (pause plus alert on pressure, auto-resume, never kills). For this round, use nice 12 as below.
- **Interpreter:** `primordial_blackhole_search/.venv/bin/python` (Python 3.12.13;
  `primordial_blackhole_search/requirements-lock.txt`).

## House rules for this round (fleet rules, 2026-11-25/26)

1. **Branch:**
   - Work on `fable/2026-11-26-coh2c`, branched from current `main`.
   - Do **not** edit `benchmark/PREREG_coherent2.md`, any `RESULTS.md`, or any `README.md`.
   - Tag every commit `[model: Fable 5.2]` and mark its contents **UNVERIFIED**.
   - Finish with a handover doc (e.g. `reports/coh2c_handover.md`) that ends with a **checklist for DeepStrain**: what
     to verify, how, and with which numbers.
2. **Compute:**
   - **One core is reserved for you** (another job is paused for the round). Timing measurements are valid on it.
   - Still run one process at a time, at normal priority (not niced), since the core is yours.
   - If you need a second core or a long run, ask the Bridge.
3. **GPU / MPS:**
   - **Timing work needs MPS**, so that it is comparable: the cost baseline (3.36 ms per template, 2²¹-sample complex
     correlation) was measured on MPS, warm median, batches of 64. A redesign must be timed on the same device, the
     same way, and its speed-up quoted against that number.
   - Memory is unified and shared with the fleet (about half free tonight).
   - Declare a peak for any MPS run and keep it **under 2 GB**. A batch of 64 snippets of 2²¹ complex64 values is
     ~1 GB on the device; the baseline measurement peaked at 1.31 GB RSS, and RSS under-reports MPS by ~1.3×.
   - No long MPS job without the Bridge's OK.
4. **No job-ending caps:** no timeout, alarm or step limits. Pause on pressure (`job_guard.sh` does this) and
   checkpoint anything over ~10 min.
5. **The 24 fresh validation segments stay unopened.** They are not on disk and must not be fetched; only the tuning
   set above may be used. **If any design choice needs a number from fresh data, stop and say so.**
6. **Seal predictions first:** any predicted speed-up or headroom is committed, with a grading rule, **before** it is
   measured.
7. **Literature:** quoted at the source, or marked "from memory".
8. **Provenance:** record the interpreter and package versions in every output (`python tools/envstamp.py` at the
   repo root).
