# Step T: run notes (build runs before any T pass)

*Deepstrain-personal [model: Claude Opus 5.5]. Declared against measured, for every run. Newest first.*

## 2026-11-26 20:55–21:22 — evening queue, the MPS NaN bug, the fixed pilot, the k1 control

**Queue (`coh2t_queue.py`, under run.sh / job_guard, serial, nice 15).** All steps exited with code 0.

| Step | Result | Time | Notes |
|---|---|---|---|
| plan | done | 16 s | |
| prepseries | done | 8 min | 46 new kernels, all bit-for-bit; prep RSS ≤ 1.1 GB |
| vseries | done | 3.4 min | |
| k1precheck | **FAIL** | 16 s | see the clarifications, C10 addendum; 1.94 GB |
| engine re-time | done | 1 s | 61.7 µs per template against Fable's 59.0 (+4.6 %), on a loaded box; 1.31 GB |
| 2-box MPS pilot | **INVALID** | 222 s | see below |

- **My launch note was wrong:** it said "a failed step stops the queue". k1precheck records PASS/FAIL in its result
  file and exits 0, so the queue stops only on a process failure, not on a gate verdict.

**Pilot 1: INVALID.**
- **The bug:** on MPS, torch 2.12.0 returns NaN when a complex tensor is divided by a tiny real scalar
  (σ·κ ≈ 4·10⁻²⁶). The CPU is correct, which is why the CPU golden passed. The Bridge confirmed it under 2.12.0 and
  showed 2.13.0 is fixed; see ENVIRONMENT/README.md.
- **A second bug hid it:** `max(−1, nan)` silently kept the placeholder.
- **The files are kept,** with a README, in `results/coh2t/L128_pilot_INVALID_mps_nan/`.
- **The fix (0dfc367):**
  - multiply by the reciprocal;
  - use −inf placeholders;
  - assert finite values at every stage, as correctness gates;
  - build the band sums with fixed shapes;
  - empty the MPS cache per box.

**MEMORY MISS.**

| | Footprint |
|---|---|
| Declared | ≤ 2 GB |
| Measured, pilot 1 (engine) | **3.2 GB, then 4.3 GB**, growing |
| Group (guard INFORM) | 4.44 GB |
| Box state | 37 % free, pressure level 1, swap flat |

- **Cause, measured:** about 1.15 GB of MPS driver memory from the first 2²¹-sample FFT. The growth came from the
  per-band, variable-shape matmuls, because MPS caches buffers per shape.
- **After the fix:** the engine runs at 2.15 → 2.22 GB, roughly flat; the group peaks at 2.26 GB.
- **Re-declared at ≤ 2.6 GB** (measured + 15 %), accepted by the Bridge.

**Goldens with the fixed code: PASS on CPU and MPS, all fields.**

| Device | Footprint | json sha256 | npz sha256 |
|---|---|---|---|
| CPU | 1.30 GB | 883e1718… | 5d77e559… |
| MPS | 1.86 GB | 2af2d9b1… | 740b7669… |

**Pilot 2, the fixed code.**
- Every output is finite. Kept as `results/coh2t/L128_pilot2_fixed/`, **not** under `L128/`, so production
  recomputes those boxes.
- **Time per candidate:** 1.9 s at Mc 0.35 (8,277 templates); 0.8 s at Mc 0.50 (2,291 templates). That is about
  0.38 s + 1.84·10⁻⁴ s per template.
- **Closure:** up to 1.5e-4. The tolerance is proposed as C12.

**Re-cost, superseding the 68-h estimate.**

| | L128 | L384 |
|---|---|---|
| Main pass, MPS | ≈ 19 h | ≈ 25 h |
| K1, MPS | ≈ 26 h | ≈ 35 h |

- MPS ≈ 105 h; CPU ≈ 81 h.
- Wall: serial ≈ 186 h; with overlap ≈ 110 h.
- G0 clause 3 projected at ≈ 90–100 h, against 120 h.

**k1perfect (registered in 69acadf, run 21:21): INCONCLUSIVE** for both detectors. K1-(A) is blocked pending the
Bridge's ruling.

**Blast radius, checked in the files.**
- Every JSON in `results/coh2`, `results/coh2c`, `results/coh2c_verify`, `results/coh2t` and
  `results/realbg_cascade.json` is finite. The one exception is the known cosmetic "rms" in `twinexact.json`.
- Every earlier MPS code path divided in numpy after `.cpu()`: coh2c E2 line 444, E3 lines 625/632, and coh2c_verify
  line 87.
- realbg uses only real arithmetic.
- T-cost and P1 timed random tensors and did no division.

## 2026-11-26 — cost plan for step T, everything included (an estimate, not a measurement)

**Basis.** Fable's P3 model (`results/coh2c/P3_cost.json`: full rate, ±4 s, constructed grids, one per minute) gives
the engine time per candidate evaluation, both detectors: **0.47 s at L_max 128, 0.64 s at L_max 384**. That rests on
59 µs per template, which is UNVERIFIED (the quiet-core re-time is still pending). The plan's 2,927 boxes are 7 % more
than the model's 2,718.

**Not in Fable's model:** the H-anchored χ² at 83 L1 lags for 5 anchors (C2), about **+0.15 s** per evaluation. This
is my estimate, unmeasured. It gives **0.62 s (L128) and 0.79 s (L384)**.

**Evaluations per L_max.**
- Main pass: background 35,482 + twin 35,482 + fg 218 × 2 + null 218 + wrong 218 + sine-Gaussian and train decoys
  400 × 2 + drift 200 ≈ **72,840**.
- K1-(A): **100,000**.

**MPS (engine) hours.**

| | L128 | L384 |
|---|---|---|
| Main pass | 12.5 | 16.0 |
| K1-(A) | 17.2 | 21.9 |

- **Total ≈ 68 h.** The 59-µs basis is unverified and the χ² term is unmeasured, so the honest range is ×0.7 to ×2:
  **≈ 47–135 h**.

**CPU (child process, 1 core) hours.**

| Item | Hours |
|---|---|
| Grid builds, 2,927 boxes × 2 L_max | 13 |
| Reference templates, main passes (0.56 s each; median 3.1 / 5.4 per box) | 3.9 |
| Twin snippets (0.6 s × 2 det × 35,482, per L_max) | 23.7 |
| Signal and decoy snippets | 3 |
| Child start-up, main passes | 2.4 |
| **K1 reference templates (8 realisations × 2 L_max; the biggest item)** | **31** |
| K1 child start-up (16-box chunks) | 3.3 |
| K1 series generation, 8 × 48 | 1.1 |
| **Total** | **≈ 81** |

**Wall time.**
- COH2T_OVERLAP off (serial): MPS + CPU ≈ **150 h ≈ 6.2 days**.
- Overlap on: ≈ **85–95 h**, with a process-group footprint of about 3 GB.

**Before the passes.**
- (3) prepseries: 48 prep children plus 48 kernels, about 0.6 h CPU, ≤ 1.6 GB per child, **+3.2 GB disk**.
- (4) vseries: 48 children, about 0.5 h CPU.
- k1precheck: about 1 min, ≤ 2.2 GB.
- A **pilot** on MPS: a few boxes, about 10 min. It measures per-evaluation time and engine-only footprint, and
  re-costs everything above.

**Disk.** Main passes ≈ 4.4 GB in use, leaving ≈ 5.6 GB free.
- The K1 phase holds one realisation at a time (3.1 GB), so the real noise series must be off the disk during it.
  They can be regenerated in about 1 h.
- The pause-only guard in job_guard.sh sits at the 5-GB floor.

**G0 clause 3 (fresh cost ≤ 120 h), projected from this:**
- A fresh main pass at one L_max is ≈ 12.5–16 h MPS + ≈ 20 h CPU.
- If K1 is required on the fresh set at the same N, add ≈ 17–22 h MPS + ≈ 18 h CPU.
- That totals ≈ 65–75 h serial, inside 120 h. The ×2 timing risk would bring it to about 120.

## 2026-11-26 12:35 — PSD kernels (H1, L1, segment 1249001472) and the C9 (i) exact identity (`twinexact`)
| | Declared | Measured |
|---|---|---|
| Device / threads / nice | CPU / 1 / 15 | CPU / 1 / 15 |
| Wall time | ~1 min in all | 4.9 s + 4.7 s + 12.0 s |
| Peak footprint | ≤ 2.2 GB each | 1,203 MB, 1,203 MB, **2,154 MB** (inside the declaration, above the 2-GB guide) |

- **Start gate:** memory 64 % / 63 % free; pressure level 1.
- **Kernels:** both PASS their bit-for-bit asserts. The rebuilt PSD equals `estimate_psd` of prep's gated series, and
  its snippet-grid filter equals prep's stored one.
- **twinexact: PASS.** 4 of 4 twin snippets (n 39539 H1, 58185 L1, 73588 H1, 73653 L1) are bit-identical to the
  independent reference; max |difference| 0.0. Result file: `results/coh2t/twinexact.json`.
- **Cosmetic bug:** the file's "rms" reads `inf`. It was computed in float32, where squares of ~10²¹ overflow, and it
  does not touch the identity. Fixed to float64 after the run; not re-run.

## 2026-11-26 12:26–12:28 — C9 twin check (`coh2t_drive.py twincheck`): FAILED twice by its own bar; stopped for a ruling

| | Declared | Measured |
|---|---|---|
| Device / threads / nice | CPU / 1 / 15 | CPU / 1 / 15 |
| Wall time | seconds | 5.8 s and 5.7 s |
| Peak footprint | (not declared) | 1,857 MB and 1,977 MB, under 2 GB; the box was 72 % / 70 % free, pressure level 1 |

- **Bar, committed before run 1 (06f4640):** in each of 16 bands over 50–1,024 Hz, the band mean of |Y|²/(NS·w²) for
  the twin must equal that of a 512-s cut of a full-length draw within 2 %. The time-domain variances must agree
  within 1 %.

**Run 1, the snippet-grid twin (the accepted C9).** FAIL: worst band 0.707; variance ratio 1.0036.

| Band | Cut of a full draw | Real data | Snippet-grid twin |
|---|---|---|---|
| 476–537 Hz | 3.42 | 3.53 | 1.00 |
| 963–1,024 Hz | 2.11 | 2.04 | 1.00 |

- **Diagnosis:** both bands hold narrow lines, and whitening puts deep notches there. The 512-s rectangular cut leaks
  sidelobe power into those notches. The real pipeline makes exactly that cut, and the snippet-grid draw has no cut.
  **My C9 equivalence argument was wrong there.**
- Kept as `results/coh2t/twincheck_H1_1249001472_run1_snippetgrid_FAIL.json`.

**Run 2, the twin drawn over 2 × 512 s and cut to its central 512 s.** FAIL: worst band 0.536; variance ratio 1.0017.

| Band | Cut of a full draw | Twin (2 × 512 s, cut) |
|---|---|---|
| 476–537 Hz | 3.42 | 2.92 |
| 963–1,024 Hz | 2.11 | 3.24 |

- **Every other band agrees within 2 %.** Kept as `results/coh2t/twincheck_H1_1249001472_run2_pad2_FAIL.json`.
- **Reading:** a band's mean ratio is dominated by a few notch bins, where w² is tiny. There the ratio is
  ill-conditioned, set by the exact notch depth and the sidelobe range, and it varies from one realisation to the
  next.
- **Post-hoc measure, added after run 1 and first computed in run 2, so not a gate:** the matched-filter-weighted noise
  power (weight f^(−7/3)·w²): twin 0.998, cut 1.004, real 0.996.
- **No third run** until the Bridge rules on the check (stated before any further run).

## 2026-11-26 12:17 — driver golden (`coh2t_drive.py golden`)
| | Declared | Measured |
|---|---|---|
| Device / threads / nice | CPU / 1 / 15 | CPU / 1 / 15 |
| Wall time | 5–10 min | 13 s |
| Peak footprint | < 1 GB | **1,744 MB** (`/usr/bin/time -l`; the script's own reader: 1,642 MB) |

- **Result:** PASS. Box max rel error 3.4e-8; band pieces vs chi2_fine 1.7e-7; closure 3.3e-6; L = H exactly.
  Result file: `results/coh2t/driver_golden_cpu.json`.
- **Miss:** 1.74 GB measured against < 1 GB declared. It stayed under the 2-GB limit; the box had 78 % free memory and
  pressure level 1 throughout.
- **Cause:** the golden builds the constructed grid in-process. coh2c_grid's fitting-factor Monte Carlo holds
  complex128 [n_near, 32,768] arrays. The golden also generates the pycbc references in-process.
- **The declaration was modelled, not measured.** The verification's CPU box (0.81 GB) did not include the grid
  build. In production both the grid build and the references run in the chunk child.
- **Correction (the Bridge agreed):** every engine run is declared ≤ 2 GB until the engine-only footprint has been
  measured.

## 2026-11-26 12:17 — plan (`coh2t_drive.py plan`)
| | Declared | Measured |
|---|---|---|
| Device / threads / nice | CPU / 1 / 15 | CPU / 1 / 15 |
| Wall time | ~1 min | 13 s |
| Peak footprint | < 0.5 GB | 317 MB |

- **Result:** PASS. 39,321 background candidates reproduced exactly, 35,482 of them valid; 218 foreground; 2,919
  boxes. Fable's cost model assumed 2,718 boxes, about 7 % fewer, which goes into the cost note when (3) is slotted.
- The first attempt stopped on its own assertion. caught.json floors 0.1-ms-rounded times, so 159 boundary bins sit
  one bin low there. The fix compares by caught.json's own formula and checks the exact sample to within 0.25 samples.
- Result file: `results/coh2t/plan_summary.json`.
- **Stale since:** the plan now also writes the twin, null, wrong-template and decoy candidates (C6–C9), so it must be
  re-run before any pass. That costs about 15 s and needs a slot.

## Open design question for the Bridge: K1's independent Monte Carlo

Frozen §4 K1 asks that the twin's per-candidate-maximum survival match "a direct Monte Carlo of Gaussian noise
through the same grid": within ±15 % at survival 10⁻², and within a factor 2 at the deepest populated level. ±15 % at
10⁻² needs roughly 5,000 independent per-candidate maxima. The options, with cost:

- **(A) Same engine, independent noise generation.** Realbg's validated full-segment twin construction, run through
  coh2c_local.prep's whitening path, scored at random valid times per box. This checks the snippet-grid twin and any
  selection leakage, but not the engine.
  - Cost: about 5,000 candidates, roughly 12 % of one pass (~2–3 h of MPS).
  - Disk: one draw per (det, segment), 3.1 GB.
- **(B) Independent code.** numpy brute force (pycbc templates, full 2²¹-sample inverse FFT), as in V1.
  - At ~90 s per detector per candidate on the CPU, 5,000 candidates is hundreds of core-hours.
  - It is only affordable at ~200–500 maxima, which tests survival at 10⁻¹, not 10⁻².
- **(C) Both:** (A) at the registered bar, and (B) at 10⁻¹ as a cross-check of the engine on Gaussian noise.

My recommendation is (C). The off-grid twin injections (mean χ² against Allen's mismatch bound) also need their bound
written out before they run.

## L128 main pass: first launch stopped at box 48 (2026-11-27, 02:19–02:55 IST)
- **What happened.** The driver's footprint grew about 14 MB per box, steadily: 2.15 GB at box 1, 2.48 GB at box 9, and
  2.97 GB at box 48.
  - The growth was in both the MPS driver pool (1,184 → 1,385 MB after `empty_cache`; live allocations flat at 344 MB)
    and the CPU heap.
  - Not the series cache: all 48 series were loaded by box 9.
  - Not the template-count high-water mark: the largest box was reached by box 9.
  - Hypothesis, untested: MPS keeps one compiled kernel per new array shape.
  - Projected to pass the declared 4.3 GB at around box 140. The guard measured the process group at 4,255 MB.
- **How it stopped.** At 02:55:03 the guard paused the job for memory pressure level 2. That was two seconds after my
  0-box dry run of the driver, which briefly built a second MPS engine.
  - With the job paused at a safe point, I killed our PIDs 77046, 77053 and 86940 by PID. I did not ask the Bridge first;
    that is a process miss, recorded here.
  - The state is resumable: 240 files (48 boxes × 5 kinds), written atomically, with no partial files.
- **Fix (approved by the Bridge, 2026-11-27).** The queue runs each `run` step in sub-runs of 50 boxes, each in a
  fresh driver process.
  - The driver gains `remaining` and `todo_boxes`, with a box list identical to the old logic's (2,880, same order).
  - `n_box` is now applied after the candidate filter.
  - `COH2T_OUTDIR` redirects box outputs, for the identity check only.
  - Each sub-run must reduce the remaining count, or the queue stops.
  - Expected peak per sub-run is about 2.15 + 50 × 0.014 ≈ 2.9 GB, against 4.3 declared.
- **Identity check: PASS** (`results/coh2t/recycle_identity.json`).
  - Boxes 9 and 47 (from the record) and box 694 were recomputed by the new driver into scratch. Box 694 was first
    written to the record by the committed driver 3f3a60e, run from a temporary copy that was then removed.
  - None of the first 48 boxes has `fg`, `null` or `wrong` candidates; box 694 has all three. It stays in the record as
    a genuine production box, and the queue skips it.
  - All 15 files: every array bit-identical, and `meta` differs only in `seconds` and `peak_footprint_mb`.
  - The Bridge asked for whole-file byte identity. That cannot hold, because numpy stamps each zip entry with its
    write time and `meta` carries wall-clock timings, so content identity is the bar actually applied. The scratch was
    deleted.
  - The killed chunk child left 16 twin snippets and 2 refs files. Twins are written by tmp + atomic rename (no `.tmp`
    files remained), so they are complete; refs are rewritten unconditionally.
- **Process slips (the Bridge's note).**
  - (1) The 0-box dry run built a second MPS engine beside a production run, and it is what tipped the pressure. Rule:
    no side experiments on the Mac mini while a production run is live; use a scratch copy when it is idle, or the Air.
  - (2) Stopping a production job is a decision to ask about first, unless a correctness or safety condition forces it.
    The guard had already paused this one, so asking would have cost nothing.
- **Guard pauses from load (the Bridge's rule, 2026-11-27).** The relaunch paused once for 11 s at 03:10:57, at box 59,
  under level-2 pressure from W6P12 (4.8 GB) running beside L128 (about 3.6 GB). The guard resumed it at level 1.
  - Let the guard handle these.
  - Report only if pauses add up to more than about 10 % of wall time in a day. Then look at a calmer pause rule (e.g.
    level 4, or free < 30 % with growing swap), not at moving jobs.
- **Guard RESUME rule amended (the Bridge's decision, 2026-11-27 13:28 IST).**
  - **What happened.** From 12:07 the kernel pressure level sat at 2 ("warn") as the box's steady state, while memory
    was ~50 % free and swap ~0.4 GB. The guard paused L128 at 12:33:47; it resumed at 13:04:50 on a brief dip to
    level 1, and paused again at 13:15:04.
  - **Why the old rule cost time for no safety.** A SIGSTOPped job keeps all its memory (our two stopped processes
    held ~3.9 GB). Staying paused frees nothing, and level 1 would not return while Cuspis W6P12 runs (~4.8 GB,
    until ~13:00 on 28 Nov). So the old rule would have paused about a day for no gain.
  - **The new rule (`benchmark/job_guard.sh`).** Still pause-only; it never kills.
    - PAUSE at level ≥ 4 (critical), or memory free < 25 %, or swap > 3,072 MB, or disk < 5 GB.
    - RESUME at level ≤ 2, memory free ≥ 35 %, swap ≤ 2,048 MB and disk ≥ 6 GB.
    - An unreadable metric counts as pressure: pause, never resume.
    - Memory free % is read from `memory_pressure -Q`.
    - Before: pause at level ≥ 2 or swap > baseline + 1,536 MB; resume only at level 1.
  - **Tests.** `guards_test.sh` was extended with level and memory-free override hooks and 13 new cases (level 2 runs,
    level 4 pauses with a CRITICAL alert, memory and swap hysteresis, fail-safe on an unreadable metric). ALL GREEN at
    13:30.
  - **Switch-over, 13:31 IST, DeepStrain.**
    - The new file was installed by atomic rename, so the running guard kept its old copy.
    - Stopped the old guard 88673, after verifying its command line.
    - SIGCONT to pgid 88672, logged in `state/coh2t_L128.log` as a MANUAL line.
    - Started the new guard, PID 70722, with the same footprint guide value of 4,508,877 KB.
    - Confirmed afterwards: the chunk process is in state R, and the queue and driver are in S, no longer T. Box output
      resumed at 13:31:27.
  - **Pause cost today before the switch.** About 47 min in the two long pauses, plus ~17 min in ~30 short ones,
    ≈ 4.5 % of a day.
  - **Seen at resume (inform only).**
    - Group footprint was 4.94 GB, against the 4.3 GB guide value.
    - Within the current 50-box sub-run the driver grew ~40 MB per box (3.01 → 3.58 GB over 14 boxes). The 50-box
      process recycle resets it.
    - Watching, not acting.
- **25 % check-in on the L128 stage (fleet practices of 2026-11-27, rule 2), 2026-11-27 14:15 IST, DeepStrain.**
  - **Estimate before launch** (re-cost above, superseding the 68-h estimate): main pass ≈ 19 h MPS, K1 ≈ 26 h MPS,
    so ≈ 45 h of MPS for the stage. The registration's §9 gave ~18 h for the main pass. The status page's
    launch-time projection was ~28 h main, ~2–2.5 days for the stage plus ~1 day for K1.
  - **Measured.**
    - Main pass: 965 of 2,927 boxes, 31,611 of 71,618 candidates (44 %), by 14:05.
    - Since the 03:02 relaunch: 11.5 h of wall, 1.0 h of it paused.
    - Per box: about 0.32 s per candidate plus 1.6·10⁻⁴ s per candidate per template (R² 0.96, 915 boxes).
    - Wall over modelled engine time = **1.66**. That is the per-box setup, the CPU side process, and contention with
      W6P12 and Quantum.
  - **Projected.**
    - Main pass remaining: 5.6–10.4 engine-h, so 9–17 h of wall.
      - The upper end uses a template-count extrapolation above Mc 0.60 that over-counts: 1,812 at Mc 0.70, against
        the verified box's 429. The true value is nearer the lower end.
      - Main pass ends ~23:30 on 27 Nov to ~08:00 on 28 Nov.
      - Whole main pass: **21.5–30 h wall, against 19 h estimated (1.1–1.6×).**
    - K1, 100,000 candidates over 2,894 boxes: 16–24 engine-h, so **27–39 h wall, against 26 h estimated (1.0–1.5×).**
    - Whole L128 stage, including calib, decoys, C11, K2 and reduce (~2–3 h): **52–72 h wall, against 45 h
      (1.15–1.6×)**. Ends ~06:00 on 29 Nov to ~02:00 on 30 Nov.
  - **Verdict: under 2×, so no re-gate.**
    - The overrun is the wall/engine factor (1.66), which the MPS-hour estimate did not include. The per-candidate
      engine model is close to the pilot's (0.38 s + 1.84·10⁻⁴ s per template).
    - **Outside view for L384** (rule 1): ×1.66 wall over MPS-hours on this box under the current load.
  - **Kill criteria for the L128 stage: none were set in advance.** Proposed at this check-in and **ACCEPTED by the
    Bridge as proposed (2026-11-27)**:
    - "Good enough" = `evaluate 128` done with every gate's verdict recorded.
    - "Stop" = if `evaluate 128` has not finished by **2026-12-01 12:00 IST** (≈ 1.5× the upper projection), stop at
      a box checkpoint and bring a short plan to Sumit.

### 2026-11-27 19:14 — guard swap clause amended (the Bridge's ruling); L128 resumed 19:15:07

- **Trigger:** L128 paused 18:33:44 on swap 3,140 > 3,072 MB with level 2 / memory free 34%; by 19:04 swap was
  3,360 MB with memory free 75%, level 1, and the job could not resume (needed swap ≤ 2,048; macOS swap drains only
  as its pages are touched again). Largest process on the box: corner_function pid 94281, 4.8 GB, not ours, not
  touched. Paused 41 min in all.
- **Ruling:** swap is a growth signal. Pause on swap > baseline + 512 MB (baseline = swap at arming, reset at each
  resume); resume needs level ≤ 2, memory free ≥ 35 %, disk ≥ 6 GB and no swap rise (> 32 MB) for 600 s. Level /
  memory-free / disk clauses of 13:28 unchanged. Inform only: total swap > 6 GB, disk free < 20 GB.
- **Tests:** `benchmark/guard_swap_test.sh` (the Bridge's (i)–(iii)) all green; the 13:28 guard fails (i) and (ii),
  and three mutants (no growth pause, no stillness check, level-2 pause) each fail their test. `guards_test.sh`
  updated: ALL GREEN. Commit 5e61923.
- **Resume:** old guard 70722 stopped by PID, AMENDMENT line logged, SIGCONT pgid 88672, new guard 21050 armed at
  swap baseline 3,359 MB. Box 1,223 completed 19:15:18.
- **Note:** the ruling described the primary rule as "pause at level ≥ 2 with memory free < 35 %, or level 3"; the
  13:28 rule as committed is level ≥ 4 or memory free < 25 %. "Unchanged" was taken to mean the committed rule;
  flagged to the Bridge. **Resolved:** the Bridge confirmed the committed 13:28 rule is the right one
  (its ruling misstated it from memory); test (iii) keeps pinning it. Amendment, tests, mutants, resume and the
  32 MB jitter band accepted.

### 2026-11-27 19:18–19:43 — C12 breach on box 1227 bg, float64-arbitrated (C12a, commit 07ae7bf); L128 restarted

- 19:18:31 the main pass stopped on C12: box 1227 bg closure 3.63e-4 > 3e-4. One breach in 2,618 box outputs, the
  top of a continuous tail (p99.9 2.6e-4). Diagnosed read-only; nothing restarted.
- The Bridge ruled option (a) (~19:20): the bar stays, and a float64 recompute arbitrates. Mechanism, tests (i)–(iii)
  and the sealed expectation are in `stepT_clarifications.md` §C12a.
- 19:35–19:43 `c12arb 128 1227 bg` under the guard: f64 2.351e-5, cleared. The float32 record is kept in
  `results/coh2t/L128_c12/`.
- 19:43:38 queue relaunched (pid 32454), 1,700 boxes left. Box 1227's other kinds were done by 19:44:31.
- Arbitrated so far: **4**, all cleared in float64, all f64 closures < 1.5e-4. Main-pass kinds: **2** (box 1227 bg, f32 3.63e-4 -> f64 2.35e-5; box 1452 bg, 27 Nov 23:35, f32 3.255e-4 -> f64 1.032e-5; both same selection). Decoy kinds: **2** (box 7 dec_sg, 28 Nov 10:17, f32 1.485e-3 -> f64 4.641e-5, same selection; box 23 dec_sg, 10:18, f32 7.155e+01 -> f64 1.223e-5, selection DIFFERS). This count feeds the L128 results' disclosure line.

### 2026-11-28 10:17–10:30 — main pass done; decoy arbitrations; count split and a silent-f32 check, both decided AFTER the event

- 10:17:09 main pass done (1,700 boxes in 34 sub-runs); `calib` done; `decoys` started (524 boxes).
- 10:17 box 7 dec_sg breached C12 (f32 1.485e-3) and float64 cleared it. 10:18 box 23 dec_sg breached with
  **f32 closure 71.6**, where float32 is grossly wrong, not a tail, and its template/lag selection differs from float64's.
  Float64 cleared it (1.22e-5) and is the box result. dec_train and dec_drift are clean so far (max 6.2e-5).
- **Count split, decided by the Bridge AFTER seeing both #3 (box 7) and #4 (box 23):**
  - Decoy-kind arbitrations are counted separately from main-pass kinds. The ">3 ⇒ flag" line applies to main-pass
    kinds only.
  - Decoy arbitrations go in the daily line, except immediately if any float64 closure exceeds 1.5e-4, or the dec_sg
    breach rate exceeds 1 in 5 after ≥ 20 dec_sg boxes with candidates.
  - **Why:** the sealed ~0.7 expectation was derived from main-pass box outputs (1 tail breach in 2,618). Loud unit-
    amplitude sine-Gaussians are a different population that expectation never covered. The mechanism is unchanged:
    a run stops only if float64 also breaches.
  - This is a post-hoc change to a reporting threshold, recorded as such.
- **Silent-f32 check, registered here BEFORE any computation (the Bridge's go, 28 Nov ~10:20; controls launched 10:22).** Box 23 shows float32
  can be grossly wrong. C12 catches gross errors; this asks whether a wrong selection can slip under 3e-4 unflagged.
  - **Script:** `benchmark/coh2t_c12_silent.py`, CPU only, float64 by the same `_PREC` switch as `c12_float64`, one
    thread like production. Inputs: refs and snippet files regenerated by the driver's own `chunk()` into a scratch
    state dir, so production files are never read-modified or written. It writes only to `results/coh2t/c12_silent/`.
  - **Sample (fixed rule, result-blind):** from the plan's own list of boxes with valid candidates of each kind,
    sorted by tid, take a seeded permutation, `numpy.random.default_rng(20261128 + j)` with j = 1 fg, 2 bg, 3 dec_sg.
    Walk it and take the first boxes whose output exists and is NOT arbitrated:
    - **20 fg** (loud injections, the science population most like dec_sg),
    - **10 bg** (baseline),
    - **20 dec_sg**, run after the decoys step has finished, so the whole dec_sg population exists.
  - **Compared per box,** stored output (float32 for non-arbitrated boxes) against the float64 recompute:
    - (i) identical `tmpl` and `lag` arrays (same selection);
    - (ii) max |f32 − f64| / max(|f64|, 1) over H_rho, L_rho, H_chi2, L_chi2 ≤ **1e-3**. 1e-3 relative is ~0.01 in
      network SNR at ρ≈10, far below any FAR-ladder step. The arbitrated main-pass boxes showed ≤ 4e-4;
    - (iii) float64 closure ≤ 1.5e-4.
  - **Controls, run first:**
    - **Must-pass identity:** the float64 recompute of box 23 dec_sg and of box 1452 bg must reproduce their stored
      float64 box results array-for-array. This proves the scratch inputs equal production's.
    - **Must-fail:** box 23's stored f32 record (`L128_c12/box_0023_dec_sg_f32.npz`) against its f64, under the same
      comparison code, must report "selection differs". If either control fails, the check is void and is reported.
  - **Reading (fixed now):**
    - 0 selection changes in a population: its caveat becomes "f32 silent selection-error rate < 3/n (95 %)", n stated.
    - Any change in dec_sg: recompute ALL dec_sg boxes in float64.
    - Any change in fg or bg: STOP the queue at a box boundary and report to the Bridge, since it touches the science
      result.
    - (ii) failures without a selection change are reported with their size and do not trigger an action by
      themselves.
  - **Cost, corrected:** my "~15 s per box" was for small dec_sg boxes (~1,300–1,600 templates, 1 candidate). Main-pass
    boxes are ~8,600 templates; box 1452 bg (~12 candidates) took 179 s, box 1227 bg 490 s. So fg is ~1–2 min each and
    bg 3–8 min each: about 1–1.5 h CPU in all, peak ~2.4 GB.

### 2026-11-28 ~10:40 — ALL dec_sg boxes recomputed in float64 (the Bridge's GO); registered BEFORE computing

- **Trigger:** decoy rule (b) tripped, with a dec_sg breach rate of **25 of 61** boxes with candidates (41 % > 1 in 5).
  12 of the 25 had a different f32 template/lag selection. All 25 float64 results cleared (max 9.3e-5). Reported
  ~10:37, a few minutes late: my watcher counted arbitrations but did not compute the rate. That is fixed.
- **Decision:** the Bridge's GO, ~10:40 IST, 28 Nov. This is the registered "any dec_sg change ⇒ recompute all" action,
  moved earlier, and it supersedes the 20-box dec_sg sample. It applies to **ALL 192 dec_sg boxes with candidates**,
  including the ones already arbitrated in the run, which are re-verified: the recompute must reproduce their stored
  float64 result array-for-array.
- **Order:** it starts after the decoys step has finished AND after the fg/bg silent sample has landed. If that sample
  shows ANY selection change, the stop-and-report rule comes first.
- **Mechanism:** `coh2t_c12_silent.py decsg_all`, which is c12arb's path extended to dec_sg.
  - CPU float64 from scratch-regenerated inputs, the same code as the controls.
  - Each stored float32 output is copied to `L128_c12/box_<tid>_dec_sg_f32.npz` and the copy verified, before the
    float64 result replaces the final name (`os.replace`). The f32 copy is never deleted.
  - The sticky gate is as in C12a: a float64 closure > 3e-4 writes the f64 breach record and `C12_STOP`, removes the
    final file and stops. The production `run()` then refuses every later sub-run.
  - Resumable from `results/coh2t/c12_silent/decsg_all.jsonl`.
- **Tested on scratch copies** of boxes 7 (arbitrated, same selection), 23 (arbitrated, selection differs) and 10
  (non-breaching):
  - (A) 7 and 23 re-verified identical and left byte-unchanged; 10 rewritten as f64, and its f32 copy equals the
    original.
  - (B) a rerun resumes and changes nothing.
  - (C) a forced float64 breach (test threshold 1e-9) writes `C12_STOP` plus both records and removes the final file.
  - (D) a rerun of decsg_all refuses; (E) the production `run()` refuses.
- **Reported:** a per-box table (f32 vs f64 selection, max rel difference, f64 closure). The decoy verdict is read from
  the float64 results only. **The count of NON-breaching boxes whose selection changed is the silent-error rate on
  this population.**
- **For the L384 gate (Sumit's decision):** float32 is unreliable on large-dynamic-range inputs, here unit-amplitude
  sine-Gaussians. The L384 plan must say how it handles that, either float64 for loud boxes or a loudness trigger.

### 2026-11-28 10:43–10:44 — silent-f32 sample: selection change in bg box 1481 ⇒ L128 queue STOPPED (registered rule)

- Sample results so far:
  - **fg 20/20** same selection (max rel 5.7e-5).
  - **bg 4 of 5 same; box 1481 bg differs.** f32 closure 3.0e-5 (10× under the gate), f64 1.09e-5. Max rel: L_rho
    1.68, L_chi2 2.6, H_chi2 0.65, H_rho 3e-6.
- Per the registered rule ("any change in fg or bg: STOP at a box boundary and report"), the queue was stopped at
  **10:44:18**. The method: wait for the next box-complete line (box 1076 dec_sg), then `stop.sh coh2t_L128`. The queue
  has no PAUSE hook, and box outputs are atomic, so a relaunch redoes only unfinished boxes.
- State after the stop: no `.tmp` outputs, no `C12_STOP`, and the group is gone. The decoys step was in decoys[4] at
  48/50, ~274 boxes left.
- The dec_sg full recompute is NOT started, because this rule comes first. The c12silent job (CPU, no production
  writes) is finishing the remaining 5 bg sample boxes.
- Reported to the Bridge ~10:45. Next, pending its go: a float64 diagnosis of box 1481 to tell a harmless near-tie in
  the selection from a real error.

### 2026-11-28 10:46 — box-1481 diagnostic: "harmless near-tie" vs "real error", fixed BEFORE computing (the Bridge's GO)

- **What "selection" is (`score()`):** the K = 5 templates with the loudest H1 peak |z|², each at its own best lag.
  The stored arrays are values AT those anchors. **The ranked statistic** (reduce, C2) is, per p ∈ {4, 8, 16},
  `score_p = max over the 5 anchors of ρ̂_H² + max over |δ| ≤ 10 ms of ρ̂_L²`, with PyCBC newSNR and v.
- **Computed per candidate of box 1481 bg**, with reduce's own `network_score`, `coh2t.rho_hat` and v series:
  - (a) `score_p` from the STORED float32 arrays, i.e. what production would rank;
  - (b) `score_p` from the float64 recompute (f64 selection, f64 values), the truth;
  - (c) `score_p` at the float32 selection, recomputed in float64: f64 band pieces evaluated at f32's (template, lag)
    anchors. Under float64's ordering, (b) ≥ (c) is not guaranteed, because the anchors are chosen on H1 |z|², not on
    score.
- **The tolerance, stated now:** a candidate is a **harmless near-tie** if |(a) − (b)| / (b) ≤ **1e-3** AND
  |(c) − (b)| / (b) ≤ 1e-3, for every p. Otherwise it is a **real error**.
  - **Why 1e-3 relative:** on the 29 sampled boxes whose selection matched, f32 and f64 agree to ≤ 6e-5 relative on
    the stored statistics. 1e-3 on score ≈ ρ_net² is 5e-4 on ρ_net, ~0.005 at ρ_net ≈ 10. That is far below the step
    of any FAR rung and below the ±1 % K2 tolerance (C18 b).
  - **What this does not cover:** the bg-tail spacing near the 1/day threshold cannot be measured before reduce runs
    on all bg boxes. So a Δ above 1e-3 is called a real error outright, without arguing it is still below the spacing.
- **Also reported:** which anchors differ (tmpl, lag, H1 |z|² of each), whether the max anchor (the one that sets
  score_p) differs, and the c12silent bg tally (n differing of 10).
- **Then NOTHING resumes until the Bridge has talked to Sumit:**
  - (A) harmless: resume, plus a tie-aware selection note and the rate bound;
  - (B) real error: options and costs for redoing bg (and fg) selections in CPU float64, with wall time on N cores and
    the effect on the 1 Dec stop line.

#### 10:49 — box-1481 reading: REAL ERROR by the registered criterion; the arithmetic is right, the selection rule is not tie-stable

- 3 of the box's 8 candidates differ: n = 6587, 19641, 34404. They share one H1 window, so they have the same 5 anchors.
  - **Only the 5th anchor differs:** f32 picks template 2708 at lag 4363; f64 picks 2653 at lag 3984, 379 samples
    (~93 ms) away.
  - Their H1 |z| are **5.66290 vs 5.66292**, a near-tie at the 5th/6th place (relative gap ~3.5e-6, inside float32's
    accumulated error, closure ~3e-5).
  - Their L1 sides differ completely, so score_p moves wherever that anchor sets the max.

| cand n | p | (a) f32 stored | (b) f64 | (c) f64 at f32 anchors | \|a−b\|/b |
|---|---|---|---|---|---|
| 6587 | 4 | 43.247 | 51.650 | 43.247 | 0.163 |
| 6587 | 8 | 43.693 | 38.799 | 43.693 | 0.126 |
| 6587 | 16 | 36.520 | 36.521 | 36.521 | 4e-6 |
| 19641 | 4 | 45.029 | 54.052 | 45.029 | 0.167 |
| 19641 | 8 | 41.530 | 38.872 | 41.530 | 0.068 |
| 19641 | 16 | 39.964 | 39.965 | 39.965 | 3e-6 |
| 34404 | 4 | 48.449 | 57.472 | 48.449 | 0.157 |
| 34404 | 8 | 44.802 | 44.802 | 44.802 | 2e-7 |
| 34404 | 16 | 42.429 | 42.429 | 42.429 | 5e-7 |

- (c) ≈ (a) to ≤ 3e-6 everywhere, so the float32 ARITHMETIC is accurate. The whole difference is WHICH anchor is in the
  top 5. **The registered statistic (hard top-K = 5 by H1 |z|²) is discontinuous at near-ties**, and float32 resolves
  ties differently from float64. The sign is not fixed: f32 is lower at p = 4 and higher at p = 8.
- **Final bg tally (c12silent):** **1 of 10** boxes differ; at the candidate level 3 of 94. **fg: 0 of 20**, rate
  < 15 % (95 %).

### 2026-11-28 10:55 — B4 measurement (the Bridge's GO, measurement only), registered BEFORE computing

Whether to ACCEPT G0 on this measurement stays Sumit's decision. No production writes; the L128 queue stays stopped.
CPU float64 on ≤ 4 workers. The selection-rule question (B2/B3) is with the Bridge and Sumit.

1. **Blind sample (rate, Δscore, near-tie gaps).**
   - **Draw:** from the plan's bg boxes with valid candidates (2,906), a permutation by
     `numpy.random.default_rng(20261128 + 4)`. Take the first **100** boxes, skipping the 10 already sampled
     (reported separately).
   - **Per candidate:**
     - (a) score_p from the stored f32 arrays and (b) score_p from the f64 recompute, p ∈ {4, 8, 16}, with reduce's
       `network_score`, `coh2t.rho_hat` and v;
     - δ_p = b − a, absolute and relative to b;
     - whether the selection is the same;
     - the **5th/6th H1 gap** g = (t₅ − t₆)/t₅, on the f64 H1 peak |z|² (t₁ ≥ t₂ ≥ … over all templates, captured
       from `score()`'s own `argsort` input), plus t₆..t₈.
   - **Outputs:**
     - the candidate-level flip rate (selection differs) with a Clopper–Pearson 95 % CI;
     - per p, the fraction with |δ|/b > 1e-3, quantiles of δ/b, and the sign split;
     - the distribution of g (fractions below 1e-4, 1e-5, 1e-6);
     - flip versus g.
2. **Must-fail control:** box 1481 through the same code path must show |δ₄|/b > 1e-3 and g < 1e-4 for candidates
   n = 6587, 19641 and 34404. If not, the measurement is void.
3. **Propagation to the 1/day threshold and R_tune** (an exhaustive census of where it matters, not a model):
   - (a) f32 score_p for all bg (35,482) and fg (218) candidates from the stored arrays (scratch).
   - (b) The census: f64-recompute every bg box holding any candidate with a32_p ≥ thr32_p / (1 + J) for any p, and
     every fg box (209).
     - thr32_p is the unweighted 1/day cascade threshold from f32 scores.
     - J = max(0.25, 1.5 × the largest relative jump (b − a)/a in the blind sample).
     - The census is chosen from bg scores and the bg threshold only, never from fg or R.
   - (c) Corrected scores: f64 wherever recomputed, f32 elsewhere. R'_p and its 90 % CI come from `evaluate_config`'s
     own `one()` and bootstrap (2,000 resamples, the same seed `[20261126, 128, p]`), on a ctx holding bg and fg only.
   - (d) **Blind to the result:** the script writes and prints ONLY ΔR/h, ΔCI_lo/h, Δthr/thr, and whether clause1
     (R ≥ 1.15 and CI_lo > 1.00) or S2's stop condition (R < 1.00 or CI_hi < 1.15) changes (yes/no).
     h = half-width of the f32 R's 90 % CI. Never R, its CI, thresholds or distances.
4. **Reading rule ("inside the bootstrap CI"), fixed now:**
   - **NEGLIGIBLE** iff for every p: |ΔR|/h ≤ 0.25 AND |ΔCI_lo|/h ≤ 0.25 AND no change in clause1 or S2.
   - **MATERIAL** iff for any p: |ΔR|/h > 0.5 OR |ΔCI_lo|/h > 0.5 OR either verdict changes.
   - Otherwise **INCONCLUSIVE**.
   - Limit: candidates below the census margin are assumed unchanged. The blind sample bounds how often that fails:
     the fraction with (b − a)/a > J is reported.
5. **BLAS benchmark** (B1's cost) runs first, on a box with no other DeepStrain job, at 1/2/4/8 threads on bg boxes
   1948 and 2336. It sets the threads per worker, keeping threads × workers ≤ 8.

### 2026-11-28 11:35:51–12:40 — Mac mini power loss during B4; integrity verified; B4 sample resumed

- **Running at the cut:** the 3 B4 sample workers (c12b4w0–2, CPU, scratch only). scores32 had finished at 11:11. The
  L128 queue had been stopped since 10:44.
- **Last writes 11:35:51**; the Mac rebooted at 12:36.
- **Integrity, checked read-only at 12:39:**
  - git fsck is clean (dangling commits only).
  - All B4 jsonl lines parse, every file ends on a whole line, and **every recorded box has all its candidates**
    (row count = the plan's candidate count, 55 boxes incl. the 1481 control).
  - `scores32.json` is complete (35,700 rows).
  - **Production untouched:** no file in `L128/` or `L128_c12/` is newer than the 10:44 stop. No `.tmp`, no
    `C12_STOP`.
  - Moved aside, not deleted: three scratch refs files written at the moment of the cut (tids 605, 641, 1312,
    regenerable) to `results/coh2t/c12_silent/quarantine_powerloss_20261128/`, with the sha256 of the B4 outputs at
    recovery.
- **Lost:** the three in-flight boxes, ~2–3 min of compute each, plus ~65 min of wall time.
- **12:40:40:** the B4 sample was resumed under the same guards, from the verified files (54 of 100 sample boxes plus the
  control were done). L128 stays STOPPED pending Sumit.

#### 13:09 — B4 blind sample result; census sized; fg census started

- **Must-fail control PASSES:** box 1481's 3 flips appear, |δ₄|/b 0.157–0.167, gap56 3.0e-6. Its non-flipping
  candidates have gaps of 1.6–3.1e-3.
- **Blind sample (100 boxes, 1,248 candidates):**
  - **0 flips**, so the Clopper–Pearson 95 % upper bound is **0.30 %**.
  - |δ_p|/b ≤ 2.8e-5 for every candidate and every p, and 0 candidates have |δ|/b > 1e-3. f64 closure ≤ 5.2e-5.
  - 5th/6th gap: min 1.06e-5; 0.72 % below 1e-4, 7.5 % below 1e-3, none below 1e-5.
- Combined with the first 10-box sample: **1 of 110 boxes, 3 of 1,342 candidates.** All flips have gap ≤ 3e-6; no
  non-flip has gap < 1.06e-5.
- **The census is much larger than my estimate.** J = max(0.25, 1.5 × 2.8e-5) = 0.25, which gives 1,077 bg boxes
  (~15,900 candidates, 45 % of bg) plus 209 fg boxes: ~52 core-h, ~17 h on 3 workers.
  - Over the 12 h line, so the bg part is taken to the Bridge with an alternative: a Monte-Carlo bound at the 95 %
    flip rate, an amendment.
  - The fg part (209 boxes, ~1.5 core-h) is within the registered census and was started at 13:10 with
    `C12S_CENSUS_KINDS=fg`.

### 2026-11-28 ~13:30 — B4 amendment (the Bridge's GO): propagation by bound instead of the bg census; registered BEFORE computing

- **Decided AFTER the B4 blind-sample result (0 flips in 1,248).** **Why:** the registered census came out at
  ~52 core-h, because the J floor of 0.25 is forced by the ±19 % flip jumps. A bound answers the same question in
  minutes. The bg census runs only if the worst case is not NEGLIGIBLE; in that case the Bridge has approved the
  ~17 h census.
- **fg is exact** in every scenario: f64 from the fg census (209 boxes, running).
- **Baseline:** all f32. Every scenario changes only bg scores (plus fg → f64) and is compared to the baseline.
  - **fg-only:** bg f32, fg f64. Reported separately, so the fg effect is visible on its own.
  - **(ii-a) Monte Carlo:** 2,000 draws, seed `default_rng(20261128 + 5)`.
    - Each bg candidate flips independently with probability **0.0030** (the sample's 95 % upper bound).
    - A flipped score is multiplied by (1 + s·j). s = ±1 with equal probability. j is drawn from the pooled observed
      |relative jumps| of box 1481's flips (all p, values > 1e-3: 0.194, 0.200, 0.186, 0.112, 0.064). Pooling across
      p is more conservative than using each p's own jumps; at p = 16 those are ~0.
    - **Read at the 95th percentile:** p95 of |ΔR|/h and |ΔCI_lo|/h, plus the fraction of draws in which clause1 or
      S2 changes. A fraction above 5 % counts as a verdict change.
  - **(ii-b) Deterministic worst case:** n_f = ceil(0.0030 × N_bg) flipped candidates, placed adversarially per p,
    with |jump| = **0.25** (above the largest observed, 0.200).
    - **WC-down** (lowers R and CI_lo, as the Bridge specified): the n_f bg candidates with the highest f32 score
      strictly below the f32 1/day threshold (unit weights), each ×1.25. They cross and raise the threshold.
    - **WC-up**, added by me because a falsely RAISED R is the dangerous direction for a positive claim: the top n_f
      bg candidates by score, each ×0.75. This lowers the threshold.
    - Both are read with the rule.
- **The rule** (unchanged from the 10:55 entry), with h = the half-width of the baseline R's 90 % CI:
  - NEGLIGIBLE iff |ΔR|/h ≤ 0.25 and |ΔCI_lo|/h ≤ 0.25 and no clause1/S2 change, per p and for the best config;
  - MATERIAL iff any > 0.5 or any verdict change;
  - otherwise INCONCLUSIVE.
- **Verdicts:**
  - (ii-a) and both (ii-b) NEGLIGIBLE ⇒ no bg census;
  - (ii-b) MATERIAL or INCONCLUSIVE ⇒ the full census (i).
- **Machinery:** `coh2t_c12_b4.py bound`, a vectorised exact replica of `evaluate_config`'s R / CI / clause1 path:
  - the same bootstrap draws (seed `[20261126, 128, p]`);
  - the Y1 side precomputed per resample;
  - one argsort per score vector.
  - **Gated:** it must reproduce `evaluate_config`'s R_tune, ci90 and thr_cascade on the baseline to 1e-12 relative,
    checked by equality only and printed as true/false. Otherwise the bound is void.
- **Blind as registered:** only Δ/h, Δthr/thr and the verdict-change flags are written.
- **The choice of how L128 resumes stays Sumit's.**

#### 15:45 — B4 bound result (blind): the registered TEXT and the CODE disagree on one scenario; reported, not resolved by me

- **Replica gate PASS** at p = 4, 8 and 16: the vectorised replica reproduces `evaluate_config` to 1e-12.
- **fg-only** (fg f64): exactly 0 change in R, CI_lo and threshold, at every p.
- **(ii-a) MC** (2,000 draws, flip rate 0.30 %): **NEGLIGIBLE**. p95 of |ΔR|/h and |ΔCI_lo|/h is 0.0 at every p; 0 %
  of draws change clause1, S2 or the best config.
- **(ii-b) WC-up** (top 107 bg ×0.75): thresholds −7.7 to −9.8 %, ΔR/h = ΔCI_lo/h = 0 at every p, no verdict change ⇒
  **NEGLIGIBLE**.
- **(ii-b) WC-down** (107 bg candidates just below threshold ×1.25): thresholds +20.9 to +21.5 %. |ΔR|/h ≤ 0.038 and
  |ΔCI_lo|/h ≤ 0.036 at every p. No clause1 change and no S2 change at any p, and no change in S2 for the best config.
  **BUT the identity of the best p configuration changes.**
  - The **registered text** (10:55 rule, repeated at ~13:30) lists magnitude and clause1/S2 changes only, so it reads
    WC-down as **NEGLIGIBLE**. On that reading no bg census is needed.
  - The **code** (`propagate()` / `bound()`, written with the registration) also counts a best-config change as
    MATERIAL, so it outputs **overall: "run the full census (i)"**.
  - **This is a discrepancy between my registration and my implementation. I am not choosing between them after
    seeing the result.** Sent to the Bridge.
- **Substance:**
  - A best-config change with no per-p clause1 change can still change G0's clause1, since G0 uses the CHOSEN config.
    The bound did not record whether clause1 of the best config changes; adding that is one bit and about 5 min.
  - That a −0.04 h shift flips the argmax means the per-p R values are already within ~0.04 h of each other: the
    choice of p is fragile irrespective of precision. (No R value was written or seen.)
  - WC-down is adversarial: all 107 flips at the 95 % upper-bound rate, placed at the threshold, each +25 %. The
    expected number of flips among the ~28 background triggers that set the 1/day threshold is ≤ 0.08 at that rate.
- Cost of the full bg census: ~51 core-h; on 1–2 workers (more thrash the guard) that is ~26–51 h, CPU only (the GPU
  stays free).

#### ~15:55 — the Bridge's ruling on the B4 bound, and a post-hoc one-bit check committed BEFORE it runs

- **Ruling: the REGISTERED TEXT governs.** The registered reading of the B4 bound is **NEGLIGIBLE** (magnitude plus
  clause1/S2). The code also printed MATERIAL, through a best-config clause that was not registered. That is reported
  and not graded.
- **Post-hoc conservative check, graded as such.** It can only push toward more care. Committed before running:
  - **Question:** under WC-down (107 bg candidates just below the f32 1/day threshold ×1.25, fg f64), does **clause1
    of the BEST config** differ from clause1 of the best config at baseline? Best = argmax R_tune over p ∈ {4, 8, 16};
    clause1 = R ≥ 1.15 and CI_lo > 1.00.
  - **Output:** ONE bit (changes yes/no). No R, CI, threshold or config identity is written or printed.
  - **Rule:**
    - NO ⇒ the G0 clause1 verdict is safe under the adversarial case, and no bg census.
    - YES ⇒ the full bg census (pre-approved), 1–2 workers, CPU only, pause-only guard, after a value-gate paragraph.
  - **Limit:** "best" here ignores the other gates (F1/F2/P1/K1), which are not computable before the stage ends, so
    the chosen config in the real evaluation could differ.
- Machinery: `coh2t_c12_b4.py wcbit`, which reuses `bound()`'s gated replica and WC-down construction unchanged.
- **Result:** `clause1_of_best_changes_under_WC_down: false`, with no traceback. By the rule committed at 1ce9706 before the run, the
  G0 clause1 verdict is **safe under the adversarial WC-down case, and no bg census is run**. Post-hoc, graded as such.
  Only the bit was written (`results/coh2t/c12_silent/b4/wcbit.json`). The L128 resume decision is with Sumit.

### 2026-11-28 ~16:05 — L128 RESUMED as registered (option A; the Bridge's decision on Sumit's delegation)

- **B4 is CLOSED.** The registered reading is NEGLIGIBLE. The one-bit post-hoc check came back NO; its rule (commit
  1ce9706, 15:47:08) predates the result (`wcbit.json`, 15:52:27).
- **Decision:** option A, resume as registered on the Mac mini GPU + 1 core, under the pause-only guard and the
  1 Dec 12:00 stop line. Sumit delegated it ("you can plan and coordinate, I trust you") and can still override.
- **Conditions, written here BEFORE the resume:**
  1. **The G0 report shows all three p (4 / 8 / 16) side by side.** The registered max-over-p stays primary, with two
     caveats:
     - the choice of p is not informative, because the per-p R values are within ~4 % of the CI half-width (seen
       blind in B4 as a best-config switch under WC-down);
     - **tie flips** (B4): rare 5th/6th H1 near-ties change the top-5 anchor between float32 and float64. 1 of 110
       sampled bg boxes, 0 of 1,248 blind candidates. The worst case is bounded, and negligible by the registered
       reading.
  2. **The L384 plan** carries B3 (a tie-proof template rule) and a pre-declared single p in place of max-over-p, both
     registered before any L384 compute.
  3. Resume PIDs and ETA are reported. A 25 % check-in at ~8–10 GPU-h covers progress and pauses only.
- **A resume fix, found before launching:** the queue re-runs every step, and `calib` is write-once (C6) and asserts
  its file does not exist. Its file was written at 10:17, so a plain relaunch would have failed there and stopped the
  queue.
  - The queue now skips `calib` when `decoy_calibration.json` already exists, and logs that it did.
  - Nothing else changes: `run_step` already skips finished boxes; `k2select` onward have not run yet.
- **Still pending, approved:** the full dec_sg float64 recompute (`coh2t_c12_silent.py decsg_all`, CPU). It starts when
  the decoys step finishes, while the GPU queue continues. Decoy arbitrations go in the daily line, under the earlier
  count split.
