# Step T: run notes (build runs before any T pass)

*Deepstrain-personal [model: Claude Opus 5.5]. Declared against measured, for every run. Newest first.*

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
