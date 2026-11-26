# Step T: run notes (build runs before any T pass)

*Deepstrain-personal [model: Claude Opus 5.5]. Declared against measured, for every run. Newest first.*

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
