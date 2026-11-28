# Referee guide: LongChirp preprint, claim by claim

Each claim in `longchirp_preprint.pdf` maps to the file that holds its number, the command that regenerates it, and
the expected output. All paths are relative to `primordial_blackhole_search/` at git tag `longchirp-v1.0.0` (commit
`492d35270ec87a656fcf4de43ccfce564ed2a849`, software DOI 10.5281/zenodo.24399603).

**Run times.** A run time is given only where the release recorded one, on macOS arm64 with the Apple GPU. "Not
recorded" means it was never timed. It does not mean it is slow.

**Setup (once).** Build the environment from `requirements-lock.txt` with Python 3.12.13, then run
`benchmark/fetch.py fetch`. The fetch downloads about 8.3 GB from GWOSC, one file at a time, and is resumable.
Do not run `discover`.

| # | Claim in the paper | File holding the number | Command to regenerate | Expected output | Run time |
|---|---|---|---|---|---|
| 1 | Y1: 57.5 Mpc (8.3 % found) at 1/day; 51.7 Mpc (7.5 %) at 1/week; thresholds 16.78 / 17.87 | `benchmark/results/y1_eval_own.json` (`at_far.1/day.chirp_distance_mpc` = 57.47453519740948) | copy `benchmark/results/y1_bins/` to `benchmark/state/y1/`, run `benchmark/y1_bank.py submit`, then `benchmark/evaluate.py <y1 submission dir>` | the same JSON values | evaluation not recorded. Regenerating Y1 from raw strain took 28.3 h (background) + ~4.5 h (foreground) |
| 2 | Y1 at ±1 s is identical | `benchmark/results/y1_eval_fixed1.json` | `evaluate.py <dir> --fixed-var 1.0` | identical to #1 (every Y1 trigger already has var = 1 s) | not recorded |
| 3 | Y2: 48.4 Mpc (7.1 %) / 46.2 Mpc (6.7 %) at its own var 20.2 s; thresholds 7.88 / 9.77 | `benchmark/results/y2_eval_own.json` | `benchmark/evaluate.py benchmark/submissions/y2` (the submission is committed) | `chirp_distance_mpc` 48.354194379119384 / 46.225956922574156 | not recorded |
| 4 | Y2 at ±1 s: 25.5 / 25.4 Mpc | `benchmark/results/y2_eval_fixed1.json` | `evaluate.py benchmark/submissions/y2 --fixed-var 1.0` | 25.539495985400656 / 25.397022820039624 | not recorded |
| 5 | Y0: 154.6 / 124.1 / 103.8 Mpc at network SNR ≥ 8 / 10 / 12 | `benchmark/results/oracle_snr.csv` + `benchmark/injections.csv` | `paper/make_figure.py` recomputes them (function `y0`). No release script prints the distances | 154.632…, 124.063…, 103.772… (330, 265, 221 found) | seconds |
| 6 | Y1/Y2 ratio 1.19 [1.10, 1.29] at 1/day, 1.12 [1.05, 1.20] at 1/week; ±1 s: 2.25, 2.04; 111 vs 96 found, 93 common | `benchmark/results/y1_vs_y2_bootstrap.json` | **not reproducible from the release:** the bootstrap script is not committed (paper, correction iii) | the stored JSON | — |
| 7 | 27.3 days of background; 1/day = 28th-loudest, 1/week = 4th; 1/month unreachable (0.90 events) | `y1_eval_own.json` (`background_livetime_days` = 27.30666666666667) | `evaluate.py` | as stored | — |
| 8 | Evaluator golden E1–E5 pass (perfect detector within 3σ; blind = 0; 27 above threshold; clustering; var rule) | `benchmark/results/golden_evaluate.json` | `benchmark/golden_evaluate.py` | `"pass": true` | not recorded (synthetic data only) |
| 9 | Injection path vs PyCBC: SNR to ≤ 9×10⁻¹⁰, merger within 0.30 ms, recovery within 2 of optimal | `benchmark/results/golden_injection.json` | `benchmark/golden_injection.py 1249001472` | `"pass": true`, `rel_err` ~8–9×10⁻¹⁰ | not recorded |
| 10 | 64 GWOSC files match GWOSC's published statistics: worst 5.6×10⁻⁸ relative, mean within 1.05×10⁻¹¹ σ | `benchmark/results/verify_published.json` | `benchmark/verify_published.py` (queries the GWOSC API) | `"all_pass": true` | not recorded |
| 11 | numpy PSD matches PyCBC's to 4.7×10⁻¹⁰; whitened to 4.5×10⁻¹²; 1.02 vs 2.10 GB peak | `benchmark/results/lowmem_psd_check.json` | `benchmark/lowmem_psd.py benchmark/data/H1_1245888512.npy` | `"pass": true` | not recorded |
| 12 | Y1 golden: whitening exact; GPU vs float64 ≤ 2.7×10⁻⁵; windowed vs whole 3.2×10⁻⁶; 5 loud dev injections at 0.77–0.91 of optimal within 59 ms | `benchmark/results/y1_golden_confirm.json` | `benchmark/y1_bank.py confirm` (development segments 3–4 only) | `"pass": true` | not recorded (the run's peak memory was 2.57 GB) |
| 13 | The earlier Y1 golden rule failed (d_c proxy) | `benchmark/results/y1_golden.json` | (superseded; kept as a record) | `"pass": false`, `loud_min_neigh_stat` 6.085 | — |
| 14 | Y1 foreground shortcut: 0–2.03 lower on 14 held-out injections; equal on 4 of 5 loud, 0.72 lower on one | `y1_golden_confirm.json` (`G4_injections[*].full_minus_neigh`) | as #12 | as stored | — |
| 15 | Y2 timing from dev only: 32 of 168 detected, offset 5.7 s, var 20.2 s | `benchmark/results/y2_calibration.json` | `benchmark/y2_cnn.py calibrate` (dev segments) | `offset_s` 5.730761289596558, `var_s` 20.15482234954834 | not recorded |
| 16 | Bank density adequate: a half-density bank differs by +1.1 / −2.5 / −2.2 % | `results/bank_adequacy_bank_dense_s0.0005.json` (an earlier, window-based test) | `scripts/bank_adequacy.py --src bank_dense_s0.0005.json` | ratios 1.0113 / 0.9745 / 0.9785 | — |
| 17 | CNN training data precede every dev and test segment | `data/manifest.json` (latest GPS 1242247048, against the first dev segment at 1245888512) | read the file | as stated | — |
| 18 | PyCBC waveform memory retention, ~0.2 → 2.0 GB over 30 random-mass waveforms | `benchmark/pycbc_leak_repro.py` (its docstring holds the measured table); upstream gwastro/pycbc#5554 | `benchmark/pycbc_leak_repro.py td` (also `td-same`, `fd`, `lalsim`) | resident memory rising with each new waveform length (`td`), flat for one length (`td-same`) | not recorded |
| 19 | Figure 1 | `paper/fig_distances.pdf` | `paper/make_figure.py` (reads the tag through `git show`) | the same figure | seconds |

## Where to look hardest
- **#6 is not reproducible from the release as it stands.** The bootstrap script is missing. A referee who needs it
  should ask for it; it belongs in the next release.
- **#13 / #12:** the Y1 recovery rule was changed after a failure and confirmed on other development segments. Judge
  whether that confirmation is adequate.
- **#9** and the choice of the 500 Mpc chirp-distance cap used test segment 1249001472. Neither tuned a search
  statistic.
- **Thresholds come from one background realisation.** No interval in the paper includes threshold uncertainty.
