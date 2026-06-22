# Primordial Black Hole Search — subsolar-mass mergers in public LIGO data

**Goal:** a deep-learning search pipeline for subsolar-mass binary black hole mergers in
public LIGO/Virgo strain data. A confirmed subsolar black hole cannot be made by a star —
it would be evidence of new physics (leading candidate: primordial black holes from the
early universe, a dark-matter candidate).

**Realistic deliverables, ranked by likelihood:**
1. **Method result** — an architecture that handles *minutes-long* faint chirps (the
   known gap: MLGWSC-1 only tested ≤20 s signals; ML hit ~70% of matched-filter
   sensitivity on real noise).
2. **A sensitivity bound** — "my pipeline excludes mergers of mass X above rate Y" →
   constraint on the PBH fraction of dark matter, for an under-covered corner
   (eccentric / asymmetric-mass binaries are nearly untouched).
3. **A candidate** (lottery ticket; vetted by two-detector coincidence ≤10 ms).

## Why this is winnable
- Waveforms for light systems are long, clean inspirals — very accurately modeled
  (the merger happens above the sensitive band; it's textbook physics).
- The limiting factor for official searches is template-bank compute, exactly what a
  learned detector amortizes.
- All data is public (GWOSC). Sensitivity is *measurable* by injection campaigns —
  a null result still produces a real number.

## Prior art & how this differs (literature-checked 2026-06-23)
Two distinct literatures touch "ML + (primordial) black holes + GW"; our niche sits precisely between
them, so we make only the **scoped** claim — NOT a broad "no ML subsolar work exists" (it does):
- **Subsolar DETECTION is matched-filter, not ML.** Every published subsolar search uses template-bank
  matched filtering — LVK [O3a 2106.08979](https://arxiv.org/abs/2106.08979),
  [O3b 2212.01477](https://arxiv.org/abs/2212.01477), [O4 bank 2412.10951](https://arxiv.org/abs/2412.10951),
  [hierarchical 2409.11317](https://arxiv.org/abs/2409.11317). Their documented bottleneck — dense banks up to
  ~O(10⁷) templates for long-duration subsolar signals — is *exactly* the wall we measured (v2 PARKED: subsolar
  needs ≤0.1 % Mꜜ spacing). That's the gap a learned trigger is meant to amortise.
- **ML + PBH exists, but for POPULATION inference, not strain-level triggering.** The deep-learning PBH line
  estimates population hyperparameters (f_PBH) from *event catalogues* via NPE / normalizing flows
  ([2503.05570](https://arxiv.org/abs/2503.05570), [2505.15530](https://arxiv.org/abs/2505.15530)) or uses DINGO
  for per-event posteriors — a different task from our strain-level CNN detection trigger.
- **The ML-vs-MF detection benchmark ([MLGWSC-1, 2209.11146](https://arxiv.org/abs/2209.11146)) only tested
  ≤20 s signals.** So the defensible niche is a *strain-level ML detection trigger for minutes-long subsolar
  signals* — the modest, scoped version is the only one we claim.

## Phases — v1 COMPLETE (see RESULTS.md for all numbers)
- **Phase 0 ✅ feasibility spike**: fetch real strain, inject subsolar chirp, recover
  (MF SNR 81.8, 16 ms timing). `spike/`
- **Phase 1 ✅ data + injections**: 24×4096 s H1 + 2 L1 segments; golden test pins the
  whitening/SNR conventions to PyCBC at 0.0%. `pbh/`, `tests/`
- **Phase 2 ✅ baseline CNN**: 1.17M params, val AUC 0.777 (after fixing overfitting
  with 2× noise + augmentation).
- **Phase 3 ✅ chunked transformer**: honest negative — 0.758, loses to the CNN.
- **Phase 4 ✅ evaluation**: **41–45% of ideal-MF sensitive distance at a zero-FA
  threshold (6.8 h real test noise)**, flat across the subsolar mass range.
- **Phase 5 ✅ search demo**: ML sweep → MF follow-up → H1×L1 coincidence; SNR-22 event
  recovered at network SNR 36.4, dt error 0.02 ms, 18 spurious peaks (incl. a louder
  glitch) rejected. Plus the template-dephasing measurement (±0.01% Mc → −28% SNR).
- **v2 next**: close the 45→70% gap; scale FAR to 1/month; fair transformer rematch;
  two-detector training; eccentric/asymmetric corners. Ranked list in RESULTS.md.

## Ground rules (honesty)
- Every sensitivity claim comes from injections into *real* noise, never Gaussian noise.
- Always report at false-alarm rate ≤ 1/month equivalent.
- Matched filtering on the same segments is the baseline to beat (or match cheaply) —
  PyCBC gives it to us for free.

## Stack
Python 3.12 (uv-managed) · PyCBC (waveforms, matched filter, PSDs) · GWpy/GWOSC (data
access) · NumPy/SciPy/Matplotlib. Training stack (PyTorch) added in Phase 2.

## Key references
- [O3a subsolar search (LVK), arXiv:2106.08979](https://arxiv.org/abs/2106.08979) ·
  [O3b subsolar search, MNRAS 524 (2023)](https://academic.oup.com/mnras/article/524/4/5984/7060405)
- [MLGWSC-1 mock data challenge, arXiv:2209.11146](https://arxiv.org/abs/2209.11146) —
  the honest ML-vs-matched-filter benchmark
- [Low-mass-ratio subsolar corner is open, arXiv:2105.11449](https://arxiv.org/abs/2105.11449)
- [GWOSC — public data](https://gwosc.org)
