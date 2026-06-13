# Echoes — hunting for quantum structure at black-hole horizons

*A real-data project: search LIGO/Virgo public strain data for gravitational-wave
**echoes** after black-hole mergers. In Einstein's classical picture the horizon is a
perfect one-way door and post-ringdown should be silence; several quantum-gravity
proposals (firewalls, fuzzballs, horizon-scale structure) instead predict a leaky
echo chamber: `bang… beep… beep… beep…` at a fixed, predictable spacing.*

**Status:** ✅ **v1 pipeline complete and validated end-to-end** (fetch → inject →
search → background → sensitivity), run on two real events. **Hardware:** runs on a
Mac. **Data:** public, from [GWOSC](https://gwosc.org).

> **v1 result:** GW150914-like echo trains with first-pulse amplitude ≥ 2
> whitened-noise σ would have been detected with ~100% probability (p < 0.01);
> the post-ringdown data of **GW150914** and **GW151226** show no such structure
> (p ≈ 0.4–1.0, both pre-registered statistics). A clean null, consistent with
> Westerweck et al. — and a measured sensitivity to hang it on.
>
> **v2 result (2026-06-12):** a noise-trained ML scorer (comb statistic on a
> conv-net's reconstruction-error envelope), judged by the *identical* harness,
> **improves amplitude sensitivity ~13×** (50% recovery at ≈ 0.11σ vs the
> comb's 1.5σ; `results/07_ml_vs_comb.png`), passes an irregular-spacing
> specificity control (6%/2% false-fire vs 100% periodic recovery), and keeps
> the on-source null (p = 0.75).
>
> **v3 (2026-06-12):** the advantage is **family-robust** — 97–100% recovery at
> 0.5σ across frequency/decay/reflectivity variations
> (`results/08_family_robustness.png`).
>
> **v4–v5 (2026-06-13, the production-path verdict):** raw-strain injection
> with measured amplitude calibration. **Band-honesty measured** (450 Hz raw
> injections die in the bandpass: 10% vs the invalid control's 100%), and the
> **fair head-to-head** through the identical path
> (`results/10_head_to_head.png`): ML 50% point ≈ 0.85σ vs comb ≈ 1.05σ —
> **a real but modest ~1.2× advantage. The earlier 13× was an artifact of the
> whitened-domain convention.** The v2→v5 arc is the project's case study in
> why production-path validation must precede sensitivity claims. Details:
> `notes/lab_notebook.md`.

---

## 1. The physics, in one breath

Two "walls" sit around a newborn black hole:

- **Inner wall (speculative):** if quantum gravity puts structure a Planck length above
  the horizon, that structure partially *reflects* infalling waves instead of swallowing
  them.
- **Outer wall (standard GR):** the **photon ring** at 1.5× the horizon radius, where
  gravity is strong enough that light orbits in circles. For outgoing waves it's a
  partial barrier.

Reflective inner wall + partially-reflective outer wall = **cavity**. Merger ringdown
energy gets trapped, bounces, and leaks out a pulse per round trip.

**The key number:** the round-trip time is
`Δt ≈ 8M·log(M/ℓ_Planck)` (plus spin corrections) — for GW150914 this is
**0.29 s** ([Abedi, Dykaar & Afshordi, PRD 96, 082004 (2017)](https://arxiv.org/abs/1612.00266);
[Wang & Afshordi, arXiv:1803.02845](https://arxiv.org/abs/1803.02845)).
Evenly-spaced pulses are the gold signature: noise rarely repeats itself at a fixed
interval, and the interval is *predicted per event* from the remnant's mass and spin.

## 2. The state of the field (verified, June 2026)

- **2016:** Abedi, Dykaar & Afshordi claim tentative evidence (~2.5σ combined) in the
  first LIGO events ([arXiv:1612.00266](https://arxiv.org/abs/1612.00266)).
- **2018:** Westerweck et al. redo the **background estimation** with more off-source
  data and find much lower significance
  ([PRD 97, 124037](https://arxiv.org/abs/1712.09966)).
- **2018:** Abedi et al. [reply](https://arxiv.org/abs/1803.08565) that even the
  critics' own p ≈ 0.02 is "moderate evidence." The argument has never fully settled.
- **Takeaway:** the bottleneck of this subfield is **methodology and statistics**, not
  ideas. Every controversy has been about background estimation. That's the standard we
  have to be paranoid about.

## 3. Where the ML angle fits (honest novelty)

Template searches must *guess the echo waveform* (nobody knows it — it depends on
unknown quantum-gravity reflectivity). Comb searches exploit only periodicity. The
opening for us:

> Train a model **purely on detector noise** from the same observing run — no signal
> assumptions — then flag post-merger segments containing structure the noise model
> can't explain, *especially repeating structure at the predicted per-event spacing*.

Honest calibration: autoencoder-style **anomaly detection on GW data exists** as a
field ([e.g. arXiv:2411.19450](https://arxiv.org/abs/2411.19450)); what we did **not**
find published is that paradigm aimed specifically at **post-merger echoes** with
injection-calibrated sensitivity. The niche is "known ML paradigm, unsolved contentious
target," not "new paradigm." That's still a real contribution, because:

- **Either outcome is reportable.** Jackpot = repeating structure at each event's
  predicted spacing across multiple events, with honest statistics. Realistic good
  outcome = **an upper limit**: "any reflected energy is below X% by this
  shape-agnostic method" — a genuine constraint on quantum-gravity proposals.

## 4. The pipeline

```
echoes/
  scripts/
    echolib.py            # shared: data access, whitening, injection, the statistic
    01_fetch_data.py      # ✅ download + whiten any cataloged event, sanity plots
    03_inject.py          # ✅ synthetic echo trains injected into real noise (demo)
    04_search.py          # ✅ on-source comb search (GW150914)
    05_background.py      # ✅ THE step: background distribution + p-values
    06_sensitivity.py     # ✅ injection-recovery curve (the money plot)
    run_event.py          # ✅ catalog-scaling entry point (per-event config)
  data/                   # cached strain (~16 MB per detector per event)
  results/                # plots + saved score arrays
  notes/lab_notebook.md   # decisions, simplifications, results, v2 roadmap
```

**Build order (v1 = all done ✅, validated on GW150914 + GW151226):**

1. ✅ Environment (venv + gwpy/gwosc).
2. ✅ **Fetch & look** — GW150914 chirp visible in whitened H1/L1 (toolchain proven).
3. ✅ **Injection framework before any search** — damped sine-Gaussian trains at
   spacing Δt in *real* noise; statistic validated on synthetic + real noise first.
4. ✅ **Baseline search** — shape-agnostic comb statistic on the energy envelope's
   autocorrelation; network-coherent (H1+L1 must agree on Δt).
5. ✅ **Background estimation** — identical statistic on event-free segment pairs;
   p-value = rank. *No detection claim ever comes from anything but this.*
6. ✅ **Sensitivity curve** — blind < 1σ, 50% at 1.5σ, 100% ≥ 2σ (p < 0.01).
7. ⬜ **v2:** the ML scorer (noise-trained anomaly model through the *identical*
   harness), independent background blocks, full O3/O4 catalog with Δt from catalog
   mass+spin, raw-strain injection, cross-event stacking. See lab notebook.

## 5. Ground rules (so we never fool ourselves)

1. **Sensitivity before search.** No detector runs on real post-merger data until its
   injection-recovery curve (amplitude vs detection probability) is measured.
2. **Background defines significance.** A score means nothing except relative to its
   off-source distribution. Westerweck's critique is our checklist.
3. **Per-event predictions are pre-registered.** Δt comes from the catalog mass/spin
   *before* looking at the post-merger data, never tuned after.
4. **Null results are results.** The honest deliverable is a sensitivity statement
   either way.

## 6. References

- [Abedi, Dykaar, Afshordi — "Echoes from the Abyss" (PRD 96, 082004)](https://arxiv.org/abs/1612.00266)
- [Westerweck et al. — "Low significance of evidence for black hole echoes" (PRD 97, 124037)](https://arxiv.org/abs/1712.09966)
- [Abedi et al. — Comment on Westerweck (arXiv:1803.08565)](https://arxiv.org/abs/1803.08565)
- [Wang & Afshordi — "Black Hole Echology: The Observer's Manual" (arXiv:1803.02845)](https://arxiv.org/abs/1803.02845)
- [Cardoso & Pani — "Tests for the existence of black holes through gravitational wave echoes" (Nat. Astron. 1, 586)](https://arxiv.org/abs/1709.01525)
- [GWOSC — Gravitational Wave Open Science Center](https://gwosc.org) · [GWpy docs](https://gwpy.github.io)
- [Unsupervised anomaly detection in GW data (arXiv:2411.19450)](https://arxiv.org/abs/2411.19450)
