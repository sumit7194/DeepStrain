# DeepStrain

**Deep-learning searches of public LIGO/Virgo data for black-hole signatures.**

Three real-data gravitational-wave sub-projects — a **subsolar / primordial-black-hole merger search**, a **post-merger echo search**, and **ringdown spectroscopy** (a no-hair test) — plus the black-hole physics notes that started it all. Everything is measured on **real O3a/O4 detector noise**: sensitivity comes from injections, significance from a measured background, and **null results are treated as results.**

> A computer engineer's deep dive into gravitational-wave astronomy. Built carefully: every load-bearing claim is checked against the literature, every sensitivity number comes from injections into real noise, and a regression gate (`./verify.sh`) asserts the headline results never silently change.

<p align="center"><img src="penrose.png" width="520" alt="Causal map of an (eternal) black hole"></p>

---

## The three searches

### 🛰️ `primordial_blackhole_search/` — subsolar-mass mergers
Below ~1 M☉ no star can collapse into a black hole, so a *subsolar* merger would be a smoking gun for **primordial black holes** (a dark-matter candidate). A CNN trigger on real O3a H1 noise reaches **41–45% of the ideal matched-filter sensitive distance** at a zero-false-alarm threshold — for *minutes-long* signals, the regime with no published ML search.

Then a methodical attempt to close the 45→70% gap, each step pre-registered:
- **Track-score aggregation** (rungs 1 & 2): two clean **negatives** — combining per-window scores can't recover what the matched filter has, confirmed even against oracle ceilings.
- **Semi-coherent matched filter** (rung 3): a chunked, phase-aware approach whose **oracle ceiling reaches 66–76%** (optimistic — true templates + a lenient threshold). A *learned* version is the current frontier.

### 🔔 `echoes/` — post-merger gravitational-wave echoes
If black-hole horizons have quantum structure, a merger might be followed by faint, repeating "echoes." A full pipeline (injection-calibrated, background-defined *p*-values) searches the GW150914/GW151226 post-ringdown. A small ML noise-model scorer gives a **modest but real ~1.2× sensitivity edge** over the classical comb through the production path — band-honest, periodicity-specific, and the on-source events are **null** (consistent with the published re-analyses).

### 🌀 `ringdown_spectroscopy/` — black-hole spectroscopy / no-hair test
Fit the post-merger ringdown tones (quasinormal modes) and test whether they imply a *consistent* mass and spin — the no-hair theorem. An amortized simulation-based-inference model with the ringdown **start-time marginalized by construction** measures the deviation δ **2.6× tighter than the classical fit**, calibration-certified (held-out coverage 0.91). On **GW250114** (the loudest event ever recorded): **δ = −0.16, consistent with a Kerr black hole.**

---

## Headline numbers

| Search | Result | Status |
|---|---|---|
| Subsolar CNN | 41–45% of ideal-MF distance · 0 false alarms in 6.8 h real noise | v1 ✓ |
| Subsolar — closing the gap | score aggregation = clean negatives; semi-coherent **oracle ceiling 66–76%** | learned model in progress |
| Echoes ML scorer | **~1.2×** over the classical comb (production path); on-source null | complete ✓ |
| Ringdown no-hair | **σ(δ) 2.6× tighter** than classical; GW250114 Kerr-consistent | complete ✓ |

---

## Layout
```
primordial_blackhole_search/   subsolar/PBH merger search (CNN + semi-coherent)
echoes/                        post-merger echo search (comb + ML scorer)
ringdown_spectroscopy/         QNM fitting + SBI no-hair test
*.md  (root)                   black-hole physics notes (holography, dimensions, paradox)
*.py  (root)                   figure generators (light cones, Penrose maps, …)
verify.sh                      regression gate over every headline artifact
dashboard.py                   live run monitor (stdlib HTTP, no deps)
```
Each sub-project is self-contained: `scripts/` (numbered steps + a shared lib), `notes/lab_notebook.md` (the raw record), `README.md` (methods + decisions), and its own `uv` virtual environment.

## Reproduce
Large, regenerable artifacts — raw LIGO strain, spectrogram shards, model weights, venvs — are **not** committed; only the code, docs, result records, and the segment `manifest.json` (so the exact GPS segments can be re-fetched from GWOSC).

```bash
cd <subproject> && uv sync               # create the per-project venv
.venv/bin/python scripts/01_*.py         # fetch data from GWOSC, then run the pipeline
./verify.sh                              # regression gate: assert the saved headline results
```

## Ground rules (the project's ethos)
- **Sensitivity from injections** into real noise — never assumed.
- **Significance from a measured background** — never a theoretical *p*-value.
- **Pre-registration** of each analysis before looking at on-source data.
- **Null results are results.** Several headline outcomes here are clean negatives, documented as carefully as the positives — including *why* they had to be negative.

## Data & credits
Built entirely on public data from the [Gravitational Wave Open Science Center (GWOSC)](https://gwosc.org/), produced by the LIGO/Virgo/KAGRA collaborations. All SNRs are band-limited to [50, 1024] Hz; whitening is normalized so whitened-domain energy equals matched-filter SNR². Stack: Python · PyTorch · `gwpy` · `gwosc` · `pycbc` · `sbi`.

*Developed iteratively with Claude.*
