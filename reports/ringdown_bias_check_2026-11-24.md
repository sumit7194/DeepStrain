# Verification of Neon's ringdown deep-research answer (T57), DeepStrain, 2026-11-24

**Reading only; nothing was run.**

**Sources:**
- arXiv API, plus full text of 2603.19021 and 2607.19293 (PDF via pdftotext) and of 2607.05486 and 2607.07070 (HTML);
- the Zenodo API (record metadata and the GWTC-5 README);
- the LIGO DCC public page.

## 1. Identifiers

| ID | Exists | Checked claim |
|---|---|---|
| 2603.19019 | ✓ GWTC-4.0 TGR I, overview | 91 events, 42 from O4a |
| 2603.19021 | ✓ GWTC-4.0 TGR III, remnants | ✓ 98.6 (+1.4/−9.4)% joint, 99.3 (+0.7/−4.5)% hierarchical, from 93.8/94.9% (GWTC-3). ✓ **Verbatim in the abstract: "Since the significance also decreases to 92.2% (96.2%) when including the more recent very loud event GW250114, there is no strong evidence for a GR deviation."** |
| 2607.05486 Kyutoku | ✓ | ✓ Eq. 16: ⟨ρ²δτ⟩/τ = 2n_A − 4n_τ + (3n_A − 3n_τ)/(2Q²) + …, i.e. ≈ +0.02 at ρ = 10 for log-flat A. ✓ Cites pSEOBNR as motivation. ✓ Caveat: investigations "should adopt more realistic waveform models and analysis pipelines". The mitigation n_τ = (n_A + 3)/2 was **not located** in the text. |
| 2310.03811 | ✓ Pacilio, Gerosa & Bhagwat | catalog variance, bootstrap mitigation |
| 2504.10130 | ✓ Pompili et al., pSEOBNRv5PHM | missing precession can fake deviations; 12 events reanalysed (numbers not checked) |
| 2607.07070 | ✓ Chandra & Calderón Bustillo | mixture model on public O1–O4a FTI / pSEOBNR / KerrPostMerger: consistent with all-GR; B ≈ 20 / 10 / 15. Does not cite Kyutoku. |
| 2607.19293 | ✓ GWTC-5.0 TGR | **See §4: the tension is NOT stale.** |
| "Chiaramello PRD" | ✓ arXiv:2511.19593, PRD 114, 044004 (2026) | eccentric + precessing pTEOBResumS, theory-agnostic tests |
| LIGO-P2600130-v1 | ✓ GWTC-4.0 TGR III data release | pSEOBNR.tar.gz 14.2 MB, pyRing.tar.gz 1.8 GB, QNMRF 581 kB, env yml 21.5 kB ✓ |
| Zenodo 21454847 | ✓ GWTC-5.0 TGR data release (2026-07-22) | pSEOB.tar.gz **711.3 MB**, RD 415.3 MB, QNMRF 1.2 MB. README: "data products necessary to reproduce the figures". |
| Zenodo 18600070 | ✓, but the title is *GW240925 and GW250207: Astrophysical Calibration of GW Detectors* | pseobnr.tar.gz 80.2 MB. It is a calibration paper's release, not a TGR release. |

## 2. "No one has tested Kyutoku on pSEOBNR"

**As far as I can find, true.**
- I searched the arXiv API and a web search.
- GWTC-5.0 TGR and 2607.07070 do not cite Kyutoku.

**But the premise underneath it is weaker than Neon says**, for two reasons.

**LVK already ran the mock catalogs** (2603.19021, Fig. 6 and its text):
- 1000 bootstrapped catalogs of 17 GR signals, NRSur7dq4 and SEOBNRv5PHM, in zero and Gaussian noise, recovered with
  pSEOBNRv5PHM under the analysis priors.
- Any finite-SNR or prior-gradient bias of Kyutoku's kind is **automatically inside** those mocks.
- They do reproduce "damping time shifted to positive values".
- Yet quantiles as large as observed occur in only **~0.7% (joint), 0.2% (1D hierarchical) and 3.5% (4D)** of
  catalogs.
- Caveats: GWTC-3 population, O4 design curves, 17 signals.

**LVK also names the mechanism in words.**
- "a systematic tendency for events with relatively low SNR to yield more samples with positive deviations",
  attributed to a mass–deviation degeneracy.
- Prior/population interaction with noise fluctuations.
- Restricting to the 11 events with log₁₀ B_noise > 8 reduces the positive bias.

⇒ **An analytic Kyutoku-on-pSEOBNR calculation would attribute an effect that LVK's own mocks already contain and
found insufficient.** That attribution is new, but its information value is modest.

## 3. Feasibility on this Mac

**Step B (beyond-Fisher bias with pyseobnr): "minutes per event" is doubtful.**
- O(ρ⁻²) needs second and third waveform derivatives in ~17 parameters (15 GR + δf̂, δτ̂). That is hundreds to
  thousands of pSEOBNRv5PHM calls per event. The per-call cost is unmeasured; at ~1 s per precessing call (an
  estimate) it is tens of minutes to hours per event.
- Finite differences of an ODE-integrated EOB waveform are numerically noisy.
- The precessing Fisher matrix is ill-conditioned; we saw cond ~1e15 in our own ringdown Fisher work.
- The expansion is asymptotic and least trustworthy exactly at ρ_RD ≈ 8–10, where the bias lives.
- Feasible in hours, but fragile.

**Step A (public posteriors): unknown until a tarball is opened.**
- The GWTC-4 pSEOBNR tarball is only 14.2 MB for ~19 events, which suggests reduced samples, possibly only δf̂ / δτ̂.
- The GWTC-5 tarball (711 MB) is "what the figures need" and may hold full posteriors.
- d_L samples and the exact priors are not confirmed.
- ρ_RD per event is not a sampled parameter; it would come from the paper's tables.
- Opening either is a download, which needs Sumit's OK.

## 4. Correction to Neon's risk (1): the tension is NOT stale

- 92.2 / 96.2% comes from adding GW250114 alone to GWTC-4.
- **GWTC-5.0, all O4b events included (Table 9 of 2607.19293):** pSEOBNR Q_GR **98.1 (+1.9/−6.9)% joint, 98.7
  (+1.3/−13.1)% hierarchical** (μ_δτ̂), and 92.5% in the 4-D hierarchical.
- The GWTC-5.0 abstract: "the GR prediction lies in the tails of the combined results, possibly due to the limited
  catalog size".
- So the question is live in the newest catalog.

## 5. Does a narrowed study add something, and recommendation

**What is left after LVK's mocks, their low-SNR remarks, and the mixture model:**
- **(i) Step A alone:** how much the GWTC-5 combined quantile moves under prior reweighting (amplitude / distance
  priors, or Kyutoku's n_τ fix). Cheap, and genuinely useful as a sensitivity number, if the samples carry the needed
  columns.
- **(ii) Step B:** an analytic attribution. Fragile, and modest in value.

**Recommendation for Sumit:**
- **Not the full narrowed plan.** Do Step A only, as a ≤ 1-day gate.
- **Cheapest first step:** with Sumit's OK, download the 14.2 MB GWTC-4 pSEOBNR tarball and read its README and
  column list: d_L? masses? priors? per-event samples? Nothing else until that answer is in.
- **If the columns are there:** reweight, and recompute the joint and hierarchical quantiles.
- **If the columns are missing, or reweighting barely moves the quantile:** stop the ringdown line, and choose
  between revised LongChirp on fresh segments and neither.
- **Step B** only if A shows the quantile is prior-sensitive.
