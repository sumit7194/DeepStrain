# DRAFT — note to the authors of arXiv:2406.11986 (METRICS, scalar-Gauss-Bonnet case)

*Status: **SENT 2026-11-17 21:54 IST** to A. K.-W. Chung (Cambridge), from Sumit's Gmail — per The Bridge, sent by a
bridge mistake (a keystroke in compose taken as Send) BEFORE Sumit had reviewed the final email; Sumit informed. Content =
this draft (4089b28) adapted to Sumit's voice, with an AI-assistance disclosure and the script link. Any reply: record it
here. All three points were checked against arXiv **v2** (last revised 7 Sep 2024); the published PRD version
(PRD 110, 064019) was **not** checked.*

---

**Subject:** Three small corrections to "Quasi-normal mode frequencies and gravitational perturbations of black
holes with any subextremal spin in modified gravity through METRICS: the scalar-Gauss-Bonnet gravity case"
(arXiv:2406.11986v2)

Dear authors,

Thank you for METRICS and for publishing the supplementary notebook and the Appendix-B tables. They let us
reproduce and use your results in detail. While doing so we noticed three points in the text of arXiv v2 that look
like typographical or transcription errors. None of them affects your computed frequencies or tables, as far as we
can tell. We report them in case a correction is useful. A short script reproduces each point from the LaTeX sources
of your paper and of Pierini & Gualtieri (link below).

**1. GR horizon angular velocity (equation for Ω_H^(0), Sec. on horizon quantities).**
- The text prints Ω_H^(0) = a/(2Mb), with b = √(1−a²).
- For the Kerr background in Boyer–Lindquist coordinates, Ω_H = −g_tφ/g_φφ at r₊ = a/(2M r₊) = a/(2M(1+b)). The
  printed form is larger by the factor (1+b)/b, e.g. 0.477 instead of 0.200 at a = 0.69 (M = 1).
- The adjacent κ^(0) = b/(2M(1+b)) is correct, so we suspect a dropped "(1+".
- The supplementary notebook appears to use the correct value: the horizon-rigidity relation
  Ω^(1) = Ω^(0)(H2 − H4) at r₊ holds to ~1e−40 with a/(2M(1+b)), and does not hold with the printed form.

**2. Sign of one coefficient in the reprinted Pierini–Gualtieri fit (Eq. `freq_Gultierai`).**
- The 033 polar linear-in-a term is printed as (−1.03488 − 0.05664 i) a.
- Rebuilding it from Pierini & Gualtieri, arXiv:2207.11267 (table with r = 1, row C₂, Im l = 3: +1.18×10⁻³) with your
  conversion (×16 for the coupling normalisation in your footnote, ×m = 3) gives **+0.05664 i**.
- The other eight reprinted coefficients match PG's tables in sign and value, so this looks like an isolated sign
  slip.

**3. The truncated fitting polynomials (Eq. `METRICS_fitting_polynomials`) and the comparison with PG.**
- The printed w_j are coefficients of a degree-8 least-squares fit over the spin range. They are not the
  small-spin Taylor coefficients.
- Truncated at a⁴ as printed, the polar 022 polynomial departs from your own Appendix-B table of 022 frequencies by 16%
  at a = 0.1 and by 383% at a = 0.2 (021P: 2% / 61%; 033P: 1% / 21%). Here the relative difference is
  |ω_fit − ω_table| / |ω_table| on the complex frequency. Split by part, 022P is Re 16.0% / Im 11.7% at a = 0.1 and
  Re 382% / Im 400% at a = 0.2.
- Its w₁ and w₂ (−2.518 − 0.411i, 48.47 + 9.60i) are also far from PG's (−0.335 + 0.001i, −0.320 + 0.304i). The text
  says that "the w_{j=0,1,2} of our fitting polynomials are close to the coefficient shown above"; that holds for w₀
  only.
- A sentence noting that the truncated polynomials should not be used as a low-spin series, and that the
  comparison with PG applies to w₀ (or to the tabulated frequencies), might help readers.

Reproduction: `ringdown_spectroscopy/scripts/42_metrics_errata_check.py` in
https://github.com/sumit7194/DeepStrain (needs only `mpmath`; output in
`ringdown_spectroscopy/results/42_metrics_errata_check.json`).

With thanks and best regards,
[name]

---

### Internal notes (not part of the message)
- Script run 2026-11-17 on the fresh arXiv sources:
  - **2406.11986v2** main.tex, sha256 `d336069790e65043…`;
  - **2207.11267v2** paper.tex, sha256 `418076f287c4c0a5…`.
- All three points were reported CONFIRMED.
- The earlier full analyses (`38_sgb_supplement.py`, `39_sgb_qnm_truncation.py`) re-ran on the same sources and
  reproduced their committed results byte for byte.
- **Bridge cross-check (V18, TheBridge 571c1a0) — reconciled 2026-11-17:**
  - E1 and E2 were confirmed independently.
  - E3 was confirmed in substance, but with 9.3% / 431%. Those are exactly what the 022P fit gives against the
    **021** table (computed: 9.3% / 431.5%).
  - The Bridge's suspected "Tables 1 and 2 swapped" is NOT in main.tex. Each `table*` carries its own caption and
    label AFTER its tabular: 022 → a=0.005 polar −0.2266−0.0752i, 033 → −0.8787−0.1150i, 021 → −0.2258−0.0753i,
    consistent with PG at a = 0 and with the fits' w₀.
  - A reader pairing each caption with the NEXT table (easy in a rendered view where the caption sits below the data)
    sees exactly the reported offset. ⇒ No fourth erratum; our E3 numbers stand, now with the measure stated.
- Scope: text errata only. Our measured truncation errors (the companion note) are a separate item.
