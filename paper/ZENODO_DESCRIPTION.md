# Zenodo record: LongChirp preprint (DOI 10.5281/zenodo.24623652, reserved; registered when the Bridge publishes)

Sumit approved the release and CC BY 4.0 on 2026-11-28 (relayed by the Bridge). The Bridge uploads and publishes.

## Metadata
- **Resource type:** Publication / Preprint
- **Title:** LongChirp: an AI-assisted benchmark for searches of minutes-long subsolar-mass inspirals in real LIGO noise
- **Author:** Sumit Pal, Independent researcher, ORCID 0009-0000-9541-2793
- **Publication date:** set on upload
- **Version:** preprint v1 (describes software release `longchirp-v1.0.0` and its corrections in `longchirp-v1.0.1`)
- **Language:** English
- **License (approved by Sumit, 2026-11-28):** Creative Commons Attribution 4.0 International (CC BY 4.0), for the
  paper text and figure. The code stays MIT in the software record.
- **Keywords:** gravitational waves; LIGO; subsolar-mass binaries; primordial black holes; matched filtering; machine
  learning; convolutional neural network; benchmark; time-slide background; MLGWSC-1; GWOSC
- **Files (previewed first):** `longchirp_preprint.pdf` (set as the default preview), `SUMMARY.pdf`,
  `REFEREE_GUIDE.md`, `README.md`, `longchirp_preprint.tex`, `SUMMARY.tex`, `make_figure.py`, `fig_distances.pdf`,
  `SHA256SUMS.txt`. Upload these as separate files, not a ZIP: a ZIP is what makes
  the current records show "Too many files inside the ZIP file" instead of a preview.

## Related identifiers
| Relation | Identifier | What |
|---|---|---|
| isSupplementedBy | 10.5281/zenodo.24623628 (DOI) | DeepStrain software, release longchirp-v1.0.1: code, data definition, results, bootstrap script |
| isSupplementedBy | 10.5281/zenodo.24399603 (DOI) | DeepStrain software, release longchirp-v1.0.0 (the version the paper's corrections refer to) |
| references | 10.1103/PhysRevD.107.023021 (DOI) | Schäfer et al., MLGWSC-1 |
| references | 10.3847/1538-4365/acdc9f (DOI) | GWOSC O3 open-data release |
| references | 10.1103/PhysRevLett.129.061104 (DOI) | LVK, O3a subsolar-mass search |
| references | 10.1093/mnras/stad588 (DOI) | LVK, O3b subsolar-mass search |
| references | 10.1103/PhysRevLett.127.151101 (DOI) | Nitz & Wang, O3a subsolar-mass search |
| references | arXiv:2602.12115 | Kacanja et al., O4a subsolar-mass search |
| references | arXiv:2412.10951 | Hanna et al., O4 subsolar-mass template bank |
| references | 10.1088/0264-9381/33/21/215004 (DOI) | Usman et al., PyCBC search |
| references | 10.5281/zenodo.596388 (DOI) | PyCBC software |

Each DOI and arXiv number was resolved at its source (Crossref, DataCite, arXiv API) on 2026-11-28. For O3b, the
arXiv entry lists the later correction (10.1093/mnras/stad3120), and the original paper is stad588. Both are cited.

## Description (paste into the record)

**Result.** On 1,344 simulated subsolar-mass binary inspirals added to 27 hours of real LIGO O3a data (Hanford and
Livingston), a semi-coherent matched-filter bank reaches a sensitive chirp distance of 57.5 Mpc at a false-alarm rate
of one per day. A convolutional neural network reaches 48.4 Mpc, or 25.5 Mpc if it must time each merger to within
one second. Both are about one third of the 155 Mpc reached by an ideal detector that knows each signal's exact
waveform (network SNR ≥ 8). The bank leads the network by a factor 1.19 in distance (90 % interval 1.10–1.29,
resampling injections only). These are numerical results from one time-slide background realisation.

**Method.** LongChirp is a fixed benchmark for signals that stay in band for minutes:
- 24 test and 8 development segments of GWOSC O3a strain (4096 s each);
- a committed table of non-spinning TaylorF2 injections (component masses 0.2–1.0 M☉, chirp distance 5–500 Mpc);
- a 27.3-day background from cross-segment time slides;
- an evaluator using MLGWSC-1's submission format and found rule, with the timing tolerance capped at 60 s and a
  chance-match bound reported.

Three yardsticks were scored on it: an ideal-SNR ceiling, a 3,235-template semi-coherent bank, and a CNN.

**Verification.** Every check is an automated test committed with the release:
- the evaluator against synthetic submissions with known answers;
- the injection path against PyCBC (SNR to 9×10⁻¹⁰);
- each downloaded file against GWOSC's published statistics (to 5.6×10⁻⁸);
- the bank pipeline against float64 and whole-series references.

The paper records a failed first version of one test, and five discrepancies between the release documentation and
its own result files.

**Limits.**
- One background realisation; the thresholds' uncertainty is not in the intervals.
- One per week at best, not one per month.
- O3a H1/L1 only, non-spinning signals only.
- The bank is not a fully coherent search.
- One machine.
- Release v1.0.0 lacked the bootstrap script behind the ratio interval. v1.0.1 adds it, rebuilt, and it reproduces
  the committed output byte for byte.

No detection or rate limit is claimed.

**AI assistance.** This work was AI-assisted. Anthropic's Claude models wrote most of the code, ran most of the
analyses and drafted the paper, under the author's direction. The author chose the questions, reviewed the work and
takes responsibility for it. Checks were automated tests plus a test run by a separate AI session. This is not human
peer review, and no external expert has read the work yet.
