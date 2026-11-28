# LongChirp preprint: record bundle (DRAFT)

A short paper and a plain summary describing the LongChirp benchmark as released in DeepStrain `longchirp-v1.0.0`
(software DOI [10.5281/zenodo.24399603](https://doi.org/10.5281/zenodo.24399603)). The paper covers only what that
release contains and verified. Work in progress after the release is not included.

| File | What it is |
|---|---|
| `longchirp_preprint.pdf` | The paper (7 pages): benchmark, three yardsticks, verification, limits, AI-assistance statement |
| `SUMMARY.pdf` | One page for non-specialists, including what we do not claim |
| `REFEREE_GUIDE.md` | Each claim → the file holding its number → the command that regenerates it → expected output → run time |
| `SHA256SUMS.txt` | Checksums of the files above |

Sources (in the git repository, not in the record):
- `longchirp_preprint.tex` and `SUMMARY.tex`, compiled with tectonic 0.17.0;
- `make_figure.py`, which draws Figure 1 from the release tag through `git show`, so it cannot drift from the
  release. Run it with `primordial_blackhole_search/.venv/bin/python paper/make_figure.py`.

Author: Sumit Pal (ORCID 0009-0000-9541-2793), independent researcher. AI-assisted: Anthropic's Claude models wrote
most of the code and drafted these documents under the author's direction. No external expert has reviewed this yet.

Check the files with `shasum -a 256 -c SHA256SUMS.txt`.
