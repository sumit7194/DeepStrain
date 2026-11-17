# Environments — what produced our numbers (fleet reproducibility policy, 2026-10-11)

Our code is used as evidence, so every number must be re-creatable. One pinned environment per place code runs,
plus a stamp of what affects numerics. Re-stamp with `python tools/envstamp.py ENVIRONMENT/<name>_env.json`
(adapted from TheBridge commit 243ec82).

| Where | Lock (exact packages) | Stamp | Numerics notes |
|---|---|---|---|
| Mac, subsolar (`primordial_blackhole_search/.venv`) | `mac_pbh_requirements.lock` (= `primordial_blackhole_search/requirements-lock.txt`, verified identical 2026-10-11) | `mac_pbh_env.json` | Python 3.12.13 (Homebrew), macOS 26.5 arm64, numpy/scipy on **Accelerate**, torch 2.12.0 (MPS available; 4 intra-op threads by default), float32 nets |
| Phone compute node (noise-variety experiment) | `phone_pbh_requirements.lock` | `phone_pbh_env.json` | same package set except `torch==2.12.0+cpu`; **OpenBLAS / Arm Compute Library**, 8 threads. Different BLAS from the Mac ⇒ last-digit float differences (measured agreement ~7 decimals); bit-identical *on the phone* across reruns and kill/resume |
| Mac, echoes (`echoes/.venv`, deleted 2026-10-03) | `echoes/requirements-lock.txt` | — (venv gone; stamp on rebuild) | Python 3.14 |
| Mac, ringdown (`ringdown_spectroscopy/.venv`, deleted) | `ringdown_spectroscopy/requirements-lock.txt` | — | Python 3.12 |
| Mac, ringdown package (`.venv311`, deleted) | `ringdown_spectroscopy/requirements-lock-py311.txt` | — | Python 3.11, jax 0.4.35 x64 |
| Google Cloud L4 GPU VM `alphaludo-l4` (Build C, C-2, 2026-06-20) | **not recorded** — VM deleted | — | CUDA; exact versions unrecoverable. Results from it are gated on saved artifacts only |

## Known numerics hazard (2026-10-11; resolved)
A vendor numerical-library defect, reported by the fleet and (per the vendor) fixed in macOS 27.0.1. **Audit of our
code: not affected; regression on 27.0.1: identical** (see the OS-update regression section below). General note: on the
Mac, torch's CPU linear algebra uses the same vendor library as numpy/scipy (`LAPACK_INFO=accelerate`), so switching to
torch is not an independent cross-check there; an OpenBLAS environment (the phone, or a Linux venv) is.

## Back-fill policy for finished results
Results committed before 2026-10-11 carry no environment line. Back-filled rule: **unless an entry says otherwise, a
Mac result used the sub-project's venv whose package set is the committed `requirements-lock*.txt`** (frozen
2026-10-03 from the venvs as they then stood). Caveat stated plainly: package versions at the *time* of an older run
are not separately recorded — the lock is the last known state, not a proof of the earlier one. Build C / C-2 ran on
the L4 VM (above). Each RESULTS / lab notebook carries a one-line pointer to this file, marked "back-filled".

## How to reproduce

**Mac (any sub-project):**
```bash
cd primordial_blackhole_search   # or echoes/, ringdown_spectroscopy/
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements-lock.txt
# (echoes: --python 3.14; ringdown package: uv venv --python 3.11 .venv311 + requirements-lock-py311.txt)
cd .. && ./verify.sh          # regression gate: must be green before any number is trusted
python tools/envstamp.py      # record what you actually ran on
```
Then run the script named in the RESULTS entry (each entry names its script and artifact).

**Phone (noise-variety experiment):** code + pre-registration are committed in
[`primordial_blackhole_search/device_stream/`](../primordial_blackhole_search/device_stream/README.md). On a Debian 13 arm64
box (Python 3.12.13 via uv):
```bash
uv venv --python 3.12 /root/pbh/.venv && uv pip install --python /root/pbh/.venv/bin/python -r ENVIRONMENT/phone_pbh_requirements.lock
mkdir -p /root/pbhcode && cp -r primordial_blackhole_search/device_stream/* primordial_blackhole_search/pbh /root/pbhcode/
# build the 600-entry bank on the Mac (device_stream/build_bank.py), copy bank/ over, verify against bank.md5
cd /root/pbhcode && sha256sum -c FROZEN.sha256 && ./run_queue.sh
```
Device setup (custom init, Wi-Fi, SSH) is deliberately not in the repo. Any aarch64 Linux box with the same lock is
expected to reproduce the phone's weights bit-for-bit (untested off the phone); the Mac agrees to ~7 decimals (measured).

## OS-update regression: macOS 27.0.1 (26A434), checked 2026-10-24
Packages unchanged (numpy 2.4.6, scipy 1.17.1, torch 2.12.0; pure wheels, nothing compiled locally). Re-ran known
results against pre-update outputs, single-threaded:
- **MPS** (cnn_w64 scoring all 384 O3a test noise windows vs the pre-update `results/track_w64/` cache, itself made on
  MPS): **bit-identical, 384/384**; zero-FA threshold 1.8170 unchanged.
- **numpy/pandas** (`device_stream/verdict.py`, `posthoc_threshold_check.py`): output JSONs **bit-identical** to the
  committed ones.
- **CPU torch** vs the same MPS cache: max |Δ| 4.4e-6 (25/384 bit-equal). That is a cross-DEVICE difference with no
  pre-update CPU reference, so it is not attributable to the OS. It is 2e-6 of the threshold, so it does not matter.
Peak RSS: scoring 3.1 GB, verdict 35 MB, post-hoc 126 MB. Details: `regression_macos27.0.1_26A434.json`.
New results from here are tagged `macOS 27.0.1 (26A434)` in their Environment line.

## OS-update regression: macOS 27.2 (26B123), 2026-11-17 — PARTIAL
- Python environment rebuilt from `primordial_blackhole_search/requirements-lock.txt`; `uv pip freeze` identical to the lock.
- **numpy/pandas path** (`device_stream/verdict.py`, `posthoc_threshold_check.py`): output JSONs **bit-identical** to the committed ones.
- **MPS/CPU torch path: NOT RUN — blocked on data.** The 6 O3a test segments (deleted in the Oct 28 cleanup) could
  not be re-downloaded. GWOSC's API answers (HTTP 200, 0.8 s), but every 4096-s strain file transfer is cut short
  (`ContentTooShortError`: 5.5 of 41.7 MB received), on all 6 segments, after 3–8 min each. That is a server-side
  transfer failure, not ours. The check script is ready (`ENVIRONMENT/regress_check.py`, fetch: `fetch_test_segments.py`);
  re-run when GWOSC transfers complete.
