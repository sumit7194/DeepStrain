# device_stream — the noise-variety experiment that runs on the phone compute node

**Question (pre-registered):** does training `cnn_w64` on ~600 h of *fresh, never-repeated* O3a noise beat the same
number of training steps on the original 18 h? Read [PREREGISTRATION.md](PREREGISTRATION.md) first — it is the
contract, with amendments appended (never edited).

## Provenance — stated plainly
These files were **written 2026-10-07 → 10-09 and kept untracked** (in the git-ignored device folder
`antigravity/fervent-carson/deepstrain_stream/`) **until 2026-10-11**, when the user clarified that only *device-setup*
material should stay untracked. They are **byte-for-byte copies** of what the phone runs (verified against the copies
on the phone by sha256 at commit time); nothing was edited on the way in. **The pre-registration's integrity rests on
`FROZEN.sha256`, not on commit timestamps:** the hashes of `PREREGISTRATION.md`, `stream_train.py`, `bank.py` and
`stream_segments.json` were fixed on 2026-10-07 before the first optimizer step (`FROZEN.sha256.orig`) and amended
only for the appended amendments (current `FROZEN.sha256`; the trainer, bank and segment list hashes have never
changed). `run_queue.sh.pre_amend3` is the queue script before Amendment 3 (stall watchdog, 2026-10-09).

## Files
| File | Role |
|---|---|
| `stream_train.py` | the trainer (`--source stream|original --seed N`), per-segment atomic checkpoints, all draws keyed to (seed, segment) |
| `run_queue.sh` | pre-registered job order V0, C0, V1, C1–C4; offline mode (Amendment 2); stall watchdog (Amendment 3) |
| `bank.py`, `build_bank.py` | 600-entry face-on h+/h× chirp bank (built on the Mac); `bank.md5` = manifest of its 600 files |
| `stream_segments.py`, `stream_segments.json` | the two disjoint pools of fresh O3a H1 segments (leakage-guarded) |
| `prefetch_control.sh` | pre-fetches the 16 control segments (enables offline mode) |
| `test_bank.py`, `test_fastpath.py` | golden tests: bank path ≡ `make_whitened_injection`; fast path ≡ bank path |
| `bench_*.py` | phone timing benchmarks used for the budget in the pre-registration |
| `FROZEN.sha256`, `FROZEN.sha256.orig` | integrity hashes (current / as frozen 2026-10-07) |

The phone runs these against the repo's own `pbh/` library (the device copy is byte-identical to
`primordial_blackhole_search/pbh/`, checked 2026-10-11).

## Not committed (large or device-specific)
- `bank/` (6.2 GB, 600 npz) — rebuild on the Mac with `build_bank.py`, then check against `bank.md5`. `bank_test/`,
  `rs_*/`, `runs_smoke/` are reproducibility/smoke outputs (18–156 MB each), kept in the device folder.
- Run outputs (`runs/<run>/ckpt.pt, final.pt, segments.jsonl, log.txt`) live on the phone under `/root/pbhcode/runs/`;
  they will be summarised in RESULTS.md when scored.
- Device setup (init, flashing, Wi-Fi, SSH keys, props) stays in the untracked folder by the user's directive.

Environment: phone — see [../../ENVIRONMENT/README.md](../../ENVIRONMENT/README.md) (`phone_pbh_requirements.lock`, `phone_pbh_env.json`).
