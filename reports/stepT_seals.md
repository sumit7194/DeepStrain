# Step T: ledger of sealed predictions

*DeepStrain [model: Claude Opus 5.5], 2026-11-26.*
- Records where each sealed prediction of step T lives and how to check that it was not changed.
- **Neither prediction is opened or edited before G0 is evaluated** (`PREREG_coherent2.md` §6b; registration §11).
- Grading follows `reports/stepT_blind_prediction_brief.md` §4. It is applied identically to both predictors.

## 1. Fable (independent, blind)

| Field | Value |
|---|---|
| Commit | `b13b4f8e1612eef3b0fe2db57b0deef269a5831c` |
| Subject | `[model: Fable 5.2] SEALED: independent blind prediction of step T (R_tune, G0, per-gate, configuration)` |
| Parent | `d01cb8daa8d7544a9ff930d2d0938e19f76b7e67` (the brief, on main) |
| Commit time | 2026-11-26 04:03:38 +0530 |
| Branch | `fable/2026-11-26-stepT-prediction` (on origin) |
| File | `reports/stepT_blind_prediction_fable.md` (the commit's only change) |
| Blob | `0a258c2b914dc3b2f624b180bb7503f148418149` |
| sha256 | `0f78841de6771dfed7d58d8e95f0f221ed89398bca685844d7ec7ef978b98eb7` |

DeepStrain checked this from git metadata and a hash of the blob, without reading the file:
- the commit exists on origin;
- its parent is the brief;
- it changes this one file;
- the recomputed sha256 matches the value Fable gave the Bridge.

## 2. DeepStrain (the registration's own)

| Field | Value |
|---|---|
| Location | `reports/stepT_registration.md` §7, "Sealed prediction" |
| Commit | `d4c3ca4eed801e871b04386e4781f00d4ba3780f`, 2026-11-26 01:08:58 +0530 (before the brief) |
| File sha256 (whole registration) | `2fb6b8d452dea4bc262d40e0c9c47d129c690f4cb3a60a777a486e52ad17670b` |
| §7 sha256 (from the "## 7." heading up to and including the "## 8" heading line, as printed by `sed -n '/## 7. Sealed prediction/,/## 8/p'`) | `7b9690d0a029cc8a01671da7aefb11e81120c7855ec4c7d25fa717a62331de78` |

**What it covers:** R_tune (point and 80% interval) and P(G0) only. It gives no per-gate probabilities and no
configuration probability. Under the brief's rules those items are graded for Fable alone; for DeepStrain they are
recorded as "no prediction", not as a miss.

## 2b. Exposure note (2026-11-26, Deepstrain-personal)

- **What happened:** On 2026-11-26, at about 11:55 IST, Deepstrain-personal read its office twin's session transcript
  when it took the project over. That transcript quotes DeepStrain's §7 numbers.
- **The seal came first:** Fable's prediction was sealed at 04:03:38 the same day, in commit `b13b4f8`. Its content
  was fixed before the exposure.
- **Status:** Fable's prediction stays **blind**. The only effect is that the predicting session now knows the other
  prediction; nothing sealed changed.
- Recorded on the Bridge's instruction.

## 3. How to check, after G0

```bash
git show b13b4f8e1612eef3b0fe2db57b0deef269a5831c:reports/stepT_blind_prediction_fable.md | shasum -a 256
```

```bash
git show d4c3ca4:reports/stepT_registration.md | sed -n '/## 7. Sealed prediction/,/## 8/p' | shasum -a 256
```

Both must print the sha256 values above. Then grade:
- **R_tune:** HIT if the measured value lies in the 80% interval. Also report |point − measured|.
- **G0 and each gate:** Brier score. Gates that were never evaluated are left out.
- **Configuration:** log score.

Environment: Mac (macOS 27.2, arm64) · git metadata and shasum only, no computation · git d01cb8d + merge of b38c5bc
