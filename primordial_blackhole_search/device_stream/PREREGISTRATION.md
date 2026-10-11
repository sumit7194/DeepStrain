# Pre-registration — does noise *variety* improve the subsolar CNN?  (FROZEN 2026-10-07, before any training step)

Lives in the git-ignored device folder by user directive. Frozen = this file's sha256 is recorded in FROZEN.sha256
before the first optimizer step of any run below. Changes after that are amendments, appended with a reason.

## Question
`cnn_w64` was trained on 16 O3a H1 segments (~18 h of noise). Does training the same network on a stream of
**fresh, never-repeated** O3a noise (~600 h per run) improve detection at a fixed false-alarm threshold, beyond the
same number of training steps on the original 18 h?

Why it is open: L6 showed the *self-supervised* pool saturates at 2.5k spectrograms; supervised noise variety was never
tested. Our response probe says false alarms are band-power fluctuations near ~110 Hz, which more noise examples could
teach the tail of — or not, since G2b showed higher AUC need not move the tail.

## Arms — differ ONLY in the noise source; both run on the phone (identical hardware and numerics)
| | Arm V (variety) | Arm C (control) |
|---|---|---|
| Noise | O3a H1 segments from `stream_segments.json`, each used once then deleted; V seed s uses pool `s % 2` (pools disjoint) | the 16 cnn_w64 train segments, recycled in a seeded order |
| Shared | SpectrogramCNN from scratch; AdamW lr 3e-4 wd 1e-3; cosine over all steps; batch 64; BCE; train.py augmentation (seeded); 64-s windows; 1,024 samples/segment = 512 injections + 512 noise | same |
| Injections | 600-entry chirp bank (`bank/`, masses log-uniform [0.2,1] M☉ each); K = 20 entries whitened per segment against that segment's PSD; sky/orientation/SNR U(5,30)/merger position as build_dataset.py, in-window SNR ≥ 4 | same |
| Seeds | **0, 1** | **0, 1, 2, 3, 4** |

At equal seed, every random draw (bank picks, injection parameters, window starts, ordering, augmentation, dropout) is
`default_rng([seed, segment_index])`, so V-s and C-s see identical injections and differ only in noise.
Run length: 586 segments × 1,024 = 600,064 samples = 9,376 steps (≈ cnn_w64's 640k sample-steps).

Verified before freezing: bank path ≡ make_whitened_injection (rel. 1.7e-8, SNR 2e-10); fast path ≡ bank path (1e-16);
a killed-and-resumed run is bit-identical to an uninterrupted one, and reruns are bit-identical.

## Leakage guards
- Stream excludes the 16 train segments (V disjoint from C's noise), and anything within ±12 h of the 2 val + 6 test
  segments. Selection otherwise = `pbh.data.discover_segments` rules (H1_DATA, 256-s margin, GWTC-2.1 events ±64 s).
- Evaluation injections use fresh continuous masses (track_eval), so no evaluation chirp is a bank chirp.

## Metric and decision rule
- **Primary:** mean over the 3 mass bins of the sensitive-distance fraction, statistic `max`, from
  `scripts/track_eval.py --window-sec 64 --weights <run>` on the fixed 6 test segments — the exact protocol of cnn_w64's
  committed 0.407 / 0.457 / 0.476. **Golden gate first:** re-running it on `cnn_w64` must reproduce those numbers;
  if it does not, nothing is scored until the discrepancy is explained.
- Effect: Δ = mean(V) − mean(C);  z = Δ / sd(C over seeds 0–4).
- **|z| < 2 ⇒ "no detectable variety effect";  z ≥ 2 ⇒ variety helps;  z ≤ −2 ⇒ variety hurts.**
- Paired check (reported, not decisive): V-s − C-s for s = 0, 1 (same injections, same init).
- Secondary (reported only): val AUC; noise-score tail quantiles on the test segments.
- Known shared caveat: the zero-FA threshold over 384 non-overlapping test windows is coarse (CLAUDE.md, rung 2); it is
  identical for every run, so it cannot favour one arm.

## Budget (measured on the phone 2026-10-07)
Per segment: fetch ≈ 417 s (network; overlaps compute via prefetch), whiten 81 s, filter 7 s, 20 × 4.4 s entries,
512 × 0.128 s injections, 512 × 0.066 s noise windows, 1,024 × 0.057 s training ⇒ ≈ 340 s CPU ⇒ ≈ 2.3–3 days per run.
Order: V0, C0, V1, C1, C2, C3, C4 (≈ 17–20 days total; the user said time is not a constraint).

## Prediction (mine, written before running)
|z| < 2 most likely (~60%); variety helps (~35%) because the false alarms are a noise-distribution property; hurts (~5%).

---
Amendments (after freeze): none yet.

**Amendment 1 (2026-10-07 ~12:25 IST, after 2 of 586 segments of stream_s0; infrastructure only).**
The trainer hit `MemoryError` at segment 3: the next segment's background download (gwpy holds several full copies)
overlapped whitening of the current one, and the two peaks exceeded the phone's RAM. Change: the next download now
starts *after* the current segment is whitened, and a crash's stub `.tmp.hdf5` is removed before re-fetching. No
numeric or random-draw change: a 3-segment run with the amended trainer is bit-identical to the frozen code's run.
Crashes never affected results anyway (all draws are keyed to (seed, segment index); a crashed segment is redone
identically from the last checkpoint). FROZEN.sha256 updated for stream_train.py; the other hashes are unchanged.

**Amendment 2 (2026-10-07 ~15:00 IST, during stream_s0; scheduling only).** The phone now reaches the internet over its
own Wi-Fi (USB/Mac as fallback). To use internet outages productively, `run_queue.sh` gained an offline mode: if a
fresh-noise job fails while the internet is down and the 16 control segments are on disk (pre-fetched once by
`prefetch_control.sh`), it works on the next unfinished control run for up to 30 min, then re-checks. This changes
only the ORDER in which segments are computed; each run's segments are deterministic in (seed, segment index) and
checkpointed atomically, so every run's weights are identical to running the queue strictly in the frozen order. The
trainer and all frozen files are unchanged (FROZEN.sha256 updated only for this file, PREREGISTRATION.md).

**Amendment 3 (2026-10-09 ~15:45 IST, during stream_s0 at segment 506/586; scheduling only).** stream_s0 sat at 506 for
78 min: after this morning's Wi-Fi drops, a GWOSC download left an ESTABLISHED TLS socket with no traffic (66 bytes in
15 s, trainer at 1% CPU in poll) and no read timeout, so it would have waited forever; the queue had no stall
handling for non-offline jobs. Killed by exact PID at 10:08 UTC (between segments: 507 had not started); the queue
retried from the 506 checkpoint as designed. Change: `run_queue.sh` now runs each job under a watchdog — kill (TERM,
then KILL, whole process group) and retry from the checkpoint when **no new segment for 45 min AND < 1 MB received on
wlan0+usb0 over the last 15 min**, so a slow-but-moving download is never killed (longest normal segment observed:
980 s). Tested on the phone with fake trainers: hang → killed, child processes included; exit 0 / exit 3 passed
through; steadily progressing job not killed; hung job with live network traffic not killed. The trainer and all
frozen files are unchanged and every run stays deterministic per (seed, segment) with atomic checkpoints, so results are
identical to the frozen order. FROZEN.sha256 updated only for this file.
