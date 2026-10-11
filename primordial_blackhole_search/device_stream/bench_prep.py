"""Device feasibility: per-sample data-prep cost of the real cnn_w64 pipeline (64-s windows), using pbh unchanged.
Steps: GWOSC fetch of one O3a H1 training segment -> whiten_segment -> N positives (sample_params,
make_whitened_injection, inject_into_window, model_input) and N negatives (model_input) as in build_dataset.py."""
import hashlib, json, sys, time
import numpy as np
from pbh import config as C
from pbh.data import fetch_segment, load_segment, whiten_segment
from pbh.spectrogram import model_input
from pbh.sweep import SweepGrid
from pbh.waveforms import inject_into_window, make_whitened_injection, sample_params

GPS, N = 1242072046, int(sys.argv[1]) if len(sys.argv) > 1 else 20
out = {}
t = time.perf_counter(); fetch_segment("H1", GPS); out["fetch_s"] = round(time.perf_counter() - t, 1)
raw, _ = load_segment("H1", GPS)
out["raw_md5"] = hashlib.md5(raw.astype("<f8").tobytes()).hexdigest()
t = time.perf_counter(); w, t0, psd = whiten_segment("H1", GPS); out["whiten_s"] = round(time.perf_counter() - t, 1)
out["whitened_std"] = float(np.std(w[C.WHITEN_CROP_SEC * C.SAMPLE_RATE: -C.WHITEN_CROP_SEC * C.SAMPLE_RATE]))
g = SweepGrid.short(64); n_win, n_time = g.win_samp, g.n_time_bins
crop = C.WHITEN_CROP_SEC * C.SAMPLE_RATE
lo, hi = crop, len(w) - crop - n_win
rng = np.random.default_rng(0)
tw, ti, ts, tries = [], [], [], 0
for _ in range(N):
    while True:
        tries += 1
        p = sample_params(rng)
        a = time.perf_counter(); h_w, snr_ref = make_whitened_injection(p, "H1", t0, psd); tw.append(time.perf_counter() - a)
        start = int(rng.integers(lo, hi)); merger_idx = int(rng.uniform(0.25, 1.30) * n_win)
        a = time.perf_counter(); window, in_snr = inject_into_window(w[start:start + n_win], h_w, snr_ref, float(rng.uniform(*C.TRAIN_SNR_RANGE)), merger_idx); ti.append(time.perf_counter() - a)
        if in_snr >= 4.0:
            break
    a = time.perf_counter(); x = model_input(window, n_time); ts.append(time.perf_counter() - a)
tn = []
for _ in range(N):
    start = int(rng.integers(lo, hi))
    a = time.perf_counter(); model_input(w[start:start + n_win], n_time); tn.append(time.perf_counter() - a)
med = lambda v: round(float(np.median(v)), 3)
out.update(shape=list(x.shape), positives=N, waveform_attempts=tries,
           waveform_s_median=med(tw), waveform_s_max=round(max(tw), 3), inject_s_median=med(ti),
           spectrogram_s_median=med(ts + tn),
           pos_sample_s=round((sum(tw) + sum(ti) + sum(ts)) / N, 3), neg_sample_s=round(sum(tn) / N, 3))
print("PREP_JSON " + json.dumps(out), flush=True)
