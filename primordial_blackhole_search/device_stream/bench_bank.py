"""On-device cost of the bank path: per-segment whiten_pair per entry, then per-sample mix+inject+spectrogram."""
import json, time, glob
import numpy as np
from pbh import config as C
from pbh.data import whiten_segment
from pbh.spectrogram import model_input
from pbh.sweep import SweepGrid
from pbh.waveforms import inject_into_window, sample_params
from bank import projected_whitened, whiten_pair

w, t0, psd = whiten_segment("H1", 1242072046)
pairs, tw, sizes = [], [], []
for f in sorted(glob.glob("bank_test/w*.npz")):
    z = np.load(f); sizes.append(len(z["hp0"]))
    a = time.perf_counter(); pairs.append(whiten_pair(z["hp0"], z["hc0"], psd)); tw.append(time.perf_counter() - a)
g = SweepGrid.short(64); n_win, n_time = g.win_samp, g.n_time_bins
crop = C.WHITEN_CROP_SEC * C.SAMPLE_RATE; lo, hi = crop, len(w) - crop - n_win
rng = np.random.default_rng(0); tm, tries = [], 0
for _ in range(200):
    a = time.perf_counter()
    while True:
        tries += 1
        p = sample_params(rng); wp, wc = pairs[int(rng.integers(len(pairs)))]
        h_w, snr_ref = projected_whitened(wp, wc, p, "H1", t0)
        start = int(rng.integers(lo, hi)); merger_idx = int(rng.uniform(0.25, 1.30) * n_win)
        window, in_snr = inject_into_window(w[start:start + n_win], h_w, snr_ref, float(rng.uniform(*C.TRAIN_SNR_RANGE)), merger_idx)
        if in_snr >= 4.0:
            break
    model_input(window, n_time); tm.append(time.perf_counter() - a)
print("BANKBENCH_JSON " + json.dumps(dict(
    whiten_pair_s_median=round(float(np.median(tw)), 3), whiten_pair_s_max=round(max(tw), 3),
    entry_seconds=[round(s / C.SAMPLE_RATE) for s in sizes],
    pos_sample_s_median=round(float(np.median(tm)), 4), pos_sample_s_mean=round(float(np.mean(tm)), 4), attempts_per_pos=round(tries / 200, 2))), flush=True)
