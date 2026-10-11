"""On-device cost of the fast bank path (cached whitening filter + window-only mixing)."""
import json, time, glob
import numpy as np
from pbh import config as C
from pbh.data import whiten_segment
from pbh.spectrogram import model_input
from pbh.sweep import SweepGrid
from pbh.waveforms import sample_params
from bank import WhitenedEntry, whitening_filter

w, t0, psd = whiten_segment("H1", 1242072046)
a = time.perf_counter(); wf = whitening_filter(psd); t_filter = time.perf_counter() - a
ents, tw = [], []
for f in sorted(glob.glob("bank_test/w*.npz")):
    z = np.load(f)
    a = time.perf_counter(); ents.append(WhitenedEntry(z["hp0"], z["hc0"], wf)); tw.append(time.perf_counter() - a)
g = SweepGrid.short(64); n_win, n_time = g.win_samp, g.n_time_bins
crop = C.WHITEN_CROP_SEC * C.SAMPLE_RATE; lo, hi = crop, len(w) - crop - n_win
rng = np.random.default_rng(0); tinj, tspec, tries = [], [], 0
for _ in range(300):
    while True:
        tries += 1
        p = sample_params(rng); e = ents[int(rng.integers(len(ents)))]
        start = int(rng.integers(lo, hi)); midx = int(rng.uniform(0.25, 1.30) * n_win)
        a = time.perf_counter(); win, snr = e.inject(w[start:start + n_win], p, "H1", t0, float(rng.uniform(*C.TRAIN_SNR_RANGE)), midx); tinj.append(time.perf_counter() - a)
        if snr >= 4.0:
            break
    a = time.perf_counter(); model_input(win, n_time); tspec.append(time.perf_counter() - a)
med = lambda v: round(float(np.median(v)), 4)
print("FAST_JSON " + json.dumps(dict(filter_s=round(t_filter, 2), whiten_entry_s_median=med(tw), whiten_entry_s_max=round(max(tw), 3),
      inject_s_median=med(tinj), spectrogram_s_median=med(tspec), attempts_per_pos=round(tries / 300, 3),
      pos_sample_s=round((sum(tinj) + sum(tspec)) / 300, 4))), flush=True)
