"""Golden test: WhitenedEntry fast path == whiten_pair + projected_whitened + inject_into_window (the slow bank path,
itself golden-tested against make_whitened_injection)."""
import numpy as np
from pbh import config as C
from pbh.data import whiten_segment
from pbh.sweep import SweepGrid
from pbh.waveforms import inject_into_window, sample_params
from bank import WhitenedEntry, face_on, projected_whitened, whiten_pair, whitening_filter

w, t0, psd = whiten_segment("H1", 1242072046)
wf = whitening_filter(psd)
g = SweepGrid.short(64); n_win = g.win_samp
crop = C.WHITEN_CROP_SEC * C.SAMPLE_RATE
rng = np.random.default_rng(11)
worst_w, worst_s = 0.0, 0.0
for k in range(6):
    m = sample_params(rng)
    hp0, hc0 = face_on(m.mass1, m.mass2)
    wp, wc = whiten_pair(hp0, hc0, psd)
    e = WhitenedEntry(hp0, hc0, wf)
    worst_w = max(worst_w, np.max(np.abs(e.wp - wp)) / np.max(np.abs(wp)))
    for _ in range(5):
        p = sample_params(rng); target = float(rng.uniform(*C.TRAIN_SNR_RANGE))
        start = int(rng.integers(crop, len(w) - crop - n_win)); midx = int(rng.uniform(0.25, 1.30) * n_win)
        win = w[start:start + n_win]
        ref, ref_snr = inject_into_window(win, *projected_whitened(wp, wc, p, "H1", t0), target, midx)
        out, snr = e.inject(win, p, "H1", t0, target, midx)
        worst_w = max(worst_w, np.max(np.abs(out - ref)) / np.max(np.abs(ref)))
        worst_s = max(worst_s, abs(snr - ref_snr) / max(ref_snr, 1e-12))
ok = worst_w < 1e-9 and worst_s < 1e-9
print(f"WORST window {worst_w:.2e} in_snr {worst_s:.2e} -> {'FASTPATH GOLDEN PASS' if ok else 'FASTPATH GOLDEN FAIL'}")
