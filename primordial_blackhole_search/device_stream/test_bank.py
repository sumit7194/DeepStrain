"""Golden test: bank path == pbh.waveforms.make_whitened_injection on a real O3a PSD (float32 bank storage)."""
import sys
import numpy as np
from pbh.data import estimate_psd, load_segment
from pbh.waveforms import make_whitened_injection, sample_params
from bank import face_on, projected_whitened, whiten_pair

strain, t0 = load_segment("H1", 1242072046)
psd = estimate_psd(strain)
rng = np.random.default_rng(7)
worst_wave, worst_snr = 0.0, 0.0
for k in range(int(sys.argv[1]) if len(sys.argv) > 1 else 12):
    p = sample_params(rng)
    ref_w, ref_snr = make_whitened_injection(p, "H1", t0, psd)
    wp, wc = whiten_pair(*face_on(p.mass1, p.mass2), psd)
    h_w, snr = projected_whitened(wp, wc, p, "H1", t0)
    dw = np.linalg.norm(h_w - ref_w) / np.linalg.norm(ref_w)
    ds = abs(snr - ref_snr) / ref_snr
    worst_wave, worst_snr = max(worst_wave, dw), max(worst_snr, ds)
    print(f"m=({p.mass1:.3f},{p.mass2:.3f}) Mc={p.chirp_mass:.3f} snr_ref={ref_snr:.4g} rel_wave_err={dw:.2e} rel_snr_err={ds:.2e}")
ok = worst_wave < 1e-5 and worst_snr < 1e-6
print(f"WORST wave {worst_wave:.2e} snr {worst_snr:.2e} -> {'GOLDEN PASS' if ok else 'GOLDEN FAIL'}")
