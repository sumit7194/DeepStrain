"""CPU-only probe (no torch): why is the far-from-candidate mean |z|^2 ~2.8 for a 332-s template (Mc 0.18, f_low 50 Hz)
in the E2 snippet, when unit-variance noise gives 2? Split the lags into those where the template WRAPS around the
snippet's seam (circular FFT correlation of a non-periodic 512-s cut) and those where it does not."""
import sys, json
from pathlib import Path
import numpy as np
B = Path(__file__).resolve().parent
sys.path.insert(0, str(B))
import coh2c_local as L
FS, NS, PRE = L.FS, L.NS, L.PRE
y, w_snip, _ = L.load_ww("H1", 1249001472, ())
st = json.loads((B / "state" / "realbg" / "caught.json").read_text())
cand = [b for b in st["candidates"]["0_0"] if (PRE + 1) * FS < b * 102 < 4096 * FS - 65 * FS][0] * 102
out = {}
for shift_s in (0, 600, 1500):                       # the E2 snippet, and two others far from that candidate
    i_c = cand + shift_s * FS
    if i_c + 65 * FS > len(y):
        i_c = cand - shift_s * FS
    ysn = L.snippet_spectrum(y, i_c)
    for mc, q in ((0.18, 0.95), (0.39, 0.6)):
        for f_lo in (50.0, 71.5):
            h = L.fd_template(mc, q, f_lo)
            sig = np.sqrt(2 * np.sum(np.abs(h * w_snip) ** 2) / NS)
            P = np.zeros(NS, complex); P[: NS // 2 + 1] = ysn * np.conj(h)
            z2 = np.abs(2 * np.fft.ifft(P) / sig) ** 2
            tau = (5 / 256) * (np.pi * f_lo) ** (-8 / 3) * (L.G_SUN * mc) ** (-5 / 3)     # template length (s)
            lag_s = np.arange(NS) / FS
            nowrap = (lag_s > tau + 8) & (lag_s < PRE - 20)          # template fully inside the snippet, before the candidate
            wrap = (lag_s > 8) & (lag_s < tau - 8)                   # template crosses the seam
            k = f"shift{shift_s}_mc{mc}_f{f_lo}"
            out[k] = dict(tau_s=float(tau), mean_z2_nowrap=float(z2[nowrap][::97].mean()) if nowrap.sum() > 5000 else None,
                          n_nowrap_s=float(nowrap.sum() / FS), mean_z2_wrap=float(z2[wrap][::97].mean()),
                          p999_nowrap=float(np.quantile(z2[nowrap][::97], 0.999)) if nowrap.sum() > 5000 else None)
            print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in out[k].items()}, flush=True)
(B / "results" / "coh2c" / "farz2_probe.json").write_text(json.dumps(dict(model="Fable 5.2", status="UNVERIFIED", rows=out), indent=1))
