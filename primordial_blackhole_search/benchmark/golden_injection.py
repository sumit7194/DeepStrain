"""Golden tests for the benchmark's injection path, on real fetched strain. Run before any yardstick is computed.

  python benchmark/golden_injection.py <segment gps>     (needs H1 and L1 of that segment in benchmark/data/)

G1  SNR: an injection's optimal SNR in the project's whitened domain (sqrt(sum h_w^2)) agrees with pycbc's sigma
    on the same PSD and band to 1%, for the lightest and the heaviest masses.
G2  Timing: the envelope peak of each injected signal sits at t_geo + the pycbc geocentre-to-detector delay + the
    peak time of the un-projected waveform on its own axis (t_c = 0) to within 1 ms. The last term matters: TaylorF2
    stops where its frequency reaches Nyquist (2048 Hz), which for 0.2+0.2 Msun is ~18 ms before the formal t_c
    (first run measured -21 ms against t_c alone; that was the test, not the injection). This independently checks project_wave and the sub-sample placement in add_signal. The H1-L1 difference is
    reported against the geometric delay.
G3  Recovery: after adding the signal to REAL strain, the true-template matched-filter SNR at the injected time is
    within 2 of the optimal SNR. Unit-variance noise adds ~N(0,1) to it, so 2 is a 2-sigma bar.
Writes benchmark/results/golden_injection.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

B = Path(__file__).resolve().parent
sys.path.insert(0, str(B.parent)); sys.path.insert(0, str(B))
from pbh import config as C  # noqa: E402
from pbh.data import estimate_psd, whiten  # noqa: E402
import injections as I  # noqa: E402


def main(gps):
    from pycbc.detector import Detector
    from pycbc.filter import sigma
    from pycbc.types import TimeSeries
    from scipy.signal import hilbert
    fs = C.SAMPLE_RATE
    strain = {d: np.load(B / "data" / f"{d}_{gps}.npy").astype(np.float64) for d in ("H1", "L1")}
    psd = {d: estimate_psd(strain[d]) for d in strain}
    out, ok = {"segment": gps, "cases": []}, True
    for m1, m2 in ((0.2, 0.2), (1.0, 1.0)):
        row = dict(ra=1.1, dec=0.4, psi=0.3, inclination=0.5, coa_phase=0.7, mass1=m1, mass2=m2,
                   t_geo=gps + 2000.123456, distance=40.0 * (I.chirp_mass(m1, m2) / I.MC_BNS) ** (5 / 6))
        case, tpk = {"masses": [m1, m2]}, {}
        from pycbc.waveform import get_td_waveform
        hp, hc = get_td_waveform(approximant=C.APPROXIMANT, mass1=m1, mass2=m2, delta_t=1 / fs, f_lower=C.F_LOWER,
                                 distance=row["distance"], inclination=row["inclination"], coa_phase=row["coa_phase"])
        t_ref = float(hp.sample_times[int(np.argmax(np.abs(hp.numpy() + 1j * hc.numpy())))])
        case["own_axis_peak_ms"] = 1e3 * t_ref
        for d in ("H1", "L1"):
            sig, t0 = I.detector_signal(row, d)
            alone = I.add_signal(np.zeros_like(strain[d]), gps, sig, t0)
            hw = whiten(alone, psd[d])
            snr_w = float(np.sqrt(np.sum(hw ** 2)))
            snr_p = float(sigma(TimeSeries(alone, delta_t=1 / fs, epoch=gps), psd=psd[d],
                                low_frequency_cutoff=C.F_LOWER - 10, high_frequency_cutoff=C.F_HIGH))
            env = np.abs(hilbert(alone))
            k = int(np.argmax(env))
            a, b, c = env[k - 1], env[k], env[k + 1]  # parabolic sub-sample peak
            tpk[d] = gps + (k + 0.5 * (a - c) / (a - 2 * b + c)) / fs
            expect = t_ref + row["t_geo"] + Detector(d).time_delay_from_earth_center(row["ra"], row["dec"], row["t_geo"])
            # G3: matched filter in the whitened domain = <data_w, h_w>/|h_w| at the injected alignment
            data_w = whiten(strain[d] + alone, psd[d])
            snr_rec = float(np.dot(data_w, hw) / snr_w)
            case[d] = dict(snr_whitened=snr_w, snr_pycbc_sigma=snr_p, rel_err=abs(snr_w / snr_p - 1),
                           merger_err_ms=1e3 * (tpk[d] - expect), mf_recovered=snr_rec)
            ok &= case[d]["rel_err"] < 0.01 and abs(case[d]["merger_err_ms"]) < 1.0 and abs(snr_rec - snr_w) < 2
        geo = Detector("L1").time_delay_from_detector(Detector("H1"), row["ra"], row["dec"], row["t_geo"])
        case["L1_minus_H1_ms"] = dict(measured=1e3 * (tpk["L1"] - tpk["H1"]), geometric=1e3 * geo)
        out["cases"].append(case)
        print(json.dumps(case, indent=1), flush=True)
    out["pass"] = bool(ok)
    (B / "results").mkdir(exist_ok=True)
    (B / "results" / "golden_injection.json").write_text(json.dumps(out, indent=1))
    print("GOLDEN", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1])))
