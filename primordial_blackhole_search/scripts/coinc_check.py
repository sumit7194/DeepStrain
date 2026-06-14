"""Path G milestone G0: coincidence plumbing check.

Validate the two-detector round-trip BEFORE building the full search (G1):
inject ONE loud event into H1 + L1 with the proper per-detector antenna
response and light-travel delay (v1 search.py geometry), then recover it in
BOTH detectors with the coarse BANK (F0 machinery) and confirm the recovered
peaks are consistent in time. If this works, the bank statistic transfers to
L1 and coincident recovery is sound -> proceed to G1 (triggers + time-slide
background + sensitivity).

Run:  .venv/bin/python scripts/coinc_check.py [--bank 32] [--net-snr 20]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pbh import config as C
from pbh.data import estimate_psd, load_segment, whiten
from pbh.waveforms import InjectionParams, make_whitened_injection
from semicoherent_oracle import PAD, WIN, _corr_sq_local, newsnr
from bank_oracle import build_bank

FS = C.SAMPLE_RATE
EVENT = InjectionParams(mass1=0.62, mass2=0.44, ra=2.1, dec=-0.9,
                        psi=1.3, inclination=0.7, coa_phase=0.4)


def local_peak(d, chunks, ws):
    """(peak newSNR, sample offset from ws) over the +-PAD scan."""
    A = np.zeros(2 * PAD + 1)
    Bsum = np.zeros(2 * PAD + 1)
    for off, a, p in chunks:
        x = np.real(_corr_sq_local(d, off, a, ws))
        A += x
        Bsum += x ** 2 / p
    ns = newsnr(A, Bsum, len(chunks))
    k = int(ns.argmax())
    return float(ns[k]), k - PAD


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=int, default=32)
    ap.add_argument("--net-snr", type=float, default=20.0)
    args = ap.parse_args()

    from pycbc.detector import Detector

    manifest = json.loads((C.DATA_DIR / "manifest.json").read_text())
    gps = manifest["L1"]["coinc"][0]
    assert gps in manifest["H1"]["test"] or gps in manifest["H1"]["train"], gps
    t_geo = gps + C.SEGMENT_LEN // 2  # merger mid-segment

    # per-detector optimal SNR at 1 Mpc -> scale to the requested NETWORK SNR
    refs = {}
    for ifo in ("H1", "L1"):
        strain, t0 = load_segment(ifo, gps)
        psd = estimate_psd(strain)
        _, snr_ref = make_whitened_injection(EVENT, ifo, t_geo, psd)
        refs[ifo] = (strain, t0, psd, snr_ref)
    net_ref = np.hypot(refs["H1"][3], refs["L1"][3])
    scale = args.net_snr / net_ref

    print(f"coincident segment GPS {gps}; target network SNR {args.net_snr}")
    print(f"bank: {args.bank} templates, n=8 vetoed statistic\n")
    peaks = {}
    for ifo in ("H1", "L1"):
        strain, t0, psd, snr_ref = refs[ifo]
        w = whiten(strain, psd)
        h_w, _ = make_whitened_injection(EVENT, ifo, t_geo, psd)
        delay = Detector(ifo).time_delay_from_earth_center(EVENT.ra, EVENT.dec, t_geo)
        t_det = t_geo + delay
        merger_idx = int(round((t_det - t0) * FS))
        sig = h_w * scale
        w[merger_idx - len(sig) + 1 : merger_idx + 1] += sig
        bank, chunks = build_bank(t0, psd, args.bank)
        dwin = w[merger_idx - WIN - PAD : merger_idx + PAD]
        scored = [(local_peak(dwin, ch, PAD), bank[k][0]) for k, ch in enumerate(chunks)]
        (best_snr, off), best_mc = max(scored, key=lambda r: r[0][0])
        # diagnostic: TRUE template (exact injected shape) -> isolates bug vs mismatch
        from semicoherent_oracle import analytic_chunks
        true_chunks = analytic_chunks(h_w[-WIN:], 8)
        (true_snr, true_off) = local_peak(dwin, true_chunks, PAD)
        print(f"     [diag] TRUE-template recovered newSNR {true_snr:5.1f} "
              f"peak {true_off/FS*1e3:+.1f} ms (vs bank {best_snr:.1f})")
        t_peak = t_det + off / FS
        peaks[ifo] = dict(inj_snr=snr_ref * scale, rec_newsnr=best_snr,
                          best_mc=best_mc, off_ms=off / FS * 1e3,
                          t_peak=t_peak, t_det=t_det, delay_ms=delay * 1e3)
        p = peaks[ifo]
        print(f"  {ifo}: injected SNR {p['inj_snr']:5.1f} | recovered newSNR {best_snr:5.1f} | "
              f"best-template Mc {best_mc:.3f} (true {EVENT.chirp_mass:.3f}) | "
              f"peak {off/FS*1e3:+.1f} ms from true merger")

    dt_rec = (peaks["H1"]["t_peak"] - peaks["L1"]["t_peak"]) * 1e3
    dt_true = (peaks["H1"]["t_det"] - peaks["L1"]["t_det"]) * 1e3
    net = np.hypot(peaks["H1"]["rec_newsnr"], peaks["L1"]["rec_newsnr"])
    print(f"\n  recovered network newSNR {net:.1f}")
    print(f"  inter-site dt: recovered {dt_rec:+.2f} ms vs true {dt_true:+.2f} ms "
          f"(err {dt_rec - dt_true:+.2f} ms, light-travel budget +-10 ms)")
    ok = (peaks["H1"]["rec_newsnr"] > 6 and peaks["L1"]["rec_newsnr"] > 6
          and abs(dt_rec - dt_true) < 10)
    print("\nG0 PLUMBING " + ("PASS — bank recovers coincident signal in both detectors, "
                              "timing consistent" if ok else "CHECK — see numbers above"))


if __name__ == "__main__":
    main()
