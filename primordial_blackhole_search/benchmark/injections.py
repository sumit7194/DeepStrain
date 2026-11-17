"""LongChirp benchmark: the injection table (deterministic) and the code that adds injections to raw strain.

  python benchmark/injections.py table      # writes benchmark/injections.csv (committed; the benchmark's foreground)

Population (PLAN.md): non-spinning TaylorF2, component masses log-uniform on [0.2, 1.0] Msun, isotropic sky and
orientation, signal from 50 Hz. Distances follow the MLGWSC-1 chirp-distance convention:
d = d_c * (Mc / Mc_BNS)^(5/6), Mc_BNS = 1.4 * 2^(-1/5), with d_c uniform on [DC_LO, DC_HI] Mpc. The sensitive volume
is then estimated in chirp-distance space, as MLGWSC-1 does, so masses are comparable.

Placement: K realizations; in each test segment, geocentric merger times are spaced >= SPACING_S apart and kept
EDGE_S from the segment edges, with a uniform jitter. SPACING_S exceeds the longest in-band signal (~352 s for
0.2+0.2 Msun from 50 Hz), so no two injections overlap in either detector.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

B = Path(__file__).resolve().parent
sys.path.insert(0, str(B.parent))
from pbh import config as C  # noqa: E402

SEED = 20261117
K = 8
SPACING_S, EDGE_S, JITTER_S = 480, 400, 60
DC_LO, DC_HI = 5.0, 300.0  # Mpc, chirp distance; DC_HI set so the faintest injections are below SNR ~4 (checked)
MC_BNS = 1.4 * 2 ** -0.2
TABLE = B / "injections.csv"
FIELDS = ["id", "realization", "segment", "t_geo", "mass1", "mass2", "ra", "dec", "psi", "inclination",
          "coa_phase", "chirp_distance", "distance"]


def chirp_mass(m1, m2):
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


def table():
    man = json.loads((B / "manifest.json").read_text())
    rng = np.random.default_rng(SEED)
    rows = []
    for r in range(K):
        for g in man["test"]:
            slots = np.arange(g + EDGE_S, g + C.SEGMENT_LEN - EDGE_S - JITTER_S + 1, SPACING_S)
            for t in slots:
                m1, m2 = np.exp(rng.uniform(np.log(C.M_MIN), np.log(C.M_MAX), 2))
                m1, m2 = max(m1, m2), min(m1, m2)
                dc = rng.uniform(DC_LO, DC_HI)
                rows.append(dict(
                    id=len(rows), realization=r, segment=g, t_geo=round(float(t + rng.uniform(0, JITTER_S)), 6),
                    mass1=round(m1, 6), mass2=round(m2, 6), ra=round(rng.uniform(0, 2 * np.pi), 6),
                    dec=round(float(np.arcsin(rng.uniform(-1, 1))), 6), psi=round(rng.uniform(0, np.pi), 6),
                    inclination=round(float(np.arccos(rng.uniform(-1, 1))), 6),
                    coa_phase=round(rng.uniform(0, 2 * np.pi), 6), chirp_distance=round(dc, 4),
                    distance=round(dc * (chirp_mass(m1, m2) / MC_BNS) ** (5 / 6), 4)))
    with open(TABLE, "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} injections ({len(rows) // K} per realization) -> {TABLE.name}")


def detector_signal(row, ifo):
    """Strain of one injection in one detector: (array, start GPS). One waveform and one coa_phase are shared by
    both detectors; pycbc's project_wave applies the antenna pattern and the geocentre-to-detector delay."""
    from pycbc.detector import Detector
    from pycbc.waveform import get_td_waveform
    hp, hc = get_td_waveform(approximant=C.APPROXIMANT, mass1=row["mass1"], mass2=row["mass2"],
                             delta_t=1.0 / C.SAMPLE_RATE, f_lower=C.F_LOWER, distance=row["distance"],
                             inclination=row["inclination"], coa_phase=row["coa_phase"])
    hp.start_time += row["t_geo"]
    hc.start_time += row["t_geo"]
    h = Detector(ifo).project_wave(hp, hc, row["ra"], row["dec"], row["psi"])
    return h.numpy().astype(np.float64), float(h.start_time)


def add_signal(strain, seg_gps, sig, sig_start):
    """Add `sig` (starting at GPS sig_start) to raw strain starting at seg_gps. The sub-sample part of the offset
    is applied as an exact frequency-domain delay on a zero-padded copy, so the merger time is not rounded."""
    off = (sig_start - seg_gps) * C.SAMPLE_RATE
    i0 = int(np.floor(off))
    frac = off - i0
    n = len(sig) + 64
    f = np.fft.rfftfreq(n, d=1.0)
    shifted = np.fft.irfft(np.fft.rfft(np.concatenate([sig, np.zeros(64)])) * np.exp(-2j * np.pi * f * frac), n=n)
    lo, hi = max(i0, 0), min(i0 + n, len(strain))
    strain[lo:hi] += shifted[lo - i0:hi - i0]
    return strain


if __name__ == "__main__":
    {"table": table}[sys.argv[1]]()
