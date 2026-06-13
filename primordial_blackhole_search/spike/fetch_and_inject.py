"""Phase 0 feasibility spike.

Fetch real public LIGO strain, inject a simulated subsolar-mass chirp,
whiten, and produce the spectrogram where the chirp should appear as a
long thin rising track. Then recover it with a matched filter to confirm
the injection is sound.

Run:  uv run python spike/fetch_and_inject.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from gwpy.timeseries import TimeSeries
from pycbc.detector import Detector
from pycbc.filter import matched_filter
from pycbc.psd import interpolate, inverse_spectrum_truncation
from pycbc.types import TimeSeries as PycbcTimeSeries
from pycbc.waveform import get_td_waveform

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------- config
IFO = "H1"
SAMPLE_RATE = 4096
# A quiet stretch of O3a (H1 observing, no catalog event nearby).
GPS_START = 1242440000
DURATION = 512  # seconds of data to fetch

M1, M2 = 0.5, 0.5  # solar masses — the subsolar target
F_LOWER = 45.0  # Hz; 0.5+0.5 from 45 Hz is a ~256 s signal
DISTANCE_MPC = 10.0  # close enough to be clearly visible for the spike
APPROXIMANT = "TaylorF2"  # inspiral-only is accurate here: merger is ~4.4 kHz,
#                           above the sensitive band for a 1 M_sun total mass

RA, DEC, POL = 1.7, -1.2, 0.6  # arbitrary fixed sky location


def fetch_strain() -> TimeSeries:
    cache = OUT / f"{IFO}_{GPS_START}_{DURATION}.hdf5"
    if cache.exists():
        print(f"using cached strain: {cache.name}")
        return TimeSeries.read(cache, path="strain")
    print(f"fetching {DURATION}s of {IFO} from GWOSC at GPS {GPS_START}...")
    strain = TimeSeries.fetch_open_data(
        IFO, GPS_START, GPS_START + DURATION, sample_rate=SAMPLE_RATE, cache=False
    )
    strain.write(cache, path="strain", format="hdf5")
    return strain


def make_injection() -> PycbcTimeSeries:
    hp, hc = get_td_waveform(
        approximant=APPROXIMANT,
        mass1=M1,
        mass2=M2,
        delta_t=1.0 / SAMPLE_RATE,
        f_lower=F_LOWER,
        distance=DISTANCE_MPC,
    )
    det = Detector(IFO)
    signal = det.project_wave(hp, hc, RA, DEC, POL)
    print(
        f"injection: {M1}+{M2} Msun at {DISTANCE_MPC} Mpc, "
        f"{len(signal) / SAMPLE_RATE:.0f} s from {F_LOWER} Hz"
    )
    return signal


def main() -> None:
    strain = fetch_strain()
    signal = make_injection()

    # Place merger time ~60 s before the end of the segment.
    t_merger = GPS_START + DURATION - 60
    sig = TimeSeries(
        signal.numpy(),
        t0=t_merger + float(signal.start_time),  # signal epoch ends at merger
        sample_rate=SAMPLE_RATE,
    )
    injected = strain.inject(sig)

    # --- spectrogram (Q-transform) around the track, whitened internally
    print("computing Q-transform...")
    q_span = (t_merger - 280, t_merger + 5)
    qgram = injected.q_transform(
        outseg=q_span, qrange=(80, 110), frange=(40, 500), logf=True
    )
    plain = strain.q_transform(
        outseg=q_span, qrange=(80, 110), frange=(40, 500), logf=True
    )

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for ax, data, title in (
        (axes[0], plain, f"{IFO} real noise (no injection)"),
        (axes[1], qgram, f"{IFO} + injected {M1}+{M2} M$_\\odot$ chirp @ {DISTANCE_MPC} Mpc"),
    ):
        pcm = ax.imshow(data, vmin=0, vmax=15)
        ax.set_yscale("log")
        ax.set_ylabel("frequency [Hz]")
        ax.set_title(title)
        fig.colorbar(pcm, ax=ax, label="normalized energy")
    axes[1].set_xlabel(f"time [s] from GPS {q_span[0]}")
    fig.tight_layout()
    fig.savefig(OUT / "spectrogram_comparison.png", dpi=120)
    print(f"wrote {OUT / 'spectrogram_comparison.png'}")

    # --- matched-filter recovery: the ground-truth check
    print("matched filtering...")
    inj_pycbc = PycbcTimeSeries(
        injected.value, delta_t=1.0 / SAMPLE_RATE, epoch=float(injected.t0.value)
    )
    inj_pycbc = inj_pycbc.highpass_fir(F_LOWER - 5, 512).crop(8, 8)

    psd = inj_pycbc.psd(4)
    psd = interpolate(psd, inj_pycbc.delta_f)
    psd = inverse_spectrum_truncation(
        psd, int(4 * SAMPLE_RATE), low_frequency_cutoff=F_LOWER
    )

    template, _ = get_td_waveform(
        approximant=APPROXIMANT,
        mass1=M1,
        mass2=M2,
        delta_t=1.0 / SAMPLE_RATE,
        f_lower=F_LOWER,
        distance=DISTANCE_MPC,
    )
    template.resize(len(inj_pycbc))
    snr = matched_filter(
        template.cyclic_time_shift(template.start_time),
        inj_pycbc,
        psd=psd,
        low_frequency_cutoff=F_LOWER,
    )
    snr = snr.crop(16, 16)
    peak_idx = int(np.argmax(np.abs(snr.numpy())))
    peak_snr = float(np.abs(snr.numpy())[peak_idx])
    peak_time = float(snr.sample_times[peak_idx])
    print(f"matched-filter peak SNR = {peak_snr:.1f} at GPS {peak_time:.2f} "
          f"(true merger GPS {t_merger}, offset {peak_time - t_merger:+.3f} s)")

    fig2, ax2 = plt.subplots(figsize=(14, 4))
    ax2.plot(snr.sample_times, np.abs(snr.numpy()), lw=0.5)
    ax2.axvline(t_merger, color="r", ls="--", label="true merger time")
    ax2.set_xlabel("GPS time [s]")
    ax2.set_ylabel("|SNR|")
    ax2.set_title(f"matched-filter recovery: peak SNR {peak_snr:.1f}")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(OUT / "matched_filter_snr.png", dpi=120)
    print(f"wrote {OUT / 'matched_filter_snr.png'}")

    verdict = "PASS" if peak_snr > 8 and abs(peak_time - t_merger) < 0.1 else "FAIL"
    print(f"\nSPIKE VERDICT: {verdict}")


if __name__ == "__main__":
    main()
