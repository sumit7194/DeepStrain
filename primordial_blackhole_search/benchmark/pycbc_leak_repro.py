"""Standalone reproduction: resident memory grows with each distinct-length TaylorF2 waveform from pycbc.

    python pycbc_leak_repro.py td      # pycbc.waveform.get_td_waveform, random masses   -> RSS grows
    python pycbc_leak_repro.py td-same # same call, nearly fixed masses (fixed length) -> RSS flat
    python pycbc_leak_repro.py fd      # pycbc.waveform.get_fd_waveform, random masses   -> (tells TD conversion vs FD)

Needs only numpy and pycbc. Prints RSS (MB, from `ps`) every 5 waveforms after gc.collect().

Observed 2026-11-20 (macOS 27.2 arm64, Python 3.12.13, pycbc 2.10.0, lal 7.7.1, lalsimulation 6.2.1, numpy 2.4.6):
  td       204 -> 1357 (5) -> 1829 (10) -> 1867 (20) -> 2040 MB (30, first 4,194,304-sample waveform)
  td-same  207 -> 603 MB, flat from 5 to 30
  fd       204 -> 1299 (5) -> 1421 (15) -> 1698 MB (25-30)
Reading: memory is retained per new waveform size and not released after the arrays are deleted and gc runs; it
plateaus while sizes repeat. It is not shown to be unbounded. Same with pycbc's numpy and default FFT backends.
Where it lives (pycbc FFT plans/caches, lalsimulation, or the allocator) is not isolated here.
"""
import gc
import os
import subprocess
import sys

import numpy as np
from pycbc.waveform import get_fd_waveform, get_td_waveform


def rss_mb():
    return int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())])) // 1024


def main(mode, n=30):
    rng = np.random.default_rng(0)
    print(f"{mode}: start RSS {rss_mb()} MB", flush=True)
    for i in range(n):
        if mode == "td-same":
            m1, m2 = 0.3, 0.25 + 0.001 * i
        else:
            m1, m2 = np.exp(rng.uniform(np.log(0.2), np.log(1.0), 2))
        if mode == "fd":
            hp, hc = get_fd_waveform(approximant="TaylorF2", mass1=m1, mass2=m2, delta_f=1 / 1024, f_lower=50,
                                     distance=10)
        else:
            hp, hc = get_td_waveform(approximant="TaylorF2", mass1=m1, mass2=m2, delta_t=1 / 4096, f_lower=50,
                                     distance=10)
        n_samp = len(hp)
        del hp, hc
        gc.collect()
        if i % 5 == 4:
            print(f"{mode}: {i + 1} waveforms (last length {n_samp}), RSS {rss_mb()} MB", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
