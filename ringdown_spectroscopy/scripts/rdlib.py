"""Shared library for the ringdown spectroscopy pipeline.

Conventions:
- All fitting happens on WHITENED, band-passed (30-350 Hz) strain. The whitening
  response is locally flat near ~250 Hz, so a damped sinusoid approximately keeps
  its form; we do not trust that analytically — biases are CALIBRATED with
  injections (04_injections.py). That is the project's ethos.
- Kerr maps use the `qnm` package (Berti/Stein conventions, s=-2).
"""
from __future__ import annotations

import numpy as np
import qnm
from gwosc import datasets
from gwpy.timeseries import TimeSeries
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, least_squares

M_SUN_SECONDS = 4.925490947641267e-06  # G * M_sun / c^3
BAND = (30, 350)


# ---------------------------------------------------------------- data plumbing
def fetch_whitened(
    detector: str, gps: float, half: float = 32.0, bandpass: bool = True
) -> TimeSeries:
    """Fetch open strain around `gps`; whiten (+ optional bandpass).

    NOTE: fit on bandpass=False data — whitening alone leaves the noise white,
    which is exactly what least-squares assumes; the zero-phase bandpass smears
    a damped sinusoid in time and biases tau upward (measured in 04). Use the
    bandpass only for display."""
    strain = TimeSeries.fetch_open_data(detector, gps - half, gps + half, cache=True)
    white = strain.whiten(4, 2)
    return white.bandpass(*BAND) if bandpass else white


def find_peak(white: TimeSeries, gps: float, half: float = 0.1) -> float:
    """GPS time of |strain| maximum within +-half s of the catalog time."""
    seg = white.crop(gps - half, gps + half)
    return float(seg.times.value[np.abs(seg.value).argmax()])


def event_gps(event: str) -> float:
    return datasets.event_gps(event)


# ---------------------------------------------------------------- Kerr (f,tau) <-> (M,chi)
class KerrMap:
    """Map between (mass, spin) and (frequency, damping time) for one QNM."""

    def __init__(self, ell: int = 2, m: int = 2, n: int = 0):
        mode = qnm.modes_cache(s=-2, l=ell, m=m, n=n)
        self.chis = np.linspace(0.0, 0.998, 400)
        omegas = np.array([mode(a=c)[0] for c in self.chis])
        self._re = CubicSpline(self.chis, omegas.real)
        self._im = CubicSpline(self.chis, omegas.imag)
        # quality factor Q = pi*f*tau = -Re(omega)/(2*Im(omega)); monotonic in chi
        self._q = CubicSpline(self.chis, -omegas.real / (2 * omegas.imag))

    def f_tau(self, mass: float, chi: float) -> tuple[float, float]:
        """(frequency [Hz], damping time [s]) for detector-frame mass [M_sun]."""
        t_m = mass * M_SUN_SECONDS
        return self._re(chi) / (2 * np.pi * t_m), -t_m / self._im(chi)

    def mass_chi(self, f_hz: float, tau_s: float) -> tuple[float, float]:
        """Invert (f, tau) -> (mass [M_sun], chi). Raises if Q out of Kerr range."""
        q_target = np.pi * f_hz * tau_s
        lo, hi = self._q(self.chis[0]), self._q(self.chis[-1])
        if not (lo <= q_target <= hi):
            raise ValueError(f"Q={q_target:.2f} outside Kerr range [{lo:.2f}, {hi:.2f}]")
        chi = brentq(lambda c: self._q(c) - q_target, self.chis[0], self.chis[-1])
        mass = self._re(chi) / (2 * np.pi * f_hz) / M_SUN_SECONDS
        return float(mass), float(chi)


# ---------------------------------------------------------------- ringdown model
def damped_sinusoids(t: np.ndarray, t0: float, params: list[dict]) -> np.ndarray:
    """Sum of damped sinusoids, zero before t0.

    Each mode dict: {f, tau, amp, phi}. Times in seconds (same epoch as t0).
    """
    out = np.zeros_like(t)
    dt = t - t0
    mask = dt >= 0
    for p in params:
        out[mask] += (
            p["amp"]
            * np.exp(-dt[mask] / p["tau"])
            * np.cos(2 * np.pi * p["f"] * dt[mask] + p["phi"])
        )
    return out


# ---------------------------------------------------------------- fitting
def fit_modes(
    series: dict[str, tuple[np.ndarray, np.ndarray]],
    t0: float,
    n_modes: int,
    f_bounds=((150.0, 350.0), (150.0, 350.0)),
    tau_bounds=((1e-3, 0.02), (2e-4, 4e-3)),
    n_restarts: int = 24,
    seed: int = 0,
    window: float = 0.06,
) -> dict:
    """Least-squares fit of n_modes damped sinusoids, SHARED (f, tau) across
    detectors, per-detector amplitude & phase.

    series: {det: (times, values)} of whitened strain (absolute GPS times).
    Returns dict with f[], tau[], per-det amp/phi, cost, and the model arrays.
    """
    rng = np.random.default_rng(seed)
    dets = sorted(series)
    crops = {}
    for det in dets:
        t, v = series[det]
        m = (t >= t0 - 0.002) & (t <= t0 + window)
        crops[det] = (t[m], v[m])

    n_shared = 2 * n_modes               # f, tau per mode
    n_per_det = 2 * n_modes              # amp, phi per mode per detector

    def unpack(x):
        shared = x[:n_shared].reshape(n_modes, 2)
        rest = x[n_shared:].reshape(len(dets), n_modes, 2)
        return shared, rest

    def resid(x):
        shared, rest = unpack(x)
        res = []
        for i, det in enumerate(dets):
            t, v = crops[det]
            params = [
                dict(f=shared[k, 0], tau=shared[k, 1], amp=rest[i, k, 0], phi=rest[i, k, 1])
                for k in range(n_modes)
            ]
            res.append(v - damped_sinusoids(t, t0, params))
        return np.concatenate(res)

    lo, hi = [], []
    for k in range(n_modes):
        lo += [f_bounds[k][0], tau_bounds[k][0]]
        hi += [f_bounds[k][1], tau_bounds[k][1]]
    for _ in dets:
        for _ in range(n_modes):
            lo += [0.0, -np.pi]
            hi += [50.0, np.pi]

    # data-driven frequency init: FFT peak of the fit window (first detector)
    t_ref, v_ref = crops[dets[0]]
    dt_s = t_ref[1] - t_ref[0]
    freqs = np.fft.rfftfreq(len(v_ref), dt_s)
    spec = np.abs(np.fft.rfft(v_ref))
    in_band = (freqs >= f_bounds[0][0]) & (freqs <= f_bounds[0][1])
    f_peak = float(freqs[in_band][spec[in_band].argmax()])

    best = None
    for r in range(n_restarts):
        x0 = []
        for k in range(n_modes):
            if r < n_restarts // 2:
                f_init = np.clip(rng.normal(f_peak, 10.0), *f_bounds[k])
            else:
                f_init = rng.uniform(*f_bounds[k])
            x0 += [f_init, rng.uniform(*tau_bounds[k])]
        for _ in dets:
            for _ in range(n_modes):
                x0 += [rng.uniform(0.5, 5.0), rng.uniform(-np.pi, np.pi)]
        try:
            sol = least_squares(resid, x0, bounds=(lo, hi), max_nfev=4000)
        except Exception:
            continue
        if best is None or sol.cost < best.cost:
            best = sol

    shared, rest = unpack(best.x)
    out = {
        "cost": float(best.cost),
        "modes": [
            {"f": float(shared[k, 0]), "tau": float(shared[k, 1])} for k in range(n_modes)
        ],
        "per_det": {
            det: [
                {"amp": float(rest[i, k, 0]), "phi": float(rest[i, k, 1])}
                for k in range(n_modes)
            ]
            for i, det in enumerate(dets)
        },
        "t0": t0,
        "window": window,
    }
    # order modes by damping time (220 = longest-lived first)
    order = np.argsort([-m["tau"] for m in out["modes"]])
    out["modes"] = [out["modes"][k] for k in order]
    for det in dets:
        out["per_det"][det] = [out["per_det"][det][k] for k in order]
    return out


# ---------------------------------------------------------------- injections
def inject_ringdown(
    raw: TimeSeries, t0: float, params: list[dict]
) -> TimeSeries:
    """Add a ringdown (sum of damped sinusoids, RAW strain units) into a copy
    of `raw` at GPS time t0."""
    t = raw.times.value
    sig = damped_sinusoids(t, t0, params)
    out = raw.copy()
    out.value[:] = raw.value + sig
    return out


def progress(run: str, step: int, total: int, **metrics) -> None:
    """Heartbeat for the repo dashboard (same pattern as curvlib/echolib)."""
    import json
    import time
    from pathlib import Path

    prog = Path(__file__).resolve().parent.parent / "results" / "progress"
    prog.mkdir(parents=True, exist_ok=True)
    f = prog / f"{run}.json"
    hist = []
    if f.exists():
        try:
            hist = json.loads(f.read_text()).get("history", [])
        except json.JSONDecodeError:
            pass
    loss = metrics.get("loss")
    if loss is not None:
        hist = (hist + [[step, float(loss)]])[-200:]
    tmp = f.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(
            {"run": run, "step": int(step), "total": int(total),
             "metrics": {k: float(v) for k, v in metrics.items()},
             "history": hist, "updated": time.time()}
        )
    )
    tmp.replace(f)
    idx = prog / "index.json"
    try:
        names = json.loads(idx.read_text()) if idx.exists() else []
    except json.JSONDecodeError:
        names = []
    if run not in names:
        names.append(run)
        idx.write_text(json.dumps(names))


class heartbeat:
    """Context manager: daemon thread that heartbeats the dashboard while an
    opaque third-party call (e.g. sbi's train loop) runs. Shows elapsed seconds
    so 'alive and working' is distinguishable from 'hung'."""

    def __init__(self, run: str, every: float = 20.0):
        self.run, self.every = run, every

    def __enter__(self):
        import threading
        import time

        self.stop = threading.Event()
        t0 = time.time()

        def beat():
            n = 0
            while not self.stop.wait(self.every):
                n += 1
                progress(self.run, n, 0, elapsed_s=time.time() - t0)

        self.thread = threading.Thread(target=beat, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join(timeout=1)
        return False
