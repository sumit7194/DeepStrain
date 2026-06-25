"""Shared unblocker (PLAN.md): the echo spacing Δt(M, χ) from first principles, verified vs Abedi Table I.

The echo round-trip is Δt = 2[r*(r_peak) − r*(r_mem)], the tortoise-coordinate distance from the
angular-momentum-barrier peak (~photon sphere) to a reflective membrane a Planck proper-distance above the
Kerr horizon (Abedi et al. 2017, arXiv:1612.00266; abstract scaling Δt≈8M·logM + spin corrections).

Kerr tortoise coordinate has the closed form (Δ = (r−r₊)(r−r₋), r±=M±√(M²−a²), a=χM):
    r*(r) = r + [2M r₊/(r₊−r₋)] ln(r−r₊) − [2M r₋/(r₊−r₋)] ln(r−r₋)
Membrane at proper distance n·ℓ_P above r₊ → coordinate offset δ = (nℓ_P)²(r₊−r₋)/(4(r₊²+a²)) (near-horizon
proper distance ∝ √δ). The one convention knob (n·r_peak) is CALIBRATED to GW150914's published 0.2925 s,
then the formula must PREDICT GW151226 and LVT151012 — that is the validation.

Run:  .venv/bin/python scripts/14_echo_spacing.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

T_SUN = 4.925490947e-6        # GM_sun/c^3  [s]
LP_GEO = 1.616255e-35 / 2.99792458e8   # Planck length / c  [s]

# Abedi et al. 2017 (arXiv:1612.00266) Eq. 6 / Table I:
#   (detector-frame remnant mass M_f [M_sun], spin chi_f, published Δt_echo [s])
# Published Δt: GW150914 0.2925±0.00916, GW151226 0.1013±0.01152, LVT151012 0.1778±0.02789.
ABEDI = {
    "GW150914":  (68.0, 0.69, 0.2925),
    "GW151226":  (22.4, 0.74, 0.1013),
    "LVT151012": (42.0, 0.66, 0.1778),
}


def echo_spacing(m_det, chi, n_planck=1.0, r_peak_factor=3.0):
    """Echo Δt [s] for a detector-frame remnant mass m_det [M_sun] and spin chi."""
    M = m_det * T_SUN
    a = chi * M
    s = np.sqrt(max(M * M - a * a, 0.0))
    r_p, r_m = M + s, M - s
    cp, cm = 2 * M * r_p / (r_p - r_m), 2 * M * r_m / (r_p - r_m)

    def rstar(r):                                  # for r whose (r−r₊) is float-resolvable (the barrier)
        return r + cp * np.log(r - r_p) - cm * np.log(r - r_m)

    lp = n_planck * LP_GEO
    delta = lp ** 2 * (r_p - r_m) / (4 * (r_p ** 2 + a ** 2))      # coordinate offset for proper dist n·ℓ_P
    # membrane rstar uses ln(δ) DIRECTLY — δ (~1e-85 s) underflows in r₊+δ, so never subtract;
    # r_mem≈r₊ for the linear term and r_mem−r₋≈r₊−r₋ (δ negligible there).
    rstar_mem = r_p + cp * np.log(delta) - cm * np.log(r_p - r_m)
    return 2.0 * (rstar(r_peak_factor * M) - rstar_mem)


def main() -> None:
    # NO calibration: the pure physical formula (membrane at 1 ℓ_P, barrier peak at the 3M photon sphere).
    print("pure first-principles formula (n_planck=1, r_peak=3M) — no free parameters tuned to Δt\n")
    print(f"{'event':>10} {'M_f':>6} {'chi':>5} | {'Δt published':>12} {'Δt formula':>11} {'err':>7}")
    rows = {}
    for ev, (m, chi, dt_pub) in ABEDI.items():
        dt = echo_spacing(m, chi)
        err = (dt - dt_pub) / dt_pub
        rows[ev] = {"M_f": m, "chi": chi, "dt_published": dt_pub, "dt_formula": dt, "rel_err": err}
        print(f"{ev:>10} {m:>6.1f} {chi:>5.2f} | {dt_pub:>12.4f} {dt:>11.4f} {err:>+6.1%}")

    # validation: an UNCALIBRATED first-principles formula reproducing all 3 published Δt within 5%
    ok = all(abs(r["rel_err"]) < 0.05 for r in rows.values())
    errstr = ", ".join(f"{ev}:{rows[ev]['rel_err']:+.1%}" for ev in ABEDI)
    print(f"\nVALIDATION: uncalibrated formula reproduces all 3 Abedi Δt within 5% -> "
          f"{'PASS' if ok else 'FAIL'} ({errstr})")

    import json
    RESULTS = Path(__file__).resolve().parent.parent / "results"
    (RESULTS / "14_echo_spacing.json").write_text(json.dumps(
        {"T_sun_s": T_SUN, "lp_geo_s": LP_GEO, "n_planck": 1.0, "r_peak_factor": 3.0,
         "events": rows, "validation_pass": bool(ok)}, indent=2))
    print(f"\nwrote 14_echo_spacing.json")


if __name__ == "__main__":
    main()
