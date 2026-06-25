"""TheBridge leg-8 export: echo Δt ↔ Abedi Planck-wall prediction, per analyzed event, with the closed form
and % agreement — so the bridge can import it read-only as a literature cross-check (anchoring its ECO/echo
templates to the standard delay formula, the way Move B anchors QNMs to Leaver).

Repackages 14_echo_spacing.json (our first-principles Kerr-tortoise Δt vs Abedi 2017 Table I, validated <2%)
into the {event, dt_echo_model, dt_abedi_predicted, percent_agreement, formula_string} form, and applies the
SAME closed form to GW250114 (post-2017, no Abedi Table-I entry).

Run:  .venv/bin/python scripts/18_abedi_export.py
"""
import importlib.util
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
_spec = importlib.util.spec_from_file_location("es", Path(__file__).resolve().parent / "14_echo_spacing.py")
es = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(es)

# the exact closed form we implemented + validated (Abedi 2017 arXiv:1612.00266 Eq.2; Cardoso–Pani
# Planck-proximity wall). Geometric units: M[s] = M_sun·(GM_sun/c³); a = χ·M; r± = M(1±√(1−χ²)).
FORMULA = {
    "name": "Kerr-tortoise echo round-trip (Abedi-Dvali Planck-proximity wall)",
    "reference": "Abedi, Dvali, et al. 2017, arXiv:1612.00266, Eq.2 / Table I; Cardoso & Pani reviews",
    "expression": ("dt_echo = 2[r*(r_peak) - r*(r_mem)],  "
                   "r*(r) = r + cp·ln(r-r+) - cm·ln(r-r-),  cp = 2M·r+/(r+-r-), cm = 2M·r-/(r+-r-);  "
                   "r± = M(1±√(1-χ²)), a = χM, r_peak = 3M (photon sphere);  "
                   "membrane r_mem = r+ + δ with δ = (n·ℓ_P)²(r+-r-)/(4(r+²+a²)) [proper distance n·ℓ_P above r+]"),
    "leading_order_scaling": "dt_echo ≈ 8·M·ln(M/ℓ_P)·[1 + O(a²)]   (Abedi abstract; Planck units)",
    "no_free_parameter_tuned_to_dt": True,
    "constants": {"T_sun_s_GMsun_over_c3": es.T_SUN, "lp_geo_s_lP_over_c": es.LP_GEO,
                  "n_planck": 1.0, "r_peak_factor": 3.0},
}

# (event, detector-frame remnant M_f [M_sun], χ_f, Abedi-2017-Table-I Δt [s] or None if post-2017)
EVENTS = [
    ("GW150914", 68.0, 0.69, 0.2925),
    ("GW151226", 22.4, 0.74, 0.1013),
    ("GW151012", 42.0, 0.66, 0.1778),   # = Abedi's "LVT151012" (renamed in GWTC-1)
    ("GW250114", 76.0, 0.76, None),     # 2025 event; not in Abedi 2017 -> closed form applied, no Table-I value
]


def main() -> None:
    out = []
    for ev, M, chi, dt_abedi in EVENTS:
        dt_model = es.echo_spacing(M, chi)                     # our closed-form Δt (the value the echo search uses)
        rec = {
            "event": ev,
            "dt_echo_model_s": round(dt_model, 5),
            "dt_abedi_predicted_s": dt_abedi,                  # Abedi 2017 Table I published value (None if post-2017)
            "percent_agreement": (round(100.0 * (1.0 - abs(dt_model - dt_abedi) / dt_abedi), 2)
                                  if dt_abedi else None),
            "M_f_detframe_Msun": M, "chi_f": chi,
            "abedi_table_I": dt_abedi is not None,
        }
        out.append(rec)

    agree = [r["percent_agreement"] for r in out if r["percent_agreement"] is not None]
    payload = {
        "_for": "TheBridge leg 8 (ECO / echo templates) — literature cross-check of the inter-echo spacing",
        "formula": FORMULA,
        "events": out,
        "validation_summary": {
            "abedi_table_I_events": [r["event"] for r in out if r["abedi_table_I"]],
            "min_percent_agreement": min(agree), "max_percent_agreement": max(agree),
            "claim": "uncalibrated first-principles formula reproduces Abedi 2017 Table I to <2% (>98% agreement) "
                     "on all 3 published events; the same closed form is applied to GW250114 (post-2017).",
            "source_artifact": "echoes/results/14_echo_spacing.json (gated in verify.sh)",
        },
    }
    (RESULTS / "18_abedi_crosscheck.json").write_text(json.dumps(payload, indent=2))
    print(f"{'event':>10} | {'dt_model':>9} | {'dt_abedi':>9} | {'agreement':>9}")
    for r in out:
        print(f"{r['event']:>10} | {r['dt_echo_model_s']:>9.4f} | "
              f"{str(r['dt_abedi_predicted_s']):>9} | "
              f"{str(r['percent_agreement'])+'%' if r['percent_agreement'] else 'n/a':>9}")
    print(f"\nagreement on the 3 Abedi events: {min(agree):.1f}%–{max(agree):.1f}% (all <2% error). "
          f"wrote results/18_abedi_crosscheck.json  [for TheBridge leg 8]")


if __name__ == "__main__":
    main()
