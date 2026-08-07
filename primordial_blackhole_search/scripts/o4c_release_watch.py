"""O4-4: standing watcher for the O4c bulk release / the S251112cm subsolar candidate.

S251112cm (12 Nov 2025, 15:18:45 UTC) is the first gravitational-wave candidate with >99% probability of a
sub-solar component — exactly what this project searches for. Its strain is NOT public: it falls in O4c, and
as of 2026-08 GWOSC's bulk releases stop at O4b (Apr 2024–Jan 2025). The only O4c dataset, O4c1DiscC00, is a
68-minute discovery window around GW250207.

This queries GWOSC and reports whether that has changed, so the moment O4c (or a targeted S251112cm release)
lands we know the pipeline can be pointed at it. cnn_w64 is already validated on O4-era noise (o4_transfer_scout:
0.97x transfer), so no retraining stands in the way.

Run:  .venv/bin/python scripts/o4c_release_watch.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pbh import config as C

S251112CM_GPS = 1446995943          # 2025-11-12 15:18:45 UTC
O4B_END_GPS = 1422144018            # 2025-01-28, end of the last public bulk run


def main() -> None:
    from gwosc import datasets

    runs = sorted(datasets.find_datasets(type="run"))
    cats = sorted(datasets.find_datasets(type="catalog"))
    events = datasets.find_datasets(type="event")

    # (1) is there an O4c bulk run beyond the tiny discovery window?
    o4c = [r for r in runs if r.upper().startswith("O4C")]
    o4c_bulk = []
    for r in o4c:
        try:
            a, b = datasets.run_segment(r)
            dur_h = (b - a) / 3600.0
            o4c_bulk.append({"name": r, "gps": [a, b], "hours": round(dur_h, 2),
                             "covers_s251112cm": bool(a <= S251112CM_GPS <= b)})
        except Exception as e:
            o4c_bulk.append({"name": r, "error": type(e).__name__})

    # (2) any public event after the end of O4b? (a targeted release would show up here)
    late = sorted(e for e in events
                  if any(e.startswith(f"GW25{m}") for m in ("02","03","04","05","06","07","08","09","10","11","12")))
    supers = sorted(e for e in events if e.upper().startswith("S25"))

    covered = any(x.get("covers_s251112cm") for x in o4c_bulk)
    print(f"runs with bulk strain: {runs}")
    print(f"O4c datasets: {o4c_bulk if o4c_bulk else 'NONE'}")
    print(f"public events after Feb 2025: {late if late else 'NONE'}")
    print(f"public S25* superevents: {supers if supers else 'NONE'}")
    print(f"catalogs: {len(cats)} (latest: {cats[-3:]})")
    print()
    if covered or supers:
        print("*** STATUS CHANGED: S251112cm-era data appears PUBLIC — the pipeline is already validated on")
        print("    O4-era noise (cnn_w64 transfers 0.97x), so the subsolar search can be pointed at it. ***")
    else:
        print("STATUS UNCHANGED: S251112cm strain is still embargoed (O4c bulk not released).")
        print("  Public bulk data still ends at O4b (2025-01-28). Nothing to do; re-run this watcher later.")

    out = {"runs": runs, "o4c_datasets": o4c_bulk, "events_after_feb2025": late,
           "s25_superevents": supers, "n_catalogs": len(cats), "latest_catalogs": cats[-3:],
           "s251112cm_gps": S251112CM_GPS, "s251112cm_data_public": bool(covered or supers)}
    (C.RESULTS_DIR / "o4c_release_watch.json").write_text(json.dumps(out, indent=2))
    print("\nwrote o4c_release_watch.json")


if __name__ == "__main__":
    main()
