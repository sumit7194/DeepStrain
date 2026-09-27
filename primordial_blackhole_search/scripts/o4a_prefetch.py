"""M11 prefetch pass (The Bridge / user, 2026-09-27): cache the strain files freeze + score need, and nothing else.

GWOSC was serving ~6 KB/s on 2026-09-27, so the final stage cannot run end to end. This fetches, one pass per call:
  ESSENTIAL  H1 pool segment #1 (freeze's O4a PSD for the band factor) + H1/L1 around each PRIMARY trigger
  REPORTING  the segments around the OUT-OF-DOMAIN triggers (scored for the registered no-verdict rows)
Segment choice is `o4a_ssm_rescore.trigger_segments` -- the same function `score` uses -- so the files are
exactly the ones scoring will read. A file counts only if it is COMPLETE: 4096 s x 4096 Hz samples, all finite;
anything else is deleted and retried next pass. No freeze, no scoring. Writes results/o4a_ssm/prefetch.json.
Exit: 0 all cached, 3 essential cached only, 1 otherwise.
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
_r = importlib.util.spec_from_file_location("r", HERE / "o4a_ssm_rescore.py")
r = importlib.util.module_from_spec(_r); _r.loader.exec_module(r)
from pbh import config as C
from pbh.data import fetch_segment, load_segment, segment_path

STATUS = r.OUT / "prefetch.json"


def needed():
    out = [("H1", json.loads(r.POOL.read_text())["segments"][0], "freeze", True)]
    for t in r.triggers():
        if t["cls"] == "out-of-range":
            continue
        for d, g in sorted(r.trigger_segments(t).items()):
            out.append((d, g, t["name"], t["cls"] == "primary"))
    return out


def complete(d, g):
    p = segment_path(d, g)
    if not p.exists():
        return False
    try:
        x, _ = load_segment(d, g)
        ok = len(x) == C.SEGMENT_LEN * C.SAMPLE_RATE and bool(np.isfinite(x).all())
    except Exception:                                      # noqa: BLE001 -- unreadable = incomplete
        ok = False
    if not ok:
        p.unlink(missing_ok=True)
    return ok


def main() -> int:
    rows = []
    for d, g, why, essential in needed():
        t0 = time.time(); state = "cached"
        if not complete(d, g):
            try:
                fetch_segment(d, g)
                state = "fetched" if complete(d, g) else "incomplete"
            except Exception as exc:                        # noqa: BLE001 -- GWOSC failure; retried next pass
                state = f"FAIL {type(exc).__name__}"
        rows.append({"det": d, "gps": g, "for": why, "essential": essential, "state": state,
                     "ok": state in ("cached", "fetched"), "seconds": round(time.time() - t0, 1)})
        print(f"{d} {g} [{why}{' *' if essential else ''}] {state} ({rows[-1]['seconds']}s)", flush=True)
    ess = all(x["ok"] for x in rows if x["essential"]); allok = all(x["ok"] for x in rows)
    STATUS.write_text(json.dumps({"checked": time.ctime(), "essential_complete": ess, "all_complete": allok,
                                  "files": rows}, indent=2))
    return 0 if allok else 3 if ess else 1


if __name__ == "__main__":
    sys.exit(main())
