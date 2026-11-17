"""Fetch only the 6 O3a H1 test segments needed for the OS-regression spot check (prints progress)."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "primordial_blackhole_search"))
from pbh import config as C
from pbh.data import fetch_segment, segment_path
for g in json.loads((C.DATA_DIR / "manifest.json").read_text())["H1"]["test"]:
    t = time.time()
    try:
        fetch_segment("H1", g)
        print(g, "ok", round(segment_path("H1", g).stat().st_size / 1e6), "MB", round(time.time() - t), "s", flush=True)
    except Exception as e:
        print(g, "FAIL", type(e).__name__, str(e)[:200], round(time.time() - t), "s", flush=True)
