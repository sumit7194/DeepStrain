"""Double the training noise diversity: fetch 8 more H1 segments and append
them to the manifest train split.

Run:  uv run python scripts/fetch_more_train.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pbh import config as C
from pbh.data import discover_segments, fetch_segment

MANIFEST = C.DATA_DIR / "manifest.json"
N_EXTRA = 8


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    used = set(
        manifest["H1"]["train"] + manifest["H1"]["val"] + manifest["H1"]["test"]
    )
    candidates = discover_segments("H1", len(used) + N_EXTRA + 4)
    new = [g for g in candidates if g not in used][:N_EXTRA]
    print(f"new train segments: {new}", flush=True)
    for i, gps in enumerate(new, 1):
        print(f"[{i}/{len(new)}] fetching H1 {gps}...", flush=True)
        fetch_segment("H1", gps)
    manifest["H1"]["train"] = manifest["H1"]["train"] + new
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print("FETCH MORE DONE", flush=True)


if __name__ == "__main__":
    main()
