"""Discover and download the noise segments (Phase 1a).

H1: N_SEGMENTS_H1 segments for train/val/test.
L1: 2 segments overlapping the H1 test stretch where both were observing,
    for the Phase-5 coincidence demo.

Run:  uv run python scripts/fetch_noise.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gwosc.timeline import get_segments

from pbh import config as C
from pbh.data import discover_segments, fetch_segment

MANIFEST = C.DATA_DIR / "manifest.json"


def main() -> None:
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("discovering H1 segments...", flush=True)
    h1_starts = discover_segments("H1", C.N_SEGMENTS_H1)
    print(f"found {len(h1_starts)} H1 segments: {h1_starts[0]} .. {h1_starts[-1]}",
          flush=True)

    splits = {
        "train": h1_starts[: C.N_TRAIN_SEG],
        "val": h1_starts[C.N_TRAIN_SEG : C.N_TRAIN_SEG + C.N_VAL_SEG],
        "test": h1_starts[C.N_TRAIN_SEG + C.N_VAL_SEG :],
    }

    # L1 coincidence segments: pick test-split times where L1 was also observing
    print("finding L1 overlap for coincidence demo...", flush=True)
    l1_starts = []
    for gps in splits["test"]:
        segs = get_segments("L1_DATA", gps, gps + C.SEGMENT_LEN)
        if any(s <= gps and e >= gps + C.SEGMENT_LEN for s, e in segs):
            l1_starts.append(gps)
        if len(l1_starts) >= 2:
            break
    print(f"L1 coincident segments: {l1_starts}", flush=True)

    manifest = {"H1": splits, "L1": {"coinc": l1_starts}}
    MANIFEST.write_text(json.dumps(manifest, indent=2))

    jobs = [("H1", g) for g in h1_starts] + [("L1", g) for g in l1_starts]
    for i, (ifo, gps) in enumerate(jobs, 1):
        print(f"[{i}/{len(jobs)}] fetching {ifo} {gps}...", flush=True)
        fetch_segment(ifo, gps)
    print("ALL FETCHES DONE", flush=True)


if __name__ == "__main__":
    main()
