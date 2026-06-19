"""Build C, step 1 (GPU VM): discover + fetch many H1+L1 COINCIDENT O3a segments
for a realistic-FAR coincidence study.

Finds SEGMENT_LEN stretches where BOTH detectors observed, clean (no catalogued
event), and NOT in the cnn_w64 training/val set (no leakage). Two modes:
  (no args)   discover -> write data/manifest_far.json + print the job count
  --job=I     fetch job I (one detector-segment) -- driven by xargs -P for speed

Run:  python fetch_coinc.py
      N=$(python fetch_coinc.py | grep -oP 'jobs: \K[0-9]+'); \
      seq 0 $((N-1)) | xargs -P8 -I{} python fetch_coinc.py --job={}
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pbh import config as C
from pbh.data import fetch_segment

N_WANT = 24                 # coincident segments (each 4096 s -> ~27 h coincident livetime)
SCAN_DAYS = 150             # widen the O3a scan to find enough fresh coincident stretches
FAR_MANIFEST = C.DATA_DIR / "manifest_far.json"


def discover_coincident(n_want, exclude):
    from gwosc import datasets
    from gwosc.timeline import get_segments
    scan_end = C.SCAN_START + SCAN_DAYS * 86400
    h1 = get_segments("H1_DATA", C.SCAN_START, scan_end)
    l1 = get_segments("L1_DATA", C.SCAN_START, scan_end)
    events = datasets.find_datasets(type="events", catalog="GWTC-2.1-confident",
                                    segment=(C.SCAN_START, scan_end))
    ev = []
    for e in events:
        try:
            ev.append(datasets.event_gps(e))
        except ValueError:
            pass
    margin = 256
    starts = []
    for s, e in h1:
        t = s + margin
        while t + C.SEGMENT_LEN + margin <= e:
            gps = int(t)
            l1_ok = any(ls <= gps and le >= gps + C.SEGMENT_LEN for ls, le in l1)
            clean = not any(gps - 64 <= g <= gps + C.SEGMENT_LEN + 64 for g in ev)
            if gps not in exclude and l1_ok and clean:
                starts.append(gps)
                if len(starts) >= n_want:
                    return starts
            t += C.SEGMENT_LEN
    return starts


def main():
    if len(sys.argv) > 1 and sys.argv[1].startswith("--job"):
        i = int(sys.argv[1].split("=")[1])
        segs = json.loads(FAR_MANIFEST.read_text())["coinc"]
        jobs = [(d, g) for g in segs for d in ("H1", "L1")]
        d, g = jobs[i]
        fetch_segment(d, g)
        print(f"fetched {d} {g}", flush=True)
        return
    manifest = json.loads((C.DATA_DIR / "manifest.json").read_text())
    exclude = set(manifest["H1"]["train"] + manifest["H1"]["val"])
    print("discovering H1xL1 coincident segments (excluding cnn_w64 train/val)...", flush=True)
    segs = discover_coincident(N_WANT, exclude)
    FAR_MANIFEST.write_text(json.dumps({"coinc": segs, "scan_days": SCAN_DAYS}, indent=2))
    print(f"found {len(segs)} coincident segments: {segs[0]}..{segs[-1]}")
    print(f"wrote {FAR_MANIFEST.name}; fetch jobs: {2*len(segs)}")


if __name__ == "__main__":
    main()
