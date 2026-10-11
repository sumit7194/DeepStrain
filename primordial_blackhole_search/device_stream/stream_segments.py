"""Segment pools for Arm V (fresh O3a H1 noise). Same selection rules as pbh.data.discover_segments
(H1_DATA flag, 256-s margin, GWTC-2.1-confident events +/-64 s excluded) over the whole O3a run, minus:
the 16 cnn_w64 train segments (Arm V must be disjoint from Arm C's noise) and anything within +/-12 h of a
val/test segment (slow-drift leakage guard). Shuffled with a fixed seed and split into disjoint per-seed pools."""
import json
import numpy as np
from gwosc import datasets
from gwosc.timeline import get_segments
from pbh import config as C

o3a = datasets.run_segment("O3a")
observing = get_segments("H1_DATA", *o3a)
events = datasets.find_datasets(type="events", catalog="GWTC-2.1-confident", segment=o3a)
event_gps = []
for ev in events:
    try:
        event_gps.append(datasets.event_gps(ev))
    except ValueError:
        pass
man = json.load(open(C.DATA_DIR / "manifest.json"))["H1"]
held = man["val"] + man["test"]
guard = 12 * 3600
starts, margin = [], 256
for a, b in observing:
    t = a + margin
    while t + C.SEGMENT_LEN + margin <= b:
        bad_event = any(t - 64 <= g <= t + C.SEGMENT_LEN + 64 for g in event_gps)
        near_held = any(t < h + C.SEGMENT_LEN + guard and h < t + C.SEGMENT_LEN + guard for h in held)
        if not bad_event and not near_held and int(t) not in man["train"]:
            starts.append(int(t))
        t += C.SEGMENT_LEN
rng = np.random.default_rng(20261007)
rng.shuffle(starts)
half = len(starts) // 2
out = {"o3a_span": list(o3a), "n_events_excluded": len(event_gps), "guard_s": guard,
       "n_available": len(starts), "hours_available": round(len(starts) * C.SEGMENT_LEN / 3600, 1),
       "pool_seed0": sorted(starts[:half]), "pool_seed1": sorted(starts[half:])}
json.dump(out, open("stream_segments.json", "w"), indent=0)
assert not set(out["pool_seed0"]) & set(out["pool_seed1"])
print({k: v for k, v in out.items() if not k.startswith("pool")}, "per-pool", half)
