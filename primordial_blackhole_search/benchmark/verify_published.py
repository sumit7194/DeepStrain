"""Check every benchmark file against the statistics GWOSC publishes for it (GWOSC publishes no checksum).

  benchmark/run.sh verify benchmark/verify_published.py

fetch.py already checked HDF5 structure, GPS start and the DATA quality bit, and deleted the raw HDF5 after saving
float32 .npy. What can still be checked is content: the API's min/max/mean/stdev of the strain must be reproduced by
the saved array. float32 storage limits agreement to ~1e-7 relative for min/max/stdev, so the bar there is 1e-6
relative; the mean is ~0 relative to the spread, so it is compared in units of stdev (|d mean| < 1e-6 stdev).
The API's filesize_bytes is compared with the HTTP Content-Length of the same URL (consistency of the source); our
own downloaded raw size cannot be re-checked because the raw file is gone, but its sha256 is in manifest.json.
Writes benchmark/results/verify_published.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import requests

B = Path(__file__).resolve().parent
sys.path.insert(0, str(B.parent))
from pbh import config as C  # noqa: E402


def main():
    from gwosc.api.v2 import fetch_run_strain_files
    man = json.loads((B / "manifest.json").read_text())
    segs = sorted(man["test"] + man["dev"])
    rec, ok = {}, True
    for det in ("H1", "L1"):
        files = {m["gps_start"]: m for m in fetch_run_strain_files("O3a", detector=det, start=min(segs),
                                                                   end=max(segs) + C.SEGMENT_LEN - 1,
                                                                   sample_rate=C.SAMPLE_RATE)}
        for g in segs:
            key = f"{det}_{g}"
            meta = requests.get(files[g]["detail_url"], timeout=60).json()
            assert meta["hdf5_url"] == man["sha256"][key]["gwosc_url"], key
            clen = int(requests.head(meta["hdf5_url"], allow_redirects=True, timeout=60).headers["Content-Length"])
            x = np.load(B / "data" / f"{key}.npy").astype(np.float64)
            got = dict(min_strain=x.min(), max_strain=x.max(), stdev_strain=x.std())
            rel = {k: float(abs(got[k] / meta[k] - 1)) for k in got}
            dmean = float(abs(x.mean() - meta["mean_strain"]) / meta["stdev_strain"])
            checks = dict(gps=meta["gps_start"] == g, samples=x.size == C.SEGMENT_LEN * C.SAMPLE_RATE,
                          size_api_vs_http=meta["filesize_bytes"] == clen, stats=max(rel.values()) < 1e-6,
                          mean=dmean < 1e-6)
            rec[key] = dict(checks=checks, stats_rel_err=rel, mean_err_in_stdev=dmean, filesize_bytes=clen)
            ok &= all(checks.values())
            print(key, "ok" if all(checks.values()) else f"FAIL {checks}", f"max rel {max(rel.values()):.1e}",
                  f"dmean/sd {dmean:.1e}", flush=True)
    out = dict(n_files=len(rec), all_pass=bool(ok), files=rec)
    (B / "results" / "verify_published.json").write_text(json.dumps(out, indent=1))
    print("VERIFY", "PASS" if ok else "FAIL", len(rec), "files")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
