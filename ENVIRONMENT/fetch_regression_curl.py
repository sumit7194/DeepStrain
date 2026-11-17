"""Fetch the 6 O3a H1 test segments for the macOS 27.2 regression by resumable curl (replaces fetch_test_segments.py).

Why the route changed (2026-11-18): GWOSC was up but dropped long transfers mid-file; gwpy's fetch_open_data treats a
drop as "failed to get data from any source" and restarts from zero, so every gwpy attempt failed 6/6. curl -C resumes.

Integrity (GWOSC publishes no checksum): every raw file must (1) match the API's filesize_bytes and the HTTP
Content-Length, (2) open as HDF5 with the expected GPS start and 4096-s duration, and (3) reproduce the API's published
min/max/mean/stdev of the strain (NaN-aware) to 1e-6 relative. The segments are then cut out exactly as gwpy would
cut them ([gps, gps+4096) at 4096 Hz), checked finite, and written to data/noise/ in the format pbh.data reads.
Writes ENVIRONMENT/fetch_regression_curl.json.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parent.parent / "primordial_blackhole_search"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "benchmark"))
from pbh import config as C  # noqa: E402
from pbh.data import segment_path  # noqa: E402
from fetch import download, log  # noqa: E402

RAW = ROOT / "benchmark" / "data" / "regress_raw"
FS, L = C.SAMPLE_RATE, C.SEGMENT_LEN


def verify(meta, path):
    import h5py
    size = path.stat().st_size
    clen = int(requests.head(meta["hdf5_url"], allow_redirects=True, timeout=60).headers["Content-Length"])
    with h5py.File(path, "r") as f:
        x = f["strain/Strain"][()]
        start, dur = int(f["meta/GPSstart"][()]), int(f["meta/Duration"][()])
    pub = {k: meta[k] for k in ("min_strain", "max_strain", "mean_strain", "stdev_strain")}
    got = {"min_strain": np.nanmin(x), "max_strain": np.nanmax(x), "mean_strain": np.nanmean(x),
           "stdev_strain": np.nanstd(x)}
    rel = {k: float(abs(got[k] / pub[k] - 1)) for k in pub}
    checks = {"size_api": size == meta["filesize_bytes"], "size_http": size == clen,
              "gps": start == meta["gps_start"], "duration": dur == L, "stats": max(rel.values()) < 1e-6}
    return checks, {"bytes": size, "content_length": clen, "stats_rel_err": rel}, x


def main():
    from gwosc.api.v2 import fetch_run_strain_files
    from gwpy.timeseries import TimeSeries
    RAW.mkdir(parents=True, exist_ok=True)
    segs = json.loads((C.DATA_DIR / "manifest.json").read_text())["H1"]["test"]
    files = {m["gps_start"]: m for m in fetch_run_strain_files("O3a", detector="H1", start=min(segs) - L,
                                                               end=max(segs) + L, sample_rate=FS)}
    need = sorted({s - (s % L) for s in segs} | {s - (s % L) + L for s in segs})
    rec, data = {"files": {}, "segments": {}}, {}
    for g in need:
        meta = requests.get(files[g]["detail_url"], timeout=60).json()
        dest = RAW / meta["hdf5_url"].rsplit("/", 1)[1]
        log(f"fetching {dest.name} ({meta['filesize_bytes']} B)")
        download(meta["hdf5_url"], dest, honour_pause=False)
        checks, info, x = verify(meta, dest)
        rec["files"][g] = {"checks": checks, **info}
        log(f"verify {g}: {checks}")
        if not all(checks.values()):
            raise SystemExit(f"INTEGRITY FAIL {g}: {checks} {info}")
        data[g] = x
    for s in segs:
        a = s - (s % L)
        x = np.concatenate([data[a], data[a + L]])[(s - a) * FS:(s - a + L) * FS]
        assert x.size == L * FS and np.isfinite(x).all(), s
        TimeSeries(x, t0=s, sample_rate=FS, name="H1:GWOSC-4KHZ_R1_STRAIN").write(
            segment_path("H1", s), path="strain", format="hdf5")
        rec["segments"][s] = {"samples": int(x.size), "finite": True}
        log(f"wrote {segment_path('H1', s).name}")
    for p in RAW.glob("*.hdf5"):
        p.unlink()
    (Path(__file__).parent / "fetch_regression_curl.json").write_text(json.dumps(rec, indent=1, default=str))
    log("REGRESSION FETCH COMPLETE, integrity verified")


if __name__ == "__main__":
    main()
