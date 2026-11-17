"""LongChirp benchmark: discover the test/dev segments and fetch them, one file at a time, resumably.

  python benchmark/fetch.py discover   # writes benchmark/manifest.json (GPS lists only; network, small)
  python benchmark/fetch.py fetch      # fetches every (ifo, gps) missing from benchmark/data/, then records sha256

Each unit is one detector-segment, saved as float32 .npy and written atomically. State is in
benchmark/state/fetch_state.json. Touch benchmark/state/PAUSE to stop cleanly between units, and re-run to resume.
Rules (PLAN.md): free disk >= 5 GB, retry backoff 10/30/60 min, skip a file after 3 failures, stop after 3
consecutive failed files, and yield while the macOS 27.2 regression fetch is running.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

B = Path(__file__).resolve().parent
sys.path.insert(0, str(B.parent))
from pbh import config as C  # noqa: E402

DATA, STATE = B / "data", B / "state"
MANIFEST, STATE_F, PAUSE = B / "manifest.json", STATE / "fetch_state.json", STATE / "PAUSE"
N_TEST, N_DEV = 24, 8
TEST_FROM, O3A_END = 1249000000, 1253977218
DEV_FROM = C.SCAN_START + C.SCAN_DAYS * 86400  # after the baselines' training scan window
BACKOFF_S = (600, 1800, 3600)
MIN_FREE_GB = 5.0


def log(*a):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), *a, flush=True)


def atomic_json(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def coincident(t0, t1, n, exclude):
    from gwosc import datasets
    from gwosc.timeline import get_segments
    h1, l1 = get_segments("H1_DATA", t0, t1), get_segments("L1_DATA", t0, t1)
    ev = []
    for cat in ("GWTC-2.1-confident", "GWTC-2.1-marginal"):
        for e in datasets.find_datasets(type="events", catalog=cat, segment=(t0, t1)):
            try:
                ev.append(datasets.event_gps(e))
            except ValueError:
                pass
    out = []
    for s, e in h1:  # segments = whole GWOSC files (GPS multiples of 4096), so each is ONE download
        t = -(-s // C.SEGMENT_LEN) * C.SEGMENT_LEN
        while t + C.SEGMENT_LEN <= e:
            g = int(t)
            if (g not in exclude and any(ls <= g and le >= g + C.SEGMENT_LEN for ls, le in l1)
                    and not any(g - 64 <= x <= g + C.SEGMENT_LEN + 64 for x in ev)):
                out.append(g)
            t += C.SEGMENT_LEN
    if len(out) > n:  # spread the picks evenly over the window: more noise variety than the first n in a row
        out = [out[int(i)] for i in np.linspace(0, len(out) - 1, n)]
    return out, len(ev)


def discover():
    m = json.loads((C.DATA_DIR / "manifest.json").read_text())
    used = {g for d in m.values() for v in d.values() for g in v}
    test, n_ev_t = coincident(TEST_FROM, O3A_END, N_TEST, used)
    dev, n_ev_d = coincident(DEV_FROM, TEST_FROM, N_DEV, used)
    assert len(test) == N_TEST and len(dev) == N_DEV, (len(test), len(dev))
    overlap = [g for g in test + dev if any(abs(g - u) < C.SEGMENT_LEN for u in used)]
    assert not overlap, overlap
    atomic_json(MANIFEST, {"run": "O3a", "segment_len_s": C.SEGMENT_LEN, "sample_rate": C.SAMPLE_RATE,
                           "flags": ["H1_DATA", "L1_DATA"], "excluded_catalogs": ["GWTC-2.1-confident",
                           "GWTC-2.1-marginal"], "event_pad_s": 64, "test": test, "dev": dev, "sha256": {}})
    log(f"test {len(test)} ({test[0]}..{test[-1]}), dev {len(dev)} ({dev[0]}..{dev[-1]}); "
        f"events excluded in windows: {n_ev_t}/{n_ev_d}")


def free_gb():
    return shutil.disk_usage("/").free / 1e9


def regression_fetch_running():
    ps = subprocess.run(["ps", "-axo", "command"], capture_output=True, text=True).stdout
    return any("fetch_test_segments.py" in ln and "grep" not in ln for ln in ps.splitlines())


def download(url, dest, honour_pause=True):
    """curl with resume. GWOSC drops long transfers mid-file (2026-11-17: gwpy reported 'failed to get data from
    any source' for every file while 1 MB ranged reads worked), so a dropped connection that made progress is
    resumed, not counted as a failure. Only a pass with no new bytes raises. The transfer is interrupted (and later
    resumed from the partial file) on PAUSE or while the 27.2 regression fetch runs: one GWOSC transfer at a time."""
    while True:
        before = dest.stat().st_size if dest.exists() else 0
        p = subprocess.Popen(["curl", "-sfL", "-C", "-", "--connect-timeout", "60", "--speed-limit", "2000",
                              "--speed-time", "300", "-o", str(dest), url])
        while p.poll() is None:
            if honour_pause and (PAUSE.exists() or regression_fetch_running()):
                p.terminate(); p.wait()
                if PAUSE.exists():
                    raise SystemExit(log("PAUSE flag present: transfer interrupted, partial file kept for resume"))
                log("yielding mid-file: 27.2 regression fetch is running (partial file kept)")
                while regression_fetch_running() and not PAUSE.exists():
                    time.sleep(60)
                break
            time.sleep(15)
        if p.returncode == 0:
            return
        if p.returncode < 0:  # we interrupted it to yield; resume from the partial file
            continue
        r = p
        after = dest.stat().st_size if dest.exists() else 0
        if after <= before:
            raise RuntimeError(f"curl exit {r.returncode}, no progress at {after} B")
        log(f"  resumed transfer: {before} -> {after} B (curl exit {r.returncode})")
        time.sleep(30)


def fetch_one(ifo, gps):
    import h5py
    from gwosc.locate import get_urls
    urls = [u for u in get_urls(ifo, gps, gps + C.SEGMENT_LEN, sample_rate=C.SAMPLE_RATE)
            if u.endswith(f"-{gps}-{C.SEGMENT_LEN}.hdf5")]
    assert len(urls) == 1 and f"-{gps}-{C.SEGMENT_LEN}.hdf5" in urls[0], urls
    raw = DATA / urls[0].rsplit("/", 1)[1]
    download(urls[0], raw)
    raw_sha = hashlib.sha256(raw.read_bytes()).hexdigest()
    with h5py.File(raw, "r") as f:
        assert int(f["meta/GPSstart"][()]) == gps, f["meta/GPSstart"][()]
        x = np.asarray(f["strain/Strain"][()], dtype=np.float32)
        dq = np.asarray(f["quality/simple/DQmask"][()])
    assert x.size == C.SEGMENT_LEN * C.SAMPLE_RATE and np.isfinite(x).all(), (x.size, np.isfinite(x).mean())
    assert (dq & 1).all(), "DATA bit not set over the whole file"
    out = DATA / f"{ifo}_{gps}.npy"
    tmp = DATA / f"{ifo}_{gps}.tmp.npy"
    np.save(tmp, x)
    os.replace(tmp, out)
    raw.unlink()
    return {"gwosc_url": urls[0], "gwosc_sha256": raw_sha, "npy_sha256": hashlib.sha256(out.read_bytes()).hexdigest()}


def fetch():
    DATA.mkdir(exist_ok=True); STATE.mkdir(exist_ok=True)
    man = json.loads(MANIFEST.read_text())
    st = json.loads(STATE_F.read_text()) if STATE_F.exists() else {"failures": {}, "skipped": []}
    units = [(i, g) for g in man["test"] + man["dev"] for i in ("H1", "L1")]
    consecutive = 0
    for ifo, g in units:
        key = f"{ifo}_{g}"
        if key in man["sha256"] or key in st["skipped"]:
            continue
        while True:
            if PAUSE.exists():
                log("PAUSE flag present: stopping cleanly"); return 0
            if free_gb() < MIN_FREE_GB:
                log(f"STOP: free disk {free_gb():.1f} GB < {MIN_FREE_GB}"); return 3
            if regression_fetch_running():
                log("yielding: 27.2 regression fetch is running"); time.sleep(300); continue
            t = time.time()
            try:
                sha = fetch_one(ifo, g)
            except Exception as e:  # noqa: BLE001 -- every GWOSC failure mode is handled the same way: back off
                n = st["failures"].get(key, 0) + 1
                st["failures"][key] = n
                log(f"FAIL {key} attempt {n}: {type(e).__name__} {str(e)[:160]} ({time.time() - t:.0f} s)")
                if n >= len(BACKOFF_S):
                    st["skipped"].append(key); atomic_json(STATE_F, st); consecutive += 1
                    if consecutive >= 3:
                        log("STOP: 3 consecutive files failed; GWOSC likely degraded"); return 2
                    break
                atomic_json(STATE_F, st)
                for _ in range(BACKOFF_S[n - 1] // 30):
                    if PAUSE.exists():
                        break
                    time.sleep(30)
                continue
            man["sha256"][key] = sha
            atomic_json(MANIFEST, man); atomic_json(STATE_F, st)
            consecutive = 0
            log(f"ok {key} {time.time() - t:.0f} s | done {len(man['sha256'])}/{len(units)} | free {free_gb():.1f} GB")
            break
    log(f"FETCH COMPLETE: {len(man['sha256'])}/{len(units)} ok, skipped {st['skipped']}")
    return 0


if __name__ == "__main__":
    sys.exit({"discover": discover, "fetch": fetch}[sys.argv[1]]())
