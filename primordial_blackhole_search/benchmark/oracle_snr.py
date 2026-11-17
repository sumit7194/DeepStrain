"""Y0 oracle: optimal network SNR of every injection (true template, real PSD of its own test segment, band [40, 1024]
Hz, the project's whitened-domain convention verified against pycbc sigma in golden_injection.py). A ceiling, not a
pipeline. Resumable: one checkpoint file per segment in benchmark/state/oracle/.

  python benchmark/oracle_snr.py [segment ...]     (default: every test segment whose H1 and L1 are on disk)
Writes benchmark/results/oracle_snr.csv when all segments are done.
"""
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

B = Path(__file__).resolve().parent
sys.path.insert(0, str(B.parent)); sys.path.insert(0, str(B))
from pbh.data import estimate_psd, whiten  # noqa: E402
import injections as I  # noqa: E402

CK = B / "state" / "oracle"


def segment(g, rows):
    out = {}
    for d in ("H1", "L1"):
        x = np.load(B / "data" / f"{d}_{g}.npy").astype(np.float64)
        psd = estimate_psd(x)
        for r in rows:
            sig, t0 = I.detector_signal(r, d)
            out.setdefault(r["id"], {})[d] = float(np.sqrt(np.sum(whiten(I.add_signal(np.zeros_like(x), g, sig, t0), psd) ** 2)))
    return {i: dict(v, net=float(np.hypot(v["H1"], v["L1"]))) for i, v in out.items()}


def main():
    CK.mkdir(parents=True, exist_ok=True)
    with open(I.TABLE) as f:
        rows = [{k: (float(v) if k not in ("id", "realization", "segment") else int(v)) for k, v in r.items()}
                for r in csv.DictReader(f)]
    segs = [int(a) for a in sys.argv[1:]] or json.loads((B / "manifest.json").read_text())["test"]
    for g in segs:
        ck = CK / f"{g}.json"
        if ck.exists() or not all((B / "data" / f"{d}_{g}.npy").exists() for d in ("H1", "L1")):
            continue
        res = segment(g, [r for r in rows if r["segment"] == g])
        tmp = ck.with_suffix(".tmp"); tmp.write_text(json.dumps(res)); os.replace(tmp, ck)
        print(g, "done", len(res), flush=True)
    done = {int(k): v for p in CK.glob("*.json") for k, v in json.loads(p.read_text()).items()}
    if len(done) == len(rows):
        (B / "results").mkdir(exist_ok=True)
        with open(B / "results" / "oracle_snr.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["id", "snr_H1", "snr_L1", "snr_net"])
            for i in sorted(done):
                w.writerow([i, *(round(done[i][k], 4) for k in ("H1", "L1", "net"))])
        print("ORACLE COMPLETE", len(done))


if __name__ == "__main__":
    main()
