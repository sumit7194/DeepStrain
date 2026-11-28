"""Unplanned recovery step (approved by the Bridge, 2026-11-28; run notes): regenerate the 48 real-noise series
(ww_*.npy) that the L128 queue's dropseries step deleted before K1, so the dec_sg float64 recompute can finish, then
delete them again. Built so that NOTHING the running queue reads is written:
  * the series are rebuilt by the same coh2c_local.prep(), in a child process whose STATE points at a scratch dir on
    the same volume (state/coh2c_regen/), never at state/coh2c/ where K1 reads wsnip_*.npy;
  * hash gate: the regenerated ww, wsnip and rows must ALL equal results/coh2t/series_sha256.json; any mismatch STOPS
    (no workaround);
  * only ww_*.npy is moved into state/coh2c/, by an atomic same-volume rename; the live wsnip/rows are never written
    and are re-hashed against the record at the end.
Why not the driver's prepseries: coh2c_local.load_ww() runs prep only when rows_*.json is missing, and dropseries
keeps rows/wsnip, so prepseries would skip prep and then fail to load the absent series.

  python benchmark/coh2t_regen_series.py regen      # all 48, one at a time; resumable (skips ww already in place)
  python benchmark/coh2t_regen_series.py drop       # re-hash each ww against the record, then delete it (as dropseries)
  python benchmark/coh2t_regen_series.py check      # re-hash the live wsnip/rows (and any ww) against the record
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

B = Path(__file__).resolve().parent
LIVE = B / "state" / "coh2c"
REGEN = B / "state" / "coh2c_regen"
WANT = json.loads((B / "results" / "coh2t" / "series_sha256.json").read_text())
SEGS = json.loads((B / "manifest.json").read_text())["test"]
ENV = dict(os.environ, OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def names(det, g):
    t = f"{det}_{g}"
    return f"ww_{t}.npy", f"wsnip_{t}.npy", f"rows_{t}.json"


def regen():
    REGEN.mkdir(parents=True, exist_ok=True)
    assert REGEN.resolve() != LIVE.resolve()
    for det in ("H1", "L1"):
        for g in SEGS:
            ww, wsnip, rows = names(det, g)
            if (LIVE / ww).exists():
                assert sha(LIVE / ww) == WANT[ww], f"{ww} present in the live dir but does not match the record; STOP"
                continue
            code = (f"import sys; sys.path.insert(0, {str(B)!r}); from pathlib import Path; import coh2c_local as L; "
                    f"L.STATE = Path({str(REGEN)!r}); L.prep({det!r}, {int(g)})")
            subprocess.run([sys.executable, "-u", "-c", code], check=True, env=ENV, cwd=str(B.parent))
            for n in (ww, wsnip, rows):
                got = sha(REGEN / n)
                assert got == WANT[n], f"{n}: regenerated sha256 {got} != recorded {WANT[n]}; STOP, nothing moved"
            assert not (LIVE / ww).exists(), f"{ww} appeared in the live dir; refusing to clobber it; STOP"
            os.replace(REGEN / ww, LIVE / ww)                     # atomic: same volume
            print(f"rename {REGEN / ww} -> {LIVE / ww}", flush=True)
            assert sha(LIVE / ww) == WANT[ww]
            (REGEN / wsnip).unlink()                              # scratch duplicates of the live files
            (REGEN / rows).unlink()
            print(f"regen {ww}: bit-identical to the record; moved into place", flush=True)
    left = sorted(p.name for p in REGEN.iterdir())
    assert not left, f"unexpected files left in the scratch dir: {left}"
    REGEN.rmdir()                                                 # no stray second copy of the series remains
    print("regen: all 48 series in place, every one bit-identical to series_sha256.json; scratch dir removed", flush=True)


def drop():
    for det in ("H1", "L1"):                                      # dropseries' rule: hash every one first ...
        for g in SEGS:
            ww = names(det, g)[0]
            assert sha(LIVE / ww) == WANT[ww], f"{ww} changed since it was recorded; not deleting"
    for det in ("H1", "L1"):                                      # ... then delete
        for g in SEGS:
            (LIVE / names(det, g)[0]).unlink()
    print("drop: 48 ww series re-hashed against the record, then deleted (no queue marker written)", flush=True)


def check():
    bad = [n for det in ("H1", "L1") for g in SEGS for n in names(det, g)[1:] if sha(LIVE / n) != WANT[n]]
    present = [names(det, g)[0] for det in ("H1", "L1") for g in SEGS if (LIVE / names(det, g)[0]).exists()]
    print(json.dumps(dict(live_wsnip_rows_mismatch=bad, ww_present=len(present))), flush=True)
    assert not bad


if __name__ == "__main__":
    {"regen": regen, "drop": drop, "check": check}[sys.argv[1]]()
