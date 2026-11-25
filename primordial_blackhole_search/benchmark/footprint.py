"""Physical footprint of THIS process (what `top -stats mem` and `footprint` report), current and lifetime peak, read
through libproc's proc_pid_rusage. RSS badly under-reports MPS/Metal memory on this Mac (a paused job showed 36 MB RSS
and 5,837 MB MEM on 2026-11-26), so peaks are declared and recorded with this instead.   [model: Fable 5.2] UNVERIFIED

The field order of struct rusage_info_v4 is written from memory, so it is CHECKED, not trusted:
    python benchmark/footprint.py     allocates known amounts and compares against `top -l 1 -pid <self> -stats mem`.
"""
import ctypes
import os

_N = 36                                        # uint64 fields after the 16-byte uuid in rusage_info_v4
_I_FOOTPRINT, _I_PEAK = 7, 28                  # ri_phys_footprint, ri_lifetime_max_phys_footprint


class _Info(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16), ("v", ctypes.c_uint64 * _N)]


_lib = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)


def footprint_mb(pid=None):
    """(current, lifetime peak) physical footprint in MB."""
    info = _Info()
    rc = _lib.proc_pid_rusage(int(pid or os.getpid()), 4, ctypes.byref(info))
    if rc != 0:
        raise OSError(ctypes.get_errno(), "proc_pid_rusage failed")
    return info.v[_I_FOOTPRINT] / 2 ** 20, info.v[_I_PEAK] / 2 ** 20


def _top_mb():
    import subprocess
    out = subprocess.run(["top", "-l", "1", "-pid", str(os.getpid()), "-stats", "mem"], capture_output=True,
                         text=True).stdout.strip().splitlines()[-1].strip()
    mult = {"K": 1 / 1024, "M": 1.0, "G": 1024.0}[out.rstrip("+-")[-1]]
    return float(out.rstrip("+-")[:-1]) * mult


if __name__ == "__main__":
    import numpy as np
    rows, keep = [], []
    for add in (0, 300, 600):
        if add:
            keep.append(np.ones(add * 2 ** 20 // 8))          # touched, so it is resident
        cur, peak = footprint_mb()
        rows.append((add, cur, peak, _top_mb()))
        print(f"after +{add:4d} MB: footprint {cur:7.1f} MB, lifetime peak {peak:7.1f} MB, top MEM {rows[-1][3]:7.1f} MB")
    keep.clear()
    cur, peak = footprint_mb()
    top_after = _top_mb()
    print(f"after freeing : footprint {cur:7.1f} MB, lifetime peak {peak:7.1f} MB, top MEM {top_after:7.1f} MB "
          f"(the allocator may keep freed pages, so no drop is required)")
    ok = (all(abs(r[1] - r[3]) < 0.05 * r[3] + 8 for r in rows) and abs(cur - top_after) < 0.05 * top_after + 8
          and abs((rows[2][1] - rows[0][1]) - 900) < 60 and peak >= rows[2][1] - 1)
    print("footprint reader", "PASS" if ok else "FAIL", "(reader == top at every stage; +900 MB seen; peak >= largest)")
