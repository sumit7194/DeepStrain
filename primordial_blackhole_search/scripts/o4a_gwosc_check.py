"""M11 launch pre-check: can GWOSC deliver O4a strain for BOTH detectors right now?

PASS RULE (fixed 2026-09-26, before the autopilot's first check): 64 s of H1 AND 64 s of L1 from pool segment #25
(a segment M11 still needs) each arrive complete and finite within 300 s. Anything else -- a timeout, an exception,
a short or non-finite read -- is a FAIL. Exit code 0 = PASS, 1 = FAIL. One line per detector is printed.
"""
import json
import signal
import sys
import time
from pathlib import Path

POOL = Path(__file__).resolve().parent.parent / "results" / "o4a_ssm" / "pool.json"
LIMIT_S = 300


class Timeout(Exception):
    pass


def _alarm(*_):
    raise Timeout()


def main() -> int:
    from gwpy.timeseries import TimeSeries
    signal.signal(signal.SIGALRM, _alarm)
    g = json.loads(POOL.read_text())["segments"][24] + 1024
    ok = True
    for d in ("H1", "L1"):
        t = time.time()
        signal.alarm(LIMIT_S)
        try:
            ts = TimeSeries.fetch_open_data(d, g, g + 64, sample_rate=4096, cache=False)
            good = len(ts) == 64 * 4096 and bool((abs(ts.value) < 1).all())
            print(f"{d} {g}: {len(ts)} samples, finite={good} in {time.time() - t:.1f}s", flush=True)
            ok &= good
        except Timeout:
            print(f"{d} {g}: TIMEOUT after {LIMIT_S}s", flush=True); ok = False
        except Exception as exc:                            # noqa: BLE001 -- any failure is a FAIL, reported
            print(f"{d} {g}: FAIL {type(exc).__name__}: {str(exc)[:80]}", flush=True); ok = False
        finally:
            signal.alarm(0)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
