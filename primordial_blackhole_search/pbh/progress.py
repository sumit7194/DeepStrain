"""Heartbeat for the repo dashboard (same pattern as echolib/rdlib)."""

from __future__ import annotations

import json
import time

from . import config as C

PROGRESS = C.RESULTS_DIR / "progress"


def progress(run: str, step: int, total: int, **metrics) -> None:
    PROGRESS.mkdir(parents=True, exist_ok=True)
    f = PROGRESS / f"{run}.json"
    hist = []
    if f.exists():
        try:
            hist = json.loads(f.read_text()).get("history", [])
        except json.JSONDecodeError:
            pass
    loss = metrics.get("loss")
    if loss is not None:
        hist = (hist + [[step, float(loss)]])[-200:]
    tmp = f.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(
            {"run": run, "step": int(step), "total": int(total),
             "metrics": {k: float(v) for k, v in metrics.items()},
             "history": hist, "updated": time.time()}
        )
    )
    tmp.replace(f)
    idx = PROGRESS / "index.json"
    try:
        names = json.loads(idx.read_text()) if idx.exists() else []
    except json.JSONDecodeError:
        names = []
    if run not in names:
        names.append(run)
        idx.write_text(json.dumps(names))
