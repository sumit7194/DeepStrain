"""Build a chirp bank: one .npz per (m1, m2) holding raw face-on hp0, hc0 (float32) — see bank.py.
Masses follow pbh.waveforms.sample_params (log-uniform [M_MIN, M_MAX] each). Usage: build_bank.py OUT_DIR N SEED"""
import json, sys, time
from pathlib import Path
import numpy as np
from pbh import config as C
from bank import face_on

out, n, seed = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(seed)
index = []
for k in range(n):
    m1, m2 = (float(np.exp(rng.uniform(np.log(C.M_MIN), np.log(C.M_MAX)))) for _ in range(2))
    f = out / f"w{k:05d}.npz"
    if not f.exists():
        hp0, hc0 = face_on(m1, m2)
        tmp = out / f".w{k:05d}.npz"
        np.savez(tmp, hp0=hp0, hc0=hc0, m1=m1, m2=m2)
        tmp.rename(f)
    index.append(dict(file=f.name, m1=m1, m2=m2, mc=(m1 * m2) ** 0.6 / (m1 + m2) ** 0.2))
json.dump(index, open(out / "index.json", "w"), indent=1)
print("done", n)
