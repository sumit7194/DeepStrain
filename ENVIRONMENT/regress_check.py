"""OS-update regression spot check: re-score the 384 O3a test noise windows with cnn_w64 on MPS and on CPU and
compare with the cached pre-update scores (results/track_w64/noise_<gps>.npy, made on MPS). Single-threaded.
Usage (from primordial_blackhole_search/): .venv/bin/python ../ENVIRONMENT/regress_check.py OUT.json"""
import json, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parent.parent / "primordial_blackhole_search"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
torch.set_num_threads(1)
import decoy_test as D  # noqa: E402
from pbh import config as C  # noqa: E402
from pbh.data import whiten_segment  # noqa: E402
from pbh.models import make_model  # noqa: E402
from pbh.sweep import CROP, score_windows  # noqa: E402
test = json.loads((C.DATA_DIR / "manifest.json").read_text())["H1"]["test"]
whitened = {g: whiten_segment("H1", g)[0] for g in test}
out = {}
for dev in ("mps", "cpu"):
    m = make_model("cnn"); m.load_state_dict(torch.load(C.MODEL_DIR / "cnn_w64.pt", map_location=dev, weights_only=True)); m.to(dev).eval()
    diffs, exact = [], 0
    for g in test:
        ref = np.load(C.RESULTS_DIR / "track_w64" / f"noise_{g}.npy")
        w = whitened[g]; h = D.GRID.win_hop_samp
        mine = score_windows(m, dev, np.stack([D.model_in(w[CROP + k * h: CROP + k * h + D.WIN]) for k in range(len(ref))]).astype(np.float32))
        diffs.append(np.abs(mine - ref)); exact += int(np.sum(mine == ref))
    d = np.concatenate(diffs)
    out[dev] = dict(n=int(d.size), max_abs_diff=float(d.max()), bit_identical=int(exact))
    print(dev, out[dev], flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
