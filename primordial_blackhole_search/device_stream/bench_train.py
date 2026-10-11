"""Device feasibility benchmark for streaming training of cnn_w64 (SpectrogramCNN on 128x63 specs).
1) golden: load cnn_w64.pt, logits on a fixed numpy-seeded input -> compare against the Mac reference.
2) inference throughput, 3) full training step (AdamW + BCE, batch 64, the train.py recipe)."""
import json, sys, time, statistics
import numpy as np
import torch
from pbh.models import SpectrogramCNN

def load():
    m = SpectrogramCNN()
    sd = torch.load("cnn_w64.pt", map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    m.load_state_dict(sd)
    return m

x_ref = torch.from_numpy(np.random.default_rng(0).standard_normal((4, 1, 128, 63)).astype(np.float32))
mode = sys.argv[1] if len(sys.argv) > 1 else "bench"
m = load().eval()
with torch.no_grad():
    logits = m(x_ref).tolist()
if mode == "golden":
    print(json.dumps({"logits": logits}))
    sys.exit()

out = {"logits": logits, "torch": torch.__version__}
for threads in (8, 4):
    torch.set_num_threads(threads)
    xb = torch.randn(64, 1, 128, 63)
    yb = (torch.rand(64) > 0.5).float()
    m = load().eval()
    with torch.no_grad():
        for _ in range(2):
            m(xb)
        t = []
        for _ in range(5):
            t0 = time.perf_counter(); m(xb); t.append(time.perf_counter() - t0)
    inf = statistics.median(t)
    m = load().train()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-3)
    lossf = torch.nn.BCEWithLogitsLoss()
    t = []
    for i in range(8):
        t0 = time.perf_counter()
        opt.zero_grad(); loss = lossf(m(xb), yb); loss.backward(); opt.step()
        if i >= 2:
            t.append(time.perf_counter() - t0)
    step = statistics.median(t)
    out[f"threads{threads}"] = {
        "infer_ms_per_batch64": round(inf * 1e3, 1), "infer_samples_per_s": round(64 / inf, 1),
        "train_ms_per_step64": round(step * 1e3, 1), "train_samples_per_s": round(64 / step, 1),
        "train_samples_per_day": int(64 / step * 86400), "step_spread_ms": [round(min(t)*1e3,1), round(max(t)*1e3,1)],
    }
    print(threads, out[f"threads{threads}"], flush=True)
print("BENCH_JSON " + json.dumps(out))
