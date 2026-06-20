"""Build C-2 (GPU VM): a LEARNED coincidence statistic — can exploiting H1-L1
consistency beat the plain sum?

The sum statistic (sH1+sL1) ignores that a real signal has CORRELATED morphology
across detectors while a glitch-coincidence does not. Here a small head is trained
on the per-detector cnn_w64 EMBEDDINGS (256-d penultimate features, not the scalar
score) to separate REAL coincident signals (injections) from ACCIDENTAL ones
(time-slide pairs). Then the same FAR sweep as coinc_far compares the learned
statistic's sensitive distance to the sum baseline.

Honest fail condition: if it does not beat sum, that's the G2a finding extended
(the operating point is tail-separation limited, not combination-rule limited).

Run:  python coinc_learned.py [--n-inj 6000] [--slides 4000]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pbh import config as C
from pbh.models import make_model
from coinc_eval import bin_snr50
from coinc_far import FARS, SEED, WIN, _segment_worker  # reuse the parallel feature pipeline


@torch.no_grad()
def embed(model, device, feats):
    """256-d penultimate embeddings (model.net[:6] = up to Flatten) + scalar scores."""
    emb, sc = [], []
    for i in range(0, len(feats), 4096):
        b = torch.from_numpy(feats[i:i+4096]).float().unsqueeze(1).to(device)
        e = model.net[:6](b)
        emb.append(e.cpu().numpy()); sc.append(model.net[6:](e).squeeze(-1).cpu().numpy())
    return np.concatenate(emb), np.concatenate(sc)


def pair_feats(eH, eL):
    """Symmetric H1-L1 consistency features: [eH, eL, |eH-eL|, eH*eL] -> 4*256."""
    return np.concatenate([eH, eL, np.abs(eH - eL), eH * eL], axis=1).astype(np.float32)


class CoincHead(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4*d, 128), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-inj", type=int, default=6000)
    ap.add_argument("--slides", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_model("cnn")
    model.load_state_dict(torch.load(C.MODEL_DIR / "cnn_w64.pt", map_location=dev))
    model.to(dev).eval()
    segs = json.loads((C.DATA_DIR / "manifest_far.json").read_text())["coinc"]
    per = max(1, args.n_inj // len(segs))
    print(f"dev {dev} | {len(segs)} segs | {per}/seg inj | {args.slides} slides", flush=True)

    t0 = time.time()
    jobs = [(g, per, SEED + 555 + i) for i, g in enumerate(segs)]
    res = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(_segment_worker, jobs):
            res[r[0]] = r
            print(f"  seg {len(res)}/{len(segs)} ({time.time()-t0:.0f}s)", flush=True)

    # --- embeddings: noise (pooled, aligned) + injections ---
    nH, nL, iH, iL, meta = [], [], [], [], []
    for g in segs:
        _, noiseH, noiseL, fH, fL, metas = res[g]
        eH, _ = embed(model, dev, noiseH); eL, _ = embed(model, dev, noiseL)
        n = min(len(eH), len(eL)); nH.append(eH[:n]); nL.append(eL[:n])
        ejH, sjH = embed(model, dev, fH); ejL, sjL = embed(model, dev, fL)
        iH.append(ejH); iL.append(ejL)
        meta += [dict(chirp_mass=mc, target_snr=t, sH1=float(a), sL1=float(b))
                 for (mc, t), a, b in zip(metas, sjH, sjL)]
    nH = np.concatenate(nH); nL = np.concatenate(nL); N = len(nH)
    iH = np.concatenate(iH); iL = np.concatenate(iL)
    df = pd.DataFrame(meta); df["coinc_sum"] = df.sH1 + df.sL1
    T_real = N * 64.0
    print(f"\npooled {N} noise windows | {len(df)} injections ({time.time()-t0:.0f}s)", flush=True)

    # --- train/eval split on injections (no leakage) ---
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(df)); ntr = int(0.6 * len(df))
    tr, ev = idx[:ntr], idx[ntr:]
    # positives = real coincident injections; negatives = time-slide noise pairs
    Xpos = pair_feats(iH[tr], iL[tr])
    k = rng.integers(1, N, size=len(tr))
    Xneg = pair_feats(nH[:len(tr)], nL[(np.arange(len(tr)) + k) % N])
    X = np.concatenate([Xpos, Xneg]); y = np.concatenate([np.ones(len(tr)), np.zeros(len(tr))])
    mu, sd = X.mean(0), X.std(0) + 1e-6                       # standardize
    head = CoincHead().to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-3)
    lf = nn.BCEWithLogitsLoss()
    Xt = torch.tensor((X - mu) / sd).to(dev); yt = torch.tensor(y, dtype=torch.float32).to(dev)
    print("training coincidence head ...", flush=True)
    for ep in range(60):
        head.train(); p = torch.randperm(len(Xt))
        for b in range(0, len(Xt), 1024):
            j = p[b:b+1024]; opt.zero_grad()
            loss = lf(head(Xt[j]), yt[j]); loss.backward(); opt.step()
    head.eval()

    @torch.no_grad()
    def learned_stat(eH, eL):
        x = torch.tensor((pair_feats(eH, eL) - mu) / sd).float().to(dev)
        return head(x).cpu().numpy()

    # --- learned statistic on eval injections + time-slide background ---
    df_ev = df.iloc[ev].copy()
    df_ev["coinc_learned"] = learned_stat(iH[ev], iL[ev])
    bg_l = np.concatenate([learned_stat(nH, np.roll(nL, kk, axis=0))
                           for kk in range(1, args.slides + 1)])
    bg_l.sort()
    # sum-statistic background: pooled scalar noise scores (same windows), time-slid
    sNH, sNL = [], []
    for g in segs:
        nh = embed(model, dev, res[g][1])[1]; nl = embed(model, dev, res[g][2])[1]
        n = min(len(nh), len(nl)); sNH.append(nh[:n]); sNL.append(nl[:n])
    sNH = np.concatenate(sNH); sNL = np.concatenate(sNL)
    bg_s = np.concatenate([sNH + np.roll(sNL, kk) for kk in range(1, args.slides + 1)]); bg_s.sort()
    T_bg = args.slides * T_real

    def thr(bg, far):
        return float(bg[-max(1, int(round(far * T_bg)))])

    ML = ("0.17-0.35", "0.35-0.55", "0.55-0.88")
    print(f"\n=== learned vs sum coincidence: sensitive distance vs FAR ({len(df_ev)} eval inj) ===")
    print(f"{'FAR':>8} | {'sum':>32} | {'learned':>32}")
    out = {}
    for name, far in FARS.items():
        if far * T_bg < 1:
            continue
        df_ev["_s"] = df_ev.coinc_sum > thr(bg_s, far)
        df_ev["_l"] = df_ev.coinc_learned > thr(bg_l, far)
        fs = bin_snr50(df_ev, "_s", 8); fl = bin_snr50(df_ev, "_l", 8)
        out[name] = {"sum": {m: fs[m][1] for m in ML}, "learned": {m: fl[m][1] for m in ML}}
        ss = " ".join(f"{fs[m][1]:.3f}" for m in ML); ll = " ".join(f"{fl[m][1]:.3f}" for m in ML)
        print(f"{name:>8} | {ss:>32} | {ll:>32}")

    (C.RESULTS_DIR / "coinc_learned.json").write_text(json.dumps(
        {"n_eval_inj": int(len(df_ev)), "slides": args.slides, "bg_days": T_bg/86400,
         "vs_far": out}, indent=2))
    # verdict
    hi = "0.55-0.88"
    wins = sum(out[n]["learned"][hi] > out[n]["sum"][hi] + 0.01 for n in out)
    print(f"\nVERDICT: learned beats sum (high-mass, >+0.01) at {wins}/{len(out)} FARs "
          + ("-> LEARNED COINCIDENCE HELPS" if wins >= len(out)//2 else "-> sum already optimal (honest)"))
    print("wrote coinc_learned.json")


if __name__ == "__main__":
    main()
