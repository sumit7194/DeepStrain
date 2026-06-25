"""N4 step 2 (PLAN.md): does the SSL-pretrained backbone beat from-scratch at SCARCE labels? (the data-wall test)

Takes the masked-autoencoder encoder (ssl_pretrain.py → ssl_encoder.pt, pretrained on 20k UNLABELED noise
spectrograms) and fine-tunes the full SpectrogramCNN (pretrained backbone + fresh head) on a REDUCED labeled
budget, vs the identical model trained from random init on the SAME budget. If pretrained > from-scratch when
labels are scarce, self-supervision on abundant unlabeled noise beats the labeled-data wall. Input standardized
with the SSL mu/sd (the encoder's convention) for BOTH models — a fair comparison.

Run:  .venv/bin/python scripts/ssl_finetune.py [--budgets 1500 4000] [--epochs 40] [--smoke]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pbh import config as C
from pbh.models import make_model
from pbh.progress import progress

SHARDS = C.DATA_DIR / "shards_w64"
CKPT = C.MODEL_DIR / "ssl_encoder.pt"


def auc(scores, labels):
    s, y = np.asarray(scores), np.asarray(labels)
    r = s.argsort().argsort()
    n1 = int(y.sum()); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def load_split(split, mu, sd):
    x = np.load(SHARDS / f"x_{split}.npy", mmap_mode="r")
    y = pd.read_parquet(SHARDS / f"meta_{split}.parquet").label.to_numpy().astype(np.float32)
    return x, y, mu, sd


def train_model(model, idx, x, y, mu, sd, dev, epochs, rng):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lf = nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        model.train(); perm = rng.permutation(idx)
        for b in range(0, len(perm), 128):
            sel = np.sort(perm[b:b + 128])
            xb = (np.asarray(x[sel], dtype=np.float32) - mu) / sd
            xb = xb + rng.normal(0, 0.05, xb.shape).astype(np.float32)        # light augmentation
            xb = torch.tensor(xb, device=dev).unsqueeze(1); yb = torch.tensor(y[sel], device=dev)
            opt.zero_grad(); loss = lf(model(xb), yb); loss.backward(); opt.step()
        sched.step()


@torch.no_grad()
def val_auc(model, xv, yv, mu, sd, dev):
    model.eval(); scores = []
    for b in range(0, len(yv), 512):
        xb = torch.tensor((np.asarray(xv[b:b + 512], dtype=np.float32) - mu) / sd, device=dev).unsqueeze(1)
        scores.append(model(xb).cpu().numpy())
    return auc(np.concatenate(scores), yv)


def make_pretrained(dev):
    ck = torch.load(CKPT, map_location=dev)
    model = make_model("cnn").to(dev)
    enc = nn.Sequential(*list(model.net[:4]))      # the 4 conv blocks = the MAE encoder
    enc.load_state_dict(ck["encoder"])             # load pretrained weights in-place
    return model, ck["mu"], ck["sd"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", type=int, nargs="+", default=[1500, 4000])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    epochs = 4 if args.smoke else args.epochs
    budgets = [800] if args.smoke else args.budgets
    seeds = 1 if args.smoke else args.seeds

    ck = torch.load(CKPT, map_location="cpu")
    mu, sd = ck["mu"], ck["sd"]
    xt, yt, _, _ = load_split("train", mu, sd)
    xv, yv, _, _ = load_split("val", mu, sd)
    pos, neg = np.where(yt == 1)[0], np.where(yt == 0)[0]
    print(f"N4 fine-tune: pretrained ({ck['n_pretrain']} unlabeled) vs from-scratch | val n={len(yv)} | dev {dev}\n", flush=True)

    out = {}
    print(f"{'budget':>7} | {'scratch AUC':>22} | {'pretrained AUC':>22} | Δ")
    for budget in budgets:
        sc, pr = [], []
        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            k = budget // 2
            idx = np.concatenate([rng.choice(pos, k, replace=False), rng.choice(neg, k, replace=False)])
            # from scratch
            torch.manual_seed(seed); ms = make_model("cnn").to(dev)
            train_model(ms, idx, xt, yt, mu, sd, dev, epochs, np.random.default_rng(seed))
            sc.append(val_auc(ms, xv, yv, mu, sd, dev))
            # SSL-pretrained backbone
            torch.manual_seed(seed); mp, _, _ = make_pretrained(dev)
            train_model(mp, idx, xt, yt, mu, sd, dev, epochs, np.random.default_rng(seed))
            pr.append(val_auc(mp, xv, yv, mu, sd, dev))
            progress("ssl_finetune", budgets.index(budget) * seeds + seed + 1, len(budgets) * seeds)
        sc, pr = np.array(sc), np.array(pr)
        out[budget] = {"scratch_auc": sc.tolist(), "pretrained_auc": pr.tolist(),
                       "scratch_mean": float(sc.mean()), "pretrained_mean": float(pr.mean()),
                       "delta_mean": float(pr.mean() - sc.mean())}
        print(f"{budget:>7} | {sc.mean():.4f} ± {sc.std():.4f} ({seeds}s) | "
              f"{pr.mean():.4f} ± {pr.std():.4f} ({seeds}s) | {pr.mean()-sc.mean():+.4f}", flush=True)

    helps = all(o["delta_mean"] > 0 for o in out.values())
    biggest = max(out.values(), key=lambda o: o["delta_mean"])
    print(f"\nVERDICT: SSL pretraining beats from-scratch at every tested budget -> "
          f"{'YES' if helps else 'MIXED/NO'} (biggest gain Δ={biggest['delta_mean']:+.4f}); "
          f"{'data-wall win' if helps else 'inconclusive at this scale'}")

    import json
    (C.RESULTS_DIR / "ssl_finetune.json").write_text(json.dumps(
        {"budgets": budgets, "epochs": epochs, "seeds": seeds, "n_unlabeled_pretrain": ck["n_pretrain"],
         "results": out, "ssl_helps": bool(helps)}, indent=2))
    print("wrote results/ssl_finetune.json")


if __name__ == "__main__":
    main()
