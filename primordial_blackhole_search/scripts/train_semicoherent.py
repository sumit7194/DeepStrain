"""Stage 1: train the learned semi-coherent model on whitened strain.

On-the-fly injection (pbh.straindata) from the pre-built waveform pool. Train/val
split by BOTH noise segment and pool waveform (no leakage). --overfit runs the
skill's capacity gate (memorize one batch -> ~100%) before any full run.

Run:  .venv/bin/python scripts/train_semicoherent.py --overfit
      .venv/bin/python scripts/train_semicoherent.py --lr 1e-3 --epochs 16
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pbh import config as C
from pbh.models import make_model
from pbh.progress import progress
from pbh.straindata import StrainInjectionDataset, load_noise

POOL_DIR = C.DATA_DIR / "waveform_pool"


@torch.no_grad()
def auc_eval(model, loader, device) -> float:
    model.eval()
    scores, labels = [], []
    for x, y in loader:
        scores.append(model(x.to(device)).float().cpu().numpy())
        labels.append(y.numpy())
    s, y = np.concatenate(scores), np.concatenate(labels)
    order = np.argsort(s)
    rank = np.empty_like(order, dtype=np.float64)
    rank[order] = np.arange(len(s))
    npos, nneg = int(y.sum()), int((1 - y).sum())
    return float((rank[y == 1].sum() - npos * (npos - 1) / 2) / (npos * nneg))


def load_pool():
    x = np.load(POOL_DIR / "x_pool.npy", mmap_mode="r")
    snr = pd.read_parquet(POOL_DIR / "pool_meta.parquet")["snr_ref"].to_numpy()
    return x, snr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--train-samples", type=int, default=20000)
    ap.add_argument("--val-samples", type=int, default=4000)
    ap.add_argument("--out", default="semicoherent")
    ap.add_argument("--overfit", action="store_true", help="capacity gate: memorize one batch")
    args = ap.parse_args()

    torch.manual_seed(C.SEED)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")

    pool_x, pool_snr = load_pool()
    n_tr_pool = int(0.8 * len(pool_x))
    manifest = json.loads((C.DATA_DIR / "manifest.json").read_text())
    print("loading + whitening noise segments...", flush=True)
    train_noise = load_noise(manifest["H1"]["train"])
    val_noise = load_noise(manifest["H1"]["val"])
    print(f"pool {len(pool_x)} ({n_tr_pool} tr / {len(pool_x)-n_tr_pool} val), "
          f"noise {len(train_noise)} tr / {len(val_noise)} val segs")

    model = make_model("semicoherent").to(device)
    print(f"semicoherent: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")
    loss_fn = nn.BCEWithLogitsLoss()

    if args.overfit:
        ds = StrainInjectionDataset(train_noise, pool_x[:n_tr_pool], pool_snr[:n_tr_pool],
                                    n_samples=args.batch, deterministic=True)
        xb = torch.stack([ds[i][0] for i in range(args.batch)]).to(device)
        yb = torch.tensor([ds[i][1] for i in range(args.batch)]).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        for step in range(300):
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            if step % 50 == 0 or step == 299:
                acc = ((model(xb) > 0).float() == yb).float().mean().item()
                print(f"  step {step:3d}: loss {float(loss):.4f}  train acc {acc:.3f}", flush=True)
        assert acc > 0.95, f"OVERFIT GATE FAILED: acc {acc:.3f} — capacity/architecture issue"
        print("OVERFIT GATE PASS — the architecture can fit; proceed to full training")
        return

    train_ds = StrainInjectionDataset(train_noise, pool_x[:n_tr_pool], pool_snr[:n_tr_pool],
                                      n_samples=args.train_samples)
    val_ds = StrainInjectionDataset(val_noise, pool_x[n_tr_pool:], pool_snr[n_tr_pool:],
                                    n_samples=args.val_samples, deterministic=True)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=False, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=64, num_workers=0)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    C.MODEL_DIR.mkdir(exist_ok=True)
    run = f"train_{args.out}"
    steps_per = len(train_dl)
    best_auc, history = 0.0, []
    for epoch in range(args.epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for b, (x, y) in enumerate(train_dl):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            tot += float(loss) * len(y); n += len(y)
            if b % 20 == 0:
                progress(run, epoch * steps_per + b, args.epochs * steps_per,
                         loss=tot / max(n, 1), epoch=float(epoch))
        sched.step()
        auc = auc_eval(model, val_dl, device)
        progress(run, (epoch + 1) * steps_per, args.epochs * steps_per,
                 loss=tot / n, epoch=float(epoch), val_auc=auc)
        history.append({"epoch": epoch, "train_loss": tot / n, "val_auc": auc,
                        "sec": time.time() - t0})
        print(f"epoch {epoch:2d}  loss {tot/n:.4f}  val AUC {auc:.4f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), C.MODEL_DIR / f"{args.out}.pt")
    (C.MODEL_DIR / f"{args.out}_history.json").write_text(json.dumps(history))
    print(f"BEST VAL AUC {best_auc:.4f} -> models/{args.out}.pt  (cnn_w64 ref: 0.793)")


if __name__ == "__main__":
    main()
