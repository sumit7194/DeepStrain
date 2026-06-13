"""Stage 1: train the learned semi-coherent model on whitened strain.

On-the-fly injection (pbh.straindata) from the pre-built waveform pool. Train/val
split by BOTH noise segment and pool waveform (no leakage).
  --overfit : capacity gate (memorize one batch -> ~100%) before any full run.
  --sweep   : 3-point LR probe (3e-4/1e-3/3e-3), then full-train the winner —
              one job, noise loaded once (per the skill's "LR is not a default").

Run:  .venv/bin/python scripts/train_semicoherent.py --overfit
      .venv/bin/python scripts/train_semicoherent.py --sweep --epochs 16
"""

from __future__ import annotations

import argparse
import json
import os
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
LRS = (3e-4, 1e-3, 3e-3)


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


def atomic_torch_save(obj, path) -> None:
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)  # never leave a half-written checkpoint on power loss


def run_training(model, train_dl, val_dl, device, lr, epochs, run,
                 save_path=None, ckpt_path=None, resume=False):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.BCEWithLogitsLoss()
    start, best_auc, history = 0, 0.0, []
    if resume and ckpt_path and Path(ckpt_path).exists():
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start, best_auc, history = ck["epoch"] + 1, ck["best_auc"], ck["history"]
        print(f"  [{run}] RESUMED at epoch {start} (best AUC {best_auc:.4f})", flush=True)
    steps = len(train_dl)
    for epoch in range(start, epochs):
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
                progress(run, epoch * steps + b, epochs * steps,
                         loss=tot / max(n, 1), epoch=float(epoch))
        sched.step()
        auc = auc_eval(model, val_dl, device)
        progress(run, (epoch + 1) * steps, epochs * steps,
                 loss=tot / n, epoch=float(epoch), val_auc=auc)
        history.append({"epoch": epoch, "train_loss": tot / n, "val_auc": auc,
                        "sec": time.time() - t0})
        print(f"  [{run}] epoch {epoch:2d}  loss {tot/n:.4f}  val AUC {auc:.4f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        if auc > best_auc:
            best_auc = auc
            if save_path:
                atomic_torch_save(model.state_dict(), save_path)
        if ckpt_path:  # resumable checkpoint every epoch (model + opt + sched + epoch)
            atomic_torch_save(
                {"model": model.state_dict(), "opt": opt.state_dict(),
                 "sched": sched.state_dict(), "epoch": epoch,
                 "best_auc": best_auc, "history": history}, ckpt_path)
    return best_auc, history


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
    ap.add_argument("--sweep", action="store_true", help="3-point LR probe then train the winner")
    ap.add_argument("--sweep-epochs", type=int, default=3)
    ap.add_argument("--resume", action="store_true",
                    help="resume from the last epoch checkpoint (and skip a cached sweep)")
    args = ap.parse_args()

    torch.manual_seed(C.SEED)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")

    pool_x, pool_snr = load_pool()
    n_tr = int(0.8 * len(pool_x))
    manifest = json.loads((C.DATA_DIR / "manifest.json").read_text())
    print("loading + whitening noise segments...", flush=True)
    train_noise = load_noise(manifest["H1"]["train"])
    val_noise = load_noise(manifest["H1"]["val"])
    print(f"pool {len(pool_x)} ({n_tr} tr / {len(pool_x)-n_tr} val), "
          f"noise {len(train_noise)} tr / {len(val_noise)} val segs")
    C.MODEL_DIR.mkdir(exist_ok=True)
    loss_fn = nn.BCEWithLogitsLoss()

    if args.overfit:
        model = make_model("semicoherent").to(device)
        print(f"semicoherent: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")
        ds = StrainInjectionDataset(train_noise, pool_x[:n_tr], pool_snr[:n_tr],
                                    n_samples=args.batch, deterministic=True)
        xb = torch.stack([ds[i][0] for i in range(args.batch)]).to(device)
        yb = torch.tensor([ds[i][1] for i in range(args.batch)]).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        acc = 0.0
        for step in range(300):
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            if step % 50 == 0 or step == 299:
                acc = ((model(xb) > 0).float() == yb).float().mean().item()
                print(f"  step {step:3d}: loss {float(loss):.4f}  train acc {acc:.3f}", flush=True)
        assert acc > 0.95, f"OVERFIT GATE FAILED: acc {acc:.3f}"
        print("OVERFIT GATE PASS")
        return

    val_ds = StrainInjectionDataset(val_noise, pool_x[n_tr:], pool_snr[n_tr:],
                                    n_samples=args.val_samples, deterministic=True)
    val_dl = DataLoader(val_ds, batch_size=64, num_workers=0)
    train_ds = StrainInjectionDataset(train_noise, pool_x[:n_tr], pool_snr[:n_tr],
                                      n_samples=args.train_samples)
    train_dl = DataLoader(train_ds, batch_size=args.batch, num_workers=0)

    sweep_path = C.MODEL_DIR / f"{args.out}_sweep.json"
    if args.sweep:
        if args.resume and sweep_path.exists():
            args.lr = float(json.loads(sweep_path.read_text())["best_lr"])
            print(f"RESUMED: skipping sweep, cached best lr={args.lr:.0e}", flush=True)
        else:
            probe_ds = StrainInjectionDataset(train_noise, pool_x[:n_tr], pool_snr[:n_tr],
                                              n_samples=8000)
            probe_dl = DataLoader(probe_ds, batch_size=args.batch, num_workers=0)
            sweep = {}
            for lr in LRS:
                torch.manual_seed(C.SEED)
                m = make_model("semicoherent").to(device)
                auc, _ = run_training(m, probe_dl, val_dl, device, lr, args.sweep_epochs,
                                      f"sweep_lr{lr:.0e}")
                sweep[lr] = auc
                print(f"LR {lr:.0e}: probe val AUC {auc:.4f}", flush=True)
            args.lr = max(sweep, key=sweep.get)
            tmp = sweep_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({"best_lr": args.lr,
                                       "aucs": {f"{k:.0e}": v for k, v in sweep.items()}}))
            os.replace(tmp, sweep_path)
            print(f"SWEEP winner: lr={args.lr:.0e}  "
                  f"({ {f'{k:.0e}': round(v,4) for k,v in sweep.items()} })", flush=True)

    torch.manual_seed(C.SEED)
    model = make_model("semicoherent").to(device)
    print(f"full training: lr={args.lr:.0e}, {args.epochs} epochs, "
          f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M params", flush=True)
    best_auc, history = run_training(
        model, train_dl, val_dl, device, args.lr, args.epochs, f"train_{args.out}",
        save_path=C.MODEL_DIR / f"{args.out}.pt",
        ckpt_path=C.MODEL_DIR / f"{args.out}_ckpt.pt", resume=args.resume)
    (C.MODEL_DIR / f"{args.out}_history.json").write_text(json.dumps(history))
    print(f"BEST VAL AUC {best_auc:.4f} -> models/{args.out}.pt  (cnn_w64 ref 0.793)", flush=True)


if __name__ == "__main__":
    main()
