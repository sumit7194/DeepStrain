"""Phase 2/3: train a detector on the spectrogram shards.

Run:  uv run python scripts/train.py --model cnn
      uv run python scripts/train.py --model transformer
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
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pbh import config as C
from pbh.models import make_model
from pbh.progress import progress


class ShardDataset(Dataset):
    def __init__(self, split: str, shard_dir, augment: bool = False):
        self.x = np.load(shard_dir / f"x_{split}.npy", mmap_mode="r")
        self.meta = pd.read_parquet(shard_dir / f"meta_{split}.parquet")
        self.y = self.meta.label.to_numpy().astype(np.float32)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        x = np.asarray(self.x[i], dtype=np.float32)
        if self.augment:
            rng = np.random.default_rng()
            # time shift (edge-padded): a chirp is a chirp wherever it sits
            k = int(rng.integers(-12, 13))
            if k:
                x = np.roll(x, k, axis=1)
                if k > 0:
                    x[:, :k] = x[:, k : k + 1]
                else:
                    x[:, k:] = x[:, k - 1 : k]
            # small frequency shift (chirp-rate jitter within calibration)
            f = int(rng.integers(-2, 3))
            if f:
                x = np.roll(x, f, axis=0)
            # additive noise on log-power (decorrelates memorized textures)
            x = x + rng.normal(0, 0.05, x.shape).astype(np.float32)
        return torch.from_numpy(x).unsqueeze(0), self.y[i]


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    scores, labels = [], []
    for x, y in loader:
        scores.append(model(x.to(device)).float().cpu().numpy())
        labels.append(y.numpy())
    s, y = np.concatenate(scores), np.concatenate(labels)
    # AUC via rank statistic
    order = np.argsort(s)
    rank = np.empty_like(order, dtype=np.float64)
    rank[order] = np.arange(len(s))
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    auc = (rank[y == 1].sum() - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg)
    # efficiency at FPR = 1e-3 (proxy for low-FAR operation)
    thresh = np.quantile(s[y == 0], 1 - 1e-3)
    eff = float((s[y == 1] > thresh).mean())
    return {"auc": float(auc), "eff_at_1e-3": eff, "scores": s, "labels": y}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["cnn", "transformer"], required=True)
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--shard-dir", default=str(C.SHARD_DIR),
                    help="dataset dir (default v1 shards; use data/shards_w64 for rung 2)")
    ap.add_argument("--out", default=None,
                    help="output model name (default = --model, e.g. cnn_w64)")
    args = ap.parse_args()
    shard_dir = Path(args.shard_dir)
    out_name = args.out or args.model

    torch.manual_seed(C.SEED)
    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"device: {device}")

    train_ds = ShardDataset("train", shard_dir, augment=True)
    val_ds = ShardDataset("val", shard_dir)
    print(f"train {len(train_ds)}, val {len(val_ds)}")
    train_dl = DataLoader(
        train_ds, batch_size=C.BATCH_SIZE, shuffle=True, num_workers=4,
        persistent_workers=True,
    )
    val_dl = DataLoader(val_ds, batch_size=256, num_workers=2)

    model = make_model(args.model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{args.model}: {n_params / 1e6:.2f} M params")

    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.BCEWithLogitsLoss()

    C.MODEL_DIR.mkdir(exist_ok=True)
    run = f"train_{out_name}"
    total_steps = args.epochs * len(train_dl)
    best_auc, history = 0.0, []
    for epoch in range(args.epochs):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for b, (x, y) in enumerate(train_dl):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += float(loss) * len(y)
            n += len(y)
            if b % 20 == 0:  # keep the dashboard card live within long epochs
                progress(run, epoch * len(train_dl) + b, total_steps,
                         loss=total / max(n, 1), epoch=float(epoch))
        sched.step()

        m = evaluate(model, val_dl, device)
        progress(run, (epoch + 1) * len(train_dl), total_steps,
                 loss=total / n, epoch=float(epoch), val_auc=m["auc"])
        history.append(
            {"epoch": epoch, "train_loss": total / n, "val_auc": m["auc"],
             "val_eff_at_1e-3": m["eff_at_1e-3"], "sec": time.time() - t0}
        )
        print(
            f"epoch {epoch:2d}  loss {total / n:.4f}  val AUC {m['auc']:.4f}  "
            f"eff@1e-3 {m['eff_at_1e-3']:.3f}  ({time.time() - t0:.0f}s)",
            flush=True,
        )
        if m["auc"] > best_auc:
            best_auc = m["auc"]
            torch.save(model.state_dict(), C.MODEL_DIR / f"{out_name}.pt")

    (C.MODEL_DIR / f"{out_name}_history.json").write_text(json.dumps(history))
    print(f"BEST VAL AUC {best_auc:.4f} -> models/{out_name}.pt")


if __name__ == "__main__":
    main()
