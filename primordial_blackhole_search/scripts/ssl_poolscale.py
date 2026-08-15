"""L6: is N4's self-supervised win still CLIMBING with unlabeled pool size, or already saturated?

N4 showed SSL pretraining on 20,000 unlabeled noise spectrograms beats from-scratch at scarce labels
(+0.124 val AUC at 1,000 labels, +0.021 at 4,000), and that the win translates to sensitive distance. Its
recorded caveat was that the unlabeled pool WAS the labeled set's own 20k noise, so "more unlabeled O3 noise
would likely give more" stayed a hypothesis. GraviBERT (arXiv:2512.21390) makes the same bet at far larger
scale.

The obvious move is to fetch a bigger pool -- but that competes with the running L2 deep-background job for a
flaky GWOSC and for disk, and it is premature anyway: **if the gain has already saturated by 20k, fetching
more is pointless.** This measures the scaling curve from data already on disk, which decides whether the
expensive version (L6b) is worth doing at all.

WHAT IS LOCALLY AVAILABLE (checked, not assumed): shards_w64 has 20,000 H1 noise specs over the 16 train
segments. shards_w64_hl has 26,250 -- but 20,000 are the SAME 16 H1 segments (duplicates); only its **6,250
L1** specs are new, a +31% pool. Neither touches H1 val or test (leakage check: 0 segments in either).

TWO QUESTIONS
  S1  SCALING. Pretrain on 2,500 / 5,000 / 10,000 / 20,000 unlabeled specs, fine-tune each at a FIXED scarce
      label budget, and read the SSL gain against pool size. Still rising at 20k => a fetched pool is
      justified. Flat => L6 closes here, and N4's caveat is answered with a number instead of a guess.
  S2  CROSS-DETECTOR. Add the 6,250 L1 specs to the 20,000 H1 ones. The detector model runs on H1, so this
      asks whether unlabeled noise from a DIFFERENT interferometer still helps -- the transfer claim
      foundation models rest on, tested for free. A gain means the pool can grow across detectors, not just
      across time.

PRE-REGISTERED READING: SSL scaling is typically ~log-linear in pool size. Judgement fixed before running --
gain(20k) - gain(5k) > 0.02 AUC => still climbing, fetch more; otherwise saturated.

REUSE: SpecMAE, random_mask and the fine-tune/eval loops are imported from ssl_pretrain.py / ssl_finetune.py
rather than reimplemented, so this curve is directly comparable to N4's numbers (an earlier draft rewrote the
autoencoder with a different channel ladder, which would have made the comparison meaningless).

Run:  .venv/bin/python scripts/ssl_poolscale.py [--budget 1000] [--seeds 3]
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pbh import config as C
from pbh.models import make_model

HERE = Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


pre = _load("ssl_pretrain")
fin = _load("ssl_finetune")

SH = C.DATA_DIR / "shards_w64"
SH_HL = C.DATA_DIR / "shards_w64_hl"
POOLS = [2500, 5000, 10000, 20000]
OUT = C.RESULTS_DIR / "ssl_poolscale.json"


def pretrain_encoder(pool_x, mu, sd, dev, epochs, seed, mask_frac=0.6):
    """Same masked-autoencoder recipe as ssl_pretrain.py, on an arbitrary pool. Returns the encoder state."""
    torch.manual_seed(seed)
    model = pre.SpecMAE().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    n, bs, last = len(pool_x), 128, float("nan")
    for _ in range(epochs):
        order = rng.permutation(n)                     # shuffle INDICES, not the 645 MB array itself
        for b in range(0, n, bs):
            sel = np.sort(order[b:b + bs])
            xb = torch.tensor((np.asarray(pool_x[sel], dtype=np.float32) - mu) / sd,
                              device=dev).unsqueeze(1)
            xm, m = pre.random_mask(xb, frac=mask_frac, rng=rng)
            loss = (((model(xm) - xb) ** 2) * m).sum() / m.sum().clamp(min=1.0)   # masked pixels only
            opt.zero_grad(); loss.backward(); opt.step()
            last = float(loss.item())
    return model.encoder.state_dict(), last


def build_model(enc_sd, dev):
    model = make_model("cnn").to(dev)
    if enc_sd is not None:
        nn.Sequential(*list(model.net[:4])).load_state_dict(enc_sd)
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=1000, help="labeled budget (N4's biggest-gain point)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--pre-epochs", type=int, default=40)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    ep_ft = 2 if args.smoke else args.epochs
    ep_pre = 1 if args.smoke else args.pre_epochs
    pools = [500, 1000] if args.smoke else POOLS
    seeds = 1 if args.smoke else args.seeds

    meta = pd.read_parquet(SH / "meta_train.parquet")
    x = np.load(SH / "x_train.npy", mmap_mode="r")
    y = meta["label"].to_numpy().astype(np.float32)
    noise_idx = np.where(y == 0)[0]
    xv = np.load(SH / "x_val.npy", mmap_mode="r")
    yv = pd.read_parquet(SH / "meta_val.parquet").label.to_numpy().astype(np.float32)

    rng = np.random.default_rng(C.SEED + 606)
    sample = np.asarray(x[noise_idx[:2000]], dtype=np.float32)
    mu, sd = float(sample.mean()), float(sample.std() + 1e-6)
    print(f"L6 pool scaling | budget {args.budget} labels | {seeds} seeds | dev {dev}")
    print(f"  val {len(yv)} | unlabeled H1 available {len(noise_idx)} | pretrain {ep_pre} ep, finetune {ep_ft} ep\n",
          flush=True)

    out = {"budget": args.budget, "seeds": seeds, "epochs_ft": ep_ft, "epochs_pre": ep_pre,
           "n_val": int(len(yv)), "n_unlabeled_available": int(len(noise_idx)), "pools": {}}

    def run(enc_sd, seed):
        model = build_model(enc_sd, dev)
        idx = np.random.default_rng(C.SEED + seed).choice(len(y), size=args.budget, replace=False)
        fin.train_model(model, idx, x, y, mu, sd, dev, ep_ft, np.random.default_rng(seed))
        return fin.val_auc(model, xv, yv, mu, sd, dev)

    scratch = [run(None, s) for s in range(seeds)]
    out["scratch_auc"] = scratch; out["scratch_mean"] = float(np.mean(scratch))
    print(f"{'pool':>8} {'SSL AUC':>10} {'gain':>9}")
    print(f"{'scratch':>8} {np.mean(scratch):>10.4f} {'--':>9}", flush=True)

    for pool in pools:
        sel = np.sort(rng.choice(noise_idx, size=min(pool, len(noise_idx)), replace=False))
        px = np.asarray(x[sel], dtype=np.float32)
        aucs = []
        for s in range(seeds):
            enc_sd, _ = pretrain_encoder(px, mu, sd, dev, ep_pre, seed=s)
            aucs.append(run(enc_sd, s))
        out["pools"][str(pool)] = {"auc": aucs, "mean": float(np.mean(aucs)),
                                   "gain": float(np.mean(aucs) - out["scratch_mean"])}
        print(f"{pool:>8} {np.mean(aucs):>10.4f} {np.mean(aucs)-out['scratch_mean']:>+9.4f}", flush=True)

    # ---- S2 cross-detector -------------------------------------------------------------------------------
    mhl = pd.read_parquet(SH_HL / "meta_train.parquet")
    xhl = np.load(SH_HL / "x_train.npy", mmap_mode="r")
    l1 = np.where((mhl["label"].values == 0) & (mhl["ifo"].values == "L1"))[0]
    if len(l1) and xhl.shape[1:] == x.shape[1:] and not args.smoke:
        px = np.concatenate([np.asarray(x[np.sort(noise_idx)], dtype=np.float32),
                             np.asarray(xhl[np.sort(l1)], dtype=np.float32)])
        aucs = []
        for s in range(seeds):
            enc_sd, _ = pretrain_encoder(px, mu, sd, dev, ep_pre, seed=s)
            aucs.append(run(enc_sd, s))
        base = out["pools"][str(pools[-1])]["mean"]
        out["cross_detector"] = {"pool": int(len(px)), "n_l1": int(len(l1)), "auc": aucs,
                                 "mean": float(np.mean(aucs)),
                                 "gain_vs_h1_only": float(np.mean(aucs) - base)}
        print(f"\nS2 CROSS-DETECTOR: +{len(l1)} L1 specs -> pool {len(px)} | AUC {np.mean(aucs):.4f} "
              f"({np.mean(aucs)-base:+.4f} vs H1-only {pools[-1]})", flush=True)

    g_small, g_big = out["pools"][str(pools[1])]["gain"], out["pools"][str(pools[-1])]["gain"]
    out["gain_slope"] = g_big - g_small
    out["still_climbing"] = bool(g_big - g_small > 0.02)
    print(f"\nS1 SCALING: gain {g_small:+.4f} @{pools[1]} -> {g_big:+.4f} @{pools[-1]} (slope {g_big-g_small:+.4f})")
    print(f"=> {'STILL CLIMBING -- a fetched pool (L6b) is justified' if out['still_climbing'] else 'SATURATED -- fetching more unlabeled noise would NOT help; L6 closes here'}")
    OUT.write_text(json.dumps(out, indent=2))
    print("wrote ssl_poolscale.json")


if __name__ == "__main__":
    main()
