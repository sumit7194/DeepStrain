"""N4 (PLAN.md): self-supervised pretraining of the SpectrogramCNN backbone via MASKED-SPECTROGRAM reconstruction.

The data wall: the subsolar detector is limited by LABELED data. But unlabeled real-noise spectrograms are
abundant. This pretrains the SpectrogramCNN conv backbone (the exact 4 `_block`s, so the weights transfer) as a
masked autoencoder on noise spectrograms — randomly mask patches, reconstruct them — so the encoder learns the
noise's time-frequency structure with NO labels. N4 step 2 (separate) fine-tunes the detection head on a REDUCED
labeled budget and compares to from-scratch; if pretrained > from-scratch at scarce labels, SSL beats the wall.

Run:  .venv/bin/python scripts/ssl_pretrain.py [--epochs 40] [--smoke]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from pbh import config as C
from pbh.models import _block
from pbh.progress import progress

SHARDS = C.DATA_DIR / "shards_w64"
ENC_OUT = C.MODEL_DIR / "ssl_encoder.pt"


class SpecMAE(nn.Module):
    """Masked autoencoder. Encoder = SpectrogramCNN's 4 conv blocks (transferable); decoder = interpolate+conv."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(_block(1, 32), _block(32, 64), _block(64, 128), _block(128, 256))
        self.dec = nn.ModuleList([
            nn.Conv2d(256, 128, 3, padding=1), nn.Conv2d(128, 64, 3, padding=1),
            nn.Conv2d(64, 32, 3, padding=1), nn.Conv2d(32, 1, 3, padding=1)])

    def encode(self, x):                       # (B,1,128,T) -> (B,256,8,T/16)
        return self.encoder(x)

    def forward(self, x):
        h = self.encode(x)
        sizes = [(16, x.shape[3] // 8), (32, x.shape[3] // 4), (64, x.shape[3] // 2), (128, x.shape[3])]
        for i, conv in enumerate(self.dec):
            h = F.interpolate(h, size=sizes[i], mode="bilinear", align_corners=False)
            h = conv(h)
            if i < len(self.dec) - 1:
                h = F.silu(h)
        return h                               # (B,1,128,T)


def random_mask(x, frac=0.6, patch=(16, 8), rng=None):
    """Zero random (freq×time) patches of each spectrogram; return (masked_x, mask) where mask=1 on masked pixels."""
    B, _, Fq, T = x.shape
    pf, pt = patch
    nf, nt = Fq // pf, max(1, T // pt)
    m = (torch.rand(B, 1, nf, nt, device=x.device) < frac).float()
    m = F.interpolate(m, size=(nf * pf, nt * pt), mode="nearest")
    m = F.pad(m, (0, T - m.shape[3], 0, Fq - m.shape[2]))      # pad to (Fq,T) if not divisible
    return x * (1 - m), m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--mask-frac", type=float, default=0.6)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    epochs = 2 if args.smoke else args.epochs

    # unlabeled pool = all NOISE (label 0) spectrograms (abundant; no labels used)
    meta = pd.read_parquet(SHARDS / "meta_train.parquet")
    x = np.load(SHARDS / "x_train.npy", mmap_mode="r")
    noise_idx = np.where(meta["label"].values == 0)[0]
    if args.smoke:
        noise_idx = noise_idx[:512]
    print(f"SSL pretrain: {len(noise_idx)} unlabeled noise spectrograms {x.shape[1:]} on {dev}", flush=True)

    # standardize with the train-noise mean/std (saved so fine-tune uses the same)
    sample = np.asarray(x[noise_idx[:2000]], dtype=np.float32)
    mu, sd = float(sample.mean()), float(sample.std() + 1e-6)

    model = SpecMAE().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(0)
    bs = 128
    for ep in range(epochs):
        perm = rng.permutation(len(noise_idx))
        tot, nb = 0.0, 0
        for b in range(0, len(perm), bs):
            sel = np.sort(noise_idx[perm[b:b + bs]])
            xb = (np.asarray(x[sel], dtype=np.float32) - mu) / sd
            xb = torch.tensor(xb, device=dev).unsqueeze(1)     # (B,1,128,T)
            xm, m = random_mask(xb, args.mask_frac, rng=rng)
            rec = model(xm)
            loss = (((rec - xb) ** 2) * m).sum() / (m.sum() + 1e-6)   # MSE on masked pixels only
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
            progress("ssl_pretrain", ep * (len(perm) // bs + 1) + nb, epochs * (len(perm) // bs + 1),
                     loss=float(loss))
        print(f"  epoch {ep+1}/{epochs}: masked-recon MSE = {tot/nb:.4f}", flush=True)

    # save ONLY the encoder weights (the transferable SpectrogramCNN backbone) + the standardization
    torch.save({"encoder": model.encoder.state_dict(), "mu": mu, "sd": sd,
                "epochs": epochs, "mask_frac": args.mask_frac, "n_pretrain": len(noise_idx)}, ENC_OUT)
    print(f"\nwrote {ENC_OUT.name} (encoder backbone for fine-tuning; mu={mu:.3f} sd={sd:.3f})")


if __name__ == "__main__":
    main()
