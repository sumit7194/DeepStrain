"""Does a per-SEGMENT constant let the learned coincidence head cheat? (tabula's confound mechanism, on us)

THE WARNING. The tabula sibling planted a per-realization nuisance channel -- a calibration-offset stand-in
with zero dynamical meaning -- and their invariant engine ranked it as MORE conserved than the genuine
invariant, passing out-of-sample validation completely. The generalisable mechanism:

    held-out validation catches OVERFITTING, not CONFOUNDING, because a nuisance constant generalises
    flawlessly precisely by being genuinely constant.

WHY THAT POINTS AT US. Build C-2's learned coincidence statistic (coinc_learned.py) trains a head on
[eH, eL, |eH-eL|, eH.eL] to separate real coincident injections from time-slid noise pairs, and we called
`--holdout-segments` (train 16 segments, evaluate on 8 unseen) the gold-standard leakage control. But look at
how the two classes are built:

    positives: pair_feats(iH[tr], iL[tr])        # an injection's H1 and L1 -> SAME segment, same time
    a = noise_tr[rng.integers(0, len(noise_tr))] # H1 noise, uniform over the whole pool
    b = noise_tr[rng.integers(0, len(noise_tr))] # L1 noise, INDEPENDENT draw -> usually a DIFFERENT segment

So positives are same-segment pairs and negatives are overwhelmingly cross-segment pairs. If the 256-d
embeddings carry ANY per-segment constant -- a residual of that segment's PSD, a calibration-like offset,
anything constant-within and varying-across -- then |eH-eL| and eH.eL separate the classes with **no
gravitational-wave content at all**. And `--holdout-segments` cannot detect it: the same-segment-vs-cross
structure is present in the held-out segments too, so the confound generalises perfectly. Our strongest
leakage control is blind to exactly this.

THE CHEAP DECISIVE TEST, run before any expensive re-run. Ask whether the cheating channel EXISTS at all,
using noise only and no injections:

  C1  CHANNEL EXISTENCE. Take pure-noise windows. Label a pair 1 if its H1 and L1 windows come from the SAME
      segment, 0 if from different segments. Train the identical CoincHead on the identical features. There
      is no signal anywhere in this data, so a head that separates the classes is reading a per-segment
      constant and nothing else. **AUC ~ 0.5 => no channel, Build C-2 is safe. AUC >> 0.5 => the channel is
      real and the gain must be re-derived with same-segment negatives.**

  C2  MAGNITUDE. If the channel exists, how much of the embedding is per-segment constant? Compare the
      between-segment variance of the mean embedding against the within-segment variance (a one-way F-like
      ratio, per dimension). This says whether the effect is a lurking few dimensions or pervasive.

Run:  .venv/bin/python scripts/coinc_confound.py [--n-pair 4000] [--seeds 3]
"""
import argparse
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

SH_HL = C.DATA_DIR / "shards_w64_hl"
OUT = C.RESULTS_DIR / "coinc_confound.json"


def auc(scores, labels):
    s, y = np.asarray(scores), np.asarray(labels)
    r = s.argsort().argsort()
    n1 = int(y.sum()); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


class CoincHead(nn.Module):
    """The same small head Build C-2 uses on the pair features."""

    def __init__(self, d=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4 * d, 128), nn.ReLU(), nn.Linear(128, 32), nn.ReLU(),
                                 nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def pair_feats(eH, eL):
    """Exactly Build C-2's feature construction."""
    return np.concatenate([eH, eL, np.abs(eH - eL), eH * eL], axis=1)


@torch.no_grad()
def embed(model, dev, x, bs=256):
    """256-d penultimate features from cnn_w64 -- the same representation the learned head consumes."""
    outs = []
    for b in range(0, len(x), bs):
        xb = torch.tensor(np.asarray(x[b:b + bs], dtype=np.float32), device=dev).unsqueeze(1)
        h = xb
        for layer in list(model.net)[:-1]:
            h = layer(h)
        outs.append(h.flatten(1).float().cpu().numpy())
    return np.concatenate(outs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pair", type=int, default=4000, help="pairs per class")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"

    meta = pd.read_parquet(SH_HL / "meta_train.parquet")
    x = np.load(SH_HL / "x_train.npy", mmap_mode="r")
    noise = meta[meta.label == 0]
    segs = sorted(set(noise[noise.ifo == "H1"].gps.unique()) & set(noise[noise.ifo == "L1"].gps.unique()))
    print(f"confound test | {len(segs)} segments with BOTH detectors | dev {dev}")

    model = make_model("cnn").to(dev)
    model.load_state_dict(torch.load(C.MODEL_DIR / "cnn_w64.pt", map_location=dev))
    model.eval()

    # per-segment, per-detector embeddings of PURE NOISE -- no injections anywhere in this script
    E = {}
    for g in segs:
        for ifo in ("H1", "L1"):
            idx = noise[(noise.gps == g) & (noise.ifo == ifo)].row.to_numpy()
            E[(g, ifo)] = embed(model, dev, x[np.sort(idx)])
        print(f"  seg {g}: H1 {E[(g,'H1')].shape} L1 {E[(g,'L1')].shape}", flush=True)
    d = E[(segs[0], "H1")].shape[1]

    # ---- C2 how much of the embedding is a per-segment constant? ---------------------------------------
    ratios = {}
    for ifo in ("H1", "L1"):
        means = np.stack([E[(g, ifo)].mean(0) for g in segs])            # (n_seg, d)
        within = np.mean([E[(g, ifo)].var(0) for g in segs], axis=0)     # (d,)
        between = means.var(0)                                            # (d,)
        r = between / (within + 1e-12)
        ratios[ifo] = {"median": float(np.median(r)), "p90": float(np.percentile(r, 90)),
                       "max": float(r.max()), "frac_gt_1": float((r > 1).mean())}
        print(f"\nC2 per-segment structure in {ifo} embeddings (between/within variance, {d} dims):")
        print(f"   median {np.median(r):.3f} | p90 {np.percentile(r,90):.3f} | max {r.max():.3f} | "
              f"{100*(r>1).mean():.0f}% of dims have between > within")

    # ---- C1 can a head detect SAME-SEGMENT from noise alone? -------------------------------------------
    print(f"\nC1 channel-existence test: same-segment vs cross-segment, PURE NOISE, no injections")
    aucs = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(1000 + seed)
        torch.manual_seed(seed)

        def make(n, same):
            H, L = [], []
            for _ in range(n):
                gi = rng.integers(len(segs))
                gj = gi if same else (gi + 1 + rng.integers(len(segs) - 1)) % len(segs)
                eh = E[(segs[gi], "H1")]; el = E[(segs[gj], "L1")]
                H.append(eh[rng.integers(len(eh))]); L.append(el[rng.integers(len(el))])
            return np.stack(H), np.stack(L)

        hp, lp = make(args.n_pair, True)
        hn, ln = make(args.n_pair, False)
        X = np.concatenate([pair_feats(hp, lp), pair_feats(hn, ln)])
        y = np.concatenate([np.ones(args.n_pair), np.zeros(args.n_pair)])
        cut = int(0.7 * len(X)); order = rng.permutation(len(X))
        tr, ev = order[:cut], order[cut:]
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xt = torch.tensor((X - mu) / sd, dtype=torch.float32).to(dev)
        yt = torch.tensor(y, dtype=torch.float32).to(dev)

        head = CoincHead(d).to(dev)
        opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-3)
        lf = nn.BCEWithLogitsLoss()
        for ep in range(args.epochs):
            perm = np.random.default_rng(seed * 97 + ep).permutation(tr)
            for b in range(0, len(perm), 256):
                sel = perm[b:b + 256]
                opt.zero_grad(); lf(head(Xt[sel]), yt[sel]).backward(); opt.step()
        with torch.no_grad():
            a = auc(head(Xt[ev]).float().cpu().numpy(), y[ev])
        aucs.append(a)
        print(f"   seed {seed}: held-out AUC {a:.4f}", flush=True)

    m = float(np.mean(aucs))
    out = {"n_segments": len(segs), "embed_dim": int(d), "n_pair": args.n_pair, "seeds": args.seeds,
           "between_within": ratios, "c1_auc": aucs, "c1_auc_mean": m,
           "channel_exists": bool(m > 0.6),
           "holdout_segments_would_catch_it": False}
    print(f"\nC1 mean AUC {m:.4f} over {args.seeds} seeds")
    if m > 0.6:
        print("   => THE CONFOUND CHANNEL EXISTS. A head can tell same-segment from cross-segment pairs using")
        print("      embeddings of PURE NOISE, with no signal present. Build C-2's negatives are drawn")
        print("      cross-segment while its positives are same-segment, so part (or all) of its learned gain")
        print("      may be this channel. --holdout-segments CANNOT detect it. The gain must be re-derived")
        print("      with same-segment negatives before the +0.02-0.05 claim can stand.")
    elif m > 0.55:
        print("   => WEAK channel. Present but small; quantify against the claimed gain before trusting it.")
    else:
        print("   => NO usable channel: the head cannot identify same-segment pairs from noise embeddings.")
        print("      Build C-2's learned gain cannot be explained by a per-segment constant, and the")
        print("      --holdout-segments control, though blind to this class of confound in principle, was")
        print("      not hiding one here.")
    out["verdict"] = ("channel exists -- Build C-2 gain must be re-derived with same-segment negatives"
                      if m > 0.6 else
                      "weak channel -- quantify before trusting" if m > 0.55 else
                      "no channel -- Build C-2 gain is not explained by a per-segment constant")
    OUT.write_text(json.dumps(out, indent=2))
    print("\nwrote coinc_confound.json")


if __name__ == "__main__":
    main()
