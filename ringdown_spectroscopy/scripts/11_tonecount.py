#!/usr/bin/env python
"""Milestone 11 (v4): amortized, start-time-marginalized TONE-COUNT selection.

How many QNM tones are in the data -- 220 alone, or 220 + the 221 overtone?
A binary classifier trained on balanced 1-tone / 2-tone simulations IS the
amortized tone-count Bayes factor: its output is P(2-tone | data), with start
time and amplitudes marginalized by construction (the crux of the GW150914
overtone controversy). Reuses sbilib.simulate_tonecount + Embed.

Pre-registered gates T1-T4 in notes/lab_notebook.md (2026-06-15).
Usage:  python 11_tonecount.py [--smoke]
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from gwpy.timeseries import TimeSeries

import rdlib
import sbilib
from sbilib import Embed, N_SAMP, kerr220, kerr221

torch.manual_seed(0)
rng = np.random.default_rng(0)
PLOTS = Path(__file__).resolve().parent.parent / "plots"
RESULTS = Path(__file__).resolve().parent.parent / "results"
MODELS = Path(__file__).resolve().parent.parent / "models"
MODELS.mkdir(exist_ok=True)
EVENTS = {"GW250114_082203": "GW250114", "GW150914": "GW150914"}
SEG = sbilib.SEG
AMP_FRAC = (0.1, 1.5)          # 221/220 amplitude-ratio prior for 2-tone training
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


class ToneCounter(torch.nn.Module):
    """Embed CNN -> scalar logit = log-odds P(2-tone)."""

    def __init__(self):
        super().__init__()
        self.embed = Embed(n_out=64)
        self.head = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Linear(64, 1))

    def forward(self, x):
        return self.head(self.embed(x)).squeeze(-1)


def make_batch(n, rng):
    """Balanced 1-tone/2-tone. Returns (x (n, 2*N_SAMP), label, overtone_snr)."""
    x = np.empty((n, 2 * N_SAMP), dtype=np.float32)
    y = np.empty(n, dtype=np.float32)
    osnr = np.empty(n, dtype=np.float32)
    for k in range(n):
        m = rng.uniform(40, 120)
        c = rng.uniform(0.05, 0.95)
        two = k % 2 == 1
        frac = rng.uniform(*AMP_FRAC) if two else 0.0
        seg, s = sbilib.simulate_tonecount(m, c, 2 if two else 1, frac, rng)
        x[k] = seg.reshape(-1)
        y[k] = 1.0 if two else 0.0
        osnr[k] = s
    return x, y, osnr


def auc(scores, labels):
    order = np.argsort(scores)
    rank = np.empty_like(order, dtype=float)
    rank[order] = np.arange(len(scores))
    npos, nneg = labels.sum(), (1 - labels).sum()
    return float((rank[labels == 1].sum() - npos * (npos - 1) / 2) / (npos * nneg))


@torch.no_grad()
def predict(model, x):
    out = []
    for i in range(0, len(x), 512):
        b = torch.tensor(x[i:i + 512]).to(DEVICE)
        out.append(torch.sigmoid(model(b)).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    n_train = 4000 if args.smoke else 80_000
    epochs = 4 if args.smoke else 30
    print(f"device {DEVICE} | n_train {n_train} | epochs {epochs}")

    # -------------------------------------------------------------- train
    print("simulating training set ...")
    xtr, ytr, _ = make_batch(n_train, rng)
    rdlib.progress("11_tonecount_sims", n_train, n_train)
    model = ToneCounter().to(DEVICE)
    model(torch.tensor(xtr[:8]).to(DEVICE))  # init LazyLinear
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    xt = torch.tensor(xtr); yt = torch.tensor(ytr)
    bs, nb = 256, (n_train + 255) // 256
    print("training tone-count classifier ...")
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        tot = 0.0
        for b in range(nb):
            idx = perm[b * bs:(b + 1) * bs]
            xb, yb = xt[idx].to(DEVICE), yt[idx].to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
            if b % 20 == 0:
                rdlib.progress("11_tonecount_train", ep * nb + b, epochs * nb,
                               loss=tot / max((b + 1) * bs, 1), epoch=float(ep))
        sched.step()
        print(f"  epoch {ep:2d}  loss {tot/n_train:.4f}", flush=True)
    model.eval()
    torch.save(model.state_dict(), MODELS / "11_tonecount.pt")

    # ---------------------------------------------------- T1 capacity (AUC)
    xte, yte, ote = make_batch(4000, rng)
    p_te = predict(model, xte)
    A = auc(p_te, yte)
    print(f"\nT1 held-out AUC = {A:.4f}  (gate > 0.90)")

    # ------------------------------- T2 sensitivity vs overtone SNR + specificity
    # dedicated 2-tone test spanning overtone SNR; 1-tone test for specificity
    x2, y2, o2 = make_batch(8000, rng)              # balanced -> use the 2-tone half
    p2 = predict(model, x2)
    is2 = y2 == 1
    snr2, pred2 = o2[is2], p2[is2]
    edges = np.linspace(0, np.percentile(snr2, 98), 11)
    cen, eff = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (snr2 >= lo) & (snr2 < hi)
        if m.sum() >= 20:
            cen.append((lo + hi) / 2); eff.append(float((pred2[m] > 0.5).mean()))
    snr50 = float(np.interp(0.5, eff, cen)) if len(cen) > 1 and max(eff) >= 0.5 else float("nan")
    specificity = float((p2[~is2] < 0.5).mean())    # 1-tone correctly called 1-tone
    rdlib.progress("11_tonecount_referee", 1, 3)
    print(f"T2 specificity (1-tone called 1-tone) = {specificity:.3f}  (gate >= 0.90)")
    print(f"   overtone SNR for 50% detection = {snr50:.2f}")

    # ----------------------------------------------------- T3 calibration (ECE)
    bins = np.linspace(0, 1, 11)
    ece, rel = 0.0, []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p_te >= lo) & (p_te < hi)
        if m.sum() > 0:
            conf, acc_ = p_te[m].mean(), yte[m].mean()
            ece += m.mean() * abs(conf - acc_)
            rel.append((float(conf), float(acc_), int(m.sum())))
    print(f"T3 expected calibration error = {ece:.3f}  (lower = better; gate < 0.10)")
    rdlib.progress("11_tonecount_referee", 2, 3)

    # ------------------------------- T2b honest cross-check: real O4-noise injections
    real = {"1tone": [], "2tone": []}
    base = rdlib.event_gps("GW250114_082203")
    F1, TAU1 = kerr220.f_tau(68.0, 0.69)
    F2, TAU2 = kerr221.f_tau(68.0, 0.69)
    k = 0
    for kind, n_t in (("1tone", 1), ("2tone", 2)):
        got = 0
        while got < 6 and k < 40:
            center = base - 300 - 137 * k
            k += 1
            try:
                segs = []
                for det in ("H1", "L1"):
                    raw = TimeSeries.fetch_open_data(det, center - 32, center + 32, cache=True)
                    if not np.isfinite(raw.value).all():
                        raise ValueError("NaNs")
                    params = [dict(f=F1, tau=TAU1, amp=2.0e-21, phi=rng.uniform(-np.pi, np.pi))]
                    if n_t == 2:
                        params.append(dict(f=F2, tau=TAU2, amp=1.4e-21, phi=rng.uniform(-np.pi, np.pi)))
                    inj = rdlib.inject_ringdown(raw, center, params).whiten(4, 2)
                    seg = inj.crop(center - 0.002, center - 0.002 + SEG + 0.01).value[:N_SAMP]
                    assert len(seg) == N_SAMP
                    segs.append(seg)
                xo = np.stack(segs).reshape(1, -1).astype(np.float32)
                real[kind].append(float(predict(model, xo)[0]))
                got += 1
            except Exception as e:
                print(f"  [real {kind} noise {k}] skip: {e}")
    for kind in real:
        v = np.array(real[kind]) if real[kind] else np.array([np.nan])
        print(f"T2b real-noise {kind}: P(2-tone) = {np.nanmean(v):.2f} ± {np.nanstd(v):.2f} (n={len(real[kind])})")
    rdlib.progress("11_tonecount_referee", 3, 3)

    # --------------------------------------------------------- T4 application
    app = {}
    for ev, label in EVENTS.items():
        try:
            gps = rdlib.event_gps(ev)
            segs = []
            for det in ("H1", "L1"):
                white = rdlib.fetch_whitened(det, gps, bandpass=False)
                pk = rdlib.find_peak(white.bandpass(*rdlib.BAND), gps)
                seg = white.crop(pk, pk + SEG + 0.01).value[:N_SAMP]
                assert len(seg) == N_SAMP
                segs.append(seg)
            xo = np.stack(segs).reshape(1, -1).astype(np.float32)
            p = float(predict(model, xo)[0])
            app[label] = p
            print(f"T4 {label}: P(2-tone) = {p:.3f}")
        except Exception as e:
            app[label] = None
            print(f"T4 {label}: skipped ({e})")

    # ---------------------------------------------------------------- save
    out = dict(auc=A, specificity=specificity, overtone_snr50=snr50, ece=ece,
               reliability=rel, real_noise={k: real[k] for k in real},
               application=app, n_train=n_train, amp_frac_prior=AMP_FRAC)
    (RESULTS / "11_tonecount.json").write_text(json.dumps(out, indent=2))

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    ax[0].plot(cen, eff, "o-"); ax[0].axhline(0.5, color="gray", ls=":")
    if np.isfinite(snr50):
        ax[0].axvline(snr50, color="crimson", ls="--", label=f"50% @ SNR {snr50:.1f}")
    ax[0].set_xlabel("injected overtone (221) SNR"); ax[0].set_ylabel("P(2-tone)>0.5 efficiency")
    ax[0].set_title(f"T2 sensitivity (specificity {specificity:.2f})"); ax[0].legend(); ax[0].grid(alpha=0.3)
    if rel:
        cf, ac, _ = zip(*rel)
        ax[1].plot([0, 1], [0, 1], "k:"); ax[1].plot(cf, ac, "o-")
    ax[1].set_xlabel("predicted P(2-tone)"); ax[1].set_ylabel("empirical fraction 2-tone")
    ax[1].set_title(f"T3 calibration (ECE {ece:.3f})"); ax[1].grid(alpha=0.3)
    labs = list(app); vals = [app[l] if app[l] is not None else 0 for l in labs]
    ax[2].bar(labs, vals, color=["tab:green", "tab:orange"][:len(labs)])
    ax[2].axhline(0.5, color="gray", ls=":"); ax[2].set_ylim(0, 1)
    ax[2].set_ylabel("P(2-tone)"); ax[2].set_title("T4 real events")
    fig.tight_layout(); fig.savefig(PLOTS / "11_tonecount.png", dpi=140)
    print(f"\nwrote {RESULTS/'11_tonecount.json'} + {PLOTS/'11_tonecount.png'}")
    rdlib.progress("11_tonecount_done", 1, 1, auc=A, specificity=specificity, ece=ece)


if __name__ == "__main__":
    main()
