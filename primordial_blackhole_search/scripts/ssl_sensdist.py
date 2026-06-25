"""N4 follow-up (PLAN.md): does the SSL val-AUC win translate to the HEADLINE metric — SENSITIVE DISTANCE?

ssl_finetune showed SSL-pretrained > from-scratch on val AUC at scarce labels. AUC ≠ sensitive distance (the
latter is set by the zero-FA threshold + SNR50, i.e. the score TAILS). The w64 val shards already carry each
injection's `in_window_snr` (5–30) and `chirp_mass`, so we read the efficiency-vs-SNR curve straight from the
val scores: zero-FA threshold = max over held-out val NOISE; SNR50 = where efficiency crosses 0.5 vs in-window
SNR; sensitive-distance fraction = 8/SNR50 (ideal MF detects at SNR 8). Computed per mass bin for SSL-pretrained
vs from-scratch at a reduced labeled budget (input standardized to the SSL mu/sd — the models' convention).

Run:  .venv/bin/python scripts/ssl_sensdist.py [--budgets 1000 4000] [--epochs 40] [--smoke]
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

from pbh import config as C
from pbh.models import make_model

_ft = importlib.util.spec_from_file_location("ft", Path(__file__).resolve().parent / "ssl_finetune.py")
ft = importlib.util.module_from_spec(_ft); _ft.loader.exec_module(ft)

SHARDS = C.DATA_DIR / "shards_w64"
MASS_EDGES = [0.17, 0.35, 0.55, 0.88]
MASS_LABELS = ["0.17-0.35", "0.35-0.55", "0.55-0.88"]


@torch.no_grad()
def score_val(model, xv, mu, sd, dev):
    model.eval(); out = []
    for b in range(0, len(xv), 512):
        xb = torch.tensor((np.asarray(xv[b:b + 512], dtype=np.float32) - mu) / sd, device=dev).unsqueeze(1)
        out.append(model(xb).cpu().numpy())
    return np.concatenate(out)


def dist_fraction(meta_inj, scores_inj, thresh):
    """per mass bin: efficiency vs in_window_snr at the zero-FA threshold -> SNR50 -> 8/SNR50."""
    df = meta_inj.assign(score=scores_inj, detected=scores_inj > thresh)
    snr_bins = np.linspace(5, 30, 11)
    out = {}
    for lo_m, hi_m, lab in zip(MASS_EDGES[:-1], MASS_EDGES[1:], MASS_LABELS):
        sub = df[(df.chirp_mass >= lo_m) & (df.chirp_mass < hi_m)]
        cen, eff = [], []
        for lo_s, hi_s in zip(snr_bins[:-1], snr_bins[1:]):
            s = sub[(sub.in_window_snr >= lo_s) & (sub.in_window_snr < hi_s)]
            if len(s) >= 8:
                cen.append((lo_s + hi_s) / 2); eff.append(float(s.detected.mean()))
        snr50 = float(np.interp(0.5, eff, cen)) if len(cen) > 1 and max(eff) >= 0.5 else np.nan
        out[lab] = round(8.0 / snr50, 4) if np.isfinite(snr50) else 0.0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", type=int, nargs="+", default=[1000, 4000])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    epochs = 4 if args.smoke else args.epochs
    budgets = [1000] if args.smoke else args.budgets
    seeds = 1 if args.smoke else args.seeds

    ck = torch.load(ft.CKPT, map_location="cpu"); mu, sd = ck["mu"], ck["sd"]
    xt, yt, _, _ = ft.load_split("train", mu, sd)
    xv = np.load(SHARDS / "x_val.npy", mmap_mode="r")
    mv = pd.read_parquet(SHARDS / "meta_val.parquet")
    yv = mv.label.to_numpy()
    noise_mask = yv == 0
    meta_inj = mv[yv == 1][["chirp_mass", "in_window_snr"]].reset_index(drop=True)
    pos, neg = np.where(yt == 1)[0], np.where(yt == 0)[0]
    print(f"N4 sensitive distance: val {len(yv)} ({int((yv==1).sum())} inj) | dev {dev}\n", flush=True)

    def measure(model):
        sc = score_val(model, xv, mu, sd, dev)
        noise = sc[noise_mask]
        res = {}
        for name, thr in (("zeroFA", float(noise.max())),          # strict: max over held-out val noise
                          ("FAR1pct", float(np.quantile(noise, 0.99)))):   # softer operating point (~1/125)
            frac = dist_fraction(meta_inj, sc[yv == 1], thr)
            res[name] = {"frac": frac, "mean": float(np.mean(list(frac.values())))}
        return res

    out = {}
    print(f"{'budget':>7} {'model':>8} | {'zeroFA mean':>11} | {'FAR1pct mean':>12}")
    for budget in budgets:
        agg = {("scratch", "zeroFA"): [], ("scratch", "FAR1pct"): [],
               ("ssl", "zeroFA"): [], ("ssl", "FAR1pct"): []}
        last = {}
        for seed in range(seeds):
            rng = np.random.default_rng(seed); k = budget // 2
            idx = np.concatenate([rng.choice(pos, k, replace=False), rng.choice(neg, k, replace=False)])
            for tag, builder in (("scratch", lambda: make_model("cnn").to(dev)),
                                 ("ssl", lambda: ft.make_pretrained(dev)[0])):
                torch.manual_seed(seed); model = builder()
                ft.train_model(model, idx, xt, yt, mu, sd, dev, epochs, np.random.default_rng(seed))
                res = measure(model); last[tag] = res
                for thr in ("zeroFA", "FAR1pct"):
                    agg[(tag, thr)].append(res[thr]["mean"])
        for tag in ("scratch", "ssl"):
            print(f"{budget:>7} {tag:>8} | {np.mean(agg[(tag,'zeroFA')]):>11.3f} | "
                  f"{np.mean(agg[(tag,'FAR1pct')]):>12.3f}", flush=True)
        out[budget] = {thr: {"scratch_mean": float(np.mean(agg[("scratch", thr)])),
                             "ssl_mean": float(np.mean(agg[("ssl", thr)])),
                             "delta_mean": float(np.mean(agg[("ssl", thr)]) - np.mean(agg[("scratch", thr)])),
                             "scratch_frac": last["scratch"][thr]["frac"], "ssl_frac": last["ssl"][thr]["frac"]}
                       for thr in ("zeroFA", "FAR1pct")}
        d0, ds = out[budget]["zeroFA"]["delta_mean"], out[budget]["FAR1pct"]["delta_mean"]
        print(f"{budget:>7} {'Δ ssl':>8} | {d0:>+11.3f} | {ds:>+12.3f}\n")

    soft_helps = any(o["FAR1pct"]["delta_mean"] > 0.02 and o["FAR1pct"]["ssl_mean"] > 0 for o in out.values())
    zf_helps = any(o["zeroFA"]["ssl_mean"] > 0 for o in out.values())
    print(f"VERDICT: at the HEADLINE zero-FA threshold the reduced-budget distance is "
          f"{'>0' if zf_helps else '0 for BOTH (models too weak; SSL AUC win does NOT reach zero-FA distance)'}; "
          f"at a softer (1%) FAR the SSL win {'DOES translate to sensitive distance' if soft_helps else 'still does not clearly translate'}.")
    (C.RESULTS_DIR / "ssl_sensdist.json").write_text(json.dumps(
        {"budgets": budgets, "epochs": epochs, "seeds": seeds, "results": out,
         "metric": "sensitive-distance fraction 8/SNR50 from val injections (in_window_snr) at two thresholds",
         "zeroFA_distance_defined": bool(zf_helps), "ssl_helps_at_softFAR": bool(soft_helps)}, indent=2))
    print("wrote results/ssl_sensdist.json")


if __name__ == "__main__":
    main()
