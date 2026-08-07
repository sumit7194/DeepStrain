"""O4-3: O4b-era sensitive distance & search reach campaign for subsolar PBH search.

Evaluates absolute search reach in Mpc and surveyed volume in Mpc³ across subsolar mass bins
(0.17-0.35, 0.35-0.55, 0.55-0.88 Msun) for both single-detector (H1) and double-detector coincidence
(H1×L1) in both O3a (2019) and O4b (2024-2025) data eras.

Run: .venv/bin/python scripts/o4_sensitive_distance.py [--n-inj 2400] [--smoke]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pbh import config as C
from pbh.data import whiten_segment
from pbh.metrics import MASS_EDGES, MASS_LABELS
from pbh.models import make_model
from pbh.spectrogram import spectrogram
from pbh.sweep import SweepGrid, pool_and_log, score_windows
from pbh.waveforms import make_whitened_injection, sample_params

GRID = SweepGrid.short(64)
WIN = GRID.win_samp                       # 262144 (64 s)
NBINS = GRID.n_time_bins                  # 63
NET_SNR_RANGE = (4.0, 40.0)
SNR_BINS = np.linspace(*NET_SNR_RANGE, 13)
SEED = C.SEED + 9999

O3A_SEGS = [1242193730, 1242197826, 1242201922, 1242206018, 1242210114]
O4B_SEGS = [1396913920, 1396933632, 1396937728, 1396941824, 1396946944, 1396951040, 1396959488, 1396963584]


def score_wins(model, device, wins):
    feats = np.stack([pool_and_log(spectrogram(w), NBINS) for w in wins])
    return score_windows(model, device, feats)


def compute_metrics(df: pd.DataFrame, col: str, min_count: int = 10):
    out = {}
    for lo_m, hi_m, lab in zip(MASS_EDGES[:-1], MASS_EDGES[1:], MASS_LABELS):
        sub = df[(df.chirp_mass >= lo_m) & (df.chirp_mass < hi_m)]
        cen, eff = [], []
        for blo, bhi in zip(SNR_BINS[:-1], SNR_BINS[1:]):
            s = sub[(sub.target_snr >= blo) & (sub.target_snr < bhi)]
            if len(s) >= min_count:
                cen.append((blo + bhi) / 2)
                eff.append(float(s[col].mean()))
        snr50 = float(np.interp(0.5, eff, cen)) if len(cen) > 1 and max(eff) >= 0.5 else np.nan
        snr_ref_avg = float(sub.snr_ref_net.mean())
        
        # Ideal matched filter reach (SNR 8 at 1 Mpc)
        d_ideal = 1.0 * snr_ref_avg / 8.0 if snr_ref_avg > 0 else 0.0
        
        # Model reach (SNR50 at 1 Mpc)
        frac = 8.0 / snr50 if np.isfinite(snr50) else 0.0
        d_model = 1.0 * snr_ref_avg / snr50 if np.isfinite(snr50) else 0.0
        v_model = (4.0 / 3.0) * np.pi * (d_model ** 3) if d_model > 0 else 0.0
        
        out[lab] = {
            "snr50": round(snr50, 3) if np.isfinite(snr50) else None,
            "dist_fraction": round(frac, 4),
            "snr_ref_net_avg": round(snr_ref_avg, 3),
            "d_ideal_mpc": round(d_ideal, 3),
            "d_reach_mpc": round(d_model, 3),
            "v_reach_mpc3": round(v_model, 3),
        }
    return out


def evaluate_era(era_name: str, segs: list[int], model, device, n_inj: int, smoke: bool):
    crop = C.WHITEN_CROP_SEC * C.SAMPLE_RATE
    data = {}
    noise = {"H1": {}, "L1": {}}
    usable_segs = []
    
    print(f"\n--- [{era_name}] Evaluating {len(segs)} segments ---", flush=True)
    t0_ = time.time()
    for g in segs:
        try:
            wH, tH, psdH = whiten_segment("H1", g)
            wL, tL, psdL = whiten_segment("L1", g)
            if not (np.isfinite(wH).all() and np.isfinite(wL).all()):
                continue
            wcH = wH[crop:-crop]
            wcL = wL[crop:-crop]
            nwin = min((len(wcH) - WIN) // WIN, (len(wcL) - WIN) // WIN)
            startsH = np.arange(nwin) * WIN
            startsL = np.arange(nwin) * WIN
            data[("H1", g)] = (wcH, tH, psdH, startsH)
            data[("L1", g)] = (wcL, tL, psdL, startsL)
            noise["H1"][g] = score_wins(model, device, [wcH[s:s + WIN] for s in startsH])
            noise["L1"][g] = score_wins(model, device, [wcL[s:s + WIN] for s in startsL])
            usable_segs.append(g)
        except Exception as e:
            print(f"  SKIP seg {g}: {e}", flush=True)
            continue
            
    print(f"  [{era_name}] scored noise on {len(usable_segs)} usable segs in {time.time()-t0_:.1f}s", flush=True)
    
    # Thresholds
    all_h1_noise = np.concatenate([noise["H1"][g] for g in usable_segs])
    thr_single = float(all_h1_noise.max())
    
    # Time-slides for H1xL1 coincidence
    bg_slides = []
    n_lag_total = 0
    for g in usable_segs:
        h1_sc = noise["H1"][g]
        l1_sc = noise["L1"][g]
        n_w = min(len(h1_sc), len(l1_sc))
        for lag in range(1, n_w):
            bg_slides.append(h1_sc[:n_w] + np.roll(l1_sc[:n_w], lag))
        n_lag_total += (n_w - 1)
    bg_slides = np.concatenate(bg_slides)
    thr_coinc_matched = float(np.sort(bg_slides)[-n_lag_total])
    
    print(f"  [{era_name}] thr_single: {thr_single:.3f} | thr_coinc_matched: {thr_coinc_matched:.3f}", flush=True)
    
    # Injections
    n_inj_per_seg = max(10, n_inj // len(usable_segs))
    rows = []
    t_inj = time.time()
    for g in usable_segs:
        (wcH, tH, psdH, startsH) = data[("H1", g)]
        (wcL, tL, psdL, startsL) = data[("L1", g)]
        rng = np.random.default_rng([SEED, int(g)])
        nwin = min(len(startsH), len(startsL))
        winsH, winsL, metas = [], [], []
        
        for _ in range(n_inj_per_seg):
            p = sample_params(rng)
            t_geo = tH + C.SEGMENT_LEN // 2
            hwH, refH = make_whitened_injection(p, "H1", t_geo, psdH)
            hwL, refL = make_whitened_injection(p, "L1", t_geo, psdL)
            net_ref = float(np.hypot(refH, refL))
            target = float(rng.uniform(*NET_SNR_RANGE))
            scale = target / net_ref
            
            wi = int(rng.integers(0, nwin))
            m = int(rng.integers(WIN // 2, WIN))
            wH = wcH[startsH[wi] : startsH[wi] + WIN].copy()
            wL = wcL[startsL[wi] : startsL[wi] + WIN].copy()
            wH[:m] += (hwH * scale)[-WIN:][WIN - m :]
            wL[:m] += (hwL * scale)[-WIN:][WIN - m :]
            winsH.append(wH)
            winsL.append(wL)
            metas.append((p.chirp_mass, target, net_ref, refH, refL))
            
        sH = score_wins(model, device, winsH)
        sL = score_wins(model, device, winsL)
        print(f"    seg {g} injections done ({len(winsH)} wins)", flush=True)
        
        for (mc, target, net_ref, refH, refL), sh, sl in zip(metas, sH, sL):
            s_coinc = float(sh + sl)
            rows.append({
                "era": era_name,
                "gps": g,
                "chirp_mass": mc,
                "target_snr": target,
                "snr_ref_net": net_ref,
                "snr_ref_h1": refH,
                "snr_ref_l1": refL,
                "sH1": float(sh),
                "sL1": float(sl),
                "s_coinc": s_coinc,
                "det_single": bool(sh > thr_single),
                "det_coinc": bool(s_coinc > thr_coinc_matched),
            })
            
    print(f"  [{era_name}] {len(rows)} injections finished in {time.time()-t_inj:.1f}s", flush=True)
    df = pd.DataFrame(rows)
    
    single_metrics = compute_metrics(df, "det_single", min_count=5 if smoke else 10)
    coinc_metrics = compute_metrics(df, "det_coinc", min_count=5 if smoke else 10)
    
    return {
        "era": era_name,
        "n_usable_segs": len(usable_segs),
        "thr_single_zeroFA": thr_single,
        "thr_coinc_matchedFAR": thr_coinc_matched,
        "single_detector": single_metrics,
        "coincidence": coinc_metrics,
        "df": df,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-inj", type=int, default=2400)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    
    if args.smoke:
        args.n_inj = 160
        o3_segs = O3A_SEGS[:2]
        o4_segs = O4B_SEGS[:2]
    else:
        o3_segs = O3A_SEGS
        o4_segs = O4B_SEGS
        
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = make_model("cnn")
    model.load_state_dict(torch.load(C.MODEL_DIR / "cnn_w64.pt", map_location=device))
    model.to(device).eval()
    
    res_o3 = evaluate_era("O3a", o3_segs, model, device, args.n_inj, args.smoke)
    res_o4 = evaluate_era("O4b", o4_segs, model, device, args.n_inj, args.smoke)
    
    # Compute overall gain ratios
    comparison = {}
    for mode in ("single_detector", "coincidence"):
        comparison[mode] = {}
        for lab in MASS_LABELS:
            d_o3 = res_o3[mode][lab]["d_reach_mpc"]
            d_o4 = res_o4[mode][lab]["d_reach_mpc"]
            v_o3 = res_o3[mode][lab]["v_reach_mpc3"]
            v_o4 = res_o4[mode][lab]["v_reach_mpc3"]
            
            d_gain = d_o4 / d_o3 if d_o3 > 0 else 0.0
            v_gain = v_o4 / v_o3 if v_o3 > 0 else 0.0
            
            comparison[mode][lab] = {
                "d_o3a_mpc": d_o3,
                "d_o4b_mpc": d_o4,
                "d_gain_ratio": round(d_gain, 3),
                "v_o3a_mpc3": v_o3,
                "v_o4b_mpc3": v_o4,
                "v_gain_ratio": round(v_gain, 3),
            }
            
    print("\n=======================================================", flush=True)
    print("O4b vs O3a SEARCH REACH SUMMARY (Coincidence Mode)", flush=True)
    print("=======================================================", flush=True)
    for lab in MASS_LABELS:
        c = comparison["coincidence"][lab]
        print(f"Mc [{lab} Msun]: O3a {c['d_o3a_mpc']:.2f} Mpc ({c['v_o3a_mpc3']:.2f} Mpc³) -> "
              f"O4b {c['d_o4b_mpc']:.2f} Mpc ({c['v_o4b_mpc3']:.2f} Mpc³) | "
              f"GAIN: {c['d_gain_ratio']:.2f}x distance, {c['v_gain_ratio']:.2f}x volume", flush=True)
        
    out = {
        "O3a": {k: v for k, v in res_o3.items() if k != "df"},
        "O4b": {k: v for k, v in res_o4.items() if k != "df"},
        "comparison": comparison,
    }
    
    C.RESULTS_DIR.mkdir(exist_ok=True)
    out_json = C.RESULTS_DIR / f"o4_sensitive_distance{'_smoke' if args.smoke else ''}.json"
    out_json.write_text(json.dumps(out, indent=2))
    print(f"\nWrote results to {out_json}", flush=True)
    
    # Generate Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # Left: Reach Distance (Mpc)
    bar_width = 0.35
    x = np.arange(len(MASS_LABELS))
    d_o3_coinc = [res_o3["coincidence"][l]["d_reach_mpc"] for l in MASS_LABELS]
    d_o4_coinc = [res_o4["coincidence"][l]["d_reach_mpc"] for l in MASS_LABELS]
    
    axes[0].bar(x - bar_width/2, d_o3_coinc, bar_width, label="O3a (2019)", color="steelblue")
    axes[0].bar(x + bar_width/2, d_o4_coinc, bar_width, label="O4b (2024-25)", color="darkorange")
    axes[0].set_ylabel("Search Reach Distance [Mpc]")
    axes[0].set_xlabel("Chirp Mass Bin [Msun]")
    axes[0].set_title("H1×L1 Coincidence Search Reach (Mpc)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(MASS_LABELS)
    axes[0].legend()
    axes[0].grid(alpha=0.3, axis="y")
    
    for i in range(len(MASS_LABELS)):
        gain = comparison["coincidence"][MASS_LABELS[i]]["d_gain_ratio"]
        axes[0].text(i + bar_width/2, d_o4_coinc[i] + 0.05 * max(d_o4_coinc), f"{gain:.2f}x", ha="center", fontsize=9, fontweight="bold")
        
    # Right: Surveyed Volume (Mpc³)
    v_o3_coinc = [res_o3["coincidence"][l]["v_reach_mpc3"] for l in MASS_LABELS]
    v_o4_coinc = [res_o4["coincidence"][l]["v_reach_mpc3"] for l in MASS_LABELS]
    
    axes[1].bar(x - bar_width/2, v_o3_coinc, bar_width, label="O3a (2019)", color="steelblue")
    axes[1].bar(x + bar_width/2, v_o4_coinc, bar_width, label="O4b (2024-25)", color="darkorange")
    axes[1].set_ylabel("Surveyed Volume [Mpc³]")
    axes[1].set_xlabel("Chirp Mass Bin [Msun]")
    axes[1].set_title("H1×L1 Coincidence Surveyed Volume (Mpc³)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(MASS_LABELS)
    axes[1].legend()
    axes[1].grid(alpha=0.3, axis="y")
    
    for i in range(len(MASS_LABELS)):
        vgain = comparison["coincidence"][MASS_LABELS[i]]["v_gain_ratio"]
        axes[1].text(i + bar_width/2, v_o4_coinc[i] + 0.05 * max(v_o4_coinc), f"{vgain:.2f}x", ha="center", fontsize=9, fontweight="bold")
        
    fig.tight_layout()
    plot_path = C.RESULTS_DIR / f"o4_sensitive_distance{'_smoke' if args.smoke else ''}.png"
    fig.savefig(plot_path, dpi=120)
    print(f"Wrote plot to {plot_path}", flush=True)


if __name__ == "__main__":
    main()
