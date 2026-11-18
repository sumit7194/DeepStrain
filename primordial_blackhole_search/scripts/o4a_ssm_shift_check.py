"""M11 post-hoc check (2026-11-18): were LVK's O4a trigger windows placed 5 s off our convention?

FINDING (measured 2026-11-18): pycbc get_td_waveform(TaylorF2, f_lower=50, 4096 Hz) returns 5.000 s of samples AFTER
coalescence for every mass we use, and pbh.waveforms keeps them, so every whitened injection and every bank template in
this repo has its merger 5.000 s BEFORE the end of the array (not at the end, as projected_waveform's docstring said).
M11's injections and templates both come from those helpers, so the thresholds and eps in frozen.json are internally
consistent. But `score` placed each TRIGGER window to END at the reported GPS, i.e. at the true merger: the CNN saw the
trigger 5 s later in its window than any training or injection signal, and the bank templates were aligned 5 s early,
far outside their +-0.5 s scan.

THIS SCRIPT (no committed result is changed): for each primary trigger and detector, the two statistics at
  shift 0      -- must reproduce results/o4a_ssm/rescore.json (golden: the same code path, else nothing is reported)
  shift +5 s   -- the window end moved 5.000 s after the reported GPS, where our convention puts it
then FAR from the cached background and the pre-registered verdict rule, with eps unchanged (eps comes from injections,
which were placed consistently). Pre-stated expectation: eps < 0.1 for all four looks, so a look can only become
"HIGH SCORE, NOISE EXCURSION FAVOURED" (if 4 x FAR <= 1/yr) or stay UNINFORMATIVE; it cannot become SUPPORT.

    nohup .venv/bin/python scripts/o4a_ssm_shift_check.py      (fetches 4 trigger segments, purged after use)
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pbh.data import segment_path, whiten_segment  # noqa: E402

HERE = Path(__file__).resolve().parent
_m = importlib.util.spec_from_file_location("m11", HERE / "o4a_ssm_rescore.py")
M = importlib.util.module_from_spec(_m); _m.loader.exec_module(M)

SHIFTS_S = (0.0, 5.0)
OUTFILE = M.OUT / "shift_check.json"


def verdict(sig, eps):
    return ("INDEPENDENT SUPPORT" if sig and eps >= 0.5 else
            "HIGH SCORE, NOISE EXCURSION FAVOURED" if sig else
            "TENSION" if eps >= 0.5 else
            "UNINFORMATIVE" if eps < 0.1 else "WEAKLY INFORMATIVE")


def main():
    fr = json.loads(M.FROZEN.read_text())
    tsets = json.loads(M.TSETS.read_text())
    committed = {t["name"]: t for t in json.loads(M.RESULT.read_text())["triggers"]}
    _, cat = M.load_bg()
    model, dev = M.load_model()
    out = {"shifts_s": list(SHIFTS_S), "triggers": []}
    for t in M.primaries():
        seg = M.trigger_segments(t)
        assert len(seg) == 2, (t["name"], seg)
        vals = {s: {} for s in SHIFTS_S}
        for d in M.DETS:
            M.fetch(d, seg[d])
            w, t0, psd = whiten_segment(d, seg[d])
            wc = w[M.CROP:-M.CROP]
            e0 = int(round((t["gps"] - t0) * M.FS)) - M.CROP
            for s in SHIFTS_S:
                e = e0 + int(round(s * M.FS))
                vals[s][f"cnn_{d}"] = M.cnn_score(model, dev, wc[e - M.WIN:e])
                dwin = wc[e - M.WIN - M.PAD:e + M.PAD]
                vals[s][f"bank_{d}"] = max(M.so.local_stats(dwin.astype(np.float64), M.template_chunks(mc, d, t0, psd),
                                                            M.PAD)[1] for mc in tsets[t["name"]]["template_mcs"])
            segment_path(d, seg[d]).unlink(missing_ok=True)
        ref = committed[t["name"]]["scores"]
        repro = {k: abs(vals[0.0][k] - ref[k]) for k in ref}
        assert max(repro.values()) < 1e-4, ("shift 0 does not reproduce rescore.json", repro)
        row = {"name": t["name"], "segments": seg, "shift0_max_abs_diff_vs_committed": max(repro.values()), "by_shift": {}}
        for s in SHIFTS_S:
            row["by_shift"][str(s)] = {}
            for stat in ("cnn", "bank"):
                key = (lambda d: f"cnn_{d}") if stat == "cnn" else (lambda d: f"bank_{t['name']}_{d}")
                x = vals[s][f"{stat}_H1"] + vals[s][f"{stat}_L1"]
                far = M.far_of(cat(key("H1")), cat(key("L1")), x)
                eps = fr["stats"][f"{t['name']}|{stat}"]["eps_at_rho_T"]
                row["by_shift"][str(s)][stat] = {"H1": vals[s][f"{stat}_H1"], "L1": vals[s][f"{stat}_L1"], "x": x,
                                                 "far_per_yr": far, "far_x_looks": M.LOOKS * far, "eps_at_rho_T": eps,
                                                 "verdict": verdict(M.LOOKS * far <= 1.0, eps)}
                print(f"{t['name']} shift {s:+.1f} s [{stat}] x = {x:.3f}  FAR {far:.3g}/yr  eps {eps:.3f}  -> "
                      f"{row['by_shift'][str(s)][stat]['verdict']}", flush=True)
        out["triggers"].append(row)
    OUTFILE.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUTFILE}")


if __name__ == "__main__":
    main()
