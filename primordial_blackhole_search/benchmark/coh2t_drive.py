"""Step T build step 1 (registration reports/stepT_registration.md §8.1): the box-major stage-2 driver.  [model: Claude Opus 5.5]
UNVERIFIED: code only, nothing has been run with it yet.

What it does. Every stage-2 candidate is examined over the box around its Y1 template (chirp mass +-2.20 %, D5; q over
the population range; merger time +-1 s at full rate, D1). Candidates are grouped by box: a box is fixed by the Y1
template index and L_max, so its constructed grid and its reference lattice are built ONCE and then every candidate of
that box is processed: background, its Gaussian twin (K1), the caught injections (fg), the F1 nulls (null: the
injection windows without the signal; wrong: with the signal, box at Mc +5 %) and the three F2 decoy families. Output
is checkpointed per (box, kind) with an atomic write; a restart skips finished files. Kinds and their exact rules:
reports/stepT_clarifications.md (C1-C9).

Stage-1 candidates and their box (identical rule for background and foreground):
  background  Y1's clustered network triggers above thr1 (one per minute, caught.json), re-derived here from Y1's
              stored bins so that each keeps its H1 bin. The box centre is that bin's H1 template, which is the
              convention Y1's foreground triggers already use (y1_bank.network_trigger returns the H1 tid) and the one
              the stage-1 bias (T1_mc_bias.json) was measured with. The plan step refuses to proceed unless the
              re-derived triggers reproduce submissions/y1/bg_k*.csv and caught.json exactly.
  foreground  caught tuning injections at their Y1 trigger (time and tid from results/y1_bins/fg_*.json), scored by
              linearity: noise snippet + gated, double-whitened signal snippet (the same arithmetic as
              coh2c_local.prep; gate K2 checks it against really injected strain).
  Both: H1 at segment test[i], L1 at test[(i + k) mod 24] (k = 0 for the foreground), same in-segment sample.
  Edge rule (realbg_cascade's, for both kinds): the candidate's merger sample must lie in [(384 + 8) s, 4096 - 8 s];
  candidates outside are recorded as excluded, and their count is reported, so livetime can be scaled.

Per candidate (engine: coh2c_local, verified as D1; clarifications in reports/stepT_clarifications.md): a 512-s
snippet per detector (start clamped inside the segment), one reference correlation per reference used by the box,
every grid template's complex z over the +-1-s lags in H1 and L1. The network value is the registered H-anchored form
(registration §3), with the L1 term on the SAME template, as realbg_cascade.survival paired them:
  * H1's K = 5 loudest templates, each at its own best H1 lag (per-template de-duplication, clarification C1);
  * at each: rho_H and Allen's chi^2 (p = 4, 8, 16 equal-SNR bins, gate E3's exact route), and the same template's
    rho_L and chi^2 at EVERY L1 lag within +-10 ms (41 samples each side), with their GPS times;
  * the plain-coherent ablation, same form over all templates: max over H1 lags of |z_H|^2 + max_{+-10 ms} |z_L|^2.
PSD variation v and newSNR are applied afterwards by PyCBC itself in its own process (coh2t.v_at / rho_hat), so the
engine process never imports pycbc. The candidate score is then max over the K templates of
rho^_H^2 + max over delta of rho^_L^2, with v read at each point's own time.

pycbc runs only in short-lived children: noise series (coh2c_local prep), PyCBC v series, and per chunk of boxes the
reference templates and the foreground signal snippets, which the engine deletes once their box is done.

  python benchmark/coh2t_drive.py plan                         candidates and boxes -> state/coh2t/plan.json  (CPU, s)
  python benchmark/coh2t_drive.py golden                       driver path vs results/coh2c_verify (one box)   (CPU)
  python benchmark/coh2t_drive.py prepseries                   noise series for every (det, segment)  (children)
  python benchmark/coh2t_drive.py vseries                      PyCBC v series for every (det, segment) (children)
  python benchmark/coh2t_drive.py run <l_max> [kinds] [n_box]  box-major loop, e.g. run 128 bg,fg 3     (MPS)
  python benchmark/coh2t_drive.py calib <l_max>                decoy amplitude target from the fg pass (once)
  Order per L_max: run bg,twin,fg,null,wrong -> calib -> run dec_sg,dec_train,dec_drift
  python benchmark/coh2t_drive.py status <l_max>
  (internal) chunk <l_max> <tid,...>                           child: references + foreground signal snippets
Device: COH2C_DEV, as the engine (MPS in production).
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

B = Path(__file__).resolve().parent
sys.path.insert(0, str(B)); sys.path.insert(0, str(B.parent)); sys.path.insert(0, str(B.parent.parent / "tools"))
import coh2c_local as L  # noqa: E402
from pbh import config as C  # noqa: E402

FS, NS = L.FS, L.NS
N = C.SEGMENT_LEN * FS
STATE = B / "state" / "coh2t"
OUT = B / "results" / "coh2t"
HALF_WIDTH = 0.022                 # D5: uniform 2.20 % chirp-mass half-width
K_PEAKS = 5                        # registration §3: chi^2 at the top K = 5 peaks per detector
P_BINS = (4, 8, 16)                # registration §4: one pass serves all three p
DELTA = int(round(0.010 * FS))     # +-10 ms network window (41 samples)
EDGE_LO, EDGE_HI = (384 + 8) * FS, N - 8 * FS
BATCH = 128                        # registration §9: template batches <= 128 at M = 32,768
CHUNK = 2                          # boxes per child; the next chunk's child runs while this one is scored
KINDS = ("bg", "fg", "twin", "null", "wrong", "dec_sg", "dec_train", "dec_drift", "k1mc",
         "allen_off", "allen_on", "allen_fail")
ALLEN = ("allen_off", "allen_on", "allen_fail")
N_ALLEN = 200                      # C11: per family, per L_max, each with both detectors
DECOYS = ("dec_sg", "dec_train", "dec_drift")
N_DECOY = 200                      # registration §5 F2: 200 per family
SEED = 202611261                   # this build's own seed (twin draws, decoy choice and parameters); NOT the fresh seed
WRONG_MC = 1.05                    # F1(b): box offset to chirp mass +5 %
N_MC, N_REAL, SLOT_S = 100_000, 8, 4.0    # C10: K1-(A) maxima, independent pycbc realisations, slot length
K1_SEED0 = 7_000_000               # C10: lal ranlux seed = K1_SEED0 + 1000 * realisation + 2 * segment + detector
CLOSURE_MAX = 3e-4                 # C12 (ruled): closure gate, relative to max(|z|, 1); breach stops at the box boundary
UNIT_STRAIN = 1e-21                # decoys are built at this raw-strain amplitude, then scaled (C6)
Y1_BIN, Y1_WIN, Y1_CROP = 256, 64 * FS, C.WHITEN_CROP_SEC * FS


def atomic_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def segments():
    return json.loads((B / "manifest.json").read_text())["test"]


# ------------------------------------------------------------------ plan: candidates and boxes (CPU, numpy only)
def y1_triggers(h, lv_src, g):
    """y1_bank.series_triggers, unchanged, except that it also returns the kept H1 bin indices."""
    lv = np.maximum.reduce([np.pad(lv_src["val"], 1, mode="edge")[s:s + len(lv_src["val"])] for s in (0, 1, 2)])
    net = np.sqrt(np.maximum(h["val"], 0) ** 2 + np.maximum(lv, 0) ** 2)
    assert np.isfinite(net).all(), "non-finite Y1 network bin"
    t = g + C.WHITEN_CROP_SEC + (np.arange(len(net)) * Y1_BIN + h["off"] + Y1_WIN - 1) / FS
    half = int(round(1.0 * FS / Y1_BIN))
    taken = np.zeros(len(net), bool)
    keep = []
    for i in np.argsort(-net):
        if not taken[i]:
            keep.append(i)
            taken[max(0, i - half):i + half + 1] = True
    keep = np.sort(keep)
    return keep, t[keep], net[keep]


def plan():
    import csv
    segs = segments()
    st = json.loads((B / "state" / "realbg" / "caught.json").read_text())
    thr1 = st["thr1"]
    z = {g: np.load(B / "results" / "y1_bins" / f"noise_{g}.npz") for g in segs}
    get = lambda g, d: {k: z[g][f"{d}_{k}"] for k in ("val", "off", "tid")}
    cands, n_bad = [], 0
    for k in range(24):
        with open(B / "submissions" / "y1" / f"bg_k{k}.csv") as f:
            csv_rows = np.array([[float(r["time"]), float(r["stat"])] for r in csv.DictReader(f)])
        tt, ss, info = [], [], []
        for i, g in enumerate(segs):
            h = get(g, "H1")
            keep, t, s = y1_triggers(h, get(segs[(i + k) % 24], "L1"), g)
            tt.append(t); ss.append(s)
            info += [(i, int(b), int(h["off"][b]), int(h["tid"][b])) for b in keep]
        tt, ss = np.concatenate(tt), np.concatenate(ss)
        assert len(tt) == len(csv_rows), (k, len(tt), len(csv_rows))
        dt = np.abs(np.round(tt, 4) - csv_rows[:, 0]).max()
        ds = np.abs(np.round(ss.astype(np.float64), 5) - csv_rows[:, 1]).max()
        assert dt < 2e-4 and ds < 2e-5, ("Y1 bins do not reproduce bg_k%d.csv" % k, dt, ds)
        for (i, b, off, tid), (t_csv, s_csv) in zip(info, csv_rows):
            if s_csv > thr1:                                  # caught()'s own selection, on the CSV values
                m = Y1_CROP + b * Y1_BIN + off + Y1_WIN - 1   # merger sample from the segment start (exact)
                assert abs(m - (t_csv - segs[i]) * FS) <= 0.25, (k, i, m, t_csv)    # CSV time is rounded to 0.1 ms
                cands.append(dict(kind="bg", k=k, i=i, iL=(i + k) % 24, i_c=m, tid=tid, y1=s_csv,
                                  valid=bool(EDGE_LO <= m <= EDGE_HI),
                                  bin102=int(np.floor((t_csv - segs[i]) * FS / 102))))   # caught()'s own formula
    assert len(cands) == st["n1"], (len(cands), st["n1"])
    want = {key: sorted(v) for key, v in st["candidates"].items()}
    got = {}
    # caught.json floors the 0.1-ms-rounded CSV time, so a merger exactly on a 102-sample boundary sits one bin low
    # there; the driver uses the exact sample. Counted for the record:
    n_floor = sum(c["i_c"] // 102 != c["bin102"] for c in cands)
    for c in cands:
        got.setdefault(f"{c['k']}_{c['i']}", []).append(c.pop("bin102"))
    n_bad = sum(sorted(got.get(key, [])) != v for key, v in want.items())
    assert n_bad == 0 and set(got) <= set(want), f"{n_bad} (slide, segment) cells disagree with caught.json"
    # foreground: caught injections at their Y1 trigger
    caught = {r["id"]: r for r in st["rows"] if r["caught"]}
    trig = {x["id"]: x for p in sorted((B / "results" / "y1_bins").glob("fg_*.json")) for x in json.loads(p.read_text())}
    for iid, r in sorted(caught.items()):
        x = trig[iid]
        i = segs.index(r["segment"])
        m = int(round((x["time"] - r["segment"]) * FS))
        cands.append(dict(kind="fg", k=0, i=i, iL=i, i_c=m, tid=int(x["tid"]), y1=float(x["stat"]), inj=iid,
                          valid=bool(EDGE_LO <= m <= EDGE_HI)))
    import y1_bank as Y1
    mcs = np.asarray(Y1.MCS, dtype=np.float64)
    fg_c = [c for c in cands if c["kind"] == "fg"]
    bg_v = [c for c in cands if c["kind"] == "bg" and c["valid"]]
    # twin (K1): every valid background candidate again, on PSD-drawn Gaussian noise (C9)
    cands += [dict(c, kind="twin") for c in bg_v]
    # F1(a) null: the caught injections' windows with the signal left out; F1(b) wrong: the signal in, box at Mc +5 %
    cands += [dict(c, kind="null") for c in fg_c]
    cands += [dict(c, kind="wrong", tid=int(np.argmin(np.abs(np.log(mcs / (mcs[c["tid"]] * WRONG_MC))))),
                   tid_y1=c["tid"]) for c in fg_c]
    # F2 decoys (C6-C8): 200 valid background candidate times per family, parameters drawn now
    rng = np.random.default_rng(SEED)
    for fam in DECOYS:
        for j in rng.choice(len(bg_v), N_DECOY, replace=False):
            c = dict(bg_v[int(j)], kind=fam)
            c["decoy"] = decoy_params(fam, rng)
            cands.append(c)
    # K1-(A) (C10): N_MC samples in N_REAL independent pycbc realisations. Each takes the box and segment pairing of a
    # random twin candidate, at a time slot of its realisation that no other sample uses in that H1 or L1 series,
    # so no two maxima of one realisation share a +-1-s window.
    twins = [c for c in cands if c["kind"] == "twin"]
    slots = np.arange(EDGE_LO + int(SLOT_S * FS) // 2, EDGE_HI - int(SLOT_S * FS) // 2, int(SLOT_S * FS))
    rng_k = np.random.default_rng([SEED, 1])
    for r in range(N_REAL):
        usedH = np.zeros((24, len(slots)), bool); usedL = np.zeros((24, len(slots)), bool)
        for _ in range(N_MC // N_REAL):
            while True:
                t = twins[int(rng_k.integers(len(twins)))]
                free = np.where(~usedH[t["i"]] & ~usedL[t["iL"]])[0]
                if len(free):
                    break
            j = int(free[rng_k.integers(len(free))])
            usedH[t["i"], j] = usedL[t["iL"], j] = True
            cands.append(dict(kind="k1mc", k=t["k"], i=t["i"], iL=t["iL"], tid=t["tid"], y1=None,
                              i_c=int(slots[j]), real=r, valid=True))
    # C11: off-grid / on-grid / must-fail injections into twin noise at random twin candidates' boxes and times
    rng_a = np.random.default_rng([SEED, 2])
    for fam in ALLEN:
        for _ in range(N_ALLEN):
            t = twins[int(rng_a.integers(len(twins)))]
            cands.append(dict(kind=fam, k=t["k"], i=t["i"], iL=t["iL"], tid=t["tid"], y1=None, i_c=t["i_c"], valid=True,
                              inj_par={det: dict(dt=float(rng_a.uniform(-0.5, 0.5)), rho=float(rng_a.uniform(8, 25)),
                                                 u=float(rng_a.uniform(0, 1)), v=float(rng_a.uniform(-1, 1)))
                                       for det in ("H1", "L1")}))
    for n, c in enumerate(cands):
        c["n"] = n
    boxes = sorted({c["tid"] for c in cands if c["valid"]})
    bg = [c for c in cands if c["kind"] == "bg"]
    fg = [c for c in cands if c["kind"] == "fg"]
    summ = dict(n_bg=len(bg), n_bg_valid=sum(c["valid"] for c in bg), n_fg=len(fg), n_fg_valid=sum(c["valid"] for c in fg),
                n_by_kind={k: sum(c["kind"] == k for c in cands) for k in KINDS},
                n_boxes_bg_fg=len({c["tid"] for c in cands if c["valid"] and c["kind"] in ("bg", "fg")}),
                n_boxes=len(boxes), n_caughtjson_bin_rounding=n_floor, thr1=thr1, half_width=HALF_WIDTH, edge_s=[EDGE_LO / FS, EDGE_HI / FS])
    atomic_json(STATE / "plan.json", dict(summary=summ, boxes=boxes, mcs=[float(m) for m in mcs], candidates=cands))
    log(f"plan: {summ}")


def decoy_params(fam, rng):
    """Decoy parameters (C7), drawn once in the plan. Times are fractions u of the template span before the
    candidate (t = i_c - u * L_max), so one draw serves both L_max passes."""
    def sg():
        return dict(u=float(rng.uniform(0, 1)), f0=float(np.exp(rng.uniform(np.log(50.0), np.log(400.0)))),
                    q=float(rng.uniform(4.0, 20.0)), phi=float(rng.uniform(0, 2 * np.pi)), amp=1.0)
    if fam == "dec_sg":
        return {"H1": [sg()], "L1": [sg()]}
    if fam == "dec_train":
        out = {}
        for det in ("H1", "L1"):
            n = int(rng.integers(3, 11))
            out[det] = [dict(sg(), amp=float(rng.uniform(0.5, 1.0))) for _ in range(n)]
        return out
    return {det: dict(u=float(rng.uniform(0, 1)), a=float(rng.uniform(0.10, 0.30)), dur=float(rng.uniform(10.0, 100.0)))
            for det in ("H1", "L1")}


def decoy_raw(fam, prm, i_c, l_max):
    """Additive raw-strain decoy on the full segment grid (length N), before gating and whitening. For power drift it
    is the CHANGE of the noise, x * a * bump(t), so the caller passes the raw strain x via prm['x']."""
    if fam in ("dec_sg", "dec_train"):
        out = np.zeros(N)
        for g in prm:
            t0 = i_c - g["u"] * l_max * FS
            tau = g["q"] / (np.sqrt(2) * np.pi * g["f0"]) * FS               # samples
            lo, hi = int(max(0, t0 - 6 * tau)), int(min(N, t0 + 6 * tau + 1))
            tt = np.arange(lo, hi) - t0
            out[lo:hi] += g["amp"] * UNIT_STRAIN * np.exp(-(tt / tau) ** 2) * np.cos(2 * np.pi * g["f0"] * tt / FS + g["phi"])
        return out
    c0, half = i_c - prm["u"] * l_max * FS, prm["dur"] * FS / 2
    lo, hi = int(max(0, c0 - half)), int(min(N, c0 + half + 1))
    bump = np.zeros(N)
    tt = (np.arange(lo, hi) - c0) / half
    bump[lo:hi] = 0.5 * (1 + np.cos(np.pi * tt))                         # Hann bump, peak 1 at the centre
    return prm["x"] * prm["a"] * bump


def load_plan():
    return json.loads((STATE / "plan.json").read_text())


# ------------------------------------------------------------------ grids (engine process; lalsimulation + numpy only)
def grid(tid, l_max):
    """Constructed grid + reference lattice + assignment for one box, cached. The grid is coh2c_grid.build's (via
    coh2c_verify.grid_points), with its fitting-factor Monte Carlo kept as a per-box record."""
    path = STATE / "grids" / f"L{l_max}_tid{tid:04d}.npz"
    if path.exists():
        z = dict(np.load(path, allow_pickle=False))
        z["coeffs"] = np.array([L.coeffs(mc, q) for mc, q in z["pts"]])
        return z
    from coh2c_verify import grid_points
    mc_star = float(load_plan()["mcs"][tid])            # Y1's bank, stored by plan (no y1_bank import here)
    pts, gsum = grid_points(mc_star, HALF_WIDTH, l_max)
    refs, lat_worst = L.reference_lattice(mc_star, HALF_WIDTH, l_max, L.dtau_max(4.0))
    cs = np.array([L.coeffs(mc, q) for mc, q in pts])
    asg = [L.assign(c, refs) for c in cs]
    worst = max(d for _, d in asg)
    assert all(np.isfinite(d) for _, d in asg) and worst <= L.dtau_max(4.0), (tid, l_max, worst)   # V3, per box
    out = dict(mc_star=np.float64(mc_star), pts=np.array(pts), coeffs=cs, ref_of=np.array([j for j, _ in asg]),
               ref_mcq=np.array([(r[0], r[1], r[2]) for r in refs]), ref_coeffs=np.array([r[3] for r in refs]),
               worst_dtau=np.float64(worst), ff_min=np.float64(gsum["ff_min"]), ff_p01=np.float64(gsum["ff_p01"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **{k: v for k, v in out.items() if k != "coeffs"})
    os.replace(tmp, path)
    return out


# ------------------------------------------------------------------ children (pycbc lives only here)
def extra_files(c):
    """Per-candidate snippet files a box needs from its child, by kind (deleted after the box)."""
    if c["kind"] in ("fg", "wrong"):               # separate names: an fg box and its +5 % box may share a chunk
        return {det: STATE / "chunk" / f"sig_{det}_{c['inj']}_{c['kind']}.npy" for det in ("H1", "L1")}
    if c["kind"] in DECOYS:
        return {det: STATE / "chunk" / f"dec_{det}_{c['n']}.npy" for det in ("H1", "L1")}
    if c["kind"] == "twin" or c["kind"] in ALLEN:        # Allen injections sit in twin noise of their own seed
        return {det: STATE / "chunk" / f"twin_{det}_{c['n']}.npy" for det in ("H1", "L1")}
    return {}


def chunk(l_max, tids, kinds=KINDS):
    """Child: reference templates (conj, complex64) for each box, and the additive snippets its candidates need:
    injected signals (fg, wrong) and decoys at unit amplitude (dec_*), each gated with the noise's own gates and
    double-whitened with the noise's PSD, exactly as coh2c_local.prep adds injections. Written to state/coh2t/chunk/."""
    import injections as I
    import realbg_cascade as R
    pl = load_plan()
    segs = segments()
    d = STATE / "chunk"
    d.mkdir(parents=True, exist_ok=True)
    table = L._table()
    wcache = {}

    def seg_state(det, gps, need_strain):
        if wcache.get("key") != (det, gps):
            wcache.clear()
            psd = psd_from_kernel(det, gps)                                  # == estimate_psd(x * g), bit for bit
            wcache.update(key=(det, gps), psd=psd, w=R.whiten_filter(psd))
        if need_strain and "x" not in wcache:
            wcache["x"] = np.load(B / "data" / f"{det}_{gps}.npy").astype(np.float64)
            wcache["g"] = R.gate_window(det, gps)[0].astype(np.float32)     # as coh2c_local.prep
        return wcache.get("x"), wcache.get("g"), wcache["psd"], wcache["w"]
    for tid in tids:
        gr = grid(tid, l_max)
        h0 = np.stack([np.conj(L.fd_template(mc, q, f_lo)).astype(np.complex64) for mc, q, f_lo in gr["ref_mcq"]])
        tmp = d / f"refs_L{l_max}_tid{tid:04d}.tmp.npy"
        np.save(tmp, h0)
        os.replace(tmp, d / f"refs_L{l_max}_tid{tid:04d}.npy")
    tidset = set(tids)
    jobs = [(det, segs[c["i"] if det == "H1" else c["iL"]], c, out) for c in pl["candidates"]
            if c["tid"] in tidset and c["valid"] and c["kind"] in kinds and c["kind"] != "null"
            for det, out in extra_files(c).items()]
    for det, gps, c, out in sorted(jobs, key=lambda j: (j[0], j[1], j[2]["n"])):   # one PSD per (det, segment)
        if out.exists():
            continue
        is_twin = c["kind"] == "twin" or c["kind"] in ALLEN
        x, g, psd, w = seg_state(det, gps, need_strain=not is_twin)
        if is_twin:
            tmp = out.with_name(out.stem + ".tmp.npy")
            np.save(tmp, twin_snippet(psd, w, c, det))
            os.replace(tmp, out)
            continue
        if c["kind"] in ("fg", "wrong"):
            s_, s0, _, _ = R._injection_template(table[c["inj"]], det, gps)
            raw = I.add_signal(np.zeros(R.N), gps, s_, s0)
            del s_
        elif c["kind"] == "dec_drift":
            raw = decoy_raw(c["kind"], dict(c["decoy"][det], x=x), c["i_c"], l_max)
        else:
            raw = decoy_raw(c["kind"], c["decoy"][det], c["i_c"], l_max)
        ys = np.fft.irfft(np.fft.rfft(raw * g) * w * w, n=R.N)
        lo = snippet_start(c["i_c"])
        tmp = out.with_name(out.stem + ".tmp.npy")
        np.save(tmp, ys[lo: lo + NS].astype(np.float32))
        os.replace(tmp, out)
        del raw, ys
    log(f"chunk L{l_max} {len(tids)} boxes done | peak footprint {L.peak_footprint_mb()} MB")


def psd_with_kernel(x):
    """lowmem_psd.estimate_psd, line for line, also returning the nonzero ends of its inverse-spectrum-truncation
    kernel q (L/2 samples at each end; zero in between). 1/|rfft(q)|^2 IS the PSD, so ~0.5 MB rebuilds it exactly."""
    import lowmem_psd as LP
    fs, n = C.SAMPLE_RATE, len(x)
    Lw = int(round(C.PSD_SEG_SEC * fs)); S = Lw // 2
    nseg = n // S
    if (nseg - 1) * S + Lw > n:
        nseg -= 1
    data_len = (nseg - 1) * S + Lw
    if data_len < n:
        diff = n - data_len
        a = diff // 2 + (diff % 2)
        x = x[a:n - diff // 2]
    win = np.hanning(Lw)
    dt, df = 1.0 / fs, fs / Lw
    P = np.empty((nseg, Lw // 2 + 1))
    for i in range(nseg):
        P[i] = np.abs(np.fft.rfft(x[i * S:i * S + Lw] * win) * dt) ** 2
    P[:, 0] /= 2; P[:, -1] /= 2
    psd = np.median(P, axis=0) / LP._median_bias(nseg)
    del P
    psd *= 2 * df * Lw / (win * win).sum()
    new_df = fs / n
    new_n = int(np.rint((len(psd) - 1) * df / new_df + 1))
    psd = np.interp(np.arange(new_n) * new_df, np.arange(len(psd)) * df, psd)
    Nn = (len(psd) - 1) * 2
    kmin = int((C.F_LOWER - 10) / new_df)
    inv = np.zeros(len(psd))
    inv[kmin:Nn // 2] = psd[kmin:Nn // 2] ** -0.5
    del psd
    q = np.fft.irfft(inv, n=Nn)
    del inv
    q[Lw // 2:Nn - Lw // 2] = 0
    ends = (q[:Lw // 2].copy(), q[Nn - Lw // 2:].copy())
    return 1.0 / np.abs(np.fft.rfft(q)) ** 2, ends


def _psd_from_ends(qa, qb):
    q = np.zeros(N)
    q[:len(qa)], q[N - len(qb):] = qa, qb
    return 1.0 / np.abs(np.fft.rfft(q)) ** 2


def psd_from_kernel(det, gps):
    z = np.load(STATE / "kern" / f"kern_{det}_{gps}.npz")
    return _psd_from_ends(z["qa"], z["qb"])


def kernel(det, gps):
    """Child: the PSD kernel of one gated tuning series, with two bit-for-bit guards: it reproduces
    lowmem_psd.estimate_psd on the same gated input, and its whitening filter on the snippet grid equals the one
    coh2c_local.prep stored with the noise series."""
    import realbg_cascade as R
    from lowmem_psd import estimate_psd
    gps = int(gps)
    x = np.load(B / "data" / f"{det}_{gps}.npy").astype(np.float64)
    x *= R.gate_window(det, gps)[0].astype(np.float32)                     # exactly prep's gating
    psd, (qa, qb) = psd_with_kernel(x)
    assert np.array_equal(psd, estimate_psd(x)), "kernel PSD differs from lowmem_psd.estimate_psd"
    rebuilt = _psd_from_ends(qa, qb)
    assert np.array_equal(rebuilt, psd), "PSD not rebuilt exactly from its kernel"
    w_snip = np.load(L.STATE / f"wsnip_{L.ww_tag(det, gps)}.npy")
    assert np.array_equal(R.whiten_filter(rebuilt)[::8], w_snip), "filter differs from prep's"
    out = STATE / "kern" / f"kern_{det}_{gps}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez(tmp, qa=qa, qb=qb)
    os.replace(tmp, out)
    log(f"kernel {det} {gps}: PSD and filter reproduced bit for bit")


def twin_snippet(psd, w, c, det):
    """C9 as amended (option (i), the Bridge 2026-11-26): realbg_cascade's twin construction on the FULL 4096-s grid
    (Gaussian, the gated series' own PSD, double-whitened), with an independent seed per candidate and detector,
    then the same 512-s rectangular cut the real series gets. Float32, as the noise series."""
    rng = np.random.default_rng([SEED, int(c["n"]), 0 if det == "H1" else 1])
    X = np.sqrt(N * FS * psd / 4.0) * (rng.standard_normal(len(psd)) + 1j * rng.standard_normal(len(psd)))
    X[0], X[-1] = X[0].real, X[-1].real
    y = np.fft.irfft(X * w * w, n=N)
    lo = snippet_start(c["i_c"])
    return y[lo: lo + NS].astype(np.float32)


def twinexact():
    """Registered check for option (i), bars fixed before its first run: for 4 twin candidates (both detectors, the
    first tuning segment, different seeds and cut positions), the driver's twin snippet (kernel PSD -> twin_snippet, the chunk child's exact path) must be
    BIT-IDENTICAL to a reference built independently from the data: lowmem_psd.estimate_psd of prep's gated series,
    realbg_cascade.whiten_filter, realbg's twin lines written out again here, the same seed, the same 512-s cut.
    Requires that segment's kernels (`kernel H1|L1 <gps>`). Tests that the code does what (i) says; K1-(A) tests the statistics."""
    import realbg_cascade as R
    from lowmem_psd import estimate_psd
    pl = load_plan()
    segs = segments()
    # segment 0: the one segment whose noise series and filter already exist (prep), in both detectors
    tH = [c for c in pl["candidates"] if c["kind"] == "twin" and c["i"] == 0]
    tL = [c for c in pl["candidates"] if c["kind"] == "twin" and c["iL"] == 0]
    picks = [(tH[0], "H1"), (tL[len(tL) // 2], "L1"), (tH[-1], "H1"), (tL[-1], "L1")]
    rows = []
    for c, det in picks:
        gps = segs[c["i"] if det == "H1" else c["iL"]]
        psd_k = psd_from_kernel(det, gps)
        mine = twin_snippet(psd_k, R.whiten_filter(psd_k), c, det)
        x = np.load(B / "data" / f"{det}_{gps}.npy").astype(np.float64)
        x *= R.gate_window(det, gps)[0].astype(np.float32)
        psd = estimate_psd(x)
        del x
        w = R.whiten_filter(psd)
        rng = np.random.default_rng([SEED, int(c["n"]), 0 if det == "H1" else 1])
        sig = np.sqrt(N * FS * psd / 4.0)
        X = sig * (rng.standard_normal(len(psd)) + 1j * rng.standard_normal(len(psd)))
        X[0] = X[0].real; X[-1] = X[-1].real
        full = np.fft.irfft(X * w * w, n=N)
        lo = snippet_start(c["i_c"])
        ref = full[lo: lo + NS].astype(np.float32)
        rows.append(dict(n=c["n"], det=det, segment=gps, i_c=c["i_c"], identical=bool(np.array_equal(mine, ref)),
                         max_abs_diff=float(np.max(np.abs(mine.astype(np.float64) - ref))), rms=float(np.std(ref.astype(np.float64)))))   # float64: float32 squares of ~1e21 overflow
        log(f"twinexact n {c['n']} {det} {gps}: identical {rows[-1]['identical']}")
    ok = all(r["identical"] for r in rows)
    res = dict(check="C9 (i) exact identity", rows=rows, passed=bool(ok), peak_footprint_mb=L.peak_footprint_mb(),
               model="Claude Opus 5.5", env=L.stamp())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "twinexact.json").write_text(json.dumps(res, indent=1))
    log(f"C9 (i) exact identity {'PASS' if ok else 'FAIL'} | footprint {L.peak_footprint_mb()} MB")


def k1_path(det, gps, r):
    return STATE / "k1" / f"pyc_{det}_{gps}_r{r}.npy"


def k1_series(det, gps, r):
    """C10: pycbc.noise.noise_from_psd (lal ranlux + lalsimulation.SimNoise) at the segment's exact pipeline PSD,
    sampled at its own 1/16-Hz resolution and extended flat below 40 Hz, then prep's whitening path: rfft x w^2,
    with the same full-grid filter as the real series. Synthetic noise is not gated. Float32, as the noise series."""
    import realbg_cascade as R
    from pycbc.noise.gaussian import noise_from_psd
    from pycbc.types import FrequencySeries
    segs = segments()
    psd = psd_from_kernel(det, gps)
    p16 = psd[::256].copy()                                            # bins at k/16 Hz
    f16 = np.arange(len(p16)) / 16.0
    k40 = int(np.searchsorted(f16, C.F_LOWER - 10))
    p16[:k40] = p16[k40]
    seed = K1_SEED0 + 1000 * int(r) + 2 * segs.index(int(gps)) + (0 if det == "H1" else 1)
    x = noise_from_psd(N, 1.0 / FS, FrequencySeries(p16, delta_f=1.0 / 16), seed=seed).numpy()
    w = R.whiten_filter(psd)
    return np.fft.irfft(np.fft.rfft(x) * w * w, n=N).astype(np.float32), x, psd, seed


def k1gen(det, gps, r):
    """Child: one pycbc realisation of one (det, segment), written for the engine to memory-map."""
    out = k1_path(det, int(gps), int(r))
    if out.exists():
        return
    y, _, _, seed = k1_series(det, int(gps), int(r))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.npy")
    np.save(tmp, y)
    os.replace(tmp, out)
    log(f"k1gen {det} {gps} r{r} (seed {seed}) | footprint {L.peak_footprint_mb()} MB")


def k1gen_all(r):
    """Realisation r for every (det, segment); any other realisation's series are deleted first (disk, C10)."""
    for f in (STATE / "k1").glob("pyc_*.npy"):
        if not f.stem.endswith(f"_r{r}"):
            f.unlink()
    for det in ("H1", "L1"):
        for g in segments():
            if not k1_path(det, g, r).exists():
                subprocess.run([sys.executable, "-u", __file__, "k1gen", det, str(g), str(r)], check=True)


def k1precheck():
    """C10 (1), bars fixed before the first run: for tuning segment 0, H1 and L1, the pipeline's own Welch estimate
    (lowmem_psd.estimate_psd) of the generated, un-whitened pycbc series over the target PSD, averaged in each of 16
    equal bands over [50, 1024] Hz, within +-1.5 % (about 6 sigma of the expected 0.25 % band-mean scatter)."""
    from lowmem_psd import estimate_psd
    gps = segments()[0]
    rows, ok = [], True
    for det in ("H1", "L1"):
        _, x, psd, seed = k1_series(det, gps, 99)                     # realisation 99: never used by K1 itself
        est = estimate_psd(x)
        f = np.fft.rfftfreq(N, d=1.0 / FS)
        edges = np.linspace(L.F_LO, L.F_HI, 17)
        bands = [float(np.mean((est / psd)[(f >= a) & (f < b)])) for a, b in zip(edges[:-1], edges[1:])]
        assert all(np.isfinite(v) for v in bands), "non-finite band mean"
        worst = max(abs(v - 1) for v in bands)
        ok &= worst <= 0.015
        rows.append(dict(det=det, segment=gps, seed=seed, band_means=bands, worst_dev=worst))
        log(f"k1precheck {det}: worst band deviation {worst:.4f}")
    res = dict(check="C10 (1) pycbc noise reproduces the target PSD", bar=0.015, rows=rows, passed=bool(ok),
               peak_footprint_mb=L.peak_footprint_mb(), model="Claude Opus 5.5", env=L.stamp())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "k1precheck.json").write_text(json.dumps(res, indent=1))
    log(f"C10 (1) PSD pre-check {'PASS' if ok else 'FAIL'} | footprint {L.peak_footprint_mb()} MB")


SERIES_SHA = OUT / "series_sha256.json"


def _sha(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def _series_files(det, g):
    tag = L.ww_tag(det, g)
    return [L.STATE / f"ww_{tag}.npy", L.STATE / f"wsnip_{tag}.npy", L.STATE / f"rows_{tag}.json"]


def k1perfect():
    """C10 addendum control (registered in 69acadf before this ran): a PERFECT generator (Gaussian on the full
    4096-s circle drawn exactly from the target PSD, X_k = sqrt(N fs S_k / 4)(a + i b), irfft; no pycbc), through the
    same estimate_psd and the same 16 bands as k1precheck. Decision per detector on the two notch bands (476-537,
    963-1024 Hz): ILL-POSED if the perfect generator also fails +-1.5 % in BOTH and in each 2/3 <= (r_perf - 1) /
    (r_pycbc - 1) <= 3/2; GENERATOR AT FAULT if it passes all 16 bands; INCONCLUSIVE otherwise. Also the line heights
    (target PSD peak / band median; fraction of bins above 10x the band median) per band."""
    from lowmem_psd import estimate_psd
    pyc = {r["det"]: r for r in json.loads((OUT / "k1precheck.json").read_text())["rows"]}
    gps = segments()[0]
    f = np.fft.rfftfreq(N, d=1.0 / FS)
    edges = np.linspace(L.F_LO, L.F_HI, 17)
    notch = (7, 15)                                                     # band indices of 476-537 and 963-1024 Hz
    rows = []
    for det in ("H1", "L1"):
        psd = psd_from_kernel(det, gps)
        rng = np.random.default_rng([SEED, 3, 0 if det == "H1" else 1])
        X = np.sqrt(N * FS * psd / 4.0) * (rng.standard_normal(len(psd)) + 1j * rng.standard_normal(len(psd)))
        X[0], X[-1] = X[0].real, X[-1].real
        x = np.fft.irfft(X, n=N)
        del X
        est = estimate_psd(x)
        del x
        bands, heights = [], []
        for a_, b_ in zip(edges[:-1], edges[1:]):
            m = (f >= a_) & (f < b_)
            bands.append(float(np.mean((est / psd)[m])))
            med = float(np.median(psd[m]))
            heights.append(dict(peak_over_median=float(psd[m].max() / med), frac_above_10x=float(np.mean(psd[m] > 10 * med))))
        assert all(np.isfinite(v) for v in bands)
        perf_fail = [abs(bands[i] - 1) > 0.015 for i in notch]
        ratio = [(bands[i] - 1) / (pyc[det]["band_means"][i] - 1) for i in notch]
        passes_all = all(abs(v - 1) <= 0.015 for v in bands)
        verdict = ("ILL-POSED" if all(perf_fail) and all(2 / 3 <= r_ <= 1.5 for r_ in ratio) else
                   "GENERATOR AT FAULT" if passes_all else "INCONCLUSIVE")
        rows.append(dict(det=det, segment=gps, perfect_band_means=bands, pycbc_band_means=pyc[det]["band_means"],
                         notch_bands_hz=[[float(edges[i]), float(edges[i + 1])] for i in notch],
                         excess_ratio_perfect_over_pycbc=ratio, perfect_passes_all=passes_all, line_heights=heights,
                         verdict=verdict))
        log(f"k1perfect {det}: notch perfect {[round(bands[i], 4) for i in notch]} vs pycbc "
            f"{[round(pyc[det]['band_means'][i], 4) for i in notch]} -> ratio {[round(r_, 3) for r_ in ratio]} | {verdict}")
    res = dict(control="C10 addendum perfect generator", registered="69acadf", rows=rows,
               peak_footprint_mb=L.peak_footprint_mb(), model="Claude Opus 5.5", env=L.stamp())
    (OUT / "k1perfect.json").write_text(json.dumps(res, indent=1))


def k1diag():
    """C13 (registered before this ran): what makes pycbc's noise depart from the exact generator. Variants per
    detector on segment 0 (seed of K1 realisation 99, as k1precheck): V0 pycbc df 1/16 (must reproduce k1precheck
    exactly), V1 df 1/64, V2 df 1/8, V3 df 1/16 with the PSD above 1024 Hz flat at its 1024-Hz value, V4 the perfect
    generator fed the 1/16-sampled PSD (linear interpolation to the full grid), V5 the perfect generator (must
    reproduce k1perfect exactly). Explanation rule, deviations and outputs: reports/stepT_clarifications.md C13."""
    from lowmem_psd import estimate_psd
    from pycbc.noise.gaussian import noise_from_psd
    from pycbc.types import FrequencySeries
    gps = segments()[0]
    f = np.fft.rfftfreq(N, d=1.0 / FS)
    edges = np.linspace(L.F_LO, L.F_HI, 17)
    pyc = {r["det"]: r for r in json.loads((OUT / "k1precheck.json").read_text())["rows"]}
    perf = {r["det"]: r for r in json.loads((OUT / "k1perfect.json").read_text())["rows"]}
    devs = {"D1": ("L1", 15, 1.572 / 1.018), "D2": ("H1", 7, 11.07 / 28.81), "D3": ("H1", 15, 11.14 / 52.24)}

    def bands(est, psd):
        return [float(np.mean((est / psd)[(f >= a_) & (f < b_)])) for a_, b_ in zip(edges[:-1], edges[1:])]

    def pycbc_noise(psd, step, flat_above_1024=False, seed=None):
        ps = psd.copy()
        if flat_above_1024:
            k = int(np.searchsorted(f, L.F_HI))
            ps[k:] = ps[k]
        pd = ps[::step].copy()
        fd = np.arange(len(pd)) * (FS / N) * step
        k40 = int(np.searchsorted(fd, C.F_LOWER - 10))
        pd[:k40] = pd[k40]
        return noise_from_psd(N, 1.0 / FS, FrequencySeries(pd, delta_f=FS / N * step), seed=seed).numpy()

    def perfect(psd_used, det, tag):
        rng = np.random.default_rng([SEED, tag, 0 if det == "H1" else 1])
        X = np.sqrt(N * FS * psd_used / 4.0) * (rng.standard_normal(len(psd_used)) + 1j * rng.standard_normal(len(psd_used)))
        X[0], X[-1] = X[0].real, X[-1].real
        return np.fft.irfft(X, n=N)
    out = {}
    for det in ("H1", "L1"):
        psd = psd_from_kernel(det, gps)
        seed = K1_SEED0 + 1000 * 99 + 0 + (0 if det == "H1" else 1)
        res = {}
        for name, gen in (("V0_df16", lambda: pycbc_noise(psd, 256, seed=seed)),
                          ("V1_df64", lambda: pycbc_noise(psd, 64, seed=seed)),
                          ("V2_df8", lambda: pycbc_noise(psd, 512, seed=seed)),
                          ("V3_df16_flat_above_1024", lambda: pycbc_noise(psd, 256, True, seed=seed)),
                          ("V4_perfect_fed_df16", lambda: perfect(np.interp(f, f[::256], psd[::256]), det, 4)),
                          ("V5_perfect", lambda: perfect(psd, det, 3))):
            x = gen()
            res[name] = bands(estimate_psd(x), psd)
            del x
            log(f"k1diag {det} {name}: 476-537 {res[name][7]:.4f}  963-1024 {res[name][15]:.4f}")
        repro = dict(V0_equals_k1precheck=res["V0_df16"] == pyc[det]["band_means"],
                     V5_equals_k1perfect=res["V5_perfect"] == perf[det]["perfect_band_means"])
        out[det] = dict(bands=res, reproduction=repro)
    verdicts = {}
    for dname, (det, bi, dval) in devs.items():
        r = out[det]["bands"]
        perf_b = r["V5_perfect"][bi]
        rel = {k: r[k][bi] / perf_b for k in r}
        within = lambda k: abs(rel[k] - 1) <= 0.25 * abs(dval - 1)
        sampling = (within("V1_df64") or within("V2_df8")) and (
            np.sign(rel["V4_perfect_fed_df16"] - 1) == np.sign(dval - 1) and abs(rel["V4_perfect_fed_df16"] - 1) >= 0.5 * abs(dval - 1))
        above = within("V3_df16_flat_above_1024")
        trend = [rel["V2_df8"], rel["V0_df16"], rel["V1_df64"]]
        monotone = bool(trend == sorted(trend) or trend == sorted(trend, reverse=True))
        verdicts[dname] = dict(det=det, band=[float(edges[bi]), float(edges[bi + 1])], deviation_registered=dval,
                               variant_over_perfect=rel, explained_by_psd_sampling=bool(sampling),
                               explained_by_psd_above_1024=bool(above),
                               verdict=("EXPLAINED: PSD sampling" if sampling else "EXPLAINED: PSD above 1024 Hz" if above
                                        else "UNEXPLAINED"), moves_with_df_monotone=monotone)
        log(f"k1diag {dname} {det} band {bi}: {verdicts[dname]['verdict']} | df 1/8, 1/16, 1/64 -> "
            f"{[round(v, 3) for v in trend]} | flat>1024 {rel['V3_df16_flat_above_1024']:.3f} | perfect fed df16 "
            f"{rel['V4_perfect_fed_df16']:.3f}")
    res = dict(control="C13 pycbc deviation diagnosis", runs=out, deviations=verdicts,
               peak_footprint_mb=L.peak_footprint_mb(), model="Claude Opus 5.5", env=L.stamp())
    (OUT / "k1diag.json").write_text(json.dumps(res, indent=1))


# ------------------------------------------------------------------ K2 (linearity vs really injected strain) and N1
K2_DIR = STATE / "k2"


def k2select():
    """K2 sample (registration §5): 15 caught injections, stratified by chirp mass (terciles) x q (< / >= 0.5),
    at least 3 with optimal SNR > 30 (Y0's oracle, used only to select), and one whose template span
    [merger - 384 s, merger] holds a gate (a gated glitch). Seeded; written once."""
    import csv
    pl = load_plan()
    out = K2_DIR / "k2_selection.json"
    if out.exists():
        return
    with open(B / "results" / "oracle_snr.csv") as f:
        orc = {int(r["id"]): float(r["snr_net"]) for r in csv.DictReader(f)}
    tab = L._table()
    import injections as I
    segs = segments()
    fg = [c for c in pl["candidates"] if c["kind"] == "fg" and c["valid"]]
    mc = {c["inj"]: I.chirp_mass(tab[c["inj"]]["mass1"], tab[c["inj"]]["mass2"]) for c in fg}
    qq = {c["inj"]: min(tab[c["inj"]]["mass1"], tab[c["inj"]]["mass2"]) / max(tab[c["inj"]]["mass1"], tab[c["inj"]]["mass2"]) for c in fg}
    t1, t2 = np.quantile(list(mc.values()), [1 / 3, 2 / 3])
    rng = np.random.default_rng([SEED, 5])
    pick = []
    for lo, hi in ((0, t1), (t1, t2), (t2, 9)):
        for qlo, qhi in ((0, 0.5), (0.5, 1.01)):
            cell = [c for c in fg if lo <= mc[c["inj"]] < hi and qlo <= qq[c["inj"]] < qhi and c not in pick]
            pick += [cell[int(j)] for j in rng.choice(len(cell), min(2, len(cell)), replace=False)]
    loud = [c for c in fg if orc[c["inj"]] > 30 and c not in pick]
    while sum(orc[c["inj"]] > 30 for c in pick) < 3 and loud:
        pick.append(loud.pop(int(rng.integers(len(loud)))))

    def gated(c):
        gps = segs[c["i"]]
        for det in ("H1", "L1"):
            times = json.loads((B / "state" / "realbg" / f"gates_{det}_{gps}.json").read_text())
            t_c = gps + c["i_c"] / FS
            if any(t_c - 384 <= t <= t_c for t in times):
                return True
        return False
    if not any(gated(c) for c in pick):
        cand = [c for c in fg if gated(c) and c not in pick]
        assert cand, "no caught injection sits on a gated glitch"
        pick.append(cand[int(rng.integers(len(cand)))])
    pick = pick[:15] if len(pick) > 15 else pick
    K2_DIR.mkdir(parents=True, exist_ok=True)
    atomic_json(out, dict(injections=[c["inj"] for c in pick], n=len(pick),
                          n_loud=int(sum(orc[c["inj"]] > 30 for c in pick)), n_gated=int(sum(gated(c) for c in pick))))
    log(f"K2 selection: {len(pick)} injections")


def _gate_window(times, gps):
    from pycbc.types import TimeSeries
    g = TimeSeries(np.ones(N), delta_t=1.0 / FS, epoch=gps)
    for t in times:
        g = g.gate(t, window=0.25, method="taper", copy=False, taper_width=0.25)
    return g.numpy().copy()


def k2gates(det, gps, inj):
    """Child: PyCBC's glitch detector on the REALLY injected strain (realbg F2b: detection in its own process)."""
    import injections as I
    import realbg_cascade as R
    from pycbc.strain.strain import detect_loud_glitches
    from pycbc.types import TimeSeries
    gps, inj = int(gps), int(inj)
    out = K2_DIR / f"gates_{det}_{inj}.json"
    if out.exists():
        return
    x = np.load(B / "data" / f"{det}_{gps}.npy").astype(np.float64)
    s_, s0, _, _ = R._injection_template(L._table()[inj], det, gps)
    x += I.add_signal(np.zeros(N), gps, s_, s0)
    atomic_json(out, [float(t) for t in detect_loud_glitches(TimeSeries(x, delta_t=1.0 / FS, epoch=gps))])
    log(f"k2 gates {det} {inj} | footprint {L.peak_footprint_mb()} MB")


def k2prep(det, gps, inj):
    """Child: the really injected series: raw + signal, gated at ITS OWN detected gates, PSD and whitening from the
    injected gated strain, double-whitened (as coh2c_local.prep); its snippet-grid filter; its PyCBC v series."""
    import injections as I
    import realbg_cascade as R
    import coh2t
    from lowmem_psd import estimate_psd
    from pycbc.psd.variation import calc_filt_psd_variation
    from pycbc.types import TimeSeries
    gps, inj = int(gps), int(inj)
    tag = f"{det}_{inj}"
    if (K2_DIR / f"v_{tag}.npz").exists():
        return
    x = np.load(B / "data" / f"{det}_{gps}.npy").astype(np.float64)
    s_, s0, _, _ = R._injection_template(L._table()[inj], det, gps)
    x += I.add_signal(np.zeros(N), gps, s_, s0)
    g = _gate_window(json.loads((K2_DIR / f"gates_{tag}.json").read_text()), gps).astype(np.float32)
    x *= g
    w = R.whiten_filter(estimate_psd(x))
    y = np.fft.irfft(np.fft.rfft(x) * w * w, n=N).astype(np.float32)
    np.save(K2_DIR / f"ww_{tag}.npy", y)
    np.save(K2_DIR / f"wsnip_{tag}.npy", w[::8].copy())
    pv = calc_filt_psd_variation(TimeSeries(x, delta_t=1.0 / FS, epoch=gps), **coh2t.V_ARGS)
    np.savez(K2_DIR / f"v_{tag}.npz", values=np.asarray(pv.numpy(), np.float64), start=float(pv.start_time),
             delta_t=float(pv.delta_t))
    log(f"k2 prep {tag} | footprint {L.peak_footprint_mb()} MB")


def k2run(l_max):
    """K2: each selected injection scored twice in its own box, by the shortcut (noise series + gated signal snippet,
    the production path) and on the really injected series; both outputs stored for the reducer (which applies each
    strain's own v). Bars (registration §5): rho^_net and chi^2_r within 2 %, every injection."""
    l_max = int(l_max)
    sel = json.loads((K2_DIR / "k2_selection.json").read_text())["injections"]
    pl = load_plan()
    segs = segments()
    fgc = {c["inj"]: c for c in pl["candidates"] if c["kind"] == "fg"}
    dev = L.pick_dev()
    sign = json.loads((L.OUT / "E1_phase_model.json").read_text())["sign"]
    eng, fine = L.Engine(dev, 4.0, 1), Fine(dev)
    res = []
    for inj in sel:
        c = fgc[inj]
        gps = segs[c["i"]]
        for det in ("H1", "L1"):
            subprocess.run([sys.executable, "-u", __file__, "k2gates", det, str(gps), str(inj)], check=True)
            subprocess.run([sys.executable, "-u", __file__, "k2prep", det, str(gps), str(inj)], check=True)
        subprocess.run([sys.executable, "-u", __file__, "chunk", str(l_max), str(c["tid"]), "fg"], check=True)
        box = Box(c["tid"], l_max, dev, eng, sign, np.load(STATE / "chunk" / f"refs_L{l_max}_tid{c['tid']:04d}.npy"))
        series = lambda det, g, _c={}: _c.setdefault((det, g), L.load_ww(det, g))
        sig = {det: np.load(f) for det, f in extra_files(c).items()}
        short = candidate(box, fine, series, c, sig)
        # the really injected strain carries its OWN gates, PSD and filter, so its own snippet-grid filter is used
        series_real = lambda det, g: (np.load(K2_DIR / f"ww_{det}_{inj}.npy", mmap_mode="r"),
                                      np.load(K2_DIR / f"wsnip_{det}_{inj}.npy"), {})
        real = candidate(box, fine, series_real, dict(c, kind="k2real"))
        res.append(dict(inj=inj, short=short, real=real))
        for det in ("H1", "L1"):
            (K2_DIR / f"ww_{det}_{inj}.npy").unlink(missing_ok=True)
        for f in list(extra_files(c).values()) + [STATE / "chunk" / f"refs_L{l_max}_tid{c['tid']:04d}.npy"]:
            f.unlink(missing_ok=True)
        log(f"K2 inj {inj}: done | footprint {L.peak_footprint_mb()} MB")
    keys = [k for k in res[0]["short"] if k in res[0]["real"]]
    np.savez(OUT / f"K2_L{l_max}.npz", inj=np.array(sel),
             **{f"short_{k}": np.array([r["short"][k] for r in res]) for k in keys},
             **{f"real_{k}": np.array([r["real"][k] for r in res]) for k in keys})


def n1():
    """N1 (registration §5): float32 vs float64 through the identical chain for a 384-s light template: the
    lightest-Mc background box at L_max 384, its grid template nearest the centre, a rho = 20 injection of that
    template in real tuning noise; the engine's complex z over the +-1-s lags in float32 (production) and with every
    tensor in float64. Bar: normalised match |<z32, z64>| / (|z32| |z64|) >= 0.9999."""
    import torch
    pl = load_plan()
    segs = segments()
    bgv = [c for c in pl["candidates"] if c["kind"] == "bg" and c["valid"]]
    c = min(bgv, key=lambda c_: pl["mcs"][c_["tid"]])
    dev = L.pick_dev()
    sign = json.loads((L.OUT / "E1_phase_model.json").read_text())["sign"]
    gr = grid(c["tid"], 384)
    refs = [(r_[0], r_[1], r_[2]) for r_ in gr["ref_mcq"]]
    h0 = np.stack([np.conj(L.fd_template(mc, q, f_lo)).astype(np.complex64) for mc, q, f_lo in refs])
    eng, fine = L.Engine(dev, 4.0, 1), Fine(dev)
    box = Box(c["tid"], 384, dev, eng, sign, h0, gr=gr)
    tsel = int(np.argmin(np.abs(np.log(gr["pts"][:, 0] / gr["mc_star"])) + np.abs(gr["pts"][:, 1] - gr["pts"][:, 1].mean())))
    y, w_snip, _ = L.load_ww("H1", segs[c["i"]])
    ysn, kappa = noise_spectrum(c, y)
    lo = snippet_start(c["i_c"])
    ysn = ysn + allen_signal(box, w_snip, gr["coeffs"][tsel], 20.0, c["i_c"] - lo, sign, int(gr["ref_of"][tsel]))
    j = int(gr["ref_of"][tsel])
    sig0 = np.sqrt(2 * np.sum((np.abs(h0[j]) * w_snip) ** 2) / NS)
    out = {}
    for name, cdt, rdt in (("f32", torch.complex64, torch.float32), ("f64", torch.complex128, torch.float64)):
        e = L.Engine("cpu", 4.0, 1)
        e.down, e.basis = e.down.to(cdt), e.basis.to(rdt)
        yt = e.window(L.analytic_corr(torch, torch.tensor(ysn * kappa, dtype=cdt), torch.tensor(h0[j], dtype=cdt)),
                      c["i_c"] - lo)
        dco = torch.tensor((gr["coeffs"][tsel] - gr["ref_coeffs"][j])[None, :], dtype=rdt)
        out[name] = e.z(yt, dco, sign)[0].numpy().astype(np.complex128) * (1.0 / (sig0 * kappa))
    a, b = out["f32"], out["f64"]
    match = float(abs(np.vdot(b, a)) / np.sqrt(np.vdot(a, a).real * np.vdot(b, b).real))
    peak_rel = float(abs(np.abs(a).max() / np.abs(b).max() - 1))
    res = dict(gate="N1", tid=c["tid"], mc_star=float(gr["mc_star"]), template=tsel, match=match, bar=0.9999,
               peak_rel_diff=peak_rel, peak_abs_f64=float(np.abs(b).max()), passed=bool(match >= 0.9999),
               peak_footprint_mb=L.peak_footprint_mb(), model="Claude Opus 5.5", env=L.stamp())
    (OUT / "N1.json").write_text(json.dumps(res, indent=1))
    log(f"N1 {'PASS' if res['passed'] else 'FAIL'}: match {match:.7f}, peak rel diff {peak_rel:.1e} | "
        f"footprint {L.peak_footprint_mb()} MB")


# ------------------------------------------------------------------ check (i), C17
CI_DIR = STATE / "checki"
CI_SERIES, CI_SNIPS, CI_TEMPLATES, DELTA_STAR = 144, 31, 7, 0.003
CI_GENS = ("exact_a", "exact_b", "planted", "pycbc_v3", "pycbc_orig")


def ci_boxes():
    pl = load_plan()
    has = sorted({c["tid"] for c in pl["candidates"] if c["kind"] == "bg" and c["valid"]})
    return [min(has, key=lambda t: abs(pl["mcs"][t] - m)) for m in (0.20, 0.35, 0.60)]


def ci_series(gen, det, j):
    """Child: one 4,096-s series of generator gen, whitened by prep's path (x w^2, the target filter, no gating),
    written as float32 for the engine. Seeds distinct per (generator, detector, series)."""
    import realbg_cascade as R
    gi, di, j = CI_GENS.index(gen), (0 if det == "H1" else 1), int(j)
    gps = segments()[0]
    out = CI_DIR / f"{gen}_{det}_{j}.npy"
    if out.exists():
        return
    psd = psd_from_kernel(det, gps)
    w = R.whiten_filter(psd)
    if gen.startswith("pycbc"):
        from pycbc.noise.gaussian import noise_from_psd
        from pycbc.types import FrequencySeries
        f = np.fft.rfftfreq(N, d=1.0 / FS)
        ps = psd.copy()
        if gen == "pycbc_v3":                                  # C16: flat above F_HIGH
            k = int(np.searchsorted(f, L.F_HI)); ps[k:] = ps[k]
        p16 = ps[::256].copy()
        k40 = int(np.searchsorted(np.arange(len(p16)) / 16.0, C.F_LOWER - 10)); p16[:k40] = p16[k40]
        x = noise_from_psd(N, 1.0 / FS, FrequencySeries(p16, delta_f=1.0 / 16), seed=9_000_000 + 10_000 * gi + 1_000 * di + j).numpy()
    else:
        ps = psd.copy()
        if gen == "planted":                                   # C17 must-fire: PSD x (1 + delta*) over 50-300 Hz
            f = np.fft.rfftfreq(N, d=1.0 / FS)
            ps[(f >= 50) & (f <= 300)] *= 1 + DELTA_STAR
        rng = np.random.default_rng([SEED, 17, gi, di, j])
        X = np.sqrt(N * FS * ps / 4.0) * (rng.standard_normal(len(ps)) + 1j * rng.standard_normal(len(ps)))
        X[0], X[-1] = X[0].real, X[-1].real
        x = np.fft.irfft(X, n=N)
    y = np.fft.irfft(np.fft.rfft(x) * w * w, n=N).astype(np.float32)
    CI_DIR.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.npy")
    np.save(tmp, y)
    os.replace(tmp, out)


def checki(gens):
    """C17 runner: for each generator in gens (comma list) and detector, CI_SERIES series, each scored on CI_SNIPS
    snippets (centres 128 s apart in the valid window) by CI_TEMPLATES templates of each of the 3 boxes at L_max 128
    and 384: the mean over the +-1-s lags of noise-only |z|^2. Saves [series, snippet, box_L, template] per
    (generator, detector), checkpointed per series."""
    import torch
    dev = L.pick_dev()
    sign = json.loads((L.OUT / "E1_phase_model.json").read_text())["sign"]
    eng, fine = L.Engine(dev, 4.0, 1), Fine(dev)
    gps = segments()[0]
    centres = (392 + 64 + 128 * np.arange(CI_SNIPS)) * FS
    boxes = []
    for l_max in (128, 384):
        for tid in ci_boxes():
            subprocess.run([sys.executable, "-u", __file__, "chunk", str(l_max), str(tid), "bg"], check=True)
            bx = Box(tid, l_max, dev, eng, sign, np.load(STATE / "chunk" / f"refs_L{l_max}_tid{tid:04d}.npy"))
            sel = np.linspace(0, bx.n - 1, CI_TEMPLATES).round().astype(int)
            boxes.append((l_max, tid, bx, sel))
    for gen in gens.split(","):
        assert gen in CI_GENS
        for det in ("H1", "L1"):
            w_snip = L.load_ww(det, gps)[1]
            res_path = OUT / "checki" / f"{gen}_{det}.npz"
            done = dict(np.load(res_path)) if res_path.exists() else {}
            arr = done.get("mz2", np.full((CI_SERIES, CI_SNIPS, len(boxes), CI_TEMPLATES), np.nan))
            # per box: sigma per reference (depends only on the template and this detector's filter), and the
            # selected templates grouped by reference, so each reference correlation is computed once per snippet
            prep_b = []
            for l_max, tid, bx, sel in boxes:
                sig = [np.sqrt(2 * np.sum((np.abs(bx.h0[jr]) * w_snip) ** 2) / NS) for jr in range(len(bx.h0))]
                groups = []
                for jr in sorted(set(int(bx.ref_of[t_]) for t_ in sel)):
                    ti = [k for k, t_ in enumerate(sel) if int(bx.ref_of[t_]) == jr]
                    dco = torch.tensor((bx.gr["coeffs"][sel[ti]] - bx.gr["ref_coeffs"][jr]).astype(np.float32), device=dev)
                    groups.append((jr, ti, dco))
                prep_b.append((bx, sig, groups))
            for j in range(CI_SERIES):
                if np.isfinite(arr[j]).all():
                    continue
                subprocess.run([sys.executable, "-u", __file__, "ciseries", gen, det, str(j)], check=True)
                y = np.load(CI_DIR / f"{gen}_{det}_{j}.npy", mmap_mode="r")
                for si, i_c in enumerate(centres):
                    ysn, kappa = noise_spectrum(dict(kind="bg", i_c=int(i_c), n=0), y)
                    ysn_t = torch.tensor((ysn * kappa).astype(np.complex64), device=dev)
                    i_s = int(i_c) - snippet_start(int(i_c))
                    for bi, (bx, sig, groups) in enumerate(prep_b):
                        for jr, ti, dco in groups:
                            yt = eng.window(L.analytic_corr(torch, ysn_t, bx.h0c[jr]), i_s)
                            a2 = (eng.z(yt, dco, sign) * float(1.0 / (sig[jr] * kappa))).abs() ** 2
                            assert bool(torch.isfinite(a2).all()), "non-finite |z|^2 (check i)"
                            arr[j, si, bi, ti] = a2.mean(dim=1).cpu().numpy()
                (CI_DIR / f"{gen}_{det}_{j}.npy").unlink()
                (OUT / "checki").mkdir(parents=True, exist_ok=True)
                tmp = res_path.with_name(res_path.stem + ".tmp.npz")
                np.savez(tmp, mz2=arr, boxes=np.array([(l_, t_) for l_, t_, _, _ in boxes]))
                os.replace(tmp, res_path)
                if dev == "mps":
                    torch.mps.empty_cache()
                log(f"checki {gen} {det} series {j + 1}/{CI_SERIES} | footprint {L.peak_footprint_mb()} MB")


def checki_eval(a, b, tag):
    """C17 decision: pooled ratio R = mean(gen a) / mean(gen b) per (L_max, detector) cell, SE from a bootstrap over
    series (B = 2,000; each generator's series resampled independently). Pass: |R - 1| <= 3 SE in all 4 cells."""
    rng = np.random.default_rng([SEED, 18])
    out = dict(a=a, b=b, cells={})
    for det in ("H1", "L1"):
        A = np.load(OUT / "checki" / f"{a}_{det}.npz")["mz2"]; Bb = np.load(OUT / "checki" / f"{b}_{det}.npz")["mz2"]
        assert np.isfinite(A).all() and np.isfinite(Bb).all(), "incomplete series"
        for li, l_max in enumerate((128, 384)):
            sl = slice(3 * li, 3 * li + 3)
            sa, sb = A[:, :, sl, :].mean(axis=(1, 2, 3)), Bb[:, :, sl, :].mean(axis=(1, 2, 3))   # per-series means
            R = sa.mean() / sb.mean()
            boot = np.array([sa[rng.integers(0, len(sa), len(sa))].mean() / sb[rng.integers(0, len(sb), len(sb))].mean()
                             for _ in range(2000)])
            se = float(boot.std(ddof=1))
            per_t = (A[:, :, sl, :].mean(axis=(0, 1)) / Bb[:, :, sl, :].mean(axis=(0, 1))).ravel().tolist()
            out["cells"][f"L{l_max}_{det}"] = dict(R=float(R), se=se, within_3se=bool(abs(R - 1) <= 3 * se),
                                                   mean_a=float(sa.mean()), mean_b=float(sb.mean()),   # expect ~2 (chi^2_2)
                                                   per_template_ratios=per_t)
    out["all_within"] = all(c_["within_3se"] for c_ in out["cells"].values())
    out["max_se"] = max(c_["se"] for c_ in out["cells"].values())
    (OUT / "checki" / f"eval_{tag}.json").write_text(json.dumps(dict(out, env=L.stamp()), indent=1))
    log(f"checki eval {tag}: {json.dumps({k: (round(v['R'], 5), round(v['se'], 5)) for k, v in out['cells'].items()})} "
        f"all within 3 SE: {out['all_within']} | max SE {out['max_se']:.5f}")
    return out


def prepseries():
    """(3): noise series + PSD kernels for every (det, segment). If series_sha256.json exists (written before the
    K1 phase deleted the series), every series is checked against it and the step STOPS on any mismatch (the Bridge's
    condition, 2026-11-26): a regenerated series is used only if it is bit-identical to the one the passes used."""
    want = json.loads(SERIES_SHA.read_text()) if SERIES_SHA.exists() else None
    for det in ("H1", "L1"):
        for g in segments():
            L.load_ww(det, g)                   # runs coh2c_local prep in its own child if the series is missing
            if not (STATE / "kern" / f"kern_{det}_{g}.npz").exists():
                subprocess.run([sys.executable, "-u", __file__, "kernel", det, str(g)], check=True)
            if want is not None:
                for f in _series_files(det, g):
                    got = _sha(f)
                    assert got == want[f.name], f"{f.name}: sha256 {got} != recorded {want[f.name]}; STOP"
    log("noise series and PSD kernels ready" + (" (all sha256 match the record)" if want else ""))


def hashseries():
    """Record the sha256 of every noise-series file (series, snippet-grid filter, rows), before any deletion."""
    rec = {f.name: _sha(f) for det in ("H1", "L1") for g in segments() for f in _series_files(det, g)}
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(SERIES_SHA, rec)
    log(f"recorded sha256 of {len(rec)} series files -> {SERIES_SHA.name}")


def dropseries():
    """Before the K1 phase (disk, C10): delete the real noise series (ww_*.npy), keeping their small filter and rows
    files, ONLY after re-hashing each one against the committed record."""
    want = json.loads(SERIES_SHA.read_text())
    for det in ("H1", "L1"):
        for g in segments():
            f = _series_files(det, g)[0]
            assert _sha(f) == want[f.name], f"{f.name} changed since it was recorded; not deleting"
    for det in ("H1", "L1"):
        for g in segments():
            _series_files(det, g)[0].unlink()
    log("real noise series deleted (hashes verified first); regenerate with prepseries, which re-checks them")


def vseries():
    """PyCBC's v series per (det, segment), each in its own child (coh2t.psd_variation)."""
    for det in ("H1", "L1"):
        for g in segments():
            out = STATE / "v" / f"v_{det}_{g}.npz"
            if out.exists():
                continue
            subprocess.run([sys.executable, "-u", __file__, "vone", det, str(g)], check=True)
    log("v series ready")


def vone(det, gps):
    import coh2t
    pv = coh2t.psd_variation(det, int(gps))
    out = STATE / "v" / f"v_{det}_{gps}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez(tmp, values=np.asarray(pv.numpy(), np.float64), start=float(pv.start_time), delta_t=float(pv.delta_t))
    os.replace(tmp, out)


# ------------------------------------------------------------------ engine: one candidate, both detectors
def snippet_start(i_c):
    return int(np.clip(i_c - L.PRE * FS, 0, N - NS))


class Box:
    """Everything that is fixed for one (tid, l_max): grid, references on the device, per-template phase offsets."""

    def __init__(self, tid, l_max, dev, eng, sign, h0, gr=None):
        import torch
        self.t, self.dev, self.eng, self.sign = torch, dev, eng, sign
        gr = grid(tid, l_max) if gr is None else gr
        self.gr, self.n = gr, len(gr["coeffs"])
        self.ref_of = gr["ref_of"]
        self.h0 = h0                                                         # [n_ref, NS/2+1] complex64 (numpy)
        self.h0c = [torch.tensor(h, device=dev) for h in h0]
        self.groups = []
        for j in range(len(h0)):
            idx = np.where(self.ref_of == j)[0]
            dco = torch.tensor((gr["coeffs"][idx] - gr["ref_coeffs"][j]).astype(np.float32), device=dev)
            self.groups.append((idx, dco))


class Fine:
    """Band bins for the exact chi^2 route (gate E3), shared by every box."""

    def __init__(self, dev):
        import torch
        f = np.arange(NS // 2 + 1) * (FS / NS)
        self.kb = np.where((f >= L.F_LO - 12) & (f <= L.F_HI))[0]
        self.basis = torch.tensor(L.basis(f[self.kb]).astype(np.float32), device=dev)
        self.k_idx = torch.tensor(self.kb, device=dev, dtype=torch.int64)
        self.deltas = torch.arange(-DELTA, DELTA + 1, device=dev, dtype=torch.int64)
        # exact lag phases exp(2 pi i k delta / NS) for every band bin and every delta, built once (integer phase), as
        # real cos / sin matrices [nb, 83]: every band sum is then a fixed-shape matmul (MPS caches per shape; the
        # first version's per-band slices grew the footprint to 4.3 GB over two boxes)
        cs, sn = [], []
        for a in range(0, len(self.kb), 65536):
            ph = ((self.k_idx[a:a + 65536].unsqueeze(1) * self.deltas.unsqueeze(0)) % NS).to(torch.float32) * (2 * np.pi / NS)
            cs.append(torch.cos(ph)); sn.append(torch.sin(ph))
            del ph
        self.cosE, self.sinE = torch.cat(cs), torch.cat(sn)


def noise_spectrum(c, y):
    """The snippet's noise spectrum from a series (the real gated, double-whitened one, or a twin snippet already cut
    to 512 s). kappa, the float32 range scale, always comes from the noise."""
    if len(y) == NS:
        ysn = np.fft.rfft(np.asarray(y, dtype=np.float64))
    else:
        lo = snippet_start(c["i_c"])
        ysn = np.fft.rfft(np.asarray(y[lo: lo + NS], dtype=np.float64))
    kappa = 1.0 / np.median(np.abs(ysn[NS // 2048: NS // 8]))
    assert np.isfinite(ysn).all() and np.isfinite(kappa) and kappa > 0, "non-finite snippet spectrum"
    return ysn, kappa


def prepare(box, fine, ysn, kappa, w_snip, i_c):
    """One detector's snippet spectrum on the device, and per reference its sigma, equal-SNR band edges (p = 16;
    p = 8 and 4 are unions of them, asserted) and the fine-resolution product p0 for chi^2."""
    t = box.t
    lo = snippet_start(i_c)
    ysn_t = t.tensor((ysn * kappa).astype(np.complex64), device=box.dev)
    refs = []
    for j in range(len(box.h0)):
        pw = (np.abs(box.h0[j]) * w_snip) ** 2
        sig0 = np.sqrt(2 * pw.sum() / NS)
        cum = np.cumsum(pw[fine.kb]) / pw[fine.kb].sum()
        ke = {p: np.concatenate([[0], np.searchsorted(cum, np.arange(1, p) / p), [len(fine.kb)]]) for p in P_BINS}
        for p in P_BINS:                         # nested bands: every p-edge is a 16-edge
            assert np.array_equal(ke[p], ke[16][:: 16 // p]), (p, "band edges are not nested")
        refs.append(dict(sig0=sig0, edges16=[(int(a), int(b)) for a, b in zip(ke[16][:-1], ke[16][1:])],
                         p0=(ysn_t * box.h0c[j])[fine.k_idx]))
    return dict(lo=lo, i_s=i_c - lo, kappa=kappa, ysn_t=ysn_t, refs=refs)


def band_series(box, fine, r, tidx, n0):
    """Allen's 16 band pieces z_j at snippet samples n0 + delta, delta in +-10 ms (gate E3's exact route: heterodyned
    at full snippet resolution, integer lag phase; the delta phase is an exact integer too). -> [n_delta, 16], normalised."""
    t = box.t
    j = int(box.ref_of[tidx])
    rf = r["refs"][j]
    dco = t.tensor((box.gr["coeffs"][tidx] - box.gr["ref_coeffs"][j]).astype(np.float32), device=box.dev)
    ph = (dco @ fine.basis) * float(-box.sign)
    ph = ph + ((fine.k_idx * int(n0)) % NS).to(t.float32) * (2 * np.pi / NS)
    zt0 = rf["p0"] * t.complex(t.cos(ph), t.sin(ph)) * (2.0 / NS)
    if "member" not in rf:                                   # [16, nb] band membership, fixed shape
        mem = t.zeros(16, len(fine.kb), device=box.dev)
        for jj, (a, b) in enumerate(rf["edges16"]):
            mem[jj, a:b] = 1.0
        rf["member"] = mem
    wa, wb = rf["member"] * zt0.real.unsqueeze(0), rf["member"] * zt0.imag.unsqueeze(0)
    re = wa @ fine.cosE - wb @ fine.sinE                      # sum_k zt0_k exp(i 2 pi k delta / NS), per band
    im = wa @ fine.sinE + wb @ fine.cosE
    out = (re.cpu().numpy().astype(np.float64) + 1j * im.cpu().numpy().astype(np.float64)).T   # [n_delta, 16]
    assert np.isfinite(out).all(), "non-finite chi^2 band pieces: correctness gate"
    return out * (1.0 / (rf["sig0"] * r["kappa"]))


def chi2_of(z16):
    """[..., 16] band pieces -> rho and Allen chi^2 for p = 4, 8, 16 (p-bands are unions of 16-bands)."""
    z = z16.sum(axis=-1)
    out = {}
    for p in P_BINS:
        zp = z16.reshape(z16.shape[:-1] + (p, 16 // p)).sum(axis=-1)
        out[p] = p * np.sum(np.abs(zp - z[..., None] / p) ** 2, axis=-1)
    return np.abs(z), out


def score(box, fine, rH, rL, plain_only=False):
    """Registration §3, H-anchored: for each of H1's K loudest templates (each template's own best lag), chi^2 at that
    H1 point, and the SAME template's L1 rho and chi^2 at every lag within +-10 ms of it (v and newSNR are applied
    later by PyCBC, which then takes the max over delta). Also the plain-coherent ablation, same form, all templates."""
    t, eng = box.t, box.eng
    assert rH["i_s"] == rL["i_s"]
    n_lag = 2 * eng.half + 1
    tpeak, tlag = np.zeros(box.n), np.zeros(box.n, np.int64)
    zH_pk, zL_at = np.zeros(box.n, np.complex128), np.zeros(box.n, np.complex128)
    plain = -np.inf
    for j, (idx, dco) in enumerate(box.groups):
        if len(idx) == 0:
            continue
        yt = {}
        for det, r in (("H1", rH), ("L1", rL)):
            x0 = L.analytic_corr(t, r["ysn_t"], box.h0c[j])
            yt[det] = eng.window(x0, r["i_s"])
            del x0
        for s in range(0, len(idx), BATCH):
            part = idx[s:s + BATCH]
            # multiply by the reciprocal: on MPS, complex / (a float ~4e-26) returns NaN (found by the 2026-11-26 pilot)
            zH = eng.z(yt["H1"], dco[s:s + BATCH], box.sign) * float(1.0 / (rH["refs"][j]["sig0"] * rH["kappa"]))
            zL = eng.z(yt["L1"], dco[s:s + BATCH], box.sign) * float(1.0 / (rL["refs"][j]["sig0"] * rL["kappa"]))
            aH, aL = zH.abs() ** 2, zL.abs() ** 2
            assert bool(t.isfinite(aH).all()) and bool(t.isfinite(aL).all()), "non-finite |z|^2: correctness gate"
            lmax = t.nn.functional.max_pool1d(aL.unsqueeze(1), 2 * DELTA + 1, stride=1, padding=DELTA).squeeze(1)
            plain = max(plain, float((aH + lmax).max()))
            pk, am = aH.max(dim=1)
            rows = t.arange(len(part), device=box.dev)
            tpeak[part], tlag[part] = pk.cpu().numpy(), am.cpu().numpy()
            zH_pk[part] = zH[rows, am].cpu().numpy()
            zL_at[part] = zL[rows, am].cpu().numpy()
            del zH, zL, aH, aL, lmax
        del yt
    assert np.isfinite(tpeak).all() and np.isfinite(plain), "non-finite |z|^2: correctness gate, stopping"
    if plain_only:
        return dict(tpeak_max=float(tpeak.max()), plain_net=plain)
    top = np.argsort(-tpeak)[:K_PEAKS]
    out = dict(tmpl=top, lag=tlag[top], H_rho=np.zeros(K_PEAKS), H_chi2=np.zeros((K_PEAKS, 3)),
               L_rho=np.zeros((K_PEAKS, 2 * DELTA + 1)), L_chi2=np.zeros((K_PEAKS, 3, 2 * DELTA + 1)),
               closure=np.zeros((K_PEAKS, 2)))
    for a, tidx in enumerate(top):
        n0 = rH["i_s"] - eng.half + int(tlag[tidx])
        zh16 = band_series(box, fine, rH, tidx, n0)[DELTA]                    # delta = 0 only
        zl16 = band_series(box, fine, rL, tidx, n0)                           # every delta
        rho_h, c_h = chi2_of(zh16)
        rho_l, c_l = chi2_of(zl16)
        out["H_rho"][a], out["H_chi2"][a] = rho_h, [c_h[p] for p in P_BINS]
        out["L_rho"][a], out["L_chi2"][a] = rho_l, np.stack([c_l[p] for p in P_BINS])
        out["closure"][a] = (abs(zh16.sum() - zH_pk[tidx]) / max(abs(zH_pk[tidx]), 1.0),
                             abs(zl16[DELTA].sum() - zL_at[tidx]) / max(abs(zL_at[tidx]), 1.0))
    out.update(tpeak_max=float(tpeak.max()), plain_net=plain, n_lag=n_lag)
    return out


_TABLE = None


def rho_opt_trunc(box, w_snip, ys, row):
    """C15: the optimal |z| of the TRUE template truncated as the engine's templates are: the injection's own phase
    (lalsimulation coefficients) on its nearest box reference's amplitude and start frequency (D2), evaluated on the
    signal-only snippet spectrum ys, maximised over all lags; numpy on the CPU (complex128)."""
    import injections as I
    mc_t = float(I.chirp_mass(row["mass1"], row["mass2"]))
    q_t = float(min(row["mass1"], row["mass2"]) / max(row["mass1"], row["mass2"]))
    c_t = L.coeffs(mc_t, q_t)
    refs = [(r_[0], r_[1], r_[2], c_) for r_, c_ in zip(box.gr["ref_mcq"], box.gr["ref_coeffs"])]
    j = L.assign(c_t, refs)[0]
    h0 = np.conj(box.h0[j].astype(np.complex128))
    f = np.arange(NS // 2 + 1) * (FS / NS)
    nz = np.abs(h0) > 0
    dpsi = np.zeros(len(f))
    dpsi[nz] = (c_t - box.gr["ref_coeffs"][j]) @ L.basis(f[nz])
    H_t = h0 * np.exp(1j * box.sign * dpsi)
    sig0 = np.sqrt(2 * np.sum((np.abs(h0) * w_snip) ** 2) / NS)
    P = np.zeros(NS, complex)
    P[: NS // 2 + 1] = ys * np.conj(H_t)
    z = 2 * np.fft.ifft(P) / sig0
    v = float(np.abs(z).max())
    assert np.isfinite(v), "non-finite rho_opt"
    return v


def candidate(box, fine, series, c, extra=None, target=None, twin=None):
    """Score one candidate. extra: per-detector additive snippets (injected signal, or a unit-amplitude decoy).
    fg also records the signal-only plain network value (the decoy calibration's yardstick, C6); dec_sg / dec_train
    are scaled so that their own decoy-only plain network value equals target (C6); dec_drift is not scaled."""
    segs = segments()
    spec, extra_spec = {}, {}
    for det, si in (("H1", c["i"]), ("L1", c["iL"])):
        y, w_snip, _ = series(det, segs[si])
        spec[det] = noise_spectrum(c, y if twin is None else twin[det]) + (w_snip,)
        if extra is not None:
            extra_spec[det] = np.fft.rfft(extra[det].astype(np.float64))

    def run_on(fn, plain_only=False):
        r = {det: prepare(box, fine, fn(det), spec[det][1], spec[det][2], c["i_c"]) for det in ("H1", "L1")}
        return r, score(box, fine, r["H1"], r["L1"], plain_only)
    o_extra, scale = {}, 1.0
    if c["kind"] in ("fg", "wrong"):                       # C15: P1's rho_opt, per detector and network
        global _TABLE
        if _TABLE is None:
            _TABLE = L._table()
        ro = {det: rho_opt_trunc(box, spec[det][2], extra_spec[det], _TABLE[c["inj"]]) for det in ("H1", "L1")}
        o_extra.update(rho_opt_trunc_H1=ro["H1"], rho_opt_trunc_L1=ro["L1"],
                       rho_opt_trunc_net=float(np.hypot(ro["H1"], ro["L1"])))
    if c["kind"] in ("fg", "dec_sg", "dec_train"):
        _, o_alone = run_on(lambda det: extra_spec[det], plain_only=True)
        o_extra["extra_only_plain"] = o_alone["plain_net"]
        if c["kind"] != "fg":
            assert target is not None, "decoys need the calibration target (run 'calib' after fg)"
            assert np.isfinite(target) and np.isfinite(o_alone["plain_net"]) and o_alone["plain_net"] > 0
            scale = float(np.sqrt(target / o_alone["plain_net"]))
    o_extra["extra_scale"] = scale
    r, o = run_on(lambda det: spec[det][0] + scale * extra_spec[det] if extra is not None else spec[det][0])
    lag0 = r["H1"]["i_s"] - box.eng.half
    o["H_gps"] = segs[c["i"]] + (r["H1"]["lo"] + lag0 + o["lag"]) / FS                 # H1 peak time
    o["L_gps0"] = segs[c["iL"]] + (r["L1"]["lo"] + lag0 + o["lag"] - DELTA) / FS       # L1 time at delta = -10 ms
    o.update(o_extra)
    return o


def peaks_one(box, r):
    """Single-detector |z|^2 peak of every grid template (max over the +-1-s lags) and its lag."""
    t, eng = box.t, box.eng
    tpeak, tlag = np.zeros(box.n), np.zeros(box.n, np.int64)
    for j, (idx, dco) in enumerate(box.groups):
        if len(idx) == 0:
            continue
        x0 = L.analytic_corr(t, r["ysn_t"], box.h0c[j])
        yt = eng.window(x0, r["i_s"])
        del x0
        for s_ in range(0, len(idx), BATCH):
            part = idx[s_:s_ + BATCH]
            a2 = (eng.z(yt, dco[s_:s_ + BATCH], box.sign) * float(1.0 / (r["refs"][j]["sig0"] * r["kappa"]))).abs() ** 2
            pk, am = a2.max(dim=1)
            tpeak[part], tlag[part] = pk.cpu().numpy(), am.cpu().numpy()
        del yt
    assert np.isfinite(tpeak).all(), "non-finite |z|^2: correctness gate, stopping"
    return tpeak, tlag


def allen_signal(box, w_snip, coef, rho, n_inj, sign, j=None):
    """C11 injection on the snippet grid, built with the engine's verified phase model: reference j's TaylorF2
    amplitude and band, the signal's own phase (lalsimulation coefficients), merger at snippet sample n_inj, double-
    whitened like the data (x w^2). Amplitude A = rho / sigma_ref, so the exact template's noise-free |z| is rho.
    Sharing the reference's amplitude and start frequency means signal and templates share band edges, so Allen's
    tighter bound (his eqs. 75 / 100) also applies; it is reported for information."""
    if j is None:
        refs = [(r_[0], r_[1], r_[2], c_) for r_, c_ in zip(box.gr["ref_mcq"], box.gr["ref_coeffs"])]
        j = L.assign(coef, refs)[0]
    h0 = np.conj(box.h0[j].astype(np.complex128))
    f = np.arange(NS // 2 + 1) * (FS / NS)
    nz = np.abs(h0) > 0
    dpsi = np.zeros(len(f))
    dpsi[nz] = (coef - box.gr["ref_coeffs"][j]) @ L.basis(f[nz])
    sig0 = np.sqrt(2 * np.sum((np.abs(h0) * w_snip) ** 2) / NS)
    k = np.arange(NS // 2 + 1)
    return (rho / sig0) * h0 * np.exp(1j * sign * dpsi) * w_snip ** 2 * np.exp(-2j * np.pi * ((k * n_inj) % NS) / NS)


def allen_candidate(box, fine, series, c, twin, l_max):
    """C11, per detector: inject (off-grid / exactly on a grid template / just outside the box), find the best grid
    template noise-free (match m), then chi^2 at that template's own peak in the noisy data."""
    segs = segments()
    out = {}
    mc_star = float(box.gr["mc_star"])
    qr = L.q_range(mc_star)
    for det, si in (("H1", c["i"]), ("L1", c["iL"])):
        _, w_snip, _ = series(det, segs[si])
        ysn_n, kappa = noise_spectrum(c, twin[det])
        par = c["inj_par"][det]
        lo = snippet_start(c["i_c"])
        n_inj = c["i_c"] - lo + int(round(par["dt"] * FS))
        q = qr[0] + par["u"] * (qr[1] - qr[0])
        j_fix, delta, tsel = None, None, None
        if c["kind"] == "allen_on":
            tsel = min(int(par["u"] * box.n), box.n - 1)
            coef, j_fix = box.gr["coeffs"][tsel], int(box.ref_of[tsel])
        elif c["kind"] == "allen_off":
            coef = L.coeffs(mc_star * np.exp(par["v"] * HALF_WIDTH), q)

        def noise_free(cf, jf=None):
            ys = allen_signal(box, w_snip, cf, par["rho"], n_inj, box.sign, jf)
            tp, _ = peaks_one(box, prepare(box, fine, ys, kappa, w_snip, c["i_c"]))
            return ys, tp
        if c["kind"] == "allen_fail":             # the first delta (0.05 % steps) whose best-grid match is <= 0.9
            for step in range(1, 200):
                delta = 0.0005 * step
                coef = L.coeffs(mc_star * np.exp(HALF_WIDTH + delta), q)
                ys, tp = noise_free(coef)
                if np.sqrt(tp.max()) / par["rho"] <= 0.9:
                    break
        else:
            ys, tp = noise_free(coef, j_fix)
        best = int(np.argmax(tp))
        m = float(np.sqrt(tp.max()) / par["rho"])
        assert np.isfinite(m), "non-finite match"
        r = prepare(box, fine, ysn_n + ys, kappa, w_snip, c["i_c"])
        _, tln = peaks_one(box, r)
        n0 = r["i_s"] - box.eng.half + int(tln[best])
        rho_obs, ch = chi2_of(band_series(box, fine, r, best, n0)[DELTA])
        rho = par["rho"]
        rec = dict(rho=rho, m=m, rho_obs=float(rho_obs), best_tmpl=best, chi2=[float(ch[p_]) for p_ in P_BINS],
                   bound=[2 * (p_ - 1) + rho ** 2 * (1 - m ** 2) for p_ in P_BINS],
                   bound_nominal=[2 * (p_ - 1) + rho ** 2 * (1 - 0.97 ** 2) for p_ in P_BINS],
                   bound_allen_tight=[2 * (p_ - 1) + (p_ - 1) * (1 / max(m, 1e-9) - 1) ** 2 * (rho * m) ** 2
                                      for p_ in P_BINS],
                   delta=-1.0 if delta is None else delta, on_tmpl=-1 if tsel is None else tsel,
                   on_found=bool(tsel is None or best == tsel))
        out[det] = rec
    return {f"{det}_{k}": v for det, rec in out.items() for k, v in rec.items()}


META = ("n", "kind", "k", "i", "iL", "i_c", "y1", "inj", "tid_y1", "decoy", "real", "inj_par")


# ------------------------------------------------------------------ the box-major loop
def run(l_max, kinds="bg,fg", n_box=None, real=None):
    l_max = int(l_max)
    kinds = [k for k in kinds.split(",")]
    for k in kinds:
        assert k in KINDS, f"unknown kind {k!r}"
    real = None if real is None else int(real)
    assert ("k1mc" in kinds) == (real is not None) and (real is None or kinds == ["k1mc"]), \
        "k1mc runs alone, one realisation at a time: run <l_max> k1mc - <r>"
    if real is not None:
        k1gen_all(real)
    target = None
    if any(k in ("dec_sg", "dec_train") for k in kinds):
        cal = OUT / f"L{l_max}" / "decoy_calibration.json"
        assert cal.exists(), "decoy calibration missing: run fg, then 'calib', before dec_sg / dec_train"
        target = json.loads(cal.read_text())["target_plain"]
    pl = load_plan()
    dev = L.pick_dev()
    sign = json.loads((L.OUT / "E1_phase_model.json").read_text())["sign"]
    eng, fine = L.Engine(dev, 4.0, 1), Fine(dev)
    cache = {}

    def series(det, gps):
        if (det, gps) not in cache:
            if real is None:
                cache[(det, gps)] = L.load_ww(det, gps)
            else:                                      # K1-(A): pycbc noise with prep's own snippet-grid filter
                tag_ = L.ww_tag(det, gps)              # (the real noise series need not be on disk in this phase)
                cache[(det, gps)] = (np.load(k1_path(det, gps, real), mmap_mode="r"),
                                     np.load(L.STATE / f"wsnip_{tag_}.npy"), {})
        return cache[(det, gps)]
    done_dir = OUT / f"L{l_max}"
    tag = "" if real is None else f"_r{real}"
    todo = [tid for tid in pl["boxes"] if any(not (done_dir / f"box_{tid:04d}_{k}{tag}.npz").exists() for k in kinds)]
    if n_box is not None:
        todo = todo[: int(n_box)]
    by_box = {}
    for c in pl["candidates"]:
        if c["valid"] and c["kind"] in kinds and (real is None or c.get("real") == real):
            by_box.setdefault(c["tid"], []).append(c)
    todo = [tid for tid in todo if tid in by_box]
    if os.environ.get("COH2T_BOXES"):                    # pilot: an explicit list of box tids
        want = [int(v) for v in os.environ["COH2T_BOXES"].split(",")]
        todo = [tid for tid in want if tid in todo]
    log(f"run L{l_max} {kinds}: {len(todo)} boxes to do on {dev}")
    t_start = time.time()
    overlap = os.environ.get("COH2T_OVERLAP") == "1"       # child for chunk s+1 runs while chunk s is scored

    chunk_n = 16 if kinds == ["k1mc"] else CHUNK          # K1 boxes need only references: fewer, larger children

    def launch(i):
        part = todo[i:i + chunk_n]
        return subprocess.Popen([sys.executable, "-u", __file__, "chunk", str(l_max), ",".join(map(str, part)),
                                 ",".join(kinds)]) if part else None
    nxt = launch(0)
    for s in range(0, len(todo), chunk_n):
        part = todo[s:s + chunk_n]
        assert nxt.wait() == 0, "chunk child failed"
        nxt = launch(s + chunk_n) if overlap else None
        for tid in part:
            t0 = time.time()
            refs_path = STATE / "chunk" / f"refs_L{l_max}_tid{tid:04d}.npy"
            box = Box(tid, l_max, dev, eng, sign, np.load(refs_path))
            for kind in kinds:
                out = done_dir / f"box_{tid:04d}_{kind}{tag}.npz"
                if out.exists():
                    continue
                cs = [c for c in by_box.get(tid, []) if c["kind"] == kind]
                res = []
                for c in cs:
                    files = extra_files(c) if kind != "null" else {}
                    loaded = {det: np.load(f) for det, f in files.items()} or None
                    if kind in ALLEN:
                        res.append(allen_candidate(box, fine, series, c, loaded, l_max))
                    elif kind == "twin":
                        res.append(candidate(box, fine, series, c, None, target, twin=loaded))
                    else:
                        res.append(candidate(box, fine, series, c, loaded, target))
                meta = dict(tid=tid, l_max=l_max, kind=kind, mc_star=float(box.gr["mc_star"]), n_tmpl=box.n,
                            n_refs=len(box.h0c), worst_dtau=float(box.gr["worst_dtau"]), ff_min=float(box.gr["ff_min"]),
                            candidates=[{k: c.get(k) for k in META} for c in cs], seconds=round(time.time() - t0, 2),
                            peak_footprint_mb=L.peak_footprint_mb(), model="Claude Opus 5.5", status="UNVERIFIED")
                arrs = {k: np.array([r[k] for r in res]) for k in res[0]} if res else {}
                if "closure" in arrs:
                    meta["closure_max"] = float(np.max(arrs["closure"]))
                for k in ("H_rho", "H_chi2", "L_rho", "L_chi2", "closure"):
                    if k in arrs:
                        arrs[k] = arrs[k].astype(np.float32)
                done_dir.mkdir(parents=True, exist_ok=True)
                tmp = out.with_name(out.stem + ".tmp.npz")
                np.savez_compressed(tmp, meta=json.dumps(meta), **arrs)
                os.replace(tmp, out)
                # C12: the box is written first (on the record), then a closure breach stops the run here
                assert meta.get("closure_max", 0.0) <= CLOSURE_MAX, \
                    f"closure {meta['closure_max']:.2e} > {CLOSURE_MAX:g} in box {tid} {kind}: correctness gate (C12)"
            for pth in [refs_path] + [f for c in by_box.get(tid, []) for f in extra_files(c).values()]:
                pth.unlink(missing_ok=True)
            del box
            mem = ""
            if dev == "mps":
                import torch
                pre = (torch.mps.current_allocated_memory() >> 20, torch.mps.driver_allocated_memory() >> 20)
                torch.mps.empty_cache()
                mem = (f" | mps cur/drv {pre[0]}/{pre[1]} MB -> after empty_cache {torch.mps.current_allocated_memory() >> 20}/"
                       f"{torch.mps.driver_allocated_memory() >> 20} MB | footprint now {L.footprint_now_mb()} MB")
            log(f"box {tid}{mem} ({s + part.index(tid) + 1}/{len(todo)}) {time.time() - t0:.1f} s | footprint "
                f"{L.peak_footprint_mb()} MB | {(time.time() - t_start) / 3600:.2f} h")
        if not overlap and s + chunk_n < len(todo):
            nxt = launch(s + chunk_n)


def calib(l_max):
    """C6: the decoy amplitude target = the median, over valid caught injections, of their SIGNAL-ONLY plain network
    value (recorded by the fg pass). Written once; refuses to overwrite, so it cannot be re-tuned after decoys ran."""
    pl = load_plan()
    d = OUT / f"L{int(l_max)}"
    out = d / "decoy_calibration.json"
    assert not out.exists(), "calibration already fixed"
    want = sorted({c["tid"] for c in pl["candidates"] if c["kind"] == "fg" and c["valid"]})
    vals = []
    for tid in want:
        z = np.load(d / f"box_{tid:04d}_fg.npz")
        vals += list(np.atleast_1d(z["extra_only_plain"]))
    assert len(vals) == sum(c["kind"] == "fg" and c["valid"] for c in pl["candidates"]), "fg pass incomplete"
    assert np.isfinite(vals).all() and min(vals) > 0, "non-finite or non-positive signal-only plain value"
    atomic_json(out, dict(target_plain=float(np.median(vals)), n=len(vals), quantiles=[float(q) for q in
                np.quantile(vals, [0.1, 0.25, 0.5, 0.75, 0.9])], rule="C6", model="Claude Opus 5.5", env=L.stamp()))
    log(f"decoy calibration L{l_max}: target plain {np.median(vals):.1f} from {len(vals)} caught injections")


def status(l_max):
    pl = load_plan()
    d = OUT / f"L{int(l_max)}"
    for k in KINDS:
        n = sum((d / f"box_{tid:04d}_{k}.npz").exists() for tid in pl["boxes"])
        print(f"L{l_max} {k}: {n}/{len(pl['boxes'])} boxes")
    print(json.dumps(pl["summary"]))


# ------------------------------------------------------------------ golden: the driver path against the verification
def golden():
    """One box of the verification (H1 noise, segment 1249001472, Mc* 0.70, L_max 128, pooled half-width 2.15 %), by
    the driver's own engine path with the same H1 series in both detector slots. Bars:
      * box max of |z_H|^2 = the same-device verification box's box_max_local within 1e-5; same template count;
      * band pieces at delta = 0 = coh2c_local.chi2_fine (gate E3's function) within 1e-5, for p = 4, 8, 16;
      * closure (sum of bands vs the engine's z) <= 1e-4 at every top point, both slots;
      * identical slots: L at delta = 0 equals H; plain network value >= 2 x the box max;
      * ALL FIELDS against the pre-rewrite CPU golden (results/coh2t/driver_golden_cpu_27d3bf9.json): the chosen
        top templates and lags identical; H_rho and H_chi2 within 1e-5 relative (the Bridge, 2026-11-26);
      * on MPS, also against the CPU golden of THIS code (driver_golden_cpu.npz): templates and lags identical, every
        band piece (H anchors and the whole L window) within 1e-5 of max(|z|, 1), box max within 1e-5.
    Writes driver_golden_<dev>.json + .npz (the band pieces), and their sha256 in the json."""
    import torch
    dev_ = L.pick_dev()
    ref = json.loads((B / "results" / "coh2c_verify" / f"box_H1_1249001472_noise_mc0.700_L128_{dev_}.json").read_text())
    old = json.loads((OUT / "driver_golden_cpu_27d3bf9.json").read_text())
    st = json.loads((B / "state" / "realbg" / "caught.json").read_text())
    gps = segments()[0]
    i_c = [b for b in st["candidates"]["0_0"] if (L.PRE + 1) * FS < b * 102 < 4096 * FS - 65 * FS][0] * 102
    dev = dev_
    sign = json.loads((L.OUT / "E1_phase_model.json").read_text())["sign"]
    from coh2c_verify import grid_points
    hw = 0.021510884647971934
    pts, _ = grid_points(0.70, hw, 128)
    refs, _ = L.reference_lattice(0.70, hw, 128, L.dtau_max(4.0))
    cs = np.array([L.coeffs(mc, q) for mc, q in pts])
    gr = dict(coeffs=cs, ref_coeffs=np.array([r[3] for r in refs]), ref_of=np.array([L.assign(c, refs)[0] for c in cs]))
    h0 = np.stack([np.conj(L.fd_template(mc, q, f_lo)).astype(np.complex64) for mc, q, f_lo, _ in refs])
    eng, fine = L.Engine(dev, 4.0, 1), Fine(dev)
    box = Box(None, 128, dev, eng, sign, h0, gr=gr)
    y, w_snip, _ = L.load_ww("H1", gps)
    ysn, kappa = noise_spectrum(dict(kind="bg", i_c=i_c, n=0), y)
    r = prepare(box, fine, ysn, kappa, w_snip, i_c)
    o = score(box, fine, r, r)
    rel = abs(o["tpeak_max"] / ref["box_max_local"] - 1)
    z16_H = np.stack([band_series(box, fine, r, int(tx), r["i_s"] - eng.half + int(lg))[DELTA]
                      for tx, lg in zip(o["tmpl"], o["lag"])])
    z16_L0 = band_series(box, fine, r, int(o["tmpl"][0]), r["i_s"] - eng.half + int(o["lag"][0]))
    tidx, n0 = int(o["tmpl"][0]), r["i_s"] - eng.half + int(o["lag"][0])
    j = int(gr["ref_of"][tidx])
    dco = torch.tensor((cs[tidx] - gr["ref_coeffs"][j]).astype(np.float32), device=dev).unsqueeze(0)
    e3 = []
    for p in P_BINS:
        ed = r["refs"][j]["edges16"]
        edp = [(ed[k][0], ed[k + 16 // p - 1][1]) for k in range(0, 16, 16 // p)]
        zj = L.chi2_fine(torch, r["refs"][j]["p0"], dco, sign, torch.tensor([n0], device=dev), edp, fine.basis,
                         fine.k_idx)[0].cpu().numpy().astype(np.complex128) / (r["refs"][j]["sig0"] * r["kappa"])
        e3.append(float(np.abs(z16_H[0].reshape(p, 16 // p).sum(axis=1) - zj).max() / max(abs(zj.sum()), 1.0)))
    assert all(np.isfinite(e3)) and np.isfinite(o["closure"]).all() and np.isfinite(rel), "non-finite golden field"
    e3_err, clos = max(e3), float(o["closure"].max())
    same_hl = float(np.abs(o["L_rho"][:, DELTA] - o["H_rho"]).max() / o["H_rho"].max())
    old_tmpl_same = list(map(int, o["tmpl"])) == old["top"]["tmpl"] and list(map(int, o["lag"])) == old["top"]["lag"]
    old_rho = float(np.max(np.abs(o["H_rho"] / np.array(old["top"]["H_rho"]) - 1)))
    old_chi = float(np.max(np.abs(o["H_chi2"] / np.array(old["top"]["H_chi2_p4_p8_p16"]) - 1)))
    checks = dict(box_max_rel=rel, n_templates_match=box.n == ref["n_templates"], band_vs_chi2_fine_max=e3_err,
                  closure_max=clos, identical_slot_L_vs_H=same_hl, plain_ge_2x=bool(o["plain_net"] >= 2 * o["tpeak_max"] * (1 - 1e-6)),
                  vs_pre_rewrite_cpu=dict(tmpl_lag_identical=old_tmpl_same, H_rho_rel_max=old_rho, H_chi2_rel_max=old_chi))
    ok = (rel <= 1e-5 and checks["n_templates_match"] and e3_err <= 1e-5 and clos <= 1e-4 and same_hl <= 1e-6
          and checks["plain_ge_2x"] and old_tmpl_same and old_rho <= 1e-5 and old_chi <= 1e-5)
    OUT.mkdir(parents=True, exist_ok=True)
    npz = OUT / f"driver_golden_{dev}.npz"
    np.savez(npz, tmpl=o["tmpl"], lag=o["lag"], z16_H=z16_H, z16_L0=z16_L0, tpeak_max=o["tpeak_max"])
    if dev == "mps":
        c = np.load(OUT / "driver_golden_cpu.npz")
        scale = np.maximum(np.abs(c["z16_H"].sum(axis=1)), 1.0)[:, None]
        xd = dict(tmpl_lag_identical=bool(np.array_equal(c["tmpl"], o["tmpl"]) and np.array_equal(c["lag"], o["lag"])),
                  z16_H_max=float(np.max(np.abs(z16_H - c["z16_H"]) / scale)),
                  z16_L0_max=float(np.max(np.abs(z16_L0 - c["z16_L0"])) / max(abs(c["z16_L0"].sum(axis=1)).max(), 1.0)),
                  box_max_rel=float(abs(o["tpeak_max"] / float(c["tpeak_max"]) - 1)))
        checks["vs_cpu_same_code"] = xd
        ok = ok and xd["tmpl_lag_identical"] and xd["z16_H_max"] <= 1e-5 and xd["z16_L0_max"] <= 1e-5 and xd["box_max_rel"] <= 1e-5
    res = dict(gate="driver golden", device=dev, n_templates=box.n, box_max_driver=o["tpeak_max"],
               box_max_verify=ref["box_max_local"], checks=checks, plain_net=o["plain_net"],
               top=dict(tmpl=o["tmpl"].tolist(), lag=o["lag"].tolist(), H_rho=o["H_rho"].tolist(),
                        H_chi2_p4_p8_p16=o["H_chi2"].tolist()), npz=npz.name, npz_sha256=_sha(npz),
               passed=bool(ok), peak_footprint_mb=L.peak_footprint_mb(), model="Claude Opus 5.5", env=L.stamp())
    (OUT / f"driver_golden_{dev}.json").write_text(json.dumps(res, indent=1))
    log(f"driver golden ({dev}) {'PASS' if ok else 'FAIL'}: {json.dumps(checks)} | footprint {L.peak_footprint_mb()} MB")


if __name__ == "__main__":
    a = sys.argv
    cmd = a[1]
    if cmd == "chunk":
        chunk(int(a[2]), [int(v) for v in a[3].split(",")], tuple(a[4].split(",")) if len(a) > 4 else KINDS)
    elif cmd == "vone":
        vone(a[2], a[3])
    elif cmd == "kernel":
        kernel(a[2], a[3])
    elif cmd == "k1gen":
        k1gen(a[2], a[3], a[4])
    elif cmd == "ciseries":
        ci_series(a[2], a[3], a[4])
    elif cmd == "checki_eval":
        checki_eval(a[2], a[3], a[4])
    elif cmd in ("k2gates", "k2prep"):
        {"k2gates": k2gates, "k2prep": k2prep}[cmd](a[2], a[3], a[4])
    else:
        dict(plan=plan, golden=golden, prepseries=prepseries, vseries=vseries, run=run, calib=calib,
             twinexact=twinexact, k1precheck=k1precheck, k1perfect=k1perfect, k1diag=k1diag, k2select=k2select, k2run=k2run, n1=n1, checki=checki, hashseries=hashseries, dropseries=dropseries,
             status=status)[cmd](*[None if v == "-" else v for v in a[2:]])
