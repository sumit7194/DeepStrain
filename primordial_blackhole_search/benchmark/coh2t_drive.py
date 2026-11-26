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
KINDS = ("bg", "fg", "twin", "null", "wrong", "dec_sg", "dec_train", "dec_drift", "k1mc")
DECOYS = ("dec_sg", "dec_train", "dec_drift")
N_DECOY = 200                      # registration §5 F2: 200 per family
SEED = 202611261                   # this build's own seed (twin draws, decoy choice and parameters); NOT the fresh seed
WRONG_MC = 1.05                    # F1(b): box offset to chirp mass +5 %
N_MC, N_REAL, SLOT_S = 100_000, 8, 4.0    # C10: K1-(A) maxima, independent pycbc realisations, slot length
K1_SEED0 = 7_000_000               # C10: lal ranlux seed = K1_SEED0 + 1000 * realisation + 2 * segment + detector
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
    assert worst <= L.dtau_max(4.0), (tid, l_max, worst)            # V3, per box
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
    if c["kind"] == "twin":
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
        x, g, psd, w = seg_state(det, gps, need_strain=c["kind"] != "twin")
        if c["kind"] == "twin":
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
        worst = max(abs(v - 1) for v in bands)
        ok &= worst <= 0.015
        rows.append(dict(det=det, segment=gps, seed=seed, band_means=bands, worst_dev=worst))
        log(f"k1precheck {det}: worst band deviation {worst:.4f}")
    res = dict(check="C10 (1) pycbc noise reproduces the target PSD", bar=0.015, rows=rows, passed=bool(ok),
               peak_footprint_mb=L.peak_footprint_mb(), model="Claude Opus 5.5", env=L.stamp())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "k1precheck.json").write_text(json.dumps(res, indent=1))
    log(f"C10 (1) PSD pre-check {'PASS' if ok else 'FAIL'} | footprint {L.peak_footprint_mb()} MB")


def prepseries():
    for det in ("H1", "L1"):
        for g in segments():
            L.load_ww(det, g)                   # runs coh2c_local prep in its own child if the series is missing
            if not (STATE / "kern" / f"kern_{det}_{g}.npz").exists():
                subprocess.run([sys.executable, "-u", __file__, "kernel", det, str(g)], check=True)
    log("noise series and PSD kernels ready")


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


def noise_spectrum(c, y):
    """The snippet's noise spectrum from a series (the real gated, double-whitened one, or a twin snippet already cut
    to 512 s). kappa, the float32 range scale, always comes from the noise."""
    if len(y) == NS:
        ysn = np.fft.rfft(np.asarray(y, dtype=np.float64))
    else:
        lo = snippet_start(c["i_c"])
        ysn = np.fft.rfft(np.asarray(y[lo: lo + NS], dtype=np.float64))
    return ysn, 1.0 / np.median(np.abs(ysn[NS // 2048: NS // 8]))


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
    out = []
    for a, b in rf["edges16"]:
        e = ((fine.k_idx[a:b].unsqueeze(1) * fine.deltas.unsqueeze(0)) % NS).to(t.float32) * (2 * np.pi / NS)
        out.append(zt0[a:b] @ t.complex(t.cos(e), t.sin(e)))
    return (t.stack(out, dim=1).cpu().numpy().astype(np.complex128)) / (rf["sig0"] * r["kappa"])


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
    plain = -1.0
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
            zH = eng.z(yt["H1"], dco[s:s + BATCH], box.sign) / float(rH["refs"][j]["sig0"] * rH["kappa"])
            zL = eng.z(yt["L1"], dco[s:s + BATCH], box.sign) / float(rL["refs"][j]["sig0"] * rL["kappa"])
            aH, aL = zH.abs() ** 2, zL.abs() ** 2
            lmax = t.nn.functional.max_pool1d(aL.unsqueeze(1), 2 * DELTA + 1, stride=1, padding=DELTA).squeeze(1)
            plain = max(plain, float((aH + lmax).max()))
            pk, am = aH.max(dim=1)
            rows = t.arange(len(part), device=box.dev)
            tpeak[part], tlag[part] = pk.cpu().numpy(), am.cpu().numpy()
            zH_pk[part] = zH[rows, am].cpu().numpy()
            zL_at[part] = zL[rows, am].cpu().numpy()
            del zH, zL, aH, aL, lmax
        del yt
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
    if c["kind"] in ("fg", "dec_sg", "dec_train"):
        _, o_alone = run_on(lambda det: extra_spec[det], plain_only=True)
        o_extra["extra_only_plain"] = o_alone["plain_net"]
        if c["kind"] != "fg":
            assert target is not None, "decoys need the calibration target (run 'calib' after fg)"
            scale = float(np.sqrt(target / o_alone["plain_net"]))
    o_extra["extra_scale"] = scale
    r, o = run_on(lambda det: spec[det][0] + scale * extra_spec[det] if extra is not None else spec[det][0])
    lag0 = r["H1"]["i_s"] - box.eng.half
    o["H_gps"] = segs[c["i"]] + (r["H1"]["lo"] + lag0 + o["lag"]) / FS                 # H1 peak time
    o["L_gps0"] = segs[c["iL"]] + (r["L1"]["lo"] + lag0 + o["lag"] - DELTA) / FS       # L1 time at delta = -10 ms
    o.update(o_extra)
    return o


META = ("n", "kind", "k", "i", "iL", "i_c", "y1", "inj", "tid_y1", "decoy", "real")


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
    log(f"run L{l_max} {kinds}: {len(todo)} boxes to do on {dev}")
    t_start = time.time()
    overlap = os.environ.get("COH2T_OVERLAP") == "1"       # child for chunk s+1 runs while chunk s is scored

    def launch(i):
        part = todo[i:i + CHUNK]
        return subprocess.Popen([sys.executable, "-u", __file__, "chunk", str(l_max), ",".join(map(str, part)),
                                 ",".join(kinds)]) if part else None
    nxt = launch(0)
    for s in range(0, len(todo), CHUNK):
        part = todo[s:s + CHUNK]
        assert nxt.wait() == 0, "chunk child failed"
        nxt = launch(s + CHUNK) if overlap else None
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
                    if kind == "twin":
                        res.append(candidate(box, fine, series, c, None, target, twin=loaded))
                    else:
                        res.append(candidate(box, fine, series, c, loaded, target))
                meta = dict(tid=tid, l_max=l_max, kind=kind, mc_star=float(box.gr["mc_star"]), n_tmpl=box.n,
                            n_refs=len(box.h0c), worst_dtau=float(box.gr["worst_dtau"]), ff_min=float(box.gr["ff_min"]),
                            candidates=[{k: c.get(k) for k in META} for c in cs], seconds=round(time.time() - t0, 2),
                            peak_footprint_mb=L.peak_footprint_mb(), model="Claude Opus 5.5", status="UNVERIFIED")
                arrs = {k: np.array([r[k] for r in res]) for k in res[0]} if res else {}
                for k in ("H_rho", "H_chi2", "L_rho", "L_chi2", "closure"):
                    if k in arrs:
                        arrs[k] = arrs[k].astype(np.float32)
                done_dir.mkdir(parents=True, exist_ok=True)
                tmp = out.with_name(out.stem + ".tmp.npz")
                np.savez_compressed(tmp, meta=json.dumps(meta), **arrs)
                os.replace(tmp, out)
            for pth in [refs_path] + [f for c in by_box.get(tid, []) for f in extra_files(c).values()]:
                pth.unlink(missing_ok=True)
            del box
            log(f"box {tid} ({s + part.index(tid) + 1}/{len(todo)}) {time.time() - t0:.1f} s | footprint "
                f"{L.peak_footprint_mb()} MB | {(time.time() - t_start) / 3600:.2f} h")
        if not overlap and s + CHUNK < len(todo):
            nxt = launch(s + CHUNK)


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
    the driver's own engine path with the same H1 series in both detector slots:
      * the box maximum of |z_H|^2 equals results/coh2c_verify's box_max_local within 1e-5, same template count;
      * the band pieces at delta = 0 equal coh2c_local.chi2_fine (gate E3's function) within 1e-5, for p = 4, 8, 16;
      * closure (sum of bands vs the engine's z) <= 1e-4 at every top point, in both slots;
      * with identical slots, L at delta = 0 equals H, and the plain network value is >= 2 x the box maximum.
    """
    import torch
    ref = json.loads((B / "results" / "coh2c_verify" / "box_H1_1249001472_noise_mc0.700_L128_cpu.json").read_text())
    st = json.loads((B / "state" / "realbg" / "caught.json").read_text())
    gps = segments()[0]
    i_c = [b for b in st["candidates"]["0_0"] if (L.PRE + 1) * FS < b * 102 < 4096 * FS - 65 * FS][0] * 102
    dev = L.pick_dev()
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
    ysn, kappa = noise_spectrum(dict(kind="bg", i_c=i_c, n=0), "H1", y, w_snip)
    r = prepare(box, fine, ysn, kappa, w_snip, i_c)
    o = score(box, fine, r, r)
    rel = abs(o["tpeak_max"] / ref["box_max_local"] - 1)
    # band pieces vs gate E3's own function, at the top point
    tidx, n0 = int(o["tmpl"][0]), r["i_s"] - eng.half + int(o["lag"][0])
    j = int(gr["ref_of"][tidx])
    z16 = band_series(box, fine, r, tidx, n0)[DELTA]
    dco = torch.tensor((cs[tidx] - gr["ref_coeffs"][j]).astype(np.float32), device=dev).unsqueeze(0)
    e3_err = 0.0
    for p in P_BINS:
        ed = r["refs"][j]["edges16"]
        edp = [(ed[k][0], ed[k + 16 // p - 1][1]) for k in range(0, 16, 16 // p)]
        zj = L.chi2_fine(torch, r["refs"][j]["p0"], dco, sign, torch.tensor([n0], device=dev), edp, fine.basis,
                         fine.k_idx)[0].cpu().numpy().astype(np.complex128) / (r["refs"][j]["sig0"] * r["kappa"])
        mine = z16.reshape(p, 16 // p).sum(axis=1)
        e3_err = max(e3_err, float(np.abs(mine - zj).max() / max(abs(zj.sum()), 1.0)))
    clos = float(o["closure"].max())
    same_hl = float(np.abs(o["L_rho"][:, DELTA] - o["H_rho"]).max() / o["H_rho"].max())
    ok = (rel <= 1e-5 and box.n == ref["n_templates"] and e3_err <= 1e-5 and clos <= 1e-4 and same_hl <= 1e-6
          and o["plain_net"] >= 2 * o["tpeak_max"] * (1 - 1e-6))
    res = dict(gate="driver golden", device=dev, n_templates=box.n, box_max_driver=o["tpeak_max"],
               box_max_verify=ref["box_max_local"], rel=rel, band_vs_chi2_fine_max=e3_err, closure_max=clos,
               identical_slot_L_vs_H=same_hl, plain_net=o["plain_net"],
               top=dict(tmpl=o["tmpl"].tolist(), lag=o["lag"].tolist(), H_rho=o["H_rho"].tolist(),
                        H_chi2_p4_p8_p16=o["H_chi2"].tolist()),
               passed=bool(ok), peak_footprint_mb=L.peak_footprint_mb(), model="Claude Opus 5.5", env=L.stamp())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"driver_golden_{dev}.json").write_text(json.dumps(res, indent=1))
    log(f"driver golden {'PASS' if ok else 'FAIL'}: box max {o['tpeak_max']:.6f} vs {ref['box_max_local']:.6f} "
        f"(rel {rel:.1e}), bands vs chi2_fine {e3_err:.1e}, closure {clos:.1e}, L=H {same_hl:.1e}, {box.n} templates "
        f"| footprint {L.peak_footprint_mb()} MB")


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
    else:
        dict(plan=plan, golden=golden, prepseries=prepseries, vseries=vseries, run=run, calib=calib,
             twinexact=twinexact, k1precheck=k1precheck, status=status)[cmd](*[None if v == "-" else v for v in a[2:]])
