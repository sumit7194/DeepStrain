#!/bin/bash
# BlackHole regression gate — the LIGO-data projects (echoes, ringdown_spectroscopy,
# primordial_blackhole_search). Asserts each project's headline artifacts against saved
# results. (moved out of SpaceTime 2026-06-13.)
set -u
cd "$(dirname "$0")"
FAIL=0

echo "--- echoes artifacts"
./echoes/.venv/bin/python - << 'PYEOF' || FAIL=1
import numpy as np
d = np.load("echoes/results/07_ml_scorer.npy", allow_pickle=True).item()
amps, eff = d["amps"], d["eff_v2"]
assert eff[list(amps).index(0.5)] >= 0.9, f"v2 sensitivity regressed: {eff}"
assert d["on_p_pred"] > 0.05, f"on-source no longer null: {d['on_p_pred']}"
assert max(d["control_irregular"].values()) <= 0.15, "specificity regressed"
print("PASS  echoes v2/v3 headline artifacts")
PYEOF

echo "--- ringdown recalibration artifacts (10)"
./ringdown_spectroscopy/.venv/bin/python - << 'PYEOF3' || FAIL=1
import json
r = json.loads(open("ringdown_spectroscopy/results/10_recalibration.json").read())
assert r["kerr_inside_90"] and all(0.85 <= c <= 0.95 for c in r["coverage_heldout"]), r
print("PASS  ringdown recalibration artifacts")
PYEOF3

echo "--- ringdown no-hair artifacts (09)"
./ringdown_spectroscopy/.venv/bin/python - << 'PYEOF' || FAIL=1
import json
d = json.loads(open("ringdown_spectroscopy/results/09_nohair_GW250114.json").read())
assert d["kerr_inside_90"] is True, "GW250114 no longer Kerr-consistent?!"
assert 0.80 <= d["coverage"]["delta"] <= 0.96, f"delta coverage: {d['coverage']}"
print("PASS  ringdown v2 headline artifacts")
PYEOF

echo "--- ringdown v5 delta-stacking METHOD (12: sigma tightens as sqrt(N) on informative injections)"
./ringdown_spectroscopy/.venv/bin/python - << 'PYEOFS' || FAIL=1
import json
d = json.loads(open("ringdown_spectroscopy/results/12_stacking.json").read())
big = d["injection"][-1]  # N=8
assert big["N"] == 8 and abs(big["sigma_stack"] - big["expect"]) / big["expect"] < 0.15, \
    f"stacking method no longer ~sqrt(N): {big['sigma_stack']} vs {big['expect']}"
assert d["gates"]["S1_unbiased"] and d["gates"]["S3_coverage"], "stacking S1/S3 regressed"
# NOTE: NO real-event "stack < singles" assertion -- 13_more_events.py stress-test showed only
# GW250114 is informative; the 2-event tightening was a Gaussian-approx-of-prior artifact (corrected).
print("PASS  ringdown v5 stacking METHOD (sigma(delta)~sqrt(N) on informative injections; real stack parked)")
PYEOFS

echo "--- ringdown v5 stress-test (13: only GW250114 informative, rest ~prior)"
./ringdown_spectroscopy/.venv/bin/python - << 'PYEOFT' || FAIL=1
import json, numpy as np
d = json.loads(open("ringdown_spectroscopy/results/13_more_events.json").read())
prior = 1.0/np.sqrt(12)
def dsig(v): return (v["delta"][2]-v["delta"][1])/(2*1.645)
g = d["GW250114_082203"]
assert dsig(g)/prior < 0.85, "GW250114 no longer informative?!"
faint = [k for k in d if d[k] and k != "GW250114_082203"]
assert all(dsig(d[k])/prior > 0.88 for k in faint), "a faint event became informative -- recheck stacking"
print("PASS  ringdown v5 stress-test (only GW250114 measures delta; fainter events ~prior -> no real stack)")
PYEOFT

echo "--- pbh sensitivity artifacts (eval_cnn)"
./primordial_blackhole_search/.venv/bin/python - << 'PYEOF4' || FAIL=1
import json
d = json.loads(open("primordial_blackhole_search/results/eval_cnn.json").read())
fracs = d["mf_distance_fraction"]
assert all(0.38 <= f <= 0.48 for f in fracs.values()), f"MF distance fraction regressed: {fracs}"
assert d["thresh_zero_fa"] > 0, f"zero-FA threshold missing: {d['thresh_zero_fa']}"
print("PASS  pbh CNN sensitivity artifacts")
PYEOF4

echo "--- pbh track-aggregation artifacts (eval_cnn_track)"
./primordial_blackhole_search/.venv/bin/python - << 'PYEOF5' || FAIL=1
import json
v1 = json.loads(open("primordial_blackhole_search/results/eval_cnn.json").read())
d = json.loads(open("primordial_blackhole_search/results/eval_cnn_track.json").read())
assert abs(d["thresholds"]["max"] - v1["thresh_zero_fa"]) < 1e-6, "max-control no longer anchors v1 threshold"
fr = [f for s in d["results"].values() for f in s["mf_distance_fraction"].values()]
assert all(0.35 <= f <= 0.50 for f in fr), f"track fractions out of band: {fr}"
print("PASS  pbh track-aggregation artifacts (negative result: agg ~= max)")
PYEOF5

echo "--- pbh rung-2 track artifacts (eval_cnn_w64_track_w64)"
./primordial_blackhole_search/.venv/bin/python - << 'PYEOF6' || FAIL=1
import json
d = json.loads(open("primordial_blackhole_search/results/eval_cnn_w64_track_w64.json").read())
r = d["results"]
for m in ("0.17-0.35", "0.35-0.55", "0.55-0.88"):
    mx = r["max"]["mf_distance_fraction"][m]
    st = r["sum_track"]["mf_distance_fraction"][m]
    orc = r["oracle"]["mf_distance_fraction"][m]
    assert 0.35 <= mx <= 0.55, f"rung-2 max out of band [{m}]: {mx}"
    assert abs(st - mx) < 0.03 and abs(orc - mx) < 0.03, \
        f"rung-2 aggregation no longer ~= max [{m}]: max={mx} sum={st} oracle={orc}"
print("PASS  pbh rung-2 artifacts (negative: accumulation ~= max, oracle ceiling flat)")
PYEOF6

echo "--- pbh rung-3 stage-0 oracle (oracle_semicoherent: n=8 vetoed clears the gate)"
./primordial_blackhole_search/.venv/bin/python - << 'PYEOF7' || FAIL=1
import json
d = json.loads(open("primordial_blackhole_search/results/oracle_semicoherent.json").read())
gate = d["cnn_w64_gate"]
v8 = d["ceilings"]["vetoed"]["n8"]["mf_distance_fraction"]
for m in ("0.17-0.35", "0.35-0.55", "0.55-0.88"):
    assert v8[m] >= gate[m] + 0.05, f"n=8 vetoed no longer clears the gate [{m}]: {v8[m]} vs {gate[m]}"
# n<=4 must stay glitch-limited (the chunk-veto sweet-spot story)
assert d["ceilings"]["vetoed"]["n4"]["mf_distance_fraction"]["0.55-0.88"] < 0.1, "n=4 vetoed no longer glitch-limited?"
print("PASS  pbh rung-3 oracle (n=8 vetoed 0.66/0.76/0.75 clears cnn_w64; ceiling, optimistic)")
PYEOF7

echo "--- pbh rung-3 stage-1 learned (definitive negative: both designs -> 0 sensitive distance)"
./primordial_blackhole_search/.venv/bin/python - << 'PYEOF8' || FAIL=1
import json
RES = "primordial_blackhole_search/results"
for tag, auc in (("semicoherent_v1def", 0.706), ("semicoherent_v2", 0.691)):
    d = json.loads(open(f"{RES}/eval_semicoherent_{tag}.json").read())
    fr = d["mf_distance_fraction"]
    for m in ("0.17-0.35", "0.35-0.55", "0.55-0.88"):
        assert fr[m] == 0.0, f"{tag} no longer 0 sensitive distance [{m}]: {fr[m]}"
print("PASS  pbh rung-3 stage-1 (learned V1 0.706 / V2 0.691 AUC both -> 0.0 dist; gap needs coherent method)")
PYEOF8

echo "--- pbh path-G coincidence (coinc_eval: H1xL1 beats single-det at matched FAR)"
./primordial_blackhole_search/.venv/bin/python - << 'PYEOF9' || FAIL=1
import json
d = json.loads(open("primordial_blackhole_search/results/coinc_eval.json").read())
s, c = d["single_det_fraction"], d["coinc_fraction_matchedFAR"]
for m in ("0.17-0.35", "0.35-0.55", "0.55-0.88"):
    assert c[m] > s[m] + 0.03, f"coincidence no longer beats single-det [{m}]: {c[m]} vs {s[m]}"
assert d["coinc_fraction_matchedFAR"]["0.55-0.88"] >= 0.40, "high-mass coinc distance regressed below 0.40"
print("PASS  pbh path-G coincidence (H1xL1 +1.3-1.5x sensitive distance over single-det, matched FAR)")
PYEOF9

echo "========================================"
[ $FAIL -eq 0 ] && echo "BLACKHOLE GATE: ALL GREEN" || echo "BLACKHOLE GATE: FAILURES"
exit $FAIL
