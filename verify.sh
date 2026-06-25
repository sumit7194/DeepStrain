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

echo "--- echoes E1 production-path upper limit (12: ML scorer does NOT tighten the exclusion -- honest negative)"
./echoes/.venv/bin/python - << 'PYEOFE1' || FAIL=1
import json
d = json.loads(open("echoes/results/12_ul_production.json").read())
i = d["dt"].index(min(d["dt"], key=lambda x: abs(x - d["dt_pred"])))   # predicted-Δt index
a90_cb, a90_ml = d["a90"]["comb"][i], d["a90"]["ml"][i]
assert 1.0 <= a90_cb <= 2.0, f"production-path comb A90 out of sane range: {a90_cb}"
# the ML scorer must NOT look ">1.2x tighter" (that's the whitened-domain artifact); honest result is ML ~= comb
assert d["ratio_at_pred"] <= 1.15, f"ML A90 'tighter' than comb by >15% -> whitened-domain artifact leaked in: {d['ratio_at_pred']}"
assert d["ratio_at_pred"] >= 0.80, f"ML A90 implausibly worse than comb: {d['ratio_at_pred']}"
print(f"PASS  echoes E1 (production-path: ML A90={a90_ml:.2f} ~ comb A90={a90_cb:.2f}; ML does NOT tighten the UL)")
PYEOFE1

echo "--- echoes E2 independent background (13: GW150914 null holds vs a different-time, 4x-larger background)"
./echoes/.venv/bin/python - << 'PYEOFE2' || FAIL=1
import json
d = json.loads(open("echoes/results/13_independent_bg_GW150914.json").read())
assert d["n_bg"] >= 300, f"independent background too small to be meaningful: {d['n_bg']}"
assert d["n_bg"] > 159, "independent background not larger than the shared-block one"   # genuinely more data
assert min(d["p_max"], d["p_pred"]) > 0.05, f"on-source no longer null vs independent bg: {d['p_max']}, {d['p_pred']}"
print(f"PASS  echoes E2 (independent {d['n_bg']}-pair background: null holds, p_max={d['p_max']:.3f} p_pred={d['p_pred']:.3f})")
PYEOFE2

echo "--- echoes echo-spacing formula (14: first-principles Δt(M,χ) reproduces Abedi Table I to <5%)"
./echoes/.venv/bin/python - << 'PYEOFES' || FAIL=1
import json
d = json.loads(open("echoes/results/14_echo_spacing.json").read())
assert d["validation_pass"], "echo-spacing formula no longer reproduces Abedi Table I"
for ev, r in d["events"].items():
    assert abs(r["rel_err"]) < 0.05, f"{ev} echo Δt off by {r['rel_err']:.1%} (formula regressed)"
# the bug this caught: GW151226 must be ~0.1013 (Abedi), NOT the old wrong 0.0579
assert abs(d["events"]["GW151226"]["dt_published"] - 0.1013) < 1e-4, "GW151226 reference Δt wrong again"
print("PASS  echoes echo-spacing formula (uncalibrated Kerr-tortoise Δt reproduces all 3 Abedi values <2%)")
PYEOFES

echo "--- echoes N1 joint ringdown<->echo (15: mass-conditioned echo search on GW250114 is more sensitive)"
./echoes/.venv/bin/python - << 'PYEOFN1' || FAIL=1
import json
d = json.loads(open("echoes/results/15_joint_ringdown_echo.json").read())
assert d["trials_ratio"] > 1.5, f"conditioned window not meaningfully tighter: {d['trials_ratio']}x"
assert 0.30 < d["dt_prior"]["median"] < 0.40, f"GW250114 echo Δt prior off (verified formula): {d['dt_prior']['median']}"
assert d["A90"]["conditioned"] <= d["A90"]["flat"], "conditioned search not >= as sensitive as flat"
assert d["thr_cond"] < d["thr_flat"], "conditioned threshold not lower than flat"
assert min(d["on_source_p"]["flat"], d["on_source_p"]["conditioned"]) > 0.05, "GW250114 echo no longer null"
print(f"PASS  echoes N1 (ringdown-conditioned echo search {d['sensitivity_gain']:.2f}x more sensitive; GW250114 null)")
PYEOFN1

echo "--- echoes N3 stacked multi-event echo search (16: population null + combined limit tighter than best single)"
./echoes/.venv/bin/python - << 'PYEOFN3' || FAIL=1
import json
d = json.loads(open("echoes/results/16_stacked_echo.json").read())
assert len(d["events"]) >= 4, "stacked over fewer than 4 events"
assert d["p_stacked"] > 0.05, f"stacked echo search no longer null: p={d['p_stacked']}"
assert d["a90_stacked"] < d["best_single_a90"], "stacking did not tighten the limit vs best single"
assert d["stack_gain"] > 1.0, f"stack gain <= 1: {d['stack_gain']}"
print(f"PASS  echoes N3 (stacked {len(d['events'])}-event echo search: null p={d['p_stacked']:.2f}, limit {d['stack_gain']:.2f}x tighter than best single)")
PYEOFN3

echo "--- ringdown R3 IMR referee (15: no-hair NPE unbiased on tones, start-time systematic on realistic IMR)"
./ringdown_spectroscopy/.venv/bin/python - << 'PYEOFR3' || FAIL=1
import json
d = json.loads(open("ringdown_spectroscopy/results/15_imr_referee.json").read())
assert abs(d["control_delta_mean"]) < 0.10, f"analytic control no longer unbiased: {d['control_delta_mean']}"
assert d["peak_bias"] > 0.20, f"IMR-from-peak bias vanished (expected the systematic): {d['peak_bias']}"
assert d["bias_shrinks_late"], "bias no longer shrinks at later start -> mechanism not reproduced"
sweep = d["start_time_sweep_ms"]
late = sweep.get("6.0", sweep[max(sweep, key=lambda k: float(k))])
assert abs(late) < 0.10, f"bias not gone by the latest start offset: {late}"
print(f"PASS  ringdown R3 (control δ={d['control_delta_mean']:+.2f}; IMR-peak δ≈{-d['peak_bias']:.2f} systematic, decays to {late:+.2f} by 6ms)")
PYEOFR3

echo "--- ringdown R3 capstone (16: GW250114 δ vs start time -- reproduces 09 at peak, Kerr-consistent throughout)"
./ringdown_spectroscopy/.venv/bin/python - << 'PYEOFC3' || FAIL=1
import json
d = json.loads(open("ringdown_spectroscopy/results/16_gw250114_starttime.json").read())
peak = d["delta_vs_start"]["0.0"]["median"]
assert abs(peak - (-0.16)) < 0.06, f"peak-cropped δ no longer reproduces 09's -0.16: {peak}"   # validation
assert d["all_kerr_consistent"], "a start offset is no longer Kerr-consistent"                 # headline holds
# the late-start (systematic-mitigated) δ sits closer to Kerr than the peak value
late = d["delta_vs_start"][str(d["offsets_ms"][-1])]["median"]
assert abs(late) <= abs(peak) + 0.05, f"late-start δ not closer-or-equal to Kerr: late={late} peak={peak}"
print(f"PASS  ringdown R3 capstone (GW250114 δ: peak {peak:+.2f} reproduces 09, late-start {late:+.2f}; all Kerr-consistent)")
PYEOFC3

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

echo "--- ringdown v6 delta-measurability threshold (14: sigma(delta) shrinks with SNR; GW250114 at the edge)"
./ringdown_spectroscopy/.venv/bin/python - << 'PYEOFD' || FAIL=1
import json
d = json.loads(open("ringdown_spectroscopy/results/14_delta_threshold.json").read())
c = d["curve"]
faint = next(r for r in c if r["a220"] == 2.0); loud = next(r for r in c if r["a220"] == 12.0)
assert faint["ratio"] >= 0.95, f"faint ringdown no longer ~prior: {faint['ratio']}"          # uninformative when quiet
assert loud["ratio"] <= 0.90, f"loudest trained ringdown not informative: {loud['ratio']}"     # monotone shrink
assert loud["ratio"] > d["gw250114_ratio"], "trained edge passed GW250114 -- recheck loudness mapping"
assert d["best_ratio"] >= 0.80, f"single-event delta got implausibly tight: {d['best_ratio']}" # still prior-limited
print("PASS  ringdown v6 threshold (delta informative only at GW250114-class loudness; ~13% tighter, the SNR wall)")
PYEOFD

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

echo "--- pbh Build C-2 (coinc_learned_segments: LEARNED H1xL1 coincidence beats sum, leakage-free + significant)"
./primordial_blackhole_search/.venv/bin/python - << 'PYEOFL' || FAIL=1
import json
d = json.loads(open("primordial_blackhole_search/results/coinc_learned_segments.json").read())
assert "HELD-OUT SEGMENTS" in d["mode"], "not the gold-standard cross-segment run"   # no noise/segment leakage
ML = ("0.17-0.35", "0.35-0.55", "0.55-0.88")
# learned beats sum (high-mass) at every honestly-supported FAR ...
for far, v in d["vs_far"].items():
    assert v["learned"]["0.55-0.88"] > v["sum"]["0.55-0.88"], f"learned <= sum high-mass at {far}"
# ... AND the HIGH-MASS gain (the headline) is significant (bootstrap 90% CI lower bound > 0) at every honest FAR.
# (the light bin 0.17-0.35 is the weakest -- marginal at the loosest FAR -- so we gate the robust high-mass claim;
#  honest distinct-lag slides: 504 eval-noise windows -> bg ~0.5 yr -> 1/year is NOT supported, auto-dropped.)
for far in d["bootstrap"]:
    lo = d["bootstrap"][far]["0.55-0.88"]["ci90"][0]
    assert lo > 0, f"high-mass learned-sum gain not significant [{far}]: CI lower bound {lo}"
# mid bin significant at the strictest honest FAR too
assert d["bootstrap"]["1/month"]["0.35-0.55"]["ci90"][0] > 0, "mid-mass gain not significant at 1/month"
assert "1/year" not in d["vs_far"], "1/year present -> overcounted slides regressed (honest bg is ~0.5 yr)"
print("PASS  pbh Build C-2 (learned coincidence: high-mass +0.02-0.05 over sum, cross-segment, 90% CI>0, honest FAR<=1/month)")
PYEOFL

echo "--- pbh Build C-2 lower-FAR (coinc_learned_holdout: leakage-clean, reaches 1/year, learned still > sum)"
./primordial_blackhole_search/.venv/bin/python - << 'PYEOFN' || FAIL=1
import json
d = json.loads(open("primordial_blackhole_search/results/coinc_learned_holdout.json").read())
assert "HELD-OUT noise" in d["mode"], "not the leakage-clean held-out-noise run"   # head never saw eval-bg noise
assert 1.0 < d["bg_days"]/365 < 1.5, f"held-out-noise honest bg should be ~1.16 yr: {d['bg_days']/365:.2f}"
assert "1/year" in d["vs_far"], "held-out-noise (1.16 yr bg) should reach 1/year"
yr = d["vs_far"]["1/year"]
assert yr["learned"]["0.55-0.88"] > yr["sum"]["0.55-0.88"], "learned <= sum high-mass at clean 1/year"
lo = d["bootstrap"]["1/year"]["0.55-0.88"]["ci90"][0]
assert lo > 0, f"clean 1/year gain not significant: CI lower bound {lo}"
print("PASS  pbh Build C-2 lower-FAR (leakage-clean 1/year: learned > sum, 90% CI>0)")
PYEOFN

echo "--- pbh Build C (coinc_far: coincidence holds at realistic FAR, down to 1/year)"
./primordial_blackhole_search/.venv/bin/python - << 'PYEOFC' || FAIL=1
import json
d = json.loads(open("primordial_blackhole_search/results/coinc_far.json").read())
ML = ("0.17-0.35", "0.35-0.55", "0.55-0.88")
s = d["single_det_floor_frac"]
day = d["coinc_vs_far"]["1/day"]["frac"]; yr = d["coinc_vs_far"]["1/year"]["frac"]
# cross-check: coinc @1/day reproduces the local G1 +1.3-1.5x over single-det floor
for m in ML:
    assert 1.2 <= day[m]/s[m] <= 1.6, f"Build C 1/day gain off [{m}]: {day[m]/s[m]:.2f}"
# graceful: even at 1/year (FAR a single detector can't reach) coinc still beats the single-det floor
for m in ML:
    assert yr[m] > s[m], f"Build C coinc @1/year no longer beats single-det floor [{m}]"
assert d["bg_days"] > 365, "Build C background livetime < 1 yr -- cannot probe realistic FAR"
print("PASS  pbh Build C (coinc FAR-robust: 1/day ~1.4x = local G1; 1/year still > single-det floor)")
PYEOFC

echo "========================================"
[ $FAIL -eq 0 ] && echo "BLACKHOLE GATE: ALL GREEN" || echo "BLACKHOLE GATE: FAILURES"
exit $FAIL
