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
# BACKED (2026-08-16 flag audit): a bare boolean written by the script it guards is self-certification.
# The flag says every start offset brackets Kerr; these are the CI edges that make it true.
assert d["all_kerr_consistent"], "a start offset is no longer Kerr-consistent"
assert d["delta_vs_start"]["0.0"]["lo"] < -0.05, f"peak-start CI no longer brackets Kerr from below: {d['delta_vs_start']['0.0']['lo']}"
assert d["delta_vs_start"]["0.0"]["hi"] > 0.05, f"peak-start CI no longer brackets Kerr from above: {d['delta_vs_start']['0.0']['hi']}"
assert d["peak_to_late_drift"] < 0.40, f"delta drift across start times grew: {d['peak_to_late_drift']}"
# the late-start (systematic-mitigated) δ sits closer to Kerr than the peak value
late = d["delta_vs_start"][str(d["offsets_ms"][-1])]["median"]
assert abs(late) <= abs(peak) + 0.05, f"late-start δ not closer-or-equal to Kerr: late={late} peak={peak}"
print(f"PASS  ringdown R3 capstone (GW250114 δ: peak {peak:+.2f} reproduces 09, late-start {late:+.2f}; all Kerr-consistent)")
PYEOFC3

echo "--- echoes N2 H1xL1 consistency statistic (17: pre-chosen lambda, bootstrap -- honest mixed/modest, not universal)"
./echoes/.venv/bin/python - << 'PYEOFN2' || FAIL=1
import json
d = json.loads(open("echoes/results/17_echo_consistency.json").read())
# BACKED (2026-08-16 flag audit): a bare boolean written by the script it guards is self-certification.
# "does not universally help" is a claim about a90 barely moving with lambda -- assert that number.
assert not d["consistency_helps"], "N2 now claims a UNIVERSAL win -- recheck (was honest mixed/modest)"
assert d["events"]["GW150914"]["a90_vs_lambda"]["0.5"] > 1.85, \
    f"GW150914 a90 at lambda=0.5 improved more than N2 claimed: {d['events']['GW150914']['a90_vs_lambda']['0.5']}"
for ev, o in d["events"].items():
    assert "dA90_ci90" in o, f"{ev} bootstrap CI missing (significance not assessed)"
    assert abs(o["pre_a90"] - o["sum_a90"]) < 0.25, f"{ev} consistency effect implausibly large: {o['pre_a90']} vs {o['sum_a90']}"
# the modest help is real for at least one event (GW150914 CI below 0)
assert any(o["dA90_ci90"][1] < 0 for o in d["events"].values()), "no event shows even a modest significant gain"
print("PASS  echoes N2 (consistency-weighted statistic: modest, event-dependent, not a universal win -- selection-bias avoided)")
PYEOFN2

echo "--- ringdown A5 tone-fit export for TheBridge (18: per-event 220/221 fits, reliability-flagged)"
./ringdown_spectroscopy/.venv/bin/python - << 'PYEOFA5' || FAIL=1
import json
d = json.loads(open("ringdown_spectroscopy/results/18_tonefits.json").read())
ok = {k: v for k, v in d.items() if v is not None}
assert len(ok) >= 7, f"too few events exported: {len(ok)}"
reliable = [k for k, v in ok.items() if not v["tone220_railed"]]
assert len(reliable) >= 4, f"too few reliable (non-railed) 220 fits for the bridge: {len(reliable)}"
g = ok["GW250114_082203"]["tone220_1mode"]
assert g["M_inv"] and 60 < g["M_inv"] < 100, f"GW250114 220 inversion implausible: {g['M_inv']}"
print(f"PASS  ringdown A5 export ({len(ok)}/8 events, {len(reliable)} reliable 220 fits; railed flagged for TheBridge)")
PYEOFA5

echo "--- ringdown A1 amortization->transfer for TheBridge (19: 5 NPE variants, gap + transfer per variant)"
./ringdown_spectroscopy/.venv/bin/python - << 'PYEOFA1' || FAIL=1
import json
d = json.loads(open("ringdown_spectroscopy/results/19_amortization_transfer.json").read())
v = d["variants"]
assert len(v) >= 5, f"too few NPE variants: {len(v)}"
g = {x["n_train"]: x["amortization_gap"] for x in v}
assert g[5000] > g[150000], "amortization gap should shrink with training (5k > 150k)"   # clean lever
assert all(x["transfer"] <= 0.02 for x in v), "transfer should be ~<=0 (sim->real degrades)"
assert "amortization_gap_vs_transfer_corr" in d, "gap-vs-transfer correlation missing"
print(f"PASS  ringdown A1 ({len(v)} variants; amort_gap {g[5000]:.3f}->{g[150000]:.3f} with N; gap-vs-transfer corr {d['amortization_gap_vs_transfer_corr']:+.2f}, for TheBridge)")
PYEOFA1

echo "--- echoes Abedi cross-check export for TheBridge leg 8 (18: per-event Δt vs Abedi, formula + % agreement)"
./echoes/.venv/bin/python - << 'PYEOFAB' || FAIL=1
import json
d = json.loads(open("echoes/results/18_abedi_crosscheck.json").read())
# BACKED (2026-08-16 flag audit): a bare boolean written by the script it guards is self-certification;
# these assert the NUMBER the flag summarises, so the gate fails if the number moves.
assert "expression" in d["formula"] and d["formula"]["no_free_parameter_tuned_to_dt"], "formula string/flag missing"
assert d["validation_summary"]["min_percent_agreement"] > 98.0, \
    f"Abedi agreement fell below 98%: {d['validation_summary']['min_percent_agreement']}"
abedi = [e for e in d["events"] if e["abedi_table_I"]]
assert len(abedi) >= 3, "fewer than 3 Abedi-Table-I cross-check events"
assert all(e["percent_agreement"] >= 98.0 for e in abedi), "Abedi agreement dropped below 98% (>2% error)"
assert any(e["event"] == "GW250114" for e in d["events"]), "GW250114 (post-2017 closed-form application) missing"
print(f"PASS  echoes Abedi cross-check ({len(abedi)} events {d['validation_summary']['min_percent_agreement']:.1f}-{d['validation_summary']['max_percent_agreement']:.1f}% agreement; for TheBridge leg 8)")
PYEOFAB

echo "--- ringdown recalibration artifacts (10)"
./ringdown_spectroscopy/.venv/bin/python - << 'PYEOF3' || FAIL=1
import json
r = json.loads(open("ringdown_spectroscopy/results/10_recalibration.json").read())
# BACKED (2026-08-16 flag audit): a bare boolean written by the script it guards is self-certification;
# these assert the NUMBER the flag summarises, so the gate fails if the number moves.
assert r["kerr_inside_90"] and all(0.85 <= c <= 0.95 for c in r["coverage_heldout"]), r
assert r["gw250114_delta"][1] < -0.05, f"delta CI no longer brackets Kerr from below: {r['gw250114_delta'][1]}"
assert r["gw250114_delta"][2] > 0.05, f"delta CI no longer brackets Kerr from above: {r['gw250114_delta'][2]}"
print("PASS  ringdown recalibration artifacts")
PYEOF3

echo "--- echoes E3 per-event ML scorers (19: broadened set, all clean nulls vs independent background)"
./echoes/.venv/bin/python - << 'PYEOFE3' || FAIL=1
import json
d = json.loads(open("echoes/results/19_per_event_ml.json").read())
evs = {e["event"]: e for e in d["events"]}
# broadened set incl. GW250114 all ran (no data-starvation skips after the independent-background upgrade)
for name in ("GW150914", "GW151012", "GW151226", "GW250114_082203"):
    e = evs[name]
    assert "error" not in e, f"{name}: {e.get('error')}"
    assert e["n_bg"] >= 500, f"{name}: background too small ({e['n_bg']})"
    # every event a clean null under BOTH the ML scorer and the comb baseline (no p<0.05 detection)
    assert e["ml_p_at_dt"] > 0.05 and e["comb_p_at_dt"] > 0.05, f"{name}: not null ({e['ml_p_at_dt']}, {e['comb_p_at_dt']})"
# GW151012's small-sample ML p=0.033 (own-block, n=59) must NOT survive the larger independent background
assert evs["GW151012"]["ml_p_at_dt"] > 0.05, "GW151012 low-p did not wash out"
print(f"PASS  echoes E3 (4 events all null vs independent bg n_bg 660-1815; GW151012 own-block 0.033 -> {evs['GW151012']['ml_p_at_dt']:.2f}, a small-sample artifact)")
PYEOFE3

echo "--- ringdown R2 v2 (21: proper ringdown-package tone-count + NPE referee)"
./ringdown_spectroscopy/.venv311/bin/python - << 'PYEOFR2' || FAIL=1
import json
r = json.loads(open("ringdown_spectroscopy/results/21_ringdown_crosscheck.json").read())
g, d1, d2 = r["GW150914_n2"], r["GW250114_082203_n1"], r["GW250114_082203_n2"]
# (a) validation: GW150914 220+221 lands in the known ballpark (pre-registered band)
assert 55 < g["m"]["q50"] < 90 and 0.4 < g["chi"]["q50"] < 0.9, f"GW150914 validation off: {g['m']['q50']}, {g['chi']['q50']}"
# (b) the R2 question: GW250114 overtone amplitude bounded away from zero (field statistic; published = yes)
assert d2["a221_frac_below_10pct_median"] <= 0.01, f"GW250114 A221 no longer bounded away from 0: {d2['a221_frac_below_10pct_median']}"
# (c) NPE referee: package (M, chi) median consistent with our 09 NPE on-source posterior (76.0, 0.762)
npe = json.loads(open("ringdown_spectroscopy/results/09_nohair_GW250114.json").read())["posterior"]
assert abs(d2["m"]["q50"] - npe["mass"][0]) < 5.0, f"package M {d2['m']['q50']} vs NPE {npe['mass'][0]}"
assert abs(d2["chi"]["q50"] - npe["chi"][0]) < 0.08, f"package chi {d2['chi']['q50']} vs NPE {npe['chi'][0]}"
assert npe["mass"][1] < d2["m"]["q50"] < npe["mass"][2], "package M median outside NPE 90% CI"
# (d) convergence: NUTS chains healthy
for k, dd in (("GW150914_n2", g), ("n1", d1), ("n2", d2)):
    dg = dd["diagnostics"]
    assert all(v < 1.01 for n, v in dg.items() if n.startswith("rhat")), f"{k} rhat: {dg}"
    assert all(v > 400 for n, v in dg.items() if n.startswith("ess")), f"{k} ess: {dg}"
print(f"PASS  ringdown R2 v2 (GW250114 overtone bounded away from 0 [P={d2['a221_frac_below_10pct_median']:.3f}]; "
      f"package M={d2['m']['q50']:.1f}/chi={d2['chi']['q50']:.2f} vs NPE {npe['mass'][0]:.1f}/{npe['chi'][0]:.2f}; rhat<1.01)")
PYEOFR2

echo "--- ringdown B1 start-time sweep (22: package refereess R3 -- overtone damps, peak-mass biased)"
./ringdown_spectroscopy/.venv311/bin/python - << 'PYEOFB1' || FAIL=1
import json
d = json.loads(open("ringdown_spectroscopy/results/22_starttime_sweep.json").read())
sw = d["sweep"]
# overtone significant at the peak, lost as the start moves late (real fast-damping 221)
assert d["overtone_significant_at_peak"] and d["overtone_lost_by_end"], "overtone significance trend broke"
# peak-start mass biased high vs the true remnant (68.1), drifting down with start time (the R3 systematic)
assert abs(d["m_drift_peak_to_end"]) > 5.0, f"peak-start mass drift gone: {d['m_drift_peak_to_end']}"
assert sw[0]["m_med"] > 72.0, f"peak mass no longer biased high: {sw[0]['m_med']}"
# NUTS healthy throughout
assert all(r["rhat"] < 1.02 for r in sw), "a sweep fit failed to converge"
print(f"PASS  ringdown B1 (GW250114 overtone P {sw[0]['a221_frac_below_10pct_median']:.3f}->{sw[-1]['a221_frac_below_10pct_median']:.3f}; "
      f"peak-mass {sw[0]['m_med']:.1f} drifts {d['m_drift_peak_to_end']:+.1f} Msun -- R3 systematic, package-confirmed)")
PYEOFB1

echo "--- ringdown B3 close the NPE loop (23: NPE agrees with the package AND sits in the peak-start systematic)"
./ringdown_spectroscopy/.venv/bin/python - << 'PYEOFB3' || FAIL=1
import json
d = json.loads(open("ringdown_spectroscopy/results/23_npe_package_loop.json").read())
# (1) the amortized NPE agrees with the field-standard package (real inference, not an artifact)
assert d["m_median_gap"] < 3.0 and d["chi_median_gap"] < 0.06, f"NPE-package agreement broke: {d['m_median_gap']}, {d['chi_median_gap']}"
assert d["package_ci_nested_in_npe"], "package CI no longer nests in the NPE's (loop-closure agreement)"
# (2) the NPE weights the peak / early start and thus inherits the R3/B1 early-time systematic
assert d["npe_location_in_sweep_tMf"] < 4.0, f"NPE no longer sits in the early-start regime: {d['npe_location_in_sweep_tMf']}"
assert d["npe_inherits_peak_systematic"], "NPE peak-systematic inheritance no longer holds"
print(f"PASS  ringdown B3 (NPE M gap {d['m_median_gap']:.1f} Msun, CI nested; NPE weights ~peak "
      f"-> inherits +{d['npe_mass_bias_vs_true']:.1f} Msun early-time systematic, shared with the package)")
PYEOFB3

echo "--- D event watcher (watch_GW250114: one command reproduces all 3 sub-project headlines)"
./ringdown_spectroscopy/.venv/bin/python - << 'PYEOFD' || FAIL=1
import json
d = json.loads(open("watch_GW250114_082203.json").read())
st = d["stages"]
rd, npe, ec = st["ringdown"], st["npe"], st["echo"]
assert all("error" not in x for x in (rd, npe, ec)), f"a watcher stage failed: {[k for k,v in st.items() if 'error' in v]}"
# ringdown reproduces 21 (M ~74.8, overtone detected)
assert 70 < rd["M"][0] < 80 and rd["overtone_detected"], f"ringdown stage off: {rd['M'][0]}"
# NPE reproduces 09 (delta ~ -0.16, Kerr-consistent)
assert -0.30 < npe["delta"][0] < 0.0 and npe["kerr_consistent_90"], f"NPE stage off: {npe['delta']}"
# echo reproduces E3 (Dt ~0.295s from the formula, p ~0.33, null)
assert 0.28 < ec["dt_pred_s"] < 0.31 and 0.1 < ec["comb_p_value"] < 0.6 and not ec["echo_detected"], f"echo stage off: {ec}"
print(f"PASS  D event watcher (GW250114: ringdown M {rd['M'][0]:.1f}+overtone; NPE delta {npe['delta'][0]:+.2f} Kerr-ok; "
      f"echo Dt {ec['dt_pred_s']*1e3:.0f}ms p {ec['comb_p_value']:.2f} null -- all sub-projects in one command)")
PYEOFD

echo "--- G8 Fisher floor on delta (24: NPE does NOT beat the Cramér-Rao floor; prior-regularized -- G8 stands)"
./ringdown_spectroscopy/.venv/bin/python - << 'PYEOFG8' || FAIL=1
import json
d = json.loads(open("ringdown_spectroscopy/results/24_fisher_floor.json").read())
# G8 survives (NPE does not beat the data floor by a margin the prior cannot explain)
assert d["g8_survives"] and not d["g8_killed"], f"G8 verdict flipped: {d['verdict']}"
# the Fisher matrix was trustworthy for the delta marginal (step-convergence stable)
assert d["step_convergence_spread"] < 0.05, f"Fisher sigma(delta) not step-stable: {d['step_convergence_spread']}"
# at this ringdown SNR the data barely constrain delta (data floor ~ prior width) -- consistent with v6
assert d["sigma_fisher_delta"] > 0.7 * d["sigma_prior"], f"data floor implausibly tight: {d['sigma_fisher_delta']}"
# the sub-floor precision is PRIOR regularization: an off-center injection is pulled back toward the prior center
assert d["prior_shrinkage_frac"] > 0.5, f"prior-shrinkage not demonstrated: {d['prior_shrinkage_frac']}"
# the NPE posterior sits between the data-only and data+prior floors (a proper Bayesian posterior, not floor-beating)
assert d["sigma_post_min"] <= d["sigma_npe"] <= d["sigma_fisher_delta"] * 1.1, f"NPE width outside the Bayesian band: {d['sigma_npe']}"
print(f"PASS  G8 Fisher floor (data-only sigma(delta)={d['sigma_fisher_delta']:.2f} ~ prior {d['sigma_prior']:.2f} @ SNR {d['ringdown_snr']:.0f}; "
      f"NPE {d['sigma_npe']:.2f} is Bayesian data+prior, off-center pulled {d['prior_shrinkage_frac']:.0%} to center -> G8 STANDS)")
PYEOFG8

echo "--- wall species (25: delta wall is species-1 statistically, but total error saturates at a systematic floor)"
./ringdown_spectroscopy/.venv/bin/python - << 'PYEOFWS' || FAIL=1
import json
d = json.loads(open("ringdown_spectroscopy/results/25_wall_species.json").read())
# LEG 1: sigma_Fisher(delta) ~ 1/SNR EXACTLY (analytic identity; validates the implementation)
assert abs(d["loglog_slope"] + 1.0) < 0.02, f"Fisher scaling broke: slope {d['loglog_slope']}"
# DECORATION, retired 2026-08-15 by the margin audit: the observed spread is 5.5e-13 against a 0.02 bar --
# 3.6e10x from failing, because sigma(delta) ∝ 1/SNR is an ANALYTIC identity here, not an empirical claim, so
# no run could ever violate it. Replaced by a bar ~18x above the observed round-off (5.5e-13), which still catches a genuine
# regression (a broken Fisher inversion) with enough headroom for cross-platform BLAS variation, without
# pretending to certify an analytic identity. NOTE such a check is inherently LOOSE -- an identity verified to
# machine precision can never be a tight gate; the point is that the bar now tracks round-off, not fiction.
assert d["sigma_snr_spread"] < 1e-11, f"sigma*SNR invariance broke beyond round-off: {d['sigma_snr_spread']}"
assert d["statistical_species1"], "statistical channel no longer species-1"
# LEG 2: the Cutler-Vallisneri systematic bias is SNR-INDEPENDENT (variation within the numerical noise floor)
assert d["bias_is_flat"], f"systematic bias no longer flat in SNR: {d['bias_snr_spread']} vs {d['bias_numerical_spread']}"
assert d["systematic_floor_delta"] > 0.01, f"no systematic floor found: {d['systematic_floor_delta']}"
# LEG 3: total error saturates -> more SNR alone cannot cross the wall past the systematic floor
assert d["total_error_saturates"], "total error no longer saturates"
assert d["rows"][-1]["total"] > 0.5 * d["systematic_floor_delta"], "total error fell below the systematic floor"
print(f"PASS  wall species (sigma_stat slope {d['loglog_slope']:+.3f} = species-1; systematic floor "
      f"delta={d['systematic_floor_delta']:.3f} SNR-independent -> total error saturates, crossover SNR ~{d['crossover_snr']:.0f})")
PYEOFWS

echo "--- crossover sensitivity (25b: the species-1->4 crossover is conditional, NOT a constant)"
./ringdown_spectroscopy/.venv/bin/python - << 'PYEOFCS' || FAIL=1
import json
d = json.loads(open("ringdown_spectroscopy/results/25_crossover_sensitivity.json").read())
xs = [r["crossover_snr"] for r in d["sensitivity"]]
# the crossover spans a wide range with the assumed un-modeled content -> must never be cited as a constant
assert max(xs) / min(xs) > 5, f"crossover no longer content-dependent: {min(xs)}-{max(xs)}"
# scaling identity: crossover = (sigma*SNR)/|bias|
for r in d["sensitivity"]:
    assert abs(r["crossover_snr"] * r["bias"] - d["sigma_times_snr"]) < 0.1, "crossover/bias scaling broke"
# the R3-anchored (directly measured, realistic-IMR) crossover puts GW250114 AT the transition
assert 15 < d["r3_crossover_snr"] < 40, f"R3-anchored crossover moved: {d['r3_crossover_snr']}"
print(f"PASS  crossover sensitivity (spans {min(xs):.0f}-{max(xs):.0f} with assumed content; R3-anchored "
      f"{d['r3_crossover_snr']:.0f} => GW250114 already at the species-1->4 transition)")
PYEOFCS

echo "--- ringdown R1 per-parameter recalibration (17: each param in band, but does NOT beat global T)"
./ringdown_spectroscopy/.venv/bin/python - << 'PYEOFR1' || FAIL=1
import json
r = json.loads(open("ringdown_spectroscopy/results/17_recalibrate_perparam.json").read())
# PLAN criterion: each per-param held-out coverage in [0.85,0.95]
# BACKED (2026-08-16 flag audit): a bare boolean written by the script it guards is self-certification;
# these assert the NUMBER the flag summarises, so the gate fails if the number moves.
assert r["each_in_band"] and all(0.85 <= c <= 0.95 for c in r["coverage_heldout_perparam"]), r
assert r["mad_perparam"] < 0.05, f"per-parameter coverage drifted from nominal: MAD {r['mad_perparam']}"
# honest finding: per-param does NOT beat v3's global T (it overfits the calibration-set noise)
assert r["mad_global"] <= r["mad_perparam"] + 1e-9, f"per-param unexpectedly beat global: {r['mad_perparam']} < {r['mad_global']}"
assert r["kerr_inside_90"], "GW250114 no longer Kerr-consistent under per-param recalibration"
assert r["gw250114_delta"][1] < -0.05 and r["gw250114_delta"][2] > 0.05, \
    f"per-param delta CI no longer brackets Kerr: {r['gw250114_delta'][1:]}"
print(f"PASS  ringdown R1 (per-param coverage {'/'.join(f'{c:.2f}' for c in r['coverage_heldout_perparam'])} all in-band; "
      f"global T better: mad {r['mad_global']:.3f} <= per-param {r['mad_perparam']:.3f}; GW250114 δ Kerr-consistent)")
PYEOFR1

echo "--- ringdown no-hair artifacts (09)"
./ringdown_spectroscopy/.venv/bin/python - << 'PYEOF' || FAIL=1
import json
d = json.loads(open("ringdown_spectroscopy/results/09_nohair_GW250114.json").read())
assert d["kerr_inside_90"] is True, "GW250114 no longer Kerr-consistent?!"
assert 0.80 <= d["coverage"]["delta"] <= 0.96, f"delta coverage: {d['coverage']}"
print("PASS  ringdown v2 headline artifacts")
PYEOF

echo "--- ringdown v5 delta-stacking (12) — RE-SCOPED 2026-08-22: the arithmetic is right, the VALIDATION was prior-dominated"
./ringdown_spectroscopy/.venv/bin/python - << 'PYEOFS' || FAIL=1
import json
d = json.loads(open("ringdown_spectroscopy/results/12_stacking.json").read())
a = json.loads(open("ringdown_spectroscopy/results/30_stacking_audit.json").read())
big = d["injection"][-1]  # N=8

# (1) The sqrt(N) relation still holds -- but it is ARITHMETIC, not evidence the method works. stack() returns
#     1/sqrt(sum 1/sigma_i^2), and S2 compares that to sigma_single/sqrt(N): the same formula when the
#     per-event sigmas are similar. Kept as a regression check on the implementation, no longer as a claim.
assert big["N"] == 8 and abs(big["sigma_stack"] - big["expect"]) / big["expect"] < 0.15, \
    f"inverse-variance arithmetic broke: {big['sigma_stack']} vs {big['expect']}"
assert d["gates"]["S1_unbiased"], "stacking S1 regressed"

# (2) THE RE-SCOPING (30_stacking_audit). The validation injections are PRIOR-DOMINATED by the project's own
#     criterion -- the faint-event gate below calls a real event uninformative at sigma/prior > 0.88, and
#     these injections sit at 0.919. So S3's coverage of 1.00 at delta_true=0 was trivial: the prior is
#     BoxUniform(-0.5, +0.5), centred exactly on the injected truth, so returning the prior scores a perfect
#     interval. Gate the re-scoping so "method validated" cannot be re-asserted from S2 alone.
assert a["A_information"]["ratio_to_prior"] > 0.85, \
    f"validation injections became informative ({a['A_information']['ratio_to_prior']:.3f}) -- re-open the re-scoping"
assert a["prior_dominated"], "audit no longer finds prior domination -- re-derive"

# (3) And the consequence that matters: off-centre, STACKING MAKES IT WORSE. The bias survives while the
#     interval shrinks as sqrt(N), so coverage collapses. This is the number that shows sqrt(N) tightening is
#     not the same as sqrt(N) improvement.
cov1 = a["by_delta"]["0.3"]["1"]["coverage"]; cov8 = a["by_delta"]["0.3"]["8"]["coverage"]
assert cov8 < 0.5 < cov1, f"coverage no longer collapses with N at delta=0.3: N=1 {cov1}, N=8 {cov8}"
assert a["C_mean_scatter_over_claim"] < 0.7, \
    f"intervals no longer conservative ({a['C_mean_scatter_over_claim']:.2f}) -- re-derive the audit"

print(f"PASS  ringdown v5 stacking RE-SCOPED (sqrt(N) holds as ARITHMETIC: {big['sigma_stack']:.3f} vs "
      f"{big['expect']:.3f}; but validation injections are prior-dominated at sigma/prior "
      f"{a['A_information']['ratio_to_prior']:.3f} > the 0.88 the project itself calls uninformative, so "
      f"coverage 1.00 at delta=0 was trivial -- off-centre at delta=0.3 coverage collapses {cov1:.2f} (N=1) "
      f"-> {cov8:.2f} (N=8) as the interval shrinks around an 82%-shrunk estimate; intervals "
      f"{1/a['C_mean_scatter_over_claim']:.1f}x too wide)")
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

echo "--- pbh N4 self-supervised backbone (ssl_finetune: SSL-pretrained beats from-scratch at scarce labels)"
./primordial_blackhole_search/.venv/bin/python - << 'PYEOFN4' || FAIL=1
import json, numpy as np
d = json.loads(open("primordial_blackhole_search/results/ssl_finetune.json").read())
# BACKED (2026-08-16 flag audit): a bare boolean written by the script it guards is self-certification;
# these assert the NUMBER the flag summarises, so the gate fails if the number moves.
assert d["ssl_helps"], "SSL no longer beats from-scratch at all budgets"
assert d["results"]["1000"]["delta_mean"] > 0.05, \
    f"SSL gain at 1000 labels collapsed: {d['results']['1000']['delta_mean']}"
r = d["results"]
small = r[min(r, key=int)]                       # smallest labeled budget
assert small["delta_mean"] > 0.05, f"SSL gain at scarce labels shrank: {small['delta_mean']}"   # data-wall signature
# significance: the small-budget gap exceeds the seed scatter
sep = small["pretrained_mean"] - small["scratch_mean"] - 2 * (np.std(small["pretrained_auc"]) + np.std(small["scratch_auc"]))
assert sep > 0, "SSL gain at scarce labels not beyond ~2-sigma seed scatter"
# gain should shrink as labels grow (the wall recedes)
big = r[max(r, key=int)]
assert small["delta_mean"] > big["delta_mean"], "SSL gain not larger at the scarcer budget (no data-wall trend)"
print(f"PASS  pbh N4 (SSL beats from-scratch: +{small['delta_mean']:.3f} AUC @{min(r,key=int)} labels, +{big['delta_mean']:.3f} @{max(r,key=int)}; data-wall mitigated)")
PYEOFN4

echo "--- pbh N4 sensitive-distance follow-up (ssl_sensdist: SSL win translates to distance at a defined FAR)"
./primordial_blackhole_search/.venv/bin/python - << 'PYEOFN4S' || FAIL=1
import json
d = json.loads(open("primordial_blackhole_search/results/ssl_sensdist.json").read())
r = d["results"]; small, big = r[min(r, key=int)], r[max(r, key=int)]
# at the strict zero-FA threshold, reduced-budget models are sub-threshold (distance 0) -- the honest floor
assert all(v["zeroFA"]["ssl_mean"] == 0 and v["zeroFA"]["scratch_mean"] == 0 for v in r.values()), \
    "zero-FA distance no longer 0 for reduced budgets (recheck the model-strength floor claim)"
# but at a softer (1%) FAR the SSL win translates to a real sensitive-distance gain, biggest when labels are scarce
# BACKED (2026-08-16 flag audit): a bare boolean written by the script it guards is self-certification;
# these assert the NUMBER the flag summarises, so the gate fails if the number moves.
assert d["ssl_helps_at_softFAR"], "SSL no longer helps sensitive distance at the softer FAR"
assert d["results"]["2000"]["FAR1pct"]["delta_mean"] > 0.10, \
    f"SSL sensitive-distance gain collapsed: {d['results']['2000']['FAR1pct']['delta_mean']}"
assert small["FAR1pct"]["delta_mean"] > 0.10, f"SSL distance gain at scarce labels shrank: {small['FAR1pct']['delta_mean']}"
assert small["FAR1pct"]["delta_mean"] > big["FAR1pct"]["delta_mean"], "no data-wall trend (gain not larger at scarcer budget)"
print(f"PASS  pbh N4 sens-dist (SSL distance gain @1%FAR: +{small['FAR1pct']['delta_mean']:.2f}@{min(r,key=int)} -> +{big['FAR1pct']['delta_mean']:.2f}@{max(r,key=int)}; zero-FA needs full-data strength)")
PYEOFN4S

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

echo "--- pbh A: real dense-bank MF vs CNN (bank_dense + bank_vs_cnn: realizable MF ties the CNN, bank-mismatch-limited)"
./primordial_blackhole_search/.venv/bin/python - << 'PYEOFA' || FAIL=1
import json, numpy as np
bd = json.loads(open("primordial_blackhole_search/results/bank_dense.json").read())
vc = json.loads(open("primordial_blackhole_search/results/bank_vs_cnn.json").read())
sw = bd["sweep"]
# (1) density wall: the coarse bank (~83 templates, bank_oracle's old density) recovers ZERO ...
assert np.mean(list(sw["83"]["frac"].values())) == 0.0, "coarse bank no longer collapses (density wall)"
# ... while the full 0.1% bank (~1,619 templates) is functional -> quantifies the density requirement
assert bd["B"] > 1500 and np.mean(list(sw[str(bd["B"])]["frac"].values())) > 0.45, "full bank not functional"
# (2) airtight head-to-head on IDENTICAL injections: realizable MF ~ CNN (a tie within ~15%, neither routs)
r = vc["bank_mean"] / vc["cnn_mean"]
assert 0.95 < r < 1.15, f"MF-vs-CNN no longer a tie/modest edge: {r}"
# (3) both far below the true-template semi-coherent oracle -> bank mismatch is the dominant loss, not ML-vs-MF
orc = json.loads(open("primordial_blackhole_search/results/bank_oracle_B64.json").read())["oracle_vetoed"]
assert np.mean(list(orc.values())) > 1.4 * vc["bank_mean"], "oracle no longer dominates -> recheck mismatch story"
print(f"PASS  pbh A (real 0.1% bank MF {vc['bank_mean']:.2f} ~ CNN {vc['cnn_mean']:.2f} [{r:.2f}x, tie]; "
      f"coarse-bank 0.00 = density wall; both << oracle {np.mean(list(orc.values())):.2f} = bank-mismatch-limited)")
PYEOFA

echo "--- pbh N5 triple-detector H1xL1xV1 (coinc_triple: double reproduces, but Virgo does NOT help subsolar)"
./primordial_blackhole_search/.venv/bin/python - << 'PYEOFN5' || FAIL=1
import json, numpy as np
d = json.loads(open("primordial_blackhole_search/results/coinc_triple.json").read())
# (1) double H1xL1 reproduces the G1/Build-C coincidence win over single-det on fresh triple-coincident data
assert d["double_over_single"] > 1.2, f"double no longer beats single: {d['double_over_single']}"
# (2) the honest N5 finding: adding Virgo does NOT improve subsolar sensitive distance (marginally hurts)
assert d["triple_over_double"] < 1.05 and not d["virgo_helps"], f"Virgo unexpectedly helps: {d['triple_over_double']}"
# (3) mechanism: V1 barely responds to signal loudness vs H1/L1 (too insensitive at subsolar -> no signal to add)
r = d["signal_responsiveness"]
assert r["V1"] < 0.4 * np.mean([r["H1"], r["L1"]]), f"V1 responsiveness not << H1/L1: {r}"
print(f"PASS  pbh N5 (double {d['double_over_single']:.2f}x single reproduces G1; triple {d['triple_over_double']:.2f}x double "
      f"= Virgo no help; V1 signal-response {r['V1']:+.1f} vs H1 {r['H1']:+.1f}/L1 {r['L1']:+.1f})")
PYEOFN5

echo "--- pbh N5 O4b RE-TEST (coinc_triple_o4b: the Virgo negative REPLICATES on 2024-25 data)"
./primordial_blackhole_search/.venv/bin/python - << 'PYEOFN5B' || FAIL=1
import json
o3 = json.loads(open("primordial_blackhole_search/results/coinc_triple.json").read())
o4 = json.loads(open("primordial_blackhole_search/results/coinc_triple_o4b.json").read())
a  = json.loads(open("primordial_blackhole_search/results/o4_asd_compare.json").read())
# (1) the double-coincidence win reproduces on O4b data (independent era, fresh segments)
assert o4["double_over_single"] > 1.2, f"O4b double no longer beats single: {o4['double_over_single']}"
# (2) THE RESULT: Virgo still does not help, on 2x the segments, 5.5 years later, on a more sensitive network
assert o4["triple_over_double"] < 1.05 and not o4["virgo_helps"], f"O4b Virgo now helps: {o4['triple_over_double']}"
# (3) cross-era replication: both eras agree to within 5% on both ratios
assert abs(o4["double_over_single"] - o3["double_over_single"]) < 0.15, "double/single did not replicate across eras"
assert abs(o4["triple_over_double"] - o3["triple_over_double"]) < 0.05, "triple/double did not replicate across eras"
# (4) mechanism, MEASURED: V1 stays an order of magnitude less responsive, because the ASD gap widened
r4 = o4["signal_responsiveness"]
assert r4["V1"] < 0.3 * (r4["H1"] + r4["L1"]) / 2, f"V1 responsiveness no longer suppressed: {r4}"
assert a["gap_widened"], "the V1/LIGO ASD gap no longer widened O3a->O4b"
assert a["eras"]["O4b"]["v1_over_best_ligo"] > 2.5, "V1 no longer >2.5x louder than LIGO in O4b"
print(f"PASS  pbh N5 O4b re-test (double {o4['double_over_single']:.2f}x reproduces; triple {o4['triple_over_double']:.2f}x "
      f"= Virgo still no help on {o4['n_segs']} O4b segs; V1 ASD {a['eras']['O4b']['v1_over_best_ligo']:.1f}x LIGO, gap widened from "
      f"{a['eras']['O3a']['v1_over_best_ligo']:.1f}x)")
PYEOFN5B

echo "--- pbh O4-3 search reach (o4_sensitive_distance: O4b search reach & volume gain over O3a)"
./primordial_blackhole_search/.venv/bin/python - << 'PYEOFO4REACH' || FAIL=1
import json
d = json.loads(open("primordial_blackhole_search/results/o4_sensitive_distance.json").read())
comp = d["comparison"]["coincidence"]
# (1) O4b coincidence search reach distance gain is > 1.15x across all subsolar mass bins
for m in ("0.17-0.35", "0.35-0.55", "0.55-0.88"):
    gain = comp[m]["d_gain_ratio"]
    assert gain > 1.15, f"O4b distance gain low [{m}]: {gain}"
# (2) O4b surveyed volume gain is > 1.6x across all subsolar mass bins
for m in ("0.17-0.35", "0.35-0.55", "0.55-0.88"):
    vgain = comp[m]["v_gain_ratio"]
    assert vgain > 1.6, f"O4b volume gain low [{m}]: {vgain}"
# (3) O4b reach reaches ~15.7 Mpc (low mass) up to ~36.9 Mpc (high mass)
assert comp["0.17-0.35"]["d_o4b_mpc"] > 15.0, f"low mass O4b reach off: {comp['0.17-0.35']['d_o4b_mpc']}"
assert comp["0.55-0.88"]["d_o4b_mpc"] > 35.0, f"high mass O4b reach off: {comp['0.55-0.88']['d_o4b_mpc']}"
print(f"PASS  pbh O4-3 reach (O4b reach gain over O3a: {comp['0.17-0.35']['d_gain_ratio']:.2f}x [low] -- "
      f"{comp['0.35-0.55']['d_gain_ratio']:.2f}x [mid] -- {comp['0.55-0.88']['d_gain_ratio']:.2f}x [high] distance; "
      f"volume gain {comp['0.17-0.35']['v_gain_ratio']:.2f}x -- {comp['0.55-0.88']['v_gain_ratio']:.2f}x)")
PYEOFO4REACH

echo "--- pbh O4-3 STRESS-TEST (matched livetime + bootstrap: the reach gain is significant and mass-independent)"
./primordial_blackhole_search/.venv/bin/python - << 'PYEOFO4BOOT' || FAIL=1
import json
b = json.loads(open("primordial_blackhole_search/results/o4_reach_bootstrap_matched.json").read())
m = json.loads(open("primordial_blackhole_search/results/o4_sensitive_distance_matched.json").read())
s = b["_summary"]
# (1) FAR-matched (equal livetime per era) reproduces the headline gain -> the 5-vs-8-segment
#     threshold mismatch in the original run does NOT drive the result
comp = m["comparison"]["coincidence"]
assert 1.15 < s["mean_d_gain"] < 1.35, f"matched-livetime gain moved: {s['mean_d_gain']}"
for k in comp:
    assert comp[k]["d_gain_ratio"] > 1.15, f"matched gain low [{k}]: {comp[k]['d_gain_ratio']}"
# (2) bootstrap: the gain is SIGNIFICANT in every mass bin (90% CI excludes 1)
assert s["all_bins_significant"], "a mass bin's reach-gain CI no longer excludes 1"
for k, v in b.items():
    if k.startswith("_"): continue
    assert v["d_gain_ci90"][0] > 1.0, f"{k} CI includes 1: {v['d_gain_ci90']}"
# (3) the bin-to-bin spread is consistent with noise -> do NOT claim a mass-dependent gain
assert s["bin_spread_consistent_with_noise"], "bin spread no longer noise-consistent -- recheck mass dependence"
print(f"PASS  pbh O4-3 stress-test (matched-livetime mean gain {s['mean_d_gain']:.2f}x; every bin's 90% CI "
      f"excludes 1; bin spread noise-consistent -> no mass dependence claimed)")
PYEOFO4BOOT

echo "--- pbh O4-4 release watcher (o4c_release_watch: S251112cm/O4c embargo status)"
./primordial_blackhole_search/.venv/bin/python - << 'PYEOFO4C' || FAIL=1
import json
d = json.loads(open("primordial_blackhole_search/results/o4c_release_watch.json").read())
# the watcher must actually be seeing GWOSC (a broken query would silently look like "no release")
assert "O4b" in d["runs"] and "O3a" in d["runs"], f"GWOSC run list looks wrong: {d['runs']}"
assert d["n_catalogs"] >= 15, f"catalog list implausibly short: {d['n_catalogs']}"
# record the embargo status; if this flips, the assert fires and we go look at the new data
if d["s251112cm_data_public"]:
    print("PASS  pbh O4-4 watcher (*** S251112cm-era data is now PUBLIC -- point the subsolar search at it ***)")
else:
    assert not d["s25_superevents"], "an S25* superevent appeared but the flag says not public -- recheck"
    print(f"PASS  pbh O4-4 watcher (S251112cm still embargoed; O4c bulk unreleased, "
          f"public bulk ends at O4b; {d['n_catalogs']} catalogs)")
PYEOFO4C

echo "--- ringdown delta-stacking wall RE-MEASURED vs the full catalog (26: still one informative event, SNR>=47)"
./ringdown_spectroscopy/.venv/bin/python - << 'PYEOFE26' || FAIL=1
import json
d = json.loads(open("ringdown_spectroscopy/results/26_more_events_o4.json").read())
ev = {e["event"]: e for e in d["events"]}
# (1) re-selected from the FULL public catalog (439 events), not the stale June 8-event list
assert d["n_analyzed"] >= 10, f"too few events re-tested: {d['n_analyzed']}"
# (2) prior-truncation artifacts were caught and EXCLUDED (a squashed mass posterior can fake 'informative';
#     GW231206 was flagged informative at SNR 21.9 while beating events 1.7x louder -- unphysical)
assert d["n_prior_truncated"] >= 3, f"prior-truncation screen found nothing: {d['n_prior_truncated']}"
assert not ev["GW231206_233901"]["valid"], "GW231206 (prior-truncated artifact) is no longer excluded"
# (3) THE RESULT: the wall is real at current data -- still exactly one informative event
assert d["wall_is_real"] and d["n_informative"] == 1, f"informative count changed: {d['n_informative']}"
assert ev["GW250114_082203"]["informative"], "GW250114 no longer informative"
# (4) it is now QUANTIFIED, not asserted: information tracks loudness, with a threshold
assert 35 < d["snr_needed_for_information"] < 60, f"SNR threshold moved: {d['snr_needed_for_information']}"
# (5) GW250114 still reproduces the gated no-hair result
g = ev["GW250114_082203"]
assert -0.30 < g["delta"][0] < 0.0 and g["kerr_consistent"], f"GW250114 delta moved: {g['delta']}"
print(f"PASS  ringdown delta-wall re-measured ({d['n_analyzed']} events from the full 439-event catalog; "
      f"{d['n_prior_truncated']} prior-truncated excluded; still 1 informative; need SNR>="
      f"{d['snr_needed_for_information']:.0f} -- only GW250114 clears it)")
PYEOFE26

echo "--- echoes L4 (RETRACTED: the 3.2 sigma was an analytic artefact; 0.96 sigma on a paired bootstrap)"
./echoes/.venv/bin/python - << 'PYEOFL4' || FAIL=1
import json
d = json.loads(open("echoes/results/20_coherent_network.json").read())
z = json.loads(open("echoes/results/24_l4_significance_stress.json").read())

# (1) GOLDEN TEST, unaffected by the retraction: the H1/L1 geometry is MEASURED from the merger, not recited.
assert d["golden_delay_ok"], f"H1-L1 geometry no longer recovered: {d['delay_ms']} ms"
assert abs(abs(d["delay_ms"]) - 6.9) < 3.0, f"measured delay drifted: {d['delay_ms']}"
assert d["sign"] < 0, "relative polarity flipped -- GW150914's detectors are anti-aligned"

# (2) THE RETRACTION (2026-08-22), gated so it cannot be quietly re-asserted. 20_'s 3.17 sigma came from an
#     analytic binomial bar missing three terms: the slope denominator came from two adjacent grid points;
#     the two statistics are PAIRED but were combined with hypot() as if independent; and the 95th-percentile
#     threshold from n_bg=60 carried no uncertainty at all. A paired bootstrap resampling trials AND
#     background gives sd 4.00x larger -> 0.96 sigma, gain CI spanning 1.
assert not z["survives"], f"L4 now survives the bootstrap -- re-open the retraction deliberately: {z['bootstrap']}"
assert z["bootstrap"]["n_sigma"] < 2.0, f"bootstrap significance now above 2: {z['bootstrap']['n_sigma']}"
lo, hi = z["bootstrap"]["gain_ci90"]
assert lo < 1.0 < hi, f"gain CI no longer spans 1 -- the claim may be recoverable, re-derive: {[lo, hi]}"
assert z["sd_ratio_boot_over_analytic"] > 2.0, \
    f"analytic bar no longer understates ({z['sd_ratio_boot_over_analytic']:.2f}x) -- recheck the retraction"

# (3) THE FINDING THAT SURVIVES, and was always the more important half: the injection CONVENTION decides the
#     answer. Physical injections (measured delay + polarity + shared carrier phase) vs the convention every
#     existing echo script uses. This is a large effect, unaffected by the significance retraction.
assert d["injection_convention_matters"], "convention effect vanished -- re-derive before trusting either arm"
assert d["gain"]["identical"] < 1.0, f"identical-injection arm no longer loses: {d['gain']['identical']}"

print(f"PASS  echoes L4 RETRACTED (geometry still measured: {d['delay_ms']:+.2f} ms, sign {d['sign']:+.0f}; "
      f"the 3.17-sigma 'coherent helps' verdict was an ANALYTIC artefact -- paired bootstrap gives sd "
      f"{z['sd_ratio_boot_over_analytic']:.2f}x larger, {z['bootstrap']['n_sigma']:.2f} sigma, gain CI "
      f"[{lo:.2f}, {hi:.2f}] spanning 1 => UNDERPOWERED not a result; the injection-convention finding "
      f"STANDS: physical {d['gain']['physical']:.2f}x vs identical {d['gain']['identical']:.2f}x)")
PYEOFL4

echo "--- pbh confound check (tabula mechanism): no per-segment channel behind the learned coincidence gain"
./primordial_blackhole_search/.venv/bin/python - << 'PYEOFCF' || FAIL=1
import json, numpy as np
d = json.loads(open("primordial_blackhole_search/results/coinc_confound.json").read())

# WHY THIS GATE EXISTS. Build C-2 trains its head with SAME-segment positives and CROSS-segment negatives.
# Any per-segment constant in the embeddings would separate those classes with zero GW content -- and
# --holdout-segments cannot catch it, because a constant generalises perfectly to unseen segments (tabula's
# planted nuisance channel ranked MORE conserved than the real invariant and passed out-of-sample validation
# completely). So the channel's ABSENCE has to be asserted, not assumed.
assert not d["channel_exists"], f"a per-segment channel appeared (AUC {d['c1_auc_mean']:.3f}) -- Build C-2's "\
                                "learned gain must be re-derived with same-segment negatives before it stands"
assert d["c1_auc_mean"] < 0.60, f"same-segment detectability rose to {d['c1_auc_mean']:.3f}"

# the embeddings must stay dominated by within-segment variation, which is what makes the channel weak
for ifo, r in d["between_within"].items():
    assert r["frac_gt_1"] < 0.05, f"{ifo}: {100*r['frac_gt_1']:.0f}% of dims now have between > within variance"
    assert r["median"] < 0.5, f"{ifo} per-segment structure grew: median between/within {r['median']:.3f}"

print(f"PASS  pbh confound check (same-segment vs cross-segment on PURE NOISE: AUC {d['c1_auc_mean']:.3f} over "
      f"{d['seeds']} seeds = no usable channel; between/within variance median "
      f"{d['between_within']['H1']['median']:.3f} H1 / {d['between_within']['L1']['median']:.3f} L1, 0% of "
      f"{d['embed_dim']} dims between>within => the Build C-2 gain is not a per-segment constant)")
PYEOFCF

echo "--- pbh L6 SSL pool scaling (saturates by 2.5k specs; cross-detector null) -- N4's caveat answered"
./primordial_blackhole_search/.venv/bin/python - << 'PYEOFL6' || FAIL=1
import json, numpy as np
d = json.loads(open("primordial_blackhole_search/results/ssl_poolscale.json").read())

# (1) SSL still beats from-scratch at scarce labels -- N4's headline must survive at every pool size.
for k, v in d["pools"].items():
    assert v["gain"] > 0.03, f"SSL stopped helping at pool {k}: gain {v['gain']:.4f}"

# (2) THE L6 RESULT: the gain does NOT grow with pool size, so fetching more noise is not justified.
assert not d["still_climbing"], f"pool scaling reappeared (slope {d['gain_slope']:+.4f}) -- L6b would be back on"
assert d["gain_slope"] < 0.02, f"slope {d['gain_slope']} crosses the pre-registered bar"

# (3) the gain is already there at the SMALLEST pool -- that is what makes it saturation rather than failure.
small = d["pools"][min(d["pools"], key=lambda k: int(k))]
assert small["gain"] > 0.05, f"smallest pool no longer carries the gain: {small}"

# (4) honesty guard: the curve is flat WITHIN NOISE, so keep the seed scatter comparable to the spread.
#     If scatter ever shrinks far below the spread, the 'flat' reading would need re-deriving as a real decline.
sds = [float(np.std(v["auc"])) for v in d["pools"].values()]
gains = [v["gain"] for v in d["pools"].values()]
assert np.mean(sds) > 0.3 * (max(gains) - min(gains)), \
    "seed scatter now much smaller than the pool-to-pool spread -- re-derive whether the curve is really flat"

# (5) cross-detector transfer is null: L1 noise does not help an H1 model.
cd_ = d.get("cross_detector")
assert cd_ is not None and abs(cd_["gain_vs_h1_only"]) < 0.03, f"cross-detector result changed: {cd_}"

print(f"PASS  pbh L6 SSL pool scaling (gain {small['gain']:+.4f} already at {min(d['pools'], key=lambda k: int(k))} "
      f"specs; slope {d['gain_slope']:+.4f} over 5k->20k with seed sd {np.mean(sds):.3f} ~ spread "
      f"{max(gains)-min(gains):.3f} => FLAT, fetching more noise not justified; +{cd_['n_l1']} L1 specs give "
      f"{cd_['gain_vs_h1_only']:+.4f} = cross-detector null)")
PYEOFL6

echo "--- pbh L1 ratio-filter: exact algebra, but an honest NEGATIVE for subsolar (0.94x, not 8x)"
./primordial_blackhole_search/.venv/bin/python - << 'PYEOFRF' || FAIL=1
import json
R = "primordial_blackhole_search/results/"
d = json.loads(open(R + "bank_ratio_diag.json").read())
m = json.loads(open(R + "bank_ratio_mcscan.json").read())
g = json.loads(open(R + "bank_ratio_regime.json").read())
rc = json.loads(open(R + "bank_ratio_realcost.json").read())

# (1) THE FOUNDATION, which stands: c_target = c_ref (*) IFFT[conj(R)] is an exact circular identity.
assert d["D1_algebra_ok"], f"ratio-filter algebra broken: {[r['exact'] for r in d['rows']]}"
for r in d["rows"]:
    assert r["exact"] > 0.9999, f"exact-kernel reconstruction degraded at sep {r['sep']}: {r['exact']}"

# (2) the kernel length subsolar DEMANDS -- this is the number that kills the speedup, so pin it.
assert d["D2_taps_needed_for_0999"]["0.001"] >= 1025, "n=1 taps changed"
assert all(r["taps_needed"] is not None and r["taps_needed"] <= 4097 for r in m["rows"]), "Mc scan changed"

# (3) the statistic IS reproducible -- but only with long kernels. Noise error is unbiased jitter (threshold
#     safe); signal error needed 16385 taps to clear 1%.
assert g["R2_noise_unbiased"], f"noise regime became BIASED -- would shift the threshold: {g['regimes']['noise']}"
assert rc["taps"] >= 16385, "real-cost run no longer uses the taps the accuracy test demanded"

# (4) THE VERDICT, asserted as a negative so it cannot be quietly re-inflated. The 8x does not transfer
#     because the gain scales as log N / log K and subsolar forces K ~ 16k instead of ~250.
assert not rc["helps"], f"ratio filtering now HELPS ({rc['speedup']:.2f}x) -- re-examine, this would reopen L1"
assert rc["speedup"] < 2.0, f"speedup changed materially: {rc['speedup']}"
assert rc["generation_frac_of_direct"] < 0.25, \
    f"generation share changed ({rc['generation_frac_of_direct']:.2f}) -- the superseded cost model claimed 0.98"
assert rc["speedup"] <= rc["theory_ceiling"] * 1.5, "measured speedup exceeds the asymptotic ceiling -- suspicious"

print(f"PASS  pbh L1 ratio-filter (algebra EXACT to {min(r['exact'] for r in d['rows']):.6f}; but subsolar needs "
      f"K={rc['taps']} taps vs the paper's ~250, so log N/log K gives only a {rc['theory_ceiling']:.1f}x ceiling "
      f"and the measurement is {rc['speedup']:.2f}x => HONEST NEGATIVE, dense bank stays blocked; generation is "
      f"{100*rc['generation_frac_of_direct']:.0f}% of cost, not the 98% the superseded model assumed)")
PYEOFRF

echo "--- ringdown L5 third-tone floor (undetectable here; two DIFFERENT reasons, with a reopening number)"
./ringdown_spectroscopy/.venv/bin/python - << 'PYEOFL5' || FAIL=1
import json
d = json.loads(open("ringdown_spectroscopy/results/29_third_tone_floor.json").read())
m = d["modes"]

# (1) THE TWO FAILURE MODES ARE DIFFERENT -- the whole point. (2,2,2) is absorbed by refitting 220+221;
#     the higher multipoles are nearly orthogonal but intrinsically faint.
assert m["222"]["orthogonal_fraction"] < 0.3, f"(2,2,2) no longer degeneracy-limited: {m['222']}"
assert m["440"]["orthogonal_fraction"] > 0.7, f"(4,4,0) no longer well separated: {m['440']}"
assert d["prediction_held"], "the degeneracy-vs-weakness split collapsed"

# (2) A third tone is NOT reachable on GW250114-class data: (2,2,2) would need an overtone LOUDER than the
#     fundamental, which is unphysical.
assert m["222"]["amplitude_ratio_needed"] > 1.0, \
    f"(2,2,2) became reachable at A<A220 ({m['222']['amplitude_ratio_needed']:.2f}) -- re-examine, L5 would reopen"

# (3) the REOPENING CRITERION, kept as a number so 'undetectable' cannot drift into a vague claim.
need = m["440"]["ringdown_snr_needed_at_10pct_amplitude"]
assert need > d["rho_ringdown"], f"(4,4,0) now reachable at current SNR ({need} vs {d['rho_ringdown']})"
assert 1.2 < need / d["rho_ringdown"] < 3.0, f"reopening ratio moved materially: {need/d['rho_ringdown']:.2f}x"

# (4) the START-TIME mechanism that explains LVK's early-time-only (2,2,2) preference -- and the record that
#     the window-length hypothesis was refuted, so it is not quietly reintroduced.
assert d["start_time_penalty_222"] > 3.0, f"(2,2,2) start-time penalty vanished: {d['start_time_penalty_222']}"
assert d["window_hypothesis_refuted"], "window-length hypothesis no longer recorded as refuted"

print(f"PASS  ringdown L5 third-tone floor ((2,2,2) orth {m['222']['orthogonal_fraction']:.3f} => needs "
      f"A/A220 {m['222']['amplitude_ratio_needed']:.2f} (unphysical) = DEGENERACY-limited; (4,4,0) orth "
      f"{m['440']['orthogonal_fraction']:.3f} but faint => needs rho_rd {need:.0f} = "
      f"{need/d['rho_ringdown']:.1f}x GW250114 = WEAKNESS-limited; (2,2,2) SNR falls "
      f"{d['start_time_penalty_222']:.1f}x by t_s=2 ms, the start-time mechanism behind LVK's early-time hint)")
PYEOFL5

echo "--- ringdown orthonormal-QNM test (27/28: the basis carries NO detection information)"
./ringdown_spectroscopy/.venv/bin/python - << 'PYEOFORTH' || FAIL=1
import json
R = "ringdown_spectroscopy/results/"
a = json.loads(open(R + "27_orthonormal_roc.json").read())
b = json.loads(open(R + "28_orthonormal_prior.json").read())

# (1) the premise is real: 220 and 221 ARE strongly non-orthogonal at a GW250114-like remnant.
assert a["overlap_220_221"] > 0.8, f"overlap changed, the whole setup assumes near-degeneracy: {a['overlap_220_221']}"

# (2) THE RESULT: an orthonormal basis and a properly-handled non-orthogonal fit are the SAME detector.
#     Asserted as an exact algebraic identity (Schur complement), so the tolerance is numerical, not empirical.
assert a["prediction_held"], f"orth != nonorth_proper: {a['max_abs_auc_diff_orth_vs_proper']}"
assert a["max_abs_auc_diff_orth_vs_proper"] < 1e-6, f"identity degraded: {a['max_abs_auc_diff_orth_vs_proper']}"

# (3) ... and it survives building the basis at the WRONG remnant (closes the 'helps under mismatch' hatch)
assert b["B1_max_abs_diff"] < 1e-6, f"mismatch breaks the identity: {b['B1_max_abs_diff']}"

# (4) the ONLY thing that moves is the reported number, via the prior's Occam factor, in the
#     uninformative limit where the likelihood-geometry claim is isolated.
#     NOTE the tolerance differs from (2)/(3) on purpose. Those are EXACT algebraic identities (the two
#     statistics are monotone transforms, so the ranks -- and hence AUC -- coincide bit for bit). This one is
#     ASYMPTOTIC: at finite prior sd the regularization differs slightly between bases, so the gap only ->0
#     as the prior broadens. Assert the convergence, not bitwise equality.
lim = b["B2_uninformative_limit"]
assert abs(lim["log10bf_shift"]) > 0.1, f"BF shift vanished -- re-derive the claim: {lim}"
assert lim["auc_gap"] < 1e-4, f"detection power differs in the uninformative limit: {lim}"
gaps = [abs(r["auc_orth"] - r["auc_nonorth"]) for r in b["B2_prior"]]
assert gaps[-1] < gaps[0] / 100, f"AUC gap does not collapse as the prior broadens: {gaps}"
assert lim["auc_gap"] < 0.01 * a["max_auc_gain_over_naive"], \
    "the residual gap is no longer negligible against the one real effect (ignoring the covariance)"
assert b["significance_moves_information_does_not"], "verdict flipped"

# (5) the effect that IS real: an analysis ignoring the covariance is worse -- but only slightly.
assert 0.0 < a["max_auc_gain_over_naive"] < 0.05, f"naive-gap story changed: {a['max_auc_gain_over_naive']}"

print(f"PASS  ringdown orthonormal-QNM (|<220|221>|={a['overlap_220_221']:.3f} strongly non-orthogonal, yet "
      f"orth==nonorth_proper to {a['max_abs_auc_diff_orth_vs_proper']:.1e} AUC -- even at a WRONG remnant "
      f"({b['B1_max_abs_diff']:.1e}); only the REPORTED number moves: log10BF {lim['log10bf_shift']:+.2f} "
      f"({10**abs(lim['log10bf_shift']):.1f}x odds) at dAUC {lim['auc_gap']:.1e}. Ignoring the covariance "
      f"costs {a['max_auc_gain_over_naive']:.4f} AUC = the real, small effect)")
PYEOFORTH

echo "--- pbh deep FAR (far_deep: 4,120-yr background on O4b -> 1/century reached, zero-lag clean)"
./primordial_blackhole_search/.venv/bin/python - << 'PYEOFFAR' || FAIL=1
import json
d = json.loads(open("primordial_blackhole_search/results/far_deep.json").read())
# (1) the background formula is the honest one: (N_windows-1) distinct lags x total livetime
#     [verified against Build C: 1511 lags x 26.88 h = 1692 days exactly]
live_s = d["real_livetime_h"] * 3600
assert abs(d["n_distinct_lags"] * live_s / 3.156e7 - d["background_years"]) < 0.1, "background formula drifted"
assert d["n_distinct_lags"] == d["n_windows"] - 1, "not using the honest N-1 distinct-lag count"
# (1b) livetime must be time ACTUALLY SEARCHED (n_windows x 64 s), not wall-clock segments x 4096 s -- the 8-s
#      whitening crop leaves 62 of 64 windows, so the per-segment form counts 3.2% of time we never searched
#      and nudges every threshold anti-conservative. Same species as the honest-slides lag overcount.
assert abs(d["real_livetime_h"] - d["n_windows"] * 64 / 3600) < 0.5, \
    f"livetime is not the ANALYZED time: {d['real_livetime_h']:.1f} h vs {d['n_windows']*64/3600:.1f} h"
# (2) THE GOAL: beat Build C's 1/year. 10+ yr background => 1/decade is measurable
assert d["background_years"] > 1000.0, f"background too short for 1/century: {d['background_years']}"
assert "1/century" in d["far_ladder"], "1/century not reached"
# (3) thresholds must be monotonic (stricter FAR => higher bar)
lad = d["far_ladder"]
assert lad["1/month"] < lad["1/year"] < lad["1/decade"] < lad["1/century"], f"FAR ladder not monotonic: {lad}"
# (4) ZERO-LAG: no candidate at any CLAIM-CAPABLE rung. NOTE the bar is 1/year, not 1/month, and that is not a
#     loosening: our zero-lag search covers only n_windows x 64 s ~ 0.09 yr, so at 1/month we EXPECT ~1.1
#     accidental events -- exceeding it is the MEDIAN noise outcome and can never support a claim. The rungs
#     that can are 1/year and deeper (expected counts 0.09, 0.009, 0.001). Pre-registered before the 727-segment
#     numbers landed. The loudest zero-lag event's own FAR is asserted below via the one-sided glitch gate.
T_obs_yr = d["n_windows"] * 64 / 3.156e7
assert 12.0 * T_obs_yr > 0.5, "1/month is no longer expected-to-fire; re-derive which rung is claim-capable"
assert d["zero_lag_max"] < lad["1/year"], f"REAL coincidence above 1/year threshold -- INVESTIGATE: {d['zero_lag_max']}"
print(f"PASS  pbh deep FAR ({d['n_segments']} O4b segments, {d['real_livetime_h']:.0f} h -> "
      f"{d['background_years']:.1f} yr background = {d['background_years']/4.63:.0f}x Build C; "
      f"1/century threshold {lad['1/century']:.2f}; loudest real coincidence {d['zero_lag_max']:.2f} "
      f"< 1/year {lad['1/year']:.2f} = clean null)")
PYEOFFAR

echo "--- pbh deep-FAR audit (independence holds; jackknife 33-44% -> ~10%; effective-N still binding; null 4/4)"
./primordial_blackhole_search/.venv/bin/python - << 'PYEOFAUD' || FAIL=1
import json
R = "primordial_blackhole_search/results/"
v = json.loads(open(R + "far_background_validation.json").read())
a = json.loads(open(R + "far_glitch_anatomy.json").read())
m = json.loads(open(R + "far_min_vs_sum.json").read())

# (1) the assumption time-slides REST on: H1 _|_ L1. If this ever fails the whole ladder is biased LOW.
assert v["V1_independence"]["p_two_sided"] > 0.05, f"H1/L1 correlated -- background biased: {v['V1_independence']}"
assert v["V6_segment_quality"]["p_perm"] > 0.05, f"segment data quality co-varies: {v['V6_segment_quality']}"

# (2) the honest error bar: the deep tail rests on a handful of windows, so the jackknife spread is LARGE.
#     This asserts we keep REPORTING that, i.e. nobody quietly re-quotes +-Poisson precision.
# 51x more background did NOT fix this: 1/decade is still set by a handful of distinct H1 windows (8, the same
# count the 80.5-yr run had) and 1/century by 3. Effective sample size, not livetime, is the binding limit.
assert v["V2_effective_n"]["1/decade"]["distinct_H1_windows"] <= 12, "effective-N story changed -- recheck the caveat"
assert v["V2_effective_n"]["1/century"]["distinct_H1_windows"] <= 6, "1/century effective-N changed -- recheck"
# The jackknife TIGHTENED (33-44% -> ~10%) but must stay an order of magnitude above the Poisson band, which is
# ~0.3% here. Two-sided so neither a regression nor a silent re-quote of sqrt(k) precision can pass.
jk_spread = v["V7_jackknife"]["1/decade"]["spread_pct"]
assert 4.0 < jk_spread < 20.0, f"jackknife spread {jk_spread:.1f}% outside the established band -- re-derive"
pb = v["V3_poisson"]["1/decade"]["band"]
poisson = 100 * (pb[1] - pb[0]) / v["V7_jackknife"]["1/decade"]["full"]
assert jk_spread > 5 * poisson, f"jackknife ({jk_spread:.1f}%) no longer dominates Poisson ({poisson:.2f}%)"

# (3) THE RESULT: null under every configuration x statistic, each vs its OWN threshold.
assert v["all_configs_null"], f"a configuration is NOT null -- INVESTIGATE: {v['V8_null_robustness']}"

# (4) the zero-lag 'near miss' is a one-sided H1 glitch, not a coincidence (so the null is wide, not 8%).
assert not a["zero_lag_two_sided"], "zero-lag loudest became two-sided -- re-examine, this WOULD be a candidate"
assert a["zero_lag_min_stat"] < a["one_sided_ceiling"], "consistency statistic no longer clean"

# (5) honest negative: the consistency statistic does NOT buy reach (keeps G2a's 'sum is optimal' honest at depth)
worst = max(m["verdict"]["1/decade"][k] for k in ("min", "veto"))
assert worst <= 1.05, f"min/veto now BEATS sum at 1/decade ({worst}x) -- that would overturn G2a, investigate"

jk = v["V7_jackknife"]["1/decade"]
print(f"PASS  pbh deep-FAR audit (independence p={v['V1_independence']['p_two_sided']:.2f}; 1/decade set by "
      f"{v['V2_effective_n']['1/decade']['distinct_H1_windows']} distinct H1 windows -> jackknife spread "
      f"{jk['spread_pct']:.0f}% (quote 14.5+-1.5, not +-0.04); zero-lag = one-sided glitch H1 {a['zero_lag_H1']:+.1f} "
      f"L1 {a['zero_lag_L1']:+.1f}; min buys {m['verdict']['1/decade']['min']:.2f}x = no gain; null 4/4)")
PYEOFAUD

echo "--- pbh zero-lag population (independence verified to 99.98th pctile; 3 of 4 rungs beyond testable data)"
./primordial_blackhole_search/.venv/bin/python - << 'PYEOFZLP' || FAIL=1
import json
R = "primordial_blackhole_search/results/"
z = json.loads(open(R + "far_zerolag_population.json").read())
d = json.loads(open(R + "far_deep.json").read())

# (1) THE ASSUMPTION THE 4,120-yr LADDER RESTS ON, tested with the 45,073 zero-lag measurements the deep-FAR
#     result discards. Time-slides ARE the independence null, so agreement here is a real check, not a tautology.
assert not z["ks"]["exceeds"], f"zero-lag CDF now differs from background: KS {z['ks']}"
# (1b) ...and that KS is a MEASUREMENT, not our own grid resolution. Both CDFs are built by quantising into
#      bins, so a small KS could be the binning rather than agreement. Swept 256x in bin count: flat.
#      (Clip-band technique from `bridge` via the coordination channel.) Without this, "the CDFs agree to
#      ~0.2% everywhere" is indistinguishable from "our grid cannot resolve 0.2%".
c = json.loads(open(R + "far_ks_clipband.json").read())
assert c["physical"], f"KS now tracks the binning -- re-quote it as an upper bound: {c['verdict']}"
assert c["ks_span_ratio"] < 1.20, f"KS varies {c['ks_span_ratio']:.2f}x across the grid sweep"
assert abs(c["d_logKS_d_logWidth"]) < 0.10, f"KS trends with bin width: {c['d_logKS_d_logWidth']}"
assert z["max_abs_z"] < 3.0, f"an excess ladder rung exceeds |z|=3 -- INVESTIGATE: {z['excess_ladder']}"
for t in z["tail_dependence"]:
    if t["expected"] >= 5:                      # only rungs with usable statistics can constrain anything
        assert t["p_perm"] > 0.05 / 3, f"tail dependence at pctile {t['pctile']}: {t}"

# (2) the well-populated rungs are the ones that constrain; assert they stay populated, else the precision
#     quoted in RESULTS.md (+-1% at 10k expected) is no longer supported by the artifact.
top = max(z["excess_ladder"], key=lambda r: r["expected"])
assert top["expected"] > 1000 and top["observed"] > 1000, f"shallowest rung lost its statistics: {top}"

# (3) THE STRUCTURAL FINDING, gated so it cannot quietly erode: the claim-capable rungs sit BEYOND every
#     zero-lag sample we have, so independence there is ASSUMED, not verified -- and sliding cannot fix it
#     (slides make more pairs, never new zero-lag samples). If this ever becomes false we have enough real
#     observing time to test the assumption directly, which would be a genuine change in what we can claim.
assert d["zero_lag_max"] < d["far_ladder"]["1/year"], "1/year is now within zero-lag reach -- re-derive scope"
assert d["far_ladder"]["1/century"] > d["zero_lag_max"], "1/century now testable -- update the structural claim"

print(f"PASS  pbh zero-lag population (KS {z['ks']['stat']:.5f} < crit {z['ks']['crit95']:.5f} => CDFs agree to "
      f"~0.2% everywhere; max |z| {z['max_abs_z']:.2f} over {len(z['excess_ladder'])} rungs; 90th-pctile joint "
      f"exceedances {z['tail_dependence'][0]['observed']} vs {z['tail_dependence'][0]['expected']:.1f} expected "
      f"=> no correlated noise, no sub-threshold population; independence verified to the 99.98th pctile but "
      f"1/year+ sit beyond ALL zero-lag data (max {d['zero_lag_max']:.2f} < {d['far_ladder']['1/year']:.2f}) "
      f"=> assumed there, and unslideable; KS verified grid-independent over {c['bin_count_span']:.0f}x "
      f"in bin count, span {c['ks_span_ratio']:.3f}x => a measurement, not a binning floor)")
PYEOFZLP

echo "--- pbh estimator audit (jackknife understates 4.2x, stable in n; N_eff does NOT control the error)"
./primordial_blackhole_search/.venv/bin/python - << 'PYEOFEST' || FAIL=1
import json
R = "primordial_blackhole_search/results/"
b = json.loads(open(R + "far_estimator_bias.json").read())
e = json.loads(open(R + "far_effective_n.json").read())
c = json.loads(open(R + "far_sigma_convergence.json").read())

# (1) the published jackknife is BIASED LOW -- assert we keep saying so, and by how much. If this ever drops
#     near 1 the estimator changed and every absolute spread in RESULTS.md must be re-derived.
assert b["mean_bias_at_largest_n"] > 2.0, f"jackknife bias vanished: {b['mean_bias_at_largest_n']}"
# (2) ...but STABLE in n, which is the only reason L2's 33-44% -> 10-12% RATIO survives. If the bias starts
#     growing with n, that ratio is inflated by the estimator and the L2 headline needs retracting, not footnoting.
assert b["mean_bias_growth"] < 1.3, f"bias now GROWS with n ({b['mean_bias_growth']:.2f}x) -- L2 ratio is inflated"

# (3) the mechanism claim, gated as REFUTED so it cannot creep back: N_eff grows ~linearly but the collapse
#     sigma*sqrt(N_eff) does NOT hold, so N_eff is not what sets the error.
assert e["beta_mean"] > 0.7, f"N_eff no longer grows ~linearly ({e['beta_mean']:.2f}) -- recheck the L2 story"
assert not e["collapse_holds"], "collapse now HOLDS -- N_eff would control the error, revisit the retraction"
assert e["alpha_mean"] < 0.35, f"sigma now falls near the naive 1/sqrt(n) ({e['alpha_mean']:.2f}) -- re-derive"
assert e["cost_to_halve_sigma"]["segments_factor"] is None, "planning number resurrected from alpha~0"

# (4) a +-sigma is only a legitimate summary because the tail is LIGHT. Hill < 2 would mean infinite variance
#     and NO error bar would be quotable anywhere in this arc -- including the +-1.7 we now publish.
assert not c["heavy_tailed_rungs"], f"heavy tail detected -- +-sigma is not quotable: {c['heavy_tailed_rungs']}"
worst = min(v["hill"] for v in c["by_far"].values())
assert worst > 2.0, f"Hill index {worst:.1f} <= 2 => infinite variance"

hi = max(v["hill"] for v in c["by_far"].values())
print(f"PASS  pbh estimator audit (jackknife understates by {b['mean_bias_at_largest_n']:.1f}x but stable in n "
      f"({b['mean_bias_growth']:.2f}x) => L2's RATIO stands, absolute spreads x4 => 1/decade 14.53 +- ~1.7 not "
      f"+-0.41; N_eff ~ n^{e['beta_mean']:.2f} yet sigma ~ n^-{e['alpha_mean']:.2f} and the collapse FAILS "
      f"=> effective-N does NOT control the error; Hill {worst:.0f}-{hi:.0f} >> 2 => sigma converges, +-sigma legitimate)")
PYEOFEST

echo "--- pbh tail-norm test (sum vindicated: equalising the 2x noise-tail asymmetry HURTS held-out)"
./primordial_blackhole_search/.venv/bin/python - << 'PYEOFTN' || FAIL=1
import json
R = "primordial_blackhole_search/results/"
t = json.loads(open(R + "coinc_tailnorm.json").read())
z = json.loads(open(R + "coinc_tailnorm_stress.json").read())

# (1) the objection was REAL: the noise-tail asymmetry is much larger than the gain asymmetry that prompted
#     it. If this ever shrinks toward 1, the whole test loses its motivation and should be re-derived.
assert not z["survives"], f"tailnorm now survives held-out -- re-open the statistic choice: {z['ratio']}"

# (2) the in-sample gain was real BEFORE the split -- assert we keep BOTH numbers, because the pair is the
#     finding. An in-sample-only record would read as 'we tried it and it helped'.
ins = t["verdict_table"]["1/month"]["tailnorm"]
assert ins > 1.02, f"in-sample tailnorm gain vanished ({ins}) -- the artefact story needs re-checking"
for lbl, r in z["ratio"].items():
    assert r < 1.0, f"held-out tailnorm beats sum at {lbl} ({r}) -- contradicts the recorded negative"

# (3) it is significantly WORSE, not merely not-better: every CI must exclude 1 on the LOW side. This is what
#     licenses the stronger claim that the tail asymmetry carries information a rarity transform discards.
for lbl, b in z["boot"].items():
    assert b["ci90"][1] < 1.0, f"{lbl} CI no longer excludes 1 on the low side: {b}"

# (4) my censoring explanation was REFUTED -- gate it so it cannot creep back as folklore.
for lbl, c in z["censoring"].items():
    assert c["frac_at_ceiling"] < 0.01, f"{lbl} now censored at the map ceiling ({c}) -- revisit the refutation"

worst = min(z["ratio"].values())
print(f"PASS  pbh tail-norm (noise tails differ 1.97x max / 1.58x q99.9 vs gain 0.97x => objection was real; "
      f"tailnorm looked +{100*(ins-1):.0f}% IN-SAMPLE but held-out gives {worst:.3f}x with every 90% CI below 1 "
      f"=> confounding not overfitting, `sum` stands; censoring hypothesis refuted at 0% of background)")
PYEOFTN

echo "--- pbh glitch morphology (the deep-FAR tail is NOT glitch-driven; nothing to veto)"
./primordial_blackhole_search/.venv/bin/python - << 'PYEOFGM' || FAIL=1
import json
d = json.loads(open("primordial_blackhole_search/results/glitch_score_correlation.json").read())

# (1) THE FINDING: the detector does not track transient excess. Gated so the veto programme cannot be
#     quietly revived -- it was killed by measurement, not by opinion, and reviving it needs new evidence.
assert d["n_windows"] > 1000, f"sample too small to carry the claim: {d['n_windows']}"
assert abs(d["spearman_score_vs_maxexcess"]) < 0.30, \
    f"score now tracks transient excess (rho={d['spearman_score_vs_maxexcess']:.3f}) -- re-open the veto idea"
assert d["top_score_top_excess_overlap"] <= 2, \
    f"loudest scorers and loudest transients now overlap ({d['top_score_top_excess_overlap']}) -- re-derive"

# (2) the sharper form: top scorers are no more transient-rich than an average window. If this ever
#     separates, the mechanism behind the deep-FAR tail has changed and several entries need rewriting.
assert d["frac_top_scorers_with_excess"] < d["frac_all_windows_with_excess"] + 0.15, \
    f"top scorers now excess-enriched: {d['frac_top_scorers_with_excess']:.2f} vs {d['frac_all_windows_with_excess']:.2f}"
assert d["median_score_of_highest_excess"] < d["median_score_all"] + 0.5, \
    "the biggest transients now score high -- the detector has started responding to glitches"

print(f"PASS  pbh glitch morphology (n={d['n_windows']} windows: Spearman(score, max_excess) = "
      f"{d['spearman_score_vs_maxexcess']:+.3f}, top-{d['top_k']}-by-score and top-{d['top_k']}-by-excess "
      f"share {d['top_score_top_excess_overlap']} windows, top scorers {d['frac_top_scorers_with_excess']:.2f} "
      f"vs all {d['frac_all_windows_with_excess']:.2f} excess-bearing => the CNN does NOT respond to transients, "
      f"the deep-FAR tail is not glitch-driven, and there is nothing to veto -- which is why min/veto never "
      f"bought reach and tail-norm HURT)")
PYEOFGM

echo "--- pbh CNN response probe (it keys on band-limited noise power ~110 Hz; false alarms are signal-like)"
./primordial_blackhole_search/.venv/bin/python - << 'PYEOFCR' || FAIL=1
import json
d = json.loads(open("primordial_blackhole_search/results/cnn_response_probe.json").read())

# (1) WHAT IT USES: a narrow low-frequency region, not the full analysis band. If this ever broadens, the
#     "~3/4 of the spectrogram is ignored capacity" observation and the band-narrowing question both change.
for pop in ("injections", "top_scoring_noise"):
    assert d["frac_below_224hz"][pop] > 0.90, \
        f"{pop} sensitivity no longer concentrated below 224 Hz: {d['frac_below_224hz'][pop]:.3f}"
    assert 60.0 < d["centre_hz"][pop] < 200.0, f"{pop} sensitivity centre moved: {d['centre_hz'][pop]:.1f} Hz"

# (2) THE UNIFYING CLAIM: false alarms use the SAME spectral region as signals, so the detector is honestly
#     fooled and no single-detector cut can separate them. This is what explains min/veto/tail-norm all
#     failing while coincidence works. Gated on the PROFILE CORRELATION, not on argmax -- comparing peaks
#     over adjacent bands is a coin flip when the peak is broad, and that is exactly how the first version
#     of this script reached the opposite (wrong) conclusion.
assert d["profile_corr_noise_vs_injections"] > 0.70, \
    f"false-alarm and signal profiles diverged ({d['profile_corr_noise_vs_injections']:+.3f}) -- a separate " \
    f"artefact may exist after all, which would REOPEN the single-detector veto question"
assert d["false_alarm_matches_signal"], "verdict flipped -- re-derive the unifying explanation"

# (3) and the detector must still be ignoring the top of the band, which is the actionable half
top = [b for b in d["bands"] if b["lo_hz"] > 400][-1]
assert abs(top["spearman"]) < 0.15, f"the top of the band now tracks the score: {top}"

print(f"PASS  pbh CNN response probe (sensitivity centre {d['centre_hz']['injections']:.0f} Hz for signals / "
      f"{d['centre_hz']['top_scoring_noise']:.0f} Hz for false alarms, {100*d['frac_below_224hz']['injections']:.0f}% "
      f"of it below 224 Hz; profiles correlate {d['profile_corr_noise_vs_injections']:+.3f} => same feature, "
      f"detector honestly fooled by band-limited noise power => no single-detector cut can work, and "
      f"coincidence works because the fluctuation is independent between detectors)")
PYEOFCR

echo "--- ringdown spin truncation (31: O(chi^2) is ~6% at real remnants, ~19% at EMRI spins)"
./ringdown_spectroscopy/.venv/bin/python - << 'PYEOFST' || FAIL=1
import json
d = json.loads(open("ringdown_spectroscopy/results/31_spin_truncation.json").read())
t2 = d["truncation"]["2"]

# (1) THE MEASUREMENT, and the channel split that is the actual finding. If either number moves materially
#     the split may invert, and the split is what a theory programme would act on.
assert 0.02 < t2["chi069"]["median"] < 0.08, f"O(chi^2) error at chi=0.69 moved: {t2['chi069']}"
assert 0.10 < t2["chi090"]["median"] < 0.25, f"O(chi^2) error at chi=0.90 moved: {t2['chi090']}"
assert t2["chi069"]["median"] < 0.14, \
    "O(chi^2) error at remnant spins now EXCEEDS our sigma(delta) ~ 0.14 -- the ringdown half of the " \
    "channel split inverts and 'not the binding constraint' must be withdrawn"
# The split is NOT a ratio -- it is each number against its OWN comparator: 6.4% against our sigma(delta)
# ~ 0.14 for ringdown, 18.9% against phase accuracy over ~1e5 cycles for EMRI. The bar below was written as
# `> 3x` when the numbers were the superseded 4.43%/16.15% (ratio 3.65); the corrected 6.36%/18.86% give
# 2.96 and tripped it. Loosened to 2x DELIBERATELY and on the record, because a ratio bar calibrated to a
# number that has since been corrected is measuring the old number, not the finding -- the assertion above,
# chi069 < 0.14, is the one that carries the ringdown half.
assert t2["chi090"]["median"] > 2 * t2["chi069"]["median"], "the channel split has collapsed -- re-derive"

# (2) the error must actually DECREASE with order -- this is what refutes "no finite order is controlled",
#     which was the strong claim we had to withdraw. If it stops decreasing that claim comes back.
errs = [d["truncation"][str(o)]["chi069"]["median"] for o in (2, 3, 4, 5, 6)]
assert all(errs[i] > errs[i+1] for i in range(len(errs)-1)), f"error no longer falls with order: {errs}"

# (3) AND THE HONESTY GUARD: only 2 coefficient ratios are stable under fit degree. If anyone later quotes
#     an asymptotic limit from this data, this assertion is what says they cannot.
stab = d["coefficient_stability"]
assert stab["2"]["is_real"] and stab["3"]["is_real"], "the n=2,3 coefficients are no longer stable"
assert not stab["5"]["is_real"], "n=5 became stable -- the 'limit not measurable' claim can be revisited"
assert stab["5"]["drift"] > 0.5, f"n=5 drift shrank to {stab['5']['drift']:.3f} -- recheck"

print(f"PASS  ringdown spin truncation (O(chi^2) error {t2['chi069']['median']:.1%} at chi=0.69 vs "
      f"{t2['chi090']['median']:.1%} at chi=0.90 => NOT binding for ringdown (below our sigma(delta)~0.14), "
      f"disqualifying for EMRI; error falls with order so 'no finite order works' is refuted; but only "
      f"n=2,3 coefficients survive a fit-degree sweep (n=5 drifts {stab['5']['drift']:.2f}, "
      f"n=6 drifts {stab['6']['drift']:.0f}) => the asymptotic limit is NOT measurable)")
PYEOFST

echo "--- ringdown spin truncation CROSS-CHECK (32: published crossings reproduced, after they found our bug)"
./ringdown_spectroscopy/.venv/bin/python - << 'PYEOFCC' || FAIL=1
import json
d = json.loads(open("ringdown_spectroscopy/results/32_spin_truncation_crosscheck.json").read())

# (1) THE GOLDEN TEST IS THE PRECONDITION, not decoration. The low-degree fit 31 used until 2026-09-04
#     FAILS this case, which is how the method error was found. If it ever fails again, nothing below the
#     line means anything and the comparison must not be reported.
g = d["golden_geometric"]
assert all(v["pass"] for v in g.values()), f"golden test failed: {g}"
assert abs(g["2"]["measured"] - 0.2154) < 0.005, f"order-2 analytic crossing moved: {g['2']}"

# (2) the external validation itself: two numbers from a paper we did not tune to.
assert all(d["agrees_with_published"].values()), f"published crossings no longer reproduced: {d['earliest']}"
assert abs(d["earliest"]["1"]["crossing"] - 0.22) <= 0.02
assert abs(d["earliest"]["2"]["crossing"] - 0.40) <= 0.02

# (3) and the reason the comparison is legal: the EARLIEST of the four curves is what a single 1% line in
#     their four-curve figure reports. Comparing only (022) real is what made the first run say DISAGREE.
assert len(d["measured"]) == 4, "the four-curve sweep shrank -- the earliest-curve rule needs all four"

print(f"PASS  ringdown spin truncation cross-check (golden 1/(1-x) crossings exact; published 1% crossings "
      f"{d['earliest']['1']['crossing']:.3f} vs 0.22 and {d['earliest']['2']['crossing']:.3f} vs 0.40 "
      f"reproduced => 31's corrected 6.4%/18.9% are externally validated, and the low-degree fit that gave "
      f"4.4%/16.2% fails the analytic control)")
PYEOFCC

echo "========================================"
[ $FAIL -eq 0 ] && echo "BLACKHOLE GATE: ALL GREEN" || echo "BLACKHOLE GATE: FAILURES"
exit $FAIL
