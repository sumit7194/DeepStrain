#!/usr/bin/env python
"""Follow-up D — the EVENT WATCHER: turn the stack into a standing instrument for new loud events.

One command runs the whole black-hole pipeline on an event and emits a one-page report:
  1. ringdown remnant + overtone     (ringdown package, venv311)   -> 220+221 M, chi, A221 significance
  2. no-hair test                    (amortized NPE, ringdown .venv) -> M, chi, delta, Kerr consistency
  3. echo Δt prediction + search     (echoes venv)                 -> Kerr-tortoise Δt(M_f,chi_f) + comb p-value
Amortized NPE + cached strain -> seconds per event; O4b/O5 will supply GW250114-class events (the only ones
loud enough to measure delta -- v6 delta-threshold wall). Reference run: GW250114.

Each stage runs in its own venv via subprocess (three Python environments); a stage that fails is reported as
such and does not sink the others. Usage:  python3 watch_event.py GW250114_082203  [--skip-npe]
"""
import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RD, EC = ROOT / "ringdown_spectroscopy", ROOT / "echoes"

# verified event registry (sources in each field): sky loc + detector-frame remnant (M_f, chi_f)
EVENTS = {
    "GW250114_082203": dict(gps=1420878141.2362, ra=2.35, dec=0.22, psi=1.37, M_f=68.1, chi_f=0.68,
                            band=[30.0, 350.0]),   # LVK max-L (arXiv:2601.05734) + remnant (arXiv:2509.08099)
    "GW150914": dict(gps=1126259462.4083147, ra=1.95, dec=-1.27, psi=0.82, M_f=68.0, chi_f=0.69,
                     band=[30.0, 350.0]),          # ringdown docs target + Abedi Table-I remnant
}

STAGES = [
    ("ringdown", RD / ".venv311/bin/python", RD / "scripts/watch_ringdown.py"),
    ("npe",      RD / ".venv/bin/python",     RD / "scripts/watch_npe.py"),
    ("echo",     EC / ".venv/bin/python",     EC / "scripts/watch_echo.py"),
]


def run_stage(py, script, cfg, cwd):
    out = Path(tempfile.mktemp(suffix=".json"))
    cfg = {**cfg, "_out": str(out)}
    cfgfile = Path(tempfile.mktemp(suffix=".json")); cfgfile.write_text(json.dumps(cfg))
    try:
        subprocess.run([str(py), str(script), str(cfgfile)], cwd=str(cwd), check=True,
                       capture_output=True, text=True, timeout=900)
        return json.loads(out.read_text())
    except subprocess.CalledProcessError as e:
        return {"error": (e.stderr or e.stdout or "failed").strip().splitlines()[-1][:200]}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    finally:
        for f in (out, cfgfile):
            f.unlink(missing_ok=True)


def report(name, cfg, R):
    L = [f"# Event watcher — {name}", "",
         f"GPS {cfg['gps']} | sky (ra {cfg['ra']}, dec {cfg['dec']}, psi {cfg['psi']}) | "
         f"remnant M_f {cfg['M_f']} M_sun, chi {cfg['chi_f']}", ""]
    rd, npe, ec = R.get("ringdown", {}), R.get("npe", {}), R.get("echo", {})
    L.append("## Ringdown (field-standard package, 220+221)")
    if "error" in rd:
        L.append(f"- ⚠️ stage failed: {rd['error']}")
    else:
        L += [f"- remnant: **M = {rd['M'][0]:.1f} [{rd['M'][1]:.1f}, {rd['M'][2]:.1f}] M_sun**, "
              f"chi = {rd['chi'][0]:.3f} [{rd['chi'][1]:.3f}, {rd['chi'][2]:.3f}]  (rhat {rd['rhat']:.3f})",
              f"- overtone (221): A221/A220 = {rd['a221_over_a220']:.2f}, P(A221≈0) = {rd['overtone_p_below']:.3f} "
              f"→ **{'DETECTED' if rd['overtone_detected'] else 'not significant'}**"]
    L.append(""); L.append("## No-hair test (amortized NPE, start-time-marginalized)")
    if "error" in npe:
        L.append(f"- ⚠️ stage failed: {npe['error']}")
    else:
        L += [f"- M = {npe['M'][0]:.1f} [{npe['M'][1]:.1f}, {npe['M'][2]:.1f}], "
              f"chi = {npe['chi'][0]:.2f} [{npe['chi'][1]:.2f}, {npe['chi'][2]:.2f}]",
              f"- **delta = {npe['delta'][0]:+.2f} [{npe['delta'][1]:+.2f}, {npe['delta'][2]:+.2f}]** → "
              f"Kerr {'consistent ✓' if npe['kerr_consistent_90'] else 'EXCLUDED at 90%'}"]
    L.append(""); L.append("## Echo search (Kerr-tortoise Δt prediction + comb)")
    if "error" in ec:
        L.append(f"- ⚠️ stage failed: {ec['error']}")
    else:
        L += [f"- predicted Δt = **{ec['dt_pred_s']*1e3:.1f} ms** (from remnant, verified formula), band {ec['band']} Hz",
              f"- on-source comb p-value = **{ec['comb_p_value']:.3f}** (n_bg {ec['n_bg']}) → "
              f"**{'ECHO CANDIDATE' if ec['echo_detected'] else 'null (no echo)'}**"]
    # one-line verdict
    ot = "overtone detected" if rd.get("overtone_detected") else "no overtone"
    kerr = "Kerr-consistent" if npe.get("kerr_consistent_90") else ("Kerr-EXCLUDED" if npe else "NPE n/a")
    echo = "echo null" if (ec and not ec.get("echo_detected")) else ("ECHO CANDIDATE" if ec else "echo n/a")
    L += ["", f"**Verdict:** {ot}; {kerr}; {echo}."]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("event", choices=list(EVENTS))
    ap.add_argument("--skip-npe", action="store_true", help="skip the NPE stage (slow venv load)")
    args = ap.parse_args()
    cfg = {"name": args.event, **EVENTS[args.event]}

    # ensure the ringdown package has strain extracted (idempotent)
    subprocess.run([str(RD / ".venv/bin/python"), str(RD / "scripts/20_extract_strain.py")],
                   cwd=str(RD), capture_output=True, text=True)

    R, t0 = {}, time.time()
    for tag, py, script in STAGES:
        if tag == "npe" and args.skip_npe:
            continue
        print(f"[watch] running {tag} ...", flush=True)
        R[tag] = run_stage(py, script, cfg, script.parent.parent)

    md = report(args.event, cfg, R)
    out_md = ROOT / f"watch_{args.event}.md"; out_md.write_text(md)
    out_json = ROOT / f"watch_{args.event}.json"
    out_json.write_text(json.dumps({"event": args.event, "config": EVENTS[args.event],
                                    "stages": R, "elapsed_s": time.time() - t0}, indent=2))
    print("\n" + md)
    print(f"\nwrote {out_md.name} + {out_json.name} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
