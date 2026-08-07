#!/usr/bin/env python
"""Milestone 26 — re-open the δ-stacking wall against the CURRENT catalog (was decided on 8 events in June).

v5/v6 concluded that only GW250114 measures δ and that multi-event stacking is blocked by an SNR wall. That
rested on a hand-picked 8-event list from O1–O3, chosen BEFORE GWTC-4.0/5.0 published hundreds of O4 events.
Re-selecting from the full public catalog (439 unique events) under the constraints our NPE actually has:

  SELECTION (all three required — a violation makes the answer meaningless, not just noisy):
    1. BBH, not BNS            (a neutron-star merger has no black-hole ringdown)
    2. detector-frame remnant mass in the NPE prior [40, 120] Msun  (source mass x (1+z))
    3. network SNR > 20        (below this nothing has ever been informative for delta)
  -> 13 usable events, of which only 3 were in the June list.

  PER-EVENT VALIDITY (the correctness-critical part — an event only counts if the NPE is trustworthy on it):
    (a) mass sanity: recovered M within a factor of the published detector-frame remnant, allowing the KNOWN
        +10% peak-start bias measured in R3/B1. A wild miss means the fit failed, not that Kerr is violated.
    (b) not railed: posterior not pinned at a prior edge (M at 40/120, chi at 0.05/0.95).
    (c) informative: sigma(delta) / sigma_prior < 0.90 (the v6 criterion). sigma_prior = 1/sqrt(12) = 0.2887
        for delta ~ U[-0.5, 0.5]. An uninformative event returns the prior and CANNOT help a stack.

Only events passing (a)+(b) are reported; (c) decides whether the stacking wall is real at current data.

Run:  .venv/bin/python scripts/26_more_events_o4.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

import rdlib
import sbilib
from sbilib import Embed, N_SAMP, SEG  # Embed at module level: the pickled posterior resolves it

RESULTS = Path(__file__).resolve().parent.parent / "results"
SIGMA_PRIOR = 1.0 / np.sqrt(12)      # delta ~ U[-0.5, 0.5]
INFORMATIVE_FRAC = 0.90              # v6 criterion: sigma_delta/sigma_prior below this = carries information
M_LO, M_HI = 40.0, 120.0
CHI_LO, CHI_HI = 0.05, 0.95

# selected by the criteria above from the full public catalog (see /tmp/usable_events.json provenance);
# m_det = published source-frame remnant x (1+z) — the frame the NPE works in.
EVENTS = [
    ("GW250114_082203", 78.6, 68.6), ("GW230814_230901", 43.0, 62.5),
    ("GW240920_124024", 37.4, 79.4), ("GW231226_101520", 34.7, 88.1),
    ("GW241127_061008", 31.3, 100.4), ("GW240621_195059", 28.2, 77.4),
    ("GW200129_065458", 26.8, 71.0), ("GW240615_113620", 26.4, 74.6),
    ("GW150914", 26.0, 67.7), ("GW190521_074359", 25.9, 87.8),
    ("GW231206_233901", 21.9, 80.8), ("GW230927_153832", 20.3, 44.9),
    ("GW200224_222234", 20.0, 90.7),
]


def main() -> None:
    posterior = torch.load(RESULTS / "09_posterior_150k.pt", weights_only=False)
    T = json.loads((RESULTS / "10_recalibration.json").read_text())["T"]
    print(f"NPE + recalibration T={T} | sigma_prior(delta)={SIGMA_PRIOR:.4f} | "
          f"informative if sigma/prior < {INFORMATIVE_FRAC}\n")

    def post(x_obs, n=4000):
        s = posterior.sample((n,), x=x_obs, show_progress_bars=False).numpy()
        med = np.median(s, 0)
        return med + T * (s - med)      # v3 temperature recalibration

    rows, failures = [], []
    hdr = f"{'event':>18} {'SNR':>5} {'M_pub':>6} | {'M_npe':>17} {'chi':>6} | {'delta [90% CI]':>21} {'s/prior':>8} {'info?':>6}"
    print(hdr); print("-" * len(hdr))
    for ev, snr, m_pub in EVENTS:
        try:
            gps = rdlib.event_gps(ev)
            segs = []
            for det in ("H1", "L1"):
                white = rdlib.fetch_whitened(det, gps, bandpass=False)
                pk = rdlib.find_peak(white.bandpass(*rdlib.BAND), gps)
                seg = white.crop(pk, pk + SEG + 0.01).value[:N_SAMP]
                if len(seg) != N_SAMP or not np.isfinite(seg).all():
                    raise ValueError("bad/short segment")
                segs.append(seg)
            x = torch.tensor(np.stack(segs).reshape(1, -1).astype(np.float32))
            s = post(x)
        except Exception as e:
            failures.append((ev, f"{type(e).__name__}: {str(e)[:60]}"))
            print(f"{ev:>18} {snr:>5.1f} {m_pub:>6.1f} |  SKIP — {type(e).__name__}", flush=True)
            continue

        m_q = np.percentile(s[:, 0], [50, 5, 95])
        c_q = np.percentile(s[:, 1], [50, 5, 95])
        d_q = np.percentile(s[:, 2], [50, 5, 95])
        sig_d = float(s[:, 2].std())
        frac = sig_d / SIGMA_PRIOR

        # (a) mass sanity — the NPE carries a known +10% peak-start bias (R3/B1), so allow generous room;
        #     a miss beyond this means the fit failed on this event and it must not enter any conclusion.
        mass_ok = 0.6 * m_pub <= m_q[0] <= 1.6 * m_pub
        # (b) railing at a prior edge — check the CI, not just the median. A posterior TRUNCATED by the mass
        #     prior has distorted geometry, and delta can look artificially narrow (a fake "informative" read).
        #     This caught GW231206_233901: M 106.4 vs published 80.8, CI hitting 118 of a 120 ceiling, and a
        #     sigma/prior that beat events 1.7x louder — unphysical.
        railed = (m_q[0] < M_LO + 2 or m_q[0] > M_HI - 2 or m_q[2] >= M_HI - 2 or m_q[1] <= M_LO + 2
                  or c_q[0] < CHI_LO + 0.03 or c_q[0] > CHI_HI - 0.03)
        informative = bool(frac < INFORMATIVE_FRAC)
        valid = bool(mass_ok and not railed)

        rows.append(dict(event=ev, snr=snr, m_pub=m_pub, M=list(m_q), chi=list(c_q), delta=list(d_q),
                         sigma_delta=sig_d, sigma_over_prior=frac, mass_ok=bool(mass_ok),
                         railed=bool(railed), valid=valid, informative=informative,
                         kerr_consistent=bool(d_q[1] <= 0.0 <= d_q[2])))
        flag = "" if valid else ("  <-- INVALID: " + ("mass miss" if not mass_ok else "railed"))
        print(f"{ev:>18} {snr:>5.1f} {m_pub:>6.1f} | {m_q[0]:>6.1f}[{m_q[1]:>4.0f},{m_q[2]:>4.0f}] {c_q[0]:>6.2f} | "
              f"{d_q[0]:>+6.2f}[{d_q[1]:>+5.2f},{d_q[2]:>+5.2f}] {frac:>8.3f} {'YES' if informative else 'no':>6}{flag}",
              flush=True)

    valid_rows = [r for r in rows if r["valid"]]
    info_rows = [r for r in valid_rows if r["informative"]]
    print(f"\nanalyzed {len(rows)}/{len(EVENTS)} | valid (mass-sane, not railed): {len(valid_rows)} | "
          f"INFORMATIVE for delta: {len(info_rows)}")
    if info_rows:
        print("informative events:")
        for r in sorted(info_rows, key=lambda x: x["sigma_over_prior"]):
            print(f"   {r['event']:>18}  SNR {r['snr']:5.1f}  sigma/prior {r['sigma_over_prior']:.3f}  "
                  f"delta {r['delta'][0]:+.2f} [{r['delta'][1]:+.2f},{r['delta'][2]:+.2f}]")
    # QUANTIFY the wall instead of just asserting it: on clean events, sigma/prior falls with ln(SNR).
    # Fit it and report the loudness an event must reach to carry delta information at all.
    snr_needed = None
    if len(valid_rows) >= 4:
        sn = np.array([r["snr"] for r in valid_rows]); fr = np.array([r["sigma_over_prior"] for r in valid_rows])
        corr = float(np.corrcoef(sn, fr)[0, 1])
        c = np.polyfit(np.log(sn), fr, 1)
        snr_needed = float(np.exp((INFORMATIVE_FRAC - c[1]) / c[0]))
        print(f"\ntrend on {len(valid_rows)} clean events: corr(SNR, sigma/prior) = {corr:+.2f}")
        print(f"  fit sigma/prior = {c[0]:+.4f}*ln(SNR) + {c[1]:.3f}  ->  need SNR >= {snr_needed:.0f} "
              f"for an event to carry delta information")

    n_new_info = len([r for r in info_rows if r["event"] not in
                      ("GW250114_082203", "GW150914", "GW200129_065458")])
    print(f"\nNEW informative events beyond the June list: {n_new_info}")
    print("VERDICT:", "the delta-stacking WALL IS REAL at current data (still only one informative event)"
          if len(info_rows) <= 1 else
          f"the WALL IS NOT REAL — {len(info_rows)} events carry delta information; stacking is now worth doing")

    (RESULTS / "26_more_events_o4.json").write_text(json.dumps(
        {"sigma_prior": SIGMA_PRIOR, "informative_frac": INFORMATIVE_FRAC, "events": rows,
         "failures": failures, "n_analyzed": len(rows), "n_valid": len(valid_rows),
         "n_informative": len(info_rows), "n_new_informative": n_new_info,
         "wall_is_real": bool(len(info_rows) <= 1), "snr_needed_for_information": snr_needed,
         "n_prior_truncated": len(rows) - len(valid_rows)}, indent=2))
    print("wrote 26_more_events_o4.json")


if __name__ == "__main__":
    main()
