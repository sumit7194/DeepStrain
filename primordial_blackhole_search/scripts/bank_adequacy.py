"""Is the 0.1% bank ADEQUATE, or merely the densest one we could afford? The pre-registered saturation test.

WHY. Follow-up A's headline is that a CNN TIES a realizable matched-filter bank (1,619 templates at 0.1% Mc
spacing, 1.03x). `bridge` put the objection precisely: 0.1% is CHEAPER, which is not ADEQUATE. If the CNN ties
there and the bank was still climbing, the tie is a statement about a bank that had not saturated -- and
answering that after seeing the result is the worst time to answer it.

RESULTS.md pre-registers the criterion BEFORE this ran: adequacy is a SATURATION argument, not a density one.
The bank is adequate only if matched-filter sensitive distance has FLATTENED with density, declared bar
MF(dense)/MF(coarser) < 1.10 -- and only then is a CNN-vs-MF ratio quotable.

NO NEW COMPUTE. `bank_dense.json` already carries a sub-sampled density sweep (B = 83, 164, 326, 649, 1619)
from the committed run, so the question is answerable from data on disk -- which the standing rule says to
check before recording anything as blocked.

Run:  .venv/bin/python scripts/bank_adequacy.py
"""
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pbh import config as C

SRC = C.RESULTS_DIR / "bank_dense.json"
BAR = 1.10          # pre-registered: MF(dense)/MF(coarser) below this = flattened
BINS = ("0.17-0.35", "0.35-0.55", "0.55-0.88")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC.name, help="density-sweep json in results/ (default: the 0.1%% run)")
    src = C.RESULTS_DIR / ap.parse_args().src
    out = C.RESULTS_DIR / ("bank_adequacy.json" if src.name == SRC.name else f"bank_adequacy_{src.stem}.json")
    d = json.loads(src.read_text())
    sweep = {int(k): v for k, v in d["sweep"].items()}
    Bs = sorted(sweep)
    dense, coarse = Bs[-1], Bs[-2]

    print(f"density sweep from {src.name} (spacing {d['spacing']}, n_inj {d['n_inj']})")
    print(f"{'B':>7} " + "".join(f"{b:>12}" for b in BINS))
    for B in Bs:
        print(f"{B:>7} " + "".join(f"{sweep[B]['frac'][b]:12.3f}" for b in BINS))

    print(f"\nPRE-REGISTERED TEST: MF(B={dense}) / MF(B={coarse}) < {BAR} in every bin?")
    rows, verdicts = {}, []
    for b in BINS:
        hi, lo = sweep[dense]["frac"][b], sweep[coarse]["frac"][b]
        ratio = (hi / lo) if lo > 0 else float("inf")
        ok = ratio < BAR
        verdicts.append(ok)
        rows[b] = {"coarse": lo, "dense": hi, "ratio": ratio, "flattened": bool(ok)}
        shown = f"{ratio:8.2f}" if lo > 0 else "     inf"
        note = "" if lo > 0 else "   <- bin is ZERO at the coarser bank: a threshold being crossed, not a plateau"
        print(f"  {b:>12}  {lo:.3f} -> {hi:.3f}   ratio {shown}  {'FLAT' if ok else 'STILL CLIMBING'}{note}")

    adequate = all(verdicts)
    res = {"source": src.name, "spacing": d["spacing"], "bar": BAR,
           "dense_B": dense, "coarse_B": coarse, "bins": rows,
           "adequate": bool(adequate),
           "cnn_w64": d["cnn_w64"], "mf_at_dense": {b: sweep[dense]["frac"][b] for b in BINS}}
    res["verdict"] = (
        f"ADEQUATE -- MF has flattened by B={dense}, so the CNN-vs-MF comparison at this density is quotable"
        if adequate else
        f"NOT ADEQUATE -- MF is still climbing at B={dense}. Per the pre-registration NO CNN-vs-MF verdict is "
        f"quoted from this bank; the deliverable is the density curve. Follow-up A's '1.03x tie' remains true "
        f"as written -- it says REALIZABLE bank, and 1,619 was the laptop ceiling -- but it does NOT license "
        f"'a CNN ties matched filtering', because the bank had not saturated. The filter is now 2.5-3.5x "
        f"cheaper, so the next rung is affordable and that is what settles it.")
    print(f"\nVERDICT: {res['verdict']}")
    out.write_text(json.dumps(res, indent=2))
    print(f"wrote {out.name}")


if __name__ == "__main__":
    main()
