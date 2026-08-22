"""Do our gates FAIL when the artifact is wrong? The margin audit could not tell us.

WHY THIS EXISTS. gate_margin_audit.py asks whether each gate's threshold is reachable -- whether any run
lands near the bar. That is a real question and it caught one decoration. But it is fundamentally a question
about NUMBERS IN A FILE, and it cannot detect a gate that is wired wrong:

  * an assertion on a mistyped or renamed key raises KeyError -- which the runner reports as a FAILURE, so
    that one is safe -- but an assertion on the WRONG key, one that happens to exist, passes forever;
  * a gate whose asserted quantity no longer influences the claim it guards;
  * a gate whose comparison is trivially satisfied by the shape of the data rather than its value.

All three look identical to a healthy gate when you read them, and identical when you measure their margins.
The only way to distinguish them is to make the artifact WRONG and check the gate notices.

    bridge, after two of their own checks passed by eye and failed when run:
    "these checks don't work as reading comprehension -- grepping for whether a gate exists returns
     'fine' in exactly the cases that matter."

METHOD. For each gate block in verify.sh, extract its inline Python, then execute it with `open` replaced by
one that serves MUTATED copies of the JSON artifacts: every number scaled hard, every boolean flipped, every
list of numbers scaled. Nothing on disk is touched -- the mutation happens in memory, so a power cut mid-run
cannot leave a corrupted artifact behind (this machine has cut six times).

A healthy gate raises AssertionError on garbage. A gate that PASSES a fully mutated artifact is asserting on
nothing that matters, and is reported as UNGUARDED.

READING THE OUTPUT. `fails-on-garbage` is the good outcome. Two results are NOT failures of the gate:
  * ERROR (KeyError/TypeError) -- the gate still stops the suite, which is the behaviour we want, though a
    clean assertion message would be better;
  * gates whose artifact has no mutable numbers (pure string/structure checks) -- reported separately rather
    than scored, the same discipline the margin audit uses for zero-threshold comparisons.

Run:  python3 scripts/gate_mutation_test.py
"""
import io
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERIFY = ROOT / "verify.sh"
BLOCK = re.compile(r'^echo "--- (?P<name>.+?)"\s*\n\./(?P<venv>\S+?)\s*-\s*<<\s*\'(?P<tag>\w+)\'.*?\n'
                   r'(?P<body>.*?)\n(?P=tag)\s*$', re.M | re.S)


# A SINGLE MUTATION IS NOT ENOUGH, and the first version of this tool proved it on itself. It used
# x -> 37x + 11: monotone INCREASING, so it satisfies every one-sided `>=` assertion (n_bg >= 300,
# n_catalogs >= 15, p > 0.05) and preserves every ordering (a > b + eps). It reported 7 gates as UNGUARDED
# that were all perfectly healthy -- the mutation simply could not break them. A mutation tester whose
# perturbation cannot violate the assertion under test measures nothing, which is the same failure as a
# gate whose bar cannot be crossed. Apply a FAMILY that moves values in both directions and destroys
# orderings, and call a gate unguarded only if it survives EVERY one.
STRATEGIES = {
    "inflate":  lambda x: x * 37.0 + 11.0,      # breaks upper bounds
    "collapse": lambda x: x * 1e-4,             # breaks lower bounds and >= thresholds
    "negate":   lambda x: -abs(x) - 1.0,        # breaks positivity and most orderings
    "zero":     lambda x: 0.0,                  # breaks ratios, non-zero requirements
}


def mutate(obj, fn):
    if isinstance(obj, bool):
        return not obj
    if isinstance(obj, (int, float)):
        return fn(obj)
    if isinstance(obj, list):
        return [mutate(v, fn) for v in obj]
    if isinstance(obj, dict):
        return {k: mutate(v, fn) for k, v in obj.items()}
    return obj


def main() -> None:
    text = VERIFY.read_text()
    blocks = list(BLOCK.finditer(text))
    print(f"{len(blocks)} gate blocks parsed from verify.sh\n")

    healthy, unguarded, errored, nodata = [], [], [], []
    for m in blocks:
        name, body = m.group("name"), m.group("body")
        outcomes, touched_any = {}, False
        for sname, fn in STRATEGIES.items():
            touched = []

            def fake_open(path, *a, _fn=fn, _t=touched, **k):
                p = str(path)
                if p.endswith(".json"):
                    full = ROOT / p
                    if not full.exists():
                        raise FileNotFoundError(p)
                    _t.append(p)
                    return io.StringIO(json.dumps(mutate(json.loads(full.read_text()), _fn)))
                return open(full if not Path(p).is_absolute() else p, *a, **k)

            g = {"__name__": "__main__", "open": fake_open, "json": json,
                 "print": lambda *a, **k: None}
            try:
                exec(compile(body, f"<gate:{name}>", "exec"), g)
                outcomes[sname] = "passed"
            except AssertionError:
                outcomes[sname] = "caught"
            except Exception as e:
                outcomes[sname] = f"error:{type(e).__name__}"
            touched_any = touched_any or bool(touched)
        if not touched_any:
            nodata.append(name)
        elif any(v == "caught" for v in outcomes.values()):
            healthy.append(name)
        elif any(str(v).startswith("error") for v in outcomes.values()):
            errored.append((name, ",".join(f"{k}={v}" for k, v in outcomes.items())))
        else:
            unguarded.append(name)

    tot = len(blocks)
    print(f"FAILS ON GARBAGE (healthy)      {len(healthy):3d}/{tot}")
    print(f"errors on garbage (still stops) {len(errored):3d}/{tot}")
    print(f"NO JSON READ (not scoreable)    {len(nodata):3d}/{tot}")
    print(f"PASSES ON GARBAGE (UNGUARDED)   {len(unguarded):3d}/{tot}")
    if unguarded:
        print("\n!! these gates accept a fully mutated artifact -- they assert on nothing that matters:")
        for n in unguarded:
            print(f"     - {n}")
    if errored:
        print("\n(errored -- still halts the suite, but an assertion message would be clearer:)")
        for n, e in errored[:12]:
            print(f"     - [{e}] {n}")
    if nodata:
        print("\n(no JSON artifact read -- structural/string checks, not scoreable here:)")
        for n in nodata[:12]:
            print(f"     - {n}")
    out = ROOT / "gate_mutation_test.json"
    out.write_text(json.dumps({"total": tot, "healthy": healthy, "unguarded": unguarded,
                               "errored": errored, "no_json": nodata}, indent=2))
    print(f"\nwrote {out.name}")


if __name__ == "__main__":
    main()
