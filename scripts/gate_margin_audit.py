"""Are our 52 gates actually gates, or decoration? Margin audit across the whole regression suite.

THE TEST, from quantum via TheBridge Round 12, after G3 produced four gates that certified nothing (a dead
conjunct, a vacuously-passing control, a non-discriminating statistic, and an integration guard whose observed
quantity sat 4.6e7 below its own threshold, having never rejected anything in 599 orbits) and tabula
independently found a clause gated at a bar unreachable by construction:

    "a control that cannot fail is not a control, and the check is trivial and almost never run -- what is the
     observed distribution of the guarded quantity relative to its threshold? If no run is within orders of
     magnitude of the gate, the gate is decoration."

We run more gates than any sibling and had never checked this. A first pass covering the 22 gates added this
week found 0 decoration -- but also produced a false positive from its OWN encoding (a gate written as
`3.0 - dist > 0` has threshold zero, so the margin divided by zero and reported infinity, which is exactly the
signature being hunted). This version parses the real assertions out of verify.sh instead of re-encoding them
by hand.

METHOD. Split verify.sh into gate blocks, find the artifacts each block loads, then evaluate every assertion
of the form `<subscript chain> <op> <literal>` against those artifacts and report |value/threshold|. Coverage
is partial by construction -- compound conditions, `all(...)`/`any(...)` generators and boolean flags are not
numeric comparisons and are counted separately rather than silently skipped.

    margin > 100x  -> DECORATION CANDIDATE: no plausible run comes near this bar
    10-100x        -> loose (fine when a large margin IS the finding, e.g. a null result)
    < 10x          -> a real gate

Run:  python3 scripts/gate_margin_audit.py
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERIFY = ROOT / "verify.sh"

GATE_RE = re.compile(r'^echo "--- (.+?)"', re.M)
OPEN_RE = re.compile(r'open\(\s*"([^"]+\.json)"\s*\)')
# assert <chain of ["key"] / [int] on a bare name> <op> <number>
ASSERT_RE = re.compile(
    r'^assert\s+(?P<lhs>(?:abs\()?[A-Za-z_]\w*(?:\[[^\]]+\])+\)?)\s*(?P<op>[<>]=?)\s*(?P<thr>-?\d+\.?\d*(?:[eE][-+]?\d+)?)')


def resolve(obj, chain):
    for key in re.findall(r'\[([^\]]+)\]', chain):
        key = key.strip()
        if key.startswith(('"', "'")):
            key = key[1:-1]
            if not isinstance(obj, dict) or key not in obj:
                return None
            obj = obj[key]
        else:
            try:
                obj = obj[int(key)]
            except (ValueError, TypeError, KeyError, IndexError):
                return None
    return obj


def main() -> None:
    text = VERIFY.read_text()
    marks = [(m.start(), m.group(1)) for m in GATE_RE.finditer(text)]
    blocks = []
    for i, (pos, name) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        blocks.append((name, text[pos:end]))

    rows, uncovered, missing = [], 0, set()
    for name, body in blocks:
        arts = {}
        for rel in OPEN_RE.findall(body):
            p = ROOT / rel
            if p.exists():
                try:
                    arts[rel] = json.loads(p.read_text())
                except Exception:
                    pass
            else:
                missing.add(rel)
        n_assert = len(re.findall(r'^assert ', body, re.M))
        matched = 0
        for line in body.splitlines():
            m = ASSERT_RE.match(line.strip())
            if not m:
                continue
            lhs, op, thr = m.group("lhs"), m.group("op"), float(m.group("thr"))
            chain = lhs[4:-1] if lhs.startswith("abs(") else lhs
            val = None
            for a in arts.values():
                v = resolve(a, chain)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    val = abs(v) if lhs.startswith("abs(") else v
                    break
            if val is None:
                continue
            matched += 1
            # A ratio is the wrong figure of merit when either side is zero, and reporting inf there
            # manufactures exactly the signature being hunted. Classify those cases instead of scoring them:
            #   threshold 0  -> a SIGN/positivity test (e.g. "is this CI lower bound above zero?"), which can
            #                   be perfectly tight (0.0097 > 0 is a hair's breadth) or categorical (sign = -1)
            #   value 0      -> the quantity sits AT the extreme; a ratio says nothing useful
            if thr == 0:
                kind, margin = "sign-test", None
            elif val == 0:
                kind, margin = "at-extreme", None
            else:
                kind = "ratio"
                margin = abs(val / thr) if op.startswith(">") else abs(thr / val)
            rows.append((name[:44], chain[:40], val, op, thr, margin, kind))
        uncovered += n_assert - matched

    ratios = [r for r in rows if r[6] == "ratio"]
    others = [r for r in rows if r[6] != "ratio"]
    ratios.sort(key=lambda r: -r[5])
    print(f"{'gate':>45} {'quantity':>41} {'value':>11} {'op':>3} {'thresh':>9} {'margin':>9}  verdict")
    dec, loose, tight = [], 0, 0
    for name, q, val, op, thr, mg, _ in ratios:
        if mg > 100:
            v = "DECORATION?"; dec.append((name, q, val, op, thr, mg))
        elif mg > 10:
            v = "loose"; loose += 1
        else:
            v = "real gate"; tight += 1
        print(f"{name:>45} {q:>41} {val:>11.4g} {op:>3} {thr:>9.4g} {mg:>9.1f}  {v}")
    print(f"\nnot ratio-scorable ({len(others)}): threshold or value is zero, so a margin is meaningless")
    for name, q, val, op, thr, mg, kind in others:
        print(f"   [{kind:>10}] {name}: {q} = {val:.4g} {op} {thr:g}")

    print(f"\ncovered {len(rows)} numeric assertions | not numeric-comparable (flags, all()/any(), compound): "
          f"{uncovered}")
    if missing:
        print(f"artifacts referenced but absent: {sorted(missing)}")
    print(f"real gates (<10x): {tight} | loose (10-100x): {loose} | DECORATION candidates (>100x): {len(dec)}")
    for name, q, val, op, thr, mg in dec:
        print(f"   >100x  {name}: {q} = {val:.4g} {op} {thr} ({mg:.0f}x from its bar)")
    out = {"covered": len(rows), "uncovered": uncovered, "tight": tight, "loose": loose,
           "decoration": [{"gate": n, "quantity": q, "value": v, "op": o, "threshold": t, "margin": m}
                          for n, q, v, o, t, m in dec]}
    (ROOT / "gate_margin_audit.json").write_text(json.dumps(out, indent=2))
    print("\nwrote gate_margin_audit.json")


if __name__ == "__main__":
    main()
