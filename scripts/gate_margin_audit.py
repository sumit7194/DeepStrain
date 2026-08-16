"""Are our gates actually gates, or decoration? Margin audit over the whole regression suite.

THE TEST, from quantum (vestigium) via TheBridge Round 12, after G3 produced four gates certifying nothing and
tabula found a clause gated at a bar unreachable by construction:

    "a control that cannot fail is not a control, and the check is trivial and almost never run -- what is the
     observed distribution of the guarded quantity relative to its threshold? If no run is within orders of
     magnitude of the gate, the gate is decoration."

TWO WARNINGS THAT MUST TRAVEL WITH THIS TOOL, both learned by it failing on us:

  1. **The instrument manufactures its own positives.** A margin is a ratio, and dividing by a zero threshold
     yields infinity -- EXACTLY the signature being hunted, so a false positive is indistinguishable from a
     detection. One such "infinite margin" gate turned out to be the TIGHTEST gate in the suite (a bootstrap
     CI lower bound sitting 0.0097 above zero). Zero-sided comparisons are CLASSIFIED, never scored.
  2. **Self-audits gravitate to recent work, which is exactly where the problem isn't.** The first pass here
     covered only the 22 gates written that week and found zero decoration; the one real instance sat in a
     gate three weeks older.

COVERAGE IS THE OTHER HALF OF HONESTY. An earlier version scored 39 of 193 assertions and reported the rest as
"not numeric-comparable" -- but most of them were, and the tool simply could not parse them: it missed
`open(R + "name.json")` (its own gates use that form), `all(...)` generators, range checks, abs-difference
checks, and comparisons whose right-hand side is another artifact value. Reporting 20% coverage as a property
of the SUITE rather than of the PARSER was itself a small instance of the failure being hunted.

WHAT REMAINS UNSCOREABLE IS THE INTERESTING PART. A bare boolean flag (`assert d["verdict_holds"]`) written by
the very script it guards is close to self-certification: the script decides it passed and the gate agrees.
Those cannot carry a margin, so the tool instead reports whether each bare flag has a NUMERIC COMPANION in the
same gate. A flag backed by a number is fine; a flag standing alone is where decoration hides.

    margin > 100x  -> DECORATION CANDIDATE
    10-100x        -> loose (legitimate when a large margin IS the finding, e.g. a null result)
    < 10x          -> a real gate

Run:  python3 scripts/gate_margin_audit.py
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERIFY = ROOT / "verify.sh"

GATE_RE = re.compile(r'^echo "--- (.+?)"', re.M)
JSON_RE = re.compile(r'"([A-Za-z0-9_./-]+\.json)"')
CHAIN = r'[A-Za-z_]\w*(?:\[[^\]]+\])+'
NUM = r'-?\d+\.?\d*(?:[eE][-+]?\d+)?'

P_ABS = re.compile(rf'^abs\((?P<lhs>{CHAIN})\s*-\s*\(?(?P<ref>{NUM})\)?\)\s*(?P<op><=?)\s*(?P<thr>{NUM})$')
P_RANGE = re.compile(rf'^(?P<lo>{NUM})\s*<=?\s*(?P<lhs>{CHAIN})\s*<=?\s*(?P<hi>{NUM})$')
P_LIT = re.compile(rf'^(?P<lhs>(?:abs\()?{CHAIN}\)?)\s*(?P<op>[<>]=?)\s*(?P<thr>{NUM})$')
P_ALL = re.compile(rf'^all\((?P<expr>.+?)\s*(?P<op>[<>]=?)\s*(?P<thr>{NUM})\s+for\s+\w+\s+in\s+(?P<it>.+)\)$')
P_PAIR = re.compile(rf'^(?P<lhs>{CHAIN})\s*(?P<op>[<>]=?)\s*(?P<rhs>{CHAIN})$')
P_FLAG = re.compile(rf'^(?:not\s+)?{CHAIN}$')


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
            except Exception:
                return None
    return obj


def lookup(ns, chain):
    v = resolve_ns(ns, chain)
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def lookup_any(ns, chain):
    return resolve_ns(ns, chain)


def score(val, op, thr):
    """Margin, or a classification when a ratio is meaningless (either side zero)."""
    if thr == 0:
        return None, "sign-test"
    if val == 0:
        return None, "at-extreme"
    return (abs(val / thr) if op.startswith(">") else abs(thr / val)), "ratio"


ASSIGN_JSON = re.compile(r'^(\w+)\s*=\s*json\.loads\(open\(')
ASSIGN_CHAIN = re.compile(rf'^(\w+)\s*=\s*({CHAIN})$')
ASSIGN_TUPLE = re.compile(rf'^([\w,\s]+)=\s*((?:{CHAIN}\s*,\s*)+{CHAIN})$')


def load_artifacts(body):
    """Build the gate's NAMESPACE, not just its files.

    Gates bind sub-objects to short names (`rd, npe, ec = st["ringdown"], st["npe"], st["echo"]`), so a
    resolver that only applies subscript chains to whole artifacts silently fails on every such assertion --
    and then mislabels that gate's boolean flags as UNBACKED because it could not see the numbers sitting
    right next to them. That false-negative-then-false-positive chain is why this is worth parsing properly."""
    ns = {}
    for line in body.splitlines():
        line = line.strip()
        m = ASSIGN_JSON.match(line)
        if m:
            rel = JSON_RE.search(line)
            if not rel:
                continue
            base = Path(rel.group(1)).name
            for c in [ROOT / rel.group(1), *ROOT.glob(f"*/results/{base}"), *ROOT.glob(f"results/{base}"),
                      *ROOT.glob(f"*/{base}"), *ROOT.glob(base)]:
                if c.exists():
                    try:
                        ns[m.group(1)] = json.loads(c.read_text()); break
                    except Exception:
                        pass
            continue
        m = ASSIGN_CHAIN.match(line)
        if m and m.group(1) not in ns:
            v = resolve_ns(ns, m.group(2))
            if v is not None:
                ns[m.group(1)] = v
            continue
        m = ASSIGN_TUPLE.match(line)
        if m:
            names = [x.strip() for x in m.group(1).split(",") if x.strip()]
            vals = [x.strip() for x in re.split(r',\s*(?![^\[]*\])', m.group(2))]
            if len(names) == len(vals):
                for n, ch in zip(names, vals):
                    v = resolve_ns(ns, ch)
                    if v is not None:
                        ns[n] = v
    return ns


def resolve_ns(ns, chain):
    """Resolve `name["a"]["b"][0]` against the namespace, honouring the bound base name."""
    m = re.match(r'^(\w+)((?:\[[^\]]+\])*)$', chain.strip())
    if not m:
        return None
    base, rest = m.group(1), m.group(2)
    if base not in ns:
        return None
    return resolve(ns[base], rest) if rest else ns[base]


def evaluate(part, arts):
    """-> (value, op, threshold, kind) or None if this form is not a numeric comparison."""
    m = P_ABS.match(part)
    if m:
        v = lookup(arts, m.group("lhs"))
        return None if v is None else (abs(v - float(m.group("ref"))), m.group("op"),
                                       float(m.group("thr")), "abs")
    m = P_RANGE.match(part)
    if m:
        v = lookup(arts, m.group("lhs"))
        if v is None:
            return None
        lo, hi = float(m.group("lo")), float(m.group("hi"))
        near = lo if (v - lo) <= (hi - v) else hi          # score against the nearer edge
        return (v, ">" if near == lo else "<", near, "range")
    m = P_LIT.match(part)
    if m:
        lhs = m.group("lhs")
        chain = lhs[4:-1] if lhs.startswith("abs(") else lhs
        v = lookup(arts, chain)
        if v is None:
            return None
        return (abs(v) if lhs.startswith("abs(") else v, m.group("op"), float(m.group("thr")), "lit")
    m = P_ALL.match(part)
    if m:
        it = lookup_any(arts, m.group("it").strip()) if re.match(rf'^{CHAIN}$', m.group("it").strip()) else None
        if not isinstance(it, (list, dict)):
            return None
        seq = list(it.values()) if isinstance(it, dict) else it
        keys = [k.strip().strip('"\'') for k in re.findall(r'\[([^\]]+)\]', m.group("expr"))]
        vals = []
        for e in seq:
            x = e
            for k in keys:
                x = x.get(k) if isinstance(x, dict) else None
            if isinstance(x, (int, float)) and not isinstance(x, bool):
                vals.append(float(x))
        if not vals:
            return None
        op = m.group("op")
        return (max(vals) if op.startswith("<") else min(vals), op, float(m.group("thr")), "all")
    m = P_PAIR.match(part)
    if m:
        a1, a2 = lookup(arts, m.group("lhs")), lookup(arts, m.group("rhs"))
        return None if (a1 is None or a2 is None) else (a1, m.group("op"), a2, "pair")
    return None


def main() -> None:
    text = VERIFY.read_text()
    marks = [(m.start(), m.group(1)) for m in GATE_RE.finditer(text)]
    blocks = [(n, text[p:(marks[i + 1][0] if i + 1 < len(marks) else len(text))])
              for i, (p, n) in enumerate(marks)]

    rows, flags, structural, unparsed = [], [], 0, []
    for name, body in blocks:
        arts = load_artifacts(body)
        gate_numbers, pending = 0, []
        for raw in body.splitlines():
            raw = raw.strip()
            if not raw.startswith("assert "):
                continue
            expr = raw[len("assert "):]
            cut = re.search(r',\s*f?"', expr)
            if cut:
                expr = expr[:cut.start()]
            # A line-continued assertion (`assert X > 0.05, \` + message on the next line) leaves a trailing
            # comma once the message is stripped, which silently breaks every pattern match -- and then the
            # gate's flags get reported as UNBACKED even though the number is sitting right there. Found by
            # this tool failing to see backing assertions THIS AUDIT had just added.
            expr = expr.strip().rstrip("\\").strip().rstrip(",").strip()
            for part in re.split(r'\s+and\s+', expr):
                part = part.strip()
                if not part:
                    continue
                got = evaluate(part, arts)
                if got:
                    val, op, thr, kind = got
                    mg, cls = score(val, op, thr)
                    rows.append((name[:40], part[:46], val, op, thr, mg, cls, kind))
                    gate_numbers += 1
                elif P_FLAG.match(part):
                    # A flag is also BACKED when its own quantity is numerically compared in the SAME
                    # assertion -- `assert g["M_inv"] and 60 < g["M_inv"] < 100` is a null-guard plus a
                    # range check, not self-certification. Judge that textually: the value may be
                    # unresolvable (here `g` comes from a dict comprehension a static parser cannot follow),
                    # but the backing is plainly present in the source.
                    chain = re.sub(r'^not\s+', '', part).strip()
                    same_line = bool(re.search(re.escape(chain) + r'\s*[<>]|[<>]=?\s*' + re.escape(chain), expr))
                    pending.append((name[:40], part[:46], same_line))
                elif re.search(r'\bin\b|len\(|isinstance|\.get\(', part):
                    structural += 1
                else:
                    unparsed.append((name[:40], part[:60]))
        for n, p, own in pending:
            flags.append((n, p, own or gate_numbers > 0))

    ratios = sorted([r for r in rows if r[6] == "ratio"], key=lambda r: -r[5])
    zeros = [r for r in rows if r[6] != "ratio"]
    print(f"{'gate':>41} {'assertion':>47} {'value':>11} {'thresh':>10} {'margin':>9}  verdict")
    dec, loose, tight = [], 0, 0
    for name, part, val, op, thr, mg, _, _ in ratios:
        v = "DECORATION?" if mg > 100 else ("loose" if mg > 10 else "real gate")
        if mg > 100:
            dec.append((name, part, val, thr, mg))
        elif mg > 10:
            loose += 1
        else:
            tight += 1
        print(f"{name:>41} {part:>47} {val:>11.4g} {thr:>10.4g} {mg:>9.1f}  {v}")

    bare = [f for f in flags if not f[2]]
    total = len(rows) + len(flags) + structural + len(unparsed)
    print(f"\nSCORED {len(ratios)} ratios | zero-sided (classified) {len(zeros)} | flags {len(flags)} | "
          f"structural {structural} | unparsed {len(unparsed)}   [of ~{total} conjuncts]")
    print(f"real gates (<10x): {tight} | loose (10-100x): {loose} | DECORATION (>100x): {len(dec)}")
    for name, part, val, thr, mg in dec:
        print(f"   >100x  {name}: {part} = {val:.4g} vs {thr:.4g} ({mg:.0f}x)")
    print(f"\nBOOLEAN FLAGS: {len(flags)} | UNBACKED (no numeric assertion anywhere in their gate): {len(bare)}")
    for n, p, _ in bare:
        print(f"   UNBACKED  {n}: assert {p}")
    if unparsed:
        print(f"\nunparsed ({len(unparsed)}):")
        for n, p in unparsed[:10]:
            print(f"   {n}: {p}")

    out = {"scored_ratios": len(ratios), "zero_sided": len(zeros), "flags_total": len(flags),
           "flags_unbacked": len(bare), "structural": structural, "unparsed": len(unparsed),
           "tight": tight, "loose": loose,
           "decoration": [{"gate": n, "assertion": p, "value": v, "threshold": t, "margin": m}
                          for n, p, v, t, m in dec],
           "unbacked": [{"gate": n, "assertion": p} for n, p, _ in bare]}
    (ROOT / "gate_margin_audit.json").write_text(json.dumps(out, indent=2))
    print("\nwrote gate_margin_audit.json")


if __name__ == "__main__":
    main()
