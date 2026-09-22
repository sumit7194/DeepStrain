"""Parse the METRICS sGB supplement's 40-order spin series and gate the parse before anything is measured.

WHY. `ansatz`'s substrate is a double truncation at O(zeta) and O(a^2), and its §1 caveat has been carrying
OUR Kerr-QNM number as the quantitative statement of ITS limitation -- a number about a different object.
arXiv:2406.11986's supplement carries the sGB metric corrections to 40 orders in spin as raw, unresummed
series, so the caveat can be replaced by a measurement of their actual object.

SYMBOLS -- read from main.tex and the notebook's own header, NOT inferred from a title:
  * `a` is the DIMENSIONLESS SPIN (angular momentum per unit mass, over M).
  * `chi` is cos(theta), the polar coordinate. It is NOT spin.
  * zeta = alpha^2 / M^4 (main.tex, Eq. zeta).
  ⚠️ An earlier census of this notebook attributed the 40-order even series to chi and built a u = chi^2
  plan on it. `a`, not chi, is the expansion variable. A name in a title is not a measurement of what a
  symbol means.

PARITY, measured from the raw boxes with a control-validated pattern:
  phi, H1..H4, kappa : a^2..a^40, zero bare `a`   -> EVEN in spin
  Omega^(1)          : a x (a^0..a^38)           -> ODD in spin
  Frame dragging must reverse with the spin; the metric functions must not. The split is physics, so it is
  a STRUCTURAL CHECK ON THE PARSER: a parsed H that comes back odd, or an Omega that comes back even, means
  the parse is broken whatever else agrees.

EVALUATION IS FROM THE BOX TREE, NOT FROM GENERATED PYTHON. The first version emitted Python source and
eval'd it. Two reasons that was the wrong design: eval on text from a downloaded file, and Python's own
parser raises RecursionError on a flat sum of ~10^5 terms, which these cells approach. It also needed
string-level guesses about implicit multiplication (its first draft turned `416 (M^3)` into a FUNCTION CALL).
Now each RowBox's flat infix sequence goes through a shunting-yard into a postfix program; the only
recursion is over box NESTING (tens deep), unknown leaves raise, and numbers are mpmath at 50 digits --
the coefficients are 30-40 digit rationals and float64 would round the very cancellations the gates check.

THE GATES. Nothing downstream is reported unless all pass:
  G1  every cell evaluates.
  G2  parity per cell, from the PARSED expression by exact DFT in `a` (the parser never used the census).
  G3  HORIZON RIGIDITY -- an identity tying three separately parsed cells together at all orders in `a`:
      the metric (main.tex) gives Omega_H = -g_tphi/g_phiphi at r_+, and the horizon stays at Delta = 0
      because g^rr = Delta / (Sigma (1 + zeta H3)). So at O(zeta)
          Omega^(1) = Omega^(0) * (H2 - H4)|_{r = r_+},   Omega^(0) = -g_tphi/g_phiphi at zeta = 0,
      and the right side must NOT depend on chi (a Killing horizon rotates rigidly). A FractionBox mis-nest
      in H2 or H4 breaks both the equality and the chi-independence.
      ⚠️ main.tex PRINTS Omega^(0) = a/(2Mb), b = sqrt(1-a^2) (b defined in its glossary). That disagrees
      with its OWN metric by exactly (1+b)/b -- a/(2M) at small spin where the metric gives a/(4M). Using
      the printed form, G3 failed by 2.005/2.048/2.155/2.400 at a = 0.1/0.3/0.5/0.7 while chi-independence
      held to 1e-44: a wrong prefactor, not a wrong parse. The notebook's Omega^(1) agrees with the
      METRIC-derived Omega^(0), so the error is confined to the printed formula in the arXiv source we hold
      (journal version not checked).
  G5  SURFACE GRAVITY -- kappa^(1) re-derived from H1..H4 through main.tex's kappa formula, with Kerr's
      printed kappa^(0) = b/(2M(1+b)) reproduced first as the control. Brings H1 and H3 under a gate, which
      G3 does not reach; also checks that the O(zeta) part of xi^2 vanishes at r_+, i.e. that the horizon
      genuinely stays at Delta = 0 as the gauge claims.
  G4  PUBLISHED: Yunes & Stein 2011 (arXiv:1101.2921, Eq. 8), static scalar field in Schwarzschild coords
          theta_bar = (alpha3 / beta) (2 / (M r)) (1 + M/r + 4 M^2 / (3 r^2)),
      read at the primary source 2026-09-22. The RATIOS of the 1/r^2 and 1/r^3 coefficients to the 1/r one
      (M and 4M^2/3) are normalisation-free, so they test the scalar cell's a^0 part without mapping
      couplings between papers. Reported side by side, reproduced and published.

PRE-REGISTRATION (written 2026-09-22 after the gates passed, BEFORE any truncation or radius number was
computed; committed in that state):
  M1  PRIMARY -- the number `ansatz`'s §1 caveat actually needs: the relative error of truncating each
      sGB correction at O(a^2), against the a^40 series, at a = 0.69 (universal remnant) and a = 0.90.
      Objects: kappa^(1), Omega^(1)/a (horizon; no r or chi), and H1..H4, phi at r = 3M and r = 6M, chi
      in {0, 0.5}. Reference validity: the a^40 reference is used only where its own last-term size is
      < 1% of the O(a^2)-truncation error, otherwise that entry is reported as UNRESOLVED, not as a number.
      PREDICTION: larger than Kerr's own 6.36% at 0.69 (the paper stresses sGB and GR trajectories
      separate as spin grows) -- 10-30% for the horizon quantities, and I would not be surprised by a
      single metric function exceeding that. Stated so it can be wrong.
  M2  SECONDARY -- radius of convergence in u = a^2 of the horizon series (exact rational coefficients,
      21 per cell, so no fit and no extraction error). Bands as for P0: R in [0.98, 1.02] => the Kerr
      extremal singularity carries over (H); R < 0.95 => sGB moves it inward (not-H). PREDICTION: H,
      because the O(zeta) corrections are built on the Kerr background, whose branch point is b = 0.
      Caveat carried in, not discovered: Domb-Sykes on 20 ratios is blind to a same-sign competitor
      (bridge, 2026-09-22), so an H reading is weaker evidence than a not-H one.

Run:  .venv/bin/python scripts/38_sgb_supplement.py --nb <path to Supplementary_materials.nb>
"""
import argparse
import json
import re
import time
from pathlib import Path

import mpmath as mp

RESULTS = Path(__file__).resolve().parent.parent / "results"
OUT = RESULTS / "38_sgb_supplement.json"

VARS = {"a": "a", "r": "r", "M": "M", r"\[Chi]": "chi", r"\[Alpha]": "alpha"}
DPS = 50
NDFT = 48                                  # > 41, the top power of `a` -> DFT coefficients exact, no aliasing


def tokenize(s):
    """Split a Mathematica box expression into HEAD[ , ] structure, strings and punctuation."""
    out, i, n = [], 0, len(s)
    while i < n:
        c = s[i]
        if c.isspace():
            i += 1
        elif c == '"':                                   # a quoted leaf
            j = i + 1
            buf = []
            while j < n and s[j] != '"':
                if s[j] == "\\" and j + 1 < n:
                    # Mathematica wraps long integers with backslash-newline. Those continuations are
                    # cosmetic and must be REASSEMBLED here, or a 40-digit coefficient splits in two.
                    # 1,721 of them in H1 alone. Other escapes are kept verbatim.
                    if s[j + 1] == "\n":
                        j += 2
                    else:
                        buf.append(s[j:j + 2]); j += 2
                else:
                    if s[j] != "\n":
                        buf.append(s[j])
                    j += 1
            out.append(("STR", "".join(buf))); i = j + 1
        elif c in "[]{},":
            out.append(("PUNC", c)); i += 1
        else:
            j = i
            while j < n and (s[j].isalnum() or s[j] in "_$"):
                j += 1
            if j == i:
                i += 1; continue
            out.append(("SYM", s[i:j])); i = j
    return out


def parse(tokens, pos=0):
    """HEAD[args] -> ('call', HEAD, [args]);  {a,b} -> ('list', [..]);  leaves as ('str'|'sym', text)."""
    kind, val = tokens[pos]
    if kind == "PUNC" and val == "{":
        items, pos = [], pos + 1
        while not (tokens[pos] == ("PUNC", "}")):
            item, pos = parse(tokens, pos)
            items.append(item)
            if tokens[pos] == ("PUNC", ","):
                pos += 1
        return ("list", items), pos + 1
    if kind == "STR":
        return ("str", val), pos + 1
    if kind == "SYM":
        if pos + 1 < len(tokens) and tokens[pos + 1] == ("PUNC", "["):
            args, pos = [], pos + 2
            while not (tokens[pos] == ("PUNC", "]")):
                a, pos = parse(tokens, pos)
                args.append(a)
                if tokens[pos] == ("PUNC", ","):
                    pos += 1
            return ("call", val, args), pos + 1
        return ("sym", val), pos + 1
    raise ValueError(f"unexpected token {tokens[pos]} at {pos}")


# ---- box tree -> postfix program ------------------------------------------------------------------------
OPS = {"+": 1, "-": 1, "*": 2, "/": 2}


def _items(node):
    """The flat item list of a RowBox."""
    args = node[2]
    return args[0][1] if args and args[0][0] == "list" else args


def to_postfix(node, prog):
    t = node[0]
    if t == "str":
        s = node[1]
        if s.isdigit():
            prog.append(("c", mp.mpf(s)))
        elif s in VARS:
            prog.append(("v", VARS[s]))
        else:
            raise ValueError(f"unknown leaf {s!r} -- refusing to guess")
        return
    if t != "call":
        raise ValueError(f"unexpected node {node[:2]}")
    head, args = node[1], node[2]
    if head == "RowBox":
        _rowbox(_items(node), prog)
    elif head == "FractionBox":
        to_postfix(args[0], prog); to_postfix(args[1], prog); prog.append(("/",))
    elif head == "SuperscriptBox":
        to_postfix(args[0], prog); to_postfix(args[1], prog); prog.append(("^",))
    elif head in ("StyleBox", "FormBox", "TagBox"):
        to_postfix(args[0], prog)
    else:
        raise ValueError(f"unknown box head {head!r} -- refusing to guess")


def _rowbox(items, prog):
    """Shunting-yard over one RowBox's flat infix list. Adjacent operands (with or without Mathematica's
    " " leaf between them) multiply; a '-' with no left operand is negation."""
    stack, prev = [], "start"                     # prev in {start, operand, op, lparen}

    def push_op(op):
        while stack and stack[-1] != "(" and OPS[stack[-1]] >= OPS[op]:
            prog.append((stack.pop(),))
        stack.append(op)

    for it in items:
        s = it[1] if it[0] == "str" else None
        if s == " ":
            continue                               # implicit multiplication is decided by adjacency below
        if s in ("+", "-"):
            if prev in ("start", "op", "lparen"):
                if s == "-":
                    prog.append(("c", mp.mpf(-1))); push_op("*")
                prev = "op"
                continue
            push_op(s); prev = "op"
        elif s == "(":
            if prev == "operand":
                push_op("*")
            stack.append("("); prev = "lparen"
        elif s == ")":
            while stack[-1] != "(":
                prog.append((stack.pop(),))
            stack.pop(); prev = "operand"
        else:
            if prev == "operand":
                push_op("*")
            to_postfix(it, prog); prev = "operand"
    while stack:
        op = stack.pop()
        if op == "(":
            raise ValueError("unbalanced parenthesis")
        prog.append((op,))


def run(prog, env):
    st = []
    for op in prog:
        k = op[0]
        if k == "c":
            st.append(op[1])
        elif k == "v":
            st.append(env[op[1]])
        else:
            y = st.pop(); x = st.pop()
            st.append(x + y if k == "+" else x - y if k == "-" else x * y if k == "*" else x / y if k == "/"
                      else x ** y)
    if len(st) != 1:
        raise ValueError(f"postfix program left {len(st)} values on the stack")
    return st[0]


# ---- notebook -> cells ----------------------------------------------------------------------------------
def input_cells(raw):
    """Every Cell[BoxData[...], "Input"] in file order, as box-block strings."""
    blocks = []
    for m in re.finditer(r"Cell\[BoxData\[", raw):
        i, depth = m.start() + len("Cell[BoxData"), 0
        while True:
            if raw[i] == "[":
                depth += 1
            elif raw[i] == "]":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        if '"Input"' in raw[i + 1:i + 40]:
            blocks.append(raw[m.start() + len("Cell[BoxData"):i + 1])
    return blocks


def _leaves(node):
    if node[0] == "str":
        return [node[1]]
    if node[0] == "list":
        return [x for c in node[1] for x in _leaves(c)]
    if node[0] == "call":
        return [x for c in node[2] for x in _leaves(c)]
    return []


def cell_program(block):
    """Box block -> (name, postfix program). The cell is `NAME = RHS;`: find the RowBox holding '='."""
    toks = tokenize(block)
    tree, end = parse(toks, 1)
    if end < len(toks) - 1:
        raise SystemExit(f"parse INCOMPLETE: consumed {end}/{len(toks)} tokens")
    node = tree
    while True:
        items = _items(node)
        strs = [x[1] if x[0] == "str" else None for x in items]
        if "=" in strs:
            k = strs.index("=")
            break
        node = items[0]
    name_leaves = "".join(_leaves(("list", items[:k])))
    name = next(n for n, key in (("Omega", "CapitalOmega"), ("kappa", "Kappa"), ("phi", "Phi"),
                                 ("H1", "H1"), ("H2", "H2"), ("H3", "H3"), ("H4", "H4")) if key in name_leaves)
    rhs = [x for x in items[k + 1:] if not (x[0] == "str" and x[1] == ";")]
    prog = []
    _rowbox(rhs, prog)
    return name, prog, len(toks)


def evaluator(prog):
    def f(**kw):
        env = {"M": mp.mpf(1), "alpha": mp.mpf(1)}
        env.update(kw)
        return run(prog, env)
    return f


def metric(F, a, r, chi, zeta, M=mp.mpf(1)):
    """main.tex Eq. (metric): the sGB metric at O(zeta) with the parsed H_i. Units M = 1."""
    H = {k: F[k](a=a, r=r, chi=chi) for k in ("H1", "H2", "H3", "H4")}
    S = r * r + M * M * a * a * chi * chi
    D = r * r - 2 * M * r + M * M * a * a
    s2 = 1 - chi * chi
    gtt = -(1 - 2 * M * r / S - zeta * H["H1"])
    gtp = -(1 + zeta * H["H2"]) * 2 * M * M * a * r * s2 / S
    gpp = (1 + zeta * H["H4"]) * s2 * (r * r + M * M * a * a + 2 * M ** 3 * a * a * r * s2 / S)
    grr_up = D / (S * (1 + zeta * H["H3"]))
    xi2 = gtt - gtp * gtp / gpp
    # the O(zeta) part of xi^2 -- must vanish at r_+ if the horizon really stays at Delta = 0
    gtp0 = -2 * M * M * a * r * s2 / S
    gpp0 = s2 * (r * r + M * M * a * a + 2 * M ** 3 * a * a * r * s2 / S)
    xi2_z = H["H1"] - (2 * H["H2"] - H["H4"]) * gtp0 * gtp0 / gpp0
    return {"gtt": gtt, "gtp": gtp, "gpp": gpp, "grr_up": grr_up, "xi2": xi2, "xi2_zeta_part": xi2_z}


def kappa_from_metric(F, a, rp, zeta, chi=mp.mpf("0.3")):
    c = mp.diff(lambda r: metric(F, a, r, chi, zeta)["xi2"], rp)
    d = mp.diff(lambda r: metric(F, a, r, chi, zeta)["grr_up"], rp)
    return mp.sqrt(abs(c) * abs(d)) / 2


def a_coeffs(f, n=NDFT, rho=mp.mpf("0.5"), **kw):
    """Exact Taylor coefficients in the spin `a` at fixed (r, chi): DFT on |a| = rho. The cells are
    polynomials of degree <= 41 in `a`, so with n = 48 points there is no aliasing at all."""
    vals = [f(a=rho * mp.expjpi(mp.mpf(2 * k) / n), **kw) for k in range(n)]
    return [sum(v * mp.expjpi(-mp.mpf(2 * k * j) / n) for k, v in enumerate(vals)) / n / rho ** j
            for j in range(n)]


def main() -> None:
    mp.mp.dps = DPS
    ap = argparse.ArgumentParser()
    ap.add_argument("--nb", required=True)
    args = ap.parse_args()
    raw = Path(args.nb).read_bytes().decode("utf8", "ignore")
    report = {"source": "arXiv:2406.11986 Supplementary_materials.nb", "dps": DPS, "gates": {}}

    F, sizes = {}, {}
    for blk in input_cells(raw):
        name, prog, ntok = cell_program(blk)
        F[name] = evaluator(prog); sizes[name] = len(prog)
        print(f"  cell {name:6s}: {ntok:>8,} tokens -> {len(prog):>8,} postfix ops", flush=True)
    need = {"phi", "H1", "H2", "H3", "H4", "Omega", "kappa"}
    if need - set(F):
        raise SystemExit(f"missing cells: {sorted(need - set(F))}")
    report["postfix_ops"] = sizes

    # ---- G1: every cell evaluates ------------------------------------------------------------------------
    pt = dict(r=mp.mpf(3), chi=mp.mpf("0.3"))
    g1 = {}
    for k, f in F.items():
        t = time.time()
        g1[k] = float(f(a=mp.mpf("0.4"), **pt))
        print(f"  G1 {k:6s}(a=0.4, r=3, chi=0.3) = {g1[k]:+.12g}   ({time.time()-t:.2f} s)", flush=True)
    report["gates"]["G1_evaluates"] = g1

    # ---- G2: parity in `a`, from the parsed expression ---------------------------------------------------
    expect = {k: ("odd" if k == "Omega" else "even") for k in F}
    g2, ok2 = {}, True
    for k, f in F.items():
        c = a_coeffs(f, **pt)
        big = max(abs(x) for x in c)
        live = [j for j in range(NDFT) if abs(c[j]) > mp.mpf(10) ** (-35) * big]
        odd = [j for j in live if j % 2]
        even = [j for j in live if not j % 2]
        got = "odd" if (odd and not even) else "even" if (even and not odd) else "MIXED"
        g2[k] = {"parity": got, "expected": expect[k], "powers": live}
        ok2 &= (got == expect[k]) and max(live) <= 41
        print(f"  G2 {k:6s}: parity {got:5s} (expected {expect[k]:4s})  powers a^{live[0]}..a^{live[-1]} "
              f"({len(live)} terms)", flush=True)
    print(f"  G2 {'PASS' if ok2 else 'FAIL'}")
    report["gates"]["G2_parity"] = g2
    if not ok2:
        OUT.write_text(json.dumps(report, indent=2))
        raise SystemExit("G2 FAILED -- parity does not match the physics; nothing downstream is reportable")

    # ---- G3: horizon rigidity, Omega^(1) = Omega^(0) (H2 - H4) at r_+, independent of chi ---------------
    # ⚠️ Omega^(0) is computed from the PAPER'S OWN METRIC at zeta = 0, not copied from its printed formula.
    # The first run used the printed Omega^(0) = a/(2Mb) with b = sqrt(1-a^2) and failed by a factor that
    # tracked (1+b)/b exactly (2.005, 2.048, 2.155, 2.400 at a = 0.1..0.7) while chi-independence held to
    # 1e-44 -- the signature of a wrong prefactor, not a wrong parse. The printed form disagrees with the
    # paper's own metric (see `printed_Omega0_vs_metric` in the report); the notebook's cell does not.
    g3, printed = [], []
    TA = ("0.1", "0.3", "0.5", "0.7")
    for a in TA:
        a = mp.mpf(a); b = mp.sqrt(1 - a * a); rp = 1 + b
        om0 = -metric(F, a, rp, mp.mpf("0.3"), 0)["gtp"] / metric(F, a, rp, mp.mpf("0.3"), 0)["gpp"]
        printed.append({"a": float(a), "metric": float(om0), "printed_a_over_2Mb": float(a / (2 * b)),
                        "ratio_printed_over_metric": float(a / (2 * b) / om0)})
        lhs = F["Omega"](a=a, r=rp, chi=mp.mpf(0))
        rhs = [om0 * (F["H2"](a=a, r=rp, chi=mp.mpf(x)) - F["H4"](a=a, r=rp, chi=mp.mpf(x)))
               for x in ("0", "0.3", "0.6", "0.9")]
        spread = max(abs(x - rhs[0]) for x in rhs)
        rel = abs(rhs[0] - lhs) / abs(lhs)
        g3.append({"a": float(a), "Omega1_cell": float(lhs), "Omega0_H2_minus_H4": float(rhs[0]),
                   "rel_mismatch": float(rel), "chi_spread_rel": float(spread / abs(lhs))})
        print(f"  G3 a={float(a):.1f}: Omega1 cell {float(lhs):+.15f}   Omega0*(H2-H4) {float(rhs[0]):+.15f}"
              f"   rel diff {float(rel):.1e}   chi-spread {float(spread/abs(lhs)):.1e}", flush=True)
    # The identity holds EXACTLY for the untruncated series; both sides are truncated at a^40, so the bar is
    # the truncation scale a^42, not a fixed epsilon (a = 0.7 -> 3e-7 is expected, not a failure).
    ok3 = all(x["rel_mismatch"] < 100 * x["a"] ** 42 + 1e-30 and x["chi_spread_rel"] < 100 * x["a"] ** 42 + 1e-30
              for x in g3)
    print(f"  G3 {'PASS' if ok3 else 'FAIL'} -- three separately parsed cells satisfy the horizon identity "
          f"(bar: 100 a^42, the truncation scale)")
    for p in printed:
        print(f"     printed Omega0 a/(2Mb) / metric-derived Omega0 at a={p['a']:.1f}: "
              f"{p['ratio_printed_over_metric']:.6f}")
    report["gates"]["G3_rigidity"] = g3
    report["printed_Omega0_vs_metric"] = printed

    # ---- G5: surface gravity from the metric (main.tex) vs the kappa^(1) cell -- brings H1, H3 in -------
    # For xi^2 = g_tt - g_tphi^2/g_phiphi ~ c (r - r_+) and g^rr ~ d (r - r_+), kappa = sqrt(|c| d) / 2.
    # Control first: at zeta = 0 this must give Kerr's printed kappa^(0) = b / (2M(1+b)).
    g5 = []
    for a in TA:
        a = mp.mpf(a); b = mp.sqrt(1 - a * a); rp = 1 + b
        k0 = kappa_from_metric(F, a, rp, 0)
        eps = mp.mpf(10) ** -22
        k1 = (kappa_from_metric(F, a, rp, eps) - kappa_from_metric(F, a, rp, -eps)) / (2 * eps)
        cell = F["kappa"](a=a, r=rp, chi=mp.mpf(0))
        xi_h = metric(F, a, rp, mp.mpf("0.3"), 1)["xi2_zeta_part"]
        g5.append({"a": float(a), "kappa0_metric": float(k0), "kappa0_printed": float(b / (2 * (1 + b))),
                   "kappa1_metric": float(k1), "kappa1_cell": float(cell),
                   "rel_mismatch": float(abs(k1 - cell) / abs(cell)), "xi2_Ozeta_at_horizon": float(xi_h)})
        print(f"  G5 a={float(a):.1f}: kappa0 metric {float(k0):.12f} printed {float(b/(2*(1+b))):.12f} | "
              f"kappa1 metric {float(k1):+.12f} cell {float(cell):+.12f}  rel diff "
              f"{float(abs(k1-cell)/abs(cell)):.1e} | xi^2 O(zeta) at r_+ {float(xi_h):.1e}", flush=True)
    ok5 = all(abs(x["kappa0_metric"] - x["kappa0_printed"]) < 1e-30
              and x["rel_mismatch"] < 100 * x["a"] ** 42 + 1e-15 for x in g5)
    print(f"  G5 {'PASS' if ok5 else 'FAIL'} -- kappa^(1) cell reproduced from H1..H4 through the metric")
    report["gates"]["G5_surface_gravity"] = g5

    # ---- G4: published a^0 scalar profile, Yunes & Stein 2011 Eq. 8 -------------------------------------
    # The a^0 part of phi is chi-independent and polynomial in 1/r; sample it at 8 radii and solve exactly
    # for the 1/r^p coefficients, p = 0..7. Published: p = 1, 2, 3 only, in ratio 1 : M : 4M^2/3.
    rs = [mp.mpf(k) for k in (2, 3, 5, 7, 11, 13, 17, 19)]
    c0 = [F["phi"](a=mp.mpf(0), r=r, chi=mp.mpf("0.3")) for r in rs]
    A = mp.matrix([[r ** (-p) for p in range(len(rs))] for r in rs])
    coef = mp.lu_solve(A, mp.matrix(c0))
    c = [coef[p] for p in range(len(rs))]
    rep = {"c2_over_c1": float(c[2] / c[1]), "c3_over_c1": float(c[3] / c[1]),
           "c0_over_c1": float(c[0] / c[1]), "c4plus_over_c1": [float(x / c[1]) for x in c[4:]],
           "c1": float(c[1])}
    print(f"  G4 scalar a^0 profile, 1/r^p coefficients relative to 1/r  (M = 1):")
    print(f"       p=2   reproduced {rep['c2_over_c1']:.15f}   published 1")
    print(f"       p=3   reproduced {rep['c3_over_c1']:.15f}   published 4/3 = {4/3:.15f}")
    print(f"       p=0   reproduced {rep['c0_over_c1']:.1e}   published 0")
    print(f"       p>=4  reproduced {['%.1e' % x for x in rep['c4plus_over_c1']]}   published 0")
    ok4 = (abs(rep["c2_over_c1"] - 1) < 1e-12 and abs(rep["c3_over_c1"] - 4 / 3) < 1e-12
           and abs(rep["c0_over_c1"]) < 1e-12 and all(abs(x) < 1e-12 for x in rep["c4plus_over_c1"]))
    print(f"  G4 {'PASS' if ok4 else 'FAIL'} -- Yunes & Stein 2011 Eq. 8 (arXiv:1101.2921)")
    report["gates"]["G4_published_scalar"] = {"reproduced": rep,
                                              "published": {"c2_over_c1": 1.0, "c3_over_c1": 4 / 3},
                                              "ref": "Yunes & Stein 2011, arXiv:1101.2921, Eq. 8"}
    report["all_gates_pass"] = bool(ok2 and ok3 and ok4 and ok5)
    OUT.write_text(json.dumps(report, indent=2))
    print(f"wrote {OUT.name}   all gates: {'PASS' if report['all_gates_pass'] else 'FAIL'}")
    if not report["all_gates_pass"]:
        return
    measure(F, report)
    OUT.write_text(json.dumps(report, indent=2))
    print(f"wrote {OUT.name}")


def truncation(c, a, keep=2):
    """Relative error of keeping powers <= keep against the full a^40 series, plus the reference's own
    validity: the pre-registered rule compares the LAST term to the error being quoted; the geometric tail
    q/(1-q) |last| is reported beside it because a slowly converging series makes the last term an
    underestimate of what was cut off."""
    full = sum(c[k] * a ** k for k in range(len(c)))
    t = sum(c[k] * a ** k for k in range(keep + 1))
    err = abs(t - full)
    top = max(k for k in range(len(c)) if c[k] != 0)
    prev = max(k for k in range(top) if c[k] != 0)
    last = abs(c[top] * a ** top)
    q = abs(c[top] / c[prev]) * a ** (top - prev)
    tail = last * q / (1 - q) if q < 1 else mp.inf
    return {"rel_err": float(err / abs(full)), "full": float(full), "truncated": float(t),
            "last_term_over_err": float(last / err), "geom_tail_over_err": float(tail / err),
            "resolved": bool(last < mp.mpf("0.01") * err)}


def radius(d, windows=(6, 10, 16)):
    """Domb-Sykes on a series in u: |d_k/d_(k-1)| vs 1/k, linear over the last w ratios; R_u = 1/intercept,
    R_a = sqrt(R_u). Cross-window agreement is the convergence evidence, as for P0."""
    rat = [abs(d[k] / d[k - 1]) for k in range(1, len(d))]
    signs = "".join("+" if (d[k] / d[k - 1]) > 0 else "-" for k in range(1, len(d)))
    out = {"ratios": [float(x) for x in rat], "ratio_signs": signs, "windows": {}}
    for w in windows:
        ks = list(range(len(rat) - w + 1, len(rat) + 1))
        A = mp.matrix([[1, mp.mpf(1) / k] for k in ks])
        y = mp.matrix([rat[k - 1] for k in ks])
        sol = mp.qr_solve(A, y)[0]
        icpt = sol[0]
        out["windows"][w] = float(mp.sqrt(1 / icpt)) if icpt > 0 else None
    return out


def measure(F, report):
    """M1 and M2 exactly as pre-registered in the module docstring."""
    z = dict(r=mp.mpf(3), chi=mp.mpf(0))
    objs = {"kappa1": (F["kappa"], z), "Omega1": (F["Omega"], z)}
    for r in (3, 6):
        for chi in ("0", "0.5"):
            for k in ("H1", "H2", "H3", "H4", "phi"):
                objs[f"{k}(r={r},chi={chi})"] = (F[k], dict(r=mp.mpf(r), chi=mp.mpf(chi)))
    coefs = {}
    for name, (f, kw) in objs.items():
        c = [mp.re(x) for x in a_coeffs(f, **kw)]
        big = max(abs(x) for x in c)
        coefs[name] = [x if abs(x) > mp.mpf(10) ** -35 * big else mp.mpf(0) for x in c]

    print("\n  M1  O(a^2) truncation error against the a^40 series  (pre-registered; Kerr 220 reference 6.36% "
          "at 0.69, 18.86% at 0.90)")
    m1 = {}
    for name, c in coefs.items():
        row = {}
        for a in ("0.69", "0.90"):
            row[a] = truncation(c, mp.mpf(a))
        m1[name] = row
        cell = lambda x: (f"{100*x['rel_err']:8.2f}%" if x["resolved"] else f"{100*x['rel_err']:7.2f}%?")
        print(f"     {name:22s} a=0.69 {cell(row['0.69'])}  (last/err {row['0.69']['last_term_over_err']:.0e}, "
              f"tail/err {row['0.69']['geom_tail_over_err']:.0e})   a=0.90 {cell(row['0.90'])}  "
              f"(last/err {row['0.90']['last_term_over_err']:.0e}, tail/err {row['0.90']['geom_tail_over_err']:.0e})")
    print("     ? = UNRESOLVED by the pre-registered rule (the a^40 reference's last term is >= 1% of the error)")
    report["M1_truncation"] = m1

    # ---- M2 with its controls on the SAME scale: 21 exact coefficients in u, identical estimator ----------
    def taylor_u(g, n=21, N=128, rho=mp.mpf("0.5")):
        # By DFT, not mp.taylor: numerical differentiation to order 20 sheds digits the gates just earned.
        # Aliasing error is |c_(k+N)| rho^N ~ 0.5^128 -- nil.
        vals = [g(rho * mp.expjpi(mp.mpf(2 * k) / N)) for k in range(N)]
        return [mp.re(sum(v * mp.expjpi(-mp.mpf(2 * k * j) / N) for k, v in enumerate(vals)) / N / rho ** j)
                for j in range(n)]
    controls = {
        "kappa0 = b/(2(1+b)), R=1":            (lambda u: mp.sqrt(1 - u) / (2 * (1 + mp.sqrt(1 - u))), 1.0),
        "Omega0/a = 1/(2(1+b)), R=1":          (lambda u: 1 / (2 * (1 + mp.sqrt(1 - u))), 1.0),
        "sqrt(1-u)/(1-u/1.3), same-sign, R=1": (lambda u: mp.sqrt(1 - u) / (1 - u / mp.mpf("1.3")), 1.0),
        "sqrt(1-u/0.8)*sqrt(1-u), R=sqrt0.8":  (lambda u: mp.sqrt(1 - u / mp.mpf("0.8")) * mp.sqrt(1 - u),
                                                 float(mp.sqrt(mp.mpf("0.8")))),
    }
    print("\n  M2  radius of convergence in a (Domb-Sykes in u = a^2, windows of 6/10/16 ratios)")
    print("     CONTROLS, identical estimator, 21 exact coefficients:")
    m2 = {"controls": {}, "series": {}}
    for name, (g, truth) in controls.items():
        res = radius(taylor_u(g)); res["truth"] = truth
        m2["controls"][name] = res
        print(f"       {name:38s} truth {truth:.4f}  ->  " +
              "  ".join(f"w{w}: {v:.4f}" if v else f"w{w}: --" for w, v in res["windows"].items()))
    for name, parity in (("kappa1", 0), ("Omega1", 1)):
        d = [coefs[name][2 * k + parity] for k in range(21) if 2 * k + parity < len(coefs[name])]
        d = d[:20] if parity else d
        res = radius(d)
        m2["series"][name] = res
        print(f"     {name:40s}  ->  " + "  ".join(f"w{w}: {v:.4f}" if v else f"w{w}: --"
                                                     for w, v in res["windows"].items())
              + f"   ratio signs {res['ratio_signs']}")

    # ---- EXPLORATORY, NOT pre-registered: effective exponent at u = 1 ------------------------------------
    # For (1-u)^g the ratio is exactly 1 - (1+g)/k, so g_k = k (1 - ratio_k) - 1. The Kerr control, whose
    # true exponent is +1/2, reads 0.41 -> 0.43 at k = 15..20: this estimator runs LOW and converges slowly,
    # so only the SIGN is quoted as a finding. Negative => the correction DIVERGES at extremality.
    print("\n  EXPLORATORY (not pre-registered) effective exponent g_k at u = 1, last 6 orders:")
    expl = {}
    for name, res in [("kappa0 control (true g = +0.5)", m2["controls"]["kappa0 = b/(2(1+b)), R=1"])] + \
            list(m2["series"].items()):
        g = [k * (1 - x) - 1 for k, x in enumerate(res["ratios"], 1)]
        # g_k = g_inf + c/k over the last 6, least squares. Validated on the control before being read.
        ks = list(range(len(g) - 5, len(g) + 1))
        sol = mp.qr_solve(mp.matrix([[1, mp.mpf(1) / k] for k in ks]), mp.matrix([g[k - 1] for k in ks]))[0]
        expl[name] = {"g_k": [float(x) for x in g], "g_inf_1overk": float(sol[0])}
        print(f"     {name:32s} " + " ".join(f"{x:+.3f}" for x in g[-6:])
              + f"   -> 1/k-extrapolated {float(sol[0]):+.3f}")
    m2["exploratory_exponent"] = expl
    report["M2_radius"] = m2


if __name__ == "__main__":
    main()
