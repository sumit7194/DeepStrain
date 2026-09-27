"""Behaviour-preservation check (The Bridge, 2026-09-27): the trigger_segments() refactor must pick exactly the
segments the REGISTERED score code picked. If they differ at all, the registered (old) choice governs.

OLD = the inline block in score() at commit 980aa5d (the last commit before the refactor). It is copied below
verbatim, and the copy is verified against `git show 980aa5d:...` as TEXT before it is run, so the "old" side
cannot silently be a paraphrase. NEW = o4a_ssm_rescore.trigger_segments. Timeline queries only; no strain data.
"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
_r = importlib.util.spec_from_file_location("r", HERE / "o4a_ssm_rescore.py")
r = importlib.util.module_from_spec(_r); _r.loader.exec_module(r)
from pbh import config as C

REGISTERED = "980aa5d"
OLD_BLOCK = '''        dets = [d for d in DETS if d[0] in t["ifos"]]
        # a 4096-s stretch fully inside each detector's DATA segments with the trigger well inside the crop
        seg = {}
        for d in dets:
            ok = get_segments(f"{d}_DATA", int(t["gps"]) - 4200, int(t["gps"]) + 4200)
            for off in (2048, 1024, 3072, 512, 3584):
                g = int(t["gps"]) - off
                if any(a <= g and g + C.SEGMENT_LEN <= b for a, b in ok) and \\
                        g + (CROP + WIN + PAD) / FS + 1 < t["gps"] < g + C.SEGMENT_LEN - CROP / FS - 1:
                    seg[d] = g; break
'''


def old_choice(t):
    """OLD_BLOCK, as a function; the text check below guarantees it is the registered code."""
    from gwosc.timeline import get_segments
    DETS, CROP, WIN, PAD, FS = r.DETS, r.CROP, r.WIN, r.PAD, r.FS
    dets = [d for d in DETS if d[0] in t["ifos"]]
    seg = {}
    for d in dets:
        ok = get_segments(f"{d}_DATA", int(t["gps"]) - 4200, int(t["gps"]) + 4200)
        for off in (2048, 1024, 3072, 512, 3584):
            g = int(t["gps"]) - off
            if any(a <= g and g + C.SEGMENT_LEN <= b for a, b in ok) and \
                    g + (CROP + WIN + PAD) / FS + 1 < t["gps"] < g + C.SEGMENT_LEN - CROP / FS - 1:
                seg[d] = g; break
    return seg


def main() -> int:
    src = subprocess.run(["git", "show", f"{REGISTERED}:primordial_blackhole_search/scripts/o4a_ssm_rescore.py"],
                         capture_output=True, text=True, check=True, cwd=HERE.parent.parent).stdout
    text_ok = OLD_BLOCK in src
    print(f"registered block at {REGISTERED} matches the copy verbatim: {text_ok}")
    rows, same = [], text_ok
    for t in r.triggers():
        if t["cls"] == "out-of-range":
            continue
        o, n = old_choice(t), r.trigger_segments(t)
        eq = o == n
        same &= eq
        rows.append({"trigger": t["name"], "cls": t["cls"], "old": o, "new": n, "identical": eq})
        print(f"{t['name']} ({t['cls']}): old {o}  new {n}  -> {'IDENTICAL' if eq else 'DIFFERENT'}")
    (r.OUT / "segchoice_check.json").write_text(json.dumps(
        {"registered_commit": REGISTERED, "text_verbatim": text_ok, "all_identical": same, "rows": rows}, indent=2))
    print(f"ALL IDENTICAL: {same}")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
