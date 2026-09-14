"""Classify a golden/fresh divergence: did a NUMBER move, or only the record?

    python3 restructure/compare.py GOLDEN FRESH

`gate.sh` runs this only after the strict `diff -ur` has already failed. Its
job is to say WHICH KIND of divergence it is, because the gate's items 2-5
protect two different things that one `diff` cannot tell apart:

    THE NUMBERS      every energy, count, cycle, exit status and percentage --
                     every numeric leaf of every JSON, every numeric cell of
                     every CSV, every numeric token of every changed stdout
                     line. A moved number is ALWAYS red. Nothing here can lift
                     that.

    THE RECORD       the `config` dict every manifest and result file carries
                     (`Config.to_dict()`), plus prose: caveat sentences, banner
                     rows, provenance notes. EnvReorganisation phases 1-4
                     DELETE knobs, and a deleted knob is a key that vanishes
                     from the record of every result -- with no number moving.
                     That is reported, not passed: the gate stays red until the
                     person running it reads the list and records the reason
                     with `--accept-record "<why>"`, exactly as `--accept-tests`
                     records a deliberate drop in the test count.

RULES, in full, so a reader can predict the verdict:

  * The two trees must hold the same files (a file that appeared or vanished
    is red). `fingerprints.tsv` and `pytest.txt` are the gate's own items 1
    and 6 and are not this script's business.
  * JSON, walked in parallel:
      - a numeric leaf (int, float, bool) must be equal, same type;
      - `None` must stay `None`;
      - a string may differ: PROSE, listed with its path;
      - a list whose elements are all strings (or empty) may change length:
        PROSE. Any other list must keep its length, elementwise as above;
      - a dict key present on ONE side only: RECORD if the subtree under it
        holds NO numeric leaf, red if it holds one (a number vanished or
        appeared). Inside a `config` dict the same rule applies and the keys
        are listed separately, because that is the list a phase has to match
        against the knobs it said it deleted -- with ONE reading peculiar to
        the record: a BOOLEAN there is a switch, not a measured quantity, so a
        config key that vanishes with only booleans (or strings, or None)
        under it is RECORD. A boolean whose VALUE changes is still red,
        inside the record or out of it (2026-09-14, phase 2: the two deleted
        switches `recon_placement_charges_decode` and
        `recon_require_group_residency`).
  * CSV: same row count; columns matched by NAME; a column on one side only
    is RECORD if every cell of it is non-numeric, red otherwise; a numeric
    cell (parses as a float) must be byte-identical; a text cell may differ:
    PROSE.
  * Any other text (stdout logs, stages.tsv, context.md): `difflib` opcodes.
    A REPLACED block must carry the same multiset of numeric tokens on both
    sides; otherwise red. Inserted or deleted lines are PROSE, flagged
    `[carries numbers]` when they do, so the reader looks at those first.

Exit 0: no number moved (the divergence is RECORD and/or PROSE only).
Exit 1: a number moved, a file appeared or vanished, or a numeric subtree did.
"""
from __future__ import annotations

import csv
import difflib
import io
import json
import pathlib
import re
import sys

#: Files the gate compares on its own, under its own rules.
SKIP = {"fingerprints.tsv", "pytest.txt", "pytest.log"}

_NUM = re.compile(r"[-+]?(?:\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?|\.\d+(?:[eE][-+]?\d+)?)")


class Report:
    def __init__(self):
        self.red = []        # a number moved / a file or numeric subtree appeared or vanished
        self.record = []     # config-record keys added or removed (no number in them)
        self.prose = []      # strings, string lists, text lines

    @property
    def ok(self):
        return not self.red


def _is_num(v):
    return isinstance(v, (int, float, bool)) and not isinstance(v, str)


def _has_num(v, bools_count=True):
    """Does a subtree hold a numeric leaf? Inside a config record a boolean is
    a switch and does not count (`bools_count=False`); everywhere else it does."""
    if isinstance(v, bool):
        return bools_count
    if _is_num(v):
        return True
    if isinstance(v, dict):
        return any(_has_num(x, bools_count) for x in v.values())
    if isinstance(v, list):
        return any(_has_num(x, bools_count) for x in v)
    return False


def _walk(a, b, path, rep, in_config=False):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            p = f"{path}/{k}"
            if k not in b or k not in a:
                side, sub = ("golden only", a[k]) if k in a else ("fresh only", b[k])
                if _has_num(sub, bools_count=not (in_config or k == "config")):
                    rep.red.append(f"{p}: key {side} and its subtree carries NUMBERS")
                elif in_config or k == "config":
                    rep.record.append(f"{p}: {side}")
                else:
                    rep.record.append(f"{p}: {side} (no number under it)")
                continue
            _walk(a[k], b[k], p, rep, in_config or k == "config")
        return
    if isinstance(a, list) and isinstance(b, list):
        strings = all(isinstance(x, str) for x in a + b)
        if len(a) != len(b):
            if strings:
                rep.prose.append(f"{path}: string list {len(a)} -> {len(b)} entries")
            else:
                rep.red.append(f"{path}: list length {len(a)} -> {len(b)} (not a string list)")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            _walk(x, y, f"{path}[{i}]", rep, in_config)
        return
    if a is None or b is None:
        if a is not b:
            rep.red.append(f"{path}: {a!r} -> {b!r}")
        return
    if _is_num(a) or _is_num(b):
        if type(a) is not type(b) or a != b:
            rep.red.append(f"{path}: {a!r} -> {b!r}")
        return
    if isinstance(a, str) and isinstance(b, str):
        if a != b:
            rep.prose.append(f"{path}: {a[:70]!r} -> {b[:70]!r}")
        return
    rep.red.append(f"{path}: {type(a).__name__} -> {type(b).__name__}")


def _floaty(s):
    try:
        float(s.replace(",", ""))
        return True
    except ValueError:
        return False


def _csv(a_text, b_text, rel, rep):
    ra = list(csv.reader(io.StringIO(a_text)))
    rb = list(csv.reader(io.StringIO(b_text)))
    if not ra or not rb:
        rep.red.append(f"{rel}: empty CSV on one side")
        return
    if len(ra) != len(rb):
        rep.red.append(f"{rel}: {len(ra)} -> {len(rb)} rows")
        return
    ha, hb = ra[0], rb[0]
    for name in sorted(set(ha) ^ set(hb)):
        side, rows, hdr = ("golden only", ra, ha) if name in ha else ("fresh only", rb, hb)
        col = hdr.index(name)
        cells = [r[col] for r in rows[1:] if col < len(r)]
        if any(_floaty(c) for c in cells if c.strip()):
            rep.red.append(f"{rel}: column {name!r} {side} and carries numbers")
        else:
            rep.record.append(f"{rel}: column {name!r} {side} (text only)")
    for name in [h for h in ha if h in hb]:
        ca, cb = ha.index(name), hb.index(name)
        for i, (x, y) in enumerate(zip(ra[1:], rb[1:]), start=2):
            va = x[ca] if ca < len(x) else ""
            vb = y[cb] if cb < len(y) else ""
            if va == vb:
                continue
            if _floaty(va) or _floaty(vb):
                rep.red.append(f"{rel}:{i} [{name}] {va!r} -> {vb!r}")
            else:
                rep.prose.append(f"{rel}:{i} [{name}] {va[:50]!r} -> {vb[:50]!r}")


def _nums(lines):
    out = []
    for ln in lines:
        out.extend(_NUM.findall(ln))
    return sorted(out)


def _text(a_text, b_text, rel, rep):
    la, lb = a_text.splitlines(), b_text.splitlines()
    sm = difflib.SequenceMatcher(a=la, b=lb, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue
        if op == "replace":
            if _nums(la[i1:i2]) != _nums(lb[j1:j2]):
                rep.red.append(f"{rel}:{i1 + 1}: numbers differ in a changed block:\n"
                               + "\n".join("      - " + l for l in la[i1:i2])
                               + "\n" + "\n".join("      + " + l for l in lb[j1:j2]))
            else:
                rep.prose.append(f"{rel}:{i1 + 1}: {i2 - i1} line(s) reworded, numbers identical")
            continue
        lines, sign = (la[i1:i2], "-") if op == "delete" else (lb[j1:j2], "+")
        for l in lines:
            flag = "  [carries numbers]" if _NUM.search(l) else ""
            rep.prose.append(f"{rel}:{i1 + 1}: {sign} {l.strip()[:90]}{flag}")


def compare(golden, fresh):
    golden, fresh = pathlib.Path(golden), pathlib.Path(fresh)
    rep = Report()
    fa = {p.relative_to(golden).as_posix() for p in golden.rglob("*") if p.is_file()}
    fb = {p.relative_to(fresh).as_posix() for p in fresh.rglob("*") if p.is_file()}
    fa = {f for f in fa if pathlib.Path(f).name not in SKIP}
    fb = {f for f in fb if pathlib.Path(f).name not in SKIP}
    for f in sorted(fa - fb):
        rep.red.append(f"{f}: in the golden snapshot, not in the fresh one")
    for f in sorted(fb - fa):
        rep.red.append(f"{f}: in the fresh snapshot, not in the golden one")
    for rel in sorted(fa & fb):
        a = (golden / rel).read_text(errors="replace")
        b = (fresh / rel).read_text(errors="replace")
        if a == b:
            continue
        if rel.endswith(".json"):
            try:
                _walk(json.loads(a), json.loads(b), rel, rep)
            except ValueError as exc:
                rep.red.append(f"{rel}: not comparable as JSON ({exc})")
        elif rel.endswith(".csv"):
            _csv(a, b, rel, rep)
        else:
            _text(a, b, rel, rep)
    return rep


def main(argv):
    if len(argv) != 3:
        sys.stderr.write("usage: python3 restructure/compare.py GOLDEN FRESH\n")
        return 2
    rep = compare(argv[1], argv[2])
    def block(title, items, cap=60):
        print(f"  {title}: {len(items)}")
        for it in items[:cap]:
            print(f"    {it}")
        if len(items) > cap:
            print(f"    ... and {len(items) - cap} more")
    block("NUMBERS MOVED (always red)", rep.red)
    block("RECORD keys added/removed (no number in them)", rep.record)
    block("PROSE (strings, text lines)", rep.prose, cap=40)
    if rep.red:
        print("  verdict: RED -- a number moved, or a numeric subtree appeared or vanished.")
        return 1
    if rep.record or rep.prose:
        print("  verdict: record/prose only -- every number identical. Read the lists,")
        print("           then record it:  bash restructure/gate.sh --accept-record '<why>'")
        return 0
    print("  verdict: identical")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
