"""Handing Timeloop an energy table, and reading back the one it used.

prompt_6. `ErtTables` is the production hook: take the base table Accelergy
generates once per arm under `<cache>/_ert/`, bump the two rows the arm declares
(`arch.fingerprint.ert_bump`), stage it per shape, and read it back after the
map -- because a cache entry's stored ERT is what proves which chip it is.

PASS `ERT:` AND `ART:` YAMLs BESIDE `design_inputs()`, AND PRE-WRITE THEM into
the output directory as `timeloop-mapper.ERT.yaml` / `.ART.yaml` FIRST. With a
supplied table Timeloop writes neither, timeloopfe's parser then raises after a
successful search, and the shape is counted failed.

SINCE PHASE C1.6 THE REFERENCE ARM HAS A SUPPLIED TABLE TOO -- no bump, only
`ECC_MAC_PJ_OVERRIDE` on every `compute` row, `set` rather than `add`, so
`apply_mac_override`'s ratio is exactly 1.0 and the mapper and the report price
one MAC. `Mapper.supplies_ert` is the test, never `ert_bump is not None`.

Every `self.bump[...]` in the class sits under a `bump is not None` guard, and
`test_phase_c.py` walks the source to keep it that way: the reference arm goes
through here too.

ProjectRestructure phase 3 cut this out of `timeloop.py`.
"""
from __future__ import annotations

import contextlib
import copy
import json
import os
import pathlib
import re
import shutil
import time
import yaml

from .cache import ShapeLock
from .inputs import ART_NAME, ERT_DIR, ERT_NAME, ERT_RECORD, MAPPING_SIDECAR, design_inputs, load_timeloopfe, problem_path, tool_versions


# ----------------------------------------------------- prompt_6: ERT tables
def ert_level_of(table_name):
    """`system_top_level.filter_glb[1..1]` -> `filter_glb` (timeloopfe's rule)."""
    return re.sub(r"\[\d+\.\.\d+\]", "", table_name).split(".")[-1]


def ert_prices(doc):
    """`{(level, action): pJ}` for every row of an ERT document."""
    return {(ert_level_of(t["name"]), a["name"]): float(a["energy"])
            for t in doc["ERT"]["tables"] for a in t["actions"]}


def art_areas(doc):
    return {ert_level_of(t["name"]): float(t["area"]) for t in doc["ART"]["tables"]}


def patched_ert(doc, changes):
    """Copy `doc` with `changes` = {(level, action): ("set"|"add", pJ)} applied.

    `arguments:` is emptied on every action: timeloopfe's `Ert` node declares no
    argument keys (the DRAM rows carry two), and Timeloop reads only `name` and
    `energy`. A change that matches nothing is an error, never a silent no-op.
    """
    out = copy.deepcopy(doc)
    pending = dict(changes)
    for t in out["ERT"]["tables"]:
        level = ert_level_of(t["name"])
        for a in t["actions"]:
            a["arguments"] = {}
            key = (level, a["name"])
            if key in pending:
                how, val = pending.pop(key)
                a["energy"] = float(val) if how == "set" else float(a["energy"]) + float(val)
    if pending:
        raise ValueError(f"no ERT row for {sorted(pending)}; the table has "
                         f"{sorted(set(ert_prices(doc)))}")
    return out


def write_yaml(path, doc):
    """LF, atomic: a concurrent reader sees the old file or the new one."""
    path = pathlib.Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", newline="\n") as fh:
        yaml.safe_dump(doc, fh, sort_keys=False)
    os.replace(tmp, path)
    return path


def ert_changes(bump):
    """The two rows an arm patches, as `patched_ert` takes them (prompt_6 5.1)."""
    return {(bump["level"], bump["action"]): ("add", bump["access_delta_pj"]),
            (bump["level"], "leak"): ("add", bump["leak_delta_pj"])}


#: The action name Timeloop's arithmetic level bills per MAC.
COMPUTE_ACTION = "compute"


def compute_levels(doc):
    """Every ERT level that bills a `compute` action -- the arithmetic ones.

    DERIVED FROM THE TABLE, never from the name `mac`. Two designs in this
    study spell the level differently and a hard-coded name would silently
    leave one of them priced at Accelergy's own number while the report
    charged another (prompt_6 RULE 1: one owner per effect).
    """
    return sorted({lvl for (lvl, action) in ert_prices(doc)
                   if action == COMPUTE_ACTION})


def mac_ert_changes(doc, mac_pj):
    """prompt_7 C1.6: price every arithmetic row at `ECC_MAC_PJ_OVERRIDE`.

    THE PROBLEM. `ECC_MAC_PJ_OVERRIDE` was evaluator-only: `energy.apply_mac_
    override()` rescales the Compute category AFTER the raw cache, so the
    mapper optimised a machine whose MAC cost Accelergy's 1.13555 pJ while
    every published percentage was computed at Horowitz's 0.23. Under
    `ECC_OPT_METRIC=edp` a 5x-too-expensive MAC can move the argmax -- the run
    printed that warning and nothing acted on it.

    `set`, NOT `add`: the override REPLACES a price rather than tolling it, so
    the evaluator's ratio `(macs x override) / compute` comes back exactly 1.0
    and the two cannot double-count. `tests/test_phase_c.py` asserts that.

    Returns `{}` for `mac_pj` None, which is what reproduces the pre-Phase-C
    table byte for byte.
    """
    if mac_pj is None:
        return {}
    levels = compute_levels(doc)
    if not levels:
        raise ValueError("no ERT row bills a `compute` action; this design has "
                         "no arithmetic level to price a MAC at")
    return {(lvl, COMPUTE_ACTION): ("set", float(mac_pj)) for lvl in levels}


#: Tolerance for "the same delta": RULE 4.4.5 asks for 1e-9 of the intended value.
ERT_TOL = 1e-9


def _close(a, b, tol=ERT_TOL):
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))


def same_bump(a, b):
    """Do two bump records describe one toll? Identity fields exactly, deltas to ERT_TOL."""
    if a is None or b is None:
        return a is None and b is None
    return (a.get("placement") == b.get("placement") and a.get("level") == b.get("level")
            and a.get("action") == b.get("action")
            and _close(a.get("access_delta_pj"), b.get("access_delta_pj"))
            and _close(a.get("leak_delta_pj"), b.get("leak_delta_pj")))


class ErtMismatch(SystemExit):
    """A cache entry's ERT is not the one the bar asking for it expects."""


def describe_bump(bump):
    if bump is None:
        return "reference (no ERT bump)"
    return (f"{bump['placement']}: {bump['level']}.{bump['action']} "
            f"+= {bump['access_delta_pj']:.9g} pJ, {bump['level']}.leak "
            f"+= {bump['leak_delta_pj']:.9g} pJ/instance/cycle")


class ErtTables:
    """The supplied ERT/ART of ONE ERT arm, generated once and reused per shape.

    Lives at `<out_root>/_ert/`, i.e. beside the shape entries of the arm's own
    cache directory (own slug, own fingerprint -- RULE 4.4.5). Holds:

        base.ERT.yaml / base.ART.yaml   Accelergy's own table for THIS arm's
                                        patched arch (deterministic, FINDINGS
                                        3.5, so it needs no cache entry to exist)
        timeloop-mapper.ERT.yaml        the base with the arm's two rows bumped
        timeloop-mapper.ART.yaml        the base ART, unchanged
        ert_bump.json                   the bump, the base and patched prices of
                                        the bumped rows, tool versions

    `ensure()` generates under a `ShapeLock`, so concurrent jobs of one arm
    build it once; a set already on disk is reused only if its record matches
    the requested bump.
    """

    def __init__(self, cfg, arch, arch_yaml, out_root, bump, mac_pj=None):
        if bump is None and mac_pj is None:
            raise ValueError(
                "ErtTables patches nothing: no ERT bump and no MAC price. The "
                "reference arm needs a supplied table only when "
                "ECC_MAC_PJ_OVERRIDE is set (prompt_7 C1.6); without either, "
                "let Accelergy write its own.")
        self.cfg, self.arch, self.arch_yaml, self.bump = cfg, arch, arch_yaml, bump
        #: prompt_7 C1.6: the MAC price the MAPPER optimises against, or None.
        #: It goes into `base.ERT.yaml` as well as the patched one, so it is
        #: part of what this arm's architecture COSTS rather than a toll on top
        #: of it -- which is what keeps `read_back_ert`'s "every other row
        #: untouched" check meaningful with a bump present.
        self.mac_pj = None if mac_pj is None else float(mac_pj)
        self.dir = pathlib.Path(out_root) / ERT_DIR
        self.ert = self.dir / ERT_NAME
        self.art = self.dir / ART_NAME
        self.base_ert = self.dir / "base.ERT.yaml"
        self.base_art = self.dir / "base.ART.yaml"
        self.record_path = self.dir / ERT_RECORD
        self.record = None

    def what_it_patches(self):
        """One phrase naming every row this arm's table changes.

        The reference arm patches only the MAC price, so `describe_bump` alone
        is not enough and `self.bump['placement']` is a crash.
        """
        parts = [describe_bump(self.bump)]
        if self.mac_pj is not None:
            parts.append(f"MAC {self.mac_pj:g} pJ")
        return ", ".join(parts)

    # ------------------------------------------------------------ validity
    def _load(self):
        """The record on disk if it describes THIS bump and MAC price, and the
        files agree with it."""
        for f in (self.ert, self.art, self.base_ert, self.base_art, self.record_path):
            if not f.exists():
                return None
        try:
            rec = json.loads(self.record_path.read_text())
            if not same_bump(rec.get("bump"), self.bump):
                return None
            # prompt_7 C1.6: a table generated before the MAC price existed,
            # or at a different one, is not this arm's table. Without this the
            # `_ert/` directory would survive a change to ECC_MAC_PJ_OVERRIDE
            # and the mapper would optimise against the old price.
            got_mac = rec.get("mac_pj")
            if (got_mac is None) != (self.mac_pj is None):
                return None
            if self.mac_pj is not None and not _close(got_mac, self.mac_pj):
                return None
            prices = ert_prices(yaml.safe_load(self.ert.read_text()))
            if self.bump is not None:
                for action, want in (rec.get("patched_pj") or {}).items():
                    if not _close(prices.get((self.bump["level"], action)), want):
                        return None
            for lvl in (rec.get("mac_levels") or []):
                if not _close(prices.get((lvl, COMPUTE_ACTION)), self.mac_pj):
                    return None
        except (OSError, ValueError, KeyError, TypeError):
            return None
        return rec

    def ensure(self, layer):
        """Generate the tables if they are not already on disk for this bump."""
        self.record = self._load()
        if self.record is not None:
            return self
        lock = ShapeLock(self.dir)
        # `describe_bump(None)` is "reference (no ERT bump)" -- since prompt_7
        # C1.6 the REFERENCE arm builds a table too, carrying only the MAC
        # price, so nothing here may assume a bump exists.
        lock.acquire(shape=f"{self.arch} ERT tables ({self.what_it_patches()})")
        try:
            self.record = self._load()
            if self.record is None:
                self._generate(layer)
        finally:
            lock.release()
        return self

    def _generate(self, layer):
        """Accelergy on THIS arm's patched arch, then the two bumped rows."""
        self.dir.mkdir(parents=True, exist_ok=True)
        scratch = self.dir / f"accelergy.{os.getpid()}"
        scratch.mkdir(parents=True, exist_ok=True)
        tl = load_timeloopfe()
        spec = tl.Specification.from_yaml_files(
            *design_inputs(self.arch_yaml, problem_path(layer), self.arch, self.cfg))
        with open(scratch / "accelergy_console.log", "w") as logf, \
                contextlib.redirect_stdout(logf), contextlib.redirect_stderr(logf):
            tl.call_accelergy_verbose(spec, output_dir=str(scratch),
                                      log_to=str(scratch / "accelergy.log"))
        erts = sorted(p for p in scratch.glob("*ERT.yaml") if "summary" not in p.name)
        arts = sorted(p for p in scratch.glob("*ART.yaml") if "summary" not in p.name)
        if not erts or not arts:
            raise SystemExit(f"ErtTables: accelergy wrote no ERT/ART under {scratch}; "
                             f"see {scratch / 'accelergy_console.log'}")
        base_ert = yaml.safe_load(erts[0].read_text())
        base_art = yaml.safe_load(arts[0].read_text())
        prices = ert_prices(base_ert)
        # prompt_7 C1.6: the MAC price is part of what THIS ARCHITECTURE COSTS,
        # so it lands in `base.ERT.yaml` too -- not as a toll on top of it.
        # That is what keeps `read_back_ert`'s "every other row untouched"
        # check meaningful once a bump is also present: the base it compares
        # against is the table the mapper was actually given, minus the bump.
        mac_changes = mac_ert_changes(base_ert, self.mac_pj)
        mac_levels = sorted({lvl for (lvl, _a) in mac_changes})
        base_doc = patched_ert(base_ert, mac_changes)
        base_prices = ert_prices(base_doc)
        record = {
            "bump": self.bump,
            "mac_pj": self.mac_pj,
            "mac_levels": mac_levels,
            "mac_base_pj": {lvl: prices[(lvl, COMPUTE_ACTION)] for lvl in mac_levels},
            "arch_yaml": str(self.arch_yaml),
            "tool_versions": tool_versions(),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        if self.bump is None:
            patched = base_doc
        else:
            level = self.bump["level"]
            for action in (self.bump["action"], "leak"):
                if (level, action) not in prices:
                    raise SystemExit(f"ErtTables: the generated ERT has no row "
                                     f"{level}.{action}; it has "
                                     f"{sorted(k for k in prices if k[0] == level)}")
            patched = patched_ert(base_doc, ert_changes(self.bump))
            pprices = ert_prices(patched)
            record["base_pj"] = {a: base_prices[(level, a)]
                                 for a in (self.bump["action"], "leak")}
            record["patched_pj"] = {a: pprices[(level, a)]
                                    for a in (self.bump["action"], "leak")}
        write_yaml(self.base_ert, base_doc)
        write_yaml(self.base_art, base_art)
        write_yaml(self.ert, patched)
        write_yaml(self.art, base_art)
        tmp = self.record_path.with_name(f".{ERT_RECORD}.{os.getpid()}.tmp")
        with open(tmp, "w", newline="\n") as fh:
            fh.write(json.dumps(record, indent=1, default=str))
        os.replace(tmp, self.record_path)
        shutil.rmtree(scratch, ignore_errors=True)
        self.record = record
        note = f"      [ert] {self.arch}: {describe_bump(self.bump)}"
        if self.bump is not None:
            act = self.bump["action"]
            note += (f"  (base {self.bump['level']}.{act} "
                     f"{record['base_pj'][act]:g} -> {record['patched_pj'][act]:g} pJ, "
                     f"leak {record['base_pj']['leak']:g} -> "
                     f"{record['patched_pj']['leak']:g})")
        if self.mac_pj is not None:
            note += ("; MAC " + ", ".join(
                f"{lvl}.{COMPUTE_ACTION} {record['mac_base_pj'][lvl]:g} -> "
                f"{self.mac_pj:g} pJ" for lvl in mac_levels)
                + " (ECC_MAC_PJ_OVERRIDE, now in the MAPPER's objective too)")
        print(note, flush=True)

    # --------------------------------------------------------------- use
    def stage(self, out_dir):
        """Pre-write the two tables into a shape's output directory."""
        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.ert, out_dir / ERT_NAME)
        shutil.copyfile(self.art, out_dir / ART_NAME)

    def inputs(self):
        """The two extra YAMLs `design_inputs()` is followed by."""
        return [str(self.ert), str(self.art)]

    def sidecar_record(self):
        """What a mapping sidecar records about its ERT (RULE 4.4.5)."""
        rec = self.record or {}
        return {"bump": self.bump, "base_pj": rec.get("base_pj"),
                "patched_pj": rec.get("patched_pj"),
                "mac_pj": rec.get("mac_pj", self.mac_pj),
                "mac_levels": rec.get("mac_levels"),
                "mac_base_pj": rec.get("mac_base_pj")}


def read_back_ert(out_dir, bump, base_prices=None, tol=ERT_TOL, mac_pj=None):
    """RULE 4.4.5, defence 3: re-open the ERT stored beside a cache entry and
    assert it is the one the bar asking for it expects.

    `bump` is the requesting bar's `archs.ert_bump()` (None for a reference bar).
    Checks, and raises `ErtMismatch` naming both sides on the first failure:

    * the sidecar's recorded bump has the same placement, level and action, and
      each delta is within `tol` of the intended value;
    * the stored table's bumped rows equal the recorded base price plus the
      intended delta, to `tol`;
    * with `base_prices` (the un-bumped table, e.g. the reference entry's ERT),
      the recorded base equals it on the bumped rows and every other row is
      untouched.

    Returns a dict describing what was verified.
    """
    out_dir = pathlib.Path(out_dir)
    side = out_dir / MAPPING_SIDECAR
    if not side.exists():
        raise ErtMismatch(f"read_back_ert: {out_dir} has no {MAPPING_SIDECAR}")
    rec = json.loads(side.read_text()).get("ert_bump")
    # prompt_7 C1.6: an entry can now carry an ERT record with NO bump -- the
    # reference arm's, whose table exists only to price the MAC. So the test
    # is on the BUMP, not on the presence of a record.
    verified = _read_back_mac(out_dir, rec, mac_pj, tol)
    if bump is None:
        if (rec or {}).get("bump") is not None:
            raise ErtMismatch(
                f"read_back_ert: {out_dir} was mapped as an ERT arm "
                f"({describe_bump(rec.get('bump'))}) but the reference arm asked for it")
        return {"arm": "reference", "verified": ["no ERT bump recorded"] + verified}
    if rec is None or rec.get("bump") is None:
        raise ErtMismatch(
            f"read_back_ert: {out_dir} records no ERT bump, but "
            f"{describe_bump(bump)} asked for it")
    got = rec["bump"]
    if not same_bump(got, bump):
        raise ErtMismatch(
            f"read_back_ert: {out_dir} was mapped as\n    {describe_bump(got)}\n"
            f"  but the bar asking for it is\n    {describe_bump(bump)}")
    ert_path = out_dir / ERT_NAME
    if not ert_path.exists():
        raise ErtMismatch(f"read_back_ert: {ert_path} is missing")
    prices = ert_prices(yaml.safe_load(ert_path.read_text()))
    level = bump["level"]
    for action, delta in ((bump["action"], bump["access_delta_pj"]),
                          ("leak", bump["leak_delta_pj"])):
        stored = prices.get((level, action))
        base = (rec.get("base_pj") or {}).get(action)
        if stored is None or base is None:
            raise ErtMismatch(f"read_back_ert: {out_dir}: no stored/base price for "
                              f"{level}.{action} (stored {stored}, base {base})")
        if not _close(stored - base, delta, tol):
            raise ErtMismatch(
                f"read_back_ert: {out_dir}: {level}.{action} stored {stored!r} - base "
                f"{base!r} = {stored - base!r}, but {describe_bump(bump)} intends "
                f"{delta!r} (tolerance {tol:g})")
        if base_prices is not None and not _close(base, base_prices.get((level, action)), tol):
            raise ErtMismatch(
                f"read_back_ert: {out_dir}: recorded base {level}.{action} = {base!r} "
                f"but the un-bumped table says {base_prices.get((level, action))!r}")
        verified.append(f"{level}.{action}: {base:g} + {delta:.9g} = {stored:g}")
    if base_prices is not None:
        bumped = {(level, bump["action"]), (level, "leak")}
        for key, want in base_prices.items():
            if key in bumped:
                continue
            if not _close(prices.get(key), want, tol):
                raise ErtMismatch(
                    f"read_back_ert: {out_dir}: {key[0]}.{key[1]} = {prices.get(key)!r} "
                    f"differs from the un-bumped table's {want!r}; only "
                    f"{level}.{bump['action']} and {level}.leak may move")
        verified.append(f"{len(base_prices) - len(bumped)} other rows untouched")
    return {"arm": bump["placement"], "level": level, "action": bump["action"],
            "verified": verified}


def _read_back_mac(out_dir, rec, mac_pj, tol=ERT_TOL):
    """RULE 4.4.5 for the MAC price (prompt_7 C1.6): the entry was mapped at
    the price this run charges, or it is not this run's entry.

    Checks the recorded price AND the stored table's own `compute` rows, so a
    sidecar that merely CLAIMS the price cannot pass on its own.
    """
    got = (rec or {}).get("mac_pj")
    if mac_pj is None and got is None:
        return []
    if (mac_pj is None) != (got is None):
        raise ErtMismatch(
            f"read_back_ert: {out_dir} was mapped with MAC price {got!r} but "
            f"this run charges {mac_pj!r}. ECC_MAC_PJ_OVERRIDE is in the "
            f"mapper's objective since prompt_7 C1.6, so the two are different "
            f"architectures, not two ways of reporting one.")
    if not _close(got, mac_pj, tol):
        raise ErtMismatch(f"read_back_ert: {out_dir} was mapped at "
                          f"{got!r} pJ/MAC; this run charges {mac_pj!r}")
    ert_path = pathlib.Path(out_dir) / ERT_NAME
    if not ert_path.exists():
        raise ErtMismatch(f"read_back_ert: {ert_path} is missing, so the MAC "
                          f"price it was mapped at cannot be verified")
    prices = ert_prices(yaml.safe_load(ert_path.read_text()))
    levels = (rec or {}).get("mac_levels") or []
    for lvl in levels:
        if not _close(prices.get((lvl, COMPUTE_ACTION)), mac_pj, tol):
            raise ErtMismatch(
                f"read_back_ert: {out_dir}: {lvl}.{COMPUTE_ACTION} = "
                f"{prices.get((lvl, COMPUTE_ACTION))!r} in the stored table, but the "
                f"sidecar says it was mapped at {mac_pj!r}")
    return [f"{lvl}.{COMPUTE_ACTION} = {mac_pj:g} pJ (ECC_MAC_PJ_OVERRIDE)"
            for lvl in levels]


