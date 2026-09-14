"""A design is a DIRECTORY, and this is the only module that reads it.

    archs/<name>/
        arch_paper.yaml    the chip, every number cited          (Timeloop reads it)
        design.yaml        what the study needs to know ABOUT the chip: label,
                           axis order, the constrained mapspace, per-design
                           modelling facts, and since 2026-09-14 the CLOCK
                           (`clock_mhz`) and the LEAKAGE DENSITIES (`leakage_nw`)
        widths.yaml        THE WIDTH TABLE, as data: q -> {spad_width, glb_width}
        weight_path.yaml   the stages a weight crosses, outer to inner
        placements.yaml    the reconstruction boundaries
        README.md          what it is and what it is not

Before ProjectRestructure phase 5, adding one architecture meant fourteen edits
across six files in three languages, and only two of them were inside that
design's own directory: its name went into `KNOWN_ARCHS`, its label into
`ARCH_LABELS`, its constrained mapspace into `archs.py`, its weight path and its
boundaries into two tables in `recon.py`, and three places in the drivers asked
`arch.startswith("eyeriss_v2")`. All of that is data now, and it is here.

    design("eyeriss_like_wglb")["label"]        what a figure axis says
    known_archs()                               every design that declares itself
    mapspace_free_levels("eyeriss_like_wglb")   prompt_3's constrained search
    flag(arch, "weights_stored_compressed")     a per-design modelling caveat

EVERY ARCHITECTURE-RELATED FACT LIVES IN THE DIRECTORY (EnvReorganisation
phase 1, 2026-09-14). The clock, the leakage densities and THE WIDTH TABLE used
to be three bash tables in env.sh, one entry per design, beside a `derived`
field nobody could override per design anyway; they are read here now and
`config._resolve()` derives the record fields (`arch_clock_mhz`, `leakage_nw`,
`weight_width_glb_mult`) from them, so every manifest still carries the same
keys with the same values and no fingerprint moved. `dram_depth` is the one
shared fact and stays in `_shared/standard.yaml` (`study.dram.depth_words`).

IT IS READ THROUGH `paths.ARCH_SRC`, so `ECC_ARCH_PIN_DIR` covers it: a
submission that pinned `archs/` at submit time keeps reading the design it
submitted, exactly as it does for the chip's own YAML.

A FILE THAT IS WRONG FAILS AT LOAD, NAMING THE FILE. `validate_design()` is the
schema: a stage of an unknown `kind`, a placement whose `reduced` set names a
stage that does not exist, a `site_stage` that is not on the path -- each is a
`ConfigError` mentioning the path, not a `KeyError` three modules later.
"""
from __future__ import annotations

import functools

import yaml

from ..contracts.errors import ConfigError
from ..paths import ARCH_SRC
from ..physics import widths
from ..settings import guards

#: The files a design directory may declare, and whether one is required.
DESIGN_FILE = "design.yaml"
WEIGHT_PATH_FILE = "weight_path.yaml"
PLACEMENTS_FILE = "placements.yaml"
WIDTHS_FILE = "widths.yaml"

#: The three densities `study/energy.py` prices standby power from, per
#: stored bit (SRAM macro / register file) and per MAC. A design that charges
#: standby (ECC_STATIC_ENERGY=1) must declare all three; a missing one is a
#: refusal at the charging site, never a zero.
LEAKAGE_KEYS = ("sram_bit", "rf_bit", "mac_instance")

#: What a weight-path stage's `kind` may be.
STAGE_KINDS = ("dram", "storage", "network")

#: What drives a boundary's reconstruction count.
SITE_COUNTERS = ("reads", "fills", "ingresses", "deliveries")

#: Directories under `archs/` that are not designs.
RESERVED = ("_shared",)


def _read(path):
    """One YAML file, or `None` if the design does not declare it."""
    if not path.is_file():
        return None
    try:
        return yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise guards.refusal("design-yaml-parse",
            f"{path}: not valid YAML -- {exc}") from None


@functools.lru_cache(maxsize=None)
def _dirs():
    """Every design directory, in DECLARED order.

    `design.yaml`'s optional `order:` says where a design sits on a default
    axis; one that omits it lands after the ordered ones, alphabetically. That
    order is a presentation choice -- but it is also the default value of
    `ECC_SWEEP_ARCHS`, and the resolved arch list is in the mapper fingerprint,
    so it is DECLARED rather than left to whatever a directory listing returns.
    """
    if not ARCH_SRC.exists():
        return ()
    found = [p for p in ARCH_SRC.iterdir()
             if p.is_dir() and p.name not in RESERVED and (p / DESIGN_FILE).is_file()]

    def _key(p):
        o = (_read(p / DESIGN_FILE) or {}).get("order")
        return (0, int(o), p.name) if o is not None else (1, 0, p.name)

    return tuple(sorted(found, key=_key))


@functools.lru_cache(maxsize=None)
def design(name):
    """One design's `design.yaml`, validated. Raises if it does not declare one."""
    path = ARCH_SRC / name / DESIGN_FILE
    doc = _read(path)
    if doc is None:
        raise guards.refusal("design-dir-missing",
            f"{name}: no {DESIGN_FILE} in {ARCH_SRC / name}.\n"
            f"  -> a design is a directory: arch_paper.yaml, design.yaml, README.md,\n"
            f"     plus weight_path.yaml and placements.yaml if it declares boundaries.\n"
            f"     `make arch NEW={name}` scaffolds one.")
    _validate_design(name, doc, path)
    return doc


def known_archs():
    """Every design that declares itself, in declaration order.

    This is `config.KNOWN_ARCHS`. A name not here is still MAPPABLE -- the
    installer looks it up in `example_designs/` as-is and says so -- but it has
    no label, no constrained mapspace and no boundaries, which is what a design
    that has not been written down yet should look like.
    """
    return tuple(p.name for p in _dirs())


def arch_labels():
    """`{name: label}` for every declared design -- what a figure axis says.

    A LABEL MAY NAME A STRUCTURE, NEVER A CAPACITY (prompt_7 C1.7): a figure
    title is the last place a reader meets the design, so it must not be the one
    place that quotes a number the file does not declare.
    """
    return {p.name: design(p.name).get("label", p.name) for p in _dirs()}


def bracket_pairs():
    """`{name: (partner, why)}` -- designs whose number is a BOUND on its own.

    A pair differs in ONE modelling judgement the design's paper does not
    settle, and `Session.setup()` writes the caveat into the manifest whenever a
    design appears without its partner. Empty today: the Eyeriss v1 pair was
    retired on 2026-09-10 when `eyeriss_like_wglb` became the design.
    """
    out = {}
    for p in _dirs():
        d = design(p.name)
        if d.get("bracket_partner"):
            out[p.name] = (d["bracket_partner"], d.get("bracket_reason", ""))
    return out


def mapspace_free_levels(name=None):
    """prompt_3's constrained mapspace: `{dim: (levels,)}`, or `{}` if unconstrained.

    With no `name`, the whole registry -- which is what `arch/patch.py` used to
    hold as a literal.
    """
    if name is not None:
        spec = design(name).get("mapspace_free_levels") or {}
        return {dim: tuple(levels or ()) for dim, levels in spec.items()}
    return {p.name: mapspace_free_levels(p.name) for p in _dirs()
            if design(p.name).get("mapspace_free_levels")}


def flag(name, key, default=None):
    """One declared per-design modelling fact, or `default` if it declares none.

    These are the three `arch.startswith("eyeriss_v2")` branches that stood in
    the drivers until phase 5: the same code path now runs for every design and
    asks it what it is.
    """
    try:
        return design(name).get(key, default)
    except ConfigError:
        return default                 # an undeclared design declares no flags


def weight_path_doc(name):
    """`weight_path.yaml`, or `None` -- a design may declare no boundaries."""
    return _read(ARCH_SRC / name / WEIGHT_PATH_FILE)


# ------------------------------------------------ the facts env.sh used to hold
def clock_mhz(name):
    """This design's declared core clock in MHz, or `None` if it declares none
    (it then runs at `ECC_GLOBAL_CYCLE_SECONDS`). `config.Config.
    cycle_seconds_for()` is the ONLY place this becomes seconds."""
    v = flag(name, "clock_mhz")
    return None if v is None else float(v)


def clock_table():
    """`{design: MHz}` for every declared design that declares a clock -- the
    record field `arch_clock_mhz`, exactly as env.sh's table used to flatten."""
    out = {}
    for p in _dirs():
        v = clock_mhz(p.name)
        if v is not None:
            out[p.name] = v
    return out


def leakage_nw(name):
    """This design's `leakage_nw: {sram_bit, rf_bit, mac_instance}` in nW, as
    floats, or `None` if it declares none."""
    v = flag(name, "leakage_nw")
    return None if v is None else {str(k): float(x) for k, x in v.items()}


def widths_doc(name):
    """`widths.yaml`, or `None` -- the design then takes THE RULE."""
    return _read(ARCH_SRC / name / WIDTHS_FILE)


@functools.lru_cache(maxsize=None)
def width_table(name):
    """THE WIDTH TABLE this design declares, as a `physics.widths.WidthTable`.

    A design with no `widths.yaml` -- and a name that is not a declared design
    at all, looked up in `example_designs/` as-is -- gets the RULE, which is
    what every design declared before the file existed and reproduces every
    row of the study's table. So a missing file colds nothing; a file that
    departs from the rule is a different chip, and its fingerprint says so.
    """
    doc = widths_doc(name)
    path = ARCH_SRC / name / WIDTHS_FILE
    if doc is None:
        return widths.WidthTable.rule(
            source=f"the rule ({name} declares no {WIDTHS_FILE})")
    return validate_widths(name, doc, path)


def placements_doc(name):
    """`placements.yaml`, or `None` -- loaded WITH the weight path, never alone."""
    return _read(ARCH_SRC / name / PLACEMENTS_FILE)


# ------------------------------------------------------------------- the schema
def validate_widths(name, doc, path):
    """`widths:` is `q -> {spad_width, glb_width}`, every width a whole
    multiple of its own q, every GLB word a whole multiple of its scratchpad's."""
    if not isinstance(doc, dict) or not isinstance(doc.get("widths"), dict) \
            or not doc["widths"]:
        raise guards.refusal("widths-not-a-table",
            f"{path}: expected `widths:` as q -> {{spad_width, glb_width}}")
    spad, glb = {}, {}
    for q, row in doc["widths"].items():
        try:
            qi = int(q)
            if not isinstance(row, dict):
                raise TypeError
            if "spad_width" in row:
                spad[qi] = int(row["spad_width"])
            if "glb_width" in row:
                glb[qi] = int(row["glb_width"])
            if not row or set(row) - {"spad_width", "glb_width"}:
                raise TypeError
        except (TypeError, ValueError):
            raise guards.refusal("widths-not-a-table",
                f"{path}: row {q!r}: {row!r} is not {{spad_width: <int>, "
                f"glb_width: <int>}} keyed by an integer q") from None
    try:
        return widths.WidthTable(spad, glb, source=str(path))
    except ValueError as exc:
        msg = str(exc)
        if "multiple of the scratchpad" in msg or "POSITIVE multiple" in msg:
            raise guards.refusal("glb-width-mult-ge-1", msg) from None
        raise guards.refusal("widths-not-a-multiple",
            f"{msg}\n  -> fix the entry in {path}; do NOT reach for a width that "
            f"suits a DIFFERENT arm (CLAUDE.md, THE WIDTH TABLE)") from None


def _validate_design(name, doc, path):
    if not isinstance(doc, dict):
        raise guards.refusal("design-not-a-mapping",
            f"{path}: expected a mapping of fields, got {type(doc).__name__}")
    if doc.get("name", name) != name:
        raise guards.refusal("design-name-mismatch",
            f"{path}: declares name {doc['name']!r} but sits in {name}/")
    if not str(doc.get("label", "")).strip():
        raise guards.refusal("design-label-empty",
            f"{path}: `label:` is what a figure axis says; it may not be empty")
    free = doc.get("mapspace_free_levels")
    if free is not None and not isinstance(free, dict):
        raise guards.refusal("design-free-levels-shape",
            f"{path}: `mapspace_free_levels:` must be dimension -> [levels]")
    if doc.get("order") is not None:
        try:
            int(doc["order"])
        except (TypeError, ValueError):
            raise guards.refusal("design-order-not-integer",
                f"{path}: `order:` must be an integer") from None
    if doc.get("clock_mhz") is not None:
        try:
            ok = float(doc["clock_mhz"]) > 0
        except (TypeError, ValueError):
            ok = False
        if not ok:
            raise guards.refusal("clock-positive",
                f"{path}: `clock_mhz: {doc['clock_mhz']!r}` -- a clock rate is a "
                f"positive number of MHz (1000 = the study default, 200 = JSSC 2017)")
    if doc.get("leakage_nw") is not None:
        lk = doc["leakage_nw"]
        bad = (not isinstance(lk, dict)
               or any(not isinstance(v, (int, float)) or isinstance(v, bool) or v < 0
                      for v in lk.values()))
        if bad:
            raise guards.refusal("leakage-not-a-table",
                f"{path}: `leakage_nw:` must map density names to non-negative "
                f"nW, e.g. {{sram_bit: 2.693, rf_bit: 70.0, mac_instance: 7844.9}}")
    band = doc.get("noc_published_share")
    if band is not None:
        if not isinstance(band, dict) or "low" not in band or "high" not in band:
            raise guards.refusal("noc-share-needs-band",
                f"{path}: `noc_published_share:` needs `low:` and `high:`")
        if not band.get("citation"):
            raise guards.refusal("noc-share-needs-citation",
                f"{path}: `noc_published_share:` is a claim about a PUBLISHED design "
                f"and must carry its citation")
        if float(band["low"]) > float(band["high"]):
            raise guards.refusal("noc-share-low-gt-high",
                f"{path}: noc_published_share low > high")


def _refuse_todo(where, field, value):
    """A citation that says TODO is a citation nobody wrote.

    `make arch NEW=<name>` writes stubs full of them ON PURPOSE: a scaffolded
    design must fail LOUDLY until it is filled in, because a stub that validated
    would be a design whose weight path nobody wrote -- and the placement study
    would report savings against a path that was invented for it.
    """
    if "TODO" in str(value):
        raise guards.refusal("design-todo",
            f"{where}: `{field}:` still says TODO.\n"
            f"  -> this design is scaffolded, not written. Every stage needs its "
            f"evidence and every boundary its description, or the numbers it "
            f"produces cannot be checked against anything.")


def validate_weight_path(name, doc, path):
    """The stages: shape, kinds, unique keys, and a reducible DRAM stage first."""
    stages = (doc or {}).get("stages")
    if not stages:
        raise guards.refusal("weight-path-empty",
            f"{path}: `stages:` is the weight path and may not be empty")
    seen = set()
    for i, s in enumerate(stages):
        where = f"{path} stage {i} ({s.get('key', '?')})"
        for req in ("key", "label", "kind", "prefixes", "reducible"):
            if req not in s:
                raise guards.refusal("weight-path-missing-field",
                    f"{where}: missing `{req}:`")
        if s["key"] in seen:
            raise guards.refusal("weight-path-duplicate-stage",
                f"{where}: duplicate stage key")
        seen.add(s["key"])
        if s["kind"] not in STAGE_KINDS:
            raise guards.refusal("weight-path-unknown-kind",
                f"{where}: kind {s['kind']!r}; one of {STAGE_KINDS}")
        if not s["prefixes"]:
            raise guards.refusal("weight-path-stage-no-prefixes",
                f"{where}: `prefixes:` names the Timeloop levels that ARE this stage; "
                f"a stage that matches no level can never be measured")
        _refuse_todo(where, "evidence", s.get("evidence", ""))
        _refuse_todo(where, "prefixes", s["prefixes"])
        _refuse_todo(where, "key", s["key"])
    if stages[0]["kind"] != "dram":
        raise guards.refusal("weight-path-starts-at-dram",
            f"{path}: the first stage must be the DRAM -- the path is written OUTER TO "
            f"INNER and every placement's `reduced` set is a prefix of it")
    return stages


def validate_placements(name, doc, stages, path):
    """The boundaries, against the stages they name."""
    places = (doc or {}).get("placements")
    if not places:
        raise guards.refusal("placements-empty",
            f"{path}: `placements:` may not be empty. A design that declares NO "
            f"boundaries declares no {PLACEMENTS_FILE} and no {WEIGHT_PATH_FILE} "
            f"at all -- three designs are in that state today.")
    keys = {s["key"] for s in stages}
    seen = set()
    for i, p in enumerate(places):
        where = f"{path} placement {i} ({p.get('key', '?')})"
        for req in ("key", "variant", "label", "short", "reduced",
                    "site_stage", "site_counter", "description"):
            if req not in p:
                raise guards.refusal("placement-missing-field",
                    f"{where}: missing `{req}:`")
        if p["key"] in seen:
            raise guards.refusal("placement-duplicate-key",
                f"{where}: duplicate placement key")
        seen.add(p["key"])
        unknown = [r for r in (p["reduced"] or ()) if r not in keys]
        if unknown:
            raise guards.refusal("placement-prefix",
                f"{where}: `reduced:` names {unknown}, which {WEIGHT_PATH_FILE} does "
                f"not declare. The two files are edited together.")
        if p["site_stage"] not in keys:
            raise guards.refusal("placement-site-stage-unknown",
                f"{where}: `site_stage: {p['site_stage']}` is not a stage of this "
                f"design's weight path")
        if p["site_counter"] not in SITE_COUNTERS:
            raise guards.refusal("placement-site-counter-unknown",
                f"{where}: site_counter {p['site_counter']!r}; one of {SITE_COUNTERS}")
        _refuse_todo(where, "description", p["description"])
        _refuse_todo(where, "rating", p.get("rating", ""))
    return places


# THE PREFIX RULE AND THE REACHABILITY RULE ARE NOT CHECKED HERE, and that is
# deliberate. `arch.arms.validate_placement_space()` states both over the
# stages a CONFIGURATION resolves (the encoder site still reshapes a network
# boundary's count). Checking the declared file instead would have refused
# `eyeriss_v2_like_wglb` AT IMPORT while its `weight_glb` stage had no boundary
# (prompt_6 Appendix B, fixed in its placements.yaml). A design is REFUSED when
# the study asks for it, with the reason -- it does not stop every other design
# from loading.
