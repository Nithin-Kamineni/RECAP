"""`run.sh validate` and `run.sh diagnose`, as two readers of the arch YAMLs.

`validate_arch()` checks a design against `archs/_shared/standard.yaml` -- the
statements that must be IDENTICAL across designs for a cross-architecture
comparison to mean anything -- plus `provenance.yaml` and `noc.yaml`, and exits
non-zero on a violation. `audit()` / `audit_findings()` report the handful of
declarations that dominate a weight-energy study: technology node, DRAM word
width, whether the global buffer keeps or bypasses Weights, how many weights fit
on chip, and every psum level narrower than its own accumulator.

Accumulator width is deliberately NOT standardized (v1 16b, v2 20b, Simba 24b,
each cited); a level that is legitimately narrower declares
`# psum-width-ok: <reason>` in the YAML and the audit honours it.

ProjectRestructure phase 3 cut this out of `archs.py`. It sits at the top of the
`arch/` package because it reads all of the rest of it.
"""
from __future__ import annotations

import re

from ..physics import widths

from .fingerprint import components_digest
from .layout import arch_levels
from .load import _blocks, _num, accumulator_bits, arch_source, arch_standard, load_noc, load_provenance, load_standard, noc_levels, noc_terms, pe_latch_pj, spatial_containers


# ------------------------------------------ standardized-comparison validation
#: Which study-wide precision a level's dataspaces are billed at.
def _expected_operand_bits(keep, bypass, std):
    """The operand width a level holding Weights and/or Inputs must declare.

    Both are standardized to the same value today, so this is normally one
    number. It is written as a lookup anyway because the two are *conceptually*
    distinct -- Task 1 says "keep these precisions explicit and distinct" -- and
    a future run that quantizes weights to 4 bits while leaving activations at
    8 must fail validation rather than silently pick one.
    """
    holds_w = ("Weights" in keep) or (not keep and "Weights" not in bypass)
    holds_i = ("Inputs" in keep) or (not keep and "Inputs" not in bypass)
    wanted = set()
    if holds_w:
        wanted.add(int(std["weight_bits"]))
    if holds_i:
        wanted.add(int(std["activation_bits"]))
    return wanted


def validate_arch(arch, cfg):
    """Check one architecture against archs/_shared/standard.yaml.

    Returns `{"arch", "violations", "notes", "facts"}`.

    A VIOLATION is a place where the design breaks the shared contract -- a
    different operand width, a different DRAM geometry, a psum level billed
    below its own accumulator. Those are bugs: they make a cross-architecture
    number mean something other than what it claims.

    A NOTE is a legitimate difference the contract expects and wants visible --
    a design-specific accumulator width, a declared `psum-width-ok` level, a
    node deviation. Notes never fail the run.

    FACTS are the quantities Task 1 says must be PRESERVED rather than
    equalised: topology, capacity, weight bypass rules. They are printed so a
    reader can confirm the standardization did not quietly flatten them.
    """
    std = load_standard()["study"]
    spec = arch_standard(arch)
    prov = load_provenance().get(arch)
    text = arch_source(arch, cfg).read_text()

    violations, notes, facts = [], [], {}

    if spec is None:
        violations.append(
            f"{arch} is not declared in archs/_shared/standard.yaml. Add it "
            f"there (with its accumulator width and a citation) before "
            f"comparing it against the others.")
        return {"arch": arch, "violations": violations, "notes": notes, "facts": facts}
    if prov is None:
        notes.append(f"no entry in archs/_shared/provenance.yaml -- the declared "
                     f"numbers are not traced to a source.")

    acc, acc_evidence = accumulator_bits(arch, cfg)
    facts["source"] = str(arch_source(arch, cfg))
    facts["origin"] = spec.get("source", "?")
    facts["label"] = spec.get("label", arch)
    facts["accumulator_bits"] = acc
    facts["accumulator_evidence"] = acc_evidence
    facts["citation"] = spec.get("citation", "")

    if spec.get("source") == "locally_authored":
        notes.append("locally authored reference dataflow -- reproduces no paper, "
                     "so its geometry is a design choice, not a citation.")
    paper_tech = spec.get("paper_technology")
    if paper_tech and paper_tech != std["technology"]:
        notes.append(f"published at {paper_tech}, modelled at {std['technology']}. "
                     f"Deliberate: a mixed-node comparison would report silicon as "
                     f"architecture.")

    # ---- technology --------------------------------------------------------
    wanted_tech = cfg.force_technology or std["technology"]
    declared = sorted(set(re.findall(r'technology:\s*"?([\w.]+)"?', text)))
    facts["technology_declared"] = declared
    for node in declared:
        if node != wanted_tech:
            violations.append(f"declares technology {node!r}; the contract says "
                              f"{wanted_tech!r}")

    # ---- interconnect ------------------------------------------------------
    # Same rule as the process node: the wire constant is study-wide metal and
    # may not differ per design; only the switching STRUCTURE is the design's
    # own, and it must be declared and cited in archs/_shared/noc.yaml.
    if cfg.noc_enabled:
        noc_entry = (load_noc().get("architectures") or {}).get(arch)
        if noc_entry is None:
            violations.append(
                f"{arch} has no entry in archs/_shared/noc.yaml, so it would map "
                f"with a free interconnect while every other design pays for one.")
        else:
            if "wire_pj_per_bit_mm" in noc_entry:
                violations.append(
                    f"noc.yaml gives {arch} its own wire_pj_per_bit_mm. The wire "
                    f"constant is shared study-wide (noc.yaml `shared:`); a "
                    f"per-design value would report metal as architecture.")
            terms = noc_terms(arch, cfg)
            wire, router, ingress = terms["wire"], terms["router"], terms["ingress"]
            levels = noc_levels(arch)
            present = spatial_containers(text)
            if levels is not None:
                for name in levels:
                    if name not in present:
                        violations.append(
                            f"noc.yaml names `{name}` as a NoC level but this design's "
                            f"spatial containers are {present}; a typo here silently "
                            f"leaves the interconnect free.")
            charged = present if levels is None else [n for n in present if n in levels]
            per_level = {n: (terms["levels"][n] if terms["levels"] else (router, ingress))
                         for n in charged}
            widths = terms["tile_width_um"]
            for name in widths:
                if name not in charged:
                    violations.append(
                        f"noc.yaml declares tile_width_um for `{name}`, which is not one "
                        f"of this design's NoC levels {charged}.")
            if widths and set(widths) != set(charged):
                violations.append(
                    f"noc.yaml declares tile_width_um for {sorted(widths)} but this design's "
                    f"NoC levels are {charged}: a container's attributes are inherited, so "
                    f"an undeclared inner level would take the outer pitch. Declare all or none.")
            facts["noc"] = {"structure": noc_entry.get("structure"),
                            "wire_pj_per_bit_mm": wire, "source": noc_entry.get("source"),
                            "router_pj_per_flit": terms["router_pj_per_flit"],
                            "operands_per_flit": terms["operands_per_flit"],
                            "pe_latch_pj": pe_latch_pj(cfg),
                            "levels": {n: {"router_pj": r, "ingress_pj": g,
                                           "tile_width_um": widths.get(n)}
                                       for n, (r, g) in per_level.items()},
                            "datapath_levels": [n for n in present if n not in charged],
                            "components_digest": components_digest()}
            notes.append(
                f"interconnect: {noc_entry.get('structure', '?')} -- wire {wire:g} pJ/b/mm "
                f"(shared); " + "; ".join(
                    f"{n}: router {r:g} pJ, ingress {g:g} pJ"
                    + (f", hop {widths[n]:g} um (declared)" if n in widths else ", hop from area")
                    for n, (r, g) in per_level.items())
                + (f"; datapath, not NoC: {[n for n in present if n not in charged]}"
                   if len(charged) != len(present) else "")
                + " (archs/_shared/noc.yaml)")
            if noc_entry.get("source") in ("published", "reference_design") \
                    and not noc_entry.get("citation_url"):
                notes.append("noc.yaml entry cites no URL for its interconnect structure.")

    # ---- per level ---------------------------------------------------------
    levels, weight_levels, fanout = [], [], 1
    for name, body in _blocks(text):
        mesh_x, mesh_y = _num(body, "meshX"), _num(body, "meshY")
        if mesh_x or mesh_y:
            fanout *= (mesh_x or 1) * (mesh_y or 1)
            continue

        cls = re.search(r"(?:class|subclass):\s*(\S+)", body)
        cls = cls.group(1) if cls else ""
        depth, width, dw = _num(body, "depth"), _num(body, "width"), _num(body, "datawidth")

        if cls == "DRAM":
            want = std["dram"]
            for key, got, exp in (("width", width, want["width_bits"]),
                                  ("datawidth", dw, want["datawidth_bits"]),
                                  ("depth", depth, want["depth_words"])):
                if got is not None and int(got) != int(exp):
                    violations.append(f"DRAM {key}={got}; the contract says {exp}")
            facts["dram"] = {"width": width, "datawidth": dw, "depth": depth}
            continue

        adder = _num(body, "adder_width")
        mult = _num(body, "multiplier_width")
        if adder or mult:
            facts["mac"] = {"multiplier_width": mult, "adder_width": adder}
            if mult is not None and int(mult) != int(std["weight_bits"]):
                violations.append(f"mac.multiplier_width={mult}; the contract "
                                  f"standardizes operands at {std['weight_bits']}b")
            if acc is not None and adder is not None and int(adder) != int(acc):
                violations.append(
                    f"mac.adder_width={adder} but this design's declared "
                    f"accumulator is {acc}b. Accumulator STORAGE and accumulator "
                    f"ARITHMETIC must agree -- the shipped simba_like got this "
                    f"wrong in exactly this way.")
            continue

        if depth is None or dw is None:
            continue

        keep = re.search(r"keep:\s*\[([^\]]*)\]", body)
        bypass = re.search(r"bypass:\s*\[([^\]]*)\]", body)
        keep = [s.strip() for s in keep.group(1).split(",")] if keep else []
        bypass = [s.strip() for s in bypass.group(1).split(",")] if bypass else []
        holds_outputs = ("Outputs" in keep) or (not keep and "Outputs" not in bypass)
        holds_weights = ("Weights" in keep) or (not keep and "Weights" not in bypass)
        ok_note = re.search(r"#\s*psum-width-ok:\s*(.+)", body)

        # Timeloop's own requirement, checked here so a bad YAML fails at
        # validate time rather than aborting the mapper on every layer.
        if width is not None and width % dw != 0:
            violations.append(f"{name}: width {width} is not a multiple of "
                              f"datawidth {dw}; Timeloop asserts "
                              f"width % datawidth == 0 and will abort.")

        if holds_outputs and not (keep and keep != ["Outputs"] and ok_note is None):
            pass  # handled below

        if keep == ["Outputs"] or (holds_outputs and not holds_weights and
                                   "Inputs" not in keep):
            # a dedicated partial-sum level: must be at the accumulator width
            if ok_note:
                notes.append(f"{name} carries Outputs at {dw}b, below the {acc}b "
                             f"accumulator, and declares why: {ok_note.group(1).strip()}")
            elif acc is not None and int(dw) != int(acc):
                violations.append(
                    f"{name} carries partial sums at datawidth {dw} but this "
                    f"design accumulates at {acc}b. Timeloop packs "
                    f"width/datawidth values per word, so this understates both "
                    f"the per-psum energy and the access count. Declare {acc}, "
                    f"or add a `# psum-width-ok: <reason>` comment if the level "
                    f"really holds requantized values.")
        else:
            wanted = _expected_operand_bits(keep, bypass, std)
            if holds_outputs and wanted:
                # A SHARED level. Timeloop allows one datawidth per level, so a
                # buffer holding both operands and partial sums must bill one of
                # them wrongly. Billing the psums at the operand width is the
                # defect that made the stock eyeriss_like look cheaper than
                # eyeriss_v2_like on every workload: 16-bit partial sums packed
                # eight to a 64-bit word and charged as bytes. The fix is to
                # split the level by dataspace, which every arch_paper.yaml
                # does. A level that legitimately holds requantized activations
                # says so with `# psum-width-ok:` and is only a note.
                if ok_note:
                    notes.append(
                        f"{name} holds Outputs alongside operands at {dw}b and "
                        f"declares why: {ok_note.group(1).strip()}")
                elif acc is not None and int(dw) < int(acc):
                    violations.append(
                        f"{name} holds partial sums AND operands at one datawidth "
                        f"({dw}b), below this design's {acc}b accumulator. "
                        f"Timeloop packs width/datawidth values per word, so the "
                        f"psums are billed as operands -- understating both their "
                        f"per-access energy and the access count. Split the level "
                        f"by dataspace (an operand half and a psum half at {acc}b), "
                        f"or add `# psum-width-ok: <reason>` if it really holds "
                        f"requantized values.")
                else:
                    notes.append(
                        f"{name} holds Outputs alongside operands; Timeloop allows "
                        f"one datawidth per level, so it is billed at the operand "
                        f"width {sorted(wanted)}.")
            if wanted and int(dw) not in wanted:
                violations.append(
                    f"{name} declares datawidth {dw}; it holds {keep or 'everything'} "
                    f"and the contract standardizes those operands at "
                    f"{sorted(wanted)}. THIS IS THE DEFECT THE CONTRACT EXISTS "
                    f"FOR: an operand width is a study-wide choice, not an "
                    f"architectural difference.")

        entries = depth * max(1, (width // dw) if (width and dw) else 1)
        levels.append({"name": name, "class": cls, "depth": depth, "width": width,
                       "datawidth": dw, "entries_each": entries,
                       "instances": fanout, "keep": keep, "bypass": bypass})
        if holds_weights:
            weight_levels.append({"name": name, "entries_each": entries,
                                  "instances": fanout,
                                  "entries_total": entries * fanout})

    facts["levels"] = levels
    facts["total_fanout"] = fanout
    facts["weight_levels"] = weight_levels
    facts["weight_capacity"] = sum(w["entries_total"] for w in weight_levels)
    facts["glb_keeps_weights"] = any(w["instances"] == 1 for w in weight_levels)
    return {"arch": arch, "violations": violations, "notes": notes, "facts": facts}

def audit(arch, cfg):
    """Return a dict of the declarations that drive a weight-energy study."""
    text = arch_source(arch, cfg).read_text()
    # The level count is taken from the PATCHED text, not this raw one: a stock
    # design that omits its DRAM depth grows a storage level when the patch adds
    # it, and the count has to match the L0..Ln the mapper will actually build.
    levels = arch_levels(arch, cfg)
    info = {
        "arch": arch,
        "source": str(arch_source(arch, cfg)),
        "fidelity": cfg.arch_fidelity,
        "loop_levels": levels,
        "victory": cfg.victory_for(levels),
        "adder_width": None,
        "technology": sorted(set(re.findall(r'technology:\s*"?([\w.]+)"?', text))),
        "levels": [],
        "spatial": [],
        "dram_datawidth": None,
        "dram_width": None,
        "glb_keeps_weights": None,
        "weight_capacity": 0,
        "weight_levels": [],
        "psum_datawidth": [],
        "psum_width_ok": {},
    }

    fanout = 1
    for name, body in _blocks(text):
        mesh_x = _num(body, "meshX")
        mesh_y = _num(body, "meshY")
        if mesh_x or mesh_y:
            n = (mesh_x or 1) * (mesh_y or 1)
            fanout *= n
            info["spatial"].append({"name": name, "meshX": mesh_x, "meshY": mesh_y,
                                    "instances": n, "cumulative_fanout": fanout})

        cls = re.search(r"(?:class|subclass):\s*(\S+)", body)
        cls = cls.group(1) if cls else ""
        adder = _num(body, "adder_width")
        if adder:
            info["adder_width"] = adder
        depth, width, dw = _num(body, "depth"), _num(body, "width"), _num(body, "datawidth")
        keep = re.search(r"keep:\s*\[([^\]]*)\]", body)
        bypass = re.search(r"bypass:\s*\[([^\]]*)\]", body)
        keep = [s.strip() for s in keep.group(1).split(",")] if keep else []
        bypass = [s.strip() for s in bypass.group(1).split(",")] if bypass else []

        if cls == "DRAM":
            info["dram_datawidth"] = dw
            info["dram_width"] = width
            continue
        if depth is None:
            continue

        entries_per_word = max(1, (width // dw) if (width and dw) else 1)
        capacity = depth * entries_per_word
        # No explicit constraint means the level keeps every dataspace.
        holds_weights = ("Weights" in keep) or (not keep and not bypass) or (
            "Weights" not in bypass and not keep)
        level = {"name": name, "class": cls, "depth": depth, "width": width,
                 "datawidth": dw, "entries": capacity, "instances": fanout,
                 "keep": keep, "bypass": bypass, "holds_weights": bool(holds_weights)}
        info["levels"].append(level)

        # Every on-chip level that carries Outputs, at the width it is BILLED
        # at. An absent constraint means the level keeps every dataspace, so
        # those count too -- that is exactly the case the stock
        # weight-/output-stationary designs are in.
        holds_outputs = ("Outputs" in keep) or (
            not keep and "Outputs" not in bypass)
        if holds_outputs and dw:
            info["psum_datawidth"].append((name, dw))
            # A level may legitimately carry Outputs below the accumulator
            # width -- Simba requantizes in the PE post-processing unit, so its
            # chip-level GlobalBuffer holds 8-bit activations, not 24-bit
            # psums. That is a claim about the design, so the design states it,
            # in the arch.yaml, next to the datawidth it justifies:
            #     datawidth: 8   # psum-width-ok: requantized before this level
            ok = re.search(r"#\s*psum-width-ok:\s*(.+)", body)
            if ok:
                info["psum_width_ok"][name] = ok.group(1).strip()

        # A shared (fanout 1) storage level is a global buffer whatever it is
        # called. The question is whether ANY such level holds Weights: that is
        # the level of weight reuse between DRAM and the PE array.
        if fanout == 1:
            info["glb_keeps_weights"] = bool(
                info["glb_keeps_weights"]) or bool(holds_weights)

        if holds_weights:
            info["weight_capacity"] += capacity * fanout
            info["weight_levels"].append(
                {"name": name, "entries_each": capacity, "instances": fanout,
                 "entries_total": capacity * fanout})

    info["total_fanout"] = fanout
    return info


def audit_findings(infos, cfg):
    """Turn a list of audits into cross-architecture warnings."""
    findings = []
    nodes = {i["arch"]: (i["technology"] or ["<inherited>"]) for i in infos}
    distinct = {tuple(v) for v in nodes.values()}
    if len(distinct) > 1 and not cfg.force_technology:
        findings.append(
            "MIXED TECHNOLOGY NODES. "
            + "; ".join(f"{a}={'/'.join(v)}" for a, v in nodes.items())
            + ". Accelergy costs components at the node declared on their container, so "
              "part of any energy gap is silicon, not architecture. "
              "Set ECC_FORCE_TECHNOLOGY=45nm to equalise.")

    dws = {i["arch"]: i["dram_datawidth"] for i in infos if i["dram_datawidth"]}
    if len(set(dws.values())) > 1 and not cfg.force_datawidth:
        findings.append(
            "MIXED DRAM DATAWIDTHS. "
            + "; ".join(f"{a}={v}b" for a, v in dws.items())
            + f". The study declares {cfg.weight_bits}-bit weights, but a design with "
              "datawidth 16 charges twice the DRAM energy per weight. "
              f"Set ECC_FORCE_DATAWIDTH={cfg.weight_bits} to equalise.")

    glb = {i["arch"]: i["glb_keeps_weights"] for i in infos}
    if len(set(glb.values())) > 1:
        keeps = [a for a, v in glb.items() if v]
        drops = [a for a, v in glb.items() if v is False]
        findings.append(
            "GLOBAL-BUFFER WEIGHT POLICY DIFFERS. "
            f"keep Weights in the GLB: {', '.join(keeps) or 'none'}; "
            f"bypass Weights: {', '.join(drops) or 'none'}. "
            "An architecture that bypasses Weights has one fewer level of weight reuse "
            "and must refetch from DRAM far more often. This is the single biggest "
            "driver of DRAM weight traffic, which is exactly what this study measures.")

    caps = {i["arch"]: i["weight_capacity"] for i in infos if i["weight_capacity"]}
    if caps:
        lo = min(caps.values())
        hi = max(caps.values())
        if hi >= 2 * lo:
            findings.append(
                "ON-CHIP WEIGHT CAPACITY SPANS >2x. "
                + "; ".join(f"{a}={v:,} weights" for a, v in
                            sorted(caps.items(), key=lambda kv: -kv[1]))
                + ". Less capacity means more DRAM refetch, independent of dataflow.")

    # ---- the one that hid the v1-vs-v2 result ------------------------------
    # A level that carries partial sums but declares a datawidth NARROWER than
    # the design's own accumulator is billing psums as operands. Timeloop packs
    # `width / datawidth` values into each physical word, so declaring a 16-bit
    # psum at datawidth 8 halves BOTH the per-scalar energy and the word count
    # -- and the error is invisible in the totals until it is compared against
    # a design that declared its psums honestly.
    under, declared_ok = [], []
    for i in infos:
        acc = i.get("adder_width")
        if not acc:
            continue
        for name, dw in i["psum_datawidth"]:
            if dw >= acc:
                continue
            reason = i.get("psum_width_ok", {}).get(name)
            if reason:
                declared_ok.append(f"{i['arch']}.{name} at {dw}b: {reason}")
            else:
                under.append(f"{i['arch']}.{name} bills {acc}b psums as {dw}b "
                             f"(x{acc // dw} too many per word)")
    if declared_ok:
        findings.append(
            "NARROWER-THAN-ACCUMULATOR OUTPUT LEVELS, DECLARED DELIBERATE. "
            + "; ".join(declared_ok)
            + ". Each carries a `# psum-width-ok:` note in its arch.yaml saying why. "
              "Listed so the claim is visible, not because anything is wrong.")
    if under:
        findings.append(
            "PARTIAL SUMS BILLED BELOW THE ACCUMULATOR WIDTH. "
            + "; ".join(under)
            + ". Timeloop packs width/datawidth values per word, so this understates "
              "both the per-psum energy and the access count at that level. It is the "
              "defect that made the stock eyeriss_like look cheaper than eyeriss_v2_like "
              "on every workload. Fix: ECC_ARCH_FIDELITY=paper (the default), which "
              "gives each design the psum precision its paper declares.")

    psums = {i["arch"]: sorted({dw for _n, dw in i["psum_datawidth"]}) for i in infos}
    psums = {a: v for a, v in psums.items() if v}
    if len({tuple(v) for v in psums.values()}) > 1:
        findings.append(
            "PARTIAL-SUM PRECISION DIFFERS. "
            + "; ".join(f"{a}={'/'.join(str(x) for x in v)}b" for a, v in psums.items())
            + ". A wider psum costs more per accumulate and, where the buffer word packs "
              "fewer psums, multiplies the number of buffer accesses. This is a real "
              "architectural difference (Eyeriss v1 accumulates at 16b, v2 at 20b, Simba "
              "at 24b), not a defect -- unlike the finding above.")

    # ---- unequal mapper effort ---------------------------------------------
    levels = {i["arch"]: (i.get("loop_levels"), i.get("victory")) for i in infos}
    levels = {a: v for a, v in levels.items() if v[0]}
    if len({v[0] for v in levels.values()}) > 1:
        detail = "; ".join(f"{a}={lv} levels -> victory {vic}"
                           for a, (lv, vic) in levels.items())
        if cfg.victory_scaling == "none":
            findings.append(
                "LOOP-NEST DEPTH DIFFERS AND MAPPER EFFORT IS FLAT. " + detail
                + ". Timeloop's search gives up after ECC_VICTORY consecutive "
                  "non-improving mappings, so the deeper design is searched less "
                  "thoroughly and the shortfall is reported as an architecture result. "
                  "Set ECC_VICTORY_SCALING=levels (the default) to scale effort with "
                  "depth.")
        else:
            findings.append(
                "LOOP-NEST DEPTH DIFFERS (effort is being scaled for it). " + detail
                + ". Scaling is a heuristic, not a proof of equal coverage: confirm by "
                  "raising ECC_VICTORY and checking the totals do not move.")

    if cfg.opt_metric != "energy":
        findings.append(
            f"MAPPER IS OPTIMISING {cfg.opt_metric.upper()}, NOT ENERGY. This study "
            "reports energy, but the mapper is choosing mappings by a different "
            "objective. Under 'edp' an architecture with more MACs can buy latency by "
            "spending energy, so the wider array is systematically pushed to a costlier "
            "mapping. Set ECC_OPT_METRIC=energy.")
    return findings
