"""The centralized result writer. One place that decides where a result goes.

Implements `04_results_storage_spec.txt`. Nothing else in this package writes
an evaluation JSON: an experiment builds a `ResultBuilder`, adds its variants,
and calls `write()`. That is the whole interface, and it exists because the
previous arrangement -- each script inventing its own JSON -- is how two files
claiming to be the same experiment end up with different fields.

THE NAMESPACE
-------------
    results/evaluation/{Pre|Post}/{ARCH}/{MODEL}/{BCH}/{PREC}/{SCOPE}/{MAPPER}/{RUNID}.json

`Pre`   the mapping was chosen without knowing about reconstruction; the ECC
        effect is applied during energy evaluation only.
`Post`  the mapping itself was optimised for the reduced weight width.
THE PHASE IS DERIVED PER ARM (`settings.run.result_phase`, 2026-09-14): Task 1
and Task 2 are `Pre` by construction, a placement mapped on its own chip is
`Post`. `ECC_PHASE` is gone; nothing lands anywhere it did not before.

The four segments after `{BCH}` are the collision guards the spec asks for, and
`legacy/docs/RESULTS_SCHEMA.md` explains why each is needed. `{RUNID}` is
`<UTC timestamp>__<config hash>`, so a re-run NEVER silently overwrites a prior
one; `ECC_OVERWRITE=1` is required to replace a file that already exists, and
even then only that exact run id.

THE DOCUMENT SHAPE
------------------
Summary first, detail after. A reader opening the file sees, in order: what the
experiment was, which variants ran, what each cost, and how much was saved --
before any of the machinery. Everything needed to reproduce or audit the run is
below that, in `detail`.

MISSING DATA IS NEVER INVENTED
------------------------------
A variant that was not implemented, not feasible or failed is recorded with
`status` and `unavailable_reason` and a `total_energy_pJ` of `null`. It is
never given a plausible-looking number. Task 1 writes `embedded` and every
reconstruction placement exactly this way, because Task 1 has not implemented
them.
"""
from __future__ import annotations

import datetime
import getpass
import hashlib
import json
import os
import platform
import socket
import subprocess

from ..settings.run import result_phase

SCHEMA_VERSION = "1.0"

#: Every status a variant may carry. Anything else is a bug in the caller.
VARIANT_STATUSES = (
    "evaluated",          # a real number, produced by this run
    "not_implemented",    # the model for it does not exist yet (Tasks 2-5)
    "unsupported",        # infeasible for this architecture, with a reason
    "failed",             # attempted and did not produce a result
)

#: Statuses that must NOT carry an energy number.
_NO_ENERGY = ("not_implemented", "unsupported", "failed")


class ResultError(RuntimeError):
    """A result that would be wrong or unauditable if written."""


def _utc_now():
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0)


def savings_percent(reference_energy, variant_energy):
    """`100 * (reference - variant) / reference`, per the storage spec.

    Negative when the variant costs more than its reference, which is a result,
    not an error. `None` when either side is unavailable -- a saving against an
    unknown reference is not a number, and writing 0.0 there would read as "no
    change" rather than "not known".
    """
    if reference_energy in (None, 0) or variant_energy is None:
        return None
    return 100.0 * (reference_energy - variant_energy) / reference_energy


def _git_commit(root):
    try:
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            return out.stdout.strip()
        return f"unavailable (not a git repository: {out.stderr.strip()[:80]})"
    except Exception as exc:
        return f"unavailable ({type(exc).__name__})"


class Variant:
    """One ECC treatment of one architecture: a name, a status, and its energy.

    `energy_by_component_pJ` is the per-component breakdown the spec asks for.
    `extra` carries whatever is specific to this variant -- the parity account
    for the conventional baseline, the placement and reconstruction counts for
    a Task 3 variant.
    """

    def __init__(self, name, kind, status="evaluated", total_energy_pJ=None,
                 energy_by_component_pJ=None, unavailable_reason=None,
                 mapping_ids=None, extra=None, label=None):
        if status not in VARIANT_STATUSES:
            raise ResultError(f"variant {name!r}: status {status!r} is not one of "
                              f"{', '.join(VARIANT_STATUSES)}")
        if status == "evaluated" and total_energy_pJ is None:
            raise ResultError(f"variant {name!r} is marked evaluated but has no "
                              f"total_energy_pJ. A result with no number is not "
                              f"an evaluated result.")
        if status in _NO_ENERGY:
            if total_energy_pJ is not None:
                raise ResultError(
                    f"variant {name!r} has status {status!r} but carries an "
                    f"energy of {total_energy_pJ}. A variant that did not run "
                    f"must not carry a number -- that is how a placeholder "
                    f"becomes a result.")
            if not unavailable_reason:
                raise ResultError(
                    f"variant {name!r} has status {status!r} and no "
                    f"unavailable_reason. The spec requires unavailable values "
                    f"to explain why they are unavailable.")
        self.name = name
        self.label = label or name
        self.kind = kind            # baseline | embedded | reconstruction
        self.status = status
        self.total_energy_pJ = total_energy_pJ
        self.energy_by_component_pJ = energy_by_component_pJ or {}
        self.unavailable_reason = unavailable_reason
        self.mapping_ids = sorted(mapping_ids or [])
        self.extra = extra or {}

    def to_dict(self):
        return {
            "name": self.name,
            "label": self.label,
            "kind": self.kind,
            "status": self.status,
            "total_energy_pJ": self.total_energy_pJ,
            "total_energy_uJ": (None if self.total_energy_pJ is None
                                else self.total_energy_pJ / 1e6),
            "unavailable_reason": self.unavailable_reason,
            "energy_by_component_pJ": self.energy_by_component_pJ,
            "mapping_ids": self.mapping_ids,
            "detail": self.extra,
        }


class ResultBuilder:
    """Collects one experiment's variants and writes exactly one JSON.

    `reference` names the conventional-ECC baseline; `embedded_reference` names
    the embedded-only variant. Savings are reported against BOTH, as the spec
    requires, and are `null` where the reference itself is unavailable rather
    than silently 0.
    """

    def __init__(self, cfg, results, arch, model, *, experiment,
                 fixed_mapping, reference="baseline_external_parity",
                 embedded_reference="embedded_ecc", phase=None):
        self.cfg = cfg
        self.results = results
        self.arch = arch
        self.model = model
        self.experiment = experiment
        self.fixed_mapping = bool(fixed_mapping)
        self.reference = reference
        self.embedded_reference = embedded_reference
        # The RESOLVED configuration's experiment, not this builder's label:
        # the placement study labels its file `reconstruction_placement` and
        # runs under ECC_EXPERIMENT=recon, and it is the latter that says a
        # placement was mapped on its own chip.
        self.phase = phase or result_phase(getattr(cfg, "experiment", experiment),
                                           getattr(cfg, "recon_optimizer", True))
        self.variants = []
        self.warnings = []
        self.approximations = []
        self.validation = []
        self.detail = {}
        self.started = _utc_now()

    # ------------------------------------------------------------- assembling
    def add(self, variant):
        if any(v.name == variant.name for v in self.variants):
            raise ResultError(f"variant {variant.name!r} added twice")
        self.variants.append(variant)
        return variant

    def warn(self, text):
        self.warnings.append(text)

    def approximate(self, text):
        self.approximations.append(text)

    def check(self, name, passed, detail=None):
        """Record a validation that ran, whether or not it passed."""
        self.validation.append({"check": name, "passed": bool(passed),
                                "detail": detail})
        return passed

    def set_detail(self, **kw):
        self.detail.update(kw)

    # -------------------------------------------------------------- identity
    def config_hash(self):
        """Everything that defines this experiment, hashed for the run id."""
        blob = json.dumps({
            "experiment": self.experiment, "phase": self.phase,
            "arch": self.arch, "model": self.model,
            "config": self.cfg.to_dict(),
        }, sort_keys=True, default=str)
        return hashlib.sha1(blob.encode()).hexdigest()[:8]

    def run_id(self):
        return f"{self.started.strftime('%Y%m%dT%H%M%SZ')}__{self.config_hash()}"

    # -------------------------------------------------------------- assembly
    def _energy(self, name):
        for v in self.variants:
            if v.name == name:
                return v.total_energy_pJ
        return None

    def _verify_fixed_mapping(self):
        """A `Pre` result must use ONE mapping set for every evaluated variant.

        The storage spec asks this to be recorded AND verified, not asserted.
        Task 1 has a single evaluated variant so the check is trivially true;
        it exists now so that Tasks 2 and 3 cannot quietly break it.
        """
        sets = {v.name: tuple(v.mapping_ids) for v in self.variants
                if v.status == "evaluated" and v.mapping_ids}
        distinct = set(sets.values())
        if not self.fixed_mapping:
            return self.check(
                "fixed_mapping_shared_across_variants", True,
                "not applicable: this result is labelled reconstruction-aware "
                "mapping, so each variant is allowed its own mapping")
        ok = len(distinct) <= 1
        return self.check(
            "fixed_mapping_shared_across_variants", ok,
            ("all evaluated variants reference the same mapping set"
             if ok else
             f"evaluated variants do NOT share one mapping set: {sets}. A "
             f"fixed-mapping comparison whose variants were mapped differently "
             f"is not a fixed-mapping comparison."))

    def _validate(self):
        if not self.variants:
            raise ResultError("no variants: refusing to write an empty result")
        names = [v.name for v in self.variants]
        if self.reference not in names:
            raise ResultError(
                f"reference variant {self.reference!r} is not among {names}. "
                f"Savings are defined against it, so it has to be present -- "
                f"even if only as an explicitly unavailable entry.")
        for v in self.variants:
            if v.status != "evaluated":
                continue
            if not v.energy_by_component_pJ:
                self.warn(f"variant {v.name!r} has no per-component energy "
                          f"breakdown; only its total can be audited")
            total = sum(float(x) for x in v.energy_by_component_pJ.values())
            if v.energy_by_component_pJ and v.total_energy_pJ:
                drift = abs(total - v.total_energy_pJ) / v.total_energy_pJ
                self.check(f"components_sum_to_total[{v.name}]", drift < 1e-9,
                           f"components sum to {total:.6f} pJ, total is "
                           f"{v.total_energy_pJ:.6f} pJ")
        self._verify_fixed_mapping()

    def summary(self):
        ref = self._energy(self.reference)
        emb = self._energy(self.embedded_reference)
        rows = []
        for v in self.variants:
            rows.append({
                "variant": v.name,
                "kind": v.kind,
                "status": v.status,
                "total_energy_pJ": v.total_energy_pJ,
                "total_energy_uJ": (None if v.total_energy_pJ is None
                                    else v.total_energy_pJ / 1e6),
                "savings_vs_conventional_ecc_percent":
                    savings_percent(ref, v.total_energy_pJ),
                "savings_vs_embedded_only_percent":
                    savings_percent(emb, v.total_energy_pJ),
                "unavailable_reason": v.unavailable_reason,
            })
        return rows

    def best_reconstruction(self):
        """Lowest-energy EVALUATED reconstruction placement, or an explanation."""
        cands = [v for v in self.variants
                 if v.kind == "reconstruction" and v.status == "evaluated"]
        if not cands:
            return {"variant": None, "reason":
                    "no reconstruction placement has been evaluated in this run"}
        best = min(cands, key=lambda v: v.total_energy_pJ)
        return {
            "variant": best.name,
            "total_energy_pJ": best.total_energy_pJ,
            "savings_vs_conventional_ecc_percent":
                savings_percent(self._energy(self.reference), best.total_energy_pJ),
            "savings_vs_embedded_only_percent":
                savings_percent(self._energy(self.embedded_reference),
                                best.total_energy_pJ),
            "evaluated_candidates": len(cands),
        }

    # ------------------------------------------------------------------ write
    def document(self):
        from ..arch.load import accumulator_bits
        from ..paths import ROOT

        self._validate()
        acc_bits, acc_evidence = accumulator_bits(self.arch, self.cfg)
        finished = _utc_now()
        evaluated = [v for v in self.variants if v.status == "evaluated"]

        return {
            # ---- 1. what this is, and whether it can be trusted -------------
            "schema_version": SCHEMA_VERSION,
            "experiment": {
                "name": self.experiment,
                "phase": self.phase,
                "phase_meaning": (
                    "Pre: the mapping is ECC-unaware and the ECC effect is applied "
                    "during energy evaluation only"
                    if self.phase == "Pre" else
                    "Post: the mapping itself was optimised for the reduced weight "
                    "representation"),
                "mapping_mode": ("fixed-mapping" if self.fixed_mapping
                                 else "reconstruction-aware mapping"),
                "run_id": self.run_id(),
                "status": ("ok" if evaluated and not any(
                    not c["passed"] for c in self.validation) else
                    "check the validation section"),
                "units": {"energy": "pJ unless a field name says otherwise",
                          "savings": "percent",
                          "capacity": "bits or bytes as the field name says"},
            },
            "identity": {
                "architecture": self.arch,
                "architecture_label": self.cfg.arch_label(self.arch),
                "model": self.model,
                "layer_scope": self.cfg.layer_scope,
                "layers": self.cfg.layers or "full model",
                "bch": {"n": self.cfg.code_n, "k": self.cfg.code_k,
                        "t": self.cfg.code_t,
                        "label": f"BCH({self.cfg.code_n},{self.cfg.code_k})"},
                "precisions_bits": {
                    "weight": self.cfg.weight_bits,
                    "activation": self.cfg.activation_bits,
                    "accumulator": acc_bits,
                    "accumulator_policy": (
                        "paper-native (per architecture)"
                        if self.cfg.acc_bits_override is None
                        else "forced common width -- SENSITIVITY STUDY"),
                    "accumulator_evidence": acc_evidence,
                },
            },
            # ---- 2. the answer ----------------------------------------------
            "reference_variants": {
                "conventional_ecc": self.reference,
                "embedded_only": self.embedded_reference,
            },
            "summary": self.summary(),
            "best_reconstruction_placement": self.best_reconstruction(),
            # ---- 3. what to be careful about --------------------------------
            "validation": self.validation,
            "warnings": self.warnings,
            "approximations": self.approximations,
            # ---- 4. the variants in full ------------------------------------
            "variants": [v.to_dict() for v in self.variants],
            # ---- 5. everything needed to reproduce it -----------------------
            "detail": self.detail,
            "provenance": {
                "config": self.cfg.to_dict(),
                "config_hash": self.config_hash(),
                "mapper_settings": self.cfg.mapper_settings(),
                "mapper_fingerprint": self.cfg.fingerprint(),
                "command": " ".join(os.sys.argv),
                "git_commit": _git_commit(ROOT),
                "started_utc": self.started.isoformat(),
                "finished_utc": finished.isoformat(),
                "runtime_seconds": (finished - self.started).total_seconds(),
                "host": socket.gethostname(),
                "user": _safe_user(),
                "platform": platform.platform(),
                "project_root": str(ROOT),
            },
        }

    def write(self, overwrite=None):
        """Write the JSON and return its path. Refuses to clobber by default."""
        overwrite = self.cfg.overwrite if overwrite is None else overwrite
        path = self.results.evaluation_path(self.arch, self.model, self.run_id(),
                                            self.phase)
        if path.exists() and not overwrite:
            raise ResultError(
                f"{path} already exists.\n"
                f"  -> results are never silently overwritten. Either accept the "
                f"existing file, or re-run with ECC_OVERWRITE=1 to replace it.")
        doc = self.document()
        # LF: this file is read inside a Linux container and committed to a
        # repository that forces LF on *.json.
        with open(path, "w", newline="\n", encoding="utf-8") as fh:
            fh.write(json.dumps(doc, indent=1, default=str))
        _write_pointer(path)
        return path


def _safe_user():
    try:
        return getpass.getuser()
    except Exception:
        return "unavailable"


def _write_pointer(path):
    """`latest.json` beside the run files, naming the most recent one.

    A pointer, not a copy: copying would double every result on disk and give
    two files that can drift apart. Prior runs are untouched.
    """
    pointer = path.parent / "latest.json"
    with open(pointer, "w", newline="\n", encoding="utf-8") as fh:
        fh.write(json.dumps({"latest": path.name,
                             "written_utc": _utc_now().isoformat()}, indent=1))
    return pointer


def load(path):
    """Read a result back, checking it is one of ours and readable."""
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    if "schema_version" not in doc:
        raise ResultError(f"{path} has no schema_version; it is not an "
                          f"eccenergy evaluation result")
    return doc


def load_latest(results, arch, model, phase=None):
    """The most recent result for one (phase, arch, model, ...) namespace."""
    d = results.evaluation_dir(arch, model, phase or result_phase(
        results.cfg.experiment, getattr(results.cfg, "recon_optimizer", True)))
    pointer = d / "latest.json"
    if pointer.exists():
        name = json.loads(pointer.read_text(encoding="utf-8"))["latest"]
        return load(d / name)
    runs = sorted(p for p in d.glob("*.json") if p.name != "latest.json")
    if not runs:
        raise ResultError(f"no results under {d}")
    return load(runs[-1])
