# Result storage: the namespace, the document, and `Pre` vs `Post`

Implements `04_results_storage_spec.txt`. Everything here is produced by one
module, [`eccenergy/results_store.py`](../eccenergy/results_store.py); no
experiment writes an evaluation JSON by itself.

## The namespace

```
results/evaluation/{Pre|Post}/{ARCH}/{MODEL}/{BCH}/{PREC}/{SCOPE}/{MAPPER}/{RUNID}.json
```

A real path from the Task 1 run:

```
results/evaluation/Pre/eyeriss_v2_like/resnet18/bch63_51/w8a8/
  layers2__layer3_0_downsample_0__layer4_1_conv2/
  map-energy-vic100l-hybrid-seednone/
  20260906T072817Z__303cee78.json
```

The spec's namespace is `Results/evaluation/{Pre|Post}/{ARCH}/{MODEL}/{BCH}/...`;
this is that, lowercased to match the existing `ECC_RESULTS_DIR=results`, plus
four extensions. The spec asks for extensions **only where needed to prevent
collisions**, so each one has to justify itself:

| segment | example | what would collide without it |
|---|---|---|
| `{PREC}` | `w8a8`, `w8a8acc24` | The optional forced-common-accumulator sensitivity study would overwrite the paper-native primary result — the one comparison that must never be silently replaced. A different weight or activation quantization would too. |
| `{SCOPE}` | `layers-full`, `layers2__…` | A two-layer development result and a full-model result are different claims. The plan requires they not be confusable; a shared directory cannot provide that. |
| `{MAPPER}` | `map-energy-vic100l-hybrid-seednone` | Two mappings found under different search settings are different results, and the energy difference between them is not an architecture difference. |
| `{RUNID}` | `20260906T072817Z__303cee78` | The file itself. Timestamp plus a hash of the full configuration, so a re-run never silently overwrites a prior one. |

Everything from `{Pre|Post}` down to `{MAPPER}` is **deterministic**: the same
configuration always resolves to the same directory. Only `{RUNID}` moves.

Layer names are sanitised for the filesystem (`layer4.1.conv2` →
`layer4_1_conv2`) but the **unmodified** names are also in the document, under
`identity.layers`. A selection too long for a path is truncated and
disambiguated with a hash of the full list, which is still deterministic.

`latest.json` sits beside the run files and names the most recent one. It is a
pointer, not a copy — copying would double every result on disk and give two
files that can drift apart.

### Never silently overwritten

Writing to a path that already exists raises. `ECC_OVERWRITE=1` (or
`--overwrite`) is required, and even then only that exact run id is replaced.
Prior runs are never touched.

## `Pre` vs `Post`

| | `Pre` | `Post` |
|---|---|---|
| Who chose the mapping | the mapper, knowing nothing about ECC | the mapper, optimising for the reduced weight width |
| Where the ECC effect enters | energy **evaluation** only | mapping **and** evaluation |
| Which tasks | 1, 2, 3 | 4, 5 |
| Mapping across variants | one shared mapping set, **verified** | each variant may reference its own |

`Post` means the mapper was told about the reduced weight representation.
Task 1 has no reduced representation, so there is nothing it could have been
told — a Task 1 run therefore refuses to write itself as `Post`.

The `Pre` guarantee is checked, not asserted. Every result carries a
`fixed_mapping_shared_across_variants` entry in `validation`, which compares the
`mapping_ids` of every evaluated variant. A fixed-mapping comparison whose
variants were mapped differently is not a fixed-mapping comparison, and the
document says so rather than letting a reader assume otherwise.

## The document

Summary first, machinery after. Opening the file, in order:

1. **`schema_version`, `experiment`** — what this is, `Pre` or `Post`,
   fixed-mapping or reconstruction-aware, the run id, the status, the units.
2. **`identity`** — architecture, model, layer scope and the layer names,
   BCH(n,k,t), and the three precisions with the accumulator's evidence quoted.
3. **`reference_variants`** — which variant is the conventional-ECC reference
   and which is the embedded-only reference.
4. **`summary`** — one row per variant: total energy, savings against **both**
   references, and the reason if it is unavailable.
5. **`best_reconstruction_placement`** — the cheapest *evaluated* placement, or
   an explanation of why there is none.
6. **`validation`, `warnings`, `approximations`** — what was checked, what to
   be careful about, what is approximate and how.
7. **`variants`** — each variant in full, with its per-component energy
   breakdown, mapping ids, and variant-specific detail (the parity account for
   the baseline; placement and reconstruction counts for a Task 3 variant).
8. **`detail`** — architecture source and fingerprint, contract report,
   provenance, workload and layer dimensions, per-layer rows, per-level access
   counts and energies, mapping records, and paths to the caches.
9. **`provenance`** — the full config, mapper settings, command, git commit,
   timestamps, runtime, host, platform.

### Units

Explicit, in the field name: `total_energy_pJ`, `total_energy_uJ`,
`payload_bits`, `parity_bytes`, `savings_vs_conventional_ecc_percent`.
`experiment.units` restates the convention.

### Savings

```
savings_percent = 100 * (reference_energy - variant_energy) / reference_energy
```

Reported against both references. Negative values are allowed and mean the
variant costs more — a result, not an error. `null` means the reference or the
variant is unavailable; writing `0.0` there would read as "no change" rather
than "not known".

## Missing data is never invented

A variant that has not been implemented, is not feasible on this architecture,
or failed carries:

```json
{
  "name": "recon_pe_spad_input",
  "status": "not_implemented",
  "total_energy_pJ": null,
  "unavailable_reason": "Tasks 3-5 have not been implemented. ..."
}
```

The writer **refuses** to construct a variant that is unavailable but carries a
number, or one that is unavailable without a reason, or one marked `evaluated`
with no number. Those are `ResultError`s, not warnings, because a placeholder
that looks like a measurement is worse than a missing one.

`status` is one of `evaluated`, `not_implemented`, `unsupported`, `failed`.

Every Task 1 file therefore already contains all seven variants — the
conventional baseline evaluated, embedded ECC and the five reconstruction
placements as `not_implemented`. The document shape is final from Task 1
onward; later tasks fill entries in rather than changing the schema.

## Reading results back

```python
from eccenergy.paths import Results
from eccenergy.results_store import load, load_latest

doc = load_latest(Results(cfg).prepare(), "eyeriss_v2_like", "resnet18")
doc = load("results/evaluation/Pre/.../20260906T072817Z__303cee78.json")
```

`load` refuses anything without a `schema_version`.

## The test

```
python3 -m eccenergy.tests.test_results_store
```

Dependency-free and offline — no pytest, no container, no mapper. `test_roundtrip`
is the test the spec asks for: it writes one file holding the conventional
baseline, embedded ECC and four reconstruction placements (two evaluated, one
unsupported, one not implemented), asserts the exact namespace, loads it back,
and recomputes the savings against both references. The rest guard the
invariants that make such a file worth trusting — overwrite protection, the
namespace separating development from full-model and forced-precision runs, the
fixed-mapping check actually failing when variants disagree, and the parity
arithmetic including a worked example small enough to verify by hand.

## Example command and the path it produces

```bash
ECC_LAYERS="layer3.0.downsample.0 layer4.1.conv2" \
ECC_VICTORY=100 ECC_MAPPER_THREADS=8 ECC_SWEEP=arch \
  bash run.sh baseline --eval
```

```
results/evaluation/Pre/eyeriss_v2_like/resnet18/bch63_51/w8a8/
  layers2__layer3_0_downsample_0__layer4_1_conv2/
  map-energy-vic100l-hybrid-seednone/
  20260906T072817Z__303cee78.json
```
