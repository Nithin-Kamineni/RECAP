# `tests/` — the three buckets

`ProjectRestructure.md` §7.4. **Red means red**: a failure here is a failure, never
a missing cache or a forgotten environment variable.

| bucket | rule |
|---|---|
| `tests/unit/` | pure functions. No env, no cache, no filesystem. Green ALWAYS, in under 5 s. |
| `tests/contract/` | structure, not numbers — the layer rule, YAML schemas, slug distinctness, docs. Green ALWAYS, fast. |
| `tests/data/` | property tests against the real mapper cache. Green **or skipped**; never a false red. |

**The suite still lives in `eccenergy/tests/`** (15 modules, 227 tests) and moves
here bucket by bucket as later phases touch it — §4.2's tree is the destination,
not a phase-2 deliverable. Phase 2 puts the first tenant in: `contract/`, which
had nowhere to live because the thing it checks (the layer rule) did not exist
until phase 2 created the layers.

Run both roots:

    bash hpc/tl.sh python3 -m pytest eccenergy/tests/ tests/ -q
