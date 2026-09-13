# ProjectRestructure — making the code hold still while the study grows

**The study is about to gain many more architectures, each with its own weight path and its
own reconstruction boundaries. The code cannot take that today: adding ONE architecture
means editing FOURTEEN places across SIX files in THREE languages, and only two of those
places are inside that architecture's own directory. This plan moves the per-design
knowledge into `archs/<name>/`, breaks the one import cycle that makes every change
ripple, splits the six files that carry half the codebase, and sorts 106 guards and 187
tests into categories you can actually see.**

| | |
|---|---|
| **Scope** | `Energy_Modeling/` only. No modelling change. **Every step must reproduce today's numbers to the pJ.** |
| **Prerequisite** | **None.** `prompt_7.md` is fully implemented, so there is no cold pass to schedule around and no sequencing constraint — this plan can start immediately. |
| **Measured** | Every number in §2 and Appendix A was measured on this repository on 2026-09-12, not estimated. §5.1.1 was measured 2026-09-13. |
| **Written** | 2026-09-12 |
| **Status** | Phases 0-5 are DONE (2026-09-13). Phases 6 (guards) and 7 (backfill tests) remain. §9.2 carries the marks. |
| **Revised** | 2026-09-13 — §4.2 now carries the COMPLETE tree including every per-architecture directory; §5.1.1 adds the measured case against per-architecture CODE; the `prompt_7` sequencing section is gone because prompt_7 is implemented. |

---

## §0 — How to use this document

| § | what is in it | read it when |
|---|---|---|
| 1 | **Terminology** — every technical word used below, in plain English | first, always |
| 2 | **What was measured** — the evidence, with numbers | to understand *why*, or to argue with the plan |
| 3 | **Why one change breaks everything** — four root causes | same |
| 4 | **The target structure** — the layer rule, and the COMPLETE proposed tree | before any step |
| 5 | **Architectures as data** — `archs/<name>/`, your main ask, and why the CODE stays shared | before adding an architecture |
| 6 | **The guards** — audit, taxonomy, and the ablation escape hatch | before an ablation study |
| 7 | **The tests** — audit, what is redundant, the three buckets | before deleting a test |
| 8 | **Reading the code yourself** — a guided tour | when you want to look, not change |
| 9 | **The phase plan** — one phase per session, each with a gate | to do the work |
| 10 | **What not to do** | before improvising |

**One rule governs every phase: `arch_fingerprint()` must not move, and the evaluated
totals must not move.** The mapper cache is hours of compute. Moving Python does not cold
it *provided* the patched YAML text and the hashed settings stay byte-identical — and
every phase gate below checks exactly that.

---

## §1 — Terminology

Words used in this document, defined before use. (Study terms — arm, bar, boundary, ERT,
`q` — are in `prompt_6.md` §1 and `prompt_7.md` §1.)

| term | plain English |
|---|---|
| **module** | one `.py` file. |
| **package** | a directory of modules with an `__init__.py`. |
| **import graph** | who reads whom. If `a.py` says `from . import b`, there is an arrow `a → b`. |
| **fan-in** | how many modules import *this* one. High fan-in = many things break when it changes. |
| **fan-out** | how many modules *this* one imports. High fan-out = it breaks when many things change. |
| **import cycle** | `a → b → a`. Python cannot load either first, so one import gets hidden inside a function to break the deadlock. **A cycle means the two files are really one file wearing two names.** |
| **lazy import** | `import` written *inside* a function instead of at the top. Almost always a cycle being hidden. It is the fingerprint of the problem, visible from the outside. |
| **layer** | a group of modules at one level of abstraction. **The layer rule: a module may import only from strictly lower layers.** Enforcing it makes cycles impossible. |
| **god object** | one class that holds everything, so every function takes it and no signature tells you what it actually uses. `Config` (104 fields) is one. |
| **coupling** | how much one part must know about another. Low coupling is the whole goal. |
| **guard** | a runtime refusal in production code: `raise ConfigError(...)`, `assert ...`. It stops a run. |
| **test** | a check that runs only when you invoke `pytest`. It stops nothing. |
| **invariant** | a fact that must be true of *any* correct run, on *any* architecture. "Both arms declare the same width" is an invariant. "K=30" is not. |
| **property test** | a test that asserts an invariant over real data, rather than checking one hand-written example. |
| **mutation test** | deliberately breaking something to prove the test notices. A test with no mutation proves nothing. |
| **golden output** | a saved copy of today's numbers, used to prove a refactor changed nothing. |
| **ablation** | deliberately removing or zeroing one term to see how much it mattered ("what if DRAM were free?"). A legitimate experiment that the current guards block — §6.4. |
| **schema** | a machine-checkable description of what a data file must contain. Lets a YAML file be validated the way code is type-checked. |

---

## §2 — What was measured

Everything here was computed from the repository on 2026-09-12. Reproduction commands are
in Appendix A.

### 2.1 Half the code is in six files

```
   25,623 lines of Python in eccenergy/

   recon.py                2,790  ####################
   tests/test_recon.py     2,660  ###################
   experiments/recon.py    2,341  #################
   archs.py                2,232  ################
   config.py               1,945  ##############
   experiments/dilation.py 1,851  #############
   ----------------------------------------------- 13,819 = 54% in SIX files
   timeloop.py             1,202  ########
   everything else (30 modules)   ~10,600
```

And the biggest functions are longer than most files should be:

| function | lines | file |
|---|---:|---|
| `evaluate` | **381** | `experiments/recon.py` |
| `ert_aware_view` | **304** | `experiments/recon.py` |
| `evaluate_placement` | **254** | `recon.py` |
| `validate_arch` | **250** | `archs.py` |
| `dilated_view` | **251** | `experiments/recon.py` |
| `level_table` | **238** | `experiments/dilation.py` |

> A 381-line function cannot be held in a human's head, cannot be unit-tested in pieces,
> and cannot be changed without re-reading all 381 lines. This is the readability
> complaint, made concrete.

### 2.2 `Config` is a god object

```
   class Config:
       104 fields          <- any function taking `cfg` may read any of them
        42 methods
        99 ECC_* env knobs read (env.sh declares 130)
```

```
   def evaluate_placement(p, cfg):        # what does this depend on?
       ...                                 # 104 possible answers. Unknowable
                                           # without reading all 254 lines.
```

### 2.3 There is an import cycle, and it is being hidden

**Ten imports are written inside functions instead of at the top of their file:**

| module | lazy imports | what it is hiding |
|---|---:|---|
| `config.py` | **6** | `config → recon → config` |
| `archs.py` | 2 | |
| `recon.py` | 2 | |

```
             +----------------------------------------+
             |                                        |
             v                                        |
         config.py  ---- (lazy, inside __post_init__) -+---> recon.py
             ^                                              |
             |                                              |
             +----------------------------------------------+

   Python cannot import either of these first. The lazy import is the workaround.
   The consequence: config.py and recon.py are ONE unit that cannot be reasoned
   about, tested, or changed separately.
```

### 2.4 Everything funnels through three modules

| module | imported by | meaning |
|---|---:|---|
| `archs` | **15** | change it and 15 modules are in the blast radius |
| `paths` | **13** | (fine — it has fan-out 0, a proper leaf) |
| `config` | **9** | and it is in a cycle, so the blast radius is not bounded |
| `energy` | 9 | |
| `ecc`, `timeloop` | 7 each | |

`paths` is the model to copy: **13 modules depend on it, it depends on nothing, and it has
never been the cause of a break.** That is what a well-placed module looks like.

### 2.5 Adding ONE architecture takes 14 edits across 6 files in 3 languages

This is the finding that matters most for where the study is going.

| # | file | what you add | right place? |
|---|---|---|---|
| 1 | `archs/<name>/arch_paper.yaml` | the design, every number cited | ✅ |
| 2 | `archs/<name>/README.md` | what it is | ✅ |
| 3 | `archs/_shared/standard.yaml` | its entry | ⚠️ shared file |
| 4 | `archs/_shared/provenance.yaml` | its citations | ⚠️ shared file |
| 5 | `archs/_shared/noc.yaml` | its NoC containers | ⚠️ shared file |
| 6 | `eccenergy/config.py` | `KNOWN_ARCHS` | ❌ **code** |
| 7 | `eccenergy/config.py` | `ARCH_LABELS` | ❌ **code** |
| 8 | `eccenergy/config.py` | `BRACKET_PAIRS` (v2 only) | ❌ **code** |
| 9 | `eccenergy/archs.py` | `MAPSPACE_FREE_LEVELS` | ❌ **code** |
| 10 | `eccenergy/recon.py` | `WEIGHT_PATHS[<name>]` | ❌ **code, 2,790-line file** |
| 11 | `eccenergy/recon.py` | `PLACEMENTS[<name>]` | ❌ **code, same file** |
| 12 | `env.sh` | `ECC_ARCHS` | ❌ **shell** |
| 13 | `env.sh` | `ECC_RECON_PLACEMENTS[<name>]` | ❌ **shell** |
| 14 | `env.sh` | `ECC_ARCH_CLOCK_MHZ[<name>]` | ❌ **shell** |

**Two of fourteen are in the architecture's own directory.** `CLAUDE.md` already has to warn
you that #10 and #11 must be *"edited together"* — that warning exists because the structure
makes forgetting easy, and the consequence is silent: every boundary below the missing stage
keeps reporting its own saving while the new stage stays at full width.

### 2.6 The guards: 106 sites, 54% in one file

| file | guard sites |
|---|---:|
| `config.py` | **57** |
| `recon.py` | 14 |
| `archs.py` | 6 |
| `code_widths.py`, `embedded.py`, `energy.py`, `parity.py` | 5 each |
| `experiments/dilation.py` | 3 |
| `ecc.py`, `timeloop.py` | 2 each |
| `baseline_dram.py`, `generate.py` | 1 each |
| **total** | **106** |

Kinds: `ConfigError` 57, `ValueError` 45, `KeyError` 3, bare `assert` 1.

**One genuinely good result, worth saying plainly: ZERO guards hardcode an architecture,
level, model or placement name in their own source.** I checked all 106 against every known
design, level, model and placement key. `CLAUDE.md`'s rule — *"never `if key == "recon2"`"* —
is being followed. **The guards are already design-agnostic. The problem is not that they
are arch-specific; it is that they are un-categorised, un-listed, and un-overridable.**

### 2.7 The tests: 187 functions, clustered on 31% of the code

| measurement | value | what it says |
|---|---:|---|
| test functions | **187** | (177 collected — `test_latency.py` does not import cleanly) |
| public production symbols | **238** | |
| symbols with ≥ 1 test | **74 (31%)** | **69% of the code has no test at all** |
| tests targeting no production symbol | **27** | they test documentation, or nothing |
| tests with no docstring | **55** | you cannot tell what they protect |
| tests with zero assertions | **3** | they cannot fail |
| tests pinning a past state | **11** | e.g. `..._reproduces_the_pre_2026_09_09_numbers` |
| median test length | 20 lines | (but the longest is 90) |

Clustering — tests are piled on a few symbols while most of the code is bare:

```
   components                  ##############  14 tests
   patched_weight_geometry     ######           6
   assert_pair_geometry        #####            5
   account                     #####            5
   ...
   164 public symbols          (nothing)        0 tests
```

> **This is the signature of tests written reactively, one per bug, rather than designed.**
> Each is individually justified; together they over-cover the few things that broke and
> leave everything else open.

---

## §3 — Why changing one thing breaks everything

Four causes, in order of damage.

### 3.1 `cfg` is passed everywhere, so nothing declares its dependencies

```
   TODAY                                    AFTER
   -----                                    -----
   def evaluate_placement(p, cfg):          def evaluate_placement(
       ...                                      p: Placement,
                                                code: CodeSettings,
   cfg has 104 fields.                          recon: ReconSettings):
   Which does this touch?                   
   Unknowable without reading 254 lines.    The signature IS the answer.
```

Every knob is therefore *potentially* coupled to every computation. That is the spaghetti
feeling, and it is not caused by file length.

### 3.2 The cycle makes the blast radius unbounded

With `config → recon → config`, a change in either can surface in the other, and neither
can be loaded, tested or reasoned about alone. The 6 lazy imports in `config.py` are the
visible symptom.

### 3.3 Per-design knowledge lives in shared code

`WEIGHT_PATHS` and `PLACEMENTS` are Python tuples inside a 2,790-line module shared by every
architecture. Adding a design means editing a file that every other design depends on.

```
   TODAY                                    TARGET
   -----                                    ------
   recon.py (2,790 lines)                   archs/eyeriss_like_wglb/
     WEIGHT_PATHS = {                           arch_paper.yaml
       "eyeriss_like":        (...),            weight_path.yaml     <- ITS path
       "eyeriss_like_wglb":   (...),            placements.yaml      <- ITS boundaries
       "eyeriss_v2_like":     (...),            design.yaml          <- label, clock, free levels
       "simple_weight_stationary": (...),       README.md
     }                                      archs/<next design>/
     PLACEMENTS = { ...the same 4 keys... }     ...same four files, nothing else touched

   every design in one file                 one directory per design
   adding one = editing shared code         adding one = adding a directory
```

### 3.4 The tests reproduce the same coupling

`test_recon.py` is 2,660 lines — as large as the module it tests. Tests build `Config`
objects by hand, so they inherit every default, including wrong ones. That is exactly why
all 14 test failures diagnosed on 2026-09-12 were harness problems rather than model
problems.

---

## §4 — The target structure

### 4.1 The layer rule

**A module may import only from strictly lower layers.** One sentence, and it makes cycles
structurally impossible.

```
   +-------------------------------------------------------------------------+
   | L5  report/     figures, tables, manifests                              |
   |                 may import L0 L1 L2 L3 L4                               |
   +-------------------------------------------------------------------------+
   | L4  study/      the arms, placement evaluation, sweeps, dilation        |
   |                 may import L0 L1 L2 L3                                  |
   +-------------------------------------------------------------------------+
   | L3  toolchain/  Timeloop invocation, ERT/ART, stats parsing, caches     |
   |                 may import L0 L1 L2                                     |
   +-------------------------------------------------------------------------+
   | L2  arch/       load a design, patch it, fingerprint it, its weight     |
   |                 path and placements.  may import L0 L1                  |
   +-------------------------------------------------------------------------+
   | L1  physics/    parity, embedded layout, packing, granularity, widths   |
   |                 PURE FUNCTIONS. No file I/O. may import L0 only         |
   +-------------------------------------------------------------------------+
   | L0  contracts/  dataclasses and enums. NO logic. IMPORTS NOTHING.       |
   +-------------------------------------------------------------------------+

   settings/  sits beside L0: frozen dataclasses read once from the environment.
   paths/     stays exactly as it is -- it is already a perfect L0 leaf.
```

**Why this specific split:** each layer has a different reason to change. `physics` changes
when the maths changes. `arch` changes when a design is added. `toolchain` changes when
Timeloop changes. Today all three change together because they share `config.py` and
`recon.py`.

### 4.2 The directory layout

**This is the complete proposed tree.** Everything marked `NEW` does not exist today;
everything else exists and either stays where it is or moves as §4.4 says. **Nothing is
deleted.** The per-architecture directories are spelled out in full, because *"where does a
new design go"* is the question this whole plan exists to answer — the answer is: one
directory, and nowhere else.

```
Energy_Modeling/
  env.sh                          the one config file, one knob per line     (unchanged)
  run.sh                          the one command                            (unchanged)
  CLAUDE.md   FINDINGS.md   README.md   progress.txt                         (unchanged)
  GUARDS.md                  NEW  GENERATED. every guard and every test, one table
  Makefile                   NEW  make guards | make layers | make arch NEW=<name>

  archs/                          ONE DIRECTORY PER ARCHITECTURE -- see §5
      _shared/                    CROSS-DESIGN CONTRACTS ONLY, never per-design entries
          standard.yaml           what must be identical across every design
          provenance.yaml         where every shared number came from
          noc.yaml                the shared NoC statements
          components/             regfile_decoded.yaml   register_rw.yaml
                                  smartbuffer_SRAM_banked.yaml

      eyeriss_like/           --- the five files a reconstruction-capable design has ---
          arch_paper.yaml         the design, every number cited             (as today)
          design.yaml        NEW  label, source, clock MHz, mapspace free levels,
                                  leakage class, NoC containers, per-design flags
          weight_path.yaml   NEW  the stages, outer to inner; kind: dram|storage|network,
                                  Timeloop level prefixes, reducible: true|false, evidence
          placements.yaml    NEW  the boundaries: key, label, reduced[], site_stage,
                                  site_counter, description
          README.md               what it is and what it is not              (as today)

      eyeriss_like_wglb/          the same five
      eyeriss_v2_like/            the same five, plus arch.yaml and
                                  arch_physical_8x2_3x4.yaml                 (as today)
      eyeriss_v2_like_wglb/       the same five, plus arch.yaml              (as today)
      simple_weight_stationary/   the same five

      simba_like/                 arch_paper.yaml   design.yaml   README.md
      simple_input_stationary/    -- NO weight_path.yaml and NO placements.yaml: these
      simple_output_stationary/      three declare no boundaries today, and the schema
                                     must accept that rather than invent one (§5.2)

      tpu_like/                  NEW   a new design IS this directory, and nothing else

  eccenergy/
      contracts/         L0   Placement  Stage  LevelStats  Bar  Arm  Verdict (~50 ln ea)
      settings/          L0   code.py  arch.py  mapper.py  recon.py  energy.py
                              run.py  banner.py
      physics/           L1   parity.py  embedded.py  packing.py  granularity.py
                              widths.py            PURE FUNCTIONS. No file I/O.
      arch/              L2   load.py  patch.py  fingerprint.py  validate.py
                              weight_path.py  placements.py
                                                   the ONLY readers of archs/<name>/*.yaml
      toolchain/         L3   invoke.py  ert.py  stats.py  cache.py
      study/             L4   arms.py  placement_eval.py  narrowing.py  dilation.py
                              sweep.py  baseline.py  embedded.py  validate.py
                              audit.py  diagnose.py
      report/            L5   stacked.py  panels.py  style.py  tables.py  manifest.py
                              recon_view.py  results_store.py
      paths.py           L0   unchanged -- it is already a perfect L0 leaf
      __main__.py             the CLI entry point                            (unchanged)

  tests/                        MOVES OUT of eccenergy/tests/
      unit/                     pure. no env, no cache. MUST be green, always.
      contract/                 structural: the layer rule, YAML schemas, slug distinctness
      data/                     property tests against the real mapper cache. SKIP cleanly.
      conftest.py  fixtures/                                                 (as today)

  hpc/                          map.sbatch and the sweep drivers             (unchanged)
  ecc_energy_study/             THE MAPPER CACHE -- hours of compute. Never moved, never
                                renamed: its paths are keyed by arch_fingerprint().
  results/                      evaluation/  figures/  manifests/  tables/  _raw/
  data/   docs/   legacy/                                                    (unchanged)
```

**Two notes on the tree.** First, §4.4 lists only the four files that genuinely **split**;
the remaining `experiments/` drivers (`baseline`, `embedded`, `validate`, `audit`,
`diagnose`) are L4 and simply move into `study/`, and `plots/` (`stacked`, `panels`,
`style`) plus `results_store.py` are L5 and move into `report/`. A move is not a split and
needs no seam. Second, `ecc_energy_study/` is in the tree only to say that it does **not**
move: it is addressed by `arch_fingerprint()`, and a renamed path is a cold cache.

**Size targets: no file over ~400 lines, no function over ~60.**

### 4.3 Splitting `Config` into six settings objects

```
   TODAY                        TARGET
   -----                        ------
                                CodeSettings     N, K, weight bits, q, packing
   class Config:                ArchSettings     archs, fidelity, clock, bandwidth, noc
     104 fields   ------->      MapperSettings   algorithm, victory, timeout, threads
      42 methods                ReconSettings    placements, arm, gating, encoder site
      99 env knobs              EnergySettings   dram pJ/bit, mac override, leakage
                                RunSettings      phase, results dir, formats
```

Two immediate wins beyond readability:

1. **Signatures become dependency lists.** You can see what a function uses.
2. **The fingerprint becomes honest.** It hashes `ArchSettings + MapperSettings` — a stated
   set — instead of "whichever of 104 fields happens to be in the hash list". Today that
   list is maintained by hand and `CLAUDE.md` has to warn you about it.

### 4.4 The split, file by file

`recon.py` already carries banner comments at exactly the right boundaries. **The seams
exist as comments; they simply are not files yet.** This makes the split mechanical.

| today | lines | becomes | layer |
|---|---|---|---|
| `recon.py` 168–417 | 249 | `arch/weight_path.py` + data in `archs/<n>/weight_path.yaml` | L2 |
| `recon.py` 417–869 | 452 | `arch/placements.py` + data in `archs/<n>/placements.yaml` | L2 |
| `recon.py` 869–1121 | 252 | `study/arms.py` | L4 |
| `recon.py` 1121–1685 | 564 | `toolchain/stats.py` | L3 |
| `recon.py` 1685–1855 | 170 | `physics/packing.py` | L1 |
| `recon.py` 1855–1964 | 109 | `study/narrowing.py` | L4 |
| `recon.py` 1964–2190 | 226 | `study/dilation.py` | L4 |
| `recon.py` 2190–2437 | 247 | `physics/granularity.py` | L1 |
| `recon.py` 2437–end | 353 | `study/placement_eval.py` | L4 |
| `archs.py` | 2,232 | `arch/load.py` + `patch.py` + `fingerprint.py` + `arch/validate.py` | L2 |
| `config.py` | 1,945 | `settings/*.py` (six) + `settings/banner.py` | L0-adjacent |
| `experiments/recon.py` | 2,341 | `study/placement_eval.py` + `report/recon_view.py` | L4/L5 |

> **Notice that `recon.py` alone splits across four different layers.** That is the
> measurement that proves it is not one module — it is four modules in a trench coat, and
> that is why touching any part of it moves the others.

**AS BUILT, 2026-09-13 (phase 3 landed).** The table above and Appendix C each assign the
same target name to two different sources three times, so three names were settled at
implementation and are recorded in the modules' own docstrings:

| both wanted | what took the name | what the other became |
|---|---|---|
| `toolchain/stats.py` | `timeloop.py`'s parser | `toolchain/weight_stats.py` (the weight-path view) |
| `study/dilation.py` | `experiments/dilation.py`'s driver | `study/capacity.py` (the dilation correction) |
| `study/placement_eval.py` | `recon.py`'s tail — ONE boundary | `study/placement_study.py` (the driver) |

Three further deviations, each because the alternative was a cycle or a rewrite:

* **`timeloop.py` became FIVE modules, not three.** `ErtTables` and `Mapper` both stand on
  the inputs half and both take the cache lock, so `toolchain/inputs.py` and
  `toolchain/cache.py` (§4.2's own `cache.py`) hold the shared halves.
* **The mapper arms are `arch/arms.py`, not `study/arms.py`** — `config.py` and
  `toolchain/ert_probe.py` both read the arm list and neither may import a driver. That is
  what removed the `ert_probe` edge rather than re-spelling it. `study/arms.py` stays free
  for Appendix C's other claimant, `experiments/{baseline,embedded}.py`.
* **`contracts/` is still empty.** Every dataclass §4.2 lists there is defined in the same
  file as the code that builds it; splitting one away would have been a rewrite, not a
  move, and the types two layers share now sit at L2, which every layer above may import.
  Phase 4 put the first tenant in it: `errors.py`, holding `ConfigError`, which both
  `settings/` and `config.py` raise and neither may own.

**PHASE 4, AS BUILT.** `config.py` did not become `settings/` — it became the layer ABOVE
`arch/`. The knobs are L0 and frozen; RESOLVING them against the designs is a different
job, and that is where the cycle actually was. Two consequences worth knowing before
phase 5: `dataclasses.replace(cfg, ...)` is now `cfg.with_(...)`, and "update every
signature to take what it needs" was deliberately NOT done — `cfg.code` exists, every call
site still says `cfg.weight_bits`, and converting them is incremental work for phase 7
rather than one diff no gate can attribute.

---

## §5 — Architectures as data — your main ask

### 5.1 The rule

> **One code path, many data files.** `eyeriss_like_wglb` and a future `tpu_like` must differ
> only in what they *declare*, never in which function runs.

**Where your instinct needs one adjustment:** a directory per architecture for **data** is
exactly right. A directory per architecture for **code** would be wrong — you would get one
copy of the logic per design, they would drift, and a bug fixed in one would not reach the
others. Likewise **do not** make a directory per reconstruction placement: a placement is
five fields of data (`reduced`, `site_stage`, `site_counter`, `label`, `description`), not a
program.

### 5.1.1 Measured: what a per-architecture CODE directory would duplicate

The instinct behind *"give each architecture its own reconstruction code"* is right about
**where the knowledge lives** and needs one adjustment about **what form it takes**. Three
measurements, taken 2026-09-13, say why data wins. Commands in Appendix A.

| # | question | measured answer | what it means |
|---|---|---|---|
| 1 | Is a placement a program? | every entry of `_EYERISS_V1_WGLB_PLACEMENTS` (`recon.py` 707+) is `key, variant, label, short, rating, reduced[], site_stage, site_counter, description` | **No.** Fields and cited prose. Nothing in it executes. |
| 2 | How many DISTINCT placement tables do the five reconstruction-capable designs have? | **three** — `PLACEMENTS` (`recon.py` 792) maps `eyeriss_v2_like` **and** `eyeriss_v2_like_wglb` to the *same* `_V2_PLACEMENTS` | a code directory per design would **fork one shared table into two copies that drift** |
| 3 | How much production code branches on an architecture name? | **three sites**, all `startswith("eyeriss_v2")`, all NoC-related: `experiments/baseline.py:207`, `experiments/audit.py:87`, `experiments/recon.py:1923` | the real per-design specialisation is **one declared flag**, not one directory |

**Finding 3 is the answer to "and the other things specialised for that architecture".**
Those three branches *are* the specialisation, and they become a field in `design.yaml` —
the design's NoC containers, present or absent — read by the same code path for every
design. After phase 5 that count is **zero**, and a `tests/contract/` test asserts it stays
zero. This is the same result §2.6 already found for the guards: **zero of the 106 hardcode
a design name.** The code is already design-agnostic; only the two tables are not.

So the split is: **data per design, code shared.** `validate_placement_space()` enforces the
prefix rule and the reachability rule for every design, and
`test_every_placement_space_is_valid_for_every_supported_design` then covers each new design
automatically the moment that design is data.

### 5.2 What a new architecture looks like, after

```
   archs/tpu_like/            (the same shape every design has -- §4.2 shows all of them)
   +-- arch_paper.yaml     the design, every number cited        (as today)
   +-- design.yaml         NEW   label, source:, clock MHz, mapspace free levels,
   |                             leakage class, per-design NoC containers
   +-- weight_path.yaml    NEW   the stages, outer to inner, each with
   |                             kind: dram|storage|network, its Timeloop level
   |                             prefixes, reducible: true|false, and its evidence
   +-- placements.yaml     NEW   the boundaries: key, label, reduced[], site_stage,
   |                             site_counter, description
   +-- README.md           what it is and what it is not          (as today)

   EDITS ELSEWHERE:  none.
```

A design that declares **no** reconstruction boundaries carries `arch_paper.yaml`,
`design.yaml` and `README.md` only — `simba_like`, `simple_input_stationary` and
`simple_output_stationary` are in that state today. The schema must accept it and say so,
rather than invent a weight path that was never cited.

```
             BEFORE                                    AFTER
   14 edits / 6 files / 3 languages          1 directory / 5 files / 1 language
   +---------------------------+             +---------------------------+
   | archs/<name>/  .......  2 |             | archs/<name>/  ........ 5 |
   | archs/_shared/ .......  3 |             | everything else ....... 0 |
   | config.py .............  3 |             +---------------------------+
   | archs.py ..............  1 |
   | recon.py ..............  2 |             validated by a SCHEMA, so a
   | env.sh ................  3 |             half-finished design fails
   +---------------------------+             LOUDLY at load, not silently
```

### 5.3 Why this is safe, and what replaces the hand-written guards

Moving the tables into YAML does not weaken the checks — it strengthens them, because a
schema can check things a Python tuple cannot:

| what is checked today | how | after |
|---|---|---|
| `reduced` is a prefix of the reducible stages | `validate_placement_space()`, hand-written | schema + the same function, now reading data |
| every reducible stage is reached by some boundary | same | same |
| every Weights-carrying level is claimed by exactly one stage | `weight_path()` | same |
| **the two tables were edited together** | **a warning in `CLAUDE.md`** | **impossible to violate — they are two files in one directory, loaded together or not at all** |
| every declared level exists in the arch YAML | *not checked* | **schema, at load** |
| every placement key appears in `ECC_RECON_PLACEMENTS` | *not checked* | **schema, at load** |

The existing test `test_every_placement_space_is_valid_for_every_supported_design` keeps
working unchanged and now covers new designs automatically.

### 5.4 The shared files stay shared — correctly

`standard.yaml`, `provenance.yaml` and `noc.yaml` are **cross-design contracts**: they state
what must be identical across designs, and where every number came from. Splitting them per
design would defeat their purpose. They stay, and the per-design *entries* inside them move
into `archs/<name>/design.yaml`, leaving the shared files holding only genuinely shared
statements.

---

## §6 — The guards

### 6.1 Guard vs test — the distinction that is currently missing

| | **guard** | **test** |
|---|---|---|
| lives in | production code | `tests/` |
| runs | on **every** run, including your SLURM jobs | only when you type `pytest` |
| on failure | **stops the run** | prints red |
| protects against | a wrong number reaching a figure | a regression reaching `main` |
| today | **106 sites** | **187 functions** |

**Both are invisible.** There is no list of either. That is your "impossible to know what is
there", and §6.5 fixes it.

### 6.2 The audit: what the 106 guards actually are

Good news first: **none of them is design-specific.** Checked against every architecture,
level, model and placement key — zero hardcoded names. The guards already generalise.

The real problem is that four very different things are spelled the same way:

```
  +----------------------------------------------------------------------------+
  | TIER 1  PARSE       "ECC_VICTORY=abc is not an integer"                     |
  |                     ~9 sites.  ALWAYS fatal. Never a nuisance. Keep as-is.  |
  +----------------------------------------------------------------------------+
  | TIER 2  IMPOSSIBLE  "need K < N";  "width must be positive";                |
  |                     "width % datawidth != 0 -- timeloop-mapper aborts"      |
  |                     ~25 sites. Physically or arithmetically impossible.     |
  |                     ALWAYS fatal. Keep as-is.                               |
  +----------------------------------------------------------------------------+
  | TIER 3  COUPLING    "ECC_RECON_ERT_AWARE=1 needs RECON_OPTIMIZER=True       |
  |                      and ECC_PHASE=Post"                                    |
  |                     "ECC_RERUN_OPTIMISER=1 but ECC_FROM_CACHE=1 never       |
  |                      invokes Timeloop"                                      |
  |                     ~15 sites. Encodes "the combinations we thought of".    |
  |                     >>> THIS IS THE ABLATION BLOCKER <<<                    |
  +----------------------------------------------------------------------------+
  | TIER 4  DERIVED     "ECC_RECON_ERT_ARM=recon2 declares q=4, but you set     |
  |                      ECC_WEIGHT_DATAWIDTH=5. Leave it EMPTY."               |
  |                     ~4 sites. Refuses you for setting a knob the system     |
  |                     also derives.  >>> ALSO AN ABLATION BLOCKER <<<         |
  +----------------------------------------------------------------------------+
```

### 6.3 Measured: 6 of 15 plausible ablations are refused today

I tried fifteen experiments a person would reasonably want to run:

| ablation you might want | result | tier |
|---|---|---|
| gating 0% / 100% (the two bounds) | ✅ OK | |
| capacity scale 0.5 (shrink the buffer) | ✅ OK | |
| K=62 (near-zero parity) | ✅ OK | |
| `ECC_SPLIT_READ_WRITE=1` | ✅ OK | |
| sweep arch while holding arch | ✅ OK | |
| **`ECC_MAC_PJ_OVERRIDE=0`** — *what if compute were free?* | ❌ **REFUSED** | 2→**wrong** |
| **`ECC_DRAM_PJ_PER_BIT=0`** — *what if DRAM were free?* | ❌ **REFUSED** | 2→**wrong** |
| **`ECC_BASELINE_DRAM_PJ_PER_BIT=0`** — *ablate the price model* | ❌ **REFUSED** | 2→**wrong** |
| **ERT arm + explicit `ECC_WEIGHT_DATAWIDTH=5`** — *test a hypothetical q* | ❌ **REFUSED** | 4 |
| **ERT arm + my own narrow levels** — *what if only the spad narrowed?* | ❌ **REFUSED** | 4 |
| `ECC_RECON_ERT_AWARE=1` alone | ❌ REFUSED | 3 |
| `ECC_RERUN_OPTIMISER=1` + `ECC_FROM_CACHE=1` | ❌ REFUSED | 3 |

**Three of these are guards that are simply wrong.** "Must be > 0" is the correct rule for a
*width*; it is the wrong rule for a *price*. Zero is a perfectly meaningful price — it is
the ablation that measures how much that term was worth, and it is the natural upper bound
on the term's importance. The guard confuses "physically impossible" with "not the value we
used".

### 6.4 The fix: severity tiers plus a recorded override

```
   ECC_ALLOW="zero-price,derived-datawidth"     <- names guards, never a blanket "off"
```

| tier | behaviour | overridable? |
|---|---|---|
| **1 PARSE** | refuse | **never** |
| **2 IMPOSSIBLE** | refuse | **never** |
| **3 COUPLING** | refuse **unless named in `ECC_ALLOW`** | yes → becomes a loud WARNING |
| **4 DERIVED** | refuse **unless named in `ECC_ALLOW`** | yes → becomes a loud WARNING |

Plus three rules that keep it honest, and which match how this project already works:

1. **Every guard gets a stable id** (`zero-price`, `ert-arm-needs-optimiser`,
   `derived-datawidth`) printed in its own message, so the message tells you how to override
   it.
2. **An override is RECORDED**: it lands in the run manifest as `overrides: [...]` and on the
   figure's caveat list — the same mechanism `Config.recon_caveats()` already uses. You
   cannot ablate by accident and you cannot publish an ablation without it saying so.
3. **Three guards get re-tiered because they are misclassified**: the three "price must be
   > 0" guards become "price must be ≥ 0, and a zero price is recorded as an ablation".

This preserves everything the guards were protecting — nothing becomes silent — while making
the ablation studies you are about to run actually possible.

### 6.5 `GUARDS.md` — generated, never hand-written

A ~40-line script walks the code and emits one table. Regenerate it with `make guards`; a
contract test fails if it is stale.

| id | tier | kind | where | refuses |
|---|---|---|---|---|
| `pair-geometry` | 2 | guard | `arch/patch.py` | the two arms declaring different width or depth |
| `narrow-once` | 2 | guard | `study/narrowing.py` | mapper **and** evaluator both narrowing (squares the saving) |
| `placement-prefix` | 2 | guard | `arch/placements.py` | `reduced` not a prefix of the reducible stages |
| `zero-price` | 4 | guard | `settings/energy.py` | a zero per-bit price *(overridable — ablation)* |
| `ert-arm-needs-optimiser` | 3 | guard | `settings/recon.py` | `ERT_AWARE=1` without the optimiser *(overridable)* |
| `arm-slugs-distinct` | — | test | `tests/contract/` | two arms sharing one cache directory |

**This single file answers "what is protected, and by what" in one screen.** It is the direct
fix to your complaint.

---

## §7 — The tests

### 7.1 Is there redundancy? Yes — but not where you would expect

You asked whether tests are redundant. The audit says: **the problem is not duplicate tests,
it is lopsided coverage and unclear purpose.**

| finding | count | verdict |
|---|---:|---|
| **symbols with no test at all** | **164 of 238 (69%)** | ⚠️ **under-tested, not over-tested** |
| tests piled on `components` alone | 14 | over-covered |
| tests with **no docstring** | **55** | ⚠️ purpose unknowable — the real "can't tell if it's useful" |
| tests with **zero assertions** | **3** | ❌ **delete or fix — they cannot fail** |
| tests **pinning a past state** | **11** | ⚠️ review individually (§7.2) |
| tests targeting **no production symbol** | 27 | mostly documentation checks (§7.3) |

**So: do not do a mass deletion.** Three tests are provably worthless (zero assertions).
Everything else needs sorting, not culling — and 69% of the code needs tests it does not
have.

### 7.2 The 11 "historical" tests — keep, retire, or convert

These pin what the code *used to do*, e.g.
`test_controller_site_reproduces_the_pre_2026_09_09_numbers`. They are the most brittle
category, and each needs an individual decision:

| pattern | keep if | retire if |
|---|---|---|
| `..._reproduces_the_pre_<date>_numbers` | the old mode is still a supported option (e.g. `ECC_RECON_ENCODER_SITE=source`) | the old mode is gone |
| `..._must_not_come_back` | it encodes a decision recorded in `FINDINGS.md` | it does not |
| `..._reproduces_prompt_N_table_X` | that table is still the contract | the prompt was superseded |

**Convert rather than delete where you can:** `..._reproduces_prompt_6_table_5_1` is more
valuable rewritten as *"the ERT deltas equal `incremental/weights_per_cw × block_size`"* — a
property that stays true when the numbers change — than as a pinned number that fails
whenever `K` moves.

### 7.3 The documentation tests

Four test files assert on `CLAUDE.md` / `prompt_*.md` content (e.g.
`test_claude_md_still_carries_the_protected_width_section`). **Keep them, but move them to
`tests/contract/test_docs.py`** so nobody mistakes a documentation failure for a modelling
failure. They exist because the docs went stale once and then contradicted the code — a real
failure mode in this project.

### 7.4 The three buckets

```
   tests/unit/       pure functions only. No env, no cache, no filesystem.
                     MUST be green at all times. If this is red, something is broken.
                     -> physics/, contracts/, settings/ parsing

   tests/contract/   structure, not numbers: the layer rule, YAML schemas,
                     cache-slug distinctness, GUARDS.md freshness, docs.
                     MUST be green. Fast.
                     -> catches "you added an arch and forgot placements.yaml"

   tests/data/       property tests against the real mapper cache.
                     SKIPS cleanly when the cache is absent -- and skip means SKIP.
                     -> the expensive, high-value guards
```

> **"Red means red."** Today a red suite might mean a real bug, a missing cache, or a
> forgotten environment variable, and you cannot tell which without an investigation. That
> is what made you distrust the tests, and it is a plumbing problem, not a philosophy
> problem.

### 7.5 Three fixes that remove most of the noise

| problem | fix | size |
|---|---|---|
| one test leaves a poisoned env var → **22 cascading failures** | `conftest.py` autouse fixture that snapshots and restores `os.environ` | **~8 lines** |
| custom `_Skip` prints as **FAILED** | use `pytest.skip()` | 1 line × 4 |
| tests hand-build configs and inherit the wrong default design | a `cfg()` fixture pinned to one design | ~15 lines |

**Measured effect: the suite goes from 36 failures to roughly green, with no production code
touched and no test deleted.**

### 7.6 What the test suite should look like when done

| bucket | rough count | rule |
|---|---:|---|
| `tests/unit/` | ~90 | green always; runs in < 5 s; no I/O |
| `tests/contract/` | ~25 | green always; runs in < 5 s |
| `tests/data/` | ~70 | green **or skipped**; never a false red |
| deleted | 3 | zero-assertion tests |
| rewritten as properties | ~11 | the historical ones worth keeping |
| **new**, for the 69% | grows over time | one per invariant, each with a mutation |

---

## §8 — Reading the code yourself

You said you want to look at the code and understand it. Here is the shortest path, today
and after.

### 8.1 Today: follow one number end to end

To understand how **one bar on the placement figure** gets its value:

| step | file | what to read | lines |
|---|---|---|---|
| 1 | `env.sh` §4 | the knobs that define the study point | ~100 |
| 2 | `eccenergy/recon.py` 417–869 | `PLACEMENTS` — what R1…R5a mean | 452 |
| 3 | `eccenergy/recon.py` 168–417 | `WEIGHT_PATHS` — the stages they sit between | 249 |
| 4 | `eccenergy/recon.py` 2437–end | `evaluate_placement()` — the arithmetic | 353 |
| 5 | `eccenergy/experiments/recon.py` | `evaluate()` — how bars are assembled | 381 |
| 6 | `eccenergy/plots/stacked.py` | `draw_panel()` — the only place a bar is drawn | 294 |

**~1,800 lines to follow one number.** That is the readability problem in one row.

### 8.2 After: the same journey

| step | file | lines |
|---|---|---|
| 1 | `archs/eyeriss_like_wglb/placements.yaml` | ~80 (data, no code) |
| 2 | `archs/eyeriss_like_wglb/weight_path.yaml` | ~60 (data, no code) |
| 3 | `study/placement_eval.py` | ~350 |
| 4 | `report/stacked.py` | ~290 |

**~780 lines, and the first 140 are data you can read like a spec sheet.**

### 8.3 Three orientation commands worth having

```bash
  make guards     # regenerate GUARDS.md -- every guard and test in one table
  make layers     # print the import graph and flag any layer-rule violation
  make arch NEW=tpu_like    # scaffold archs/tpu_like/ with schema-valid stubs
```

---

## §9 — The phase plan

**One phase per session, each with a gate you can run.**

### 9.1 The gate, applied to every phase

```
   BEFORE the phase:  save the fingerprints and the evaluated totals
   AFTER  the phase:  they must be IDENTICAL

     1. arch_fingerprint() for EVERY (arch, model) pair -- byte-identical
     2. bash run.sh baseline --eval   -> Task 1 and 2 totals unchanged, to the pJ
     3. bash run.sh embedded --eval   -> unchanged
     4. bash run.sh recon --eval      -> every bar unchanged
     5. bash run.sh validate && bash run.sh diagnose -> clean, and diff vs before
     6. the full test suite is no worse than it was
```

If gate 1 fails, **stop** — you are about to cold hours of compute.

### 9.2 The phases

| # | phase | what | logic? | session |
|---|---|---|---|---|
| **0** ✅ | **Golden snapshot** | Save fingerprints for every (arch, model), and the evaluated JSONs for all three arms. This is the safety net for everything that follows. Commit it. | no | ½ |
| **1** ✅ | **Test plumbing** | `conftest.py` env restore; `pytest.skip` for the 4 `_Skip` sites; a `cfg()` fixture; delete the 3 zero-assertion tests. **Suite goes green.** No production code touched. | no | 1 |
| **2** ✅ | **Layers** | Create `contracts/ physics/ arch/ toolchain/ study/ report/`. `git mv` modules in. Fix imports. Add `tests/contract/test_layer_rule.py`. | no | 1 |
| **3** ✅ | **Split the big six** | Cut `recon.py`, `archs.py`, `experiments/recon.py`, `experiments/dilation.py` along their existing banners (§4.4). No function bodies change — only which file they live in. | no | 2 |
| **4** ✅ | **Settings** | `Config` → six frozen dataclasses. **The import cycle dies here.** Update every signature to take what it needs. | **yes** | 2 |
| **5** ✅ | **Architectures as data** | `WEIGHT_PATHS`/`PLACEMENTS` → `archs/<name>/weight_path.yaml` + `placements.yaml`; `KNOWN_ARCHS`/`ARCH_LABELS`/`MAPSPACE_FREE_LEVELS`/`BRACKET_PAIRS` → `design.yaml`; the three `startswith("eyeriss_v2")` branches (§5.1.1) → a declared field. Add the schema and `make arch`. | **yes** | 2 |
| **6** ✅ | **Guards + GUARDS.md** | Assign tier and id to all 106; add `ECC_ALLOW`; re-tier the three price guards; generate `GUARDS.md`; record overrides in the manifest. **Done 2026-09-13: 136 invariants over 141 sites, not 106 — the audit predates phase 5, which added `arch/design.py`'s 23-check schema.** | **yes** | 1 |
| **7** 🔄 | **Backfill tests** | Write `tests/unit/` for the 69% that has none — ongoing, not a blocking phase. **First tranche 2026-09-13: `tests/unit/` exists, +183 tests in under a second; `physics/` 70→76 of 81 and `settings/` closed. `study/` (61) and `toolchain/` (31) are next and mostly need `tests/data/`, which does not exist yet.** | no | ongoing |

**Total: roughly 10 working sessions**, of which **phases 0–3 (4½ sessions) are pure motion
with a byte-identical gate** and buy most of the readability. Phases 4–5 are what buy "change
one thing without breaking everything". Phase 6 is what unblocks your ablations.

### 9.3 If you only do three phases

**0, 1, 5.** Phase 1 stops the false alarms (under an hour). Phase 5 is the one you actually
need before adding architectures. Phase 0 makes both safe.

---

## §10 — What not to do

| ✗ | why |
|---|---|
| **Rewrite and refactor at the same time** | if a number moves you cannot attribute it. Every phase reproduces the old numbers exactly, or it is not done. |
| **Delete guards or tests in bulk** | they encode findings that cost real compute (`FINDINGS` §2.4, §2.7, §2.8, the `BCH(63,39)` width bug that killed 24 jobs). Port them; tier them; list them. Only the 3 zero-assertion tests are free to delete. |
| **A directory of CODE per architecture** | eight copies of the same logic that drift. Measured (§5.1.1): two designs already SHARE one placement table, and only three sites in all of production code branch on a design name — there is no per-design logic to put there. Data per design, code shared. |
| **A directory per reconstruction placement** | a placement is fields of data plus cited prose, not a program (§5.1.1). |
| **A blanket `ECC_GUARDS=off`** | it would be used once and then forgotten, and a wrong number would reach a figure with nothing saying so. Name the guard, record the override. |
| **Change `env.sh`'s knob names** | they are in the mapper fingerprint. Renaming a knob colds the cache. |
| **Touch `parity.py`, `baseline.py`, `external_parity()`** | still frozen (`CLAUDE.md`). They may MOVE in phase 2; their contents may not change. |

---

## Appendix A — How every number here was measured

Run from `Energy_Modeling/` with a Python that has `pandas`, `yaml`, `matplotlib`
(`module load python/3.10` on HiPerGator).

| § | claim | command |
|---|---|---|
| 2.1 | file sizes | `find eccenergy -name '*.py' -not -path '*__pycache__*' \| xargs wc -l \| sort -rn` |
| 2.1 | biggest functions | AST walk over module-level `FunctionDef`, `end_lineno - lineno` |
| 2.2 | 104 fields / 42 methods | AST: `AnnAssign` and `FunctionDef` counts in `class Config` |
| 2.2 | 99 knobs | `grep -oE '"ECC_[A-Z0-9_]+"' eccenergy/config.py \| sort -u \| wc -l` |
| 2.3 | lazy imports | AST: `ImportFrom` with `level > 0` nested inside a `FunctionDef` |
| 2.4 | fan-in / fan-out | AST import graph over `eccenergy/**/*.py` |
| 2.5 | 14 edit sites | `grep -rln` for each of `KNOWN_ARCHS`, `ARCH_LABELS`, `MAPSPACE_FREE_LEVELS`, `WEIGHT_PATHS`, `PLACEMENTS`, `BRACKET_PAIRS`, plus the `env.sh` arrays |
| 2.6 | 106 guards | AST: every `Raise` of `ConfigError/ValueError/AssertionError/RuntimeError/KeyError`, plus every `Assert` |
| 2.6 | no hardcoded design names | regex each guard's source text against every known arch, level, model, placement key → **0 hits** |
| 2.7 | 187 tests, coverage | AST over `tests/test_*.py`; symbol targeting by matching `Name`/`Attribute` nodes against public production symbols |
| 6.3 | the ablation table | build a `Config` under each setting, catch `ConfigError` |
| 5.1.1 | 3 distinct placement tables for 5 designs | read the `PLACEMENTS` dict at `eccenergy/recon.py` 792 |
| 5.1.1 | 3 sites branch on an architecture name | `grep -rnE '(==\|!=\|startswith\|in \()\s*\(?"(eyeriss\|simba\|simple)[a-z_0-9]*"' --include=*.py eccenergy/ \| grep -v '/tests/'` |

### Snapshot of the measured values, 2026-09-12

```
   lines of Python in eccenergy/ ................. 25,623
   share in the six largest files ................ 54%
   Config fields / methods / knobs ............... 104 / 42 / 99
   lazy imports (cycle workarounds) .............. 10   (config 6, archs 2, recon 2)
   fan-in: archs / paths / config / energy ....... 15 / 13 / 9 / 9
   edits to add ONE architecture ................. 14 sites, 6 files, 3 languages
   guard sites ................................... 106  (config.py 57 = 54%)
   guards naming a specific design/level/model ... 0    <- already generalised
   plausible ablations REFUSED ................... 6 of 15
   test functions ................................ 187 (177 collected)
   public symbols with >= 1 test ................. 74 of 238  (31%)
   tests with no docstring ....................... 55
   tests with zero assertions .................... 3
   tests pinning a past state .................... 11
```

---

## Appendix B — The three misclassified guards, in full

These are the clearest single example of why tiering matters. All three are in
`config.py`'s `__post_init__`:

| guard | message today | why it is wrong | after |
|---|---|---|---|
| `ECC_MAC_PJ_OVERRIDE` | *"the per-MAC energy must be positive"* | zero is the ablation "what if compute were free" — the upper bound on how much the MAC term matters | **≥ 0**, tier 4, id `zero-price`, override recorded |
| `ECC_DRAM_PJ_PER_BIT` | *"must be > 0"* | same, for the DRAM term — the single largest category in most bars | same |
| `ECC_BASELINE_DRAM_PJ_PER_BIT` | *"must be > 0"* | same, and this one directly ablates the baseline's price model, which is a live modelling choice (`FINDINGS`, DRAM cost model) | same |

A negative price stays refused — that **is** impossible. The bug is that the guard used
`> 0` where the invariant is `>= 0`.

---

## Appendix C — Where each of today's modules goes

| today | lines | → | layer |
|---|---:|---|---|
| `paths.py` | 250 | `paths.py` *(unchanged — already a perfect leaf)* | L0 |
| `config.py` | 1,945 | `settings/{code,arch,mapper,recon,energy,run}.py` + `settings/banner.py` | L0+ |
| `parity.py` | 356 | `physics/parity.py` *(frozen — moves, contents unchanged)* | L1 |
| `embedded.py` | 534 | `physics/embedded.py` | L1 |
| `code_widths.py` | 249 | `physics/widths.py` | L1 |
| `baseline_dram.py` | 246 | `physics/baseline_dram.py` | L1 |
| `archs.py` | 2,232 | `arch/{load,patch,fingerprint,validate}.py` | L2 |
| `recon.py` | 2,790 | splits across **four** layers — see §4.4 | L1–L4 |
| `timeloop.py` | 1,202 | `toolchain/{invoke,ert,stats}.py` | L3 |
| `noc_post.py` | — | `toolchain/noc_post.py` | L3 |
| `ecc.py` | 465 | `study/stacks.py` | L4 |
| `energy.py` | 612 | `study/energy.py` | L4 |
| `results_store.py` | 451 | `toolchain/results_store.py` *(still the ONLY writer)* | L3 |
| `workloads.py` | — | `arch/workloads.py` | L2 |
| `experiments/recon.py` | 2,341 | `study/placement_eval.py` + `report/recon_view.py` | L4/L5 |
| `experiments/dilation.py` | 1,851 | `study/dilation.py` + `report/dilation_view.py` | L4/L5 |
| `experiments/{baseline,embedded}.py` | 659 | `study/arms.py` | L4 |
| `experiments/{sweep,panels}.py` | — | `report/sweep.py`, `report/panels.py` | L5 |
| `experiments/{validate,diagnose,audit}.py` | 636 | `arch/validate.py`, `report/diagnose.py` | L2/L5 |
| `experiments/ert_probe.py` | 628 | `toolchain/ert_probe.py` | L3 |
| `plots/*` | 294+ | `report/*` | L5 |
| `generate.py` | — | `arch/generate.py` | L2 |
