"""L3 -- Timeloop, its outputs, and everything charged after it.

    inputs        the cloned repo, the problem YAMLs, `design_inputs()`
    ert           supplying an ERT/ART table and reading back the one used
    cache         `ShapeLock`: one cache entry, one writer
    invoke        `Mapper` -- THE only module that needs the container
    stats         parsing `timeloop-mapper.stats.txt`
    weight_stats  the WEIGHT PATH read back out of that output
    noc_post      the two NoC terms Timeloop cannot be given
    latency_post  the roofline, and the one owner of the clock period
    results_store the ONLY writer of an evaluation JSON
    ert_probe     the ERT hook's proof (FINDINGS 3.5)

May import: `arch`, `physics`, `contracts`, `settings`/`config`, `paths`.

PHASE 3 cut `timeloop.py` (1,529 lines) into the first five. It is five files
and not Appendix C's three because `ErtTables` and `Mapper` BOTH stand on the
inputs half and BOTH take the cache lock: two modules that import each other are
the one file the split was supposed to end, so the shared halves became
`inputs.py` and `cache.py` -- the latter is section 4.2's own `cache.py`.

`latency_post` is here by SYMMETRY WITH `noc_post`, not by instruction --
ProjectRestructure names neither a layer nor a package for it (Appendix C omits
the file). Both are post-mapping charges over a solved plan, so they belong
together; recorded here so the choice is visible rather than inferred.
"""
