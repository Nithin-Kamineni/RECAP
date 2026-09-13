"""L3 -- Timeloop, its outputs, and everything charged after it.

`noc_post` and `latency_post` are the two terms Timeloop cannot be given and the
evaluator charges on top of a solved mapping; `results_store` is the ONLY writer
of an evaluation JSON; `ert_probe` is the ERT hook's proof.

May import: `arch`, `physics`, `contracts`, `settings`/`config`, `paths`.

`timeloop.py` is still at the package root: PHASE 3 cuts it into
`toolchain/{invoke,ert,stats}.py`.

`latency_post` is here by SYMMETRY WITH `noc_post`, not by instruction --
ProjectRestructure names neither a layer nor a package for it (Appendix C omits
the file). Both are post-mapping charges over a solved plan, so they belong
together; recorded here so the choice is visible rather than inferred.
"""
