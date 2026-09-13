"""L0 -- the shared types, and NOTHING that computes with them.

`ProjectRestructure.md` section 4.2. STILL EMPTY AFTER PHASE 3, on purpose.

Section 4.2 lists `Placement`, `Stage`, `LevelStats`, `Bar`, `Arm` and `Verdict`
here. Phase 3 moved code and changed no body, and each of those is defined in
the same file as the code that BUILDS it -- `Placement` and `Stage` with the
tables they index (`arch/placements.py`, `arch/weight_path.py`), `StageStats`
with the parser that fills it. Splitting a dataclass away from its only
constructor would have been a rewrite, not a move, and nothing needed it: the
types two layers share now live in L2, which every layer above may import.

It earns its keep in PHASE 4, where `Config` becomes six frozen settings objects
and the types they carry have to sit below both the settings and the physics. The
package exists so the layer rule already has somewhere to point.

A module here may import nothing of ours at all.
"""
