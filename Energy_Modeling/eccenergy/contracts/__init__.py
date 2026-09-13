"""L0 -- the shared types, and NOTHING that computes with them.

`ProjectRestructure.md` section 4.2. Empty on purpose after phase 2: the types
that belong here (`Placement`, `Stage`, `LevelStats`, `Bar`, `Arm`, `Verdict`)
are today defined inside `recon.py` and `timeloop.py`, and they arrive when
PHASE 3 cuts those files. Creating the package now means the layer rule has
somewhere to point at, and a phase-3 split has somewhere to put a dataclass that
two layers both need.

A module here may import nothing of ours at all.
"""
