"""L2 -- the designs, and the workloads they run.

The only readers of `archs/<name>/*.yaml`. After phase 5 a new architecture is
one directory of DATA here and no code anywhere.

May import: `physics`, `contracts`, `settings`/`config`, `paths`.

`archs.py` (2,762 lines) is still at the package root: PHASE 3 cuts it into
`arch/{load,patch,fingerprint,validate}.py`. Moving it here first would only
have to be moved again.
"""
