"""Reading the environment -- and the ONLY place in the project that does.

`env.sh` is the one file a user edits and every knob is written `${VAR:=default}`
there, so the ENVIRONMENT WINS over the file and a one-off never needs an edit.
By the time anything else runs, the environment has been read exactly once, into
the six frozen settings objects beside this module.

Each helper turns a string into the type the knob means, and REFUSES rather than
guessing: `ECC_VICTORY=two_thousand` is a `ConfigError` naming the knob, not a
silent default. `_one()` refuses a list where one value is meant -- only the
swept axis takes a list -- `_table()` parses env.sh section 10's `key=value;`
flattening of a bash associative array, which cannot be exported any other way,
and `_scoped_list()` tells `ECC_LAYERS`' bare list from its per-model form.

ProjectRestructure phase 4 moved these out of `config.py`. The rule they enforce
is unchanged and is in CLAUDE.md: NEVER read `os.environ` outside this module.
"""
from __future__ import annotations

import os

from ..contracts.errors import ConfigError
from . import guards

#: What counts as true and false in every ECC_* boolean.
_TRUE = {"1", "true", "yes", "on", "y"}
_FALSE = {"0", "false", "no", "off", "n", ""}


def _s(name, default=""):
    return os.environ.get(name, default).strip()


def _b(name, default):
    raw = _s(name)
    if raw == "":
        return default
    low = raw.lower()
    if low in _TRUE:
        return True
    if low in _FALSE:
        return False
    raise guards.refusal("not-a-boolean",
        f"{name}={raw!r} is not a boolean (use 1/0, true/false, on/off)")


def _i(name, default):
    raw = _s(name)
    if raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise guards.refusal("not-an-integer",
            f"{name}={raw!r} is not an integer") from exc


def _f(name, default):
    raw = _s(name)
    if raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise guards.refusal("not-a-number",
            f"{name}={raw!r} is not a number") from exc


def _oi(name):
    raw = _s(name)
    return None if raw == "" else _i(name, 0)


def _of(name):
    raw = _s(name)
    return None if raw == "" else _f(name, 0.0)


def _table(name):
    """`key=value;key=value` -> `{key: float(value)}` (env.sh section 10's
    flattening of a `declare -A` table, which bash cannot export)."""
    out = {}
    for entry in os.environ.get(name, "").split(";"):
        entry = entry.strip()
        if not entry:
            continue
        key, _, val = entry.partition("=")
        try:
            out[key.strip()] = float(val)
        except ValueError:
            raise guards.refusal("not-a-key-value-table",
                f"{name}: entry {entry!r} is not `key=number`") from None
    return out


def _list(name, default="", sep=None):
    """Space- or comma-separated list; anything after '#' is a comment.

    `sep=";"` splits on semicolons and keeps each entry whole, spaces and all --
    for `ECC_RECON_PLACEMENT_LIST`, whose entries are `arch=key key key`.
    """
    raw = _s(name, default).split("#", 1)[0]
    if sep:
        return [e.strip() for e in raw.split(sep) if e.strip()]
    return [tok for tok in raw.replace(",", " ").split() if tok]


def _scoped_list(name, default=""):
    """`ECC_LAYERS`' two spellings, told apart by the `=` (EnvReorganisation 6.9).

        conv1 layer3.0.conv1                these layers, whichever model runs
        resnet18=conv1; mobilenet_v2=f.1    per MODEL; a model with no entry
                                            runs whole

    Layer names are per network, so one bare list cannot scope a run that
    spans two of them -- a name the other model has not got is refused by
    `layers-not-in-model`, and it must stay refused (a typo that silently
    swept a whole model is the worse failure). The per-model form is how a
    development scope names two layers of each of three networks.

    Returns `(bare, by_model)`; at most one of the two is non-empty, because
    the two spellings may not be mixed.
    """
    raw = _s(name, default).split("#", 1)[0]
    if "=" not in raw:
        return [tok for tok in raw.replace(",", " ").split() if tok], {}
    by_model = {}
    for entry in raw.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        key, sep, vals = entry.partition("=")
        if not sep or not key.strip():
            raise guards.refusal("not-a-key-value-list",
                f"{name}: entry {entry!r} is not `model=layer layer`. The two "
                f"spellings may not be mixed: either a bare list of layer "
                f"names, or `;`-separated `model=...` entries")
        by_model[key.strip()] = [t for t in vals.replace(",", " ").split() if t]
    return [], by_model


def _one(name, default=""):
    """A single token. Extra tokens are an error, not a silent truncation."""
    got = _list(name, default)
    if len(got) > 1:
        raise guards.refusal("one-value-not-a-list",
            f"{name} takes ONE value, not {len(got)}: {' '.join(got)}\n"
            f"  -> only the swept axis takes a list")
    return got[0] if got else ""


def _list_or_none(name, sep=None):
    """`None` if the variable is UNSET, a list if it is set -- empty or not.

    The distinction matters for exactly one knob: `ECC_SWEEP_ARCHS` unset means
    "every declared design" and is filled in by `config._resolve()`, while
    `ECC_SWEEP_ARCHS=" "` means the user asked for NOTHING and is refused. A
    plain default cannot say both, and this package may not read `archs/` to
    apply the first one itself.
    """
    if name not in os.environ:
        return None
    return _list(name, sep=sep)
