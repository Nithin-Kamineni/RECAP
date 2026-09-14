"""Reading the environment -- and the ONLY place in the project that does.

`env.sh` is the one file a user edits and every knob is written `${VAR:=default}`
there, so the ENVIRONMENT WINS over the file and a one-off never needs an edit.
By the time anything else runs, the environment has been read exactly once, into
the six frozen settings objects beside this module.

Each helper turns a string into the type the knob means, and REFUSES rather than
guessing: `ECC_VICTORY=two_thousand` is a `ConfigError` naming the knob, not a
silent default. `_one()` refuses a list where one value is meant -- only the
swept axis takes a list -- `_table()` parses env.sh section 8's `key=value;`
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
    """`key=value;key=value` -> `{key: float(value)}` (env.sh section 8's
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


def _scoped_list(name, default="", models=()):
    """`ECC_LAYERS`' THREE spellings (EnvReorganisation 6.9, extended phase 6).

        conv1 layer3.0.conv1                these layers, whichever model runs
        resnet18=conv1; mobilenet_v2=f.1    per MODEL; a model with no entry
                                            runs whole
        resnet18.conv1                      ONE layer of ONE named model
        resnet18.all                        that model, WHOLE

    Layer names are per network, so one bare list cannot scope a run that
    spans two of them -- a name the other model has not got is refused by
    `layers-not-in-model`, and it must stay refused (a typo that silently
    swept a whole model is the worse failure). The per-model form is how a
    development scope names two layers of each of three networks.

    THE DOTTED FORM IS THE ONE A PERSON TYPES (the user's ask, 2026-09-14):
    `resnet18.conv1` is one layer of one model without the `=`/`;` punctuation,
    and `resnet18.all` is that whole model. It resolves to the SAME per-model
    table as the `=` form, so nothing downstream learns a third shape.

    TELLING IT APART FROM A BARE LAYER NAME IS WHY `models` IS PASSED IN.
    Layer names contain dots (`layer3.0.conv1`, `features.9.conv.2`) and model
    names do not, so a token is the dotted form exactly when the text before
    its FIRST dot is a KNOWN MODEL. Without that list the rule would be a
    guess, and `settings/env.py` may not reach for one -- `settings/run.py`
    hands it the registry it already imports.

    Returns `(bare, by_model)`; at most one of the two is non-empty, because
    the spellings may not be mixed.
    """
    raw = _s(name, default).split("#", 1)[0]
    if "=" not in raw:
        toks = [tok for tok in raw.replace(",", " ").split() if tok]
        dotted = [t for t in toks
                  if "." in t and t.split(".", 1)[0] in tuple(models)]
        plain = [t for t in toks if t not in dotted]
        if dotted and plain:
            raise guards.refusal("not-a-key-value-list",
                f"{name}: {' '.join(plain)} is a bare layer list and "
                f"{' '.join(dotted)} is the `model.layer` form. The spellings "
                f"may not be mixed -- a bare name means 'this layer of whichever "
                f"model runs', which is not a statement about one network")
        if dotted:
            by_model, whole = {}, set()
            for tok in dotted:
                model, _, layer = tok.partition(".")
                if layer == "all":
                    whole.add(model)
                    by_model[model] = []
                    continue
                if model in whole:
                    raise guards.refusal("not-a-key-value-list",
                        f"{name}: {model}.all asks for the whole model and "
                        f"{tok} asks for one layer of it. Name one or the other")
                by_model.setdefault(model, []).append(layer)
            return [], by_model
        return plain, {}
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
