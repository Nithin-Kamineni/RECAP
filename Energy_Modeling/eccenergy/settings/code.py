"""The code and the precision: BCH(N, K) over `weight_bits`-bit weights.

One codeword definition serves all three ECC arms -- they differ only in where
the parity lives -- so this group is what a bar's arithmetic is computed from,
and `physics/` is handed these numbers rather than reaching for them.

`code_t` is derived: the BCH(63, K) design distance table gives it exactly, and
anything else falls back to the (n - k) / 6 estimate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..contracts.errors import ConfigError
from .env import _b, _f, _i, _list, _of, _oi, _one, _s, _table

#: BCH(63, K) minimum distance, from the code tables behind the DC synthesis runs.
BCH63_KTOD = {57: 3, 51: 5, 45: 7, 39: 9, 36: 11, 30: 13}

#: Where codeword boundaries fall when parity is counted. See parity.py.
PARITY_GROUPINGS = ("layer", "model")

#: The fields of this group the MAPPER sees, and therefore the ones
#: `Config.fingerprint()` hashes.
#:
#: The mapper sees the PRECISION, not the code: `weight_bits` is the word the
#: arch YAML declares and `acc_bits_override` the psum width it is patched to.
#: N and K reach it only through `weight_datawidth`, which the arm resolves.
IN_FINGERPRINT = frozenset({
    "weight_bits",
    "activation_bits",
    "acc_bits_override",
})


@dataclass(frozen=True)
class CodeSettings:
    weight_bits: int
    activation_bits: int
    acc_bits_override: Optional[int]
    code_n: int
    emb_weights_per_cw_override: Optional[float]
    parity_grouping: str
    parity_charge_padding: bool
    decode_enabled: bool
    decode_pj_base: float
    decode_pj_emb: float
    recon_charges_decode: bool
    weak_enabled: bool
    weak_n: int
    weak_k: int

    # ---- derived --------------------------------------------------------
    #: `ECC_CONST_K`, carried here because K is a property of the CODE and
    #: `const_k` is the axis knob that sets it. `config._resolve()` copies it.
    code_k: int = 0
    #: Correctable symbols, from BCH63_KTOD where the code is tabulated and the
    #: (n - k) / 6 estimate where it is not.
    code_t: int = 0

    @staticmethod
    def from_env():
        """Every `ECC_*` knob of this group, read once.

        `code_k` is `ECC_CONST_K` -- a RUN axis this group is the consumer of --
        and `code_t` comes from the BCH design-distance table; `config._resolve()`
        sets both.
        """
        return CodeSettings(
            weight_bits=_i("ECC_WEIGHT_BITS", 8),
            activation_bits=_i("ECC_ACTIVATION_BITS", 8),
            acc_bits_override=_oi("ECC_ACC_BITS"),
            code_n=_i("ECC_CODE_N", 63),
            emb_weights_per_cw_override=_of("ECC_EMB_WEIGHTS_PER_CW"),
            parity_grouping=_s("ECC_PARITY_GROUPING", "layer").lower(),
            parity_charge_padding=_b("ECC_PARITY_CHARGE_PADDING", True),
            decode_enabled=_b("ECC_DECODE", False),
            decode_pj_base=_f("ECC_DECODE_PJ_BASE", 40.0),
            decode_pj_emb=_f("ECC_DECODE_PJ_EMB", 40.0),
            recon_charges_decode=_b("ECC_RECON_CHARGES_DECODE", True),
            weak_enabled=_b("ECC_WEAK", False),
            weak_n=_i("ECC_WEAK_N", 63),
            weak_k=_i("ECC_WEAK_K", 57),
        )
