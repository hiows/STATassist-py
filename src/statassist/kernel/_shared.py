"""What every kernel needs before it can start: samples, and the three tails.

The R kernels take "a numeric vector without missing values" or "a list of them,
one per group level, named by and ordered as ``group_lv``". Those two sentences
are the whole of the input contract, and this module is where they are read and
enforced, so that no kernel has to repeat the check and none of them can disagree
about it.

Who drops the missing values is deliberately left to the caller. Independent
samples drop per group and a paired design drops whole pairs, so the same column
yields different samples depending on the design; a kernel that dropped them
itself would quietly answer a different question than the one asked.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..core.errors import SaValueError

__all__ = [
    "ALTERNATIVES",
    "as_matrix",
    "as_sample",
    "as_samples",
    "check_alternative",
    "condition_names",
]

#: The three alternatives, spelled as R spells them.
#:
#: The dot in ``"two.sided"`` is not a Python convention, and it is kept anyway:
#: it is the value a user of the R package already writes, it is what the public
#: ``compare_*`` functions of a later phase will take, and it is what the golden
#: fixtures were generated with. Renaming it here would leave the layers of the
#: port disagreeing about the spelling.
ALTERNATIVES: tuple[str, ...] = ("two.sided", "less", "greater")


def check_alternative(alternative: str) -> str:
    """Refuse an alternative the kernels do not know.

    R's ``switch()`` returns ``NULL`` for an unmatched value, which then vanishes
    from the named vector being assembled and leaves a column silently absent.
    Refusing by name is the one place this port does not reproduce R's behaviour
    on purpose.
    """
    if alternative not in ALTERNATIVES:
        raise SaValueError("`alternative` must be one of: " + ", ".join(ALTERNATIVES) + ".")
    return alternative


def as_sample(v: Any, arg: str = "v") -> np.ndarray:
    """Read one sample as a finite float array."""
    array = np.asarray(v, dtype=float).reshape(-1)
    if array.size == 0:
        raise SaValueError(f"`{arg}` holds no observation.")
    if not np.isfinite(array).all():
        raise SaValueError(
            f"`{arg}` holds a missing or infinite value; the caller is expected to "
            "have dropped those according to its own design."
        )
    return array


def as_samples(samples: Any) -> tuple[list[str], list[np.ndarray]]:
    """Read a per-level set of samples as names and arrays.

    A mapping is the usual form, since the R kernels index their error messages
    by level name. A plain sequence is accepted as well and labelled by position,
    which is what a caller with no level names would otherwise have to invent.

    Returns:
        The level names in order, and the samples in the same order.
    """
    if isinstance(samples, Mapping):
        items = list(samples.items())
    elif isinstance(samples, Sequence) and not isinstance(samples, str | bytes):
        items = [(str(index + 1), values) for index, values in enumerate(samples)]
    else:
        raise SaValueError("`samples` must be a mapping of level name to sample, or a sequence.")

    if len(items) < 2:
        raise SaValueError(f"needs at least 2 groups, got {len(items)}.")

    names = [str(name) for name, _ in items]
    arrays = [as_sample(values, f"samples[{name!r}]") for name, values in items]
    return names, arrays


def as_matrix(mat: Any, arg: str = "mat") -> np.ndarray:
    """Read a subjects-by-conditions matrix as a complete float array.

    A within-subject design keeps complete subjects only, and which rows to drop
    is decided where the pairing is resolved, so an incomplete matrix reaching a
    kernel is a caller error rather than something to clean up here.
    """
    if isinstance(mat, pd.DataFrame):
        array = mat.to_numpy(dtype=float)
    else:
        array = np.asarray(mat, dtype=float)
    if array.ndim != 2:
        raise SaValueError(f"`{arg}` must be a subjects-by-conditions matrix.")
    if not np.isfinite(array).all():
        raise SaValueError(
            f"`{arg}` holds a missing or infinite value; a within-subject design "
            "keeps complete subjects only, and dropping them is the caller's job."
        )
    return array


def condition_names(mat: Any) -> list[str]:
    """The condition labels of a matrix, by column name or by position.

    A :class:`pandas.DataFrame` carries the level names R would have read from
    ``colnames()``; a bare array has none, so its conditions are numbered.
    """
    if isinstance(mat, pd.DataFrame):
        return [str(name) for name in mat.columns]
    return [str(index + 1) for index in range(np.asarray(mat).shape[1])]
