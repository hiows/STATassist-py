"""Reading the fixtures ``tools/export_golden.R`` froze, and grading against them.

The kernels of this phase are closed-form statistics, so unlike the simulators
they have to produce R's number rather than a number with the same distribution.
``testdata/golden/<case>/`` holds the input R was given and the result R
produced; this module reads the pair back and compares.

Three conventions come from the R side and are undone here:

* A length-one vector was unboxed to a bare JSON scalar, so :func:`as_list` puts
  it back into a list.
* JSON has no infinity, so ``NA``, ``NaN``, ``Inf`` and ``-Inf`` were written as
  the strings ``"NA"``, ``"NaN"``, ``"Inf"`` and ``"-Inf"``. They are decoded
  here, which is what keeps the untested end of a one-sided interval - genuinely
  infinite - distinct from a quantity that could not be computed.
* Row indices in a table are R's one-based ones. :func:`zero_based` converts a
  column of them, and the port's own row numbers are compared against that.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "GOLDEN_ROOT",
    "RTOL",
    "as_list",
    "assert_close",
    "assert_frame_close",
    "case_names",
    "load_case",
    "load_expected",
    "samples_from_long",
    "zero_based",
]

#: Where the frozen cases live, next to the package rather than inside it: they
#: grade the port and are not shipped with it.
GOLDEN_ROOT = Path(__file__).resolve().parent.parent / "testdata" / "golden"

#: The tolerance the phase is graded at. Every closed-form quantity should reach
#: it; a case that cannot says so at the call site, with the reason.
RTOL = 1e-8


def case_names() -> list[str]:
    """Every frozen case, sorted, for a test that walks all of them."""
    return sorted(path.name for path in GOLDEN_ROOT.iterdir() if path.is_dir())


def load_case(name: str) -> tuple[pd.DataFrame, Any]:
    """The input R was given and the result R produced.

    Args:
        name: Directory name under ``testdata/golden``.

    Returns:
        The input as a frame, and the expected value as nested plain Python.
    """
    return load_input(name), load_expected(name)


def load_input(name: str) -> pd.DataFrame:
    """The input frame of one case.

    ``keep_default_na`` is left on, so R's empty CSV field becomes ``NaN``. The
    R side wrote missing values as empty rather than as ``NA`` for exactly that
    reason: a feature legitimately holding the string ``"NA"`` would otherwise be
    indistinguishable from a hole.
    """
    path = GOLDEN_ROOT / name / "input.csv"
    if not path.exists():
        raise FileNotFoundError(f"no golden input for case `{name}`: {path} does not exist.")
    return pd.read_csv(path)


#: How the R side spelled the four values JSON cannot hold. ``None`` stands for
#: "must be missing on the Python side", which is what both of R's two kinds of
#: missing value mean here.
_SPECIAL = {"NA": None, "NaN": None, "Inf": math.inf, "-Inf": -math.inf}


def _decode(value: Any) -> Any:
    """Turn R's stringified non-finite values back into numbers.

    Only exact matches are touched, so a genuine label is left alone unless it
    happens to be spelled ``"NA"`` - which no fixture in this phase uses.
    """
    if isinstance(value, str):
        return _SPECIAL[value] if value in _SPECIAL else value
    if isinstance(value, list):
        return [_decode(entry) for entry in value]
    if isinstance(value, dict):
        return {key: _decode(entry) for key, entry in value.items()}
    return value


def load_expected(name: str) -> Any:
    """The expected value of one case, as nested dicts, lists and scalars."""
    path = GOLDEN_ROOT / name / "expected.json"
    if not path.exists():
        raise FileNotFoundError(f"no golden result for case `{name}`: {path} does not exist.")
    return _decode(json.loads(path.read_text(encoding="utf-8")))


def as_list(value: Any) -> list[Any]:
    """Undo ``jsonlite``'s unboxing of a length-one vector.

    >>> as_list(3)
    [3]
    >>> as_list([3, 4])
    [3, 4]
    """
    if isinstance(value, list):
        return value
    return [value]


def zero_based(values: Any) -> list[int]:
    """R's one-based row numbers as the zero-based ones the port returns.

    >>> zero_based([2, 19])
    [1, 18]
    """
    return [int(value) - 1 for value in as_list(values)]


def samples_from_long(
    frame: pd.DataFrame,
    levels: Sequence[str],
    *,
    group_col: str = "group",
    value_col: str = "value",
    block: str | None = None,
    block_col: str = "block",
) -> dict[str, np.ndarray]:
    """Rebuild the per-level samples a kernel takes from a long input frame.

    The long shape is what the CSV holds; the kernels take one array per level,
    in the order ``levels`` gives, which is the order the golden result is in.
    """
    rows = frame if block is None else frame[frame[block_col] == block]
    out: dict[str, np.ndarray] = {}
    for level in levels:
        values = rows.loc[rows[group_col] == level, value_col].to_numpy(dtype=float)
        out[str(level)] = values[np.isfinite(values)]
    return out


def _is_missing(value: Any) -> bool:
    """Whether a Python value stands for R's ``NA`` or ``NULL``."""
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    if isinstance(value, np.floating):
        return bool(np.isnan(value))
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _describe(value: Any) -> str:
    """A short rendering for the failure message."""
    if isinstance(value, float | np.floating):
        return repr(float(value))
    return repr(value)


def assert_close(
    actual: Any,
    expected: Any,
    *,
    rtol: float = RTOL,
    atol: float = 0.0,
    path: str = "value",
) -> None:
    """Compare a result against a frozen one, structure and all.

    Walks the two together, so a missing key or an extra element is a failure in
    its own right rather than something a numeric comparison happens to notice.

    Args:
        actual: What the port produced. A mapping, a sequence, a
            :class:`pandas.Series` or a scalar.
        expected: What R produced, as :func:`load_expected` read it.
        rtol: Relative tolerance for a finite number.
        atol: Absolute tolerance, ``0`` by default. Raise it only for a quantity
            whose scale makes the relative comparison meaningless - a p-value
            that R itself returned as a small negative, say - and say why.
        path: Where in the structure the comparison is, used in the message.
    """
    if isinstance(expected, Mapping):
        _assert_mapping_close(actual, expected, rtol=rtol, atol=atol, path=path)
        return

    if isinstance(expected, list):
        _assert_sequence_close(actual, expected, rtol=rtol, atol=atol, path=path)
        return

    if expected is None:
        assert _is_missing(actual), f"{path}: expected a missing value, got {_describe(actual)}."
        return

    if isinstance(expected, bool):
        assert isinstance(actual, bool | np.bool_), (
            f"{path}: expected a boolean, got {type(actual).__name__}."
        )
        assert bool(actual) is expected, f"{path}: expected {expected}, got {bool(actual)}."
        return

    if isinstance(expected, str):
        assert str(actual) == expected, f"{path}: expected {expected!r}, got {_describe(actual)}."
        return

    _assert_number_close(actual, float(expected), rtol=rtol, atol=atol, path=path)


def _assert_mapping_close(
    actual: Any, expected: Mapping[str, Any], *, rtol: float, atol: float, path: str
) -> None:
    if isinstance(actual, pd.Series):
        actual = actual.to_dict()
    assert isinstance(actual, Mapping), (
        f"{path}: expected a mapping with keys {list(expected)}, got {type(actual).__name__}."
    )
    # The key order is part of the contract: an R kernel returns a named vector
    # whose order is the column order of the table it feeds.
    assert list(actual) == list(expected), (
        f"{path}: key order differs.\n  expected: {list(expected)}\n  got:      {list(actual)}"
    )
    for key, value in expected.items():
        assert_close(actual[key], value, rtol=rtol, atol=atol, path=f"{path}[{key!r}]")


def _assert_sequence_close(
    actual: Any, expected: list[Any], *, rtol: float, atol: float, path: str
) -> None:
    if isinstance(actual, pd.Series | np.ndarray):
        actual = list(actual)
    assert isinstance(actual, Sequence) and not isinstance(actual, str | bytes), (
        f"{path}: expected a sequence of {len(expected)}, got {type(actual).__name__}."
    )
    assert len(actual) == len(expected), (
        f"{path}: expected {len(expected)} element(s), got {len(actual)}."
    )
    for index, value in enumerate(expected):
        assert_close(actual[index], value, rtol=rtol, atol=atol, path=f"{path}[{index}]")


def _assert_number_close(
    actual: Any, expected: float, *, rtol: float, atol: float, path: str
) -> None:
    assert not _is_missing(actual), f"{path}: expected {expected!r}, got a missing value."
    got = float(actual)

    if math.isinf(expected):
        assert got == expected, f"{path}: expected {expected!r}, got {got!r}."
        return

    tolerance = atol + rtol * abs(expected)
    assert abs(got - expected) <= tolerance, (
        f"{path}: expected {expected!r}, got {got!r} "
        f"(off by {abs(got - expected):.3e}, allowed {tolerance:.3e})."
    )


def assert_frame_close(
    actual: pd.DataFrame,
    expected: Mapping[str, Any],
    *,
    rtol: float = RTOL,
    atol: float = 0.0,
    path: str = "table",
) -> None:
    """Compare a result table against a frozen one, column by column.

    The expected side is the column-oriented JSON ``jsonlite`` writes for a
    data.frame, so the comparison covers the column names, their order, the row
    count and every cell.
    """
    assert isinstance(actual, pd.DataFrame), (
        f"{path}: expected a DataFrame, got {type(actual).__name__}."
    )
    assert list(actual.columns) == list(expected), (
        f"{path}: column order differs.\n"
        f"  expected: {list(expected)}\n"
        f"  got:      {list(actual.columns)}"
    )
    for name, column in expected.items():
        wanted = as_list(column)
        assert len(actual.index) == len(wanted), (
            f"{path}[{name!r}]: expected {len(wanted)} row(s), got {len(actual.index)}."
        )
        assert_close(list(actual[name]), wanted, rtol=rtol, atol=atol, path=f"{path}[{name!r}]")
