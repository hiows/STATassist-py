"""The train/test partition for the supervised learning family.

The port of ``R/split_data.R``. A split is the first place a model can be handed
something it is not supposed to know, and the two ways that happens are
structural rather than accidental, so both are dealt with here instead of being
left to the caller: ``stratified`` keeps the balance of the whole data set in both
halves, and ``id`` keeps every row of one sampling unit on the same side.

R draws the partition with :func:`caret::createDataPartition`. There is no caret
in Python and adding a machine learning dependency to draw a stratified sample
would be the wrong trade, so the partition is drawn here. It is caret's
algorithm and not a lookalike: up to five quantile bins for a numeric
stratifier, ``ceil(n_k * p)`` taken from each stratum, a stratum of one row kept
for training, and the result sorted. **Which** rows come out differs from R,
because the two languages have different generators; **how many** come out of
each stratum is arithmetic and is the same.
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from ..core.errors import SaValueError, warn
from ..core.random import SaRandom
from ..core.result import SaSplit, metadata
from ..core.validate import check_count, check_scalar_num, fmt_num, resolve_row_vector

__all__ = ["split_data"]

#: Fewest and most quantile bins caret cuts a numeric stratifier into.
_MIN_GROUPS = 2
_MAX_GROUPS = 5


class Folded(NamedTuple):
    """Rows gathered into sampling units, with the stratifier carried up.

    Attributes:
        rows: Zero-based row positions of each unit, units in order of first
            appearance.
        names: What each unit is called, for an error message that has to point
            at one.
        stratum: One entry per unit, in the type it arrived as, or ``None`` when
            there is no stratifier.
    """

    rows: list[np.ndarray]
    names: list[str]
    stratum: pd.Series | None


def _fold_by_unit(
    id: pd.Series | None,  # noqa: A002 - matches the R argument name
    stratified: pd.Series | None,
    n: int,
) -> Folded:
    """Fold rows into sampling units and carry the stratifier up with them.

    Port of ``sa_fold_by_unit()``. Units come out in order of first appearance
    rather than sorted, so a numeric id is not silently reordered as text, which
    is the same rule :func:`~statassist.core.align_by_subject` follows.

    A unit is assigned to one side of the split as a whole, so it can only carry
    one stratum. A unit whose rows disagree is an error rather than a majority
    vote: the two are different designs, and guessing which one was meant would
    produce a split that looks stratified but is not.
    """
    if id is None:
        return Folded(
            rows=[np.array([i]) for i in range(n)],
            names=[str(i) for i in range(n)],
            stratum=stratified,
        )

    keys = [str(value) for value in id]
    units = list(dict.fromkeys(keys))
    where: dict[str, list[int]] = {name: [] for name in units}
    for position, key in enumerate(keys):
        where[key].append(position)
    rows = [np.array(where[name]) for name in units]

    stratum = None
    if stratified is not None:
        values = stratified.to_numpy()
        mixed = [
            name
            for name, index in zip(units, rows, strict=True)
            if pd.unique(values[index]).size > 1
        ]
        if mixed:
            shown = ", ".join(mixed[:5]) + (", ..." if len(mixed) > 5 else "")
            raise SaValueError(
                "`stratified` must be constant within each `id`, since a unit is "
                "assigned to one side of the split as a whole. Offending id(s): " + shown + "."
            )
        first_rows = np.array([index[0] for index in rows])
        stratum = stratified.iloc[first_rows].reset_index(drop=True)

    return Folded(rows=rows, names=units, stratum=stratum)


def _strata_keys(y: pd.Series) -> np.ndarray:
    """Which stratum each unit belongs to, the way caret decides it.

    A numeric stratifier is cut into up to five quantile bins first, which is how
    a continuous outcome is kept from landing entirely on one side of the split.
    It does mean a numeric stratifier is never matched exactly.
    """
    n = len(y)
    if not pd.api.types.is_numeric_dtype(y) or y.dtype == bool:
        return np.asarray([str(value) for value in y])

    groups = max(_MIN_GROUPS, min(_MAX_GROUPS, n))
    values = np.asarray(y, dtype=float)
    # `method="linear"` is R's `quantile()` type 7, the default there too.
    breaks = np.unique(np.quantile(values, np.linspace(0, 1, groups), method="linear"))
    if breaks.size < 2:
        return np.zeros(n, dtype=int)
    # R's `cut(include.lowest = TRUE)` closes the bins on the right, and puts the
    # smallest value in the first bin rather than outside every bin.
    return np.maximum(np.searchsorted(breaks, values, side="left"), 1) - 1


def _data_partition(
    y: pd.Series,
    times: int,
    p: float,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Draw a stratified partition, as ``caret::createDataPartition()`` does.

    Returns:
        One sorted array of zero-based positions per repeat, holding the units
        allocated to training.
    """
    keys = _strata_keys(y)
    strata = list(pd.unique(keys))
    index_by_stratum = [np.flatnonzero(keys == key) for key in strata]

    if keys.dtype.kind not in "iu":
        alone = [
            str(key) for key, index in zip(strata, index_by_stratum, strict=True) if index.size == 1
        ]
        if alone:
            warn(
                "some strata hold a single unit (" + ", ".join(alone) + ") and it "
                "will be selected for the training set, so the test set has none of it."
            )

    out = []
    for _ in range(times):
        picked = [
            index
            if index.size == 1
            else rng.choice(index, size=math.ceil(index.size * p), replace=False)
            for index in index_by_stratum
        ]
        out.append(np.sort(np.concatenate(picked)))
    return out


def split_data(
    data: Any,
    stratified: Any = None,
    id: Any = None,  # noqa: A002 - matches the R argument name
    p_train: float = 0.75,
    times: int = 1,
    seed: int | None = None,
) -> SaSplit:
    """Split data into training and test sets.

    Partitions the rows of a data set into a training half and a test half,
    optionally several times over. The partition is stratified, so the balance of
    the whole data set is preserved in both halves rather than left to the draw,
    and it can be taken over sampling units rather than over rows, so that
    repeated measurements of one subject never end up on both sides.

    Nothing is fitted or transformed here. The point of splitting first is that
    every later step - imputation, scaling, feature selection, hyperparameter
    tuning - is fitted on the training half alone, and this function is what
    makes that half well defined.

    What the partition does with ``stratified`` depends on its type, and the
    difference is worth knowing because it is invisible from the outside:

    * **Categorical or text.** Each distinct value is a stratum. A value
      occurring only once is put into the training set and the test set has none
      of it, which is warned about.
    * **Numeric.** Cut into up to five quantile bins first, and the bins are the
      strata. This is how a continuous outcome is kept from landing entirely on
      one side, but it does mean a numeric stratifier is never matched exactly.
    * **``None``.** No strata. The split is a simple random draw of
      ``ceil(n * p_train)`` rows.

    With ``id`` the whole thing moves up one level. Rows are folded into units,
    each unit takes the stratum its rows agree on, the partition is drawn over
    units, and the chosen units are expanded back into row positions. ``p_train``
    is then a proportion of units, not of rows, and the two differ whenever the
    units have unequal sizes; ``parameters["achieved_p"]`` reports the row
    proportion each repeat actually reached.

    Args:
        data: A frame in wide format, one row per observation. A 2-d array is
            read as one.
        stratified: What to preserve the balance of, either the name of a column
            of ``data`` or a vector with one entry per row. Usually the outcome
            the model will predict. ``None`` draws a simple random split.
        id: Sampling unit, either a column name or a vector with one entry per
            row. Rows sharing a value are assigned to the same side of the split,
            which is what keeps repeated measurements or technical replicates of
            one subject from appearing in both halves. ``None`` treats every row
            as its own unit.
        p_train: Proportion allocated to the training set, strictly between 0
            and 1.
        times: Number of independent splits to draw. The shape of the result does
            not depend on it: one split still comes back as a mapping of one.
        seed: Seed for the draw, or ``None`` to draw from the operating system's
            entropy. Unlike R, seeding is local to the call, so nothing needs
            putting back afterwards.

    Returns:
        A :class:`~statassist.core.SaSplit` of six slots.

        * ``full_data`` - the input, exactly as it was passed in. Held once
          rather than once per repeat.
        * ``datasets`` - one entry per repeat, keyed ``Resample1`` upwards, each
          a mapping of ``train_data``, ``test_data``, and the ``train_rows`` and
          ``test_rows`` they were taken from. The two frames are re-indexed from
          zero, so the row positions are the only record of where a row came from
          and they are kept beside it.
        * ``train_idx`` - the training row positions of every repeat. **Zero
          based**, where R stores one-based ones.
        * ``design`` - what the split was made on: row and unit counts, the
          labels of ``stratified`` and ``id``, and the number of units per
          stratum where the strata are discrete.
        * ``parameters`` - ``p_train``, ``times``, ``seed``, and ``achieved_p``,
          the row proportion each repeat actually reached.
        * ``metadata`` - version and platform of the run.

    Raises:
        SaValueError: If ``data`` is unusable, if there are fewer than two
            sampling units, if ``stratified`` is not constant within a unit, or
            if ``p_train`` leaves the test set empty.

    Examples:
        Stratified on a label: both halves keep the balance of the whole data set
        rather than whatever the draw happened to give.

        >>> import pandas as pd
        >>> data = pd.DataFrame({
        ...     "species": ["a"] * 20 + ["b"] * 20,
        ...     "value": range(40),
        ... })
        >>> sp = split_data(data, stratified="species", seed=1)
        >>> train = sp.datasets["Resample1"]["train_data"]
        >>> [int((train["species"] == level).sum()) for level in ("a", "b")]
        [15, 15]

        Three measurements per subject. Splitting by row would put most subjects
        in both halves; splitting by ``id`` cannot.

        >>> repeated = pd.DataFrame({
        ...     "subject": [f"s{i}" for i in range(20) for _ in range(3)],
        ...     "arm": ["control"] * 30 + ["treated"] * 30,
        ...     "value": range(60),
        ... })
        >>> sp_id = split_data(repeated, stratified="arm", id="subject", seed=1)
        >>> first = sp_id.datasets["Resample1"]
        >>> set(first["train_data"]["subject"]) & set(first["test_data"]["subject"])
        set()

        ``p_train`` is a proportion of units once ``id`` is given, and each
        stratum rounds its share up, so the proportion actually reached is
        reported rather than assumed.

        >>> sp_id.parameters["achieved_p"]["Resample1"]
        0.8
    """
    if isinstance(data, np.ndarray) and data.ndim == 2:
        data = pd.DataFrame(data)
    if not isinstance(data, pd.DataFrame):
        raise SaValueError("`data` must be a data.frame or a matrix.")
    n = len(data.index)
    if n == 0:
        raise SaValueError("`data` has zero rows.")

    p_train = check_scalar_num(p_train, "p_train", 0, 1, lower_open=True, upper_open=True)
    times = check_count(times, "times", 1)

    strat = resolve_row_vector(stratified, "stratified", data)
    unit = resolve_row_vector(id, "id", data)

    folded = _fold_by_unit(unit.value, strat.value, n)
    n_units = len(folded.rows)
    if n_units < 2:
        raise SaValueError(f"a split needs at least 2 sampling units, but `data` has {n_units}.")

    # The partition needs something to stratify on either way. A single level is
    # the honest spelling of "no strata": every unit is in the one class and the
    # draw is a simple random sample of it.
    strata = folded.stratum
    numeric_strata = (
        strata is not None and pd.api.types.is_numeric_dtype(strata) and strata.dtype != bool
    )
    if strata is None:
        y = pd.Series(["all"] * n_units)
    elif numeric_strata:
        if pd.unique(strata.to_numpy()).size < 2:
            raise SaValueError(
                "`stratified` is numeric and constant, so it defines no strata. "
                "Pass `stratified = None` for an unstratified split."
            )
        y = strata
    else:
        # Read as text, so a categorical carrying a level no unit has does not
        # become a stratum with no members.
        y = pd.Series([str(value) for value in strata])

    rand = SaRandom(seed)
    unit_idx = _data_partition(y, times, p_train, rand.rng)

    names = [f"Resample{i + 1}" for i in range(times)]
    train_rows = {
        name: np.sort(np.concatenate([folded.rows[u] for u in picked]))
        for name, picked in zip(names, unit_idx, strict=True)
    }

    if any(n - rows.size == 0 for rows in train_rows.values()):
        raise SaValueError(
            f"`p_train` = {fmt_num(p_train)} leaves the test set empty: the partition "
            f"takes `ceil(n * p_train)` units from each stratum, which here is all "
            f"{n_units} of them. Lower `p_train` or supply more units."
        )

    datasets = {}
    for name, rows in train_rows.items():
        test = np.setdiff1d(np.arange(n), rows)
        datasets[name] = {
            "train_data": data.iloc[rows].reset_index(drop=True),
            "test_data": data.iloc[test].reset_index(drop=True),
            "train_rows": rows,
            "test_rows": test,
        }

    return SaSplit(
        {
            "full_data": data,
            "datasets": datasets,
            "train_idx": train_rows,
            "design": {
                "n_rows": n,
                "n_units": n_units,
                "stratified": strat.label,
                "id": unit.label,
                # A numeric stratifier is binned into quantiles and the bins are
                # not reported back, so there is no honest count to give.
                "strata_n": None
                if strata is None or numeric_strata
                else {key: int((y == key).sum()) for key in sorted(set(y))},
            },
            "parameters": {
                "p_train": p_train,
                "times": times,
                "seed": seed,
                "achieved_p": {name: rows.size / n for name, rows in train_rows.items()},
            },
            "metadata": metadata(),
        }
    )
