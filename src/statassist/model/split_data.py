"""Train/test data partition."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

from statassist.contracts.split import sa_new_split
from statassist.utils.validate import (
    sa_check_count,
    sa_check_scalar_num,
    sa_preserve_seed,
    sa_resolve_row_vector,
)


def _fold_by_unit(
    id_: np.ndarray | None,
    stratified: np.ndarray | None,
    n: int,
) -> dict[str, Any]:
    if id_ is None:
        rows = {str(i): [i] for i in range(n)}
        return {"rows": rows, "stratum": stratified}
    units, inv = np.unique(id_, return_inverse=True)
    rows: dict[str, list[int]] = {}
    for ui, unit in enumerate(units):
        rows[str(unit)] = np.where(inv == ui)[0].tolist()
    stratum = None
    if stratified is not None:
        mixed = []
        for unit, idxs in rows.items():
            vals = np.unique(stratified[idxs])
            if len(vals) > 1:
                mixed.append(unit)
        if mixed:
            shown = mixed[:5]
            suffix = ", ..." if len(mixed) > 5 else ""
            raise ValueError(
                "`stratified` must be constant within each `id`, since a unit is "
                "assigned to one side of the split as a whole. Offending id(s): "
                f"{', '.join(shown)}{suffix}."
            )
        stratum = np.array([stratified[idxs[0]] for idxs in rows.values()])
    return {"rows": rows, "stratum": stratum}


def split_data(
    data: pd.DataFrame | np.ndarray,
    *,
    stratified: Any = None,
    id: Any = None,
    p_train: float = 0.75,
    times: int = 1,
    seed: int | None = None,
) -> dict[str, Any]:
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
    if not isinstance(data, pd.DataFrame):
        raise ValueError("`data` must be a data.frame or a matrix.")
    n = len(data)
    if n == 0:
        raise ValueError("`data` has zero rows.")

    sa_check_scalar_num(p_train, "p_train", 0, 1, lower_open=True, upper_open=True)
    times = sa_check_count(times, "times", 1)

    strat = sa_resolve_row_vector(stratified, "stratified", data)
    unit = sa_resolve_row_vector(id, "id", data)
    folded = _fold_by_unit(unit["value"], strat["value"], n)
    unit_rows = folded["rows"]
    n_units = len(unit_rows)
    if n_units < 2:
        raise ValueError(
            f"a split needs at least 2 sampling units, but `data` has {n_units}."
        )

    strata = folded["stratum"]
    if strata is None:
        y = np.array(["all"] * n_units)
        strata_factor = None
    elif np.issubdtype(np.asarray(strata).dtype, np.number):
        if len(np.unique(strata)) < 2:
            raise ValueError(
                "`stratified` is numeric and constant, so it defines no strata. "
                "Pass `stratified = None` for an unstratified split."
            )
        y = np.asarray(strata)
        strata_factor = None
    else:
        y = np.asarray(strata, dtype=str)
        strata_factor = dict(zip(*np.unique(y, return_counts=True)))

    unit_keys = list(unit_rows.keys())
    unit_indices = np.arange(n_units)
    datasets: list[dict[str, Any]] = []
    train_idx: list[list[int]] = []
    achieved: list[float] = []

    with sa_preserve_seed(seed):
        for t in range(times):
            if strata is None or (not np.issubdtype(np.asarray(strata).dtype, np.number) and len(np.unique(y)) == 1):
                n_train_units = max(1, int(np.ceil(n_units * p_train)))
                perm = np.random.permutation(n_units)
                chosen_units = perm[:n_train_units]
            else:
                splitter = StratifiedShuffleSplit(n_splits=1, train_size=p_train, random_state=None)
                chosen_units, _ = next(splitter.split(unit_indices.reshape(-1, 1), y))
            rows = sorted(
                idx for ui in chosen_units for idx in unit_rows[unit_keys[ui]]
            )
            test = sorted(set(range(n)) - set(rows))
            if not test:
                raise ValueError(
                    f"`p_train` = {p_train} leaves the test set empty: lower `p_train` "
                    f"or supply more units."
                )
            train_data = data.iloc[rows].reset_index(drop=True)
            test_data = data.iloc[test].reset_index(drop=True)
            name = f"Resample{t + 1}"
            datasets.append(
                {
                    "train_data": train_data,
                    "test_data": test_data,
                    "train_rows": rows,
                    "test_rows": test,
                }
            )
            train_idx.append(rows)
            achieved.append(len(rows) / n)

    design = {
        "n_rows": n,
        "n_units": n_units,
        "stratified": strat["label"],
        "id": unit["label"],
        "strata_n": strata_factor,
    }
    parameters = {
        "p_train": p_train,
        "times": times,
        "seed": seed,
        "achieved_p": achieved if times == 1 else dict(zip([f"Resample{i+1}" for i in range(times)], achieved)),
    }
    return sa_new_split(
        full_data=data,
        datasets=datasets,
        train_idx=train_idx,
        design=design,
        parameters=parameters,
    )
