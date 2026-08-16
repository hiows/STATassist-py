"""Centre every feature on the control group."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from statassist.utils.foldchange import sa_fc_center, sa_resolve_fc_mean
from statassist.utils.validate import sa_control_first, sa_validate_wide_input


def sa_control_baseline(
    v: np.ndarray,
    control: str,
    mean_type: str,
    input_scale: str,
) -> float:
    centre = sa_fc_center(v[np.isfinite(v)], control, mean_type, input_scale)
    if input_scale == "log2":
        baseline = np.log2(centre)
        if not np.isfinite(baseline):
            raise ValueError(
                f"the {control} centre is {centre} on the raw scale, which has no log2 to subtract."
            )
        return float(baseline)
    if not np.isfinite(centre) or centre <= 0:
        raise ValueError(
            f"the {control} centre is {centre}, and dividing by it is not valid. "
            'Pass logged values with `input_scale = "log2"` instead.'
        )
    return float(centre)


def center_by_control(
    data: pd.DataFrame | np.ndarray,
    feats: list[str],
    group: pd.Series | np.ndarray | list[Any],
    group_lv: list[str],
    *,
    control_label: str | None = None,
    fc_mean: str | None = None,
    input_scale: str = "raw",
) -> pd.DataFrame:
    if input_scale not in ("raw", "log2"):
        raise ValueError("`input_scale` must be one of 'raw' or 'log2'.")
    fc_mean = sa_resolve_fc_mean(fc_mean, input_scale, fc_mean is None)

    inp = sa_validate_wide_input(data, feats, group, group_lv)
    control = sa_control_first(list(inp["group"].categories), control_label or group_lv[0])[0]

    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
    else:
        data = data.copy()

    if inp["n_dropped"] > 0:
        print(
            f"Kept {inp['n_dropped']} row(s) belonging to a level outside "
            "`group_lv`. They are centred on the same baseline but take no "
            "part in it."
        )

    group_arr = np.asarray(group).astype(str)
    ctrl_rows = np.where(group_arr == control)[0]
    failures: dict[str, str] = {}

    for f in feats:
        v = data[f].to_numpy(dtype=float)
        try:
            baseline = sa_control_baseline(v[ctrl_rows], control, fc_mean, input_scale)
            data[f] = v - baseline if input_scale == "log2" else v / baseline
        except Exception as exc:
            failures[f] = str(exc)
            data[f] = np.nan

    if failures:
        lines = "\n".join(f"  {k}: {v}" for k, v in failures.items())
        warnings.warn(
            f"The control baseline could not be taken for {len(failures)} of "
            f"{len(feats)} feature(s); those columns are all NA:\n{lines}",
            UserWarning,
            stacklevel=2,
        )
    return data
