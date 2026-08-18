"""Simulate a two-group experiment with known truth (R simulate_two_groups.R)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from statassist.utils.rng_r import get_rng, sa_r_seed
from statassist.utils.validate import sa_check_count, sa_check_range


def simulate_two_groups(
    n_feats: int = 100,
    n_case: int = 50,
    n_control: int = 50,
    n_up: int = 15,
    n_down: int = 15,
    expr_range: tuple[float, float] = (2, 12),
    case_sd: tuple[float, float] = (1.8, 3.2),
    control_sd: tuple[float, float] = (1.2, 2.4),
    deg_log2fc: tuple[float, float] = (1, 2.5),
    group_lv: list[str] | None = None,
    seed: float | None = None,
) -> dict:
    group_lv = list(group_lv or ["control", "case"])
    n_feats = sa_check_count(n_feats, "n_feats", 1)
    n_case = sa_check_count(n_case, "n_case", 2)
    n_control = sa_check_count(n_control, "n_control", 2)
    n_up = sa_check_count(n_up, "n_up")
    n_down = sa_check_count(n_down, "n_down")
    if n_up + n_down > n_feats:
        raise ValueError(
            f"`n_up` + `n_down` is {n_up + n_down}, which is more features "
            f"than the {n_feats} that `n_feats` asks for."
        )
    if len(group_lv) != 2 or group_lv[0] == group_lv[1]:
        raise ValueError("`group_lv` must be two distinct non-missing group labels.")
    sa_check_range(expr_range, "expr_range")
    sa_check_range(case_sd, "case_sd", 0)
    sa_check_range(control_sd, "control_sd", 0)
    sa_check_range(deg_log2fc, "deg_log2fc", 0)

    with sa_r_seed(seed):
        rng = get_rng()
        feats = [f"gene_{i}" for i in range(1, n_feats + 1)]
        baseline = rng.runif(n_feats, expr_range[0], expr_range[1])
        sd_case = rng.runif(n_feats, case_sd[0], case_sd[1])
        sd_control = rng.runif(n_feats, control_sd[0], control_sd[1])
        direction = np.array(["none"] * n_feats, dtype=object)
        delta = np.zeros(n_feats)
        if n_up + n_down > 0:
            picked = rng.sample_int(n_feats, n_up + n_down) - 1
            up_idx = picked[:n_up]
            down_idx = picked[n_up:]
            direction[up_idx] = "up"
            direction[down_idx] = "down"
            delta[up_idx] = rng.runif(n_up, deg_log2fc[0], deg_log2fc[1])
            delta[down_idx] = -rng.runif(n_down, deg_log2fc[0], deg_log2fc[1])

        case_values = np.column_stack(
            [rng.rnorm(n_case, baseline[i] + delta[i], sd_case[i]) for i in range(n_feats)]
        )
        control_values = np.column_stack(
            [rng.rnorm(n_control, baseline[i], sd_control[i]) for i in range(n_feats)]
        )
        values = np.vstack([control_values, case_values])
        data = pd.DataFrame(values, columns=feats)
        group = np.repeat(group_lv, [n_control, n_case])

    return {
        "args": {
            "data": data,
            "feats": feats,
            "group": group,
            "group_lv": group_lv,
            "input_scale": "log2",
        },
        "truth": pd.DataFrame(
            {
                "features": feats,
                "direction": direction,
                "log2fc": delta,
                "baseline": baseline,
                "sd_case": sd_case,
                "sd_control": sd_control,
            }
        ),
    }
