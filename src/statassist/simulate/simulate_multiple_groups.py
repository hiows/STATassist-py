"""Simulate a control-versus-treatments experiment with known truth (R simulate_multiple_groups.R)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from statassist.utils.rng_r import get_rng, sa_r_seed
from statassist.utils.simulate_utils import (
    sa_sim_allocate,
    sa_sim_design,
    sa_sim_pattern_delta,
    sa_sim_pattern_mix,
    sa_sim_truth,
    sa_sim_truth_contrast,
    sa_sim_truth_group,
)
from statassist.utils.validate import sa_check_count, sa_check_flag, sa_check_range


def simulate_multiple_groups(
    n_feats: int = 100,
    n_control: int = 50,
    n_treat: list[int] | None = None,
    n_up: int | None = None,
    n_down: int | None = None,
    pattern_mix: dict[str, float] | None = None,
    expr_range: tuple[float, float] = (2, 12),
    control_sd: tuple[float, float] = (1.2, 2.4),
    treat_sd: tuple[float, float] = (1.8, 3.2),
    deg_log2fc: tuple[float, float] = (1, 2.5),
    paired: bool = False,
    subject_sd: tuple[float, float] = (2, 4),
    group_lv: list[str] | None = None,
    feat_prefix: str = "prot",
    seed: float | None = None,
) -> dict[str, Any]:
    use_default_n_treat = n_treat is None
    if n_treat is None:
        n_treat = [50, 50, 50]

    sa_check_flag(paired, "paired")
    design = sa_sim_design(n_control, n_treat, group_lv, use_default_n_treat, paired)
    group_lv = design["group_lv"]
    sizes = design["sizes"]
    n_lv = len(group_lv)
    n_treat_groups = n_lv - 1

    n_feats = sa_check_count(n_feats, "n_feats", 1)
    if n_up is None:
        n_up = round(0.15 * n_feats)
    if n_down is None:
        n_down = round(0.15 * n_feats)
    n_up = sa_check_count(n_up, "n_up")
    n_down = sa_check_count(n_down, "n_down")
    if n_up + n_down > n_feats:
        raise ValueError(
            f"`n_up` + `n_down` is {n_up + n_down}, which is more features "
            f"than the {n_feats} that `n_feats` asks for."
        )

    if pattern_mix is None:
        pattern_mix = {"all": 1, "gradient": 1, "single": 1}
    mix = sa_sim_pattern_mix(pattern_mix)
    sa_check_range(expr_range, "expr_range")
    sa_check_range(control_sd, "control_sd", 0)
    sa_check_range(treat_sd, "treat_sd", 0)
    sa_check_range(deg_log2fc, "deg_log2fc", 0)
    sa_check_range(subject_sd, "subject_sd", 0)
    if not isinstance(feat_prefix, str) or not feat_prefix:
        raise ValueError("`feat_prefix` must be a single non-empty string.")

    with sa_r_seed(seed):
        rng = get_rng()
        feats = [f"{feat_prefix}_{i}" for i in range(1, n_feats + 1)]
        baseline = rng.runif(n_feats, expr_range[0], expr_range[1])

        sd_mat = np.zeros((n_feats, n_lv))
        sd_mat[:, 0] = rng.runif(n_feats, control_sd[0], control_sd[1])
        for g in range(n_treat_groups):
            sd_mat[:, g + 1] = rng.runif(n_feats, treat_sd[0], treat_sd[1])

        sd_subject = (
            rng.runif(n_feats, subject_sd[0], subject_sd[1])
            if paired
            else np.full(n_feats, np.nan)
        )

        delta = np.zeros((n_feats, n_lv))
        direction = np.array(["none"] * n_feats, dtype=object)
        pattern = np.array(["none"] * n_feats, dtype=object)

        if n_up + n_down > 0:
            picked = rng.sample_int(n_feats, n_up + n_down) - 1
            up_idx = picked[:n_up]
            down_idx = picked[n_up:]
            direction[up_idx] = "up"
            direction[down_idx] = "down"

            plant_idx = np.concatenate([up_idx, down_idx])
            plant_mag = np.concatenate(
                [
                    rng.runif(n_up, deg_log2fc[0], deg_log2fc[1]),
                    -rng.runif(n_down, deg_log2fc[0], deg_log2fc[1]),
                ]
            )
            alloc_up = sa_sim_allocate(n_up, mix)
            alloc_down = sa_sim_allocate(n_down, mix)
            plant_pat = np.array(
                [nm for nm in mix for _ in range(alloc_up[nm])]
                + [nm for nm in mix for _ in range(alloc_down[nm])],
                dtype=object,
            )

            for k, i in enumerate(plant_idx):
                delta[i, 1:] = sa_sim_pattern_delta(
                    plant_mag[k], plant_pat[k], n_treat_groups
                )
                pattern[i] = plant_pat[k]

        center = baseline[:, None] + delta
        offsets = None
        if paired:
            offsets = np.column_stack(
                [rng.rnorm(sizes[0], 0, sd_subject[i]) for i in range(n_feats)]
            )

        blocks = []
        for g in range(n_lv):
            values = np.column_stack(
                [
                    rng.rnorm(sizes[g], center[i, g], sd_mat[i, g])
                    for i in range(n_feats)
                ]
            )
            if paired:
                values = values + offsets
            blocks.append(values)

        data = pd.DataFrame(np.vstack(blocks), columns=feats)
        group = np.repeat(group_lv, sizes)

        args: dict[str, Any] = {
            "data": data,
            "feats": feats,
            "group": group,
            "group_lv": group_lv,
            "input_scale": "log2",
        }
        if paired:
            subjects = [f"subject_{i}" for i in range(1, sizes[0] + 1)]
            args["id"] = np.tile(subjects, n_lv)
            args["paired"] = True

    return {
        "args": args,
        "truth": sa_sim_truth(
            feats, delta, group_lv, pattern, direction, baseline, sd_subject
        ),
        "truth_group": sa_sim_truth_group(
            feats, delta, center, sd_mat, group_lv, sizes
        ),
        "truth_contrast": sa_sim_truth_contrast(feats, delta, group_lv),
    }
