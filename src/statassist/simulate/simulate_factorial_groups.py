"""Simulate a crossed-factor experiment with known truth (R simulate_factorial_groups.R)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from statassist.utils.factorial_utils import (
    sa_fact_design,
    sa_fact_partner,
    sa_fact_plant,
    sa_fact_shapes,
    sa_fact_shuffle,
    sa_fact_truth,
    sa_fact_truth_cell,
    sa_fact_truth_contrast,
    sa_fact_truth_term,
)
from statassist.utils.rng_r import get_rng, sa_r_seed
from statassist.utils.simulate_utils import sa_sim_allocate, sa_sim_pattern_mix
from statassist.utils.validate import sa_check_count, sa_check_range, sa_check_scalar_num


def simulate_factorial_groups(
    n_feats: int = 100,
    factor_lv: dict[str, list[str]] | None = None,
    within: list[str] | None = None,
    n_per_cell: int | list[int] = 20,
    n_up: int | None = None,
    n_down: int | None = None,
    term_mix: dict[str, float] | None = None,
    pattern_mix: dict[str, float] | None = None,
    expr_range: tuple[float, float] = (2, 12),
    ref_sd: tuple[float, float] = (1.2, 2.4),
    cell_sd: tuple[float, float] = (1.8, 3.2),
    deg_log2fc: tuple[float, float] = (1, 2.5),
    interaction_scale: float = 0.8,
    subject_sd: tuple[float, float] = (2, 4),
    feat_prefix: str = "prot",
    seed: float | None = None,
) -> dict[str, Any]:
    if factor_lv is None:
        factor_lv = {
            "treatment": ["control", "treat_A", "treat_B", "treat_C"],
            "sex": ["male", "female"],
        }

    design = sa_fact_design(factor_lv, within, n_per_cell)
    factor_lv = design["factor_lv"]
    fac_names = list(factor_lv.keys())
    n_cells = design["n_cells"]
    has_within = len(design["within"]) > 0

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

    if term_mix is None:
        term_mix = {s: 1.0 for s in sa_fact_shapes()}
    if pattern_mix is None:
        pattern_mix = {"all": 1, "gradient": 1, "single": 1}
    shapes = sa_sim_pattern_mix(term_mix, tuple(sa_fact_shapes()), "term_mix")
    mix = sa_sim_pattern_mix(pattern_mix)
    sa_check_range(expr_range, "expr_range")
    sa_check_range(ref_sd, "ref_sd", 0)
    sa_check_range(cell_sd, "cell_sd", 0)
    sa_check_range(deg_log2fc, "deg_log2fc", 0)
    sa_check_scalar_num(interaction_scale, "interaction_scale", 0, lower_open=True)
    sa_check_range(subject_sd, "subject_sd", 0)
    if not isinstance(feat_prefix, str) or not feat_prefix:
        raise ValueError("`feat_prefix` must be a single non-empty string.")

    with sa_r_seed(seed):
        rng = get_rng()
        feats = [f"{feat_prefix}_{i}" for i in range(1, n_feats + 1)]
        baseline = rng.runif(n_feats, expr_range[0], expr_range[1])

        sd_mat = np.zeros((n_feats, n_cells))
        sd_mat[:, design["ref_cell"]] = rng.runif(n_feats, ref_sd[0], ref_sd[1])
        for j in range(n_cells):
            if j != design["ref_cell"]:
                sd_mat[:, j] = rng.runif(n_feats, cell_sd[0], cell_sd[1])

        sd_subject = (
            rng.runif(n_feats, subject_sd[0], subject_sd[1])
            if has_within
            else np.full(n_feats, np.nan)
        )

        delta = np.zeros((n_feats, n_cells))
        direction = np.array(["none"] * n_feats, dtype=object)
        pattern = np.array(["none"] * n_feats, dtype=object)
        spread = np.array(["none"] * n_feats, dtype=object)
        partner = np.array([None] * n_feats, dtype=object)

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
            alloc_up = sa_sim_allocate(n_up, shapes)
            alloc_down = sa_sim_allocate(n_down, shapes)
            plant_shape = np.array(
                [nm for nm in shapes for _ in range(alloc_up[nm])]
                + [nm for nm in shapes for _ in range(alloc_down[nm])],
                dtype=object,
            )
            plant_spread = np.array(
                sa_fact_shuffle(
                    [nm for nm in mix for _ in range(sa_sim_allocate(n_up, mix)[nm])]
                )
                + sa_fact_shuffle(
                    [nm for nm in mix for _ in range(sa_sim_allocate(n_down, mix)[nm])]
                ),
                dtype=object,
            )

            cells = design["cells"]
            ref_cell = design["ref_cell"]
            for k, i in enumerate(plant_idx):
                mate = sa_fact_partner(plant_shape[k], fac_names)
                eff = sa_fact_plant(
                    plant_mag[k],
                    plant_shape[k],
                    plant_spread[k],
                    mate,
                    factor_lv,
                    cells,
                    interaction_scale,
                )
                delta[i, :] = eff - eff[ref_cell]
                pattern[i] = plant_shape[k]
                spread[i] = plant_spread[k]
                partner[i] = mate

        center = baseline[:, None] + delta
        offsets = None
        if has_within:
            offsets = np.column_stack(
                [
                    rng.rnorm(design["n_units"], 0, sd_subject[i])
                    for i in range(n_feats)
                ]
            )

        cell_idx = design["cell_idx"]
        n_rows = design["n_rows"]
        values = np.column_stack(
            [
                rng.rnorm(
                    n_rows,
                    center[i, cell_idx],
                    sd_mat[i, cell_idx],
                )
                + (
                    offsets[design["subject_idx"], i]
                    if has_within
                    else 0.0
                )
                for i in range(n_feats)
            ]
        )

        data = pd.DataFrame(values, columns=feats)
        args: dict[str, Any] = {
            "data": data,
            "feats": feats,
            "factors": design["factors"],
            "factor_lv": factor_lv,
            "input_scale": "log2",
        }
        if has_within:
            args["within"] = design["within"]
            args["id"] = design["subject"]

    return {
        "args": args,
        "truth": sa_fact_truth(
            feats,
            delta,
            design,
            pattern,
            spread,
            direction,
            partner,
            baseline,
            sd_subject,
        ),
        "truth_term": sa_fact_truth_term(
            feats, delta, design, pattern != "none"
        ),
        "truth_cell": sa_fact_truth_cell(feats, delta, center, sd_mat, design),
        "truth_contrast": sa_fact_truth_contrast(feats, delta, design),
    }
