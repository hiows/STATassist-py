"""What the package exports, and that the exports fit together.

The individual functions are graded against R elsewhere. What is checked here is
the seam: that every public name is reachable from the top level, and that the
description functions accept the arguments the simulators hand out, without a
translation step in between.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statassist

#: The public surface after Wave B: seven simulators and five description
#: functions.
PUBLIC = (
    "center_by_control",
    "compare_multiple_groups",
    "compare_one_sample",
    "compare_two_groups",
    "diagnose_distribution",
    "draw_butterfly_hist",
    "draw_forest_plot",
    "draw_heatmap",
    "draw_volcano_plot",
    "estimate_significance",
    "make_block_cor",
    "screen_outliers",
    "simulate_categorical_groups",
    "simulate_classification",
    "simulate_factorial_groups",
    "simulate_multiple_groups",
    "simulate_regression",
    "simulate_two_groups",
    "split_data",
    "summarize_association_stats",
    "summarize_descriptive_stats",
)


class TestExports:
    def test_every_public_name_is_reachable_from_the_top_level(self):
        for name in PUBLIC:
            assert callable(getattr(statassist, name)), name

    def test_all_lists_exactly_the_public_names_and_the_version(self):
        assert set(statassist.__all__) == {*PUBLIC, "__version__"}

    def test_all_is_sorted_so_a_new_export_lands_in_one_obvious_place(self):
        assert statassist.__all__ == sorted(statassist.__all__)

    @pytest.mark.parametrize(
        "module",
        [
            "compare",
            "core",
            "diagnose",
            "estimate",
            "kernel",
            "plot",
            "simulate",
            "summarize",
            "transform",
        ],
    )
    def test_each_subpackage_exports_only_names_it_holds(self, module):
        import importlib

        loaded = importlib.import_module(f"statassist.{module}")
        for name in loaded.__all__:
            assert hasattr(loaded, name), f"{module}.{name}"


@pytest.fixture(scope="module")
def simulated():
    return statassist.simulate_two_groups(
        n_feats=6, n_case=18, n_control=18, n_up=2, n_down=2, seed=42
    )


class TestEndToEnd:
    """Phase 1 makes the data, Wave A describes it, with no glue in between."""

    def test_the_simulator_arguments_unpack_into_every_description_function(self, simulated):
        args = simulated.args
        described = statassist.summarize_descriptive_stats(
            args["data"], args["feats"], args["group"], args["group_lv"]
        )
        assert len(described.index) == len(args["feats"]) * len(args["group_lv"])

        diagnosed = statassist.diagnose_distribution(
            args["data"], args["feats"], args["group"], args["group_lv"]
        )
        assert diagnosed["features"] == list(args["feats"])

        associated = statassist.summarize_association_stats(args["data"], args["feats"])
        square = (len(args["feats"]),) * 2
        for method in associated["design"]["methods"]:
            assert associated[method]["corr"].shape == square

        centred = statassist.center_by_control(
            args["data"],
            args["feats"],
            args["group"],
            args["group_lv"],
            input_scale=args["input_scale"],
        )
        assert centred.shape == args["data"].shape

    def test_the_descriptive_mean_of_a_planted_feature_moves_the_way_it_was_planted(
        self, simulated
    ):
        """The first check that crosses the two phases: what Phase 1 put in the
        data is what Wave A reads back out of it."""
        args = simulated.args
        truth = simulated.truth
        described = statassist.summarize_descriptive_stats(
            args["data"], args["feats"], args["group"], args["group_lv"]
        )
        reference, other = args["group_lv"]

        planted = truth[truth["direction"] != "none"]
        for _, row in planted.iterrows():
            of_feature = described[described["features"] == row["features"]]
            control = float(of_feature.loc[of_feature["group"] == reference, "mean"].iloc[0])
            case = float(of_feature.loc[of_feature["group"] == other, "mean"].iloc[0])
            assert np.sign(case - control) == np.sign(row["log2fc"])

    def test_centring_leaves_the_distances_between_the_groups_alone(self, simulated):
        """Which is what lets the centred frame be handed to a comparison with
        the arguments unchanged: the transformation is one constant per feature."""
        args = simulated.args
        centred = statassist.center_by_control(
            args["data"],
            args["feats"],
            args["group"],
            args["group_lv"],
            input_scale=args["input_scale"],
        )
        group = pd.Series(args["group"])
        reference, other = args["group_lv"]
        for name in args["feats"]:
            before = (
                args["data"].loc[(group == other).to_numpy(), name].mean()
                - args["data"].loc[(group == reference).to_numpy(), name].mean()
            )
            after = (
                centred.loc[(group == other).to_numpy(), name].mean()
                - centred.loc[(group == reference).to_numpy(), name].mean()
            )
            assert after == pytest.approx(before)

    def test_a_multi_group_simulation_diagnoses_across_all_of_its_levels(self):
        sim = statassist.simulate_multiple_groups(
            n_feats=4, n_control=12, n_treat=[12, 12], n_up=1, n_down=1, seed=8
        )
        args = sim.args
        diagnosed = statassist.diagnose_distribution(
            args["data"], args["feats"], args["group"], args["group_lv"]
        )
        assert len(diagnosed["normality"].index) == len(args["feats"]) * len(args["group_lv"])
        assert list(diagnosed["variance"]["n_groups"]) == [len(args["group_lv"])] * len(
            args["feats"]
        )
