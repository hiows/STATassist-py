"""``compare_two_groups`` against the object R's version assembled.

Both halves of the port are graded here. The numbers come from R's three engines
and are compared at the golden tolerance; the assembly - which rows were paired,
which were dropped, what order the columns are in - is compared with them,
because a table of correct numbers in the wrong direction is the failure this
function is most exposed to.
"""

from __future__ import annotations

import functools
import math

import numpy as np
import pandas as pd
import pytest
from golden import as_list, assert_close, assert_frame_close, load_case

from statassist import compare_two_groups, simulate_two_groups
from statassist.core.contracts import test_table_columns as contract_columns
from statassist.core.errors import SaValueError, SaWarning

FC_FEATS = ["prot_1", "prot_2"]
PAIR_FEATS = ["metab_1", "metab_2"]


def _assert_case(produced, expected, path: str) -> None:
    """Grade one comparison against the frozen R object."""
    design = produced["design"]
    assert design["group_lv"] == expected["group_lv"], f"{path}: group order differs."
    assert_close(design["pairing"], expected["pairing"], path=f"{path}[pairing]")
    assert design["n_dropped"] == expected["n_dropped"]
    assert design["unmatched_ids"] == as_list(expected["unmatched_ids"])
    assert_close(produced["parameters"]["tr"], expected["tr"], path=f"{path}[tr]")
    assert produced["parameters"]["fc_mean"] == expected["fc_mean"]

    assert_frame_close(produced["effect"], expected["effect"], path=f"{path}[effect]")
    for name in ("t_test", "wilcox_test", "robust_test"):
        assert_frame_close(produced["tests"][name], expected[name], path=f"{path}[{name}]")


class TestIndependent:
    @pytest.mark.parametrize(
        ("key", "kwargs"),
        [
            ("plain", {}),
            ("greater", {"alternative": "greater"}),
            ("less_90", {"alternative": "less", "conf_level": 0.90}),
            ("reversed", {"control_label": "case"}),
            ("geom", {"fc_mean": "geom"}),
        ],
    )
    def test_matches_r(self, key, kwargs):
        frame, expected = load_case("two_group_independent")
        produced = compare_two_groups(
            frame, FC_FEATS, frame["group"], ["ctrl", "case"], diagnose=False, **kwargs
        )
        _assert_case(produced, expected[key], key)

    def test_a_level_outside_group_lv_is_dropped_rather_than_tested(self):
        frame, expected = load_case("two_group_independent")
        produced = compare_two_groups(
            frame,
            FC_FEATS,
            frame["group"],
            ["other", "case"],
            p_adjust="holm",
            diagnose=False,
        )
        assert produced["design"]["n_dropped"] == 6
        _assert_case(produced, expected["dropped"], "dropped")


class TestLog2Input:
    @pytest.mark.parametrize(
        ("key", "kwargs"),
        [("default_geom", {}), ("explicit_arith", {"fc_mean": "arith"})],
    )
    def test_matches_r(self, key, kwargs):
        frame, expected = load_case("two_group_log2")
        produced = compare_two_groups(
            frame,
            FC_FEATS,
            frame["group"],
            ["ctrl", "case"],
            input_scale="log2",
            diagnose=False,
            **kwargs,
        )
        _assert_case(produced, expected[key], key)

    def test_only_the_effect_table_leaves_the_log2_scale(self):
        """The tests run on the values as supplied, which is why they were logged."""
        frame, _ = load_case("two_group_log2")
        produced = compare_two_groups(
            frame, FC_FEATS, frame["group"], ["ctrl", "case"], input_scale="log2"
        )
        case = frame.loc[frame["group"] == "case", "prot_1"]
        assert produced["tests"]["t_test"]["x_mean"].iloc[0] == pytest.approx(case.mean())
        assert (
            produced["effect"]["x_center"].iloc[0] > produced["tests"]["t_test"]["x_mean"].iloc[0]
        )


class TestPaired:
    @pytest.mark.parametrize(
        ("key", "kwargs"),
        [
            ("by_order", {}),
            ("by_id", {"id": True}),
            ("tr_10", {"id": True, "tr": 0.1}),
            ("greater", {"id": True, "alternative": "greater"}),
        ],
    )
    def test_matches_r(self, key, kwargs):
        frame, expected = load_case("two_group_paired")
        by_id = kwargs.pop("id", False)
        produced = compare_two_groups(
            frame,
            PAIR_FEATS,
            frame["group"],
            ["pre", "post"],
            id=frame["subject"] if by_id else None,
            paired=True,
            diagnose=False,
            **kwargs,
        )
        _assert_case(produced, expected[key], key)

    def test_row_order_pairing_and_id_pairing_differ_on_reordered_rows(self):
        """The one failure mode `id` exists for, graded against both answers."""
        frame, expected = load_case("two_group_paired_id")
        by_order = compare_two_groups(
            frame, PAIR_FEATS, frame["group"], ["pre", "post"], paired=True, diagnose=False
        )
        by_id = compare_two_groups(
            frame,
            PAIR_FEATS,
            frame["group"],
            ["pre", "post"],
            id=frame["subject"],
            paired=True,
            diagnose=False,
        )
        _assert_case(by_order, expected["shuffled_by_order"], "shuffled_by_order")
        _assert_case(by_id, expected["shuffled_by_id"], "shuffled_by_id")

        # The means survive the wrong pairing; the paired standard error does not.
        assert by_order["tests"]["t_test"]["x_mean"].iloc[0] == pytest.approx(
            by_id["tests"]["t_test"]["x_mean"].iloc[0]
        )
        assert by_order["tests"]["t_test"]["stderr"].iloc[0] != pytest.approx(
            by_id["tests"]["t_test"]["stderr"].iloc[0]
        )

    def test_an_id_in_only_one_group_is_dropped(self):
        frame, expected = load_case("two_group_unmatched")
        produced = compare_two_groups(
            frame,
            PAIR_FEATS,
            frame["group"],
            ["pre", "post"],
            id=frame["subject"],
            paired=True,
            diagnose=False,
        )
        assert produced["design"]["unmatched_ids"] == ["s04"]
        _assert_case(produced, expected["holed"], "holed")

    def test_row_order_pairing_refuses_unequal_groups(self):
        frame, _ = load_case("two_group_unmatched")
        with pytest.raises(SaValueError, match="same number of rows"):
            compare_two_groups(frame, PAIR_FEATS, frame["group"], ["pre", "post"], paired=True)

    def test_id_without_paired_is_ignored_and_says_so(self):
        frame, _ = load_case("two_group_paired")
        with pytest.warns(SaWarning, match="only used to form pairs"):
            compare_two_groups(
                frame,
                PAIR_FEATS,
                frame["group"],
                ["pre", "post"],
                id=frame["subject"],
                diagnose=False,
            )


class TestUnusableFeatures:
    def test_a_feature_that_cannot_be_tested_is_a_missing_row_not_an_abort(self):
        frame, expected = load_case("two_group_gappy")
        with pytest.warns(SaWarning):
            produced = compare_two_groups(
                frame,
                ["prot_1", "prot_2", "flat"],
                frame["group"],
                ["ctrl", "case"],
                diagnose=False,
            )
        _assert_case(produced, expected["holes"], "holes")

    def test_the_sample_sizes_follow_the_holes_in_each_feature(self):
        frame, _ = load_case("two_group_gappy")
        with pytest.warns(SaWarning):
            produced = compare_two_groups(
                frame,
                ["prot_1", "prot_2", "flat"],
                frame["group"],
                ["ctrl", "case"],
                diagnose=False,
            )
        sizes = produced["tests"]["t_test"].set_index("features")["n_used"]
        assert sizes["prot_1"] < sizes["prot_2"]


@functools.lru_cache(maxsize=1)
def _planted():
    """A simulation and its comparison, built once for the contract tests."""
    sim = simulate_two_groups(n_feats=6, n_case=15, n_control=15, n_up=2, n_down=2, seed=7)
    return sim, compare_two_groups(**sim.args)


class TestContract:
    @pytest.fixture
    def result(self):
        return _planted()

    def test_a_simulation_unpacks_straight_into_the_comparison(self, result):
        sim, res = result
        assert res["features"] == list(sim.args["feats"])
        assert res["analysis"] == "two_group_comparison"

    def test_every_table_carries_the_contract_columns_in_order(self, result):
        _, res = result
        assert list(res["effect"].columns) == [
            "features",
            "x_center",
            "y_center",
            "fold_change",
            "log2fc",
        ]
        for table in res["tests"].values():
            assert list(table["features"]) == res["features"]
            for name in contract_columns():
                assert name in table.columns
            # `pval_adj` sits immediately after `pval`, so a reader scanning the
            # table finds the raw and the adjusted value side by side.
            names = list(table.columns)
            assert names[names.index("pval") + 1] == "pval_adj"

    def test_two_groups_have_no_post_hoc_stage(self):
        """The omnibus comparison is already the only contrast there is."""
        sim = simulate_two_groups(n_feats=2, n_case=10, n_control=10, n_up=1, n_down=1, seed=2)
        res = compare_two_groups(**sim.args, diagnose=False)
        assert "posthoc" not in res
        assert "pairwise" not in res
        assert "terms" not in res

    def test_the_planted_direction_comes_back_signed(self, result):
        sim, res = result
        planted = sim.truth.set_index("features")["log2fc"]
        found = res["effect"].set_index("features")["log2fc"]
        moved = planted[planted != 0].index
        assert (np.sign(found[moved]) == np.sign(planted[moved])).all()

    def test_reversing_the_reference_reverses_every_direction(self, result):
        sim, res = result
        flipped = compare_two_groups(
            **sim.args, control_label=sim.args["group_lv"][1], diagnose=False
        )
        assert flipped["design"]["group_lv"] == list(reversed(res["design"]["group_lv"]))
        assert_close(
            list(flipped["effect"]["log2fc"]),
            [-value for value in res["effect"]["log2fc"]],
        )
        assert_close(
            list(flipped["tests"]["t_test"]["mean_diff"]),
            [-value for value in res["tests"]["t_test"]["mean_diff"]],
        )
        # The two-sided p-value does not care which way round the pair is.
        assert_close(list(flipped["tests"]["t_test"]["pval"]), list(res["tests"]["t_test"]["pval"]))

    def test_a_one_sided_alternative_leaves_the_untested_side_open(self, result):
        sim, _ = result
        greater = compare_two_groups(**sim.args, alternative="greater", diagnose=False)
        for name in ("t_test", "wilcox_test"):
            assert (greater["tests"][name]["upper_conf"] == math.inf).all()
        # The robust interval is on the probability scale, so "open" is 1 and not
        # infinity.
        assert (greater["tests"]["robust_test"]["upper_conf"] == 1.0).all()

    def test_diagnostics_are_attached_only_when_asked_for(self, result):
        sim, res = result
        assert res["diagnostics"] is not None
        assert set(res["diagnostics"]) == {"normality", "variance", "summary"}
        # Reference first, the way every other per-level table in the package is.
        assert list(res["diagnostics"]["normality"]["group"])[:2] == res["design"]["group_lv"]
        assert compare_two_groups(**sim.args, diagnose=False)["diagnostics"] is None

    def test_repr_summarises_the_tests_rather_than_printing_them(self, result):
        _, res = result
        text = repr(res)
        assert "two_group_comparison" in text
        assert "(independent)" in text
        assert "$t_test" in text
        assert "$diagnostics attached" in text


class TestArgumentChecks:
    @pytest.fixture
    def frame(self):
        return pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "g": ["x", "x", "x", "y", "y", "y"],
            }
        )

    def test_more_than_two_levels_is_refused(self, frame):
        with pytest.raises(SaValueError, match="exactly 2 levels"):
            compare_two_groups(frame, "a", frame["g"], ["x", "y", "z"])

    def test_an_unknown_alternative_is_refused_by_name(self, frame):
        with pytest.raises(SaValueError, match="`alternative` must be one of"):
            compare_two_groups(frame, "a", frame["g"], ["x", "y"], alternative="bigger")

    def test_an_out_of_range_conf_level_is_refused(self, frame):
        with pytest.raises(SaValueError, match="conf_level"):
            compare_two_groups(frame, "a", frame["g"], ["x", "y"], conf_level=1.0)

    def test_a_trimming_proportion_of_a_half_is_refused(self, frame):
        with pytest.raises(SaValueError, match="`tr`"):
            compare_two_groups(frame, "a", frame["g"], ["x", "y"], tr=0.5)

    def test_an_unknown_input_scale_is_refused(self, frame):
        with pytest.raises(SaValueError, match="`input_scale` must be one of"):
            compare_two_groups(frame, "a", frame["g"], ["x", "y"], input_scale="log10")

    def test_an_unknown_adjustment_is_refused(self, frame):
        with pytest.raises(SaValueError, match="p_adjust"):
            compare_two_groups(frame, "a", frame["g"], ["x", "y"], p_adjust="bonferoni")

    def test_a_control_label_outside_group_lv_is_refused(self, frame):
        with pytest.raises(SaValueError, match="names a level `group_lv` does not hold"):
            compare_two_groups(frame, "a", frame["g"], ["x", "y"], control_label="z")
