"""Grading :mod:`statassist.transform` against R.

Two things are being checked. The arithmetic, which is one constant per feature
and is graded against the frozen columns; and the contract, which is that the
frame comes back the shape it went in - same rows, same order, same non-feature
columns - because the whole point is handing the result straight to a comparison
with the arguments unchanged.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from golden import assert_close, load_case, load_expected

from statassist.core.errors import SaValueError, SaWarning
from statassist.core.validate import UNSET
from statassist.transform import center_by_control, control_baseline
from statassist.transform._foldchange import fc_center, fold_change, resolve_fc_mean

FEATS = ["prot_1", "prot_2"]
LEVELS = ["ctrl", "case"]


@pytest.fixture(scope="module")
def raw() -> tuple[pd.DataFrame, dict]:
    return load_case("center_by_control")


@pytest.fixture(scope="module")
def logged() -> tuple[pd.DataFrame, dict]:
    return load_case("center_by_control_log2")


class TestResolveFcMean:
    def test_matches_r_on_all_four_combinations(self):
        expected = load_expected("foldchange_center")["resolved"]
        assert resolve_fc_mean(UNSET, "raw") == expected["default_raw"]
        assert resolve_fc_mean(UNSET, "log2") == expected["default_log2"]
        assert resolve_fc_mean("arith", "log2") == expected["explicit_arith"]
        assert resolve_fc_mean("geom", "raw") == expected["explicit_geom"]

    def test_the_default_is_scale_dependent_only_when_nothing_was_said(self):
        """Which is why the sentinel is needed: a formal default of ``"arith"``
        could not tell an explicit ``"arith"`` on the log2 scale apart from
        silence."""
        assert resolve_fc_mean(UNSET, "log2") == "geom"
        assert resolve_fc_mean("arith", "log2") == "arith"

    def test_an_unknown_centre_is_refused(self):
        with pytest.raises(SaValueError, match="fc_mean"):
            resolve_fc_mean("median", "raw")


class TestFcCenter:
    @pytest.mark.parametrize(
        ("case", "mean_type", "scale"),
        [
            ("arith_raw", "arith", "raw"),
            ("geom_raw", "geom", "raw"),
            ("arith_log2", "arith", "log2"),
            ("geom_log2", "geom", "log2"),
        ],
    )
    def test_matches_r(self, raw, case, mean_type, scale):
        frame, _ = raw
        expected = load_expected("foldchange_center")
        values = frame["prot_1"]
        if scale == "log2":
            values = np.round(np.log2(values), 6)
        assert_close(fc_center(values, "case", mean_type, scale), expected[case], path=case)

    def test_a_log2_centre_is_reported_on_the_raw_scale(self):
        """Which is the only scale on which a ratio is a ratio.

        The centre of ``log2(v)`` is not the log2 of the centre of ``v``, so
        undoing the transformation first is what makes
        ``fold_change == x_center / y_center`` hold.
        """
        values = np.array([2.0, 4.0, 8.0])
        got = fc_center(np.log2(values), "case", "geom", "log2")
        assert got == pytest.approx(fc_center(values, "case", "geom", "raw"))

    def test_an_empty_sample_names_the_side_it_came_from(self):
        with pytest.raises(SaValueError, match="no usable observation left in the ctrl group"):
            fc_center([], "ctrl", "arith")

    def test_a_non_positive_value_is_refused_rather_than_dropped(self):
        """Dropping it would silently return the geometric mean of the positive
        subset, which is a different quantity."""
        with pytest.raises(SaValueError, match="2 value\\(s\\) at or below zero"):
            fc_center([-1.0, 0.0, 3.0], "case", "geom")

    def test_values_that_are_not_on_the_log2_scale_are_named_as_such(self):
        with pytest.raises(SaValueError, match="overflows to infinity"):
            fc_center([2000.0, 3.0], "case", "arith", "log2")


class TestFoldChange:
    @pytest.mark.parametrize(
        ("case", "mean_type", "scale"),
        [("arith", "arith", "raw"), ("geom", "geom", "raw"), ("geom_log2", "geom", "log2")],
    )
    def test_matches_r(self, raw, case, mean_type, scale):
        frame, _ = raw
        expected = load_expected("foldchange_table")[case]
        source = frame.copy()
        if scale == "log2":
            source[FEATS] = np.round(np.log2(source[FEATS]), 6)
        samples = {
            name: {
                "x": source.loc[source["group"] == "case", name].to_numpy(float),
                "y": source.loc[source["group"] == "ctrl", name].to_numpy(float),
            }
            for name in FEATS
        }
        got = fold_change(samples, FEATS, ["case", "ctrl"], mean_type, scale)
        assert list(got.columns) == list(expected)
        for column, values in expected.items():
            assert_close(list(got[column]), values, path=column)

    def test_the_table_carries_no_adjusted_p_value(self, raw):
        """It holds no p-value at all: an effect estimate is not a test."""
        frame, _ = raw
        samples = {
            name: {
                "x": frame.loc[frame["group"] == "case", name].to_numpy(float),
                "y": frame.loc[frame["group"] == "ctrl", name].to_numpy(float),
            }
            for name in FEATS
        }
        got = fold_change(samples, FEATS, ["case", "ctrl"], "arith")
        assert "pval_adj" not in got.columns

    def test_a_zero_denominator_gives_an_infinite_ratio_rather_than_a_failure(self):
        samples = {"a": {"x": np.array([2.0, 4.0]), "y": np.array([0.0, 0.0])}}
        got = fold_change(samples, ["a"], ["case", "ctrl"], "arith")
        assert math.isinf(float(got["fold_change"].iloc[0]))
        assert math.isinf(float(got["log2fc"].iloc[0]))

    def test_a_zero_numerator_clears_any_cutoff_downwards(self):
        samples = {"a": {"x": np.array([0.0, 0.0]), "y": np.array([2.0, 4.0])}}
        got = fold_change(samples, ["a"], ["case", "ctrl"], "arith")
        assert float(got["log2fc"].iloc[0]) == -math.inf

    def test_centres_of_opposite_sign_leave_log2fc_undefined(self):
        samples = {"a": {"x": np.array([-2.0, -4.0]), "y": np.array([2.0, 4.0])}}
        got = fold_change(samples, ["a"], ["case", "ctrl"], "arith")
        assert float(got["fold_change"].iloc[0]) < 0
        assert math.isnan(float(got["log2fc"].iloc[0]))


class TestControlBaseline:
    def test_a_raw_centre_is_the_quantity_to_divide_out(self):
        assert control_baseline([2.0, 4.0, 6.0], "ctrl", "arith", "raw") == 4.0

    def test_a_log2_centre_is_the_quantity_to_subtract(self):
        """The raw centre logged, not the mean of the logs.

        Both are 2 for this sample, which is the point of choosing it: the
        geometric mean of ``2^v`` logged back is exactly ``mean(v)``.
        """
        assert control_baseline([1.0, 2.0, 3.0], "ctrl", "geom", "log2") == pytest.approx(2.0)

    def test_missing_values_are_dropped_before_the_centre_is_taken(self):
        assert control_baseline([2.0, float("nan"), 6.0], "ctrl", "arith", "raw") == 4.0

    @pytest.mark.parametrize(
        ("values", "scale", "message"),
        [
            ([0.0, 0.0], "raw", "send every value to infinity"),
            ([-2.0, -4.0], "raw", "reverse the order of every value"),
            # `2^v` underflows to zero, so the raw centre is 0 and has no log2.
            ([-2000.0, -2000.0], "log2", "has no log2 to subtract"),
        ],
    )
    def test_a_centre_that_cannot_be_applied_says_what_it_would_have_done(
        self, values, scale, message
    ):
        with pytest.raises(SaValueError, match=message):
            control_baseline(values, "ctrl", "arith", scale)

    def test_the_raw_refusal_names_the_control_level(self):
        with pytest.raises(SaValueError, match="the ctrl centre is 0"):
            control_baseline([0.0, 0.0], "ctrl", "arith", "raw")


class TestCenterByControl:
    def test_matches_r_on_the_raw_scale(self, raw):
        frame, expected = raw
        got = center_by_control(frame, FEATS, frame["group"], LEVELS)
        for name in FEATS:
            assert_close(list(got[name]), expected["raw_arith"][name], path=name)

    def test_matches_r_with_a_geometric_baseline(self, raw):
        frame, expected = raw
        got = center_by_control(frame, FEATS, frame["group"], LEVELS, fc_mean="geom")
        for name in FEATS:
            assert_close(list(got[name]), expected["raw_geom"][name], path=name)

    def test_matches_r_when_the_baseline_is_pointed_at_another_level(self, raw):
        frame, expected = raw
        got = center_by_control(
            frame, FEATS, frame["group"], ["ctrl", "case", "other"], control_label="case"
        )
        for name in FEATS:
            assert_close(list(got[name]), expected["other_control"][name], path=name)

    def test_matches_r_when_a_level_is_left_out_of_group_lv(self, raw):
        frame, expected = raw
        got = center_by_control(frame, "prot_2", frame["group"], ["other", "ctrl"])
        assert_close(list(got["prot_2"]), expected["one_feature"]["prot_2"], path="prot_2")

    def test_matches_r_on_the_log2_scale_with_the_default_centre(self, logged):
        frame, expected = logged
        got = center_by_control(frame, FEATS, frame["group"], LEVELS, input_scale="log2")
        for name in FEATS:
            assert_close(list(got[name]), expected["default_geom"][name], path=name)

    def test_matches_r_on_the_log2_scale_with_an_explicit_arithmetic_centre(self, logged):
        frame, expected = logged
        got = center_by_control(
            frame, FEATS, frame["group"], LEVELS, fc_mean="arith", input_scale="log2"
        )
        for name in FEATS:
            assert_close(list(got[name]), expected["explicit_arith"][name], path=name)


class TestContract:
    def test_no_row_is_dropped_even_when_a_level_is_left_out(self, raw):
        """The opposite of what a comparison does, and deliberately so.

        The result has to stay the same length as the ``group`` vector the caller
        still has to hand to the comparison, which drops those rows itself.
        """
        frame, _ = raw
        got = center_by_control(frame, FEATS, frame["group"], ["other", "ctrl"])
        assert len(got.index) == len(frame.index)
        assert list(got.index) == list(frame.index)
        assert set(got["group"]) == set(frame["group"])

    def test_a_column_that_is_not_a_feature_comes_back_untouched(self, raw):
        frame, _ = raw
        got = center_by_control(frame, "prot_1", frame["group"], LEVELS)
        assert list(got.columns) == list(frame.columns)
        pd.testing.assert_series_equal(got["prot_2"], frame["prot_2"])
        pd.testing.assert_series_equal(got["group"], frame["group"])

    def test_the_control_level_lands_on_one(self, raw):
        frame, _ = raw
        got = center_by_control(frame, FEATS, frame["group"], LEVELS)
        control = frame["group"] == "ctrl"
        for name in FEATS:
            assert float(got.loc[control, name].mean()) == pytest.approx(1.0)

    def test_the_control_level_lands_on_zero_on_the_log2_scale(self, logged):
        """With the geometric centre, which is the default there.

        It is the only choice that reduces to a difference of means: the
        geometric mean of ``2^v`` logged back is exactly ``mean(v)``, so
        subtracting it leaves the control at zero. The arithmetic centre lands
        near zero rather than at it, which is Jensen's inequality and not a bug.
        """
        frame, _ = logged
        got = center_by_control(frame, FEATS, frame["group"], LEVELS, input_scale="log2")
        control = frame["group"] == "ctrl"
        for name in FEATS:
            assert float(got.loc[control, name].mean()) == pytest.approx(0.0, abs=1e-12)

        arithmetic = center_by_control(
            frame, FEATS, frame["group"], LEVELS, fc_mean="arith", input_scale="log2"
        )
        for name in FEATS:
            assert float(arithmetic.loc[control, name].mean()) < 0

    def test_the_ratio_between_two_groups_survives_the_centring(self, raw):
        """Both centres are divided by the same baseline, so a comparison run
        afterwards reports the fold changes it reported before."""
        frame, _ = raw
        got = center_by_control(frame, FEATS, frame["group"], LEVELS)
        for name in FEATS:
            before = (
                frame.loc[frame["group"] == "case", name].mean()
                / frame.loc[frame["group"] == "ctrl", name].mean()
            )
            after = (
                got.loc[got.index[frame["group"] == "case"], name].mean()
                / got.loc[got.index[frame["group"] == "ctrl"], name].mean()
            )
            assert after == pytest.approx(before)

    def test_a_matrix_comes_back_as_a_frame(self):
        values = np.array([[2.0, 4.0], [4.0, 8.0], [6.0, 12.0], [8.0, 16.0]])
        got = center_by_control(
            pd.DataFrame(values, columns=["a", "b"]),
            ["a", "b"],
            ["x", "x", "y", "y"],
            ["x", "y"],
        )
        assert isinstance(got, pd.DataFrame)
        assert len(got.index) == 4


class TestFailures:
    def test_a_feature_whose_baseline_cannot_be_taken_comes_back_all_missing(self):
        frame = pd.DataFrame(
            {
                "fine": [2.0, 4.0, 6.0, 8.0],
                "zeroed": [0.0, 0.0, 5.0, 7.0],
                "g": ["x", "x", "y", "y"],
            }
        )
        with pytest.warns(SaWarning, match="could not be taken for 1 of 2 feature"):
            got = center_by_control(frame, ["fine", "zeroed"], frame["g"], ["x", "y"])
        assert got["zeroed"].isna().all()
        assert got["fine"].notna().all()

    def test_one_warning_names_every_failed_feature_rather_than_one_each(self):
        """A scan over hundreds of columns must not be abandoned because one of
        them has no usable control group."""
        frame = pd.DataFrame(
            {
                "a": [0.0, 0.0, 5.0, 7.0],
                "b": [0.0, 0.0, 3.0, 9.0],
                "g": ["x", "x", "y", "y"],
            }
        )
        with pytest.warns(SaWarning) as caught:
            center_by_control(frame, ["a", "b"], frame["g"], ["x", "y"])
        assert len(caught) == 1
        text = str(caught[0].message)
        assert "2 of 2 feature(s)" in text
        assert "  a: " in text and "  b: " in text

    def test_an_unknown_scale_is_refused(self, raw):
        frame, _ = raw
        with pytest.raises(SaValueError, match="input_scale"):
            center_by_control(frame, FEATS, frame["group"], LEVELS, input_scale="ln")

    def test_a_control_label_naming_no_level_is_refused(self, raw):
        frame, _ = raw
        with pytest.raises(SaValueError, match="control_label"):
            center_by_control(frame, FEATS, frame["group"], LEVELS, control_label="placebo")


class TestOnSimulatedData:
    def test_a_simulated_experiment_centres_on_its_own_control(self):
        """Phase 1 generates log2 data and says so, which is the scale-dependent
        default this function shares with the comparisons."""
        from statassist import simulate_two_groups

        sim = simulate_two_groups(n_feats=4, n_case=12, n_control=12, n_up=1, n_down=1, seed=13)
        args = sim.args
        got = center_by_control(
            args["data"],
            args["feats"],
            args["group"],
            args["group_lv"],
            input_scale=args["input_scale"],
        )
        control = np.asarray(args["group"]) == args["group_lv"][0]
        for name in args["feats"]:
            assert float(got.loc[control, name].mean()) == pytest.approx(0.0, abs=1e-12)
