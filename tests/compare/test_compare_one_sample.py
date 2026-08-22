"""``compare_one_sample`` against the object R's version assembled."""

from __future__ import annotations

import contextlib
import math
import warnings

import pandas as pd
import pytest
from golden import assert_close, assert_frame_close, load_case

from statassist import compare_one_sample
from statassist.compare.one_sample import one_sample_prop
from statassist.core.contracts import test_table_columns as contract_columns
from statassist.core.errors import SaValueError, SaWarning

FEATS = ["conc", "level", "flag"]


@contextlib.contextmanager
def _quiet():
    """Run without the "not a binary feature" warning.

    Which features the proportion test refuses is graded on its own; here it
    would only be noise on every case that holds a continuous feature.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SaWarning)
        yield


def _assert_case(produced, expected, path: str) -> None:
    design = produced["design"]
    assert_close(design["mu"], expected["mu"], path=f"{path}[mu]")
    assert_close(design["p"], expected["p"], path=f"{path}[p]")
    assert_close(design["success"], expected["success"], path=f"{path}[success]")
    assert produced["parameters"]["fc_mean"] == expected["fc_mean"]

    assert_frame_close(produced["effect"], expected["effect"], path=f"{path}[effect]")
    for name in ("t_test", "wilcox_test", "prop_test"):
        assert_frame_close(produced["tests"][name], expected[name], path=f"{path}[{name}]")


class TestAgainstR:
    @pytest.mark.parametrize(
        ("key", "feats", "kwargs"),
        [
            ("plain", FEATS, {"mu": 5, "p": 0.5}),
            ("greater", FEATS, {"mu": 5, "p": 0.6, "alternative": "greater"}),
            ("less_90", FEATS, {"mu": 5, "p": 0.4, "alternative": "less", "conf_level": 0.90}),
            ("geom", FEATS, {"mu": 5, "fc_mean": "geom"}),
            ("mu_zero", FEATS, {"p": 0.5, "p_adjust": "holm"}),
            ("success_zero", ["flag"], {"mu": 0.5, "p": 0.5, "success": 0}),
        ],
    )
    def test_matches_r(self, key, feats, kwargs):
        frame, expected = load_case("one_sample")
        with _quiet():
            produced = compare_one_sample(frame, feats, diagnose=False, **kwargs)
        _assert_case(produced, expected[key], key)

    @pytest.mark.parametrize(
        ("key", "kwargs"),
        [
            ("default_geom", {"mu": 2}),
            ("mu_zero", {"mu": 0}),
            ("explicit_arith", {"mu": 2, "fc_mean": "arith"}),
        ],
    )
    def test_matches_r_on_log2_input(self, key, kwargs):
        frame, expected = load_case("one_sample_log2")
        with _quiet():
            produced = compare_one_sample(
                frame, ["level"], input_scale="log2", diagnose=False, **kwargs
            )
        _assert_case(produced, expected[key], key)


class TestProportion:
    def test_matches_r_on_every_shape_of_count(self):
        params, expected = load_case("one_sample_prop")
        produced = []
        for row in params.itertuples():
            values = [1.0] * int(row.n_success) + [0.0] * int(row.n - row.n_success)
            produced.append(one_sample_prop(values, row.p, 1.0, row.alternative, row.conf_level))
        for column, wanted in expected.items():
            assert_close([row[column] for row in produced], wanted, path=column)

    def test_the_interval_stays_inside_the_unit_interval(self):
        """Which is what a Wilson interval buys over a Wald one."""
        for n_success in (1, 5, 9, 10):
            values = [1.0] * n_success + [0.0] * (10 - n_success)
            row = one_sample_prop(values, 0.5, 1.0, "two.sided", 0.95)
            assert 0.0 <= row["lower_conf"] <= row["upper_conf"] <= 1.0

    def test_a_one_sided_interval_opens_to_the_boundary_not_to_infinity(self):
        values = [1.0] * 7 + [0.0] * 3
        assert one_sample_prop(values, 0.5, 1.0, "greater", 0.95)["upper_conf"] == 1.0
        assert one_sample_prop(values, 0.5, 1.0, "less", 0.95)["lower_conf"] == 0.0

    def test_a_feature_that_is_not_binary_is_refused(self):
        with pytest.raises(SaValueError, match="needs a binary feature"):
            one_sample_prop([1.0, 2.0, 3.0], 0.5, 1.0, "two.sided", 0.95)

    def test_a_success_value_that_never_occurs_is_refused(self):
        with pytest.raises(SaValueError, match="does not occur in this feature"):
            one_sample_prop([0.0, 0.0, 0.0], 0.5, 1.0, "two.sided", 0.95)


class TestContract:
    @pytest.fixture
    def frame(self):
        return pd.DataFrame(
            {
                "a": [4.1, 5.2, 6.3, 3.8, 7.1, 5.9],
                "flag": [1.0, 0.0, 1.0, 1.0, 1.0, 0.0],
            }
        )

    def test_the_three_tests_are_the_one_sample_family(self, frame):
        res = compare_one_sample(frame, ["a", "flag"], mu=5, diagnose=False)
        assert list(res["tests"]) == ["t_test", "wilcox_test", "prop_test"]
        assert res["analysis"] == "one_sample_comparison"

    def test_the_design_carries_the_reference_instead_of_group_levels(self, frame):
        res = compare_one_sample(frame, "a", mu=5, p=0.4, success=1, diagnose=False)
        assert res["design"]["mu"] == 5
        assert res["design"]["p"] == 0.4
        assert "group_lv" not in res["design"]
        assert "posthoc" not in res
        assert "pairwise" not in res

    def test_every_table_carries_the_contract_columns_in_order(self, frame):
        res = compare_one_sample(frame, ["a", "flag"], mu=5, diagnose=False)
        assert list(res["effect"].columns) == [
            "features",
            "n_used",
            "center",
            "mu",
            "diff",
            "fold_change",
            "log2fc",
        ]
        for table in res["tests"].values():
            assert list(table["features"]) == res["features"]
            for name in contract_columns():
                assert name in table.columns

    def test_mu_at_zero_leaves_the_ratio_undecided_rather_than_infinite(self, frame):
        """An infinity would read as an infinitely large increase."""
        res = compare_one_sample(frame, "a", mu=0, diagnose=False)
        assert res["effect"]["fold_change"].isna().all()
        assert res["effect"]["log2fc"].isna().all()
        # The tests still ran: only the ratio has no answer.
        assert res["tests"]["t_test"]["pval"].notna().all()

    def test_on_the_log2_scale_mu_at_zero_is_a_reference_of_one(self, frame):
        res = compare_one_sample(frame, "a", mu=0, input_scale="log2", diagnose=False)
        assert res["effect"]["mu"].iloc[0] == 1.0
        assert res["effect"]["log2fc"].notna().all()

    def test_a_one_sided_alternative_leaves_the_untested_side_open(self, frame):
        with _quiet():
            res = compare_one_sample(
                frame, ["a", "flag"], mu=5, alternative="greater", diagnose=False
            )
        assert res["tests"]["t_test"]["upper_conf"].iloc[0] == math.inf
        assert res["tests"]["wilcox_test"]["upper_conf"].iloc[0] == math.inf
        # The proportion interval is on the probability scale, so "open" is 1.
        assert res["tests"]["prop_test"]["upper_conf"].iloc[1] == 1.0

    def test_diagnostics_name_the_single_sample_on_the_level_axis(self, frame):
        with _quiet():
            res = compare_one_sample(frame, "a", mu=5)
        assert list(res["diagnostics"]["normality"]["group"]) == ["sample"]
        # One level, so Levene and Bartlett have nothing to compare and the row
        # is missing rather than absent: the feature was still diagnosed.
        variance = res["diagnostics"]["variance"]
        assert list(variance["n_groups"]) == [1]
        assert variance["levene_pval"].isna().all()

    def test_repr_reports_mu_where_a_grouped_comparison_reports_levels(self, frame):
        with _quiet():
            text = repr(compare_one_sample(frame, "a", mu=5, diagnose=False))
        assert "one_sample_comparison" in text
        assert "mu       : 5" in text
        assert "$prop_test" in text

    def test_an_out_of_range_proportion_is_refused(self, frame):
        with pytest.raises(SaValueError, match="`p` must be in"):
            compare_one_sample(frame, "a", p=1.0)

    def test_a_missing_success_value_is_refused(self, frame):
        with pytest.raises(SaValueError, match="`success` must be a single non-missing number"):
            compare_one_sample(frame, "a", success=None)

    def test_an_unusable_feature_is_a_missing_row_not_an_abort(self, frame):
        with pytest.warns(SaWarning, match="binary feature"):
            res = compare_one_sample(frame, ["a", "flag"], mu=5, diagnose=False)
        prop = res["tests"]["prop_test"].set_index("features")
        assert prop.loc["a"].isna().all()
        assert prop.loc["flag"].notna().all()
