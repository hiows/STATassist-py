"""Grading :mod:`statassist.diagnose.distribution` against R.

The tables are made of kernel results graded elsewhere, so what is being checked
here is the assembly: which rows exist, in which order, what happens to a level
that cannot be tested, and where the two summary flags come from.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from golden import as_list, assert_frame_close, load_case, zero_based

from statassist.core.errors import SaValueError
from statassist.core.result import SaDiagnosis
from statassist.diagnose import diagnose_distribution, diagnose_samples

FEATS = ["gene_1", "gene_2"]


@pytest.fixture(scope="module")
def ungrouped() -> tuple[pd.DataFrame, dict]:
    return load_case("diagnose_ungrouped")


@pytest.fixture(scope="module")
def grouped() -> tuple[pd.DataFrame, dict]:
    return load_case("diagnose_grouped")


def assert_diagnosis_close(got: SaDiagnosis, expected: dict) -> None:
    """Every slot of one frozen diagnosis, in the order R lists them."""
    assert list(got) == [
        "analysis",
        "features",
        "design",
        "parameters",
        "normality",
        "variance",
        "outliers",
        "summary",
        "metadata",
    ]
    assert got["analysis"] == expected["analysis"]
    assert got["features"] == as_list(expected["features"])

    design = expected["design"]
    assert got["design"]["grouped"] is design["grouped"]
    wanted = design["group_lv"]
    assert got["design"]["group_lv"] == (None if wanted is None else as_list(wanted))

    for name, value in expected["parameters"].items():
        assert got["parameters"][name] == value

    assert_frame_close(got["normality"], expected["normality"], path="normality")
    assert_frame_close(got["variance"], expected["variance"], path="variance")

    outliers = dict(expected["outliers"])
    outliers["row"] = zero_based(outliers["row"])
    assert_frame_close(got["outliers"], outliers, path="outliers")

    assert_frame_close(got["summary"], expected["summary"], path="summary")
    assert len(got["variance"].index) == expected["n_variance_rows"]


class TestDiagnoseDistributionUngrouped:
    def test_matches_r(self, ungrouped):
        frame, expected = ungrouped
        got = diagnose_distribution(frame, ["gene_1", "gene_2", "gene_3"])
        assert_diagnosis_close(got, expected["plain"])

    def test_a_different_alpha_only_moves_the_flags(self, ungrouped):
        frame, expected = ungrouped
        got = diagnose_distribution(frame, "gene_1", alpha=0.1)
        assert_diagnosis_close(got, expected["one"])

    def test_without_a_grouping_there_is_nothing_to_compare_variances_across(self, ungrouped):
        """So the table is empty rather than a row of missing values.

        The columns are still there, which is what lets a caller read
        ``variance["levene_pval"]`` without asking first whether a grouping was
        supplied.
        """
        frame, expected = ungrouped
        got = diagnose_distribution(frame, FEATS)
        assert len(got["variance"].index) == 0
        assert list(got["variance"].columns) == list(expected["plain"]["variance"])
        assert got["summary"]["variance_ok"].isna().all()


class TestDiagnoseDistributionGrouped:
    def test_matches_r(self, grouped):
        frame, expected = grouped
        got = diagnose_distribution(frame, FEATS, frame["group"])
        assert_diagnosis_close(got, expected["plain"])

    def test_matches_r_with_the_levene_centre_moved_to_the_mean(self, grouped):
        frame, expected = grouped
        got = diagnose_distribution(frame, "gene_1", frame["group"], center="mean", alpha=0.1)
        assert_diagnosis_close(got, expected["mean_centre"])

    def test_matches_r_with_a_trimmed_levene_centre(self, grouped):
        frame, expected = grouped
        got = diagnose_distribution(frame, "gene_1", frame["group"], center="trimmed", trim=0.25)
        assert_diagnosis_close(got, expected["trimmed"])

    def test_matches_r_under_the_robust_z_outlier_rule(self, grouped):
        frame, expected = grouped
        got = diagnose_distribution(
            frame, "gene_1", frame["group"], criterion="robust_z", z_threshold=2
        )
        assert_diagnosis_close(got, expected["robust_z"])

    def test_the_normality_table_has_one_row_per_feature_and_level(self, grouped):
        frame, _ = grouped
        levels = list(frame["group"].unique())
        got = diagnose_distribution(frame, FEATS, frame["group"])
        assert len(got["normality"].index) == len(FEATS) * len(levels)
        # The feature is the slow axis and the level the fast one, so a feature's
        # rows sit together.
        assert list(got["normality"]["features"]) == [name for name in FEATS for _ in levels]

    def test_the_worst_level_decides_whether_a_feature_is_normal(self, grouped):
        frame, _ = grouped
        got = diagnose_distribution(frame, FEATS, frame["group"])
        for name in FEATS:
            of_feature = got["normality"]["shapiro_pval"][got["normality"]["features"] == name]
            row = got["summary"][got["summary"]["features"] == name]
            assert float(row["min_shapiro_pval"].iloc[0]) == pytest.approx(float(of_feature.min()))


class TestFlags:
    """``normal_ok`` and ``variance_ok`` are ``p > alpha``, strictly."""

    def test_a_p_value_exactly_at_alpha_is_a_failure(self):
        frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
        observed = float(diagnose_distribution(frame, "a")["summary"]["min_shapiro_pval"].iloc[0])
        at_alpha = diagnose_distribution(frame, "a", alpha=observed)
        assert bool(at_alpha["summary"]["normal_ok"].iloc[0]) is False

        just_under = diagnose_distribution(frame, "a", alpha=observed * (1 - 1e-9))
        assert bool(just_under["summary"]["normal_ok"].iloc[0]) is True

    def test_a_check_that_could_not_be_run_is_missing_rather_than_passing(self):
        """A constant column has no Shapiro-Wilk statistic, so it has no verdict.

        Calling it normal would be the wrong answer to a question that was never
        asked.
        """
        frame = pd.DataFrame({"flat": [4.0] * 10, "fine": list(range(10))})
        got = diagnose_distribution(frame, ["flat", "fine"])
        flat = got["summary"][got["summary"]["features"] == "flat"]
        assert pd.isna(flat["min_shapiro_pval"].iloc[0])
        assert pd.isna(flat["normal_ok"].iloc[0])
        assert pd.isna(got["normality"]["shapiro_stat"].iloc[0])

    def test_a_level_too_small_to_test_still_gets_its_row(self):
        """Its absence would be indistinguishable from the level not existing."""
        frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], "g": ["x"] * 5 + ["y"] * 2})
        got = diagnose_distribution(frame, "a", frame["g"])
        assert list(got["normality"]["group"]) == ["x", "y"]
        tiny = got["normality"][got["normality"]["group"] == "y"]
        assert pd.isna(tiny["shapiro_pval"].iloc[0])
        assert int(tiny["n_used"].iloc[0]) == 2


class TestRepr:
    def test_it_reports_the_counts_rather_than_the_tables(self, grouped):
        frame, _ = grouped
        text = repr(diagnose_distribution(frame, FEATS, frame["group"]))
        assert "<SaDiagnosis> distribution_diagnosis" in text
        assert "features : 2" in text
        assert "outlier criterion = iqr" in text
        assert "feature(s) fail Levene at 0.05" in text
        assert "observation(s) flagged across" in text

    def test_an_ungrouped_diagnosis_says_so_and_leaves_out_the_variance_line(self, ungrouped):
        frame, _ = ungrouped
        text = repr(diagnose_distribution(frame, FEATS))
        assert "none, so no variance test" in text
        assert "Levene" not in text


class TestRefusals:
    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"criterion": "eyeball"}, "criterion"),
            ({"center": "mode"}, "center"),
            ({"alpha": 0}, "alpha"),
            ({"trim": 0.5}, "trim"),
        ],
    )
    def test_a_setting_out_of_range_is_refused_by_name(self, ungrouped, kwargs, message):
        frame, _ = ungrouped
        with pytest.raises(SaValueError, match=message):
            diagnose_distribution(frame, "gene_1", **kwargs)


class TestDiagnoseSamples:
    """What a comparison scenario will call, with the samples it actually tested."""

    def test_independent_samples_get_both_tables(self, grouped):
        frame, expected = grouped
        levels = list(frame["group"].unique())
        per_feature = {
            name: {
                level: frame.loc[frame["group"] == level, name].dropna().to_numpy(float)
                for level in levels
            }
            for name in FEATS
        }
        got = diagnose_samples(per_feature, FEATS, levels, paired=False)
        assert list(got) == ["normality", "variance", "summary"]
        assert len(got["variance"].index) == len(FEATS)
        assert_frame_close(got["normality"], expected["plain"]["normality"], path="normality")

    def test_a_paired_design_gets_no_variance_table(self):
        """Homogeneity across independent groups is not the assumption a
        within-subject test makes. Sphericity is, and the repeated measures row
        carries it."""
        rng = np.random.default_rng(3)
        matrix = rng.normal(size=(12, 3))
        got = diagnose_samples({"a": matrix}, ["a"], ["t1", "t2", "t3"], paired=True)
        assert len(got["variance"].index) == 0
        assert list(got["normality"]["group"]) == ["t1", "t2", "t3"]
        assert got["summary"]["variance_ok"].isna().all()

    def test_a_paired_matrix_is_read_column_by_column(self):
        matrix = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [5.0, 55.0]])
        got = diagnose_samples({"a": matrix}, ["a"], ["before", "after"], paired=True)
        assert list(got["normality"]["n_used"]) == [4, 4]
        assert list(got["normality"]["group"]) == ["before", "after"]

    def test_the_summary_carries_no_outliers_because_none_were_screened(self):
        rng = np.random.default_rng(5)
        per_feature = {"a": {"x": rng.normal(size=10), "y": rng.normal(size=10)}}
        got = diagnose_samples(per_feature, ["a"], ["x", "y"], paired=False)
        assert list(got["summary"]["n_outliers"]) == [0]


class TestOnSimulatedData:
    def test_a_simulated_experiment_diagnoses_as_it_comes(self):
        """Phase 1's ``args`` unpack straight into this."""
        from statassist import simulate_two_groups

        sim = simulate_two_groups(n_feats=5, n_case=15, n_control=15, n_up=2, n_down=2, seed=11)
        args = sim.args
        got = diagnose_distribution(args["data"], args["feats"], args["group"], args["group_lv"])
        assert got["features"] == as_list(args["feats"])
        assert got["design"]["group_lv"] == as_list(args["group_lv"])
        assert len(got["normality"].index) == 5 * 2
        assert len(got["variance"].index) == 5
        # Normal data, drawn as normal: most features should pass.
        assert int(got["summary"]["normal_ok"].sum()) >= 4
