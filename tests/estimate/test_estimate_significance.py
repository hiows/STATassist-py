"""``estimate_significance`` against the verdict R read out of the same object.

The attributes are graded with the numbers. They are how the cutoffs travel to
``draw_volcano_plot()``, so a port that flags the right features and loses the
attributes has not finished the job.
"""

from __future__ import annotations

import functools
import math

import pandas as pd
import pytest
from golden import assert_close, assert_frame_close, load_case

from statassist import (
    compare_multiple_groups,
    compare_one_sample,
    compare_two_groups,
    estimate_significance,
    simulate_multiple_groups,
    simulate_two_groups,
)
from statassist.core.errors import SaValueError, SaWarning
from statassist.core.result import SIGNIFICANCE_COLUMNS

FC_FEATS = ["prot_1", "prot_2"]
EDGE_FEATS = ["moved", "flipped", "zeroed"]
ONE_FEATS = ["conc", "level", "flag"]
GENE_FEATS = ["gene_1", "gene_2", "gene_3"]
GENE_LV = ["ctrl", "treat_a", "treat_b"]


def _assert_table(produced: pd.DataFrame, expected, path: str) -> None:
    """One verdict table and the attributes it describes itself with."""
    assert_frame_close(produced, expected["table"], path=f"{path}[table]")
    for key, value in expected["attrs"].items():
        # R writes an absent attribute as a stated absence, which is the same
        # answer as a key the port never set.
        assert_close(produced.attrs.get(key), value, path=f"{path}[attrs][{key}]")


def _assert_case(produced, expected, path: str) -> None:
    assert produced["analysis_type"] == expected["analysis_type"]
    if "significance" in expected:
        _assert_table(produced["significance"], expected["significance"], path)
        return
    held = produced["significance"]
    assert list(held) == list(expected["by_contrast"]), f"{path}: contrast order differs."
    for contrast, table in expected["by_contrast"].items():
        _assert_table(held[contrast], table, f"{path}[{contrast}]")


@functools.lru_cache(maxsize=1)
def _two_group():
    frame, expected = load_case("estimate_two_group")
    produced = compare_two_groups(frame, FC_FEATS, frame["group"], ["ctrl", "case"], diagnose=False)
    return produced, expected


@functools.lru_cache(maxsize=1)
def _undecided():
    frame, expected = load_case("estimate_undecided")
    produced = compare_two_groups(
        frame, EDGE_FEATS, frame["group"], ["ctrl", "case"], diagnose=False
    )
    return produced, expected


@functools.lru_cache(maxsize=1)
def _one_sample():
    frame, expected = load_case("estimate_one_sample")
    with pytest.warns(SaWarning):
        produced = compare_one_sample(frame, ONE_FEATS, mu=5, p=0.5)
    return produced, expected


@functools.lru_cache(maxsize=1)
def _multi_group():
    frame, expected = load_case("estimate_multi_group")
    produced = compare_multiple_groups(
        frame, GENE_FEATS, frame["group"], GENE_LV, posthoc_alpha=1, diagnose=False
    )
    return produced, expected


@functools.lru_cache(maxsize=1)
def _planted_two_group():
    sim = simulate_two_groups(n_feats=8, n_case=15, n_control=15, n_up=2, n_down=2, seed=9)
    return sim, compare_two_groups(**sim.args, diagnose=False)


class TestTwoGroup:
    @pytest.fixture
    def comparison(self):
        return _two_group()

    @pytest.mark.parametrize(
        ("key", "kwargs"),
        [
            ("default", {}),
            ("loose", {"log2fc_cutoff": 0.05}),
            ("wilcox", {"test": "wilcox_test", "log2fc_cutoff": 0.05}),
            ("robust", {"test": "robust_test", "log2fc_cutoff": 0.05}),
            ("bonferroni", {"adj_type": "bonferroni", "log2fc_cutoff": 0.05}),
            ("raw", {"adj_type": "none", "log2fc_cutoff": 0.05, "pval_cutoff": 0.2}),
        ],
    )
    def test_matches_r(self, comparison, key, kwargs):
        produced, expected = comparison
        _assert_case(estimate_significance(produced, **kwargs), expected[key], key)

    def test_naming_a_method_replaces_the_adjustment_rather_than_compounding_it(self, comparison):
        produced, _ = comparison
        verdict = estimate_significance(produced, adj_type="bonferroni")
        raw = produced["tests"]["t_test"]["pval"]
        # Bonferroni over two features doubles the raw p-value, which is not what
        # doubling the comparison's Benjamini-Hochberg column would give.
        assert_close(
            list(verdict.significance["adj_pvalue"]),
            [min(1.0, 2 * value) for value in raw],
        )

    def test_the_default_test_is_the_first_the_scenario_ran(self, comparison):
        produced, _ = comparison
        assert estimate_significance(produced).significance.attrs["test"] == "t_test"


class TestUndecided:
    @pytest.fixture
    def comparison(self):
        return _undecided()

    @pytest.mark.parametrize(("key", "cutoff"), [("plain", 0.05), ("strict", 8)])
    def test_matches_r(self, comparison, key, cutoff):
        produced, expected = comparison
        _assert_case(estimate_significance(produced, log2fc_cutoff=cutoff), expected[key], key)

    def test_the_two_ways_out_of_the_log_domain_are_not_the_same_answer(self, comparison):
        """A ratio of zero is an infinite decrease; opposite signs is no ratio."""
        produced, _ = comparison
        verdict = estimate_significance(produced, log2fc_cutoff=0.05).significance
        flags = verdict.set_index("features")["is_signif"]
        assert math.isnan(verdict.set_index("features").loc["flipped", "log2fc"])
        assert flags["flipped"] is pd.NA
        assert verdict.set_index("features").loc["zeroed", "log2fc"] == math.inf
        assert bool(flags["zeroed"]) is True

    def test_the_magnitude_rule_alone_can_decide_against_a_feature(self, comparison):
        """Which is R's ``NA & FALSE`` being ``FALSE`` rather than ``NA``."""
        produced, _ = comparison
        # A p-value cutoff nothing clears, so the undecided magnitude meets a
        # FALSE rather than a TRUE and the verdict is decided after all.
        verdict = estimate_significance(
            produced, log2fc_cutoff=0.05, pval_cutoff=1e-12
        ).significance
        assert bool(verdict.set_index("features")["is_signif"]["flipped"]) is False


class TestOneSample:
    @pytest.fixture
    def comparison(self):
        return _one_sample()

    @pytest.mark.parametrize(
        ("key", "kwargs"),
        [("plain", {}), ("prop", {"test": "prop_test"})],
    )
    def test_matches_r(self, comparison, key, kwargs):
        produced, expected = comparison
        _assert_case(
            estimate_significance(produced, log2fc_cutoff=0.05, **kwargs),
            expected[key],
            key,
        )

    def test_a_comparison_with_no_group_levels_says_so(self, comparison):
        produced, _ = comparison
        assert estimate_significance(produced).significance.attrs["group_lv"] is None


class TestMultiGroup:
    @pytest.fixture
    def comparison(self):
        return _multi_group()

    @pytest.mark.parametrize(
        ("key", "kwargs"),
        [
            ("omnibus", {}),
            ("kruskal", {"test": "kruskal_test"}),
            ("by_contrast", {"by": "contrast"}),
            ("by_contrast_bh", {"by": "contrast", "adj_type": "BH"}),
        ],
    )
    def test_matches_r(self, comparison, key, kwargs):
        produced, expected = comparison
        _assert_case(
            estimate_significance(produced, log2fc_cutoff=0.05, **kwargs),
            expected[key],
            key,
        )

    def test_the_omnibus_reading_names_the_level_the_ratio_came_from(self, comparison):
        produced, _ = comparison
        verdict = estimate_significance(produced).significance
        assert list(verdict.columns) == [*SIGNIFICANCE_COLUMNS, "extreme_level"]
        assert set(verdict["extreme_level"]) <= set(GENE_LV[1:])

    def test_the_two_readings_agree_on_which_way_a_feature_moved(self, comparison):
        """Both divide by the reference, so neither can point the other way."""
        produced, _ = comparison
        omnibus = estimate_significance(produced).significance.set_index("features")
        by_pair = estimate_significance(produced, by="contrast").significance
        for name, row in omnibus.iterrows():
            contrast = f"{row['extreme_level']} - {GENE_LV[0]}"
            pair = by_pair[contrast].set_index("features")
            assert_close(pair.loc[name, "log2fc"], row["log2fc"])

    def test_a_feature_never_compared_pairwise_is_undecided_and_keeps_its_ratio(self):
        frame, expected = load_case("estimate_not_asked")
        produced = compare_multiple_groups(
            frame,
            GENE_FEATS,
            frame["group"],
            GENE_LV,
            posthoc_alpha=0.001,
            diagnose=False,
        )
        verdict = estimate_significance(produced, by="contrast", log2fc_cutoff=0.05)
        _assert_case(verdict, expected["by_contrast"], "not_asked")
        for table in verdict.significance.values():
            never = table[table["pvalue"].isna()]
            assert len(never.index) > 0
            assert never["log2fc"].notna().all()


class TestArgumentChecks:
    @pytest.fixture
    def two_group(self):
        return _planted_two_group()[1]

    def test_an_unknown_reading_is_refused(self, two_group):
        with pytest.raises(SaValueError, match="`by` must be one of"):
            estimate_significance(two_group, by="feature")

    def test_the_term_reading_is_not_part_of_this_port(self, two_group):
        with pytest.raises(SaValueError, match="needs a term axis"):
            estimate_significance(two_group, by="term")

    def test_the_contrast_reading_needs_a_pairwise_stage(self, two_group):
        with pytest.raises(SaValueError, match="needs a pairwise stage"):
            estimate_significance(two_group, by="contrast")

    def test_a_comparison_without_a_posthoc_stage_is_named_in_the_message(self):
        sim = simulate_multiple_groups(
            n_feats=2, n_control=8, n_treat=(8, 8), n_up=1, n_down=1, seed=6
        )
        res = compare_multiple_groups(**sim.args, posthoc=False, diagnose=False)
        with pytest.raises(SaValueError, match="needs a pairwise stage"):
            estimate_significance(res, by="contrast")

    def test_an_unknown_test_is_refused_by_name(self, two_group):
        with pytest.raises(SaValueError, match="`test` must name one of the tests"):
            estimate_significance(two_group, test="anova_test")

    def test_a_negative_cutoff_is_refused(self, two_group):
        with pytest.raises(SaValueError, match="log2fc_cutoff"):
            estimate_significance(two_group, log2fc_cutoff=-1)

    def test_a_pval_cutoff_at_zero_is_refused(self, two_group):
        with pytest.raises(SaValueError, match="pval_cutoff"):
            estimate_significance(two_group, pval_cutoff=0)

    def test_an_unknown_adjustment_is_refused(self, two_group):
        with pytest.raises(SaValueError, match="adj_type"):
            estimate_significance(two_group, adj_type="bonferoni")

    def test_something_that_is_not_a_comparison_is_refused(self):
        with pytest.raises(SaValueError, match="must be a comparison result"):
            estimate_significance(pd.DataFrame({"features": ["a"]}))


class TestContract:
    @pytest.fixture
    def planted(self):
        return _planted_two_group()

    def test_the_verdict_keeps_the_scenario_it_was_read_from(self, planted):
        _, res = planted
        verdict = estimate_significance(res)
        assert verdict["analysis_type"] == "two_group_comparison"
        assert list(verdict) == ["analysis_type", "significance"]

    def test_the_planted_features_are_the_ones_flagged(self, planted):
        sim, res = planted
        verdict = estimate_significance(res, log2fc_cutoff=0.5).significance
        flagged = set(verdict.loc[verdict["is_signif"] == True, "features"])  # noqa: E712
        planted_up = set(sim.truth.loc[sim.truth["direction"] != "none", "features"])
        assert flagged <= planted_up

    def test_repr_summarises_the_rule_rather_than_printing_the_table(self, planted):
        _, res = planted
        text = repr(estimate_significance(res))
        assert "two_group_comparison" in text
        assert "t_test" in text
        assert "abs(log2fc) >= 1" in text
        assert "significant" in text

    def test_repr_of_a_contrast_reading_lists_one_line_per_contrast(self):
        sim = simulate_multiple_groups(
            n_feats=3, n_control=10, n_treat=(10, 10), n_up=1, n_down=1, seed=4
        )
        res = compare_multiple_groups(**sim.args, posthoc_alpha=1, diagnose=False)
        text = repr(estimate_significance(res, by="contrast"))
        assert "one table per contrast" in text
        for contrast in res["pairwise"]["anova_test"]:
            assert contrast in text
