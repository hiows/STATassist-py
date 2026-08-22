"""``compare_multiple_groups`` against the object R's version assembled.

The pairwise stage is most of what is graded here. ``posthoc`` is the ragged
record of what was asked and ``pairwise`` is the rectangular view of the same
numbers, so both are compared: a port that rectangularises the wrong way round
produces two tables that are individually plausible and disagree with each other.
"""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd
import pytest
from golden import as_list, assert_close, assert_frame_close, load_case

from statassist import compare_multiple_groups, simulate_multiple_groups
from statassist.core.contracts import pairwise_table_columns, posthoc_table_columns
from statassist.core.contracts import test_table_columns as contract_columns
from statassist.core.errors import SaValueError, SaWarning

GENE_FEATS = ["gene_1", "gene_2", "gene_3"]
GENE_LV = ["ctrl", "treat_a", "treat_b"]
REP_FEATS = ["score", "shifted"]
REP_LV = ["t1", "t2", "t3"]

#: The effect table of a multi-group comparison, in order.
EFFECT_COLUMNS = [
    "features",
    "n_used",
    "n_groups",
    "ref_center",
    "extreme_level",
    "extreme_center",
    "fold_change",
    "log2fc",
]

#: The two procedures that read their p-value and interval off the studentised
#: range, and the tolerance those three columns are graded at.
#:
#: R integrates the distribution by the Copenhaver-Holland algorithm and
#: :class:`scipy.stats.studentized_range` by its own quadrature, which is the
#: difference ``tests/kernel/test_posthoc.py`` already documents. Every other
#: column of the same tables is still graded at the golden tolerance.
STUDENTISED = {"tukey_hsd", "games_howell"}
QTUKEY_RTOL = 1e-6
QTUKEY_PVAL = ("pval", "pval_adj")
QTUKEY_BOUNDS = ("lower_conf", "upper_conf")


def _assert_pairs_close(produced, expected, path: str, studentised: bool) -> None:
    """Grade one post-hoc or pairwise table, the quadrature columns apart."""
    if not studentised:
        assert_frame_close(produced, expected, path=path)
        return

    # Compared whole first, so the column order stays part of the contract.
    assert list(produced.columns) == list(expected), f"{path}: column order differs."
    touched = QTUKEY_PVAL + QTUKEY_BOUNDS
    tight = {name: values for name, values in expected.items() if name not in touched}
    assert_frame_close(produced[list(tight)], tight, path=path)

    loose = {name: expected[name] for name in QTUKEY_PVAL}
    assert_frame_close(produced[list(loose)], loose, rtol=QTUKEY_RTOL, path=path)

    # A bound is `estimate +/- q * stderr`, so the quadrature error lives in the
    # half-width and not in the bound. A bound that happens to sit near zero -
    # which is exactly what a contrast on the edge of significance produces -
    # would fail any relative comparison against itself, so the half-width is
    # what the tolerance is taken from.
    bounds = {name: as_list(expected[name]) for name in QTUKEY_BOUNDS}
    half = (
        np.abs(
            np.asarray(bounds["upper_conf"], dtype=float)
            - np.asarray(bounds["lower_conf"], dtype=float)
        )
        / 2
    )
    scale = float(np.nanmax(half)) if half.size and np.isfinite(half).any() else 0.0
    assert_frame_close(
        produced[list(bounds)],
        bounds,
        rtol=QTUKEY_RTOL,
        atol=QTUKEY_RTOL * scale,
        path=path,
    )


def _assert_case(produced, expected, path: str) -> None:
    """Grade one comparison against the frozen R object."""
    design = produced["design"]
    assert design["group_lv"] == as_list(expected["group_lv"]), f"{path}: group order differs."
    assert_close(design["pairing"], expected["pairing"], path=f"{path}[pairing]")
    assert design["n_dropped"] == expected["n_dropped"]
    assert design["unmatched_ids"] == as_list(expected["unmatched_ids"])
    assert_close(produced["parameters"]["tr"], expected["tr"], path=f"{path}[tr]")
    assert produced["parameters"]["fc_mean"] == expected["fc_mean"]
    assert_close(
        produced["parameters"]["n_posthoc"],
        expected["n_posthoc"],
        path=f"{path}[n_posthoc]",
    )

    assert_frame_close(produced["effect"], expected["effect"], path=f"{path}[effect]")
    assert list(produced["tests"]) == list(expected["tests"]), f"{path}: tests differ."
    for name, table in expected["tests"].items():
        assert_frame_close(produced["tests"][name], table, path=f"{path}[tests][{name}]")

    if "posthoc" not in expected:
        assert "posthoc" not in produced
        assert "pairwise" not in produced
        return

    studentised = {
        name: info["posthoc_id"] in STUDENTISED for name, info in produced["test_info"].items()
    }
    for name, table in expected["posthoc"].items():
        _assert_pairs_close(
            produced["posthoc"][name], table, f"{path}[posthoc][{name}]", studentised[name]
        )
    for name, by_contrast in expected["pairwise"].items():
        assert list(produced["pairwise"][name]) == list(by_contrast), (
            f"{path}: contrast order of {name} differs."
        )
        for contrast, table in by_contrast.items():
            _assert_pairs_close(
                produced["pairwise"][name][contrast],
                table,
                f"{path}[pairwise][{name}][{contrast}]",
                studentised[name],
            )


class TestIndependent:
    @pytest.mark.parametrize(
        ("key", "kwargs"),
        [
            ("all_posthoc", {"posthoc_alpha": 1}),
            ("default_alpha", {}),
            ("reversed", {"control_label": "treat_b", "posthoc_alpha": 1}),
            (
                "tuned",
                {
                    "conf_level": 0.90,
                    "tr": 0.1,
                    "posthoc_alpha": 1,
                    "p_adjust": "holm",
                    "posthoc_p_adjust": "BH",
                },
            ),
            ("no_posthoc", {"posthoc": False}),
        ],
    )
    def test_matches_r(self, key, kwargs):
        frame, expected = load_case("multi_group_independent")
        produced = compare_multiple_groups(
            frame, GENE_FEATS, frame["group"], GENE_LV, diagnose=False, **kwargs
        )
        _assert_case(produced, expected[key], key)

    def test_four_levels_of_unequal_size_and_spread(self):
        """Where the pooled tests and the Welch family part company."""
        frame, expected = load_case("multi_group_unbalanced")
        produced = compare_multiple_groups(
            frame,
            "value",
            frame["group"],
            ["ctrl", "low", "mid", "high"],
            posthoc_alpha=1,
            diagnose=False,
        )
        _assert_case(produced, expected["four_levels"], "four_levels")
        # Six pairs of four levels, and every one of them present per feature.
        assert len(produced["posthoc"]["anova_test"].index) == 6

    def test_a_level_outside_group_lv_is_dropped_rather_than_tested(self):
        frame, expected = load_case("multi_group_unbalanced")
        produced = compare_multiple_groups(
            frame,
            "value",
            frame["group"],
            ["ctrl", "mid", "high"],
            posthoc_alpha=1,
            diagnose=False,
        )
        assert produced["design"]["n_dropped"] == 7
        _assert_case(produced, expected["dropped"], "dropped")


class TestEffect:
    @pytest.mark.parametrize(("key", "kwargs"), [("arith", {}), ("geom", {"fc_mean": "geom"})])
    def test_matches_r(self, key, kwargs):
        frame, expected = load_case("multi_group_effect")
        produced = compare_multiple_groups(
            frame,
            ["prot_1", "prot_2"],
            frame["group"],
            ["ctrl", "case", "other"],
            posthoc_alpha=1,
            diagnose=False,
            **kwargs,
        )
        _assert_case(produced, expected[key], key)

    @pytest.mark.parametrize(
        ("key", "kwargs"),
        [("default_geom", {}), ("explicit_arith", {"fc_mean": "arith"})],
    )
    def test_log2_input_matches_r(self, key, kwargs):
        frame, expected = load_case("multi_group_log2")
        produced = compare_multiple_groups(
            frame,
            ["prot_1", "prot_2"],
            frame["group"],
            ["ctrl", "case", "other"],
            input_scale="log2",
            posthoc_alpha=1,
            diagnose=False,
            **kwargs,
        )
        _assert_case(produced, expected[key], key)

    def test_the_extreme_level_is_the_one_furthest_from_the_reference(self):
        frame, _ = load_case("multi_group_effect")
        produced = compare_multiple_groups(
            frame,
            ["prot_1", "prot_2"],
            frame["group"],
            ["ctrl", "case", "other"],
            posthoc=False,
            diagnose=False,
        )
        effect = produced["effect"].set_index("features")
        for name in effect.index:
            row = effect.loc[name]
            # The ratio puts the extreme level over the reference, so the two
            # centres and the reported ratio have to agree.
            assert row["extreme_center"] / row["ref_center"] == pytest.approx(row["fold_change"])
            assert row["extreme_level"] in {"case", "other"}

    def test_the_pairwise_ratio_and_the_estimate_point_the_same_way(self):
        """``log2fc`` divides ``group1`` by ``group2``, which ``estimate`` subtracts."""
        frame, _ = load_case("multi_group_effect")
        produced = compare_multiple_groups(
            frame,
            ["prot_1", "prot_2"],
            frame["group"],
            ["ctrl", "case", "other"],
            posthoc_alpha=1,
            diagnose=False,
        )
        for table in produced["pairwise"]["anova_test"].values():
            assert (np.sign(table["log2fc"]) == np.sign(table["estimate"])).all()


class TestRepeated:
    @pytest.mark.parametrize(
        ("key", "kwargs"),
        [
            ("plain", {}),
            ("tuned", {"conf_level": 0.90, "posthoc_p_adjust": "BH"}),
            ("reversed", {"control_label": "t3"}),
        ],
    )
    def test_matches_r(self, key, kwargs):
        frame, expected = load_case("multi_group_repeated")
        produced = compare_multiple_groups(
            frame,
            REP_FEATS,
            frame["cond"],
            REP_LV,
            id=frame["subject"],
            paired=True,
            posthoc_alpha=1,
            diagnose=False,
            **kwargs,
        )
        _assert_case(produced, expected[key], key)

    def test_the_repeated_family_is_two_tests_rather_than_four(self):
        frame, _ = load_case("multi_group_repeated")
        produced = compare_multiple_groups(
            frame,
            REP_FEATS,
            frame["cond"],
            REP_LV,
            id=frame["subject"],
            paired=True,
            diagnose=False,
        )
        assert list(produced["tests"]) == ["anova_test", "kruskal_test"]
        assert produced["test_info"]["anova_test"]["id"] == "repeated_measures_anova"

    def test_a_subject_missing_a_condition_is_dropped_whole(self):
        frame, expected = load_case("multi_group_unmatched")
        produced = compare_multiple_groups(
            frame,
            REP_FEATS,
            frame["cond"],
            REP_LV,
            id=frame["subject"],
            paired=True,
            posthoc_alpha=1,
            diagnose=False,
        )
        assert produced["design"]["unmatched_ids"] == ["s05"]
        _assert_case(produced, expected["holed"], "holed")

    def test_a_hole_in_one_feature_costs_only_that_feature(self):
        frame, expected = load_case("multi_group_gappy")
        produced = compare_multiple_groups(
            frame,
            REP_FEATS,
            frame["cond"],
            REP_LV,
            id=frame["subject"],
            paired=True,
            posthoc_alpha=1,
            diagnose=False,
        )
        _assert_case(produced, expected["per_feature_holes"], "per_feature_holes")
        sizes = produced["tests"]["anova_test"].set_index("features")["n_used"]
        assert sizes["score"] < sizes["shifted"]

    def test_paired_without_an_id_is_refused_rather_than_matched_by_order(self):
        frame, _ = load_case("multi_group_repeated")
        with pytest.raises(SaValueError, match="cannot be matched by row order"):
            compare_multiple_groups(frame, REP_FEATS, frame["cond"], REP_LV, paired=True)

    def test_an_id_without_paired_is_ignored_and_says_so(self):
        frame, _ = load_case("multi_group_repeated")
        with pytest.warns(SaWarning, match="only used to match repeated conditions"):
            compare_multiple_groups(
                frame,
                REP_FEATS,
                frame["cond"],
                REP_LV,
                id=frame["subject"],
                diagnose=False,
            )


@functools.lru_cache(maxsize=1)
def _planted():
    """A simulation and its comparison, built once for the contract tests."""
    sim = simulate_multiple_groups(
        n_feats=6, n_control=12, n_treat=(12, 12), n_up=2, n_down=2, seed=11
    )
    return sim, compare_multiple_groups(**sim.args, posthoc_alpha=1)


class TestContract:
    @pytest.fixture
    def result(self):
        return _planted()

    def test_a_simulation_unpacks_straight_into_the_comparison(self, result):
        sim, res = result
        assert res["features"] == list(sim.args["feats"])
        assert res["analysis"] == "multi_group_comparison"

    def test_every_table_carries_the_contract_columns_in_order(self, result):
        _, res = result
        assert list(res["effect"].columns) == EFFECT_COLUMNS
        for table in res["tests"].values():
            assert list(table["features"]) == res["features"]
            for name in contract_columns():
                assert name in table.columns
            names = list(table.columns)
            assert names[names.index("pval") + 1] == "pval_adj"
        for table in res["posthoc"].values():
            assert list(table.columns) == posthoc_table_columns()
        for by_contrast in res["pairwise"].values():
            for table in by_contrast.values():
                assert list(table.columns) == pairwise_table_columns()
                assert list(table["features"]) == res["features"]

    def test_every_omnibus_test_has_its_own_post_hoc_procedure(self, result):
        """A rank-based omnibus test is never followed by a parametric comparison."""
        _, res = result
        procedures = {name: info["posthoc_id"] for name, info in res["test_info"].items()}
        assert procedures == {
            "anova_test": "tukey_hsd",
            "welch_test": "games_howell",
            "robust_test": "pairwise_yuen",
            "kruskal_test": "dunn_test",
        }
        assert set(res["posthoc"]) == set(res["tests"])

    def test_an_omnibus_interval_is_absent_rather_than_invented(self, result):
        """There is no single quantity for it to be about; the contrasts have one."""
        _, res = result
        for table in res["tests"].values():
            assert table["lower_conf"].isna().all()
            assert table["upper_conf"].isna().all()
        assert res["posthoc"]["anova_test"]["lower_conf"].notna().any()

    def test_a_family_wise_procedure_is_not_adjusted_twice(self, result):
        _, res = result
        for name in ("anova_test", "welch_test"):
            table = res["posthoc"][name]
            assert_close(list(table["pval_adj"]), list(table["pval"]))
        dunn = res["posthoc"]["kruskal_test"]
        assert (dunn["pval_adj"] >= dunn["pval"]).all()

    def test_a_feature_that_did_not_qualify_is_absent_from_posthoc_and_present_in_pairwise(
        self, result
    ):
        """Absent means the question was never asked; missing means it failed."""
        sim, _ = result
        strict = compare_multiple_groups(**sim.args, posthoc_alpha=0.001, diagnose=False)
        asked = set(strict["posthoc"]["anova_test"]["features"])
        assert asked < set(strict["features"])
        for table in strict["pairwise"]["anova_test"].values():
            assert list(table["features"]) == strict["features"]
            skipped = table[~table["features"].isin(asked)]
            # The ratio survives, because dividing two centres needs no test.
            assert skipped["estimate"].isna().all()
            assert skipped["log2fc"].notna().all()

    def test_no_posthoc_leaves_the_slots_out_rather_than_empty(self, result):
        sim, _ = result
        res = compare_multiple_groups(**sim.args, posthoc=False, diagnose=False)
        assert "posthoc" not in res
        assert "pairwise" not in res
        assert set(res["parameters"]["n_posthoc"].values()) == {0}

    def test_the_planted_direction_comes_back_signed(self, result):
        sim, res = result
        planted = sim.truth.set_index("features")["log2fc"]
        found = res["effect"].set_index("features")["log2fc"]
        moved = planted[planted != 0].index
        assert (np.sign(found[moved]) == np.sign(planted[moved])).all()

    def test_repointing_the_reference_repoints_every_contrast(self, result):
        sim, res = result
        flipped = compare_multiple_groups(
            **sim.args,
            control_label=sim.args["group_lv"][-1],
            posthoc_alpha=1,
            diagnose=False,
        )
        reference = sim.args["group_lv"][-1]
        assert flipped["design"]["group_lv"][0] == reference
        # The reference is the level every contrast it takes part in subtracts,
        # so it is on the right of those and absent from the rest.
        involved = [
            contrast for contrast in flipped["pairwise"]["anova_test"] if reference in contrast
        ]
        assert len(involved) == len(flipped["design"]["group_lv"]) - 1
        for contrast in involved:
            assert contrast.endswith(f"- {reference}")
        # The omnibus question does not depend on which level is the reference.
        assert_close(
            list(flipped["tests"]["anova_test"]["pval"]),
            list(res["tests"]["anova_test"]["pval"]),
        )

    def test_diagnostics_are_attached_only_when_asked_for(self, result):
        sim, res = result
        assert set(res["diagnostics"]) == {"normality", "variance", "summary"}
        assert list(res["diagnostics"]["normality"]["group"])[:3] == res["design"]["group_lv"]
        assert compare_multiple_groups(**sim.args, diagnose=False)["diagnostics"] is None

    def test_repr_names_the_post_hoc_procedure_under_each_test(self, result):
        _, res = result
        text = repr(res)
        assert "multi_group_comparison" in text
        assert "$anova_test" in text
        assert "post-hoc:" in text
        assert "Tukey HSD" in text


class TestArgumentChecks:
    @pytest.fixture
    def frame(self):
        return pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                "g": ["x", "x", "x", "y", "y", "y", "z", "z", "z"],
            }
        )

    def test_two_levels_is_refused_as_the_wrong_function(self, frame):
        with pytest.raises(SaValueError, match="at least 3"):
            compare_multiple_groups(frame, "a", frame["g"], ["x", "y"])

    def test_an_out_of_range_posthoc_alpha_is_refused(self, frame):
        with pytest.raises(SaValueError, match="posthoc_alpha"):
            compare_multiple_groups(frame, "a", frame["g"], ["x", "y", "z"], posthoc_alpha=0)

    def test_an_unknown_posthoc_adjustment_is_refused(self, frame):
        with pytest.raises(SaValueError, match="posthoc_p_adjust"):
            compare_multiple_groups(
                frame, "a", frame["g"], ["x", "y", "z"], posthoc_p_adjust="holmes"
            )

    def test_an_unknown_input_scale_is_refused(self, frame):
        with pytest.raises(SaValueError, match="`input_scale` must be one of"):
            compare_multiple_groups(frame, "a", frame["g"], ["x", "y", "z"], input_scale="ln")

    def test_a_control_label_outside_group_lv_is_refused(self, frame):
        with pytest.raises(SaValueError, match="names a level `group_lv` does not hold"):
            compare_multiple_groups(frame, "a", frame["g"], ["x", "y", "z"], control_label="w")
