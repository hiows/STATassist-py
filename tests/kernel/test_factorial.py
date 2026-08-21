"""``kernel/factorial.py`` against the numbers R produced."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from golden import as_list, assert_close, assert_frame_close, load_case, load_expected

from statassist.core.errors import SaValueError
from statassist.core.factorial import fact_cell_labels, fact_contrast_skeleton, fact_grid
from statassist.kernel.factorial import (
    QR_RANK_TOL,
    SS_TYPES,
    FactorialFit,
    contr_sum,
    fact_cell_matrix,
    fact_ss_plan,
    factorial_anova,
    factorial_plan,
    factorial_tukey,
)

#: The crossed design the fixtures were generated from.
FACT_LV = {"treatment": ["control", "treat_A", "treat_B"], "sex": ["male", "female"]}

#: A three-factor design, for the column order of a three-way interaction block.
CUBE_LV = {"a": ["a1", "a2"], "b": ["b1", "b2"], "c": ["c1", "c2"]}

#: The cell whose sample the "one cell never observed" fixtures leave out.
HOLED_CELL = 2

#: Tolerance for anything that passes through the studentised range.
#:
#: The same relaxation ``tests/kernel/test_posthoc.py`` makes and for the same
#: reason: R's ``ptukey``/``qtukey`` and
#: :class:`scipy.stats.studentized_range` integrate the same distribution by
#: different quadratures and agree to about 1e-7 relative, which is the accuracy
#: of the quadratures rather than of the port. Every other column of the same
#: table is still graded at ``1e-8``.
QTUKEY_RTOL = 1e-6


def assert_studentised_range_close(actual, expected, *, path):
    """Grade the columns the studentised range touches apart from the rest."""
    touched = {"pval", "lower_conf", "upper_conf"}
    exact = {name: values for name, values in expected.items() if name not in touched}
    approximate = {name: values for name, values in expected.items() if name in touched}
    assert_frame_close(actual[list(exact)], exact, path=path)
    assert_frame_close(actual[list(approximate)], approximate, rtol=QTUKEY_RTOL, path=path)


def fact_samples(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """The per-cell samples of the shared crossed fixture, in cell order."""
    labels = fact_cell_labels(FACT_LV, fact_grid(FACT_LV))
    return {
        label: frame.loc[frame["cell"] == label, "value"].to_numpy(dtype=float) for label in labels
    }


def one_based(values) -> list[int]:
    """The port's zero-based indices as the one-based ones R wrote out."""
    return [int(value) + 1 for value in values]


class TestContrSum:
    def test_matches_r(self):
        expected = load_expected("factorial_cell_matrix")
        assert_close(contr_sum(2).tolist(), expected["contr_sum_2"])
        assert_close(contr_sum(4).tolist(), expected["contr_sum_4"])

    def test_the_columns_sum_to_zero_over_the_levels(self):
        assert_close(contr_sum(5).sum(axis=0).tolist(), [0.0] * 4)

    def test_a_factor_of_one_level_cannot_be_coded(self):
        with pytest.raises(SaValueError, match="at least two levels"):
            contr_sum(1)


class TestCellMatrix:
    def test_matches_r(self):
        expected = load_expected("factorial_cell_matrix")
        produced = fact_cell_matrix(FACT_LV, fact_grid(FACT_LV))
        assert_close(produced.x.tolist(), expected["x"])

    def test_the_assign_vector_matches_r_shifted_to_zero_based(self):
        expected = load_expected("factorial_cell_matrix")
        produced = fact_cell_matrix(FACT_LV, fact_grid(FACT_LV))
        # R numbers the terms from one and gives the intercept zero; the port
        # numbers them from zero and gives the intercept -1, so that
        # `terms[assign[column]]` reads without a correction.
        assert one_based(produced.assign) == as_list(expected["assign"])
        assert produced.assign[0] == -1

    def test_a_three_way_interaction_block_matches_r_column_for_column(self):
        expected = load_expected("factorial_cell_matrix")
        produced = fact_cell_matrix(CUBE_LV, fact_grid(CUBE_LV))
        assert_close(produced.x.tolist(), expected["cube_x"])
        assert one_based(produced.assign) == as_list(expected["cube_assign"])

    def test_the_width_of_a_term_is_the_product_of_its_factors_degrees(self):
        produced = fact_cell_matrix(FACT_LV, fact_grid(FACT_LV))
        widths = [int(np.sum(produced.assign == position)) for position in range(3)]
        assert widths == [2, 1, 2]

    def test_every_term_column_is_orthogonal_to_the_intercept(self):
        # Sum-to-zero coding is what makes this true, and what makes Type I, II
        # and III agree on a balanced design.
        produced = fact_cell_matrix(CUBE_LV, fact_grid(CUBE_LV))
        assert_close(produced.x[:, 1:].sum(axis=0).tolist(), [0.0] * 7)


class TestSsPlan:
    def test_matches_r_for_every_type(self):
        expected = load_expected("factorial_ss_plan")
        matrix = fact_cell_matrix(FACT_LV, fact_grid(FACT_LV))
        for ss_type in SS_TYPES:
            produced = fact_ss_plan(matrix.terms, matrix.assign, ss_type)
            wanted = expected[ss_type]
            for position, pair in enumerate(produced):
                assert one_based(pair.base) == as_list(wanted["base"][position]), (
                    f"{ss_type} term {position} base"
                )
                assert one_based(pair.full) == as_list(wanted["full"][position]), (
                    f"{ss_type} term {position} full"
                )

    def test_a_three_factor_plan_matches_r_for_every_type(self):
        expected = load_expected("factorial_ss_plan")
        matrix = fact_cell_matrix(CUBE_LV, fact_grid(CUBE_LV))
        for ss_type in SS_TYPES:
            produced = fact_ss_plan(matrix.terms, matrix.assign, ss_type)
            wanted = expected[f"cube_{ss_type}"]
            for position, pair in enumerate(produced):
                assert one_based(pair.base) == as_list(wanted["base"][position])
                assert one_based(pair.full) == as_list(wanted["full"][position])

    def test_the_full_model_always_contains_the_base_one(self):
        matrix = fact_cell_matrix(CUBE_LV, fact_grid(CUBE_LV))
        for ss_type in SS_TYPES:
            for pair in fact_ss_plan(matrix.terms, matrix.assign, ss_type):
                assert set(pair.base) <= set(pair.full)

    def test_type_three_holds_every_other_term_in(self):
        matrix = fact_cell_matrix(FACT_LV, fact_grid(FACT_LV))
        plan = fact_ss_plan(matrix.terms, matrix.assign, "III")
        for pair in plan:
            assert list(pair.full) == list(range(matrix.x.shape[1]))

    def test_type_one_is_sequential_so_the_first_term_is_added_to_nothing(self):
        matrix = fact_cell_matrix(FACT_LV, fact_grid(FACT_LV))
        plan = fact_ss_plan(matrix.terms, matrix.assign, "I")
        assert list(plan[0].base) == [0]

    def test_type_two_does_not_adjust_a_main_effect_for_its_own_interaction(self):
        matrix = fact_cell_matrix(FACT_LV, fact_grid(FACT_LV))
        plan = fact_ss_plan(matrix.terms, matrix.assign, "II")
        interaction = np.flatnonzero(matrix.assign == 2)
        assert not set(interaction) & set(plan[0].base)

    def test_an_unknown_type_is_refused(self):
        matrix = fact_cell_matrix(FACT_LV, fact_grid(FACT_LV))
        with pytest.raises(SaValueError, match="`ss_type` must be one of"):
            fact_ss_plan(matrix.terms, matrix.assign, "IV")


class TestFactorialAnova:
    def test_matches_r_for_every_type(self):
        frame, expected = load_case("factorial_anova")
        samples = fact_samples(frame)
        for ss_type in SS_TYPES:
            plan = factorial_plan(FACT_LV, fact_grid(FACT_LV), ss_type)
            fit = factorial_anova(samples, plan)
            wanted = expected[ss_type]
            assert_close(fit.model, wanted["model"], path=f"{ss_type} model")
            assert_frame_close(fit.terms, wanted["terms"], path=f"{ss_type} terms")
            assert list(fit.terms.index) == as_list(wanted["labels"])
            assert_close(list(fit.means), as_list(wanted["means"]))
            assert_close(list(fit.n), as_list(wanted["n"]))
            assert_close(fit.ms_error, wanted["ms_error"])
            assert_close(fit.df_error, wanted["df_error"])

    def test_type_one_matches_what_aov_reports_on_unbalanced_data(self):
        frame, expected = load_case("factorial_anova")
        plan = factorial_plan(FACT_LV, fact_grid(FACT_LV), "I")
        fit = factorial_anova(fact_samples(frame), plan)
        aov = expected["aov_I"]
        # `aov()` lists the residual row last; the term rows come first and in the
        # same order.
        assert_close(list(fit.terms["df"]), as_list(aov["Df"])[:-1])
        assert_close(list(fit.terms["ss"]), as_list(aov["Sum Sq"])[:-1])
        assert_close(list(fit.terms["ms"]), as_list(aov["Mean Sq"])[:-1])
        assert_close(list(fit.terms["f_stat"]), as_list(aov["F value"])[:-1])
        assert_close(list(fit.terms["pval"]), as_list(aov["Pr(>F)"])[:-1])

    def test_type_one_sums_of_squares_add_up_to_the_between_cell_total(self):
        frame, _ = load_case("factorial_anova")
        plan = factorial_plan(FACT_LV, fact_grid(FACT_LV), "I")
        fit = factorial_anova(fact_samples(frame), plan)
        grand = float(np.sum(fit.n * fit.means) / np.sum(fit.n))
        between = float(np.sum(fit.n * (fit.means - grand) ** 2))
        assert float(fit.terms["ss"].sum()) == pytest.approx(between)

    def test_the_whole_model_row_is_the_one_way_anova_over_the_cells(self):
        frame, _ = load_case("factorial_anova")
        plan = factorial_plan(FACT_LV, fact_grid(FACT_LV), "III")
        fit = factorial_anova(fact_samples(frame), plan)
        assert fit.model["n_cells"] == 6
        # `n_groups` is what the one-way kernel calls it; a crossed design counts
        # cells, and the rename keeps the column position.
        assert "n_groups" not in fit.model
        assert list(fit.model)[:2] == ["n_used", "n_cells"]

    def test_a_design_with_a_cell_never_observed_loses_the_rank_it_lost(self):
        frame, expected = load_case("factorial_anova")
        samples = fact_samples(frame)
        labels = list(samples)
        holed_cells = fact_grid(FACT_LV).drop(index=HOLED_CELL)
        holed_samples = {
            label: values
            for position, (label, values) in enumerate(samples.items())
            if position != HOLED_CELL
        }
        assert labels[HOLED_CELL] not in holed_samples

        plan = factorial_plan(FACT_LV, holed_cells, "III")
        fit = factorial_anova(holed_samples, plan)
        wanted = expected["holed"]
        assert_close(fit.model, wanted["model"], path="holed model")
        assert_frame_close(fit.terms, wanted["terms"], path="holed terms")
        # Five cells cannot support six coefficients, so the full model has rank
        # five and every term is measured against that ceiling rather than
        # against its own width. `sex` has nothing left to add once the others
        # are in, which is a degree of freedom of zero and an F statistic of
        # nothing rather than a division by nothing.
        assert list(fit.terms["df"]) == [1.0, 0.0, 1.0]
        assert np.isnan(fit.terms["f_stat"].iloc[1])
        assert np.isnan(fit.terms["pval"].iloc[1])

    def test_a_balanced_design_reads_the_same_under_all_three_types(self):
        # Sum-to-zero coding makes the term blocks mutually orthogonal when the
        # cells are equal in size, so there is nothing left for the types to
        # disagree about.
        rng = np.random.default_rng(11)
        cells = fact_grid(FACT_LV)
        samples = {
            label: rng.normal(10 + index * 0.3, 1.0, 8)
            for index, label in enumerate(fact_cell_labels(FACT_LV, cells))
        }
        tables = [
            factorial_anova(samples, factorial_plan(FACT_LV, cells, ss_type)).terms
            for ss_type in SS_TYPES
        ]
        for table in tables[1:]:
            assert_close(list(table["ss"]), list(tables[0]["ss"]), rtol=1e-12)

    def test_a_cell_with_no_observation_is_named(self):
        frame, _ = load_case("factorial_anova")
        samples = fact_samples(frame)
        samples["treat_A.male"] = np.array([])
        plan = factorial_plan(FACT_LV, fact_grid(FACT_LV), "III")
        with pytest.raises(SaValueError, match="no usable observation.*treat_A.male"):
            factorial_anova(samples, plan)

    def test_a_missing_value_reaching_the_kernel_is_the_callers_error(self):
        frame, _ = load_case("factorial_anova")
        samples = fact_samples(frame)
        samples["control.male"] = np.append(samples["control.male"], np.nan)
        plan = factorial_plan(FACT_LV, fact_grid(FACT_LV), "III")
        with pytest.raises(SaValueError, match="missing or infinite"):
            factorial_anova(samples, plan)

    def test_the_rank_tolerance_is_the_one_r_counts_with(self):
        assert QR_RANK_TOL == 1e-7


class TestFactorialTukey:
    def test_matches_r(self):
        frame, expected = load_case("factorial_tukey")
        cells = fact_grid(FACT_LV)
        fit = factorial_anova(fact_samples(frame), factorial_plan(FACT_LV, cells, "III"))
        skeleton = fact_contrast_skeleton(FACT_LV, cells)
        nmeans = as_list(expected["nmeans"])
        produced = factorial_tukey(
            fit, skeleton.sel1, skeleton.sel2, nmeans, range(len(skeleton.table))
        )
        assert_studentised_range_close(produced, expected["all"], path="all")

    def test_the_confidence_level_reaches_the_interval(self):
        frame, expected = load_case("factorial_tukey")
        cells = fact_grid(FACT_LV)
        fit = factorial_anova(fact_samples(frame), factorial_plan(FACT_LV, cells, "III"))
        skeleton = fact_contrast_skeleton(FACT_LV, cells)
        produced = factorial_tukey(
            fit,
            skeleton.sel1,
            skeleton.sel2,
            as_list(expected["nmeans"]),
            range(len(skeleton.table)),
            conf_level=0.90,
        )
        assert_studentised_range_close(produced, expected["conf_90"], path="conf_90")

    def test_the_row_order_is_the_order_asked_for(self):
        frame, expected = load_case("factorial_tukey")
        cells = fact_grid(FACT_LV)
        fit = factorial_anova(fact_samples(frame), factorial_plan(FACT_LV, cells, "III"))
        skeleton = fact_contrast_skeleton(FACT_LV, cells)
        # R was given 10, 1, 5 one-based, which is 9, 0, 4 here.
        produced = factorial_tukey(
            fit, skeleton.sel1, skeleton.sel2, as_list(expected["nmeans"]), [9, 0, 4]
        )
        assert_studentised_range_close(produced, expected["picked"], path="picked")

    def test_the_nmeans_of_a_factor_is_how_many_levels_it_has(self):
        expected = load_expected("factorial_tukey")
        skeleton = fact_contrast_skeleton(FACT_LV, fact_grid(FACT_LV))
        wanted = [3 if factor == "treatment" else 2 for factor in skeleton.table["factor"]]
        assert as_list(expected["nmeans"]) == wanted

    def test_a_marginal_estimate_is_the_unweighted_difference_of_cell_means(self):
        frame, _ = load_case("factorial_tukey")
        cells = fact_grid(FACT_LV)
        fit = factorial_anova(fact_samples(frame), factorial_plan(FACT_LV, cells, "III"))
        skeleton = fact_contrast_skeleton(FACT_LV, cells)
        row = int(np.flatnonzero(skeleton.table["stratum"].isna())[0])
        produced = factorial_tukey(
            fit, skeleton.sel1, skeleton.sel2, [3] * len(skeleton.table), [row]
        )
        by_hand = float(
            np.mean(fit.means[skeleton.sel1[row]]) - np.mean(fit.means[skeleton.sel2[row]])
        )
        assert produced["estimate"].iloc[0] == pytest.approx(by_hand)

    def test_the_columns_are_the_posthoc_layout(self):
        from statassist.kernel.posthoc import posthoc_columns

        frame, expected = load_case("factorial_tukey")
        cells = fact_grid(FACT_LV)
        fit = factorial_anova(fact_samples(frame), factorial_plan(FACT_LV, cells, "III"))
        skeleton = fact_contrast_skeleton(FACT_LV, cells)
        produced = factorial_tukey(
            fit, skeleton.sel1, skeleton.sel2, as_list(expected["nmeans"]), [0]
        )
        assert list(produced.columns) == posthoc_columns()

    def test_a_model_with_no_residual_variation_cannot_scale_a_contrast(self):
        # `factorial_anova()` never hands over such a fit - the one-way kernel it
        # delegates the whole-model row to refuses cells of zero variance first -
        # so the guard is reached by building the fit rather than by fitting one,
        # which is also the only way R's own guard can be reached.
        cells = fact_grid(FACT_LV)
        skeleton = fact_contrast_skeleton(FACT_LV, cells)
        fit = FactorialFit(
            model={},
            terms=pd.DataFrame(),
            means=np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
            n=np.full(6, 4.0),
            ms_error=0.0,
            df_error=18.0,
        )
        with pytest.raises(SaValueError, match="mean square error of the model is zero"):
            factorial_tukey(fit, skeleton.sel1, skeleton.sel2, [3] * 13, [0])
