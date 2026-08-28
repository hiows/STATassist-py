"""Asking which predictors were worth having, two ways.

The two searches share a result contract and almost nothing else. An elimination
chooses by a resampled score, so its ``profile`` rows are subset sizes and it
always holds rows out; a stepwise search chooses by a penalised likelihood
computed on the rows it was fitted to, so its rows are steps of a path and it
holds nothing out. What is checked here is the contract they share, then the
property each one is named after, then the refusals.

Two of the assertions are cross-checks rather than properties: the criterion a
step of the path reports has to be the one
:func:`~statassist.fit_linear_regression` reports for the same model on the same
rows, and the elimination's ranking has to be non-negative because a magnitude is.
Those are what tie the searches to the fits they are built out of, which is the
only thing here a planted coefficient cannot say.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import pytest

from statassist import (
    fit_linear_regression,
    fit_logistic_regression,
    perform_rfe,
    perform_stepwise,
    simulate_classification,
    simulate_regression,
)
from statassist.core import SaInternalError, SaValueError, new_selection
from statassist.core.contracts import (
    selection_profile_columns,
    selection_ranking_columns,
    stepwise_profile_columns,
)
from statassist.select.rfe import DEFAULT_SIZES, RFE_MODELS, rfe_sizes
from statassist.select.stepwise import CRITERIA, DIRECTIONS, STEPWISE_MODELS

#: Rows and candidates the shared fixtures are built at.
#:
#: Enough rows for a five-fold scheme to leave something in every fold, and few
#: enough candidates that an elimination over every subset size stays quick.
_N_SAMPLES = 90
_N_PRED = 5

#: Folds the searches that resample are run at here.
#:
#: Three rather than the default five, and one run rather than five, because what
#: is being checked is the shape of the answer rather than its precision.
_N_FOLD = 3

#: How far two criteria computed the same way may differ and still be one number.
_CRITERION_TOL = 1e-8


@pytest.fixture(scope="module")
def planted():
    """A regression with known coefficients and no factor among the candidates."""
    return simulate_regression(
        n_samples=_N_SAMPLES,
        n_pred=_N_PRED,
        n_factor_pred=0,
        p_missing=0,
        seed=7,
    )


@pytest.fixture(scope="module")
def factored():
    """The same shape, with one candidate arriving as a three-level factor."""
    return simulate_regression(
        n_samples=_N_SAMPLES,
        n_pred=_N_PRED,
        n_factor_pred=1,
        p_missing=0,
        seed=4,
    )


@pytest.fixture(scope="module")
def labelled():
    """A two-class outcome, so that the classifying paths have something to run on."""
    return simulate_classification(
        n_samples=_N_SAMPLES,
        n_pred=_N_PRED,
        n_factor_pred=0,
        p_missing=0,
        seed=11,
    )


@pytest.fixture(scope="module")
def eliminated(planted):
    return perform_rfe(**planted.args, cv_method="kfold", n_fold=_N_FOLD, seed=1)


@pytest.fixture(scope="module")
def stepped(planted):
    return perform_stepwise(**planted.args)


@pytest.fixture(scope="module")
def both(eliminated, stepped):
    """The two searches, by the name each reports as its ``analysis``."""
    return {"rfe": eliminated, "stepwise": stepped}


class TestSharedContract:
    """What holds for a selection whichever search produced it."""

    def test_the_analysis_names_the_search_that_ran(self, both):
        for name, res in both.items():
            assert res["analysis"] == name

    def test_the_candidates_are_the_row_axis_of_the_ranking(self, both, planted):
        for res in both.values():
            assert res["ranking"]["candidates"].tolist() == res["candidates"]
            assert sorted(res["candidates"]) == sorted(planted.args["predictors"])

    def test_the_ranking_carries_the_contract_columns_in_order(self, both):
        for res in both.values():
            assert list(res["ranking"]) == selection_ranking_columns()

    def test_the_ranking_is_numbered_from_one_and_never_climbs(self, both):
        for res in both.values():
            table = res["ranking"]
            assert table["rank"].tolist() == list(range(1, len(table.index) + 1))
            estimate = table["estimate"].to_numpy(dtype=float)
            assert np.all(np.diff(estimate[~np.isnan(estimate)]) <= 0)

    def test_the_selected_predictors_are_candidates_and_are_flagged_as_such(self, both):
        for res in both.values():
            table = res["ranking"]
            assert set(res["selected"]) <= set(res["candidates"])
            flagged = table.loc[table["selected"].astype(bool), "candidates"].tolist()
            assert flagged == res["selected"]

    def test_the_selection_leads_the_ranking(self, both):
        """A selection is the top of the table, not a set scattered through it."""
        for res in both.values():
            kept = res["ranking"]["selected"].astype(bool).to_numpy()
            assert kept[: len(res["selected"])].all()
            assert not kept[len(res["selected"]) :].any()

    def test_the_profile_marks_one_answer_and_it_is_the_size_that_was_kept(self, both):
        for res in both.values():
            profile = res["profile"]
            for name in selection_profile_columns():
                assert name in profile.columns
            chosen = profile["chosen"].astype(bool)
            assert int(chosen.sum()) == 1
            assert int(profile.loc[chosen, "n_vars"].iloc[0]) == len(res["selected"])

    def test_design_describes_the_rows_the_search_saw(self, both, planted):
        for res in both.values():
            design = res["design"]
            assert design["outcome"] == planted.args["outcome"]
            assert design["outcome_type"] == "continuous"
            assert design["n_obs"] == _N_SAMPLES
            assert design["n_used"] == _N_SAMPLES
            assert design["n_dropped"] == 0
            assert design["predictors"] == planted.args["predictors"]
            assert design["dropped_predictors"] == []
            assert "outcome_lv" not in design

    def test_the_engine_says_what_the_ranking_was_measured_by(self, both):
        for res in both.values():
            engine = res["engine"]
            for name in ("package", "method", "label", "metrics", "importance", "overridden"):
                assert engine.get(name) is not None
            assert engine["overridden"]

    def test_the_public_slots_survive_a_json_round_trip(self, both):
        for res in both.values():
            payload = {
                name: (value.to_dict(orient="list") if isinstance(value, pd.DataFrame) else value)
                for name, value in res.items()
            }
            restored = json.loads(json.dumps(payload))
            assert restored["analysis"] == res["analysis"]
            assert restored["selected"] == res["selected"]

    def test_the_engine_handle_is_off_the_slots(self, both):
        for res in both.values():
            assert "fit" not in dict(res)
            assert res.fit is not None

    def test_the_repr_reports_the_outcome_the_search_and_the_answer(self, both):
        for name, res in both.items():
            text = repr(res)
            assert name in text.splitlines()[0]
            assert "outcome  :" in text
            assert "search   :" in text
            assert "settings :" in text
            assert f"selected : {len(res['selected'])} of {len(res['candidates'])}" in text
            assert res["engine"]["importance"] in text
            for candidate in res["candidates"]:
                assert candidate in text


class TestElimination:
    """What is specific to a search that scores every subset size."""

    def test_every_size_that_was_scored_gets_a_row_and_the_full_set_is_among_them(self, eliminated):
        assert eliminated["profile"]["n_vars"].tolist() == list(range(1, _N_PRED + 1))

    def test_each_metric_arrives_with_its_spread_over_the_resamples(self, eliminated):
        profile = eliminated["profile"]
        for metric in eliminated["engine"]["metrics"]:
            assert metric in profile.columns
            assert f"{metric}SD" in profile.columns
            assert np.isfinite(profile[metric].to_numpy(dtype=float)).all()

    def test_the_chosen_size_is_the_one_that_placed_first_on_the_chosen_metric(self, eliminated):
        profile = eliminated["profile"]
        metric = eliminated["parameters"]["metric"]
        assert eliminated["parameters"]["maximize"] is False
        best = profile.loc[profile[metric].idxmin(), "n_vars"]
        assert int(best) == len(eliminated["selected"])

    def test_a_magnitude_cannot_be_negative(self, eliminated):
        """The two regressions rank by an absolute statistic, so the column is one."""
        assert (eliminated["ranking"]["estimate"].to_numpy(dtype=float) >= 0).all()

    def test_one_row_of_the_resampling_table_per_resample_at_the_chosen_size(self, eliminated):
        resampling = eliminated["resampling"]
        assert len(resampling.index) == _N_FOLD
        assert resampling["Resample"].tolist() == [f"Fold{n}" for n in range(1, _N_FOLD + 1)]

    def test_the_resampling_scheme_is_recorded_as_it_was_used(self, eliminated):
        params = eliminated["parameters"]
        assert params["cv_method"] == "kfold"
        assert params["n_fold"] == _N_FOLD
        assert params["n_repeat"] is None
        assert params["seed"] == 1

    def test_the_predictor_with_the_largest_planted_coefficient_survives(self, eliminated, planted):
        largest = planted.truth.loc[planted.truth["beta"].abs().idxmax(), "predictors"]
        assert largest in eliminated["selected"]
        assert eliminated["ranking"]["candidates"].iloc[0] == largest

    def test_a_named_ladder_is_scored_and_the_full_set_is_added_to_it(self, planted):
        res = perform_rfe(
            **planted.args, subset_sizes=[1, 2], cv_method="kfold", n_fold=_N_FOLD, seed=1
        )
        assert res["profile"]["n_vars"].tolist() == [1, 2, _N_PRED]

    def test_the_metric_the_caller_names_is_the_one_that_chooses(self, planted):
        res = perform_rfe(
            **planted.args, metric="Rsquared", cv_method="kfold", n_fold=_N_FOLD, seed=1
        )
        assert res["parameters"]["metric"] == "Rsquared"
        assert res["parameters"]["maximize"] is True
        profile = res["profile"]
        best = profile.loc[profile["Rsquared"].idxmax(), "n_vars"]
        assert int(best) == len(res["selected"])

    def test_a_two_class_outcome_is_ranked_by_a_wald_statistic_and_scored_as_classes(
        self, labelled
    ):
        res = perform_rfe(
            **labelled.args, model="logistic", cv_method="kfold", n_fold=_N_FOLD, seed=1
        )
        assert res["design"]["outcome_type"] == "two classes"
        assert res["design"]["outcome_lv"] == list(labelled.args["outcome_lv"])
        assert res["engine"]["metrics"] == ["Accuracy", "Kappa"]
        assert res["engine"]["importance"] == "absolute Wald z"
        assert res["parameters"]["maximize"] is True

    def test_a_forest_ranks_by_permutation_importance_and_records_its_own_arguments(self, labelled):
        res = perform_rfe(
            **labelled.args,
            model="rf",
            ntree=40,
            cv_method="kfold",
            n_fold=_N_FOLD,
            seed=1,
        )
        assert res["engine"]["importance"] == "permutation importance"
        assert res["parameters"]["ntree"] == 40
        assert res["parameters"]["nodesize"] == 1
        assert len(res["selected"]) >= 1

    def test_a_factor_is_eliminated_as_one_candidate_and_never_as_its_dummies(self, factored):
        res = perform_rfe(**factored.args, cv_method="kfold", n_fold=_N_FOLD, seed=1)
        assert sorted(res["candidates"]) == sorted(factored.args["predictors"])
        assert "predictor_lv" in res["design"]

    def test_leave_one_out_is_scored_on_the_pooled_predictions_and_reports_no_resamples(
        self, planted
    ):
        res = perform_rfe(**planted.args, subset_sizes=[2], cv_method="loocv")
        assert res["resampling"] is None
        assert res["parameters"]["n_fold"] is None
        assert "RMSESD" not in res["profile"].columns

    def test_the_same_seed_gives_the_same_search_twice(self, planted):
        first = perform_rfe(**planted.args, cv_method="kfold", n_fold=_N_FOLD, seed=3)
        again = perform_rfe(**planted.args, cv_method="kfold", n_fold=_N_FOLD, seed=3)
        assert first["selected"] == again["selected"]
        pd.testing.assert_frame_equal(first["profile"], again["profile"])


class TestPath:
    """What is specific to a search that walks one move at a time."""

    def test_the_profile_carries_both_criteria_and_the_move_that_reached_each_model(self, stepped):
        assert list(stepped["profile"]) == stepwise_profile_columns()
        assert stepped["profile"]["step"].iloc[0] == ""
        assert all(step.startswith("- ") for step in stepped["profile"]["step"].iloc[1:])

    def test_the_answer_is_where_the_search_stopped(self, stepped):
        assert bool(stepped["profile"]["chosen"].iloc[-1])
        assert int(stepped["profile"]["n_vars"].iloc[-1]) == len(stepped["selected"])

    def test_nothing_was_resampled(self, stepped):
        assert stepped["resampling"] is None
        assert stepped["parameters"]["maximize"] is False
        assert stepped["parameters"]["k"] == pytest.approx(2.0)

    def test_a_backward_path_only_ever_drops(self, stepped):
        sizes = stepped["profile"]["n_vars"].to_numpy(dtype=int)
        assert sizes[0] == _N_PRED
        assert np.all(np.diff(sizes) == -1)

    def test_the_criterion_it_moved_by_falls_at_every_step(self, stepped):
        criterion = stepped["profile"][stepped["parameters"]["criterion"]].to_numpy(dtype=float)
        assert np.all(np.diff(criterion) < 0)

    def test_the_criterion_of_a_step_is_the_one_a_fit_reports_for_that_model(
        self, stepped, planted
    ):
        """The cross-check: a step of the path and a fitted model are one number."""
        fitted = fit_linear_regression(
            planted.args["data"],
            outcome=planted.args["outcome"],
            predictors=stepped["selected"],
            cv=False,
        )
        best = stepped["profile"].loc[stepped["profile"]["chosen"].astype(bool)].iloc[0]
        assert float(best["AIC"]) == pytest.approx(fitted["fit_stats"]["aic"], abs=_CRITERION_TOL)
        assert float(best["BIC"]) == pytest.approx(fitted["fit_stats"]["bic"], abs=_CRITERION_TOL)

    def test_the_ranking_prices_what_dropping_each_predictor_would_cost(self, stepped):
        """Positive for a predictor the search kept and negative for one it left."""
        table = stepped["ranking"]
        kept = table.loc[table["selected"].astype(bool), "estimate"].to_numpy(dtype=float)
        left = table.loc[~table["selected"].astype(bool), "estimate"].to_numpy(dtype=float)
        assert (kept > 0).all()
        assert (left < 0).all()

    def test_a_heavier_charge_per_parameter_keeps_a_subset_of_what_a_lighter_one_kept(
        self, stepped, planted
    ):
        heavy = perform_stepwise(**planted.args, criterion="BIC")
        assert heavy["parameters"]["k"] == pytest.approx(np.log(_N_SAMPLES))
        assert len(heavy["selected"]) <= len(stepped["selected"])
        assert set(heavy["selected"]) <= set(stepped["selected"])

    def test_a_forward_path_starts_at_the_intercept_and_only_adds(self, planted):
        res = perform_stepwise(**planted.args, direction="forward")
        sizes = res["profile"]["n_vars"].to_numpy(dtype=int)
        assert sizes[0] == 0
        assert np.all(np.diff(sizes) == 1)
        assert all(step.startswith("+ ") for step in res["profile"]["step"].iloc[1:])

    def test_the_two_directions_that_start_at_the_full_set_agree_on_this_data(
        self, stepped, planted
    ):
        """``"both"`` is ``"backward"`` plus the option of putting a term back."""
        res = perform_stepwise(**planted.args, direction="both")
        assert res["selected"] == stepped["selected"]

    def test_a_two_class_outcome_is_priced_on_its_own_deviance(self, labelled):
        res = perform_stepwise(**labelled.args, model="logistic")
        fitted = fit_logistic_regression(
            labelled.args["data"],
            outcome=labelled.args["outcome"],
            predictors=res["selected"],
            outcome_lv=labelled.args["outcome_lv"],
            cv=False,
        )
        best = res["profile"].loc[res["profile"]["chosen"].astype(bool)].iloc[0]
        assert float(best["AIC"]) == pytest.approx(fitted["fit_stats"]["aic"], abs=_CRITERION_TOL)
        assert res["engine"]["importance"] == "AIC increase when the predictor is left out"

    def test_a_factor_moves_as_one_term_however_many_columns_it_becomes(self, factored):
        res = perform_stepwise(**factored.args)
        moves = [step[2:] for step in res["profile"]["step"].iloc[1:]]
        assert set(moves) <= set(factored.args["predictors"])

    def test_the_same_rows_give_the_same_path_every_time(self, planted):
        first = perform_stepwise(**planted.args)
        again = perform_stepwise(**planted.args)
        assert first["selected"] == again["selected"]
        pd.testing.assert_frame_equal(first["profile"], again["profile"])


class TestRefusals:
    """What is refused, and by which of the two searches."""

    def test_a_model_outside_the_list_each_search_offers(self, planted):
        with pytest.raises(SaValueError, match="`model` must be one of"):
            perform_rfe(**planted.args, model="glmnet")
        with pytest.raises(SaValueError, match="`model` must be one of"):
            perform_stepwise(**planted.args, model="rf")
        assert "rf" in RFE_MODELS
        assert "rf" not in STEPWISE_MODELS

    def test_a_criterion_or_direction_outside_the_list(self, planted):
        with pytest.raises(SaValueError, match="`criterion` must be one of"):
            perform_stepwise(**planted.args, criterion="Mallows")
        with pytest.raises(SaValueError, match="`direction` must be one of"):
            perform_stepwise(**planted.args, direction="sideways")
        assert set(CRITERIA) == {"AIC", "BIC"}
        assert set(DIRECTIONS) == {"backward", "both", "forward"}

    def test_a_straight_line_through_a_number_refuses_class_labels(self, labelled):
        for search in (perform_rfe, perform_stepwise):
            with pytest.raises(SaValueError, match="set of class labels"):
                search(**labelled.args, model="linear")

    def test_a_two_class_model_refuses_a_continuous_outcome(self, planted):
        for search in (perform_rfe, perform_stepwise):
            with pytest.raises(SaValueError, match="classifies two classes"):
                search(**planted.args, model="logistic")

    def test_naming_the_classes_two_ways_and_disagreeing(self, labelled):
        levels = list(labelled.args["outcome_lv"])
        with pytest.raises(SaValueError, match="disagree about which class"):
            perform_stepwise(
                labelled.args["data"],
                outcome=labelled.args["outcome"],
                predictors=labelled.args["predictors"],
                model="logistic",
                outcome_lv=levels,
                control_label=levels[1],
            )

    def test_a_numeric_outcome_of_two_values_is_searched_as_a_regression_with_a_note(
        self, planted, caplog
    ):
        frame = planted.args["data"].copy()
        first = planted.args["predictors"][0]
        # Tied to a candidate, so that the search has something to keep: what is
        # being checked is which reading the column got, not what a coin flip is
        # worth to a criterion.
        frame[planted.args["outcome"]] = (frame[first] > 0).astype(float)
        with caplog.at_level(logging.INFO):
            res = perform_stepwise(
                frame, outcome=planted.args["outcome"], predictors=planted.args["predictors"]
            )
        assert res["design"]["outcome_type"] == "continuous"
        assert res["selected"] == [first]
        assert "searched as a regression" in caplog.text

    def test_an_outcome_that_holds_an_infinity(self, planted):
        frame = planted.args["data"].copy()
        frame.loc[frame.index[0], planted.args["outcome"]] = np.inf
        for search in (perform_rfe, perform_stepwise):
            with pytest.raises(SaValueError, match="non-finite value"):
                search(
                    frame, outcome=planted.args["outcome"], predictors=planted.args["predictors"]
                )

    def test_a_metric_that_belongs_to_the_other_kind_of_outcome(self, planted):
        with pytest.raises(SaValueError, match="`metric` must be one of RMSE"):
            perform_rfe(**planted.args, metric="Accuracy")

    def test_subset_sizes_that_do_not_count_predictors(self, planted):
        with pytest.raises(SaValueError, match="whole numbers"):
            perform_rfe(**planted.args, subset_sizes=[1.5])
        with pytest.raises(SaValueError, match=r"must be in \[1, 5\]"):
            perform_rfe(**planted.args, subset_sizes=[0, 9])

    def test_a_search_that_walks_back_to_the_intercept_says_so(self, planted):
        """No candidate pays for itself, which is an answer this contract cannot hold."""
        frame = planted.args["data"].copy()
        rng = np.random.default_rng(0)
        frame[planted.args["outcome"]] = rng.normal(size=len(frame.index))
        with pytest.raises(SaValueError, match="walked back to the intercept"):
            perform_stepwise(
                frame,
                outcome=planted.args["outcome"],
                predictors=planted.args["predictors"],
                criterion="BIC",
            )


class TestSizeLadder:
    """The default ladder, which decides how much of the profile is worth scoring."""

    def test_the_ladder_is_capped_at_the_candidate_count(self):
        assert rfe_sizes(None, 4) == [1, 2, 3, 4]
        assert rfe_sizes(None, 12) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]

    def test_the_ladder_thins_out_where_a_size_stops_being_a_different_model(self):
        assert rfe_sizes(None, 60) == [*range(1, 11), 15, 20, 30, 50, 60]
        assert max(DEFAULT_SIZES) == 100

    def test_the_full_set_is_scored_whichever_sizes_were_asked_for(self):
        assert rfe_sizes([2], 7) == [2, 7]
        assert rfe_sizes([7], 7) == [7]


class TestContractGuards:
    """The checks inside the factory, which fire only on a mistake in the package."""

    @staticmethod
    def _parts():
        candidates = ["b", "a"]
        ranking = pd.DataFrame(
            {
                "candidates": candidates,
                "estimate": [2.0, 1.0],
                "rank": [1, 2],
                "selected": [True, False],
            }
        )
        profile = pd.DataFrame({"n_vars": [1, 2], "RMSE": [1.0, 2.0], "chosen": [True, False]})
        return {
            "analysis": "rfe",
            "candidates": candidates,
            "design": {"outcome": "y"},
            "parameters": {"model": "linear"},
            "selected": ["b"],
            "ranking": ranking,
            "profile": profile,
            "engine": {
                "package": "scikit-learn",
                "method": "rfe",
                "label": "Linear regression",
                "metrics": ["RMSE"],
                "importance": "absolute t statistic",
                "overridden": ["none"],
            },
        }

    def test_the_parts_as_they_stand_are_accepted(self):
        assert new_selection(**self._parts())["analysis"] == "rfe"

    def test_a_search_this_contract_does_not_cover(self):
        parts = self._parts()
        parts["analysis"] = "lasso"
        with pytest.raises(SaInternalError, match="`analysis` must be one of"):
            new_selection(**parts)

    def test_a_ranking_in_a_different_order_from_the_candidates(self):
        parts = self._parts()
        parts["candidates"] = ["a", "b"]
        with pytest.raises(SaInternalError, match="aligned with `candidates`"):
            new_selection(**parts)

    def test_a_selection_that_kept_nothing(self):
        parts = self._parts()
        parts["selected"] = []
        with pytest.raises(SaInternalError, match="`selected` must be a non-empty"):
            new_selection(**parts)

    def test_a_selection_that_kept_something_it_was_never_offered(self):
        parts = self._parts()
        parts["selected"] = ["z"]
        with pytest.raises(SaInternalError, match="not candidates: z"):
            new_selection(**parts)

    def test_a_flag_column_that_disagrees_with_the_selection(self):
        parts = self._parts()
        parts["selected"] = ["a"]
        with pytest.raises(SaInternalError, match="disagrees with `selected`"):
            new_selection(**parts)

    def test_a_profile_that_marks_no_answer_or_two(self):
        for marks in ([False, False], [True, True]):
            parts = self._parts()
            parts["profile"] = parts["profile"].assign(chosen=marks)
            with pytest.raises(SaInternalError, match="exactly one row of `profile`"):
                new_selection(**parts)

    def test_a_chosen_size_that_is_not_the_size_that_was_kept(self):
        parts = self._parts()
        parts["profile"] = parts["profile"].assign(chosen=[False, True])
        with pytest.raises(SaInternalError, match="is a subset of 2 variable"):
            new_selection(**parts)

    def test_an_engine_that_does_not_say_what_it_ranked_by(self):
        parts = self._parts()
        del parts["engine"]["importance"]
        with pytest.raises(SaInternalError, match="missing `importance`"):
            new_selection(**parts)

    def test_a_resampling_slot_that_is_not_a_table(self):
        parts = self._parts()
        with pytest.raises(SaInternalError, match="`resampling` must be a DataFrame"):
            new_selection(**parts, resampling="three folds")
