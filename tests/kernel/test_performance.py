"""``kernel/performance.py`` against the numbers R produced."""

from __future__ import annotations

import numpy as np
import pytest
from golden import as_list, assert_close, assert_frame_close, load_case

from statassist.core.errors import SaInternalError, SaValueError
from statassist.kernel.performance import (
    auc,
    auc_delong,
    brier,
    check_response,
    delong_test,
    idi,
    nri,
    placement_values,
    roc_points,
    threshold_scores,
)


def scored(case: str):
    """The response and the two predictors of one fixture, plus its result."""
    frame, expected = load_case(case)
    return (
        frame["response"].to_numpy(dtype=float),
        frame["predictor_old"].to_numpy(dtype=float),
        frame["predictor_new"].to_numpy(dtype=float),
        expected,
    )


class TestCheckResponse:
    def test_a_length_mismatch_is_the_callers_contract(self):
        with pytest.raises(SaInternalError, match="differ in length"):
            check_response([0, 1, 1], [0.2, 0.8])

    def test_anything_but_zero_and_one_is_the_callers_contract(self):
        with pytest.raises(SaInternalError, match="must be 0/1 with 1 for the event"):
            check_response([0, 1, 2], [0.2, 0.8, 0.5])

    def test_a_float_indicator_says_the_same_thing_as_an_integer_one(self):
        # R tests `response %in% c(0, 1)`, which a double passes; the callers
        # convert a two-level factor to a numeric indicator and a kernel has no
        # reason to care which numeric type came out of that.
        check_response([0.0, 1.0, 1.0], [0.2, 0.8, 0.5])

    def test_a_single_class_is_something_the_user_can_see_and_fix(self):
        with pytest.raises(SaValueError, match="single class"):
            check_response([0, 0, 0], [0.2, 0.8, 0.5])
        with pytest.raises(SaValueError, match="single class"):
            check_response([1, 1, 1], [0.2, 0.8, 0.5])


class TestRocPoints:
    def test_matches_r(self):
        response, old, new, expected = scored("perf_roc")
        assert_frame_close(roc_points(response, old), expected["points_old"], path="old")
        assert_frame_close(roc_points(response, new), expected["points_new"], path="new")

    def test_the_curve_runs_from_the_corner_to_the_corner(self):
        response, old, _, _ = scored("perf_roc")
        points = roc_points(response, old)
        assert points["threshold"].iloc[0] == np.inf
        assert points["sensitivity"].iloc[0] == 0.0
        assert points["specificity"].iloc[0] == 1.0
        assert points["sensitivity"].iloc[-1] == pytest.approx(1.0)
        assert points["specificity"].iloc[-1] == pytest.approx(0.0)

    def test_a_run_of_tied_predictions_is_one_point_rather_than_several(self):
        response, old, _, _ = scored("perf_roc")
        # The fixture plants ties on purpose, so there are fewer points than rows
        # plus one.
        assert len(roc_points(response, old)) == len(np.unique(old)) + 1
        assert len(np.unique(old)) < old.size


class TestPlacementValues:
    def test_matches_r(self):
        response, old, new, expected = scored("perf_roc")
        for label, scores in (("old", old), ("new", new)):
            produced = placement_values(response, scores)
            wanted = expected[f"placement_{label}"]
            assert_close(list(produced.event), as_list(wanted["event"]), path=f"{label} event")
            assert_close(list(produced.other), as_list(wanted["other"]), path=f"{label} other")

    def test_both_classes_average_to_the_auc(self):
        response, old, _, _ = scored("perf_roc")
        produced = placement_values(response, old)
        area = auc(response, old)
        assert float(np.mean(produced.event)) == pytest.approx(area)
        assert float(np.mean(produced.other)) == pytest.approx(area)


class TestAuc:
    def test_matches_r(self):
        response, old, new, expected = scored("perf_roc")
        assert_close(auc(response, old), expected["auc_old"])
        assert_close(auc(response, new), expected["auc_new"])

    def test_a_perfect_ranking_is_one_and_a_reversed_one_is_zero(self):
        response = np.array([0.0, 0.0, 1.0, 1.0])
        assert auc(response, [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
        assert auc(response, [0.9, 0.8, 0.2, 0.1]) == pytest.approx(0.0)

    def test_a_prediction_that_ties_everything_is_a_coin_flip(self):
        assert auc([0.0, 0.0, 1.0, 1.0], [0.5] * 4) == pytest.approx(0.5)

    def test_delong_matches_r(self):
        response, old, new, expected = scored("perf_roc")
        assert_close(auc_delong(response, old), expected["delong_old"])
        assert_close(auc_delong(response, new), expected["delong_new"])

    def test_the_delong_area_is_the_same_number_the_rank_form_gives(self):
        response, old, _, _ = scored("perf_roc")
        assert auc_delong(response, old)["auc"] == pytest.approx(auc(response, old))

    def test_a_class_of_one_row_has_no_spread_to_report(self):
        response, old, _, expected = scored("perf_thin")
        assert_close(auc_delong(response, old), expected["delong"])
        assert np.isnan(auc_delong(response, old)["se"])


class TestDelongTest:
    def test_matches_r(self):
        response, old, new, expected = scored("perf_compare")
        assert_close(delong_test(response, new, old), expected["delong_test"])

    def test_reversing_the_pair_reverses_the_sign_and_nothing_else(self):
        response, old, new, expected = scored("perf_compare")
        assert_close(delong_test(response, old, new), expected["delong_reversed"])
        forward = delong_test(response, new, old)
        backward = delong_test(response, old, new)
        assert backward["delta"] == pytest.approx(-forward["delta"])
        assert backward["pval"] == pytest.approx(forward["pval"])

    def test_two_models_that_rank_alike_report_nothing_rather_than_certainty(self):
        response, old, _, expected = scored("perf_compare")
        assert_close(delong_test(response, old, old), expected["delong_identical"])
        produced = delong_test(response, old, old)
        assert produced["se"] == 0.0
        assert np.isnan(produced["statistic"])
        assert np.isnan(produced["pval"])

    def test_a_class_of_one_row_leaves_the_covariance_undefined(self):
        response, old, new, expected = scored("perf_thin")
        assert_close(delong_test(response, new, old), expected["delong_test"])


class TestIdi:
    def test_matches_r(self):
        response, old, new, expected = scored("perf_compare")
        assert_close(idi(response, old, new), expected["idi"])

    def test_a_model_that_did_not_move_anything_moved_nothing(self):
        response, old, _, expected = scored("perf_compare")
        assert_close(idi(response, old, old), expected["idi_identical"])
        assert idi(response, old, old)["idi"] == 0.0

    def test_it_sees_a_shift_an_auc_cannot(self):
        # Every event pushed up by a tenth without reordering anything: the AUC is
        # untouched and the IDI is not.
        response = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 0.0])
        old = np.array([0.1, 0.2, 0.6, 0.7, 0.8, 0.15])
        new = old + np.where(response == 1, 0.1, 0.0)
        assert auc(response, new) == pytest.approx(auc(response, old))
        assert idi(response, old, new)["idi"] == pytest.approx(0.1)

    def test_a_class_of_one_row_has_no_spread_to_report(self):
        response, old, new, expected = scored("perf_thin")
        assert_close(idi(response, old, new), expected["idi"])


class TestNri:
    def test_matches_r(self):
        response, old, new, expected = scored("perf_compare")
        assert_close(nri(response, old, new), expected["nri"])

    def test_a_model_that_did_not_move_anything_reclassified_nothing(self):
        response, old, _, expected = scored("perf_compare")
        assert_close(nri(response, old, old), expected["nri_identical"])
        produced = nri(response, old, old)
        assert produced["nri"] == 0.0
        assert produced["se"] == 0.0

    def test_the_total_is_the_sum_of_the_two_class_wise_components(self):
        response, old, new, _ = scored("perf_compare")
        produced = nri(response, old, new)
        assert produced["nri"] == pytest.approx(produced["nri_event"] + produced["nri_other"])

    def test_only_the_direction_of_a_change_is_counted(self):
        # Doubling every movement leaves the reclassification unchanged, which is
        # what separates this from the IDI.
        response, old, new, _ = scored("perf_compare")
        doubled = old + 2 * (new - old)
        assert_close(nri(response, old, doubled), nri(response, old, new))

    def test_a_class_of_one_row_still_reports_the_components(self):
        response, old, new, expected = scored("perf_thin")
        assert_close(nri(response, old, new), expected["nri"])


class TestBrierAndThresholds:
    def test_brier_matches_r(self):
        response, old, new, expected = scored("perf_scores")
        assert_close(brier(response, old), expected["brier_old"])
        assert_close(brier(response, new), expected["brier_new"])

    def test_a_perfect_forecast_scores_zero(self):
        response = np.array([0.0, 1.0, 1.0, 0.0])
        assert brier(response, response) == 0.0

    def test_the_threshold_scores_match_r(self):
        response, old, _, expected = scored("perf_scores")
        produced = [threshold_scores(response, old, cut) for cut in as_list(expected["thresholds"])]
        wanted = expected["at_threshold"]
        for name in ("accuracy", "sensitivity", "specificity"):
            assert_close(
                [row[name] for row in produced],
                as_list(wanted[name]),
                path=f"at_threshold[{name!r}]",
            )

    def test_a_threshold_of_zero_calls_everything_an_event(self):
        response, old, _, _ = scored("perf_scores")
        produced = threshold_scores(response, old, 0.0)
        assert produced["sensitivity"] == 1.0
        assert produced["specificity"] == 0.0

    def test_the_call_is_at_or_above_the_threshold(self):
        # The same direction `roc_points` steps in: a row sitting exactly on the
        # threshold is called an event.
        produced = threshold_scores([0.0, 1.0], [0.5, 0.5], 0.5)
        assert produced["sensitivity"] == 1.0
        assert produced["specificity"] == 0.0
