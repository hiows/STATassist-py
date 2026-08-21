"""Wide input resolution: column-or-vector arguments, row filtering, level order."""

from __future__ import annotations

import pandas as pd
import pytest

from statassist.core import control_first, resolve_row_vector, validate_wide_input
from statassist.core.errors import SaValueError


@pytest.fixture
def wide() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample": ["s1", "s2", "s3", "s4", "s5"],
            "group": ["case", "case", "control", "control", "other"],
            "gene_1": [5.0, 6.0, 1.0, 2.0, 9.0],
            "gene_2": [1.0, 2.0, 5.0, 6.0, 9.0],
        }
    )


class TestValidateWideInput:
    def test_rows_outside_group_lv_are_dropped_and_counted(self, wide: pd.DataFrame) -> None:
        """Dropped, not coerced to missing: NaN would enter the samples silently."""
        out = validate_wide_input(wide, ["gene_1"], wide["group"], ["case", "control"])
        assert out.n_dropped == 1
        assert len(out.data.index) == 4
        assert out.data["gene_1"].tolist() == [5.0, 6.0, 1.0, 2.0]

    def test_the_filtered_frame_is_reindexed_from_zero(self, wide: pd.DataFrame) -> None:
        """R keeps the original row names; a gapped index would misalign in pandas."""
        out = validate_wide_input(wide, ["gene_1"], wide["group"], ["control", "other"])
        assert out.data.index.tolist() == [0, 1, 2]

    def test_group_lv_fixes_the_level_order(self, wide: pd.DataFrame) -> None:
        """`case` before `control` is the display order asked for, not the sorted one."""
        out = validate_wide_input(wide, ["gene_1"], wide["group"], ["control", "case"])
        assert list(out.group.categories) == ["control", "case"]
        assert list(out.group) == ["case", "case", "control", "control"]

    def test_feats_keep_the_order_given(self, wide: pd.DataFrame) -> None:
        out = validate_wide_input(wide, ["gene_2", "gene_1"], wide["group"], ["case", "control"])
        assert out.feats == ["gene_2", "gene_1"]

    def test_a_non_numeric_feature_is_refused_by_name(self, wide: pd.DataFrame) -> None:
        with pytest.raises(SaValueError, match="Not numeric: sample"):
            validate_wide_input(wide, ["gene_1", "sample"], wide["group"], ["case", "control"])

    def test_a_logical_column_is_not_numeric(self) -> None:
        """``is.numeric`` is FALSE for a logical in R, while pandas counts bool as numeric."""
        frame = pd.DataFrame({"g": ["a", "a", "b", "b"], "flag": [True, False, True, True]})
        with pytest.raises(SaValueError, match="Not numeric: flag"):
            validate_wide_input(frame, ["flag"], frame["g"], ["a", "b"])

    def test_an_unknown_feature_is_refused_by_name(self, wide: pd.DataFrame) -> None:
        with pytest.raises(SaValueError, match="not found in `data`: gene_9"):
            validate_wide_input(wide, ["gene_9"], wide["group"], ["case", "control"])

    def test_a_level_absent_from_the_data_is_refused(self, wide: pd.DataFrame) -> None:
        with pytest.raises(SaValueError, match="absent from `group`: treated"):
            validate_wide_input(wide, ["gene_1"], wide["group"], ["case", "treated"])

    def test_group_and_group_lv_must_arrive_together(self, wide: pd.DataFrame) -> None:
        with pytest.raises(SaValueError, match="both be supplied or both be `None`"):
            validate_wide_input(wide, ["gene_1"], wide["group"], None)

    def test_ungrouped_input_keeps_every_row(self, wide: pd.DataFrame) -> None:
        out = validate_wide_input(wide, ["gene_1"], None, None)
        assert out.group is None
        assert out.n_dropped == 0
        assert len(out.data.index) == 5

    def test_n_levels_pins_the_count(self, wide: pd.DataFrame) -> None:
        with pytest.raises(SaValueError, match="exactly 3 levels, but 2 were given"):
            validate_wide_input(wide, ["gene_1"], wide["group"], ["case", "control"], n_levels=3)

    def test_the_id_is_filtered_alongside_the_data(self, wide: pd.DataFrame) -> None:
        out = validate_wide_input(
            wide, ["gene_1"], wide["group"], ["case", "control"], id=wide["sample"]
        )
        assert out.id == ["s1", "s2", "s3", "s4"]

    def test_a_mismatched_group_length_is_refused(self, wide: pd.DataFrame) -> None:
        with pytest.raises(SaValueError, match="got 2 for 5 rows"):
            validate_wide_input(wide, ["gene_1"], ["case", "control"], ["case", "control"])

    def test_zero_rows_is_refused(self) -> None:
        empty = pd.DataFrame({"g": [], "x": []})
        with pytest.raises(SaValueError, match="zero rows"):
            validate_wide_input(empty, ["x"], None, None)


class TestResolveRowVector:
    def test_a_column_name_resolves_to_that_column(self, wide: pd.DataFrame) -> None:
        out = resolve_row_vector("group", "stratified", wide)
        assert out.label == "group"
        assert out.value is not None
        assert out.value.tolist() == ["case", "case", "control", "control", "other"]

    def test_a_vector_is_labelled_as_one(self, wide: pd.DataFrame) -> None:
        out = resolve_row_vector([1, 2, 3, 4, 5], "stratified", wide)
        assert out.label == "<vector>"

    def test_an_absent_argument_has_no_label(self, wide: pd.DataFrame) -> None:
        """R returns ``NA_character_`` here, which is ``None`` and not the text "NA"."""
        out = resolve_row_vector(None, "stratified", wide)
        assert out.value is None
        assert out.label is None

    def test_a_wrong_length_is_refused(self, wide: pd.DataFrame) -> None:
        with pytest.raises(SaValueError, match="got 2 for 5 row"):
            resolve_row_vector([1, 2], "stratified", wide)

    def test_a_missing_entry_is_refused_by_default(self, wide: pd.DataFrame) -> None:
        with pytest.raises(SaValueError, match="must not contain NA"):
            resolve_row_vector([1, None, 3, 4, 5], "stratified", wide)

    def test_allow_na_lets_it_through(self, wide: pd.DataFrame) -> None:
        out = resolve_row_vector([1, None, 3, 4, 5], "outcome", wide, allow_na=True)
        assert out.value is not None
        assert bool(out.value.isna().any())


class TestControlFirst:
    def test_moves_the_named_level_to_the_front(self) -> None:
        assert control_first(["a", "b", "c"], "c") == ["c", "a", "b"]

    def test_the_rest_keep_the_order_they_were_given(self) -> None:
        assert control_first(["x", "b", "a"], "b") == ["b", "x", "a"]

    def test_none_leaves_the_order_alone(self) -> None:
        assert control_first(["a", "b", "c"], None) == ["a", "b", "c"]

    def test_an_unknown_level_is_refused_with_the_present_ones(self) -> None:
        with pytest.raises(SaValueError, match="does not hold: z. Present: a, b"):
            control_first(["a", "b"], "z")

    def test_the_argument_names_are_reported_as_given(self) -> None:
        """A crossed design names one reference per factor, so the message says which."""
        with pytest.raises(SaValueError, match=r"`control_label\['sex'\]` names a level"):
            control_first(["m", "f"], "x", arg="control_label['sex']", lv_arg="group_lv['sex']")
