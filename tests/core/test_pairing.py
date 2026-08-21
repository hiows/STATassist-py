"""Pairing two groups and aligning many conditions.

Every index these return is a zero-based row position, where R's ``which()``
returns a one-based one.
"""

from __future__ import annotations

import pytest

from statassist.core import align_by_subject, pair_by_id, pair_by_order
from statassist.core.errors import SaValueError

GROUP = ["pre", "pre", "pre", "post", "post", "post"]


class TestPairByOrder:
    def test_pairs_by_row_position(self) -> None:
        out = pair_by_order(GROUP, ["pre", "post"])
        assert out.idx_x.tolist() == [0, 1, 2]
        assert out.idx_y.tolist() == [3, 4, 5]
        assert out.unmatched == []

    def test_unequal_group_sizes_are_refused(self) -> None:
        with pytest.raises(SaValueError, match="pre = 3, post = 2"):
            pair_by_order(["pre", "pre", "pre", "post", "post"], ["pre", "post"])


class TestPairById:
    def test_pairs_in_the_row_order_of_the_first_level(self) -> None:
        ids = ["s1", "s2", "s3", "s3", "s1", "s2"]
        out = pair_by_id(ids, GROUP, ["pre", "post"])
        assert out.idx_x.tolist() == [0, 1, 2]
        assert out.idx_y.tolist() == [4, 5, 3]

    def test_reports_the_ids_that_appear_in_only_one_group(self) -> None:
        ids = ["s1", "s2", "s3", "s1", "s2", "s9"]
        out = pair_by_id(ids, GROUP, ["pre", "post"])
        assert out.idx_x.tolist() == [0, 1]
        assert out.idx_y.tolist() == [3, 4]
        assert out.unmatched == ["s3", "s9"]

    def test_a_repeated_id_within_a_group_is_ambiguous(self) -> None:
        ids = ["s1", "s1", "s3", "s1", "s2", "s3"]
        with pytest.raises(SaValueError, match="Repeated id\\(s\\): s1"):
            pair_by_id(ids, GROUP, ["pre", "post"])

    def test_a_missing_id_is_refused(self) -> None:
        ids = ["s1", None, "s3", "s1", "s2", "s3"]
        with pytest.raises(SaValueError, match="must not contain NA"):
            pair_by_id(ids, GROUP, ["pre", "post"])

    def test_fewer_than_two_pairs_is_refused(self) -> None:
        ids = ["s1", "s2", "s3", "s1", "s8", "s9"]
        with pytest.raises(SaValueError, match="only 1 id\\(s\\) appear in both"):
            pair_by_id(ids, GROUP, ["pre", "post"])


class TestAlignBySubject:
    GROUP3 = ["t0", "t0", "t0", "t1", "t1", "t2", "t2"]
    IDS3 = ["s2", "s1", "s3", "s1", "s2", "s2", "s1"]

    def test_the_index_is_labelled_by_subject_and_condition(self) -> None:
        """R returns a matrix with dimnames; a frame is what carries both in pandas."""
        out = align_by_subject(self.IDS3, self.GROUP3, ["t0", "t1", "t2"])
        assert out.idx.index.tolist() == ["s2", "s1"]
        assert out.idx.columns.tolist() == ["t0", "t1", "t2"]
        assert out.idx.loc["s2"].tolist() == [0, 4, 5]
        assert out.idx.loc["s1"].tolist() == [1, 3, 6]

    def test_subjects_come_out_in_first_appearance_order(self) -> None:
        """Not sorted, so a numeric id is not silently reordered as text."""
        out = align_by_subject(self.IDS3, self.GROUP3, ["t0", "t1", "t2"])
        assert out.subjects == ["s2", "s1"]

    def test_an_incomplete_subject_is_dropped_and_reported(self) -> None:
        out = align_by_subject(self.IDS3, self.GROUP3, ["t0", "t1", "t2"])
        assert out.unmatched == ["s3"]

    def test_a_repeated_id_within_a_condition_names_the_condition(self) -> None:
        ids = ["s1", "s1", "s2", "s1", "s2", "s1", "s2"]
        with pytest.raises(SaValueError, match=r"Repeated id\(s\) in `t0`: s1"):
            align_by_subject(ids, self.GROUP3, ["t0", "t1", "t2"])

    def test_fewer_than_two_complete_subjects_is_refused(self) -> None:
        ids = ["s1", "s2", "s3", "s1", "s7", "s1", "s8"]
        with pytest.raises(SaValueError, match=r"only 1 subject\(s\) have all 3 condition"):
            align_by_subject(ids, self.GROUP3, ["t0", "t1", "t2"])

    def test_a_missing_id_is_refused(self) -> None:
        ids = ["s2", None, "s3", "s1", "s2", "s2", "s1"]
        with pytest.raises(SaValueError, match="must not contain NA"):
            align_by_subject(ids, self.GROUP3, ["t0", "t1", "t2"])
