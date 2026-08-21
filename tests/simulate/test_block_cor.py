"""Block correlation matrices.

Everything here is deterministic, so it is checked against values worked out by
hand rather than against a property. The rejections carry as much of the point as
the matrices do: the whole reason this function exists instead of a literal
matrix is that it can say which block cannot hold.
"""

from __future__ import annotations

import numpy as np
import pytest

from statassist import make_block_cor
from statassist.core.errors import SaValueError


def test_no_blocks_gives_the_default_off_the_diagonal() -> None:
    cor_mat = make_block_cor(3, default_cor=0.2)
    assert cor_mat.tolist() == [
        [1.0, 0.2, 0.2],
        [0.2, 1.0, 0.2],
        [0.2, 0.2, 1.0],
    ]


def test_a_block_holds_one_value_and_leaves_the_rest_alone() -> None:
    cor_mat = make_block_cor(4, [{"features": [0, 1], "cor": 0.8}])
    assert cor_mat[0, 1] == 0.8
    assert cor_mat[0, 2] == 0.0
    assert cor_mat[2, 3] == 0.0
    assert np.diag(cor_mat).tolist() == [1.0] * 4


def test_two_blocks_do_not_reach_across_each_other() -> None:
    cor_mat = make_block_cor(
        6, [{"features": [0, 1], "cor": 0.8}, {"features": [2, 3, 4], "cor": 0.5}]
    )
    assert cor_mat[0, 1] == 0.8
    assert cor_mat[2, 3] == cor_mat[2, 4] == cor_mat[3, 4] == 0.5
    assert cor_mat[1, 2] == 0.0
    assert cor_mat[4, 5] == 0.0


def test_against_gives_the_sign_pattern_of_an_outer_product() -> None:
    """One factor with a sign per predictor, which is what lifts the -1/(k-1) bound."""
    cor_mat = make_block_cor(6, [{"features": range(3), "cor": 0.9, "against": range(3, 6)}])
    signs = np.array([1, 1, 1, -1, -1, -1])
    expected = 0.9 * np.outer(signs, signs)
    np.fill_diagonal(expected, 1.0)
    assert cor_mat.tolist() == expected.tolist()


def test_a_split_block_survives_a_correlation_no_shared_one_could_hold() -> None:
    cor_mat = make_block_cor(6, [{"features": range(3), "cor": 0.9, "against": range(3, 6)}])
    assert np.linalg.eigvalsh(cor_mat).min() > 0


def test_the_result_is_a_correlation_matrix() -> None:
    cor_mat = make_block_cor(5, [{"features": [0, 1, 2], "cor": -0.4}], default_cor=0.1)
    assert cor_mat.tolist() == cor_mat.T.tolist()
    assert np.linalg.eigvalsh(cor_mat).min() > 0


def test_indices_are_zero_based() -> None:
    """R writes ``features = 1:2``; here the first two predictors are 0 and 1."""
    cor_mat = make_block_cor(3, [{"features": [0, 1], "cor": 0.5}])
    assert cor_mat[0, 1] == 0.5
    assert cor_mat[1, 2] == 0.0


def test_a_shared_value_below_the_block_bound_is_refused_by_name() -> None:
    with pytest.raises(SaValueError, match=r"holds only above -0\.5"):
        make_block_cor(6, [{"features": range(3), "cor": -0.6}])


def test_the_bound_depends_on_the_size_of_the_block() -> None:
    """Two predictors may disagree at -0.6; three may not."""
    assert make_block_cor(6, [{"features": [0, 1], "cor": -0.6}])[0, 1] == -0.6


def test_a_shared_value_of_one_is_refused_as_a_repeated_variable() -> None:
    with pytest.raises(SaValueError, match="perfect agreement"):
        make_block_cor(4, [{"features": [0, 1], "cor": 1}])


def test_a_default_no_matrix_of_that_size_could_hold_is_refused() -> None:
    with pytest.raises(SaValueError, match=r"`default_cor` of -0\.5 is not possible"):
        make_block_cor(4, default_cor=-0.5)


def test_blocks_that_cannot_meet_are_refused_with_the_eigenvalue() -> None:
    """Every block holds on its own here; what does not is `default_cor` beside them."""
    assert make_block_cor(4, [{"features": [0, 1], "cor": -0.9}]) is not None
    assert make_block_cor(4, default_cor=0.5) is not None
    with pytest.raises(SaValueError, match="smallest eigenvalue"):
        make_block_cor(4, [{"features": [0, 1], "cor": -0.9}], default_cor=0.5)


def test_overlapping_blocks_are_refused_rather_than_letting_the_later_one_win() -> None:
    with pytest.raises(SaValueError, match="overlaps an earlier block at predictor"):
        make_block_cor(6, [{"features": [0, 1, 2], "cor": 0.8}, {"features": [2, 3], "cor": 0.4}])


def test_a_predictor_cannot_move_against_itself() -> None:
    with pytest.raises(SaValueError, match="in both `features` and `against`"):
        make_block_cor(6, [{"features": [0, 1], "cor": 0.8, "against": [1, 2]}])


def test_against_refuses_a_negative_correlation_written_backwards() -> None:
    with pytest.raises(SaValueError, match="must be above 0 when `against` is given"):
        make_block_cor(6, [{"features": [0, 1], "cor": -0.8, "against": [2, 3]}])


def test_an_unknown_name_in_a_block_is_refused() -> None:
    with pytest.raises(SaValueError, match="which a block has no use for"):
        make_block_cor(6, [{"features": [0, 1], "cor": 0.8, "agianst": [2, 3]}])


def test_an_index_outside_the_matrix_is_refused() -> None:
    with pytest.raises(SaValueError, match="indexes predictor"):
        make_block_cor(3, [{"features": [2, 3], "cor": 0.8}])


def test_an_unsplit_block_needs_two_predictors() -> None:
    with pytest.raises(SaValueError, match="at least two distinct whole numbers"):
        make_block_cor(3, [{"features": [0], "cor": 0.8}])


def test_a_side_of_a_split_block_needs_only_one() -> None:
    cor_mat = make_block_cor(3, [{"features": [0], "cor": 0.8, "against": [1]}])
    assert cor_mat[0, 1] == -0.8


def test_repeated_indices_in_a_block_are_refused() -> None:
    with pytest.raises(SaValueError, match="distinct whole numbers"):
        make_block_cor(4, [{"features": [0, 1, 1], "cor": 0.8}])


def test_a_fractional_index_is_refused() -> None:
    with pytest.raises(SaValueError, match="whole numbers"):
        make_block_cor(4, [{"features": [0, 1.5], "cor": 0.8}])


def test_a_block_without_cor_is_refused() -> None:
    with pytest.raises(SaValueError, match="must be a mapping with `features` and `cor`"):
        make_block_cor(4, [{"features": [0, 1]}])


def test_blocks_must_be_a_sequence_of_mappings() -> None:
    with pytest.raises(SaValueError, match="must be a sequence of blocks"):
        make_block_cor(4, {"features": [0, 1], "cor": 0.5})
