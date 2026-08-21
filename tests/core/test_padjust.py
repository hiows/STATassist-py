"""Multiplicity adjustment against R's ``stats::p.adjust``.

The expected values are derived from R's algorithm by hand rather than captured
from a run, so they document the arithmetic as well as pinning it. ``p`` below is
the Benjamini-Hochberg worked example, whose BH result is widely published.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statassist.core import add_padj, p_adjust
from statassist.core.errors import SaInternalError, SaValueError

# The ten p-values of the classic BH example, already sorted.
P = [0.001, 0.008, 0.039, 0.041, 0.042, 0.060, 0.074, 0.205, 0.212, 0.216]


def test_bh_matches_r() -> None:
    expected = [0.01, 0.04, 0.084, 0.084, 0.084, 0.1, 0.10571428571428572, 0.216, 0.216, 0.216]
    assert p_adjust(P, "BH") == pytest.approx(expected, rel=1e-12)


def test_fdr_is_an_alias_of_bh() -> None:
    assert p_adjust(P, "fdr") == pytest.approx(p_adjust(P, "BH"), rel=0)


def test_holm_matches_r() -> None:
    expected = [0.01, 0.072, 0.312, 0.312, 0.312, 0.312, 0.312, 0.615, 0.615, 0.615]
    assert p_adjust(P, "holm") == pytest.approx(expected, rel=1e-12)


def test_hochberg_matches_r() -> None:
    expected = [0.01, 0.072, 0.216, 0.216, 0.216, 0.216, 0.216, 0.216, 0.216, 0.216]
    assert p_adjust(P, "hochberg") == pytest.approx(expected, rel=1e-12)


def test_bonferroni_caps_at_one() -> None:
    expected = [0.01, 0.08, 0.39, 0.41, 0.42, 0.6, 0.74, 1.0, 1.0, 1.0]
    assert p_adjust(P, "bonferroni") == pytest.approx(expected, rel=1e-12)


def test_hommel_matches_r() -> None:
    # Hand-traced through R's loop for n = 4.
    assert p_adjust([0.001, 0.008, 0.039, 0.041], "hommel") == pytest.approx(
        [0.004, 0.024, 0.041, 0.041], rel=1e-12
    )


def test_by_matches_r() -> None:
    harmonic = sum(1.0 / k for k in range(1, 11))
    raw = [0.01, 0.04, 0.084, 0.084, 0.084, 0.1, 0.10571428571428572, 0.216, 0.216, 0.216]
    expected = [min(1.0, harmonic * value) for value in raw]
    assert p_adjust(P, "BY") == pytest.approx(expected, rel=1e-12)


def test_none_returns_the_input_untouched() -> None:
    assert p_adjust(P, "none") == pytest.approx(P, rel=0)


def test_n_counts_only_the_present_p_values() -> None:
    """R's ``n = length(p)`` default is forced after the missing ones are dropped.

    Adjusting three cells of which one is missing is adjusting against two, not
    three. With n = 3 the answer would be 0.03; the test tables in this package
    routinely carry a missing p-value for a feature that failed, so this rule
    decides the numbers everywhere.
    """
    out = p_adjust([0.01, float("nan"), 0.02], "BH")
    assert out[0] == pytest.approx(0.02, rel=1e-12)
    assert out[2] == pytest.approx(0.02, rel=1e-12)
    assert np.isnan(out[1])


def test_a_family_of_one_is_returned_unchanged() -> None:
    """R returns early for ``n <= 1``, before any method runs."""
    out = p_adjust([0.5, float("nan"), float("nan")], "bonferroni")
    assert out[0] == pytest.approx(0.5, rel=0)
    assert np.isnan(out[1:]).all()


def test_unknown_method_is_refused() -> None:
    with pytest.raises(SaValueError, match="must be one of"):
        p_adjust(P, "benjamini")


def test_n_below_the_number_of_p_values_is_a_contract_breach() -> None:
    with pytest.raises(SaInternalError):
        p_adjust([0.1, 0.2, 0.3], "BH", n=2)


def test_add_padj_puts_the_column_straight_after_pval() -> None:
    df = pd.DataFrame(
        {
            "features": ["g1", "g2"],
            "n_used": [10, 10],
            "pval": [0.01, 0.5],
            "lower_conf": [0.0, 0.0],
            "upper_conf": [1.0, 1.0],
        }
    )
    out = add_padj(df, "BH")
    assert list(out.columns) == [
        "features",
        "n_used",
        "pval",
        "pval_adj",
        "lower_conf",
        "upper_conf",
    ]
    assert out["pval_adj"].tolist() == pytest.approx([0.02, 0.5], rel=1e-12)


def test_add_padj_without_a_pval_column_is_a_contract_breach() -> None:
    with pytest.raises(SaInternalError):
        add_padj(pd.DataFrame({"features": ["g1"], "estimate": [1.0]}), "BH")
