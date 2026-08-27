"""Results built by hand, for states the public functions refuse to produce.

:func:`~statassist.compare_categorical_groups` will not run a test on a table
with an empty margin, so a cell with nothing expected and a level that was named
and never seen are unreachable through it. Both are states the layers above it
have a rule for - a lift that does not exist as against one of zero, a strip that
is reported rather than drawn - and a rule that cannot be reached cannot be
checked, so the result is assembled here instead.

Assembled through ``new_categorical`` rather than around it, so the contract the
builder enforces is still what these results satisfy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from statassist.core.contracts import categorical_cell_columns
from statassist.core.result import SaCategorical, new_categorical


def crafted_categorical() -> SaCategorical:
    """A 2x2 result whose second row holds nothing.

    Row ``b`` was named and never seen, which leaves its two cells with no
    observation and one of them - the one whose column is also empty - with
    nothing expected either. Cell ``(a, q)`` is the other case: nothing observed
    where something was expected, so its lift is zero rather than missing.
    """
    cells = pd.DataFrame(
        {
            "row_level": ["a", "b", "a", "b"],
            "col_level": ["p", "p", "q", "q"],
            "observed": [10.0, 0.0, 0.0, 0.0],
            "expected": [10.0, 0.0, 5.0, 5.0],
            "residual": [0.0, np.nan, -np.sqrt(5.0), -np.sqrt(5.0)],
            "std_residual": [0.0, np.nan, -1.0, -1.0],
            "prop_total": [1.0, 0.0, 0.0, 0.0],
            "prop_row": [1.0, np.nan, 0.0, np.nan],
            "prop_col": [1.0, 0.0, np.nan, np.nan],
        }
    )[list(categorical_cell_columns())]
    test = pd.DataFrame(
        {
            "n_used": [10.0],
            "statistic": [0.0],
            "df": [1.0],
            "pval": [1.0],
            "lower_conf": [np.nan],
            "upper_conf": [np.nan],
        }
    )
    return new_categorical(
        analysis="categorical_comparison",
        variables=["v1", "v2"],
        design={
            "category_lv": {"v1": ["a", "b"], "v2": ["p", "q"]},
            "null": "independence",
            "paired": False,
            "pairing": None,
            "dim": [2, 2],
            "row_var": "v1",
            "col_var": "v2",
            "n_used": 10,
            "n_dropped": 0,
            "n_incomplete": 0,
        },
        parameters={"conf_level": 0.95, "correct": True},
        cells=cells,
        tests={"chisq_test": test},
        test_info={
            "chisq_test": {
                "id": "chisq_independence",
                "label": "Chi-square test of independence",
            }
        },
        association=pd.DataFrame(
            {
                "measure": ["cramers_v"],
                "estimate": [0.0],
                "lower_conf": [np.nan],
                "upper_conf": [np.nan],
            }
        ),
    )
