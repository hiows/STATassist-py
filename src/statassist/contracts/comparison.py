"""Comparison result contract (sa_comparison / sa_two_group)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from statassist.utils.metadata import sa_metadata


def TEST_TABLE_COLUMNS() -> list[str]:
    """Column names every test table in a comparison result must carry."""
    return sa_test_table_columns()


def sa_test_table_columns() -> list[str]:
    return ["features", "n_used", "pval", "pval_adj", "lower_conf", "upper_conf"]


def sa_posthoc_stat_columns() -> list[str]:
    return [
        c
        for c in sa_posthoc_table_columns()
        if c not in ("features", "contrast", "group1", "group2")
    ]


def sa_posthoc_table_columns() -> list[str]:
    return [
        "features",
        "contrast",
        "group1",
        "group2",
        "n1",
        "n2",
        "estimate",
        "stderr",
        "statistic",
        "df",
        "pval",
        "pval_adj",
        "lower_conf",
        "upper_conf",
    ]


def sa_pairwise_table_columns() -> list[str]:
    stat_cols = [
        c
        for c in sa_posthoc_table_columns()
        if c not in ("features", "contrast", "group1", "group2")
    ]
    return [
        "features",
        "contrast",
        "group1",
        "group2",
        "fold_change",
        "log2fc",
        *stat_cols,
    ]


class sa_result:
    """Marker base class matching the R S3 sa_result type."""


class sa_comparison(sa_result):
    """Marker class for comparison results."""


class sa_two_group(sa_comparison):
    """Marker class for two-group comparison results."""


class sa_multi_group(sa_comparison):
    """Marker class for multi-group comparison results."""


class sa_one_sample(sa_comparison):
    """Marker class for one-sample comparison results."""


class sa_factorial(sa_comparison):
    """Marker class for factorial comparison results."""


@dataclass
class ComparisonResult:
    """Structured comparison result matching the R sa_comparison contract."""

    analysis: str
    features: list[str]
    design: dict[str, Any]
    parameters: dict[str, Any]
    effect: pd.DataFrame
    tests: dict[str, pd.DataFrame]
    test_info: dict[str, dict[str, Any]]
    diagnostics: dict[str, Any] | None = None
    metadata: dict[str, str] = field(default_factory=sa_metadata)
    terms: pd.DataFrame | None = None
    cells: pd.DataFrame | None = None
    posthoc: dict[str, pd.DataFrame] | None = None
    pairwise: dict[str, dict[str, pd.DataFrame]] | None = None
    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "analysis": self.analysis,
            "features": list(self.features),
            "design": dict(self.design),
            "parameters": dict(self.parameters),
            "effect": self.effect.copy(),
            "tests": {k: v.copy() for k, v in self.tests.items()},
            "test_info": {k: dict(v) for k, v in self.test_info.items()},
            "metadata": dict(self.metadata),
        }
        if self.diagnostics is not None:
            out["diagnostics"] = self.diagnostics
        if self.terms is not None:
            out["terms"] = self.terms.copy()
        if self.cells is not None:
            out["cells"] = self.cells.copy()
        if self.posthoc is not None:
            out["posthoc"] = {k: v.copy() for k, v in self.posthoc.items()}
        if self.pairwise is not None:
            out["pairwise"] = {
                test: {ct: tbl.copy() for ct, tbl in blocks.items()}
                for test, blocks in self.pairwise.items()
            }
        return out


def sa_new_comparison(
    *,
    analysis: str,
    features: list[str],
    design: dict[str, Any],
    parameters: dict[str, Any],
    effect: pd.DataFrame,
    tests: dict[str, pd.DataFrame],
    test_info: dict[str, dict[str, Any]],
    terms: pd.DataFrame | None = None,
    cells: pd.DataFrame | None = None,
    posthoc: dict[str, pd.DataFrame] | None = None,
    pairwise: dict[str, dict[str, pd.DataFrame]] | None = None,
    diagnostics: dict[str, Any] | None = None,
    subclass: type = sa_two_group,
) -> ComparisonResult:
    if not tests or not isinstance(tests, dict):
        raise RuntimeError("internal error: `tests` must be a non-empty named list.")
    if set(tests) != set(test_info):
        raise RuntimeError(
            "internal error: `tests` and `test_info` name different tests: "
            f"{', '.join(tests)} vs {', '.join(test_info)}."
        )
    if posthoc is not None and not set(posthoc).issubset(set(tests)):
        unknown = sorted(set(posthoc) - set(tests))
        raise RuntimeError(
            f"internal error: `posthoc` names a test that was not run: "
            f"{', '.join(unknown)}."
        )
    if pairwise is not None and not set(pairwise).issubset(set(tests)):
        unknown = sorted(set(pairwise) - set(tests))
        raise RuntimeError(
            f"internal error: `pairwise` names a test that was not run: "
            f"{', '.join(unknown)}."
        )

    def _check_table(df: pd.DataFrame, what: str) -> None:
        if not isinstance(df, pd.DataFrame):
            raise RuntimeError(f"internal error: {what} must be a data.frame.")
        if not df["features"].tolist() == list(features):
            raise RuntimeError(
                f"internal error: {what} is not aligned with `features`."
            )

    _check_table(effect, "`effect`")
    contract_cols = sa_test_table_columns()
    for nm, tbl in tests.items():
        _check_table(tbl, f"`tests${nm}`")
        absent = [c for c in contract_cols if c not in tbl.columns]
        if absent:
            raise RuntimeError(
                f"internal error: `tests${nm}` is missing contract column(s): "
                f"{', '.join(absent)}."
            )

    if terms is not None:
        if not isinstance(terms, pd.DataFrame):
            raise RuntimeError("internal error: `terms` must be a data.frame.")
        unknown_feats = set(terms["features"]) - set(features)
        if unknown_feats:
            raise RuntimeError(
                "internal error: `terms` holds feature(s) absent from the "
                f"comparison: {', '.join(sorted(unknown_feats))}."
            )

    if cells is not None:
        if not isinstance(cells, pd.DataFrame):
            raise RuntimeError("internal error: `cells` must be a data.frame.")
        if list(cells["features"].unique()) != list(features):
            raise RuntimeError(
                "internal error: `cells` does not hold every feature of the "
                "comparison once, in order."
            )

    if posthoc:
        for nm, tbl in posthoc.items():
            if not isinstance(tbl, pd.DataFrame):
                raise RuntimeError(
                    f"internal error: `posthoc${nm}` must be a data.frame."
                )
            absent = [c for c in sa_posthoc_table_columns() if c not in tbl.columns]
            if absent:
                raise RuntimeError(
                    f"internal error: `posthoc${nm}` is missing contract "
                    f"column(s): {', '.join(absent)}."
                )
            unknown_feats = set(tbl["features"]) - set(features)
            if unknown_feats:
                raise RuntimeError(
                    f"internal error: `posthoc${nm}` holds feature(s) absent "
                    f"from the comparison: {', '.join(sorted(unknown_feats))}."
                )

    if pairwise:
        pw_cols = sa_pairwise_table_columns()
        for nm, blocks in pairwise.items():
            if not isinstance(blocks, dict):
                raise RuntimeError(
                    f"internal error: `pairwise${nm}` must be a list named by "
                    "contrast."
                )
            for ct, tbl in blocks.items():
                _check_table(tbl, f'`pairwise${nm}[["{ct}"]]`')
                absent = [c for c in pw_cols if c not in tbl.columns]
                if absent:
                    raise RuntimeError(
                        f"internal error: `pairwise${nm}[['{ct}']]` is missing "
                        f"contract column(s): {', '.join(absent)}."
                    )

    obj = ComparisonResult(
        analysis=analysis,
        features=list(features),
        design=design,
        parameters=parameters,
        effect=effect,
        tests=tests,
        test_info=test_info,
        terms=terms,
        cells=cells,
        posthoc=posthoc if posthoc else None,
        pairwise=pairwise if pairwise else None,
        diagnostics=diagnostics,
        metadata=sa_metadata(),
    )
    tagged_cls = type(subclass.__name__, (subclass, ComparisonResult), {})
    obj.__class__ = tagged_cls
    return obj


def sa_pick_test(res: ComparisonResult, test: str, *, arg: str = "comparison_result") -> pd.DataFrame:
    if not isinstance(res, ComparisonResult):
        raise ValueError(
            f"`{arg}` must be a comparison result, as returned by "
            "compare_two_groups()."
        )
    if not isinstance(test, str) or not test:
        raise ValueError("`test` must be a single test name.")
    if test not in res.tests:
        valid = ", ".join(res.tests)
        raise ValueError(
            f"`test` must name one of the tests in `{arg}`: {valid}. Got {test}."
        )
    return res.tests[test]
