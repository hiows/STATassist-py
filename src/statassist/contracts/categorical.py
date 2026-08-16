"""Categorical comparison result contract (sa_categorical)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from statassist.utils.metadata import sa_metadata


CATEGORICAL_NULLS = ("independence", "symmetry", "marginal_homogeneity")


def sa_categorical_cell_columns() -> list[str]:
    return [
        "row_level",
        "col_level",
        "observed",
        "expected",
        "residual",
        "std_residual",
        "prop_total",
        "prop_row",
        "prop_col",
    ]


def sa_categorical_test_columns() -> list[str]:
    return ["n_used", "statistic", "df", "pval", "lower_conf", "upper_conf"]


def sa_association_columns() -> list[str]:
    return ["measure", "estimate", "lower_conf", "upper_conf"]


class sa_result:
    pass


class sa_categorical(sa_result):
    pass


@dataclass
class CategoricalResult:
    analysis: str
    variables: list[str]
    design: dict[str, Any]
    parameters: dict[str, Any]
    cells: pd.DataFrame
    tests: dict[str, pd.DataFrame]
    test_info: dict[str, dict[str, Any]]
    association: pd.DataFrame
    diagnostics: dict[str, Any] | None = None
    metadata: dict[str, str] = field(default_factory=sa_metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis": self.analysis,
            "variables": list(self.variables),
            "design": dict(self.design),
            "parameters": dict(self.parameters),
            "cells": self.cells.copy(),
            "tests": {k: v.copy() for k, v in self.tests.items()},
            "test_info": {k: dict(v) for k, v in self.test_info.items()},
            "association": self.association.copy(),
            "diagnostics": self.diagnostics,
            "metadata": dict(self.metadata),
        }


def sa_new_categorical(
    *,
    analysis: str,
    variables: list[str],
    design: dict[str, Any],
    parameters: dict[str, Any],
    cells: pd.DataFrame,
    tests: dict[str, pd.DataFrame],
    test_info: dict[str, dict[str, Any]],
    association: pd.DataFrame,
    diagnostics: dict[str, Any] | None = None,
) -> CategoricalResult:
    obj = CategoricalResult(
        analysis=analysis,
        variables=list(variables),
        design=design,
        parameters=parameters,
        cells=cells,
        tests=tests,
        test_info=test_info,
        association=association,
        diagnostics=diagnostics,
    )
    obj.__class__ = type("CategoricalResult", (sa_categorical, CategoricalResult), {})
    return obj


def sa_null_label(null: str) -> str:
    labels = {
        "independence": "independence -- a cell is expected at the product of its margins",
        "symmetry": "symmetry -- a cell is expected at the average of it and its transpose",
        "marginal_homogeneity": "marginal homogeneity -- every condition is expected at the pooled rate",
    }
    return labels.get(null, null)
