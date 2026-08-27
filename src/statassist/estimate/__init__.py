"""Reading a verdict out of a comparison."""

from __future__ import annotations

from .categorical_significance import estimate_categorical_significance
from .significance import estimate_significance

__all__ = ["estimate_categorical_significance", "estimate_significance"]
