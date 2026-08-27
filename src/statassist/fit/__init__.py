"""Supervised model fitting: one outcome, a set of predictors, one contract.

The port of the ``fit_*`` family. Every function here answers with a
:class:`~statassist.core.result.SaModel`, so a linear regression and a random
forest are read the same way even though they have nothing else in common: the
same ``design`` describing what was seen, the same ``parameters`` recording what
was chosen, the same ``terms`` row order, and ``performance`` scored by the same
resampling scheme.

STATassist does not implement the models. The engine is ``scikit-learn``, which
stands where R stands on ``glmnet``, ``randomForest`` and ``kernlab``; what is
owned here is the input resolution, the coding of a factor predictor, the column
names and the direction rule that decides which class a coefficient describes.
"""

from __future__ import annotations

from .linear_regression import fit_linear_regression
from .logistic_regression import fit_logistic_regression

__all__ = [
    "fit_linear_regression",
    "fit_logistic_regression",
]
