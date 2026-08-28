"""The ``evaluate_*`` functions: fitted models in, a held-out score out.

Port of ``R/evaluate_regression_models.R`` and
``R/evaluate_classification_models.R``. What a ``fit_*`` result already carries
in ``performance`` is a resampled score, measured inside the folds of the data
the model was fitted to. These two are the other kind: measured once, on rows the
caller held back, and across several models at once so that they can be compared.

Both take the intersection of the rows every model could predict rather than
letting each model score whatever it managed, which is what makes the paired
tests on the classification side mean anything.
"""

from __future__ import annotations

from .classification_models import evaluate_classification_models
from .regression_models import evaluate_regression_models

__all__ = [
    "evaluate_classification_models",
    "evaluate_regression_models",
]
