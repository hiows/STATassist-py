"""The searches that answer which predictors are worth keeping.

Both are handed candidates rather than predictors and both come back with
:class:`~statassist.core.SaSelection`, whose row axis is ``candidates``. What
differs is how they decide.

:func:`perform_rfe` chooses by a resampled score: the elimination runs inside the
resampling, so every subset size is scored on rows that did not rank it.
:func:`perform_stepwise` chooses by a penalised likelihood computed on the rows
the model was fitted to, which is why it resamples nothing and why its
``resampling`` slot is ``None``.

Both answer a question :func:`~statassist.fit_elastic_net` answers from the other
end, by shrinking a coefficient to exactly zero rather than by dropping a column.
"""

from __future__ import annotations

from .rfe import perform_rfe
from .stepwise import perform_stepwise

__all__ = ["perform_rfe", "perform_stepwise"]
