"""How the package reports a refusal, a degraded result and a note.

R has three channels here and they are not interchangeable, so the port keeps
them apart rather than collapsing everything onto ``raise``:

``stop(..., call. = FALSE)``
    The call was wrong and nothing was computed. Mapped to :class:`SaValueError`.
    The message text is carried over as written, because the R messages name the
    argument and say what it should have been, which is most of their value.

``stop("internal error: ...")``
    A table assembled inside the package broke its own contract. Mapped to
    :class:`SaInternalError`, so a caller cannot catch a package bug with the
    same ``except`` clause it uses for its own bad input.

``warning()``
    Something was computed but part of it is missing. Mapped to
    :func:`warn` / :class:`SaWarning`.

``message()``
    Informational; the result is intact. Mapped to :func:`notify`, which logs
    rather than warns, so a scan reporting engine notes for two hundred features
    cannot be turned into an error by a caller's warning filter.
"""

from __future__ import annotations

import logging
import warnings

__all__ = [
    "LOGGER",
    "SaError",
    "SaInternalError",
    "SaValueError",
    "SaWarning",
    "notify",
    "warn",
]

LOGGER = logging.getLogger("statassist")


class SaError(Exception):
    """Base class for every error this package raises on purpose."""


class SaValueError(SaError, ValueError):
    """An argument was not something the function could work with.

    Subclasses :class:`ValueError` as well, so callers that already guard
    against bad input with ``except ValueError`` keep working.
    """


class SaInternalError(SaError, RuntimeError):
    """A result assembled inside the package broke its own contract.

    Raised where R says ``internal error:``. Not a :class:`ValueError`, because
    the caller's input is not what was wrong.
    """


class SaWarning(UserWarning):
    """A result came back with part of it missing."""


def warn(message: str, *, stacklevel: int = 3) -> None:
    """Report a partial result, the counterpart of R's ``warning()``.

    ``stacklevel`` defaults to 3 so the warning is attributed to the caller of
    the helper that emitted it rather than to this module.
    """
    warnings.warn(message, SaWarning, stacklevel=stacklevel)


def notify(message: str) -> None:
    """Report an informational note, the counterpart of R's ``message()``.

    Goes to the ``statassist`` logger rather than to :mod:`warnings`. R's
    ``message()`` does not mark the result as suspect and neither does this.
    """
    LOGGER.info(message)
