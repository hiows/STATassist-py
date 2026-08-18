"""Shared base for dict-shaped result contracts."""

from __future__ import annotations

from typing import Any, Callable


class SaResult(dict[str, Any]):
    """A result dict that prints the way R's ``print.sa_*`` does.

    Ordinary dicts have no place to hang a summary printer, and the JSON
    contract stores everything as named slots anyway, so subclassing the dict
    keeps ``res["parameters"]`` working while ``repr(res)`` reads like the R
    console.
    """

    _repr_fn: Callable[[Any], str] | None = None

    def __repr__(self) -> str:
        if self._repr_fn is not None:
            return self._repr_fn(self)
        return dict.__repr__(self)

    def __str__(self) -> str:
        return self.__repr__()


def _sa_result(data: dict[str, Any], repr_fn: Callable[[Any], str]) -> SaResult:
    out = SaResult(data)
    out._repr_fn = repr_fn  # type: ignore[attr-defined]
    return out
