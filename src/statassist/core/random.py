"""Seeded randomness for the functions that draw.

This replaces ``sa_preserve_seed()`` rather than translating it. In R the
simulators seed the one global stream and hand back a closure that puts it
back, because there is nothing else to seed; without the restore, calling a
simulator in the middle of a script would silently reset everything drawn after
it. Here each call owns a :class:`numpy.random.Generator` of its own, so the
problem the restore existed to solve does not arise and there is no global
state to put back.

What a seed means is narrower than in R, on purpose:

* The same seed gives the same result **within this package**. That is what
  ``seed=`` promises and it is what the tests check.
* The same seed does **not** reproduce R's numbers. R and NumPy have different
  generators and different algorithms for the same distribution, so nothing
  short of reimplementing R's RNG would make the two agree, and doing that would
  mean writing worse sampling code for no statistical gain. Numerical agreement
  with R is required of the deterministic computations, not of the draws.
"""

from __future__ import annotations

import numpy as np

from .validate import check_count

__all__ = ["SaRandom"]


class SaRandom:
    """A seeded generator, owned by the call that made it.

    Args:
        seed: A non-negative whole number, or ``None`` to draw from the operating
            system's entropy. A :class:`numpy.random.Generator` is accepted as
            well, so a caller running many simulations can thread one stream
            through them.

            R validates its seed with ``sa_check_scalar_num()`` only, so
            ``set.seed(1.5)`` is accepted there and truncated. NumPy cannot seed
            from a fraction, so ``check_count`` is used instead and ``1.5`` is
            refused by name rather than silently becoming ``1``.

    Attributes:
        seed: The seed as given, or ``None``. Recorded so a result object can
            report what it was made with.
        rng: The generator to draw from.
    """

    __slots__ = ("rng", "seed")

    def __init__(self, seed: int | np.random.Generator | None = None) -> None:
        if isinstance(seed, np.random.Generator):
            self.seed: int | None = None
            self.rng: np.random.Generator = seed
            return
        self.seed = None if seed is None else check_count(seed, "seed")
        self.rng = np.random.default_rng(self.seed)

    def __repr__(self) -> str:
        return f"SaRandom(seed={self.seed!r})"
