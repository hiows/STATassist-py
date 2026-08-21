"""Seeded randomness.

There is no R comparison here, and there is not meant to be: a seed promises
reproducibility within this package, not agreement with R's stream. See
:mod:`statassist.core.random`.
"""

from __future__ import annotations

import numpy as np
import pytest

from statassist.core import SaRandom
from statassist.core.errors import SaValueError


def test_the_same_seed_gives_the_same_draw() -> None:
    first = SaRandom(42).rng.normal(size=5)
    second = SaRandom(42).rng.normal(size=5)
    assert first.tolist() == second.tolist()


def test_a_different_seed_gives_a_different_draw() -> None:
    assert SaRandom(1).rng.normal(size=5).tolist() != SaRandom(2).rng.normal(size=5).tolist()


def test_no_seed_still_draws() -> None:
    unseeded = SaRandom()
    assert unseeded.seed is None
    assert unseeded.rng.normal(size=3).shape == (3,)


def test_the_seed_is_recorded_for_the_result_object() -> None:
    assert SaRandom(7).seed == 7


def test_it_does_not_touch_the_global_stream() -> None:
    """The whole reason ``sa_preserve_seed`` exists in R does not arise here."""
    np.random.seed(123)
    before = np.random.rand()
    np.random.seed(123)
    SaRandom(999).rng.normal(size=100)
    assert np.random.rand() == before


def test_an_existing_generator_is_threaded_through() -> None:
    """A caller running many simulations shares one stream rather than reseeding."""
    shared = np.random.default_rng(5)
    first = SaRandom(shared).rng.normal(size=3)
    second = SaRandom(shared).rng.normal(size=3)
    assert first.tolist() != second.tolist()


def test_a_fractional_seed_is_refused() -> None:
    """R would accept and truncate it; NumPy cannot seed from a fraction at all."""
    with pytest.raises(SaValueError, match="finite whole number"):
        SaRandom(1.5)


def test_a_negative_seed_is_refused() -> None:
    with pytest.raises(SaValueError, match=r"must be in \[0, Inf\]"):
        SaRandom(-1)
