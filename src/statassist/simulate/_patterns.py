"""How a planted effect is shaped across several levels, and how many get each shape.

The three shape helpers of ``R/simulate_multiple_groups.R``. They are here rather
than beside the simulator because the crossed and categorical designs weigh their
own catalogues of shapes with the same machinery, so the catalogue and the
argument name are parameters instead of a second copy of the code.

Two of the three deliberately do not draw. Which features are planted moves with
the seed; how many of them take each shape does not, because that is the kind of
thing about a simulation that should not have to be looked up.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from ..core.errors import SaInternalError, SaValueError

__all__ = ["PATTERNS", "allocate", "pattern_delta", "pattern_mix", "pick_up_down"]

#: The shapes an effect over ordered treatment levels can take.
PATTERNS = ("all", "gradient", "single")


def pattern_mix(
    mix: Mapping[str, float] | None,
    known: Sequence[str] = PATTERNS,
    arg: str = "pattern_mix",
) -> dict[str, float]:
    """Check the shape weights and drop the ones set to zero.

    Port of ``sa_sim_pattern_mix()``. R takes a named numeric vector; a mapping
    is what carries the same two things in Python, and it rules out the repeated
    name R has to check for.

    Args:
        mix: Relative weights, keyed by shape name.
        known: The shape names the caller accepts.
        arg: Argument name to name in the error.

    Returns:
        The weights that are above zero, in the order they were given.
    """
    if (
        not isinstance(mix, Mapping)
        or len(mix) == 0
        or not all(isinstance(name, str) and _is_weight(weight) for name, weight in mix.items())
    ):
        raise SaValueError(
            f"`{arg}` must be a mapping from shape name to a non-missing numeric "
            "weight, with one entry per shape. Known shapes are: " + ", ".join(known) + "."
        )

    unknown = [name for name in mix if name not in known]
    if unknown:
        raise SaValueError(
            f"`{arg}` names unknown shape(s): "
            + ", ".join(unknown)
            + ". Known shapes are: "
            + ", ".join(known)
            + "."
        )
    weights = {name: float(weight) for name, weight in mix.items()}
    if any(weight < 0 for weight in weights.values()):
        raise SaValueError(f"`{arg}` weights must not be negative.")
    if sum(weights.values()) <= 0:
        raise SaValueError(
            f"`{arg}` needs at least one positive weight, otherwise there is no "
            "shape left to plant an effect in."
        )
    return {name: weight for name, weight in weights.items() if weight > 0}


def _is_weight(value: object) -> bool:
    """Whether a value is a single non-missing number, as R's weights must be."""
    if isinstance(value, bool | np.bool_):
        return False
    if not isinstance(value, int | float | np.integer | np.floating):
        return False
    return not math.isnan(float(value))


def allocate(n: int, weights: Mapping[str, float]) -> dict[str, int]:
    """Split ``n`` between weighted shapes without drawing lots.

    Port of ``sa_sim_allocate()``. The largest remainder method rather than a
    multinomial draw, so that the counts are exactly the proportions the weights
    ask for and are a function of the arguments alone.

    Returns:
        One count per shape, in the order the weights were given, summing to
        ``n`` exactly.
    """
    out = dict.fromkeys(weights, 0)
    if n == 0:
        return out

    total = sum(weights.values())
    share = {name: n * weight / total for name, weight in weights.items()}
    out = {name: int(math.floor(value)) for name, value in share.items()}
    short = n - sum(out.values())
    if short > 0:
        # Sorting is stable in both languages, so an even mix hands the remainder
        # to the earlier shapes rather than to an arbitrary one.
        order = sorted(out, key=lambda name: -(share[name] - out[name]))
        for name in order[:short]:
            out[name] += 1
    return out


def pick_up_down(
    n: int,
    n_up: int,
    n_down: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose which items are moved up and which are moved down.

    R spells this out at each of the three places that plant a signed effect,
    with the same comment each time; here it is one function, so the trap the
    comment is about cannot be reintroduced at one of the three. That trap is
    taking the down set as the complement of the up set: with an empty up set the
    complement is everything rather than nothing.

    Returns:
        The chosen positions, up set first, taken from the head and the tail of
        one shuffled draw so that the two sets cannot overlap.
    """
    picked = rng.permutation(n)[: n_up + n_down]
    return picked[:n_up], picked[n_up:]


def pattern_delta(
    d: float,
    pattern: str,
    n_groups: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Spread one magnitude over the treatment groups according to its shape.

    Port of ``sa_sim_pattern_delta()``.

    Args:
        d: Signed magnitude of the effect, on the log2 scale.
        pattern: ``"all"``, ``"gradient"`` or ``"single"``.
        n_groups: Number of treatment groups.
        rng: The generator ``"single"`` picks its one group from. The other two
            shapes do not draw.

    Returns:
        One delta per treatment group.
    """
    if pattern == "all":
        return np.full(n_groups, float(d))
    if pattern == "gradient":
        return float(d) * np.arange(1, n_groups + 1) / n_groups
    if pattern == "single":
        out = np.zeros(n_groups)
        out[int(rng.integers(n_groups))] = float(d)
        return out
    raise SaInternalError(f"internal error: unknown effect shape `{pattern}`.")
