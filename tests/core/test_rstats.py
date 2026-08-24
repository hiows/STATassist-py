"""``r_mean`` against the rounding R's ``mean()`` aims at."""

from __future__ import annotations

import math

import numpy as np
import pytest

from statassist.core.rstats import r_mean

# Taken from the term-sign probe: prot_1 / control.female after undoing log2 and
# taking log(). numpy.mean and the compensated sum disagree by one ULP here, and
# that ULP is what flipped the sex term's log2_effect sign against R.
_DIVERGING_LOGS = [
    6.763445997559917,
    10.257367344162752,
    9.734064096696953,
    6.087673937375068,
    8.06145628173343,
    6.892694379730207,
    10.322131550978371,
    4.763522993581098,
]


class TestRMean:
    def test_disagrees_with_numpy_on_a_known_vector(self):
        produced = r_mean(_DIVERGING_LOGS)
        numpy_mean = float(np.mean(_DIVERGING_LOGS))
        assert produced != numpy_mean

    def test_matches_compensated_fsum(self):
        values = np.asarray(_DIVERGING_LOGS, dtype=float)
        centre = math.fsum(values.tolist()) / values.size
        centre += math.fsum((values - centre).tolist()) / values.size
        assert r_mean(values) == pytest.approx(centre, abs=0.0)

    def test_empty_is_missing(self):
        assert math.isnan(r_mean([]))

    def test_a_singleton_is_itself(self):
        assert r_mean([3.5]) == 3.5
