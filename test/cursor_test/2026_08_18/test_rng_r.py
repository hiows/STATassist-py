"""R-compatible RNG parity tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from statassist.utils.rng_r import RRandom

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture
def rng_golden() -> dict:
    with open(FIXTURES / "rng_stream.json", encoding="utf-8") as f:
        return json.load(f)


def test_runif_rnorm_sample_int_match_r(rng_golden: dict) -> None:
    r = RRandom(2026)
    got_u = r.runif(5)
    np.testing.assert_allclose(got_u, rng_golden["runif_5"], rtol=0, atol=1e-15)

    r = RRandom(2027)
    got_n = r.rnorm(5)
    np.testing.assert_allclose(got_n, rng_golden["rnorm_5"], rtol=1e-12, atol=1e-12)

    r = RRandom(2028)
    got_s = r.sample_int(20, 8)
    assert got_s.tolist() == rng_golden["sample_int"]


def test_sample_prob_rbinom_match_r(rng_golden: dict) -> None:
    r = RRandom(2029)
    probs = np.array([0.1, 0.1, 0.2, 0.2, 0.2, 0.2])
    got_sp = r.sample_prob_replace(probs, 30)
    assert got_sp.tolist() == rng_golden["sample_prob"]

    r = RRandom(2030)
    got_b = r.rbinom(10, 1, 0.3)
    assert got_b.tolist() == rng_golden["rbinom_10"]
