"""Shared pytest fixtures for statassist Python port."""

import pytest


@pytest.fixture
def matplotlib_use_agg():
    import matplotlib

    matplotlib.use("Agg", force=True)
