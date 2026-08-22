"""A drawing test needs a device, and it must not be a window.

Agg is chosen before :mod:`matplotlib.pyplot` is imported anywhere, so a test run
on a machine with a display does not open one, and every figure is closed after
each test, since the ``draw_*`` functions draw on the current figure and would
otherwise inherit whatever the last test left on it.
"""

from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def _clean_figure():
    import matplotlib.pyplot as plt

    plt.close("all")
    yield
    plt.close("all")
