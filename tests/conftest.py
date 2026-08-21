"""Makes ``golden`` importable from every test, however deep it sits.

``tests/`` is not a package, so pytest puts the directory of each test file on
the path rather than this one, and ``tests/kernel/test_robust.py`` would not find
``tests/golden.py``. Adding it here is one line against turning the whole tree
into packages for the sake of a single shared helper module.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
