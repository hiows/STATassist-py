"""Reproducibility metadata attached to every result."""

from __future__ import annotations

import platform
import sys
from datetime import datetime, timezone


def sa_metadata() -> dict:
    now = datetime.now(timezone.utc).astimezone()
    ts = now.strftime("%Y-%m-%dT%H:%M:%S%z")
    if len(ts) > 5 and ts[-5] not in ("+", "-"):
        ts = f"{ts[:-2]}{ts[-2:]}"

    from importlib.metadata import version

    try:
        pkg_ver = version("statassist")
    except Exception:
        pkg_ver = "0.0.0"

    return {
        "package_version": pkg_ver,
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "timestamp": ts,
    }
