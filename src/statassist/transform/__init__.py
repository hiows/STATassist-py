"""Reshaping data before it is compared, rather than as part of comparing it.

One public function so far: :func:`center_by_control`, which removes the control
group's centre so that every observation reads as its distance from the control.
It shares its arguments with the comparisons on purpose, so the same set of
arguments describes both steps.
"""

from __future__ import annotations

from .center import center_by_control, control_baseline

__all__ = ["center_by_control", "control_baseline"]
