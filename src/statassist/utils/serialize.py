"""JSON-serializable conversion for result contracts."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd


def _to_jsonable(obj: Any, *, omit_fit: bool = True) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        if np.isnan(val) or np.isinf(val):
            return val
        return val
    if isinstance(obj, np.ndarray):
        return _to_jsonable(obj.tolist(), omit_fit=omit_fit)
    if isinstance(obj, pd.DataFrame):
        return {
            "columns": list(obj.columns),
            "data": [
                [_cell(v) for v in row]
                for row in obj.itertuples(index=False, name=None)
            ],
        }
    if isinstance(obj, pd.Series):
        return {k: _cell(v) for k, v in obj.items()}
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if omit_fit and k == "fit":
                continue
            if v is None:
                continue
            if isinstance(v, dict) and len(v) == 0:
                continue
            if isinstance(v, list) and len(v) == 0:
                continue
            out[k] = _to_jsonable(v, omit_fit=omit_fit)
        return out
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x, omit_fit=omit_fit) for x in obj]
    if hasattr(obj, "to_dict"):
        return _to_jsonable(obj.to_dict(), omit_fit=omit_fit)
    raise TypeError(f"Cannot serialize type {type(obj)!r}")


def _cell(v: Any) -> Any:
    if pd.isna(v):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def to_json(obj: Any, *, omit_fit: bool = True, indent: int | None = 2) -> str:
    return json.dumps(_to_jsonable(obj, omit_fit=omit_fit), indent=indent, allow_nan=True)
