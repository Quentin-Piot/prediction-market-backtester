from __future__ import annotations

import math
from collections.abc import Mapping


def feature_as_float(features: Mapping[str, object], key: str) -> float | None:
    """Extract a feature value as a finite float, or None if missing/non-finite."""
    raw_value = features.get(key)
    if isinstance(raw_value, int | float):
        value = float(raw_value)
        if math.isfinite(value):
            return value
    return None
