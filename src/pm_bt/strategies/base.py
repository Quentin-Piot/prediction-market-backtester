from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from pm_bt.common.models import Bar, OrderIntent

FeatureMap = Mapping[str, object]


@runtime_checkable
class Strategy(Protocol):
    """Strategy contract for bar-by-bar order generation.

    Implementations may be stateless or stateful. The engine passes both:
    - `bar`: validated domain bar object for the current step
    - `features`: row-level feature mapping (same timestamp/market/outcome)
    """

    def on_bar(self, bar: Bar, features: FeatureMap) -> list[OrderIntent]:
        """Generate order intents for the current bar."""
        ...
