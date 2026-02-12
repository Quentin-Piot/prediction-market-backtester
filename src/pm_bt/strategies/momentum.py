from __future__ import annotations

from dataclasses import dataclass

from pm_bt.common.models import Bar, OrderIntent
from pm_bt.common.types import OrderSide
from pm_bt.strategies._utils import feature_as_float
from pm_bt.strategies.base import FeatureMap


@dataclass(slots=True)
class MomentumStrategy:
    """Directional strategy on implied-probability momentum."""

    threshold: float = 0.0
    qty: float = 1.0

    def __post_init__(self) -> None:
        if self.threshold < 0.0:
            raise ValueError("threshold must be >= 0")
        if self.qty <= 0.0:
            raise ValueError("qty must be > 0")

    def on_bar(self, bar: Bar, features: FeatureMap) -> list[OrderIntent]:
        momentum = feature_as_float(features, "momentum")
        if momentum is None:
            return []

        if momentum > self.threshold:
            side = OrderSide.BUY
            reason = "momentum_up"
        elif momentum < -self.threshold:
            side = OrderSide.SELL
            reason = "momentum_down"
        else:
            return []

        return [
            OrderIntent(
                ts=bar.ts_close,
                market_id=bar.market_id,
                outcome_id=bar.outcome_id,
                venue=bar.venue,
                side=side,
                qty=self.qty,
                reason=reason,
            )
        ]
