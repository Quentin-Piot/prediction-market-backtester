from __future__ import annotations

from dataclasses import dataclass

from pm_bt.common.models import Bar, OrderIntent
from pm_bt.common.types import OrderSide
from pm_bt.strategies._utils import feature_as_float
from pm_bt.strategies.base import FeatureMap


@dataclass(slots=True)
class EventThresholdStrategy:
    """Trade directional jumps when move and participation are both elevated."""

    price_jump_threshold: float = 0.05
    min_volume: float = 0.0
    qty: float = 1.0

    def __post_init__(self) -> None:
        if self.price_jump_threshold < 0.0:
            raise ValueError("price_jump_threshold must be >= 0")
        if self.min_volume < 0.0:
            raise ValueError("min_volume must be >= 0")
        if self.qty <= 0.0:
            raise ValueError("qty must be > 0")

    def on_bar(self, bar: Bar, features: FeatureMap) -> list[OrderIntent]:
        ret_1 = feature_as_float(features, "ret_1")
        if ret_1 is None:
            return []
        if abs(ret_1) < self.price_jump_threshold:
            return []
        if bar.volume < self.min_volume:
            return []

        side = OrderSide.BUY if ret_1 > 0.0 else OrderSide.SELL
        reason = "event_jump_with_volume"
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
