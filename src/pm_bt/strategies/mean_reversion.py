from __future__ import annotations

from dataclasses import dataclass

from pm_bt.common.models import Bar, OrderIntent
from pm_bt.common.types import OrderSide
from pm_bt.strategies._utils import feature_as_float
from pm_bt.strategies.base import FeatureMap


@dataclass(slots=True)
class MeanReversionStrategy:
    """Fade large one-bar moves in implied probability."""

    move_threshold: float = 0.02
    qty: float = 1.0

    def __post_init__(self) -> None:
        if self.move_threshold < 0.0:
            raise ValueError("move_threshold must be >= 0")
        if self.qty <= 0.0:
            raise ValueError("qty must be > 0")

    def on_bar(self, bar: Bar, features: FeatureMap) -> list[OrderIntent]:
        ret_1 = feature_as_float(features, "ret_1")
        if ret_1 is None:
            return []

        if ret_1 >= self.move_threshold:
            side = OrderSide.SELL
            reason = "mean_reversion_fade_up_move"
        elif ret_1 <= -self.move_threshold:
            side = OrderSide.BUY
            reason = "mean_reversion_fade_down_move"
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
