from __future__ import annotations

from datetime import UTC, datetime

from pm_bt.common.models import Bar
from pm_bt.common.types import OrderSide, Venue
from pm_bt.strategies import EventThresholdStrategy, MeanReversionStrategy, MomentumStrategy
from pm_bt.strategies.base import FeatureMap


def _make_bar(*, close: float = 0.5, volume: float = 100.0) -> Bar:
    ts = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    return Bar(
        ts_open=ts,
        ts_close=ts,
        market_id="KX-RAIN-2026-01-01",
        outcome_id="yes",
        venue=Venue.KALSHI,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=volume,
        vwap=close,
        trade_count=1,
    )


def test_momentum_strategy_emits_buy_order_on_positive_momentum() -> None:
    strategy = MomentumStrategy(threshold=0.05, qty=3.0)
    bar = _make_bar()
    features: FeatureMap = {"momentum": 0.10}

    orders = strategy.on_bar(bar, features)

    assert len(orders) == 1
    assert orders[0].side == OrderSide.BUY
    assert orders[0].qty == 3.0


def test_mean_reversion_strategy_emits_sell_order_on_large_up_move() -> None:
    strategy = MeanReversionStrategy(move_threshold=0.02, qty=2.0)
    bar = _make_bar()
    features: FeatureMap = {"ret_1": 0.05}

    orders = strategy.on_bar(bar, features)

    assert len(orders) == 1
    assert orders[0].side == OrderSide.SELL
    assert orders[0].qty == 2.0


def test_event_threshold_strategy_emits_directional_order_when_jump_and_volume_match() -> None:
    strategy = EventThresholdStrategy(price_jump_threshold=0.05, min_volume=50.0, qty=4.0)
    bar = _make_bar(volume=120.0)
    features: FeatureMap = {"ret_1": 0.08}

    orders = strategy.on_bar(bar, features)

    assert len(orders) == 1
    assert orders[0].side == OrderSide.BUY
    assert orders[0].qty == 4.0


def test_event_threshold_strategy_skips_signal_when_volume_is_too_low() -> None:
    strategy = EventThresholdStrategy(price_jump_threshold=0.05, min_volume=200.0, qty=1.0)
    bar = _make_bar(volume=100.0)
    features: FeatureMap = {"ret_1": 0.10}

    orders = strategy.on_bar(bar, features)

    assert orders == []
