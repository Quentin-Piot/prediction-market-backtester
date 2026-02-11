from __future__ import annotations

from datetime import UTC, datetime

from pm_bt.common.models import OrderIntent
from pm_bt.common.types import OrderSide, Venue
from pm_bt.execution import ExecutionConfig, ExecutionSimulator, MarketSnapshot


def _order(ts: datetime, side: OrderSide, qty: float) -> OrderIntent:
    return OrderIntent(
        ts=ts,
        market_id="KX-RAIN-2026-01-01",
        outcome_id="yes",
        venue=Venue.KALSHI,
        side=side,
        qty=qty,
    )


def _snapshot(
    ts: datetime,
    *,
    mid: float,
    spread: float | None,
    recent_volume: float = 1_000.0,
    market_id: str = "KX-RAIN-2026-01-01",
    outcome_id: str = "yes",
    recent_prices: list[float] | None = None,
) -> MarketSnapshot:
    return MarketSnapshot(
        ts=ts,
        market_id=market_id,
        outcome_id=outcome_id,
        mid_price=mid,
        spread=spread,
        recent_volume=recent_volume,
        recent_prices=recent_prices,
    )


def _assert_close(actual: float, expected: float, *, tol: float = 1e-12) -> None:
    assert abs(actual - expected) <= tol


def test_execution_fills_at_bid_ask_not_mid() -> None:
    sim = ExecutionSimulator(ExecutionConfig(default_spread=0.02), initial_cash=1_000.0)
    ts = datetime(2026, 1, 1, 9, 0, tzinfo=UTC)

    fills = sim.execute_bar(
        bar_index=0,
        snapshot=_snapshot(ts, mid=0.50, spread=0.04),
        incoming_orders=[
            _order(ts, OrderSide.BUY, 1.0),
            _order(ts, OrderSide.SELL, 1.0),
        ],
    )

    assert len(fills) == 2
    _assert_close(fills[0].price_fill, 0.52)
    _assert_close(fills[1].price_fill, 0.48)


def test_execution_fees_reduce_cash() -> None:
    sim = ExecutionSimulator(ExecutionConfig(fee_bps=100.0), initial_cash=100.0)
    ts = datetime(2026, 1, 1, 9, 0, tzinfo=UTC)

    fills = sim.execute_bar(
        bar_index=0,
        snapshot=_snapshot(ts, mid=0.60, spread=0.02),
        incoming_orders=[_order(ts, OrderSide.BUY, 10.0)],
    )

    assert len(fills) == 1
    fill = fills[0]
    _assert_close(fill.price_fill, 0.61)
    _assert_close(fill.fees, 10.0 * 0.61 * 0.01)
    _assert_close(sim.state.cash, 100.0 - (10.0 * 0.61) - fill.fees)


def test_execution_latency_shifts_fill_timing() -> None:
    sim = ExecutionSimulator(ExecutionConfig(latency_bars=1), initial_cash=1_000.0)
    ts0 = datetime(2026, 1, 1, 9, 0, tzinfo=UTC)
    ts1 = datetime(2026, 1, 1, 9, 1, tzinfo=UTC)

    fills_bar0 = sim.execute_bar(
        bar_index=0,
        snapshot=_snapshot(ts0, mid=0.50, spread=0.02),
        incoming_orders=[_order(ts0, OrderSide.BUY, 1.0)],
    )
    fills_bar1 = sim.execute_bar(
        bar_index=1,
        snapshot=_snapshot(ts1, mid=0.55, spread=0.02),
        incoming_orders=[],
    )

    assert fills_bar0 == []
    assert len(fills_bar1) == 1
    assert fills_bar1[0].ts_fill == ts1
    _assert_close(fills_bar1[0].price_fill, 0.56)
    assert fills_bar1[0].latency_ms == 60_000


def test_execution_slippage_cost_accounted() -> None:
    sim = ExecutionSimulator(ExecutionConfig(slippage_bps=100.0), initial_cash=1_000.0)
    ts = datetime(2026, 1, 1, 9, 0, tzinfo=UTC)

    fills = sim.execute_bar(
        bar_index=0,
        snapshot=_snapshot(ts, mid=0.50, spread=0.02),
        incoming_orders=[_order(ts, OrderSide.BUY, 10.0)],
    )

    assert len(fills) == 1
    fill = fills[0]
    reference = 0.51
    expected_slippage_per_unit = reference * 0.01
    _assert_close(fill.price_fill, reference + expected_slippage_per_unit)
    _assert_close(fill.slippage_cost, 10.0 * expected_slippage_per_unit)


def test_execution_constraints_prevent_overbuy_from_max_exposure() -> None:
    sim = ExecutionSimulator(
        ExecutionConfig(
            max_position_size=100.0,
            max_gross_exposure=5.0,
            default_spread=0.0,
        ),
        initial_cash=1_000.0,
    )
    ts = datetime(2026, 1, 1, 9, 0, tzinfo=UTC)

    fills = sim.execute_bar(
        bar_index=0,
        snapshot=_snapshot(ts, mid=0.50, spread=0.0),
        incoming_orders=[_order(ts, OrderSide.BUY, 20.0)],
    )

    assert len(fills) == 1
    _assert_close(fills[0].qty_filled, 10.0)
    _assert_close(sim.current_gross_exposure(), 5.0)


def test_execution_uses_size_based_slippage_when_configured() -> None:
    sim = ExecutionSimulator(
        ExecutionConfig(slippage_bps=0.0, slippage_volume_k=0.2),
        initial_cash=1_000.0,
    )
    ts = datetime(2026, 1, 1, 9, 0, tzinfo=UTC)

    fills = sim.execute_bar(
        bar_index=0,
        snapshot=_snapshot(ts, mid=0.50, spread=0.02, recent_volume=100.0),
        incoming_orders=[_order(ts, OrderSide.BUY, 10.0)],
    )

    assert len(fills) == 1
    fill = fills[0]
    # base ask = 0.51, size-based slippage = 0.2 * (10/100) = 0.02
    _assert_close(fill.price_fill, 0.53)
    _assert_close(fill.slippage_cost, 0.2)


def test_execution_estimates_spread_from_recent_prices_when_missing() -> None:
    sim = ExecutionSimulator(ExecutionConfig(default_spread=0.01), initial_cash=1_000.0)
    ts = datetime(2026, 1, 1, 9, 0, tzinfo=UTC)

    fills = sim.execute_bar(
        bar_index=0,
        snapshot=_snapshot(
            ts,
            mid=0.50,
            spread=None,
            recent_prices=[0.49, 0.50, 0.51],
        ),
        incoming_orders=[_order(ts, OrderSide.BUY, 1.0)],
    )

    assert len(fills) == 1
    # mean abs diff = 0.01 -> spread = 0.02 -> ask = 0.51
    _assert_close(fills[0].price_fill, 0.51)


def test_execution_keeps_ready_orders_for_other_markets() -> None:
    sim = ExecutionSimulator(ExecutionConfig(latency_bars=0), initial_cash=1_000.0)
    ts = datetime(2026, 1, 1, 9, 0, tzinfo=UTC)

    order_other_market = OrderIntent(
        ts=ts,
        market_id="KX-TEMP-2026-01-02",
        outcome_id="yes",
        venue=Venue.KALSHI,
        side=OrderSide.BUY,
        qty=1.0,
    )

    fills_market_a = sim.execute_bar(
        bar_index=0,
        snapshot=_snapshot(ts, mid=0.50, spread=0.02, market_id="KX-RAIN-2026-01-01"),
        incoming_orders=[order_other_market],
    )
    fills_market_b = sim.execute_bar(
        bar_index=0,
        snapshot=_snapshot(ts, mid=0.40, spread=0.02, market_id="KX-TEMP-2026-01-02"),
        incoming_orders=[],
    )

    assert fills_market_a == []
    assert len(fills_market_b) == 1
    _assert_close(fills_market_b[0].price_fill, 0.41)
