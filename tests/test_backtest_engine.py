from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import polars as pl
import pytest
from typing_extensions import override

from pm_bt.backtest import BacktestEngine
from pm_bt.common.models import BacktestConfig, Bar, OrderIntent
from pm_bt.common.types import OrderSide, Venue
from pm_bt.strategies import Strategy
from pm_bt.strategies.base import FeatureMap


def _assert_close(actual: float, expected: float, *, tol: float = 1e-12) -> None:
    assert abs(actual - expected) <= tol


def _series_value(df: pl.DataFrame, column: str, index: int) -> float:
    return float(cast(float, df[column][index]))


def _write_trades_fixture(data_root: Path) -> None:
    (data_root / "kalshi" / "trades").mkdir(parents=True, exist_ok=True)
    trades = pl.DataFrame(
        {
            "ts": [
                datetime(2026, 1, 1, 9, 0, tzinfo=UTC),
                datetime(2026, 1, 1, 9, 1, tzinfo=UTC),
            ],
            "market_id": ["KX-RAIN-2026-01-01", "KX-RAIN-2026-01-01"],
            "outcome_id": ["yes", "yes"],
            "venue": ["kalshi", "kalshi"],
            "price": [0.50, 0.60],
            "size": [100.0, 100.0],
            "side": ["buy", "buy"],
            "trade_id": ["t1", "t2"],
            "fee_paid": [0.0, 0.0],
        }
    )
    trades.write_parquet(data_root / "kalshi" / "trades" / "trades_0_2.parquet")


class _BuyFirstBarStrategy(Strategy):
    def __init__(self) -> None:
        self._emitted: bool = False

    @override
    def on_bar(self, bar: Bar, features: FeatureMap) -> list[OrderIntent]:
        if self._emitted:
            return []
        self._emitted = True
        return [
            OrderIntent(
                ts=bar.ts_close,
                market_id=bar.market_id,
                outcome_id=bar.outcome_id,
                venue=bar.venue,
                side=OrderSide.BUY,
                qty=10.0,
            )
        ]


class _RoundTripStrategy(Strategy):
    def __init__(self) -> None:
        self._step: int = 0

    @override
    def on_bar(self, bar: Bar, features: FeatureMap) -> list[OrderIntent]:
        if self._step == 0:
            self._step += 1
            return [
                OrderIntent(
                    ts=bar.ts_close,
                    market_id=bar.market_id,
                    outcome_id=bar.outcome_id,
                    venue=bar.venue,
                    side=OrderSide.BUY,
                    qty=10.0,
                )
            ]
        if self._step == 1:
            self._step += 1
            return [
                OrderIntent(
                    ts=bar.ts_close,
                    market_id=bar.market_id,
                    outcome_id=bar.outcome_id,
                    venue=bar.venue,
                    side=OrderSide.SELL,
                    qty=10.0,
                )
            ]
        return []


class _AlwaysBuyStrategy(Strategy):
    def __init__(self) -> None:
        self.calls: int = 0

    @override
    def on_bar(self, bar: Bar, features: FeatureMap) -> list[OrderIntent]:
        self.calls += 1
        return [
            OrderIntent(
                ts=bar.ts_close,
                market_id=bar.market_id,
                outcome_id=bar.outcome_id,
                venue=bar.venue,
                side=OrderSide.BUY,
                qty=10.0,
            )
        ]


class _FeatureCaptureStrategy(Strategy):
    def __init__(self) -> None:
        self.momentum_values: list[float | None] = []

    @override
    def on_bar(self, bar: Bar, features: FeatureMap) -> list[OrderIntent]:
        raw_momentum = features.get("momentum")
        self.momentum_values.append(cast(float | None, raw_momentum))
        return []


def test_backtest_engine_runs_end_to_end_on_fixture_data(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_trades_fixture(data_root)

    config = BacktestConfig(
        name="engine-e2e",
        venue=Venue.KALSHI,
        market_id="KX-RAIN-2026-01-01",
        start_ts=datetime(2026, 1, 1, 9, 0, tzinfo=UTC),
        end_ts=datetime(2026, 1, 1, 9, 2, tzinfo=UTC),
        bar_timeframe="1m",
        initial_cash=100.0,
        fee_bps=0.0,
        slippage_bps=0.0,
        latency_bars=0,
        max_position_size=100.0,
        max_gross_exposure=10_000.0,
        data_root=data_root,
    )
    engine = BacktestEngine(config=config, strategy=_BuyFirstBarStrategy())

    artifacts = engine.run()

    assert artifacts.fills.height == 1
    assert artifacts.equity_curve.height == 2
    assert artifacts.exposure_curve.height == 2

    _assert_close(_series_value(artifacts.fills, "price_fill", 0), 0.51)
    _assert_close(_series_value(artifacts.equity_curve, "cash", -1), 94.9)
    _assert_close(_series_value(artifacts.equity_curve, "equity", -1), 100.9)
    _assert_close(_series_value(artifacts.equity_curve, "cash_at_risk_gross", -1), 4.0)
    _assert_close(_series_value(artifacts.exposure_curve, "position", -1), 10.0)
    _assert_close(_series_value(artifacts.exposure_curve, "cash_at_risk", -1), 4.0)
    _assert_close(artifacts.run_result.trading_metrics["total_pnl"], 0.9)
    _assert_close(artifacts.run_result.trading_metrics["unrealized_pnl"], 0.9)
    _assert_close(artifacts.run_result.trading_metrics["realized_pnl"], 0.0)


def test_backtest_engine_accounts_realized_and_unrealized_pnl_from_round_trip() -> None:
    bars = pl.DataFrame(
        {
            "ts_open": [
                datetime(2026, 1, 1, 9, 0, tzinfo=UTC),
                datetime(2026, 1, 1, 9, 1, tzinfo=UTC),
            ],
            "ts_close": [
                datetime(2026, 1, 1, 9, 0, tzinfo=UTC),
                datetime(2026, 1, 1, 9, 1, tzinfo=UTC),
            ],
            "market_id": ["KX-RAIN-2026-01-01", "KX-RAIN-2026-01-01"],
            "outcome_id": ["yes", "yes"],
            "venue": ["kalshi", "kalshi"],
            "open": [0.50, 0.60],
            "high": [0.50, 0.60],
            "low": [0.50, 0.60],
            "close": [0.50, 0.60],
            "volume": [100.0, 100.0],
            "vwap": [0.50, 0.60],
            "trade_count": [1, 1],
            "momentum": [None, 0.2],
        }
    )
    config = BacktestConfig(
        name="engine-round-trip",
        venue=Venue.KALSHI,
        market_id="KX-RAIN-2026-01-01",
        initial_cash=100.0,
        fee_bps=0.0,
        slippage_bps=0.0,
        latency_bars=0,
        max_position_size=100.0,
        max_gross_exposure=10_000.0,
    )

    artifacts = BacktestEngine(config=config, strategy=_RoundTripStrategy()).run(bars)

    assert artifacts.fills.height == 2
    _assert_close(_series_value(artifacts.fills, "price_fill", 0), 0.51)
    _assert_close(_series_value(artifacts.fills, "price_fill", 1), 0.59)

    _assert_close(_series_value(artifacts.equity_curve, "cash", -1), 100.8)
    _assert_close(_series_value(artifacts.equity_curve, "cash_at_risk_gross", -1), 0.0)
    final_equity = _series_value(artifacts.equity_curve, "equity", -1)
    _assert_close(final_equity, 100.8)
    _assert_close(artifacts.run_result.trading_metrics["realized_pnl"], 0.8)
    _assert_close(artifacts.run_result.trading_metrics["unrealized_pnl"], 0.0)
    _assert_close(artifacts.run_result.trading_metrics["total_pnl"], 0.8)


def test_backtest_engine_passes_row_features_to_strategy() -> None:
    bars = pl.DataFrame(
        {
            "ts_open": [
                datetime(2026, 1, 1, 9, 0, tzinfo=UTC),
                datetime(2026, 1, 1, 9, 1, tzinfo=UTC),
            ],
            "ts_close": [
                datetime(2026, 1, 1, 9, 0, tzinfo=UTC),
                datetime(2026, 1, 1, 9, 1, tzinfo=UTC),
            ],
            "market_id": ["KX-RAIN-2026-01-01", "KX-RAIN-2026-01-01"],
            "outcome_id": ["yes", "yes"],
            "venue": ["kalshi", "kalshi"],
            "open": [0.50, 0.60],
            "high": [0.50, 0.60],
            "low": [0.50, 0.60],
            "close": [0.50, 0.60],
            "volume": [100.0, 100.0],
            "vwap": [0.50, 0.60],
            "trade_count": [1, 1],
            "momentum": [None, 0.2],
        }
    )
    strategy = _FeatureCaptureStrategy()
    config = BacktestConfig(
        name="feature-map",
        venue=Venue.KALSHI,
        market_id="KX-RAIN-2026-01-01",
        initial_cash=100.0,
        fee_bps=0.0,
        slippage_bps=0.0,
        latency_bars=0,
        max_position_size=100.0,
        max_gross_exposure=10_000.0,
    )

    artifacts = BacktestEngine(config=config, strategy=strategy).run(bars)

    assert strategy.momentum_values == [None, 0.2]
    assert artifacts.fills.height == 0


def test_backtest_engine_drawdown_stop_halts_new_orders_only() -> None:
    bars = pl.DataFrame(
        {
            "ts_open": [
                datetime(2026, 1, 1, 9, 0, tzinfo=UTC),
                datetime(2026, 1, 1, 9, 1, tzinfo=UTC),
                datetime(2026, 1, 1, 9, 2, tzinfo=UTC),
            ],
            "ts_close": [
                datetime(2026, 1, 1, 9, 0, tzinfo=UTC),
                datetime(2026, 1, 1, 9, 1, tzinfo=UTC),
                datetime(2026, 1, 1, 9, 2, tzinfo=UTC),
            ],
            "market_id": ["KX-RAIN-2026-01-01", "KX-RAIN-2026-01-01", "KX-RAIN-2026-01-01"],
            "outcome_id": ["yes", "yes", "yes"],
            "venue": ["kalshi", "kalshi", "kalshi"],
            "open": [0.60, 0.10, 0.10],
            "high": [0.60, 0.10, 0.10],
            "low": [0.60, 0.10, 0.10],
            "close": [0.60, 0.10, 0.10],
            "volume": [100.0, 100.0, 100.0],
            "vwap": [0.60, 0.10, 0.10],
            "trade_count": [1, 1, 1],
        }
    )
    strategy = _AlwaysBuyStrategy()
    config = BacktestConfig(
        name="drawdown-stop",
        venue=Venue.KALSHI,
        market_id="KX-RAIN-2026-01-01",
        initial_cash=100.0,
        fee_bps=0.0,
        slippage_bps=0.0,
        latency_bars=0,
        drawdown_stop_pct=0.05,
        max_position_size=100.0,
        max_gross_exposure=10_000.0,
    )

    artifacts = BacktestEngine(config=config, strategy=strategy).run(bars)

    assert strategy.calls == 2
    assert artifacts.fills.height == 2


def test_backtest_engine_rejects_multi_outcome_stream_when_market_id_is_set() -> None:
    bars = pl.DataFrame(
        {
            "ts_open": [
                datetime(2026, 1, 1, 9, 0, tzinfo=UTC),
                datetime(2026, 1, 1, 9, 1, tzinfo=UTC),
            ],
            "ts_close": [
                datetime(2026, 1, 1, 9, 0, tzinfo=UTC),
                datetime(2026, 1, 1, 9, 1, tzinfo=UTC),
            ],
            "market_id": ["KX-RAIN-2026-01-01", "KX-RAIN-2026-01-01"],
            "outcome_id": ["yes", "no"],
            "venue": ["kalshi", "kalshi"],
            "open": [0.50, 0.50],
            "high": [0.50, 0.50],
            "low": [0.50, 0.50],
            "close": [0.50, 0.50],
            "volume": [100.0, 100.0],
            "vwap": [0.50, 0.50],
            "trade_count": [1, 1],
        }
    )
    config = BacktestConfig(
        name="stream-guard",
        venue=Venue.KALSHI,
        market_id="KX-RAIN-2026-01-01",
        initial_cash=100.0,
        fee_bps=0.0,
        slippage_bps=0.0,
        latency_bars=0,
        max_position_size=100.0,
        max_gross_exposure=10_000.0,
    )

    with pytest.raises(ValueError, match="single venue/market/outcome stream"):
        _ = BacktestEngine(config=config, strategy=_BuyFirstBarStrategy()).run(bars)


class _FlipStrategy(Strategy):
    """Buy 10 on bar 0, then sell 15 on bar 1 (flips from +10 to -5)."""

    def __init__(self) -> None:
        self._step: int = 0

    @override
    def on_bar(self, bar: Bar, features: FeatureMap) -> list[OrderIntent]:
        if self._step == 0:
            self._step += 1
            return [
                OrderIntent(
                    ts=bar.ts_close,
                    market_id=bar.market_id,
                    outcome_id=bar.outcome_id,
                    venue=bar.venue,
                    side=OrderSide.BUY,
                    qty=10.0,
                )
            ]
        if self._step == 1:
            self._step += 1
            return [
                OrderIntent(
                    ts=bar.ts_close,
                    market_id=bar.market_id,
                    outcome_id=bar.outcome_id,
                    venue=bar.venue,
                    side=OrderSide.SELL,
                    qty=15.0,
                )
            ]
        return []


def test_backtest_engine_position_flip_tracks_realized_pnl_correctly() -> None:
    """Flip from long +10 to short -5 in one fill; realized PnL must reflect the closed portion."""
    bars = pl.DataFrame(
        {
            "ts_open": [
                datetime(2026, 1, 1, 9, 0, tzinfo=UTC),
                datetime(2026, 1, 1, 9, 1, tzinfo=UTC),
                datetime(2026, 1, 1, 9, 2, tzinfo=UTC),
            ],
            "ts_close": [
                datetime(2026, 1, 1, 9, 0, tzinfo=UTC),
                datetime(2026, 1, 1, 9, 1, tzinfo=UTC),
                datetime(2026, 1, 1, 9, 2, tzinfo=UTC),
            ],
            "market_id": ["KX-RAIN-2026-01-01"] * 3,
            "outcome_id": ["yes"] * 3,
            "venue": ["kalshi"] * 3,
            "open": [0.50, 0.60, 0.60],
            "high": [0.50, 0.60, 0.60],
            "low": [0.50, 0.60, 0.60],
            "close": [0.50, 0.60, 0.60],
            "volume": [100.0, 100.0, 100.0],
            "vwap": [0.50, 0.60, 0.60],
            "trade_count": [1, 1, 1],
        }
    )
    config = BacktestConfig(
        name="flip-test",
        venue=Venue.KALSHI,
        market_id="KX-RAIN-2026-01-01",
        initial_cash=100.0,
        fee_bps=0.0,
        slippage_bps=0.0,
        latency_bars=0,
        max_position_size=100.0,
        max_gross_exposure=10_000.0,
    )

    artifacts = BacktestEngine(config=config, strategy=_FlipStrategy()).run(bars)

    assert artifacts.fills.height == 2

    # Bar 0: buy 10 at ask = 0.50 + 0.02/2 = 0.51, cash = 100 - 5.1 = 94.9
    buy_fill = _series_value(artifacts.fills, "price_fill", 0)
    _assert_close(buy_fill, 0.51)

    # Bar 1: sell 15 at bid = 0.60 - 0.02/2 = 0.59
    # Closes 10 long (realized PnL = (0.59 - 0.51) * 10 = 0.8), opens 5 short
    sell_fill = _series_value(artifacts.fills, "price_fill", 1)
    _assert_close(sell_fill, 0.59)

    _assert_close(artifacts.run_result.trading_metrics["realized_pnl"], 0.8)

    # Final position is -5 short YES at mark 0.60
    # Unrealized PnL for short: (entry - mark) * qty = (0.59 - 0.60) * 5 = -0.05
    _assert_close(artifacts.run_result.trading_metrics["unrealized_pnl"], -0.05)

    # Cash: 94.9 + 15 * 0.59 = 94.9 + 8.85 = 103.75
    _assert_close(_series_value(artifacts.equity_curve, "cash", -1), 103.75)

    # Equity: cash + position * mark = 103.75 + (-5) * 0.60 = 103.75 - 3.0 = 100.75
    _assert_close(_series_value(artifacts.equity_curve, "equity", -1), 100.75)
