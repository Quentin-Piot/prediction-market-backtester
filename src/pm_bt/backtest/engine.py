from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import cast

import polars as pl

from pm_bt.common.models import (
    BacktestConfig,
    Bar,
    DatasetSlice,
    Fill,
    OrderIntent,
    RunResult,
    RunTimings,
)
from pm_bt.common.types import OrderSide, Venue
from pm_bt.common.utils import get_git_commit_hash, make_run_id
from pm_bt.data import load_trades
from pm_bt.execution import ExecutionConfig, ExecutionSimulator, MarketSnapshot
from pm_bt.features import add_basic_features, build_bars
from pm_bt.strategies.base import FeatureMap, Strategy

logger = logging.getLogger(__name__)

BAR_COLUMNS = {
    "ts_open",
    "ts_close",
    "market_id",
    "outcome_id",
    "venue",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
    "trade_count",
}


@dataclass(slots=True)
class BacktestArtifacts:
    run_result: RunResult
    equity_curve: pl.DataFrame
    fills: pl.DataFrame
    exposure_curve: pl.DataFrame


class _PnlLedger:
    def __init__(self) -> None:
        self.realized_pnl: float = 0.0
        self._avg_entry: dict[tuple[str, str], float] = {}

    def avg_entry(self, key: tuple[str, str]) -> float | None:
        return self._avg_entry.get(key)

    def apply_fill(
        self,
        *,
        fill: Fill,
        position_before: float,
    ) -> None:
        key = (fill.market_id, fill.outcome_id)
        qty = fill.qty_filled
        price = fill.price_fill
        avg_before = self._avg_entry.get(key, 0.0)

        # price_fill already includes spread/slippage impact. slippage_cost is reporting-only.
        realized_delta = -fill.fees

        if fill.side == OrderSide.BUY:
            if position_before >= 0.0:
                position_after = position_before + qty
                if position_after > 0.0:
                    weighted = (position_before * avg_before) + (qty * price)
                    self._avg_entry[key] = weighted / position_after
            else:
                close_qty = min(qty, abs(position_before))
                realized_delta += (avg_before - price) * close_qty
                position_after = position_before + qty
                if position_after < 0.0:
                    self._avg_entry[key] = avg_before
                elif position_after > 0.0:
                    self._avg_entry[key] = price
                else:
                    if key in self._avg_entry:
                        del self._avg_entry[key]
        else:
            if position_before <= 0.0:
                position_after = position_before - qty
                short_before = abs(position_before)
                short_after = abs(position_after)
                if short_after > 0.0:
                    weighted = (short_before * avg_before) + (qty * price)
                    self._avg_entry[key] = weighted / short_after
            else:
                close_qty = min(qty, position_before)
                realized_delta += (price - avg_before) * close_qty
                position_after = position_before - qty
                if position_after > 0.0:
                    self._avg_entry[key] = avg_before
                elif position_after < 0.0:
                    self._avg_entry[key] = price
                else:
                    if key in self._avg_entry:
                        del self._avg_entry[key]

        if abs(position_after) <= 1e-12 and key in self._avg_entry:
            del self._avg_entry[key]

        self.realized_pnl += realized_delta


class BacktestEngine:
    config: BacktestConfig
    strategy: Strategy

    def __init__(self, *, config: BacktestConfig, strategy: Strategy) -> None:
        self.config = config
        self.strategy = strategy

    def run(self, bars: pl.DataFrame | pl.LazyFrame | None = None) -> BacktestArtifacts:
        total_t0 = perf_counter()
        timings = RunTimings()

        bars_df: pl.DataFrame
        if bars is None:
            load_t0 = perf_counter()
            bars_df = self._build_bars_from_data()
            timings.load_s = perf_counter() - load_t0
            # Bars + features are built in one lazy plan collection.
            timings.features_s = 0.0
        else:
            bars_df = self._collect_bars(bars)

        if bars_df.is_empty():
            raise ValueError("No bars available for the configured backtest slice")
        self._assert_supported_stream(bars_df)

        execution_t0 = perf_counter()
        simulator = ExecutionSimulator(
            ExecutionConfig.from_backtest_config(self.config),
            initial_cash=self.config.initial_cash,
        )
        ledger = _PnlLedger()

        last_marks: dict[tuple[str, str], float] = {}
        fills_out: list[dict[str, object]] = []
        equity_out: list[dict[str, object]] = []
        exposure_out: list[dict[str, object]] = []

        peak_equity = self.config.initial_cash
        max_drawdown = 0.0
        # Drawdown stop halts new signal emissions but does not cancel pending orders.
        stop_emitting_orders = False

        row_columns = bars_df.columns
        column_index = {name: idx for idx, name in enumerate(row_columns)}
        feature_indices = [
            (idx, name) for idx, name in enumerate(row_columns) if name not in BAR_COLUMNS
        ]

        for bar_index, row in enumerate(bars_df.iter_rows(named=False)):
            bar = self._bar_from_tuple(row, column_index)
            key = (bar.market_id, bar.outcome_id)
            last_marks[key] = bar.close

            features: FeatureMap = {name: row[idx] for idx, name in feature_indices}
            incoming_orders = [] if stop_emitting_orders else self.strategy.on_bar(bar, features)
            self._validate_orders(incoming_orders=incoming_orders, bar=bar)

            pre_positions = dict(simulator.state.positions)
            fills = simulator.execute_bar(
                bar_index=bar_index,
                snapshot=MarketSnapshot(
                    ts=bar.ts_close,
                    market_id=bar.market_id,
                    outcome_id=bar.outcome_id,
                    mid_price=bar.close,
                    spread=None,
                    recent_volume=bar.volume,
                ),
                incoming_orders=incoming_orders,
            )

            for fill in fills:
                pos_before = pre_positions.get((fill.market_id, fill.outcome_id), 0.0)
                ledger.apply_fill(fill=fill, position_before=pos_before)
                pre_positions[(fill.market_id, fill.outcome_id)] = (
                    pos_before + fill.qty_filled
                    if fill.side == OrderSide.BUY
                    else pos_before - fill.qty_filled
                )
                fills_out.append(
                    {
                        "ts_fill": fill.ts_fill,
                        "market_id": fill.market_id,
                        "outcome_id": fill.outcome_id,
                        "venue": fill.venue.value,
                        "side": fill.side.value,
                        "qty_filled": fill.qty_filled,
                        "price_fill": fill.price_fill,
                        "fees": fill.fees,
                        "slippage_cost": fill.slippage_cost,
                        "latency_ms": fill.latency_ms,
                        "notional": fill.notional,
                    }
                )

            equity, gross_notional, cash_at_risk_gross, unrealized_pnl = self._mark_to_market(
                positions=simulator.state.positions,
                marks=last_marks,
                ledger=ledger,
                cash=simulator.state.cash,
            )

            peak_equity = max(peak_equity, equity)
            drawdown = 0.0 if peak_equity <= 0 else (peak_equity - equity) / peak_equity
            max_drawdown = max(max_drawdown, drawdown)

            equity_out.append(
                {
                    "ts": bar.ts_close,
                    "equity": equity,
                    "cash": simulator.state.cash,
                    "realized_pnl": ledger.realized_pnl,
                    "unrealized_pnl": unrealized_pnl,
                    "gross_notional_exposure": gross_notional,
                    "cash_at_risk_gross": cash_at_risk_gross,
                }
            )
            self._append_exposure_rows(
                ts=bar.ts_close,
                positions=simulator.state.positions,
                marks=last_marks,
                out=exposure_out,
            )

            if (
                self.config.drawdown_stop_pct is not None
                and drawdown >= self.config.drawdown_stop_pct
            ):
                stop_emitting_orders = True

        timings.execution_s = perf_counter() - execution_t0
        timings.total_s = perf_counter() - total_t0

        fills_df = pl.DataFrame(
            fills_out,
            schema={
                "ts_fill": pl.Datetime(time_zone="UTC"),
                "market_id": pl.Utf8,
                "outcome_id": pl.Utf8,
                "venue": pl.Utf8,
                "side": pl.Utf8,
                "qty_filled": pl.Float64,
                "price_fill": pl.Float64,
                "fees": pl.Float64,
                "slippage_cost": pl.Float64,
                "latency_ms": pl.Int64,
                "notional": pl.Float64,
            },
        )
        equity_df = pl.DataFrame(
            equity_out,
            schema={
                "ts": pl.Datetime(time_zone="UTC"),
                "equity": pl.Float64,
                "cash": pl.Float64,
                "realized_pnl": pl.Float64,
                "unrealized_pnl": pl.Float64,
                "gross_notional_exposure": pl.Float64,
                "cash_at_risk_gross": pl.Float64,
            },
        )
        exposure_df = pl.DataFrame(
            exposure_out,
            schema={
                "ts": pl.Datetime(time_zone="UTC"),
                "market_id": pl.Utf8,
                "outcome_id": pl.Utf8,
                "position": pl.Float64,
                "notional_exposure": pl.Float64,
                "cash_at_risk": pl.Float64,
            },
        )

        final_equity = (
            float(cast(float, equity_df["equity"][-1]))
            if equity_df.height
            else float(self.config.initial_cash)
        )
        final_unrealized = (
            float(cast(float, equity_df["unrealized_pnl"][-1])) if equity_df.height else 0.0
        )
        max_cash_at_risk = (
            float(cast(float, equity_df["cash_at_risk_gross"].max())) if equity_df.height else 0.0
        )
        max_gross_notional = (
            float(cast(float, equity_df["gross_notional_exposure"].max()))
            if equity_df.height
            else 0.0
        )
        total_notional = float(cast(float, fills_df["notional"].sum())) if fills_df.height else 0.0

        metrics = {
            "total_pnl": final_equity - self.config.initial_cash,
            "total_return": (final_equity / self.config.initial_cash) - 1.0,
            "realized_pnl": ledger.realized_pnl,
            "unrealized_pnl": final_unrealized,
            "max_drawdown": max_drawdown,
            "max_cash_at_risk_gross": max_cash_at_risk,
            # Backward-compatible alias with PM semantics (cash-at-risk).
            "max_gross_exposure": max_cash_at_risk,
            "max_gross_notional_exposure": max_gross_notional,
            "turnover": total_notional / self.config.initial_cash,
            "fills_count": float(fills_df.height),
            "bars_processed": float(equity_df.height),
        }

        git_commit = get_git_commit_hash(cwd=Path.cwd())
        run_result = RunResult(
            run_id=make_run_id(short_hash=git_commit),
            created_at=datetime.now(tz=UTC),
            config=self.config,
            dataset_slice=self._dataset_slice(),
            git_commit=git_commit,
            trading_metrics=metrics,
            forecasting_metrics={},
            artifacts={},
            timings=timings,
        )

        return BacktestArtifacts(
            run_result=run_result,
            equity_curve=equity_df,
            fills=fills_df,
            exposure_curve=exposure_df,
        )

    def _build_bars_from_data(self) -> pl.DataFrame:
        if self.config.venue is None:
            raise ValueError("config.venue is required when bars are not provided")

        trades = load_trades(
            self.config.venue,
            data_root=self.config.data_root,
            market_id=self.config.market_id,
            start_ts=self.config.start_ts,
            end_ts=self.config.end_ts,
        )
        bars = add_basic_features(build_bars(trades, timeframe=self.config.bar_timeframe))
        return self._collect_bars(bars)

    @staticmethod
    def _collect_bars(bars: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
        lf = bars.lazy() if isinstance(bars, pl.DataFrame) else bars
        schema = lf.collect_schema()
        missing = BAR_COLUMNS.difference(schema.names())
        if missing:
            joined = ", ".join(sorted(missing))
            raise ValueError(f"Bars are missing required columns: {joined}")

        return lf.sort(["ts_open", "venue", "market_id", "outcome_id"]).collect()

    @staticmethod
    def _bar_from_tuple(row: tuple[object, ...], column_index: dict[str, int]) -> Bar:
        raw_venue = row[column_index["venue"]]
        venue = Venue(raw_venue) if isinstance(raw_venue, str) else cast(Venue, raw_venue)
        return Bar(
            ts_open=cast(datetime, row[column_index["ts_open"]]),
            ts_close=cast(datetime, row[column_index["ts_close"]]),
            market_id=cast(str, row[column_index["market_id"]]),
            outcome_id=cast(str, row[column_index["outcome_id"]]),
            venue=venue,
            open=cast(float, row[column_index["open"]]),
            high=cast(float, row[column_index["high"]]),
            low=cast(float, row[column_index["low"]]),
            close=cast(float, row[column_index["close"]]),
            volume=cast(float, row[column_index["volume"]]),
            vwap=cast(float | None, row[column_index["vwap"]]),
            trade_count=cast(int, row[column_index["trade_count"]]),
        )

    @staticmethod
    def _validate_orders(*, incoming_orders: list[OrderIntent], bar: Bar) -> None:
        for order in incoming_orders:
            if order.market_id != bar.market_id or order.outcome_id != bar.outcome_id:
                raise ValueError(
                    "Strategy emitted order for a different market/outcome than current bar: "
                    + f"{order.market_id}/{order.outcome_id} vs {bar.market_id}/{bar.outcome_id}"
                )
            if order.venue != bar.venue:
                raise ValueError(
                    "Strategy emitted order with mismatched venue: "
                    + f"{order.venue.value} vs {bar.venue.value}"
                )

    @staticmethod
    def _mark_to_market(
        *,
        positions: dict[tuple[str, str], float],
        marks: dict[tuple[str, str], float],
        ledger: _PnlLedger,
        cash: float,
    ) -> tuple[float, float, float, float]:
        equity = cash
        gross_notional = 0.0
        gross_cash_at_risk = 0.0
        unrealized = 0.0

        for key, position in positions.items():
            if abs(position) <= 1e-12:
                continue
            mark = marks.get(key, 0.0)
            gross_notional += abs(position) * mark
            gross_cash_at_risk += BacktestEngine._cash_at_risk_for_position(
                position=position, mark=mark
            )
            equity += position * mark

            avg_entry = ledger.avg_entry(key)
            if avg_entry is None:
                continue
            if position >= 0.0:
                unrealized += (mark - avg_entry) * position
            else:
                unrealized += (avg_entry - mark) * abs(position)

        return equity, gross_notional, gross_cash_at_risk, unrealized

    @staticmethod
    def _append_exposure_rows(
        *,
        ts: datetime,
        positions: dict[tuple[str, str], float],
        marks: dict[tuple[str, str], float],
        out: list[dict[str, object]],
    ) -> None:
        for (market_id, outcome_id), position in positions.items():
            if abs(position) <= 1e-12:
                continue
            mark = marks.get((market_id, outcome_id), 0.0)
            out.append(
                {
                    "ts": ts,
                    "market_id": market_id,
                    "outcome_id": outcome_id,
                    "position": position,
                    "notional_exposure": abs(position) * mark,
                    "cash_at_risk": BacktestEngine._cash_at_risk_for_position(
                        position=position,
                        mark=mark,
                    ),
                }
            )

    @staticmethod
    def _cash_at_risk_for_position(*, position: float, mark: float) -> float:
        p = min(max(mark, 0.0), 1.0)
        if position >= 0.0:
            return position * (1.0 - p)
        return abs(position) * p

    def _assert_supported_stream(self, bars_df: pl.DataFrame) -> None:
        if self.config.market_id is None:
            return

        market_unique = bars_df["market_id"].n_unique()
        outcome_unique = bars_df["outcome_id"].n_unique()
        venue_unique = bars_df["venue"].n_unique()
        if market_unique > 1 or outcome_unique > 1 or venue_unique > 1:
            raise ValueError(
                "BacktestEngine currently supports a single venue/market/outcome stream per run "
                + "when config.market_id is set. Use stricter filtering or implement portfolio "
                + "routing."
            )

    def _dataset_slice(self) -> DatasetSlice:
        paths: list[str] = []
        if self.config.venue is not None:
            paths.append(
                str(self.config.data_root / self.config.venue.value / "trades" / "*.parquet")
            )
        return DatasetSlice(
            venue=self.config.venue,
            market_ids=[self.config.market_id] if self.config.market_id else [],
            start_ts=self.config.start_ts,
            end_ts=self.config.end_ts,
            source_paths=paths,
        )
