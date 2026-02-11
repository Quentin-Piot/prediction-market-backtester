from __future__ import annotations

import csv
from pathlib import Path

import pytest

from pm_bt.common.models import BacktestConfig, DatasetSlice, Market, RunResult, TradeTick
from pm_bt.common.types import MarketStructure, ResolutionOutcome, TradeSide, Venue
from pm_bt.common.utils import make_run_id, parse_ts_utc


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def test_markets_fixture_parses_and_contains_resolved_market(fixtures_dir: Path) -> None:
    markets_path = fixtures_dir / "markets_small.csv"
    with markets_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2

    markets: list[Market] = []
    for row in rows:
        winning_outcome = row["winning_outcome"].strip() or None
        market = Market(
            market_id=row["market_id"],
            venue=Venue(row["venue"]),
            outcome_id=row["outcome_id"],
            question=row["question"] or None,
            category=row["category"] or None,
            close_ts=parse_ts_utc(row["close_ts"]) if row["close_ts"] else None,
            resolved=_as_bool(row["resolved"]),
            winning_outcome=ResolutionOutcome(winning_outcome) if winning_outcome else None,
            resolved_ts=parse_ts_utc(row["resolved_ts"]) if row["resolved_ts"] else None,
            market_structure=MarketStructure(row["market_structure"]),
        )
        markets.append(market)

    assert any(m.resolved for m in markets)
    assert markets[0].winning_outcome == ResolutionOutcome.YES


def test_trades_fixture_parses_and_prices_are_bounded(fixtures_dir: Path) -> None:
    trades_path = fixtures_dir / "trades_small.csv"
    with trades_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 8

    trades = [
        TradeTick(
            ts=parse_ts_utc(row["ts"]),
            market_id=row["market_id"],
            outcome_id=row["outcome_id"],
            venue=Venue(row["venue"]),
            price=float(row["price"]),
            size=float(row["size"]),
            side=TradeSide(row["side"]),
            trade_id=row["trade_id"],
            fee_paid=float(row["fee_paid"]),
        )
        for row in rows
    ]
    assert min(t.price for t in trades) >= 0.0
    assert max(t.price for t in trades) <= 1.0


def test_runresult_enforces_forecasting_trading_metric_separation() -> None:
    with pytest.raises(ValueError):
        _ = RunResult(
            run_id=make_run_id(short_hash="abc123"),
            created_at=parse_ts_utc("2026-01-01T00:00:00Z"),
            config=BacktestConfig(
                strategy_name="momentum",
                initial_cash=10_000,
                max_position_size=100,
                max_gross_exposure=10_000,
            ),
            dataset_slice=DatasetSlice(),
            trading_metrics={"sharpe": 1.2},
            forecasting_metrics={"sharpe": 0.2},
        )
