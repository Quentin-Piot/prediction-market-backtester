from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pm_bt.common.types import (
    MarketId,
    MarketStructure,
    OrderSide,
    OutcomeId,
    ResolutionOutcome,
    RunId,
    TradeSide,
    Venue,
)


class DomainModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", validate_assignment=True)


class Market(DomainModel):
    market_id: MarketId
    venue: Venue
    outcome_id: OutcomeId
    question: str | None = None
    category: str | None = None
    close_ts: datetime | None = None
    resolved: bool = False
    winning_outcome: ResolutionOutcome | None = None
    resolved_ts: datetime | None = None
    market_structure: MarketStructure = MarketStructure.UNKNOWN
    metadata: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_resolution_fields(self) -> Market:
        if self.resolved and self.winning_outcome is None:
            raise ValueError("winning_outcome is required for resolved markets")
        if self.resolved and self.winning_outcome == ResolutionOutcome.UNRESOLVED:
            raise ValueError("winning_outcome cannot be unresolved when resolved=True")
        if not self.resolved and self.winning_outcome is not None:
            raise ValueError("winning_outcome must be None when resolved=False")
        return self


class TradeTick(DomainModel):
    ts: datetime
    market_id: MarketId
    outcome_id: OutcomeId
    venue: Venue
    price: float = Field(ge=0.0, le=1.0)
    size: float = Field(gt=0.0)
    side: TradeSide = TradeSide.UNKNOWN
    trade_id: str | None = None
    fee_paid: float = Field(default=0.0, ge=0.0)
    metadata: dict[str, object] = Field(default_factory=dict)


class Bar(DomainModel):
    ts_open: datetime
    ts_close: datetime
    market_id: MarketId
    outcome_id: OutcomeId
    venue: Venue
    open: float = Field(ge=0.0, le=1.0)
    high: float = Field(ge=0.0, le=1.0)
    low: float = Field(ge=0.0, le=1.0)
    close: float = Field(ge=0.0, le=1.0)
    volume: float = Field(ge=0.0)
    vwap: float | None = Field(default=None, ge=0.0, le=1.0)
    trade_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_ohlc(self) -> Bar:
        if self.high < self.low:
            raise ValueError("high must be >= low")
        if not (self.low <= self.open <= self.high):
            raise ValueError("open must be between low and high")
        if not (self.low <= self.close <= self.high):
            raise ValueError("close must be between low and high")
        if self.ts_close < self.ts_open:
            raise ValueError("ts_close must be >= ts_open")
        return self


class OrderIntent(DomainModel):
    ts: datetime
    market_id: MarketId
    outcome_id: OutcomeId
    venue: Venue
    side: OrderSide
    qty: float = Field(gt=0.0)
    limit_price: float | None = Field(default=None, ge=0.0, le=1.0)
    reason: str | None = None
    target_position: float | None = None


class Fill(DomainModel):
    ts_fill: datetime
    market_id: MarketId
    outcome_id: OutcomeId
    venue: Venue
    side: OrderSide
    qty_filled: float = Field(gt=0.0)
    price_fill: float = Field(ge=0.0, le=1.0)
    fees: float = Field(default=0.0, ge=0.0)
    slippage_cost: float = Field(default=0.0, ge=0.0)
    latency_ms: int = Field(default=0, ge=0)

    @property
    def notional(self) -> float:
        return self.qty_filled * self.price_fill


class BacktestConfig(DomainModel):
    name: str = "default"
    venue: Venue | None = None
    market_id: MarketId | None = None
    start_ts: datetime | None = None
    end_ts: datetime | None = None
    bar_timeframe: str = "1m"
    strategy_name: str = "momentum"
    strategy_params: dict[str, object] = Field(default_factory=dict)
    fee_bps: float = Field(default=0.0, ge=0.0)
    slippage_bps: float = Field(default=0.0, ge=0.0)
    latency_bars: int = Field(default=0, ge=0)
    max_position_size: float = Field(default=100.0, gt=0.0)
    max_gross_exposure: float = Field(
        default=10_000.0,
        gt=0.0,
        description="Gross risk limit under cash-at-risk semantics for binary payoffs.",
    )
    drawdown_stop_pct: float | None = Field(default=None, gt=0.0, lt=1.0)
    initial_cash: float = Field(default=10_000.0, gt=0.0)
    data_root: Path = Path("data")
    output_root: Path = Path("output/runs")

    @model_validator(mode="after")
    def validate_date_range(self) -> BacktestConfig:
        if self.start_ts and self.end_ts and self.start_ts >= self.end_ts:
            raise ValueError("start_ts must be before end_ts")
        return self


class DatasetSlice(DomainModel):
    venue: Venue | None = None
    market_ids: list[MarketId] = Field(default_factory=list)
    start_ts: datetime | None = None
    end_ts: datetime | None = None
    source_paths: list[str] = Field(default_factory=list)


class RunTimings(DomainModel):
    load_s: float = Field(default=0.0, ge=0.0)
    features_s: float = Field(default=0.0, ge=0.0)
    execution_s: float = Field(default=0.0, ge=0.0)
    reporting_s: float = Field(default=0.0, ge=0.0)
    total_s: float = Field(default=0.0, ge=0.0)


class RunResult(DomainModel):
    run_id: RunId
    created_at: datetime
    config: BacktestConfig
    dataset_slice: DatasetSlice
    git_commit: str | None = None
    trading_metrics: dict[str, float] = Field(default_factory=dict)
    forecasting_metrics: dict[str, float] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)
    timings: RunTimings = Field(default_factory=RunTimings)

    @model_validator(mode="after")
    def validate_metric_namespaces(self) -> RunResult:
        overlap = set(self.trading_metrics).intersection(self.forecasting_metrics)
        if overlap:
            joined = ", ".join(sorted(overlap))
            message = "metrics must be separated (same names present in trading and forecasting)"
            raise ValueError(f"{message}: {joined}")
        return self
