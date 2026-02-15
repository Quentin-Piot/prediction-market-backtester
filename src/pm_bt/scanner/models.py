from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import Field

from pm_bt.common.models import DomainModel
from pm_bt.common.types import AlertId, AlertSeverity, MarketId, Venue


class Alert(DomainModel):
    """A single scanner alert — deterministic by (reason, market_id, ts)."""

    alert_id: AlertId
    market_id: MarketId
    ts: datetime
    venue: Venue
    reason: str
    severity: AlertSeverity
    supporting_stats: dict[str, float] = Field(default_factory=dict)


class ScannerConfig(DomainModel):
    """Configuration for the alpha scanner."""

    # Data inputs
    data_root: Path = Path("data")
    venues: list[Venue] = Field(default_factory=lambda: [Venue.KALSHI, Venue.POLYMARKET])
    market_ids: list[MarketId] = Field(default_factory=list)
    start_ts: datetime | None = None
    end_ts: datetime | None = None

    # Market selection
    top_n: int | None = Field(default=None, gt=0)

    # Consistency thresholds
    complement_sum_tolerance: float = Field(default=0.05, gt=0.0, le=1.0)
    mutually_exclusive_tolerance: float = Field(default=0.10, gt=0.0, le=1.0)

    # Whale / impact thresholds
    whale_rolling_window: str = "1h"
    whale_size_multiplier: float = Field(default=20.0, gt=0.0)
    impact_score_threshold: float = Field(default=0.05, gt=0.0)

    # Output
    output_dir: Path = Path("output/scans")
    emit_html: bool = False


def make_alert_id(reason: str, market_id: str, ts: datetime) -> AlertId:
    """Deterministic alert identifier."""
    return f"{reason}:{market_id}:{ts.isoformat()}"
