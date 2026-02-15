from __future__ import annotations

import logging

import polars as pl

from pm_bt.common.types import AlertSeverity, Venue
from pm_bt.scanner.models import Alert, make_alert_id

logger = logging.getLogger(__name__)


def check_whale_trades(
    trades: pl.LazyFrame,
    *,
    rolling_window: str,
    size_multiplier: float,
) -> list[Alert]:
    """Detect trades whose size exceeds *size_multiplier* × rolling average.

    The rolling baseline is computed per ``(market_id, venue)`` over
    *rolling_window* (a Polars duration string such as ``"1h"``).
    """
    sorted_trades = trades.sort(["market_id", "venue", "ts"])

    with_rolling = sorted_trades.with_columns(
        pl.col("size")
        .rolling_mean_by("ts", window_size=rolling_window, closed="left")
        .over(["market_id", "venue"])
        .alias("rolling_avg_size"),
    )

    whales = (
        with_rolling.filter(
            pl.col("rolling_avg_size").is_not_null()
            & (pl.col("rolling_avg_size") > 0)
            & (pl.col("size") > size_multiplier * pl.col("rolling_avg_size"))
        )
        .with_columns(
            (pl.col("size") / pl.col("rolling_avg_size")).alias("size_ratio"),
        )
        .collect()
    )

    alerts: list[Alert] = []
    for row in whales.iter_rows(named=True):
        ratio: float = row["size_ratio"]
        if ratio > 10.0:
            severity = AlertSeverity.HIGH
        elif ratio > 5.0:
            severity = AlertSeverity.MEDIUM
        else:
            severity = AlertSeverity.LOW

        ts = row["ts"]
        market_id: str = row["market_id"]
        venue = Venue(row["venue"])
        alerts.append(
            Alert(
                alert_id=make_alert_id("whale_trade", market_id, ts),
                market_id=market_id,
                ts=ts,
                venue=venue,
                reason="whale_trade",
                severity=severity,
                supporting_stats={
                    "trade_size": row["size"],
                    "rolling_avg_size": row["rolling_avg_size"],
                    "size_ratio": ratio,
                    "price": row["price"],
                },
            )
        )
    return alerts


def check_price_impact(
    trades: pl.LazyFrame,
    *,
    impact_threshold: float,
) -> list[Alert]:
    """Detect trades with abnormally high price impact per unit size.

    ``impact_score = |price - prev_price| / size``

    Only consecutive trades within the same ``(market_id, venue)`` are
    compared.  The first trade in each group is skipped (no previous price).
    """
    sorted_trades = trades.sort(["market_id", "venue", "ts"])

    with_impact = sorted_trades.with_columns(
        pl.col("price").diff().over(["market_id", "venue"]).alias("price_diff"),
    ).with_columns(
        (pl.col("price_diff").abs() / pl.col("size")).alias("impact_score"),
    )

    impacts = with_impact.filter(
        pl.col("impact_score").is_not_null()
        & pl.col("impact_score").is_finite()
        & (pl.col("impact_score") > impact_threshold)
        & (pl.col("size") > 0)
    ).collect()

    alerts: list[Alert] = []
    for row in impacts.iter_rows(named=True):
        score: float = row["impact_score"]
        severity = AlertSeverity.HIGH if score > 3 * impact_threshold else AlertSeverity.MEDIUM

        ts = row["ts"]
        market_id: str = row["market_id"]
        venue = Venue(row["venue"])
        alerts.append(
            Alert(
                alert_id=make_alert_id("price_impact", market_id, ts),
                market_id=market_id,
                ts=ts,
                venue=venue,
                reason="price_impact",
                severity=severity,
                supporting_stats={
                    "price_diff": abs(row["price_diff"]),
                    "size": row["size"],
                    "impact_score": score,
                    "price": row["price"],
                },
            )
        )
    return alerts
