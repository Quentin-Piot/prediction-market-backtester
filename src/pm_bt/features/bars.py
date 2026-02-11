from __future__ import annotations

import polars as pl


def _as_lazy(frame: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    if isinstance(frame, pl.LazyFrame):
        return frame
    return frame.lazy()


def build_bars(
    trades: pl.DataFrame | pl.LazyFrame,
    timeframe: str,
) -> pl.LazyFrame:
    """Build time bars from normalized trade ticks."""
    lf = _as_lazy(trades).with_columns(
        [
            pl.col("ts").cast(pl.Datetime(time_zone="UTC")),
            pl.col("price").cast(pl.Float64),
            pl.col("size").cast(pl.Float64),
        ]
    )

    bars = (
        lf.sort(["venue", "market_id", "outcome_id", "ts"])
        .filter(pl.col("ts").is_not_null())
        .group_by_dynamic(
            index_column="ts",
            every=timeframe,
            period=timeframe,
            group_by=["venue", "market_id", "outcome_id"],
            closed="left",
            label="left",
        )
        .agg(
            [
                pl.col("ts").min().alias("ts_open"),
                pl.col("ts").max().alias("ts_close"),
                pl.col("price").first().alias("open"),
                pl.col("price").max().alias("high"),
                pl.col("price").min().alias("low"),
                pl.col("price").last().alias("close"),
                pl.col("size").sum().alias("volume"),
                (pl.col("price") * pl.col("size")).sum().alias("_notional"),
                pl.len().alias("trade_count"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("volume") > 0.0)
                .then(pl.col("_notional") / pl.col("volume"))
                .otherwise(pl.lit(None))
                .alias("vwap")
            ]
        )
        .drop("_notional", "ts")
        .sort(["venue", "market_id", "outcome_id", "ts_open"])
    )
    return bars


def add_basic_features(
    bars: pl.DataFrame | pl.LazyFrame,
    *,
    momentum_window: int = 3,
    volatility_window: int = 5,
) -> pl.LazyFrame:
    """Add vectorized baseline features on top of bars."""
    lf = _as_lazy(bars).sort(["venue", "market_id", "outcome_id", "ts_open"])
    grouped = ["venue", "market_id", "outcome_id"]

    return lf.with_columns(
        [
            pl.col("close").pct_change().over(grouped).alias("ret_1"),
            pl.col("close")
            .rolling_mean(window_size=momentum_window, min_samples=momentum_window)
            .over(grouped)
            .alias("rolling_mean"),
            pl.col("close")
            .rolling_std(window_size=volatility_window, min_samples=volatility_window)
            .over(grouped)
            .alias("rolling_std"),
            (pl.col("close") / pl.col("close").shift(momentum_window).over(grouped) - 1.0).alias(
                "momentum"
            ),
        ]
    )
