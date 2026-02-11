from __future__ import annotations

import polars as pl

from pm_bt.features import add_basic_features, build_bars


def test_build_bars_computes_ohlcv_and_vwap(trades_df: pl.DataFrame) -> None:
    trades = trades_df.filter(pl.col("market_id") == "KX-RAIN-2026-01-01")
    bars = build_bars(trades, "2m").collect()

    assert bars.height == 2
    assert "ts" not in bars.columns, "residual 'ts' column from group_by_dynamic must be dropped"

    first_bar = bars.row(0, named=True)
    assert first_bar["open"] == 0.44
    assert first_bar["high"] == 0.46
    assert first_bar["low"] == 0.44
    assert first_bar["close"] == 0.46
    assert first_bar["volume"] == 200.0
    assert first_bar["trade_count"] == 2
    assert first_bar["vwap"] == (0.44 * 120.0 + 0.46 * 80.0) / 200.0


def test_add_basic_features_adds_expected_columns(trades_df: pl.DataFrame) -> None:
    trades = trades_df.filter(pl.col("market_id") == "KX-RAIN-2026-01-01")
    bars = build_bars(trades, "1m")
    featured = add_basic_features(bars, momentum_window=2, volatility_window=2).collect()

    expected_columns = {"ret_1", "rolling_mean", "rolling_std", "momentum"}
    assert expected_columns.issubset(set(featured.columns))
    assert featured.height == 4
