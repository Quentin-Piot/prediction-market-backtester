from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import cast

import pytest

from pm_bt.cli import run_cli
from pm_bt.data import load_trades


@pytest.mark.skipif(
    os.getenv("PM_BT_RUN_REAL_DATA_TESTS") != "1",
    reason="Set PM_BT_RUN_REAL_DATA_TESTS=1 to run integration tests on local real data",
)
def test_real_polymarket_trades_loader_returns_valid_sample(tmp_path: Path) -> None:
    source_root = Path("data")
    trades_dir = source_root / "polymarket" / "trades"
    real_files = sorted(p for p in trades_dir.glob("*.parquet") if not p.name.startswith("._"))
    if not trades_dir.exists() or not real_files:
        pytest.skip("real polymarket trades dataset not available under data/")

    target_root = tmp_path / "data"
    target_trades = target_root / "polymarket" / "trades"
    target_trades.mkdir(parents=True, exist_ok=True)
    first_file = real_files[0]
    os.symlink(first_file.resolve(), target_trades / first_file.name)

    sample = (
        load_trades("polymarket", data_root=target_root)
        .select(["ts", "market_id", "outcome_id", "price", "size", "side"])
        .drop_nulls(["ts", "market_id", "outcome_id", "price", "size"])
        .head(1000)
        .collect()
    )

    assert sample.height > 0
    min_price = cast(float | None, sample["price"].min())
    max_price = cast(float | None, sample["price"].max())
    min_size = cast(float | None, sample["size"].min())

    assert min_price is not None
    assert max_price is not None
    assert min_size is not None

    assert min_price >= 0.0
    assert max_price <= 1.0
    assert min_size > 0.0

    sides = cast(list[str], sample["side"].drop_nulls().to_list())
    assert set(sides).issubset({"buy", "sell", "unknown"})


@pytest.mark.skipif(
    os.getenv("PM_BT_RUN_REAL_DATA_TESTS") != "1",
    reason="Set PM_BT_RUN_REAL_DATA_TESTS=1 to run integration tests on local real data",
)
def test_real_kalshi_cli_backtest_runs_on_small_sample(tmp_path: Path) -> None:
    source_root = Path("data")
    trades_dir = source_root / "kalshi" / "trades"
    real_files = sorted(p for p in trades_dir.glob("*.parquet") if not p.name.startswith("._"))
    if not trades_dir.exists() or not real_files:
        pytest.skip("real kalshi trades dataset not available under data/")

    first_file = real_files[0]
    target_root = tmp_path / "data"
    target_trades = target_root / "kalshi" / "trades"
    target_trades.mkdir(parents=True, exist_ok=True)
    os.symlink(first_file.resolve(), target_trades / first_file.name)

    sample = (
        load_trades("kalshi", data_root=target_root)
        .select(["market_id", "ts"])
        .drop_nulls(["market_id", "ts"])
        .head(1)
        .collect()
    )
    if sample.height == 0:
        pytest.skip("No valid kalshi rows found in sampled real dataset file")

    market_id = cast(str, sample["market_id"][0])
    first_ts = cast(datetime, sample["ts"][0])
    start_ts = first_ts.isoformat()
    end_ts = (first_ts + timedelta(hours=6)).isoformat()

    output_root = tmp_path / "output"
    exit_code = run_cli(
        [
            "backtest",
            "--venue",
            "kalshi",
            "--market",
            market_id,
            "--strategy",
            "momentum",
            "--config",
            "configs/momentum/default.yaml",
            "--data-root",
            str(target_root),
            "--output-root",
            str(output_root),
            "--start-ts",
            start_ts,
            "--end-ts",
            end_ts,
        ]
    )

    assert exit_code == 0
    run_dirs = sorted(path for path in output_root.iterdir() if path.is_dir())
    assert run_dirs
    assert (run_dirs[0] / "results.json").exists()
    assert (run_dirs[0] / "equity.csv").exists()
