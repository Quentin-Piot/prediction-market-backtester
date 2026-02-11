from __future__ import annotations

import os
from pathlib import Path

import pytest

from pm_bt.data import load_trades


@pytest.mark.skipif(
    os.getenv("PM_BT_RUN_REAL_DATA_TESTS") != "1",
    reason="Set PM_BT_RUN_REAL_DATA_TESTS=1 to run integration tests on local real data",
)
def test_real_polymarket_trades_loader_returns_valid_sample(tmp_path: Path) -> None:
    source_root = Path("data")
    trades_dir = source_root / "polymarket" / "trades"
    if not trades_dir.exists() or not any(trades_dir.glob("*.parquet")):
        pytest.skip("real polymarket trades dataset not available under data/")

    target_root = tmp_path / "data"
    target_trades = target_root / "polymarket" / "trades"
    target_trades.mkdir(parents=True, exist_ok=True)
    first_file = sorted(trades_dir.glob("*.parquet"))[0]
    os.symlink(first_file.resolve(), target_trades / first_file.name)

    sample = (
        load_trades("polymarket", data_root=target_root)
        .select(["ts", "market_id", "outcome_id", "price", "size", "side"])
        .drop_nulls(["ts", "market_id", "outcome_id", "price", "size"])
        .head(1000)
        .collect()
    )

    assert sample.height > 0
    assert sample["price"].min() >= 0.0
    assert sample["price"].max() <= 1.0
    assert sample["size"].min() > 0.0
    assert set(sample["side"].drop_nulls().to_list()).issubset({"buy", "sell", "unknown"})
