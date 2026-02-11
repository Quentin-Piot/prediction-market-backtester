from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture()
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture()
def trades_df() -> pl.DataFrame:
    """Load the deterministic trades fixture as a typed Polars DataFrame."""
    df = pl.read_csv(FIXTURES_DIR / "trades_small.csv")
    return df.with_columns(
        [
            pl.col("ts").str.to_datetime(strict=False, time_zone="UTC"),
            pl.col("price").cast(pl.Float64),
            pl.col("size").cast(pl.Float64),
            pl.col("fee_paid").cast(pl.Float64),
        ]
    )
