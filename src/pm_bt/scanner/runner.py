from __future__ import annotations

import logging
from time import perf_counter

import polars as pl

from pm_bt.data import load_markets, load_trades
from pm_bt.scanner.checks import (
    check_complement_sum,
    check_mutually_exclusive,
    check_price_impact,
    check_whale_trades,
)
from pm_bt.scanner.models import Alert, ScannerConfig

logger = logging.getLogger(__name__)


def run_scanner(
    config: ScannerConfig,
    *,
    trades: pl.LazyFrame | None = None,
    markets: pl.LazyFrame | None = None,
) -> list[Alert]:
    """Orchestrate all checks and return a deduplicated, sorted alert list.

    *trades* and *markets* can be injected for testing.  When ``None``,
    data is loaded from *config.data_root* for each venue in *config.venues*.
    """
    logger.info("Scanner starting: venues=%s", [v.value for v in config.venues])

    if trades is None:
        trades = _load_all_trades(config)
    if markets is None:
        markets = _load_all_markets(config)

    # Single sort shared by all checks that need ordered trades.
    trades = trades.sort(["market_id", "venue", "ts"])

    all_alerts: list[Alert] = []

    # --- Consistency checks ---
    t0 = perf_counter()
    logger.info("Running complement_sum check...")
    all_alerts.extend(check_complement_sum(trades, tolerance=config.complement_sum_tolerance))
    logger.info("  complement_sum: %d alert(s) in %.1fs", len(all_alerts), perf_counter() - t0)

    t0 = perf_counter()
    n_before = len(all_alerts)
    logger.info("Running mutually_exclusive check...")
    all_alerts.extend(
        check_mutually_exclusive(markets, trades, tolerance=config.mutually_exclusive_tolerance)
    )
    logger.info(
        "  mutually_exclusive: %d alert(s) in %.1fs",
        len(all_alerts) - n_before,
        perf_counter() - t0,
    )

    # --- Whale / impact checks ---
    t0 = perf_counter()
    n_before = len(all_alerts)
    logger.info("Running whale_trades check...")
    all_alerts.extend(
        check_whale_trades(
            trades,
            rolling_window=config.whale_rolling_window,
            size_multiplier=config.whale_size_multiplier,
        )
    )
    logger.info(
        "  whale_trades: %d alert(s) in %.1fs", len(all_alerts) - n_before, perf_counter() - t0
    )

    t0 = perf_counter()
    n_before = len(all_alerts)
    logger.info("Running price_impact check...")
    all_alerts.extend(check_price_impact(trades, impact_threshold=config.impact_score_threshold))
    logger.info(
        "  price_impact: %d alert(s) in %.1fs", len(all_alerts) - n_before, perf_counter() - t0
    )

    # Deduplicate by alert_id (deterministic: last-writer-wins within fixed
    # check ordering) and sort for reproducibility.
    seen: dict[str, Alert] = {}
    for alert in all_alerts:
        seen[alert.alert_id] = alert
    result = sorted(seen.values(), key=lambda a: (a.ts, a.market_id, a.reason))
    logger.info("Total: %d unique alert(s) after deduplication", len(result))
    return result


def _resolve_market_ids(config: ScannerConfig) -> list[str] | None:
    """Return an explicit list of market_ids to scan, or None for all."""
    if config.market_ids:
        return config.market_ids
    if config.top_n is not None:
        logger.info("Selecting top-%d markets by volume per venue...", config.top_n)
        all_ids: list[str] = []
        for venue in config.venues:
            ranks = (
                load_trades(
                    venue,
                    data_root=config.data_root,
                    start_ts=config.start_ts,
                    end_ts=config.end_ts,
                )
                .group_by("market_id")
                .agg(pl.col("size").sum().alias("volume_total"))
                .sort("volume_total", descending=True)
                .head(config.top_n)
                .collect()
            )
            ids = ranks["market_id"].to_list()
            logger.info("  %s: %d market(s) selected", venue.value, len(ids))
            all_ids.extend(ids)
        return all_ids
    return None


def _load_all_trades(config: ScannerConfig) -> pl.LazyFrame:
    market_ids = _resolve_market_ids(config)

    t0 = perf_counter()
    frames: list[pl.LazyFrame] = []
    for venue in config.venues:
        lf = load_trades(
            venue,
            data_root=config.data_root,
            start_ts=config.start_ts,
            end_ts=config.end_ts,
        )
        if market_ids:
            lf = lf.filter(pl.col("market_id").is_in(market_ids))
        frames.append(lf)
    if not frames:
        msg = "No venues configured"
        raise ValueError(msg)
    result = pl.concat(frames)
    logger.info("Trades loaded (lazy plan built) in %.1fs", perf_counter() - t0)
    return result


def _load_all_markets(config: ScannerConfig) -> pl.LazyFrame:
    t0 = perf_counter()
    frames: list[pl.LazyFrame] = []
    for venue in config.venues:
        lf = load_markets(
            venue,
            data_root=config.data_root,
            start_ts=config.start_ts,
            end_ts=config.end_ts,
        )
        if config.market_ids:
            lf = lf.filter(pl.col("market_id").is_in(config.market_ids))
        frames.append(lf)
    if not frames:
        msg = "No venues configured"
        raise ValueError(msg)
    result = pl.concat(frames)
    logger.info("Markets loaded (lazy plan built) in %.1fs", perf_counter() - t0)
    return result
