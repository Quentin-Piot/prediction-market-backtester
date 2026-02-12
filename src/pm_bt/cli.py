from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import cast

import yaml  # type: ignore[import-untyped]

from pm_bt.backtest import BacktestEngine
from pm_bt.common.models import BacktestConfig
from pm_bt.common.types import Venue
from pm_bt.common.utils import parse_ts_utc, safe_mkdir
from pm_bt.strategies import EventThresholdStrategy, MeanReversionStrategy, MomentumStrategy
from pm_bt.strategies.base import Strategy

logger = logging.getLogger(__name__)

StrategyFactory = Callable[..., Strategy]

_STRATEGY_REGISTRY: dict[str, StrategyFactory] = {
    "momentum": MomentumStrategy,
    "mean_reversion": MeanReversionStrategy,
    "event_threshold": EventThresholdStrategy,
}

_DEFAULT_CONFIG_PATHS: dict[str, Path] = {
    "momentum": Path("configs/momentum/default.yaml"),
    "mean_reversion": Path("configs/mean_reversion/default.yaml"),
    "event_threshold": Path("configs/event_threshold/default.yaml"),
}


def _as_str_object_mapping(value: object, *, field_name: str) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")

    mapping = cast(Mapping[object, object], value)

    output: dict[str, object] = {}
    for key, mapped_value in mapping.items():
        if not isinstance(key, str):
            raise ValueError(f"{field_name} keys must be strings")
        output[key] = mapped_value
    return output


def _load_yaml_config(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    payload = cast(object, yaml.safe_load(path.read_text(encoding="utf-8")))
    return _as_str_object_mapping(payload, field_name="config")


def _resolve_config_path(args: argparse.Namespace) -> Path:
    explicit = cast(str | None, args.config)
    if explicit:
        return Path(explicit)

    strategy_name = cast(str, args.strategy)
    default_path = _DEFAULT_CONFIG_PATHS.get(strategy_name)
    if default_path is None:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    return default_path


def _build_strategy(strategy_name: str, strategy_params: Mapping[str, object]) -> Strategy:
    factory = _STRATEGY_REGISTRY.get(strategy_name)
    if factory is None:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    try:
        return factory(**strategy_params)
    except TypeError as exc:
        raise ValueError(f"Invalid strategy parameters for '{strategy_name}': {exc}") from exc


def _build_backtest_config(
    args: argparse.Namespace, yaml_config: Mapping[str, object]
) -> BacktestConfig:
    """Build a BacktestConfig by merging YAML config with CLI arguments.

    CLI arguments take precedence over YAML values for: venue, market, strategy,
    data_root, output_root, name, start_ts, end_ts, bar_timeframe.
    """
    config_data: dict[str, object] = dict(yaml_config)
    name = cast(str | None, args.name)
    start_ts = cast(str | None, args.start_ts)
    end_ts = cast(str | None, args.end_ts)
    bar_timeframe = cast(str | None, args.bar_timeframe)

    config_data["venue"] = Venue(cast(str, args.venue))
    config_data["market_id"] = cast(str, args.market)
    config_data["strategy_name"] = cast(str, args.strategy)
    config_data["data_root"] = Path(cast(str, args.data_root))
    config_data["output_root"] = Path(cast(str, args.output_root))

    if name:
        config_data["name"] = name
    if start_ts:
        config_data["start_ts"] = parse_ts_utc(start_ts)
    if end_ts:
        config_data["end_ts"] = parse_ts_utc(end_ts)
    if bar_timeframe:
        config_data["bar_timeframe"] = bar_timeframe

    raw_strategy_params = config_data.get("strategy_params")
    config_data["strategy_params"] = _as_str_object_mapping(
        raw_strategy_params,
        field_name="strategy_params",
    )

    return BacktestConfig.model_validate(config_data)


def _run_backtest(args: argparse.Namespace) -> int:
    try:
        config_path = _resolve_config_path(args)
        yaml_config = _load_yaml_config(config_path)
        config = _build_backtest_config(args, yaml_config)
        strategy = _build_strategy(config.strategy_name, config.strategy_params)

        artifacts = BacktestEngine(config=config, strategy=strategy).run()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Backtest failed: %s", exc)
        return 1

    run_dir = config.output_root / artifacts.run_result.run_id
    safe_mkdir(run_dir)

    results_path = run_dir / "results.json"
    equity_path = run_dir / "equity.csv"
    trades_path = run_dir / "trades.csv"

    artifacts.equity_curve.write_csv(equity_path)
    artifacts.fills.write_csv(trades_path)

    artifacts.run_result.artifacts = {
        "results_json": str(results_path),
        "equity_csv": str(equity_path),
        "trades_csv": str(trades_path),
    }
    results_payload = artifacts.run_result.model_dump(mode="json")
    _ = results_path.write_text(
        json.dumps(results_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    logger.info("Backtest completed. Artifacts written to %s", run_dir)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pm-bt")
    subparsers = parser.add_subparsers(dest="command", required=True)

    backtest = subparsers.add_parser("backtest", help="Run a backtest and persist run artifacts")
    _ = backtest.add_argument("--venue", choices=[venue.value for venue in Venue], required=True)
    _ = backtest.add_argument("--market", required=True)
    _ = backtest.add_argument("--strategy", required=True)
    _ = backtest.add_argument("--config", required=False)
    _ = backtest.add_argument("--name", required=False)
    _ = backtest.add_argument("--data-root", default="data")
    _ = backtest.add_argument("--output-root", default="output/runs")
    _ = backtest.add_argument("--start-ts", required=False)
    _ = backtest.add_argument("--end-ts", required=False)
    _ = backtest.add_argument("--bar-timeframe", required=False)
    _ = backtest.set_defaults(handler=_run_backtest)
    return parser


def run_cli(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = cast(Callable[[argparse.Namespace], int], args.handler)
    return handler(args)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raise SystemExit(run_cli())
