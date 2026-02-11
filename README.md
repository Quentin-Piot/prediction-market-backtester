# Prediction Market Backtester

Quant-style backtesting engine for prediction markets (Polymarket + Kalshi), focused on correctness, reproducibility, and performance.

## Project Intent

- Build an engine-first backtesting and analytics system, with UI as an optional control/exploration layer.
- Reuse robust historical ingestion patterns from `prediction-market-analysis`.
- Prioritize correctness, reproducibility, and explicit execution assumptions.
- Keep strategy/execution/accounting logic in the engine (CLI/API), not in UI code.

## Domain First: Prediction Markets != Traditional Markets

This project treats prediction markets as a distinct domain with different mechanics from traditional financial markets:

- Binary settlement (`0/1`) and event resolution, instead of indefinite mark-to-market assets.
- Prices interpreted as market-implied probabilities, which can deviate from true probabilities due to microstructure and liquidity effects.
- Thin/uneven liquidity and episodic flow around news/events, requiring careful tradability assumptions.
- Outcome structure (Yes/No, mutually exclusive outcomes) that creates market-consistency checks.
- Market structure can vary by venue and era (for example CLOB-style vs AMM-style execution assumptions).
- Forecasting quality metrics (Brier/log loss) are tracked separately from trading performance metrics (PnL/drawdown).

A dedicated note is available in `docs/prediction-markets-vs-tradfi.md`.

## Quickstart

```bash
uv sync --dev
make lint
make typecheck
make test
```

## Data Setup

```bash
cp .env.example .env
# set DATA_URL (and optionally DATA_SHA256)
make setup
```

`make setup` is idempotent:
- downloads `data.tar.zst` only if missing
- optionally verifies `DATA_SHA256`
- extracts to `data/`

## Data Indexing

```bash
# all venues, markets + trades
make index

# examples
make index SOURCE=kalshi MODE=markets
make index SOURCE=polymarket MODE=trades
```

Notes:
- `make index` installs the extra `index` dependency group and runs vendored indexers.
- For Polymarket trades indexing, set `POLYGON_RPC`.

## Structure

- `src/pm_bt/common/`: shared models/types/utils
- `src/pm_bt/data/`: data loading
- `src/pm_bt/features/`: bars and indicators
- `src/pm_bt/execution/`: execution simulation
- `src/pm_bt/strategies/`: strategy implementations
- `src/pm_bt/backtest/`: engine and metrics
- `src/pm_bt/reporting/`: artifacts and plots
- `vendor/prediction-market-analysis/`: vendored data indexers/schemas reference (MIT)

## Roadmap

- Build plan and acceptance criteria: `ROADMAP.md`
- Scope and engineering rules: `SKILLS.md`
