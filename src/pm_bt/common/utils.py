from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path


def ensure_utc(ts: datetime) -> datetime:
    """Normalize a datetime to timezone-aware UTC."""
    if ts.tzinfo is None:
        return ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC)


def parse_ts_utc(value: datetime | str | int | float) -> datetime:
    """Parse a timestamp input into timezone-aware UTC datetime."""
    if isinstance(value, datetime):
        return ensure_utc(value)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=UTC)
    normalized = value.replace("Z", "+00:00")
    return ensure_utc(datetime.fromisoformat(normalized))


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_git_commit_hash(cwd: Path | None = None) -> str | None:
    """Return short commit hash if repository metadata is available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def make_run_id(
    now: datetime | None = None,
    short_hash: str | None = None,
) -> str:
    ts = ensure_utc(now or datetime.now(tz=UTC))
    suffix = short_hash or "nogit"
    return f"{ts.strftime('%Y%m%d_%H%M%S')}_{suffix}"
