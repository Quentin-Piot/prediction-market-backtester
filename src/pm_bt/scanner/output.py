from __future__ import annotations

import json
from html import escape
from pathlib import Path

import polars as pl

from pm_bt.scanner.models import Alert


def write_alerts_json(alerts: list[Alert], path: Path) -> None:
    """Serialize alerts to JSON via Pydantic ``model_dump``."""
    payload = [a.model_dump(mode="json") for a in alerts]
    _ = path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_alerts_csv(alerts: list[Alert], path: Path) -> None:
    """Write alerts to CSV.  ``supporting_stats`` is JSON-encoded."""
    if not alerts:
        pl.DataFrame(
            schema=[
                ("alert_id", pl.Utf8),
                ("market_id", pl.Utf8),
                ("ts", pl.Utf8),
                ("venue", pl.Utf8),
                ("reason", pl.Utf8),
                ("severity", pl.Utf8),
                ("supporting_stats_json", pl.Utf8),
            ]
        ).write_csv(path)
        return

    rows = [
        {
            "alert_id": a.alert_id,
            "market_id": a.market_id,
            "ts": a.ts.isoformat(),
            "venue": a.venue.value,
            "reason": a.reason,
            "severity": a.severity.value,
            "supporting_stats_json": json.dumps(a.supporting_stats, sort_keys=True),
        }
        for a in alerts
    ]
    pl.DataFrame(rows).write_csv(path)


_SEVERITY_COLORS: dict[str, str] = {
    "high": "#f87171",
    "medium": "#fb923c",
    "low": "#fbbf24",
}


def write_alerts_html(alerts: list[Alert], path: Path) -> None:
    """Write a minimal HTML report with a colour-coded table."""
    rows_html = ""
    for a in alerts:
        colour = _SEVERITY_COLORS.get(a.severity.value, "#e5e7eb")
        stats = ", ".join(f"{k}={v:.4g}" for k, v in sorted(a.supporting_stats.items()))
        rows_html += (
            f"<tr style='background:{colour}'>"
            f"<td>{escape(a.market_id)}</td>"
            f"<td>{escape(a.ts.isoformat())}</td>"
            f"<td>{escape(a.venue.value)}</td>"
            f"<td>{escape(a.reason)}</td>"
            f"<td>{escape(a.severity.value)}</td>"
            f"<td>{escape(stats)}</td>"
            f"</tr>\n"
        )

    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<title>Scanner Alerts</title>"
        "<style>table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #ccc;padding:4px 8px;text-align:left}"
        "th{background:#374151;color:#fff}</style></head><body>"
        f"<h1>Scanner Alerts ({len(alerts)})</h1>"
        "<table><tr><th>Market</th><th>Time</th><th>Venue</th>"
        "<th>Reason</th><th>Severity</th><th>Stats</th></tr>"
        f"{rows_html}</table></body></html>"
    )
    _ = path.write_text(html, encoding="utf-8")
