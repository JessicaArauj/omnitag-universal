"""Compatibility wrapper for the legacy dashboard report generator."""

from __future__ import annotations

from pathlib import Path

from .config import OUTPUT_DIR
from .save_dashboard_snapshot import (
    SNAPSHOT_NAME,
    save_dashboard_snapshot,
)

REPORT_NAME = 'dashboard_report.html'


def _as_path(value: Path | None) -> Path:
    if value is None:
        return OUTPUT_DIR
    if isinstance(value, Path):
        return value
    return Path(value)


def save_dashboard_report(
    metrics_path: Path | None = None,
    output_dir: Path | None = None,
    classified_data_path: Path | None = None,
) -> Path:
    """Generate the dashboard report HTML (legacy name)."""
    snapshot_path = save_dashboard_snapshot(
        metrics_path=metrics_path,
        output_dir=output_dir,
        classified_data_path=classified_data_path,
    )
    target_dir = _as_path(output_dir)
    report_path = target_dir / REPORT_NAME
    html = snapshot_path.read_text(encoding='utf-8')
    html = html.replace('Dashboard Snapshot', 'Dashboard Report', 1)
    report_path.write_text(html, encoding='utf-8')
    return report_path


__all__ = ['save_dashboard_report']
