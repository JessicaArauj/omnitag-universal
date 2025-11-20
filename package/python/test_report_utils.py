"""Utilities to sanitize pytest JSON reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_REPORT = Path(__file__).resolve().parents[2] / 'output' / 'test_results.json'


def sanitize_test_report(report_path: Path) -> Path:
    """Remove sensitive or noisy keys (e.g., project root) from the test report."""
    if not report_path.exists():
        raise FileNotFoundError(f'Test report not found: {report_path}')

    data = json.loads(report_path.read_text(encoding='utf-8'))
    if 'root' in data:
        data.pop('root', None)
        report_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Sanitize pytest JSON report (remove root path, etc.).'
    )
    parser.add_argument(
        '--report',
        type=Path,
        default=DEFAULT_REPORT,
        help='Path to output/test_results.json (default: output/test_results.json).',
    )
    args = parser.parse_args()
    sanitize_test_report(args.report)
    print(f'Sanitized test report: {args.report}')


if __name__ == '__main__':
    main()
