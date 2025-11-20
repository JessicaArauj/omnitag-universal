"""Generate a static HTML snapshot of the metrics dashboard."""

from __future__ import annotations

import html
import json
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

from .config import METRICS_FILE, OUTPUT_DIR, OUTPUT_FILE

SNAPSHOT_NAME = 'dashboard_snapshot.html'


def _load_metrics(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f'Metrics file not found at {path}')
    return json.loads(path.read_text(encoding='utf-8'))


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f'{value:.4f}'
    if value is None:
        return 'n/a'
    return str(value)


def _render_list_section(title: str, items: Dict[str, Any]) -> str:
    rows = ''.join(
        f'<li><strong>{html.escape(key)}:</strong> {html.escape(_format_value(value))}</li>'
        for key, value in items.items()
        if value is not None
    )
    if not rows:
        rows = '<li><em>No data available</em></li>'
    return f'<section><h2>{html.escape(title)}</h2><ul>{rows}</ul></section>'


def _render_table_section(title: str, rows: Dict[str, Any]) -> str:
    if not rows:
        return f'<section><h2>{html.escape(title)}</h2><p><em>No data available.</em></p></section>'
    header = '<tr><th>Key</th><th>Value</th></tr>'
    body = ''.join(
        f'<tr><td>{html.escape(str(key))}</td><td>{html.escape(_format_value(value))}</td></tr>'
        for key, value in rows.items()
    )
    table = f'<table>{header}{body}</table>'
    return f'<section><h2>{html.escape(title)}</h2>{table}</section>'


def _render_report_section(report: Any) -> str:
    if isinstance(report, dict):
        payload = html.escape(json.dumps(report, ensure_ascii=False, indent=2))
        content = f'<pre>{payload}</pre>'
    elif report:
        content = f'<pre>{html.escape(str(report))}</pre>'
    else:
        content = '<p><em>No classification report available.</em></p>'
    return f'<section><h2>Classification Report</h2>{content}</section>'


def _render_feature_importance(importance: Any) -> str:
    if not isinstance(importance, dict) or not importance:
        return '<section><h2>Feature Importance</h2><p><em>No data available.</em></p></section>'
    tables = []
    for label, rows in importance.items():
        header = '<tr><th>Term</th><th>Weight</th></tr>'
        body = ''.join(
            f'<tr><td>{html.escape(str(term))}</td><td>{html.escape(_format_value(weight))}</td></tr>'
            for term, weight in rows
        )
        tables.append(
            f'<h3>{html.escape(label)}</h3><table>{header}{body}</table>'
        )
    return '<section><h2>Feature Importance</h2>' + ''.join(tables) + '</section>'


def _normalize_name(value: str) -> str:
    normalized = (
        unicodedata.normalize('NFKD', str(value))
        .encode('ascii', 'ignore')
        .decode()
    )
    return ''.join(ch.lower() for ch in normalized if ch.isalnum())


def _load_classification_summary(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None

    df = pd.read_excel(path)
    if df.empty:
        return None

    category_col = None
    confidence_col = None
    for column in df.columns:
        key = _normalize_name(column)
        if key in {'categoria', 'categorizacao', 'possibilidade'}:
            category_col = column
        if key in {'confianca', 'confiancadomodelo'}:
            confidence_col = column
    if not category_col:
        return None

    counts = df[category_col].value_counts().to_dict()
    avg_confidence = (
        float(df[confidence_col].mean()) if confidence_col else None
    )
    return {'counts': counts, 'average_confidence': avg_confidence}


def _render_category_section(
    summary: Dict[str, Any] | None,
) -> Tuple[str, str]:
    if not summary or not summary.get('counts'):
        section = (
            '<section><h2>Category Distribution</h2>'
            '<p><em>No data available.</em></p></section>'
        )
        return section, ''

    counts: Dict[str, int] = summary['counts']
    labels = list(counts.keys())
    values = list(counts.values())
    avg_conf = summary.get('average_confidence')
    extra = ''
    if avg_conf is not None:
        extra = f'<p>Average confidence: <strong>{avg_conf:.2f}</strong></p>'

    rows = ''.join(
        f'<tr><td>{html.escape(label)}</td><td>{value}</td></tr>'
        for label, value in counts.items()
    )
    chart_canvas = '<canvas id="categoryChart" height="140"></canvas>'
    table = (
        f'<table><tr><th>Category</th><th>Count</th></tr>{rows}</table>'
    )
    section = (
        '<section><h2>Category Distribution</h2>'
        f'{extra}{chart_canvas}{table}</section>'
    )
    chart_template = """
    <script>
    const ctx = document.getElementById('categoryChart').getContext('2d');
    new Chart(ctx, {
      type: 'bar',
      data: {
        labels: __LABELS__,
        datasets: [{
          label: 'Records',
          data: __VALUES__,
          backgroundColor: ['#4CAF50','#FFC107','#F44336','#2196F3','#9C27B0']
        }]
      },
      options: {
        responsive: true,
        plugins: {
          legend: { display: false }
        },
        scales: {
          y: { beginAtZero: true, ticks: { precision: 0 } }
        }
      }
    });
    </script>
    """
    chart_script = (
        chart_template
        .replace('__LABELS__', json.dumps(labels, ensure_ascii=False))
        .replace('__VALUES__', json.dumps(values))
    )
    return section, chart_script


def _build_html(
    metrics: Dict[str, Any],
    classification_summary: Dict[str, Any] | None,
) -> str:
    timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
    summary = {
        'Backend': metrics.get('backend'),
        'Test size': metrics.get('test_size'),
        'Records classified': metrics.get('records_classified'),
        'Local model': metrics.get('local_model'),
        'LLM model': metrics.get('llm_model'),
        'HF space': metrics.get('hf_space'),
        'BERT model': metrics.get('bert_model'),
        'Best ML pipeline': metrics.get('best_model'),
    }
    config_snapshot = {
        'Text columns': ', '.join(metrics.get('text_columns', [])),
        'Label column': metrics.get('label_column'),
    }
    vectorizer_scores = metrics.get('vectorizer_scores') or {}
    feature_importance = metrics.get('feature_importance')
    classification_report = metrics.get('classification_reports')
    category_section, chart_script = _render_category_section(
        classification_summary
    )

    sections = [
        _render_list_section('Summary', summary),
        _render_list_section('Configuration', config_snapshot),
        _render_table_section('Vectorizer Accuracy', vectorizer_scores),
        category_section,
        _render_report_section(classification_report),
        _render_feature_importance(feature_importance),
    ]

    styles = """
    body { font-family: 'Inter', 'Segoe UI', system-ui, sans-serif; margin: 2rem; background: #0b1120; color: #e2e8f0; }
    h1 { text-align: center; color: #f8fafc; }
    section { background: #111827; padding: 1.5rem; margin-bottom: 1.5rem; border-radius: 10px; box-shadow: 0 8px 24px rgba(0,0,0,0.4); border: 1px solid #1f2937; }
    h2 { color: #f8fafc; border-bottom: 1px solid #1f2937; padding-bottom: 0.5rem; margin-bottom: 1rem; }
    table { width: 100%; border-collapse: collapse; margin-top: 1rem; color: #f8fafc; }
    th, td { border: 1px solid #1f2937; padding: 0.6rem; text-align: left; }
    th { background: #1f2937; }
    ul { padding-left: 1.5rem; }
    pre { background: #0f172a; color: #f8fafc; padding: 1rem; border-radius: 8px; overflow-x: auto; border: 1px solid #1f2937; }
    canvas { margin-top: 1rem; }
    em { color: #94a3b8; }
    """
    body_html = ''.join(sections)
    chart_bundle = (
        '<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>'
        + chart_script
        if chart_script
        else ''
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Dashboard Snapshot</title>
  <style>{styles}</style>
</head>
<body>
  <h1>Dashboard Snapshot</h1>
  <p><em>Generated on {html.escape(timestamp)}</em></p>
  {body_html}
  {chart_bundle}
</body>
</html>
"""


def save_dashboard_snapshot(
    metrics_path: Path | None = None,
    output_dir: Path | None = None,
    classified_data_path: Path | None = None,
) -> Path:
    metrics_file = metrics_path or METRICS_FILE
    target_dir = output_dir or OUTPUT_DIR
    classified_path = classified_data_path or OUTPUT_FILE
    metrics = _load_metrics(metrics_file)
    classification_summary = _load_classification_summary(classified_path)
    html_content = _build_html(metrics, classification_summary)

    target_dir.mkdir(parents=True, exist_ok=True)
    destination = target_dir / SNAPSHOT_NAME
    destination.write_text(html_content, encoding='utf-8')
    print(f'Snapshot saved to: {destination}')
    return destination


if __name__ == '__main__':
    save_dashboard_snapshot()
