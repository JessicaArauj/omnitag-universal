"""Streamlit dashboard to inspect the latest metrics summary."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st

if __package__ in {None, ''}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from package.python.config import METRICS_FILE  # type: ignore
else:
    from .config import METRICS_FILE


def load_metrics(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        st.warning(f'Metrics file not found at "{path}".')
        return None

    raw = path.read_bytes()
    for encoding in ('utf-8', 'utf-8-sig', 'latin-1'):
        try:
            return json.loads(raw.decode(encoding))
        except UnicodeDecodeError:
            continue
        except json.JSONDecodeError as exc:
            st.error(f'Invalid JSON in metrics file: {exc}')
            return None

    st.error('Unable to decode metrics file; unsupported encoding.')
    return None


def render_top_summary(data: Dict[str, Any]) -> None:
    cols = st.columns(3)
    cols[0].metric('Backend', data.get('backend', 'n/a'))
    cols[1].metric('Test size', data.get('test_size', 'n/a'))
    cols[2].metric('Records classified', data.get('records_classified', 'n/a'))


def render_config_report(data: Dict[str, Any]) -> None:
    st.subheader('Configuration report')
    st.json(
        {
            'text_columns': data.get('text_columns'),
            'label_column': data.get('label_column'),
            'model_info': {
                key: data.get(key)
                for key in (
                    'bert_model',
                    'llm_model',
                    'hf_space',
                    'hf_api_name',
                    'local_model',
                    'best_model',
                )
                if data.get(key)
            },
        }
    )


def render_vectorizer_scores(data: Dict[str, Any]) -> None:
    scores = data.get('vectorizer_scores')
    if not scores:
        return

    st.subheader('Vectorizer Accuracy')
    frame = (
        pd.Series(scores, name='accuracy')
        .rename_axis('vectorizer')
        .reset_index()
    )
    frame['accuracy'] = frame['accuracy'].astype(float)
    st.bar_chart(frame, x='vectorizer', y='accuracy')


def render_classification_report(data: Dict[str, Any]) -> None:
    st.subheader('Classification Report')
    report = data.get('classification_reports')
    if isinstance(report, dict):
        st.json(report)
    elif report:
        st.code(report)
    else:
        st.info('No classification report captured for this run.')


def render_feature_importance(data: Dict[str, Any]) -> None:
    st.subheader('Feature Importance')
    importance = data.get('feature_importance')
    if not isinstance(importance, dict) or not importance:
        st.info('Feature importance data unavailable.')
        return

    tabs = st.tabs(list(importance.keys()))
    for tab, (label, rows) in zip(tabs, importance.items(), strict=False):
        tab.table(pd.DataFrame(rows, columns=['term', 'weight']))


def main() -> None:
    st.set_page_config(page_title='Test Metrics Dashboard', layout='wide')
    st.title('Test Metrics Overview')
    st.caption(
        'Run `python -m package.python.aut` to refresh the metrics before reloading this dashboard.'
    )

    metrics = load_metrics(METRICS_FILE)
    if not metrics:
        st.stop()

    render_top_summary(metrics)
    st.markdown('---')
    render_config_report(metrics)
    st.markdown('---')
    render_vectorizer_scores(metrics)
    render_classification_report(metrics)
    render_feature_importance(metrics)

    st.markdown('---')
    st.caption(f'Data source: {METRICS_FILE}')


if __name__ == '__main__':
    main()
