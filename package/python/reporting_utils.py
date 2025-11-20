"""Utility functions for saving outputs and metrics."""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from .config import (
    EXCLUDED_RESULT_COLUMNS,
    LABEL_COLUMN,
    METRICS_FILE,
    OUTPUT_FILE,
    TEST_SIZE,
    TEXT_COLUMNS,
)


def save_metrics_summary(
    metrics: Dict[str, object] | None,
    feature_importance: Dict[str, List[Tuple[str, float]]] | None,
    extra_info: Dict[str, object] | None = None,
) -> None:
    """Persist evaluation metrics or execution metadata."""
    summary = {
        'text_columns': TEXT_COLUMNS,
        'label_column': LABEL_COLUMN,
        'test_size': TEST_SIZE,
    }

    backend = None
    if isinstance(metrics, dict):
        backend = metrics.get('backend')

    if backend == 'bert' and metrics:
        summary.update(
            {
                'backend': 'bert',
                'bert_model': metrics.get('model_name'),
                'bert_accuracy': round(metrics.get('accuracy', 0.0), 4),
                'classification_reports': metrics.get('report'),
                'bert_labels': metrics.get('labels'),
            }
        )
    elif backend == 'llm' and metrics:
        summary.update(
            {
                'backend': 'llm',
                'llm_model': metrics.get('llm_model'),
            }
        )
    elif backend == 'hf' and metrics:
        summary.update(
            {
                'backend': 'hf',
                'hf_space': metrics.get('hf_space'),
                'hf_api_name': metrics.get('hf_api_name'),
            }
        )
    elif backend == 'local' and metrics:
        summary.update(
            {
                'backend': 'local',
                'local_model': metrics.get('local_model'),
            }
        )
    elif metrics:
        summary.update(
            {
                'backend': 'ml',
                'best_model': metrics['best_key'],
                'vectorizer_scores': {
                    key: round(result['accuracy'], 4)
                    for key, result in metrics['results'].items()
                },
                'classification_reports': metrics['results'][
                    metrics['best_key']
                ]['report'],
            }
        )
        if feature_importance:
            summary['feature_importance'] = feature_importance
    elif feature_importance:
        summary['feature_importance'] = feature_importance

    if extra_info:
        summary.update(extra_info)

    METRICS_FILE.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )


def save_classified_data(df: pd.DataFrame, dataset_path: Path) -> None:
    """Persist the enriched dataframe to Excel."""
    df_output = df.copy()

    def _normalize(name: str) -> str:
        ascii_name = (
            unicodedata.normalize('NFKD', str(name))
            .encode('ascii', 'ignore')
            .decode()
        )
        return ''.join(ch.lower() for ch in ascii_name if ch.isalnum())

    if 'Categoria' not in df_output.columns:
        fallback_map = {
            'categoria': 'Categoria',
            'categorizacao': 'Categoria',
            'previsao': 'Categoria',
            'possibilidade': 'Categoria',
        }
        for column in df_output.columns:
            normalized = _normalize(column)
            if normalized in fallback_map:
                df_output['Categoria'] = df_output[column]
                break

    drop_columns = [
        column
        for column in EXCLUDED_RESULT_COLUMNS
        if column in df_output.columns
    ]
    if drop_columns:
        df_output = df_output.drop(columns=drop_columns)
    df_output.to_excel(OUTPUT_FILE, index=False)
    print(
        f'Classified data saved to: {OUTPUT_FILE} '
        f'(based on {dataset_path.name}).'
    )
