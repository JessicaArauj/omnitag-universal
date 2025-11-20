from __future__ import annotations

import json

import pandas as pd

from package.python.save_dashboard_report import save_dashboard_report


def test_save_dashboard_report_creates_file(tmp_path):
    metrics = {
        'backend': 'local',
        'test_size': 0.2,
        'records_classified': 12,
        'local_model': 'falcon',
        'text_columns': ['texto'],
        'label_column': 'label',
        'vectorizer_scores': {'tfidf': 0.9},
        'classification_reports': {'macro avg': {'f1-score': 0.8}},
        'feature_importance': {'Sim': [['agua', 0.5]]},
    }
    metrics_path = tmp_path / 'metrics.json'
    metrics_path.write_text(json.dumps(metrics), encoding='utf-8')

    classified_path = tmp_path / 'tagged.xlsx'
    pd.DataFrame(
        {
            'Categoria': ['Sim', 'Nao', 'Sim'],
            'Confianca': [0.9, 0.7, 0.95],
        }
    ).to_excel(classified_path, index=False)

    destination = save_dashboard_report(
        metrics_path=metrics_path,
        output_dir=tmp_path,
        classified_data_path=classified_path,
    )

    assert destination.exists()
    content = destination.read_text(encoding='utf-8')
    assert 'Dashboard Report' in content
    assert 'falcon' in content
    assert 'Category Distribution' in content
