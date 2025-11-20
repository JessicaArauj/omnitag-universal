import json

from package.python import reporting_utils as reporting


def test_save_metrics_summary_records_local_backend(tmp_path, monkeypatch):
    path = tmp_path / 'metrics.json'
    monkeypatch.setattr(reporting, 'METRICS_FILE', path)

    reporting.save_metrics_summary(
        metrics={'backend': 'local', 'local_model': 'falcon'},
        feature_importance=None,
        extra_info={'run_id': 42},
    )

    summary = json.loads(path.read_text(encoding='utf-8'))
    assert summary['backend'] == 'local'
    assert summary['local_model'] == 'falcon'
    assert summary['run_id'] == 42


def test_save_metrics_summary_records_ml_scores(tmp_path, monkeypatch):
    path = tmp_path / 'metrics_ml.json'
    monkeypatch.setattr(reporting, 'METRICS_FILE', path)

    metrics = {
        'backend': 'ml',
        'best_key': 'tfidf',
        'results': {
            'tfidf': {'accuracy': 0.9123, 'report': {'macro avg': {'f1-score': 0.9}}},
            'bow': {'accuracy': 0.75, 'report': {'macro avg': {'f1-score': 0.7}}},
        },
    }
    reporting.save_metrics_summary(metrics, feature_importance={'Sim': [('agua', 0.2)]})

    summary = json.loads(path.read_text(encoding='utf-8'))
    assert summary['backend'] == 'ml'
    assert summary['best_model'] == 'tfidf'
    assert summary['vectorizer_scores']['tfidf'] == 0.9123
    assert 'feature_importance' in summary
