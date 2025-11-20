from sklearn.feature_extraction.text import CountVectorizer

from package.python import modeling_utils as mu


def test_build_pipeline_returns_expected_steps():
    pipeline = mu.build_pipeline(CountVectorizer())

    assert pipeline.named_steps['vectorizer'].__class__ is CountVectorizer
    assert pipeline.named_steps['classifier'].__class__.__name__ == 'LogisticRegression'


def test_train_and_evaluate_returns_best_pipeline():
    texts = [
        'agua tratada confiavel',
        'processo seguro abastecimento',
        'contaminacao detectada urgente',
        'risco elevado não potavel',
        'sistema robusto aprovado',
        'falha grave não recomendado',
    ]
    labels = ['Sim', 'Sim', 'Não', 'Não', 'Sim', 'Não']

    result = mu.train_and_evaluate(texts, labels)

    assert result['best_key'] in result['results']
    assert 'pipeline' in result['results'][result['best_key']]


def test_extract_feature_importance_returns_weights():
    pipeline = mu.build_pipeline(CountVectorizer(max_features=5))
    texts = ['bom ruim', 'bom aprovado', 'ruim rejeitado', 'rejeitado ruim']
    labels = ['Sim', 'Sim', 'Não', 'Não']
    pipeline.fit(texts, labels)

    importances = mu.extract_feature_importance(pipeline, top_n=2)

    assert set(importances.keys()) == {'Sim', 'Não'}
    assert len(importances['Sim']) == 2
    assert isinstance(importances['Sim'][0][0], str)


def test_map_prediction_to_outputs_contains_reasonable_values():
    label, action, confidence, priority, justification = mu.map_prediction_to_outputs(
        'Sim',
        0.9,
    )

    assert label == 'Sim'
    assert confidence == 0.9
    assert priority > 0
    assert action
    assert justification
