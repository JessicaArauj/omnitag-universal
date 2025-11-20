import pandas as pd

from package.python import preprocessing_utils as prep


class _DummyStemmer:
    def stem(self, token: str) -> str:
        return token[:3]


def test_normalize_label_handles_variants():
    assert prep.normalize_label(' SIM ') == 'Sim'
    assert prep.normalize_label('nAo') == 'Nao'
    assert prep.normalize_label('unknown') == 'Unknown'


def test_preprocess_text_removes_stopwords_and_stems(monkeypatch):
    monkeypatch.setattr(prep, 'STOPWORDS', {'de'})
    monkeypatch.setattr(prep, 'STEMMER', _DummyStemmer())
    monkeypatch.setattr(
        prep,
        'word_tokenize',
        lambda text, language=None: text.split(),
    )

    result = prep.preprocess_text('Agua de qualidade')

    assert result == 'agu qua'


def test_combine_text_columns_skips_missing(monkeypatch):
    monkeypatch.setattr(prep, 'TEXT_COLUMNS', ['col_a', 'col_b'])
    row = pd.Series({'col_a': 'First', 'col_b': None})

    combined = prep.combine_text_columns(row)

    assert combined == 'First'


def test_prepare_dataframe_creates_columns(monkeypatch):
    monkeypatch.setattr(prep, 'combine_text_columns', lambda row: 'joined')
    monkeypatch.setattr(prep, 'preprocess_text', lambda text: 'clean')
    monkeypatch.setattr(prep, 'normalize_label', lambda label: 'Sim')
    df = pd.DataFrame({prep.LABEL_COLUMN: ['Sim']})

    prepared = prep.prepare_dataframe(df)

    assert prepared.at[0, 'texto_bruto'] == 'joined'
    assert prepared.at[0, 'texto_limpo'] == 'clean'
    assert prepared.at[0, prep.LABEL_COLUMN] == 'Sim'


def test_validate_training_data_filters_samples():
    df = pd.DataFrame(
        {
            prep.LABEL_COLUMN: ['Sim', 'Sim', 'Não', 'Não', 'Avaliar', 'Avaliar'],
            'texto': ['a', 'b', 'c', 'd', 'e', 'f'],
        }
    )

    filtered = prep.validate_training_data(df)

    assert set(filtered[prep.LABEL_COLUMN]) == {'Sim', 'Não', 'Avaliar'}
