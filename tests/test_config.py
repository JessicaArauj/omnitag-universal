from package.python import config


def test_parse_text_columns_prefers_multi_value(monkeypatch):
    monkeypatch.setenv('TEXT_COLUMNS', 'col_a, col_b ,col_c')
    monkeypatch.delenv('TEXT_COLUMN', raising=False)

    columns = config._parse_text_columns()

    assert columns == ['col_a', 'col_b', 'col_c']


def test_parse_text_columns_fallback_to_single(monkeypatch):
    monkeypatch.delenv('TEXT_COLUMNS', raising=False)
    monkeypatch.setenv('TEXT_COLUMN', 'only_one')

    columns = config._parse_text_columns()

    assert columns == ['only_one']
