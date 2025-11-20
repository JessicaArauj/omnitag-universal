import pandas as pd

from package.python import map as mapper


def test_get_columns_from_csv(tmp_path, monkeypatch):
    csv_path = tmp_path / 'data.csv'
    csv_path.write_text('a,b,c\n1,2,3\n', encoding='utf-8')

    result = mapper.get_columns_from_csv(csv_path)

    assert result[0]['columns'] == ['a', 'b', 'c']
    assert result[0]['file'] == 'data.csv'


def test_get_columns_from_excel(tmp_path):
    excel_path = tmp_path / 'sample.xlsx'
    df = pd.DataFrame({'a': [1], 'b': [2]})
    df.to_excel(excel_path, index=False)

    result = mapper.get_columns_from_excel(excel_path)

    assert result[0]['columns'] == ['a', 'b']
    assert result[0]['sheet'] == 'Sheet1'
