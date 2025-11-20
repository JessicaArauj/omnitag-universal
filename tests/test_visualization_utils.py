import pytest

pytest.importorskip('matplotlib')
pytest.importorskip('seaborn')
pytest.importorskip('wordcloud')

import pandas as pd

from package.python import visualization_utils as viz


class _DummyCloud:
    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN001, D401
        self.rendered = ''

    def generate(self, text: str):
        self.rendered = text
        return self

    def to_file(self, path) -> None:  # noqa: ANN001
        path = viz.Path(path)
        path.write_text(self.rendered or 'empty', encoding='utf-8')


def test_generate_wordclouds_creates_files(tmp_path, monkeypatch):
    df = pd.DataFrame({'Categoria': ['Sim'], 'texto_bruto': ['agua limpa']})
    monkeypatch.setattr(viz, 'VISUAL_DIR', tmp_path)
    monkeypatch.setattr(viz, 'WordCloud', _DummyCloud)

    paths = viz.generate_wordclouds(df, 'texto_bruto')

    assert len(paths) == 1
    assert paths[0].exists()


def test_plot_category_distribution_saves_plot(tmp_path, monkeypatch):
    df = pd.DataFrame({'Categoria': ['Sim', 'Não', 'Sim']})
    monkeypatch.setattr(viz, 'VISUAL_DIR', tmp_path)

    output = viz.plot_category_distribution(df)

    assert output.exists()
