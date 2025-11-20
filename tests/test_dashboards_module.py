import sys
import types

import pandas as pd


def _install_streamlit_stub():
    module = types.ModuleType('streamlit')

    def _noop(*args, **kwargs):
        return None

    module.set_page_config = _noop
    module.title = _noop
    module.markdown = _noop
    module.error = _noop
    module.success = _noop
    module.subheader = _noop
    module.pyplot = _noop
    module.bar_chart = _noop
    module.dataframe = _noop
    module.caption = _noop
    module.warning = _noop
    module.stop = _noop

    def columns(count):  # noqa: ANN001
        Column = types.SimpleNamespace  # noqa: N806
        return tuple(Column(metric=_noop) for _ in range(count))

    module.columns = columns
    sys.modules['streamlit'] = module


_install_streamlit_stub()

from package.python import config as config_module  # noqa: E402

sys.modules['config'] = config_module

from package.python import dashboards  # noqa: E402


def test_load_data_returns_none_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboards, 'OUTPUT_PATH', tmp_path / 'missing.xlsx')

    assert dashboards.load_data() is None


def test_load_data_reads_existing_file(tmp_path, monkeypatch):
    output = tmp_path / 'tagged_file.xlsx'
    pd.DataFrame({'Categoria': ['Sim']}).to_excel(output, index=False)
    monkeypatch.setattr(dashboards, 'OUTPUT_PATH', output)

    result = dashboards.load_data()

    assert not result.empty
