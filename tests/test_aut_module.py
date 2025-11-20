from pathlib import Path
import importlib
import sys
import types


def _load_aut_module():
    stub_name = 'package.python.local_model_utils'
    original = sys.modules.get(stub_name)
    stub = types.ModuleType(stub_name)
    stub.classify_dataframe_with_local_model = lambda df, text_column='texto_bruto': df  # noqa: E731
    sys.modules[stub_name] = stub
    try:
        sys.modules.pop('package.python.aut', None)
        return importlib.import_module('package.python.aut')
    finally:
        if original is None:
            sys.modules.pop(stub_name, None)
        else:
            sys.modules[stub_name] = original


def test_clear_hf_cache_removes_directory(tmp_path):
    aut = _load_aut_module()
    cache_dir = tmp_path / 'hf-cache'
    nested = cache_dir / 'model'
    nested.mkdir(parents=True)
    (nested / 'weights.bin').write_text('data')

    aut.clear_hf_cache(cache_dir)

    assert not cache_dir.exists()
