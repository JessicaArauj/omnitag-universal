from pathlib import Path

import pytest

from package.python import io_utils


def test_expand_candidate_supports_wildcards(tmp_path, monkeypatch):
    monkeypatch.setattr(io_utils, 'INPUT_DIR', tmp_path)
    (tmp_path / 'input_alpha.xlsx').write_text('a')
    (tmp_path / 'input_beta.xlsx').write_text('b')

    matches = io_utils._expand_candidate(tmp_path / 'input_*.xlsx')

    assert matches == sorted(tmp_path.glob('input_*.xlsx'))


def test_resolve_input_file_prefers_env_override(tmp_path, monkeypatch):
    override = tmp_path / 'custom.xlsx'
    override.write_text('content')
    monkeypatch.setattr(io_utils, 'INPUT_DIR', tmp_path)
    monkeypatch.setattr(io_utils, 'ENV_INPUT_FILE', str(override))
    monkeypatch.setattr(io_utils, 'DEFAULT_INPUT_NAME', 'missing.xlsx')
    monkeypatch.setattr(io_utils, 'FALLBACK_INPUT_NAME', 'fallback.xlsx')

    resolved = io_utils.resolve_input_file()

    assert resolved == override


def test_resolve_input_file_raises_when_no_options(tmp_path, monkeypatch):
    monkeypatch.setattr(io_utils, 'INPUT_DIR', tmp_path)
    monkeypatch.setattr(io_utils, 'ENV_INPUT_FILE', None)
    monkeypatch.setattr(io_utils, 'DEFAULT_INPUT_NAME', 'does_not_exist.xlsx')
    monkeypatch.setattr(io_utils, 'FALLBACK_INPUT_NAME', 'still_missing.xlsx')

    with pytest.raises(FileNotFoundError):
        io_utils.resolve_input_file()
