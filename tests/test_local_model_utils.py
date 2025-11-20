import sys
import types

import pytest


def _install_transformers_stub():
    module = types.ModuleType('transformers')

    class _DummyPipeline:
        def __call__(self, *args, **kwargs):  # noqa: ANN001, D401
            return []

    module.pipeline = lambda **kwargs: _DummyPipeline()  # noqa: E731
    sys.modules['transformers'] = module


if 'transformers' not in sys.modules:
    _install_transformers_stub()

from package.python import local_model_utils as local


def test_render_prompt_replaces_placeholder(monkeypatch):
    monkeypatch.setattr(local, 'LOCAL_PROMPT_TEMPLATE', 'Input: {text}')

    prompt = local._render_prompt('agua')

    assert prompt == 'Input: agua'


def test_render_prompt_raises_without_placeholder(monkeypatch):
    monkeypatch.setattr(local, 'LOCAL_PROMPT_TEMPLATE', 'no placeholder here')

    with pytest.raises(RuntimeError):
        local._render_prompt('text')


def test_parse_numeric_confidence_handles_percentage():
    assert local._parse_numeric_confidence('85') == 0.85
    assert local._parse_numeric_confidence('0.5') == 0.5
    assert local._parse_numeric_confidence('abc') is None


def test_infer_confidence_from_text_extracts_value():
    snippet = 'Confidence 78% predicted'

    confidence = local._infer_confidence_from_text(snippet)

    assert confidence == pytest.approx(0.78, rel=1e-3)


def test_parse_generated_text_handles_json_payload():
    label, confidence, reason = local._parse_generated_text(
        '{"label": "Sim", "confidence": 0.9, "reason": "ok"}'
    )

    assert label == 'Sim'
    assert confidence == 0.9
    assert reason == 'ok'
