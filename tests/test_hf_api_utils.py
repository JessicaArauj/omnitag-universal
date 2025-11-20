import pytest

from package.python import hf_api_utils as hf


def test_normalize_label_maps_variants():
    assert hf._normalize_label('SIM!') == 'Sim'
    assert hf._normalize_label('nao absoluto') == hf._NO_LABEL
    assert hf._normalize_label('talvez') == 'Avaliar'


def test_coerce_confidence_clamps_values():
    assert hf._coerce_confidence('0.8') == 0.8
    assert hf._coerce_confidence('invalid') == 0.65
    assert hf._coerce_confidence(5) == 1.0


def test_parse_numeric_confidence_supports_percentages():
    assert hf._parse_numeric_confidence('0.42') == 0.42
    assert hf._parse_numeric_confidence('80') == 0.8
    assert hf._parse_numeric_confidence('200') is None


def test_infer_confidence_from_text_extracts_hint():
    snippet = 'Modelo reportou confiança de 82,5%'

    confidence = hf._infer_confidence_from_text(snippet)

    assert confidence == pytest.approx(0.825, rel=1e-3)


def test_parse_response_handles_generated_text():
    label, confidence, reason = hf._parse_response({'generated_text': '{"label": "Sim", "confidence": 0.9, "reason": "ok"}'})

    assert label == 'Sim'
    assert confidence == 0.9
    assert reason == 'ok'
