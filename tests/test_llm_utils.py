from package.python import llm_utils as llm


def test_normalize_label_defaults_to_avaliar():
    assert llm._normalize_label('maybe?') == 'Avaliar'
    assert llm._normalize_label('SIM') == 'Sim'


def test_parse_response_handles_partial_json():
    response = 'prefix {"label": "Sim", "confidence": 1.2, "reason": "ok"} suffix'

    result = llm._parse_response(response)

    assert result.label == 'Sim'
    assert result.confidence == 1.0
    assert result.reason == 'ok'


def test_build_messages_escapes_curly_braces():
    text = 'Use {value}'

    messages = llm._build_messages(text)

    assert messages[0]['role'] == 'system'
    assert 'Use {value}' in messages[1]['content']
