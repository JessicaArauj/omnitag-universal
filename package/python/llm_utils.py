"""Utilities to classify records using LLMs."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
from openai import OpenAI, OpenAIError

from .config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MAX_RETRIES,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_SYSTEM_PROMPT,
    LLM_TEMPERATURE,
    LLM_USER_PROMPT_TEMPLATE,
    TARGET_LABELS,
)
from .modeling_utils import map_prediction_to_outputs


@dataclass
class LLMResult:
    """Standard structure for a classification response."""

    label: str
    confidence: float
    reason: str


_CLIENT: OpenAI | None = None


def _get_client() -> OpenAI:
    """Instantiate the OpenAI client only once."""
    if not LLM_API_KEY:
        raise RuntimeError(
            'Set LLM_API_KEY to enable LLM-powered classification.'
        )

    global _CLIENT  # noqa: PLW0603 - intentional cache
    if _CLIENT is None:
        kwargs: Dict[str, str] = {}
        if LLM_BASE_URL:
            kwargs['base_url'] = LLM_BASE_URL
        _CLIENT = OpenAI(api_key=LLM_API_KEY, **kwargs)
    return _CLIENT


def _normalize_label(raw: str | None) -> str:
    """Ensure the label belongs to the allowed set."""
    if not raw:
        return 'Avaliar'
    candidate = raw.strip().title()
    if candidate not in TARGET_LABELS:
        return 'Avaliar'
    return candidate


def _parse_response(content: str) -> LLMResult:
    """Deserialize the model response from JSON."""
    payload: Dict[str, object]
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        start = content.find('{')
        end = content.rfind('}')
        if start != -1 and end != -1 and end > start:
            payload = json.loads(content[start : end + 1])
        else:
            payload = {}

    label = _normalize_label(str(payload.get('label', 'Avaliar')))

    try:
        confidence = float(payload.get('confidence', 0.65))
    except (TypeError, ValueError):
        confidence = 0.65
    confidence = max(0.0, min(1.0, confidence))

    reason = str(payload.get('reason') or '').strip()
    if not reason:
        reason = 'Model did not provide an explanation.'

    return LLMResult(label=label, confidence=confidence, reason=reason)


def _build_messages(text: str) -> List[Dict[str, str]]:
    """Generate the structured prompt while keeping the customizable template."""
    sanitized = text.strip()
    placeholder = '__TEXT_PLACEHOLDER__'
    template = LLM_USER_PROMPT_TEMPLATE.replace('{text}', placeholder)
    escaped = template.replace('{', '{{').replace('}', '}}')
    escaped = escaped.replace(placeholder, '{text}')
    user_prompt = escaped.format(text=sanitized)
    return [
        {'role': 'system', 'content': LLM_SYSTEM_PROMPT},
        {'role': 'user', 'content': user_prompt},
    ]


def classify_text_with_llm(text: str) -> LLMResult:
    """Classify a text snippet using the configured model."""
    client = _get_client()
    last_error: str | None = None

    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=_build_messages(text),
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
            )
            content = response.choices[0].message.content or '{}'
            return _parse_response(content)
        except OpenAIError as exc:  # noqa: PERF203 - sequential retries
            last_error = str(exc)
            sleep_time = min(4, attempt)
            time.sleep(sleep_time)

    raise RuntimeError(
        'Failed tPonto de Coletary the LLM after '
        f'{LLM_MAX_RETRIES} attempts: {last_error}'
    )


def classify_dataframe_with_llm(df: pd.DataFrame) -> pd.DataFrame:
    """Apply LLM classification to every row in the dataframe."""
    if df.empty:
        return df

    df = df.copy()
    text_column = 'texto_bruto' if 'texto_bruto' in df else 'texto_limpo'
    results: List[LLMResult] = []
    for text in df[text_column].fillna('').astype(str):
        results.append(classify_text_with_llm(text))

    labels = [result.label for result in results]
    confidences = [round(result.confidence, 4) for result in results]
    df['Categoria'] = labels
    df['categoria'] = labels
    df['confianca_modelo'] = confidences
    df['justificativa_llm'] = [result.reason for result in results]
    df['modelo_classificacao'] = LLM_MODEL
    df['fonte_classificacao'] = 'llm'

    mapped_rows = [
        map_prediction_to_outputs(result.label, result.confidence)
        for result in results
    ]
    df[
        [
            'possibilidade_automacao',
            'acao_sugerida',
            'confianca',
            'pontuacao_prioridade',
            'justificativa',
        ]
    ] = pd.DataFrame(mapped_rows, index=df.index)
    return df
