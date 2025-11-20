"""Integration helpers for Hugging Face backends (router or Space)."""

from __future__ import annotations

import json
import re
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
from gradio_client import Client

from .config import (
    HF_API_KEY,
    HF_API_MAX_NEW_TOKENS,
    HF_API_MAX_RETRIES,
    HF_API_NAME,
    HF_API_PROMPT_TEMPLATE,
    HF_API_RETURN_FULL_TEXT,
    HF_API_SLEEP_SECONDS,
    HF_API_TEMPERATURE,
    HF_API_TOP_P,
    HF_INFERENCE_URL,
    HF_SPACE_ID,
    TARGET_LABELS,
)
from .modeling_utils import map_prediction_to_outputs

_NO_LABEL = next(
    (label for label in TARGET_LABELS if label.lower().startswith('n')),
    'Nao',
)

HEADERS = (
    {'Authorization': f'Bearer {HF_API_KEY}'}
    if HF_API_KEY
    else {}
)
_SPACE_CLIENT: Client | None = None
USE_GENERATION_PROMPT = bool(
    HF_API_PROMPT_TEMPLATE and HF_API_PROMPT_TEMPLATE.strip()
)


def _ensure_api_key() -> None:
    """Guarantee the Hugging Face API key is present before making requests."""
    if HF_INFERENCE_URL and not HF_API_KEY:
        raise RuntimeError(
            'HUGGINGFACE_API_KEY is required to call the inference endpoint. '
            'Set it in your environment or data/.env.local before running.'
        )
    if HF_INFERENCE_URL:
        return

    # Spaces accessed via gradio_client may work without a token when public,
    # but we still warn because most private spaces demand authentication.
    if not HF_API_KEY:
        raise RuntimeError(
            'HUGGINGFACE_API_KEY is missing. Provide your token to use the '
            f'Hugging Face Space "{HF_SPACE_ID}".'
        )


def _normalize_label(value: object) -> str:
    """Normalize API label to Sim/Não/Avaliar."""
    if not isinstance(value, str):
        return 'Avaliar'

    cleaned = value.strip()
    lowered = cleaned.lower()
    if lowered.startswith('sim') or lowered.startswith('yes'):
        return 'Sim'
    if lowered.startswith('nao') or lowered.startswith('não') or lowered == 'no':
        return _NO_LABEL
    if lowered.startswith('avaliar') or lowered.startswith('maybe'):
        return 'Avaliar'

    titled = cleaned.title()
    if titled in TARGET_LABELS:
        return titled
    if cleaned in TARGET_LABELS:
        return cleaned
    return 'Avaliar'


def _coerce_confidence(value: object) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.65
    return max(0.0, min(1.0, confidence))


def _parse_numeric_confidence(value: str) -> float | None:
    """Convert textual confidence (decimal or percentage) into 0-1."""
    normalized = value.strip().replace(',', '.')
    try:
        numeric = float(normalized)
    except ValueError:
        return None
    if numeric > 1:
        if numeric <= 100:
            numeric /= 100
        else:
            return None
    return max(0.0, min(1.0, numeric))


def _infer_confidence_from_text(snippet: str) -> float | None:
    """Attempt to extract a confidence hint from unstructured text."""
    if not snippet:
        return None

    normalized = snippet.replace(',', '.')
    keyword_match = re.search(
        r'(confidence|confianca|precisao|accuracy|probability|probabilidade)[^\d]*(\d{1,3}(?:\.\d+)?)',
        normalized,
        flags=re.IGNORECASE,
    )
    if keyword_match:
        parsed = _parse_numeric_confidence(keyword_match.group(2))
        if parsed is not None:
            return parsed

    percent_match = re.search(r'(\d{1,3}(?:\.\d+)?)\s?%', normalized)
    if percent_match:
        parsed = _parse_numeric_confidence(percent_match.group(1))
        if parsed is not None:
            return parsed

    decimal_match = re.search(r'(?:0?\.\d+|1\.0)', normalized)
    if decimal_match:
        parsed = _parse_numeric_confidence(decimal_match.group())
        if parsed is not None:
            return parsed

    return None


def _render_prompt(text: str) -> str:
    """Format the prompt text when a template is present."""
    if not HF_API_PROMPT_TEMPLATE:
        return text
    try:
        return HF_API_PROMPT_TEMPLATE.format(text=text)
    except KeyError as exc:  # pragma: no cover - configuration issue
        raise RuntimeError(
            'Invalid HF_API_PROMPT_TEMPLATE; ensure it only uses '
            '{text} as a placeholder.'
        ) from exc


def _extract_from_payload(payload: Dict[str, object]) -> Tuple[str, float, str]:
    label = payload.get('label') or payload.get('prediction')
    label = label or payload.get('class') or payload.get('categoria')
    confidence = (
        payload.get('confidence')
        or payload.get('score')
        or payload.get('probability')
    )
    reason = payload.get('reason') or payload.get('explanation')
    reason = reason or payload.get('details') or payload.get('texto', '')

    norm_label = _normalize_label(label)
    norm_confidence = _coerce_confidence(confidence)
    final_reason = str(reason or '').strip()
    return norm_label, norm_confidence, final_reason


def _call_router_inference(text: str) -> object:
    payload = {'inputs': text}
    if USE_GENERATION_PROMPT:
        payload['parameters'] = {
            'max_new_tokens': HF_API_MAX_NEW_TOKENS,
            'temperature': HF_API_TEMPERATURE,
            'top_p': HF_API_TOP_P,
            'return_full_text': HF_API_RETURN_FULL_TEXT,
        }
    last_error = ''

    for attempt in range(1, HF_API_MAX_RETRIES + 1):
        try:
            response = requests.post(
                HF_INFERENCE_URL,
                headers=HEADERS,
                json=payload,
                timeout=60,
            )
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict) and data.get('error'):
                    last_error = data['error']
                else:
                    return data
            else:
                last_error = response.text
        except requests.RequestException as exc:  # noqa: BLE001
            last_error = str(exc)

        if attempt < HF_API_MAX_RETRIES:
            time.sleep(HF_API_SLEEP_SECONDS)

    raise RuntimeError(
        'Failed to call Hugging Face inference endpoint '
        f'{HF_INFERENCE_URL}: {last_error}'
    )


def _get_space_client() -> Client:
    """Instantiate (or reuse) a Gradio client for the configured Space."""
    global _SPACE_CLIENT  # noqa: PLW0603
    if _SPACE_CLIENT is None:
        client_kwargs = {}
        if HF_API_KEY:
            client_kwargs['hf_token'] = HF_API_KEY
        _SPACE_CLIENT = Client(HF_SPACE_ID, **client_kwargs)
    return _SPACE_CLIENT


def _call_space_via_client(text: str) -> object:
    """Call the Hugging Face Space using gradio_client for better compatibility."""
    last_error = ''
    api_name = HF_API_NAME or '/predict'
    if api_name and not api_name.startswith('/'):
        api_name = f'/{api_name}'

    guidance = (
        'Ensure HF_SPACE_ID/HF_API_NAME point to a Space exposing a public '
        'predict API or configure HF_INFERENCE_URL with your HF inference '
        'endpoint (router).'
    )

    for attempt in range(1, HF_API_MAX_RETRIES + 1):
        try:
            client = _get_space_client()
            return client.predict(text, api_name=api_name)
        except ValueError as exc:  # JSON decode issues / empty responses
            message = str(exc)
            if 'Expecting value' in message:
                raise RuntimeError(
                    f'Hugging Face Space {HF_SPACE_ID} did not expose a valid '
                    'API response for gradio_client. '
                    + guidance
                ) from exc
            last_error = message
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
        if attempt < HF_API_MAX_RETRIES:
            time.sleep(HF_API_SLEEP_SECONDS)

    raise RuntimeError(
        f'Failed to call Hugging Face Space {HF_SPACE_ID} via gradio_client: '
        f'{last_error}. {guidance}'
    )


def _parse_generated_text(raw: str) -> Tuple[str, float, str]:
    snippet = raw.strip()
    if not snippet:
        return 'Avaliar', 0.5, ''

    candidate: Dict[str, object] | None = None
    try:
        loaded = json.loads(snippet)
        if isinstance(loaded, dict):
            candidate = loaded
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', snippet, flags=re.DOTALL)
        if match:
            try:
                inner = json.loads(match.group())
                if isinstance(inner, dict):
                    candidate = inner
            except json.JSONDecodeError:
                candidate = None

    if candidate:
        return _extract_from_payload(candidate)

    lowered = snippet.lower()
    if 'sim' in lowered:
        label = 'Sim'
    elif 'nao' in lowered or 'Não' in lowered or 'não' in lowered:
        label = _NO_LABEL
    elif 'avaliar' in lowered:
        label = 'Avaliar'
    else:
        label = 'Avaliar'
    confidence_hint = _infer_confidence_from_text(snippet)
    return label, confidence_hint if confidence_hint is not None else 0.55, snippet


def _parse_response(result: object) -> Tuple[str, float, str]:
    if isinstance(result, str):
        return _parse_generated_text(result)

    if isinstance(result, dict):
        if 'generated_text' in result:
            return _parse_generated_text(str(result['generated_text']))
        if 'data' in result:
            data = result['data']
            if isinstance(data, list) and data:
                candidate = data[0]
                if isinstance(candidate, dict):
                    return _extract_from_payload(candidate)
                if isinstance(candidate, list) and candidate:
                    first = candidate[0]
                    if isinstance(first, dict):
                        return _extract_from_payload(first)
                if isinstance(candidate, str):
                    text = candidate.strip()
                    lowered = text.lower()
                    label = 'Sim' if 'sim' in lowered else (
                        _NO_LABEL if 'nao' in lowered or 'não' in lowered else 'Avaliar'
                    )
                    return label, 0.6, text
        if 'error' in result:
            raise RuntimeError(
                f'Hugging Face inference error: {result["error"]}'
            )
        return _extract_from_payload(result)

    if isinstance(result, list) and result:
        head = result[0]
        if isinstance(head, list) and head:
            head = head[0]
        if isinstance(head, dict):
            if 'generated_text' in head:
                return _parse_generated_text(str(head['generated_text']))
            return _extract_from_payload(head)
        if isinstance(head, str):
            return _parse_generated_text(head)

    raise RuntimeError(
        'Unexpected Hugging Face response format '
        f'for {HF_SPACE_ID}: {result!r}'
    )


def classify_text_with_hf_api(text: str) -> Tuple[str, float, str]:
    """Invoke HF inference or Space and return label/confidence/justification."""
    _ensure_api_key()

    prompt_text = _render_prompt(text)

    if HF_INFERENCE_URL:
        try:
            response = _call_router_inference(prompt_text)
        except RuntimeError as exc:
            msg = str(exc).lower()
            if any(
                keyword in msg
                for keyword in ('insufficient', 'permission', 'no longer supported')
            ):
                response = _call_space_via_client(prompt_text)
            else:
                raise
    else:
        response = _call_space_via_client(prompt_text)

    return _parse_response(response)


def classify_dataframe_with_hf_api(
    df: pd.DataFrame,
    text_column: str = 'texto_bruto',
) -> pd.DataFrame:
    """Apply the HF inference endpoint to every row of the dataframe."""
    if df.empty:
        return df

    df = df.copy()
    texts = df[text_column].fillna('').astype(str).tolist()
    labels: List[str] = []
    confidences: List[float] = []
    reasons: List[str] = []

    for text in texts:
        label, confidence, reason = classify_text_with_hf_api(text)
        labels.append(label)
        confidences.append(confidence)
        reasons.append(reason)

    df['Previsão'] = labels
    mapped = [
        map_prediction_to_outputs(label, confidence)
        for label, confidence in zip(labels, confidences, strict=False)
    ]

    df['Categorização'] = labels
    df['Categoria'] = labels
    df['Confiança do modelo'] = np.round(confidences, 4)
    df[
        [
            'Possibilidade',
            'Ação',
            'Confianca',
            'Pontuação de Prioridade',
            'Justificativa',
        ]
    ] = pd.DataFrame(mapped, index=df.index)
    df['justificativa_hf_api'] = reasons
    df['fonte_classificacao'] = 'hf_api'
    df['modelo_classificacao'] = HF_SPACE_ID
    return df
