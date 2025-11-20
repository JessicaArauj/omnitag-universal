"""Local transformers pipeline backend for classification."""

from __future__ import annotations

import importlib
import importlib.util as importlib_util
import json
import os
import re
import sys
import types
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'True')

def _ensure_torch_dll_directory() -> None:
    """Ensure torch's DLL directory is discoverable on Windows."""
    if os.name != 'nt':
        return
    if getattr(_ensure_torch_dll_directory, '_configured', False):
        return
    spec = importlib_util.find_spec('torch')
    origin = getattr(spec, 'origin', None) if spec else None
    if not origin:
        return
    torch_dir = Path(origin).resolve().parent
    lib_dir = torch_dir / 'lib'
    if not lib_dir.exists():
        return
    dll_path = str(lib_dir)
    if hasattr(os, 'add_dll_directory'):
        try:
            os.add_dll_directory(dll_path)
        except (OSError, FileNotFoundError):
            pass
    os.environ['PATH'] = dll_path + os.pathsep + os.environ.get('PATH', '')
    _ensure_torch_dll_directory._configured = True  # type: ignore[attr-defined]


_ensure_torch_dll_directory()

try:  # pragma: no cover - optional dependency
    import torch  # type: ignore  # noqa: F401
except Exception as exc:  # noqa: BLE001
    torch = None  # type: ignore[assignment]
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None


def _install_torchvision_stub(import_exc: Exception) -> None:
    """Provide a minimal torchvision stub so text-only transformers can load."""
    print(
        'Warning: torchvision import failed; using a lightweight stub suitable '
        'for text-only pipelines.\n'
        f'Original error: {import_exc}'
    )
    for module_name in list(sys.modules):
        if module_name == 'torchvision' or module_name.startswith('torchvision.'):
            sys.modules.pop(module_name, None)

    stub = types.ModuleType('torchvision')
    stub.__dict__['__file__'] = '<torchvision-stub>'
    stub.__dict__['__path__'] = []
    stub.__dict__['__package__'] = 'torchvision'
    stub.__dict__['__spec__'] = importlib.machinery.ModuleSpec(
        'torchvision',
        loader=None,
        origin='torchvision-stub',
    )
    stub.__all__ = []

    extension_module = types.ModuleType('torchvision.extension')
    extension_module._HAS_OPS = False
    extension_module._has_ops = lambda: False  # noqa: E731
    stub.extension = extension_module

    class _InterpolationMode(str, Enum):
        NEAREST = 'nearest'
        NEAREST_EXACT = 'nearest_exact'
        BOX = 'box'
        BILINEAR = 'bilinear'
        HAMMING = 'hamming'
        BICUBIC = 'bicubic'
        LANCZOS = 'lanczos'

    transforms_module = types.ModuleType('torchvision.transforms')
    transforms_module.__path__ = []
    transforms_module.InterpolationMode = _InterpolationMode
    stub.transforms = transforms_module

    modules = {
        'torchvision': stub,
        'torchvision.extension': extension_module,
        'torchvision.transforms': transforms_module,
    }

    placeholder_names = (
        '_meta_registrations',
        'datasets',
        'io',
        'models',
        'ops',
        'utils',
    )
    for name in placeholder_names:
        placeholder = types.ModuleType(f'torchvision.{name}')
        setattr(stub, name, placeholder)
        modules[f'torchvision.{name}'] = placeholder

    transforms_functional = types.ModuleType('torchvision.transforms.functional')
    transforms_module.functional = transforms_functional
    modules['torchvision.transforms.functional'] = transforms_functional

    transforms_v2 = types.ModuleType('torchvision.transforms.v2')
    transforms_v2.__path__ = []
    transforms_module.v2 = transforms_v2
    modules['torchvision.transforms.v2'] = transforms_v2
    transforms_v2_functional = types.ModuleType('torchvision.transforms.v2.functional')
    transforms_v2.functional = transforms_v2_functional
    modules['torchvision.transforms.v2.functional'] = transforms_v2_functional

    for module_name, module in modules.items():
        sys.modules[module_name] = module


def _ensure_torchvision_stub_if_needed() -> None:
    if 'torchvision' in sys.modules:
        return
    try:
        importlib.import_module('torchvision')
    except Exception as exc:  # pragma: no cover - optional dependency guard
        _install_torchvision_stub(exc)


_ensure_torchvision_stub_if_needed()

os.environ.setdefault('TRANSFORMERS_NO_TORCHVISION', '1')
os.environ.setdefault('TRANSFORMERS_NO_TORCHVISION_IMPORT', '1')

try:  # pragma: no cover - optional dependency
    from transformers import pipeline as _pipeline_import
except (ImportError, OSError, AttributeError) as primary_exc:  # noqa: PERF203
    try:
        from transformers.pipelines import pipeline as _pipeline_import
    except (ImportError, OSError, AttributeError) as fallback_exc:
        pipeline = None  # type: ignore[assignment]
        PIPELINE_IMPORT_ERROR = ImportError(
            'transformers import failed and fallback also failed: '
            f'{primary_exc}; {fallback_exc}'
        )
    else:
        pipeline = _pipeline_import  # type: ignore[assignment]
        PIPELINE_IMPORT_ERROR = None
else:
    pipeline = _pipeline_import  # type: ignore[assignment]
    PIPELINE_IMPORT_ERROR = None

from .config import (
    LOCAL_DEVICE_MAP,
    LOCAL_DO_SAMPLE,
    LOCAL_MAX_NEW_TOKENS,
    LOCAL_MODEL_NAME,
    LOCAL_PIPELINE_TASK,
    LOCAL_PROMPT_TEMPLATE,
    LOCAL_RETURN_FULL_TEXT,
    LOCAL_TEMPERATURE,
    LOCAL_TOP_P,
    LOCAL_TORCH_DTYPE,
    LOCAL_TRUST_REMOTE_CODE,
)
from .modeling_utils import map_prediction_to_outputs

_PIPELINE = None
_PIPELINE_CONFIG: dict | None = None


def _resolve_device_map() -> str | None:
    if not LOCAL_DEVICE_MAP:
        return None
    device_map = LOCAL_DEVICE_MAP.lower()
    if device_map == 'auto' and importlib_util.find_spec('accelerate') is None:
        print(
            'Warning: accelerate not installed; ignoring LOCAL_DEVICE_MAP.'
        )
        return None
    return LOCAL_DEVICE_MAP


def _resolve_dtype():
    """Return an explicit torch dtype when required."""
    if torch is None:
        return None

    value = LOCAL_TORCH_DTYPE.strip()
    if not value or value.lower() == 'auto':
        return None if torch.cuda.is_available() else torch.float32

    normalized = value.lower()
    mapping = {
        'float16': torch.float16,
        'half': torch.float16,
        'float32': torch.float32,
        'fp32': torch.float32,
        'float64': torch.float64,
        'fp64': torch.float64,
        'bfloat16': torch.bfloat16,
    }
    dtype = mapping.get(normalized)
    if dtype is None and hasattr(torch, normalized):
        dtype = getattr(torch, normalized)
    if dtype in {torch.float16, torch.bfloat16} and not torch.cuda.is_available():
        print('Warning: float16/bfloat16 not supported on CPU; forcing float32.')
        return torch.float32
    return dtype


def _convert_dtype(value):
    if torch is None or value is None:
        return None
    if isinstance(value, torch.dtype):
        return value
    normalized = str(value).lower()
    mapping = {
        'float16': torch.float16,
        'half': torch.float16,
        'float32': torch.float32,
        'fp32': torch.float32,
        'float64': torch.float64,
        'fp64': torch.float64,
        'bfloat16': torch.bfloat16,
    }
    if normalized in mapping:
        return mapping[normalized]
    return getattr(torch, normalized, None)


def _force_cpu_float32_if_needed(pipe):
    if pipe is None or torch is None:
        return
    if torch.cuda.is_available():
        return
    try:
        pipe.model.to(torch.float32)
        pipe.model.config.torch_dtype = torch.float32
    except Exception:  # noqa: BLE001
        pass


def _apply_pipeline_dtype(pipe, dtype_value):
    if pipe is None or torch is None or dtype_value is None:
        return
    target_dtype = _convert_dtype(dtype_value)
    if target_dtype is None:
        return
    try:
        pipe.model.to(target_dtype)
        pipe.model.config.torch_dtype = target_dtype
    except Exception:  # noqa: BLE001
        _force_cpu_float32_if_needed(pipe)


def _get_pipeline():
    """Instantiate the transformers pipeline lazily."""
    if pipeline is None:  # pragma: no cover - defensive
        message = (
            f"transformers/torch is not available: {PIPELINE_IMPORT_ERROR}"
            if 'PIPELINE_IMPORT_ERROR' in globals() and PIPELINE_IMPORT_ERROR
            else 'Install transformers[torch] to enable the local backend.'
        )
        raise RuntimeError(message)

    global _PIPELINE, _PIPELINE_CONFIG  # noqa: PLW0603

    pipeline_kwargs = {
        'task': LOCAL_PIPELINE_TASK,
        'model': LOCAL_MODEL_NAME,
        'trust_remote_code': LOCAL_TRUST_REMOTE_CODE,
    }
    device_map = _resolve_device_map()
    if device_map:
        pipeline_kwargs['device_map'] = device_map
    dtype_value = _resolve_dtype()
    if dtype_value is not None:
        pipeline_kwargs['dtype'] = dtype_value

    def _normalized_config(params: dict) -> dict:
        normalized = params.copy()
        for key in ('dtype', 'torch_dtype'):
            if key in normalized and normalized[key] is not None:
                normalized[key] = str(normalized[key])
        return normalized

    normalized_kwargs = _normalized_config(pipeline_kwargs)
    if _PIPELINE is not None and _PIPELINE_CONFIG == normalized_kwargs:
        if dtype_value is None:
            _force_cpu_float32_if_needed(_PIPELINE)
        else:
            _apply_pipeline_dtype(_PIPELINE, dtype_value)
        return _PIPELINE

    try:
        _PIPELINE = pipeline(**pipeline_kwargs)
    except ValueError as exc:
        message = str(exc).lower()
        unsupported_generation = (
            pipeline_kwargs.get('task') == 'text-generation'
            and 'not supported for text-generation' in message
        )
        if unsupported_generation:
            print(
                'Model does not support text-generation; retrying with '
                'text2text-generation.'
            )
            pipeline_kwargs['task'] = 'text2text-generation'
            _PIPELINE = pipeline(**pipeline_kwargs)
        else:
            raise
    _PIPELINE_CONFIG = normalized_kwargs
    if dtype_value is None:
        _force_cpu_float32_if_needed(_PIPELINE)
    else:
        _apply_pipeline_dtype(_PIPELINE, dtype_value)
    return _PIPELINE


def _render_prompt(text: str) -> str:
    placeholder = '{text}'
    if placeholder not in LOCAL_PROMPT_TEMPLATE:
        raise RuntimeError(
            'Invalid LOCAL_PROMPT_TEMPLATE; include the {text} placeholder.'
        )
    return LOCAL_PROMPT_TEMPLATE.replace(placeholder, text)


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
                parsed = json.loads(match.group())
                if isinstance(parsed, dict):
                    candidate = parsed
            except json.JSONDecodeError:
                candidate = None

    if candidate:
        label = candidate.get('label') or candidate.get('prediction')
        confidence = (
            candidate.get('confidence')
            or candidate.get('score')
            or candidate.get('probability')
        )
        reason = candidate.get('reason') or candidate.get('explanation') or ''
        label = str(label or '').strip().title() or 'Avaliar'
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.6
        return label, max(0.0, min(1.0, confidence)), str(reason).strip()

    lowered = snippet.lower()
    if 'sim' in lowered:
        label = 'Sim'
    elif 'não' in lowered or 'nao' in lowered:
        label = 'Não'
    elif 'avaliar' in lowered:
        label = 'Avaliar'
    else:
        label = 'Avaliar'
    confidence_hint = _infer_confidence_from_text(snippet)
    return label, confidence_hint if confidence_hint is not None else 0.55, snippet


def _classify_text(text: str) -> Tuple[str, float, str]:
    prompt = _render_prompt(text)
    pipe = _get_pipeline()
    tokenizer = getattr(pipe, 'tokenizer', None)
    supports_chat = bool(
        tokenizer
        and getattr(tokenizer, 'chat_template', None)
    )

    generation_kwargs = {
        'max_new_tokens': LOCAL_MAX_NEW_TOKENS,
        'temperature': LOCAL_TEMPERATURE,
        'top_p': LOCAL_TOP_P,
        'do_sample': LOCAL_DO_SAMPLE,
        'return_full_text': LOCAL_RETURN_FULL_TEXT,
    }

    if supports_chat:
        inputs = [{'role': 'user', 'content': prompt}]
    else:
        inputs = prompt

    outputs = pipe(inputs, **generation_kwargs)
    if not outputs:
        raise RuntimeError('Local pipeline returned no output.')
    result = outputs[0]
    if isinstance(result, dict):
        text_output = result.get('generated_text')
        if isinstance(text_output, list) and text_output:
            text_output = text_output[-1]
        if isinstance(text_output, dict):
            text_output = text_output.get('content') or text_output.get('text')
        if text_output is None:
            text_output = result.get('text') or result.get('content')
        if isinstance(text_output, str):
            return _parse_generated_text(text_output)
    if isinstance(result, str):
        return _parse_generated_text(result)

    raise RuntimeError(f'Unexpected pipeline output: {result!r}')


def classify_dataframe_with_local_model(
    df: pd.DataFrame,
    text_column: str = 'texto_bruto',
) -> pd.DataFrame:
    """Classify every row using the local transformers pipeline."""
    if df.empty:
        return df

    df = df.copy()
    labels: List[str] = []
    confidences: List[float] = []
    reasons: List[str] = []

    texts = df[text_column].fillna('').astype(str).tolist()
    iterator = texts
    progress_iter = None
    try:  # pragma: no cover - optional dependency
        from tqdm.auto import tqdm

        progress_iter = tqdm(
            texts,
            desc='Classifying rows',
            unit='row',
        )
        iterator = progress_iter
    except ImportError:
        iterator = texts

    for text in iterator:
        label, confidence, reason = _classify_text(text)
        labels.append(label)
        confidences.append(confidence)
        reasons.append(reason)

    if progress_iter is not None:  # pragma: no cover - visual feedback
        progress_iter.close()

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
    df['justificativa_local_llm'] = reasons
    df['fonte_classificacao'] = 'local_llm'
    df['modelo_classificacao'] = LOCAL_MODEL_NAME
    return df




