"""Central configuration and constant definitions for the pipeline."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List

LOGGER = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parents[2]
LOCAL_ENV_PATH = BASE_DIR / 'data' / '.env.local'


def _load_local_env_file() -> None:
    """Load key=value pairs from data/.env.local into the environment."""
    if not LOCAL_ENV_PATH.exists():
        return

    try:
        lines = LOCAL_ENV_PATH.read_text(encoding='utf-8').splitlines()
    except OSError as exc:  # noqa: BLE001
        LOGGER.warning('Unable to read %s: %s', LOCAL_ENV_PATH, exc)
        return

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue
        if '=' not in line:
            LOGGER.warning('Ignoring malformed env line: %s', raw_line)
            continue
        key, value = line.split('=', 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


_load_local_env_file()

INPUT_DIR = BASE_DIR / 'inputs'
OUTPUT_DIR = BASE_DIR / 'output'
DEFAULT_INPUT_NAME = '*.xlsx'
FALLBACK_INPUT_NAME = 'sample.xlsx'
OUTPUT_FILE = OUTPUT_DIR / 'tagged_file.xlsx'
METRICS_FILE = OUTPUT_DIR / 'nlp_metrics.json'
TEST_REPORT_FILE = OUTPUT_DIR / 'test_results.json'
VISUAL_DIR = OUTPUT_DIR / 'nlp_visualizations'
EXCLUDED_RESULT_COLUMNS = [
    'fonte_classificacao',
    'modelo_classificacao',
    'texto_bruto',
    'texto_limpo',
    'Possibilidade',
    'justificativa_local_llm',
]
DEFAULT_TEXT_COLUMNS = [
    'Procedência da Coleta',
    'Ponto de Coleta',
    'Grupo de parâmetros',
    'Parâmetro (demais parâmetros)',
    'LD',
    'LQ',
    'Resultado',
]
MODEL_BACKEND = os.getenv('MODEL_BACKEND', 'hf').strip().lower() or 'hf'
if MODEL_BACKEND not in {'ml', 'llm', 'bert', 'hf', 'local'}:
    MODEL_BACKEND = 'hf'


def _parse_text_columns() -> List[str]:
    """Return list of text columns defined via environment variables."""
    raw = os.getenv('TEXT_COLUMNS')
    if raw:
        columns = [col.strip() for col in raw.split(',') if col.strip()]
        if columns:
            return columns

    fallback = os.getenv('TEXT_COLUMN')
    if fallback:
        fallback = fallback.strip()
        if fallback:
            return [fallback]

    return DEFAULT_TEXT_COLUMNS.copy()


TEXT_COLUMNS = _parse_text_columns()
LABEL_COLUMN = os.getenv('LABEL_COLUMN', 'Grupo de parâmetros')
ENV_INPUT_FILE = os.getenv('INPUT_FILE')

EMAIL_ENABLED = os.getenv('EMAIL_ENABLED', 'false').lower() in {
    '1',
    'true',
    'yes',
}
EMAIL_FROM = os.getenv('EMAIL_FROM')
EMAIL_TO = os.getenv('EMAIL_TO', '')
EMAIL_SUBJECT = os.getenv(
    'EMAIL_SUBJECT',
    'Resultado de categorizacao de viabilidade tecnica',
)
SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))
SMTP_USER = os.getenv('SMTP_USER')
SMTP_PASSWORD = os.getenv('SMTP_PASSWORD')
SMTP_USE_TLS = os.getenv('SMTP_USE_TLS', 'true').lower() in {
    '1',
    'true',
    'yes',
}

TEST_SIZE = max(0.1, min(0.3, float(os.getenv('TEST_SIZE', '0.2'))))
RANDOM_STATE = int(os.getenv('RANDOM_STATE', '42'))
MAX_FEATURES = int(os.getenv('MAX_FEATURES', '5000'))
NGRAM_RANGE = (1, 2)

TARGET_LABELS = {'Sim', 'Não', 'Avaliar'}
LABEL_NORMALIZATION = {
    'sim': 'Sim',
    's': 'Sim',
    'yes': 'Sim',
    'positiva': 'Sim',
    'nao': 'Nao',
    'n': 'Nao',
    'não': 'Nao',
    'negativa': 'Nao',
    'avaliar': 'Avaliar',
    'avaliacao': 'Avaliar',
    'analise': 'Avaliar',
    'a': 'Avaliar',
}

CATEGORY_PLAYBOOK = {
    'Sim': {
        'action': (
            'Autorizar o consumo da fonte e registrar como adequada no SISAGUA.'
        ),
        'justification': (
            'Análises indicam conformidade com os limites de qualidade e potabilidade.'
        ),
    },
    'Nao': {
        'action': (
            'Interditar o consumo da fonte e acionar tratamento ou correção.'
        ),
        'justification': (
            'Parâmetros apresentam risco ao consumo humano ou estao fora dos limites regulamentares.'
        ),
    },
    'Avaliar': {
        'action': (
            'Solicitar coletas adicionais e investigar parâmetros pendentes.'
        ),
        'justification': (
            'Faltam dados ou há divergências nas análises que impedem concluir a potabilidade.'
        ),
    },
}

LLM_MODEL = os.getenv(
    'LLM_MODEL',
    'carlosdelfino/eli5_clm-model',
)
LLM_API_KEY = os.getenv('HUGGINGFACE_API_KEY')
# When using Hugging Face Inference Endpoints with Cross-platform schema,
# you must set LLM_BASE_URL to that endpoint (e.g., https://api-inference.huggingface.co/v1).
LLM_BASE_URL = os.getenv('LLM_BASE_URL')
LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', '0.1'))
LLM_MAX_TOKENS = int(os.getenv('LLM_MAX_TOKENS', '350'))
LLM_MAX_RETRIES = max(1, int(os.getenv('LLM_MAX_RETRIES', '3')))
LLM_USER_PROMPT_TEMPLATE = os.getenv(
    'LLM_USER_PROMPT_TEMPLATE',
    (
        'Voce e um especialista da vigilancia da qualidade da agua (SISAGUA). '
        'Analise o texto a seguir, que resume resultados laboratoriais e contexto operacional, '
        'e classifique a fonte como "Sim" (propria), "Não" (impropria) ou '
        '"Avaliar" (dados inconclusivos). Considere parametros LD/LQ, resultado agregado '
        'e contexto regional antes de decidir. Responda exclusivamente em JSON bem formado '
        'seguindo {"label": "<Sim|Não|Avaliar>", "confidence": <float 0-1>, '
        '"reason": "<2-3 frases com parametros relevantes ou lacunas>"}. Texto: "{text}"'
    ),
)
LLM_SYSTEM_PROMPT = os.getenv(
    'LLM_SYSTEM_PROMPT',
    (
        'Voce e um especialista em vigilancia da qualidade da agua do SISAGUA '
        'e precisa decidir se cada fonte esta propria para consumo. Responda '
        'somente com JSON bem formado.'
    ),
)
BERT_MODEL_NAME = os.getenv(
    'BERT_MODEL_NAME',
    'carlosdelfino/eli5_clm-model',
)
BERT_MAX_LENGTH = max(64, int(os.getenv('BERT_MAX_LENGTH', '256')))
BERT_LEARNING_RATE = float(os.getenv('BERT_LEARNING_RATE', '2e-5'))
BERT_WEIGHT_DECAY = float(os.getenv('BERT_WEIGHT_DECAY', '0.01'))
BERT_WARMUP_RATIO = float(os.getenv('BERT_WARMUP_RATIO', '0.1'))
BERT_EPOCHS = max(1, int(os.getenv('BERT_EPOCHS', '3')))
BERT_BATCH_SIZE = max(2, int(os.getenv('BERT_BATCH_SIZE', '8')))
BERT_MODEL_DIR = OUTPUT_DIR / os.getenv(
    'BERT_OUTPUT_SUBDIR',
    'bertimbau_model',
)

HF_API_PROMPT_TEMPLATE = os.getenv('HF_API_PROMPT_TEMPLATE')
HF_API_MAX_NEW_TOKENS = max(
    16,
    int(os.getenv('HF_API_MAX_NEW_TOKENS', '256')),
)
HF_API_TEMPERATURE = float(os.getenv('HF_API_TEMPERATURE', '0.1'))
HF_API_TOP_P = float(os.getenv('HF_API_TOP_P', '0.95'))
HF_API_RETURN_FULL_TEXT = os.getenv(
    'HF_API_RETURN_FULL_TEXT',
    'false',
).lower() in {'1', 'true', 'yes'}
HF_SPACE_ID = os.getenv(
    'HF_SPACE_ID',
    'carlosdelfino/eli5_clm-model',
)
HF_API_NAME = os.getenv('HF_API_NAME', '/predict').strip() or '/predict'
HF_API_KEY = os.getenv('HUGGINGFACE_API_KEY')
HF_API_MAX_RETRIES = max(1, int(os.getenv('HF_API_MAX_RETRIES', '3')))
HF_API_SLEEP_SECONDS = float(os.getenv('HF_API_SLEEP_SECONDS', '1.0'))
HF_INFERENCE_URL = os.getenv('HF_INFERENCE_URL')
HF_SPACE_RUN_URL = os.getenv('HF_SPACE_RUN_URL')

LOCAL_MODEL_NAME = os.getenv('LOCAL_MODEL_NAME', 'tiiuae/falcon-11B')
LOCAL_PIPELINE_TASK = os.getenv('LOCAL_PIPELINE_TASK', 'text-generation')
LOCAL_PROMPT_TEMPLATE = os.getenv(
    'LOCAL_PROMPT_TEMPLATE',
    HF_API_PROMPT_TEMPLATE
    or (
        'Voce e um especialista da vigilancia da qualidade da agua (SISAGUA). '
        'Analise o texto a seguir e classifique a fonte como "Sim", "Não" ou "Avaliar" '
        'com base em parametros LD/LQ, resultado agregado e contexto regional. '
        'Responda unicamente com JSON no formato {"label": "<Sim|Não|Avaliar>", '
        '"confidence": <float 0-1>, "reason": "<2-3 frases com justificativas>"}. '
        'Texto: "{text}"'
    ),
)
LOCAL_MAX_NEW_TOKENS = max(
    16,
    int(os.getenv('LOCAL_MAX_NEW_TOKENS', '256')),
)
LOCAL_TEMPERATURE = float(os.getenv('LOCAL_TEMPERATURE', '0.1'))
LOCAL_TOP_P = float(os.getenv('LOCAL_TOP_P', '0.9'))
LOCAL_RETURN_FULL_TEXT = os.getenv(
    'LOCAL_RETURN_FULL_TEXT',
    'false',
).lower() in {'1', 'true', 'yes'}
LOCAL_DO_SAMPLE = os.getenv(
    'LOCAL_DO_SAMPLE',
    'true',
).lower() in {'1', 'true', 'yes'}
LOCAL_TRUST_REMOTE_CODE = os.getenv(
    'LOCAL_TRUST_REMOTE_CODE',
    'true',
).lower() in {'1', 'true', 'yes'}
LOCAL_DEVICE_MAP = os.getenv('LOCAL_DEVICE_MAP', '').strip()
LOCAL_TORCH_DTYPE = os.getenv('LOCAL_TORCH_DTYPE', 'auto').strip()
