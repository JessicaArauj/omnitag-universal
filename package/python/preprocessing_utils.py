"""Text preprocessing, dataset preparation, and validation helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

import nltk
import pandas as pd
import unidecode
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from nltk.tokenize import word_tokenize

from .config import (
    LABEL_COLUMN,
    LABEL_NORMALIZATION,
    TARGET_LABELS,
    TEXT_COLUMNS,
)

NON_WORD_PATTERN = re.compile(r'[^a-z0-9\s]')
STOPWORDS: set[str] = set()
STEMMER: RSLPStemmer | None = None
NLTK_PACKAGES = ['stopwords', 'punkt', 'rslp', 'punkt_tab']


def ensure_nltk_resources() -> None:
    """Download required NLTK packages and populate globals."""
    for resource in NLTK_PACKAGES:
        try:
            nltk.download(resource, quiet=True)
        except Exception as exc:  # noqa: BLE001 - informational
            print(f'Warning downloading {resource}: {exc}')

    global STOPWORDS, STEMMER  # noqa: PLW0603 - intentional cache
    STOPWORDS = set(stopwords.words('portuguese'))
    STOPWORDS.update({'a', 'ou', 'pra', 'vai', 'ta'})
    STEMMER = RSLPStemmer()


def normalize_label(value: object) -> str | None:
    """Normalize labels to Sim, Não or Avaliar when possible."""
    if not isinstance(value, str):
        return None

    cleaned = unidecode.unidecode(value.strip().lower())
    if not cleaned:
        return None

    if cleaned in LABEL_NORMALIZATION:
        return LABEL_NORMALIZATION[cleaned]
    return value.strip().title()


def preprocess_text(text: object) -> str:
    """Clean, tokenize, remove stopwords and stem text."""
    if not isinstance(text, str):
        return ''

    normalized = unidecode.unidecode(text.strip().lower())
    normalized = NON_WORD_PATTERN.sub(' ', normalized)
    tokens = word_tokenize(normalized, language='portuguese')
    processed_tokens: List[str] = []

    for token in tokens:
        if len(token) < 2 or token in STOPWORDS:
            continue
        if STEMMER:
            processed_tokens.append(STEMMER.stem(token))
        else:
            processed_tokens.append(token)

    return ' '.join(processed_tokens)


def load_dataset(path):
    """Read Excel file and validate required columns."""
    if not Path(path).exists():
        raise FileNotFoundError(f'Input file not found: {path}')
    df = pd.read_excel(path)
    required_columns = list(dict.fromkeys(TEXT_COLUMNS + [LABEL_COLUMN]))
    missing_cols: List[str] = []
    normalized_lookup = {
        unidecode.unidecode(str(col)).strip().lower(): col for col in df.columns
    }

    for required in required_columns:
        if required in df.columns:
            continue
        normalized_required = unidecode.unidecode(required).strip().lower()
        match = normalized_lookup.get(normalized_required)
        if match:
            df = df.rename(columns={match: required})
        else:
            missing_cols.append(required)

    if missing_cols:
        raise ValueError(
            'Required columns not found: '
            f'{", ".join(missing_cols)}.'
        )
    return df


def combine_text_columns(row: pd.Series) -> str:
    """Concatenate multiple configured text columns into one string."""
    parts: List[str] = []
    for column in TEXT_COLUMNS:
        if column not in row or pd.isna(row[column]):
            continue
        value = str(row[column]).strip()
        if value:
            parts.append(value)
    return ' '.join(parts)


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Create normalized text column and clean labels."""
    df = df.copy()
    df['texto_bruto'] = df.apply(combine_text_columns, axis=1)
    df['texto_limpo'] = df['texto_bruto'].apply(preprocess_text)
    df[LABEL_COLUMN] = df[LABEL_COLUMN].apply(normalize_label)
    return df


def validate_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return labeled subset ensuring there are enough samples."""
    labeled = df.dropna(subset=[LABEL_COLUMN])
    labeled = labeled[labeled[LABEL_COLUMN].isin(TARGET_LABELS)]

    if labeled.empty:
        raise ValueError(
            'No labeled rows were found. Fill the label column '
            f'("{LABEL_COLUMN}") with Sim/Não/Avaliar.'
        )

    label_counts = labeled[LABEL_COLUMN].value_counts()
    if any(count < 2 for count in label_counts):
        raise ValueError(
            'Each class must contain at least 2 records for training. '
            f'Current distribution: {label_counts.to_dict()}.'
        )

    if len(label_counts) < 2:
        raise ValueError(
            'At least two distinct classes are required to train the '
            'logistic regression model.'
        )

    return labeled
