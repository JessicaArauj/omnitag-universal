"""Visualization helpers (wordclouds and distribution plots)."""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from .config import TARGET_LABELS, VISUAL_DIR
from .preprocessing_utils import STOPWORDS
try:  # pragma: no cover - optional dependency
    from unidecode import unidecode as _transliterate
except ImportError:  # pragma: no cover - fallback when optional dependency missing
    def _transliterate(value: str) -> str:
        return value

sns.set_theme(style='whitegrid')


def _resolve_category_column(df) -> str | None:
    """Return the best column name representing the predicted categories."""
    target_keys = {
        'categoria',
        'categorizacao',
        'previsao',
        'possibilidade',
    }

    def normalize(name: str) -> str:
        normalized = _transliterate(str(name)).lower()
        normalized = ''.join(ch for ch in normalized if ch.isalnum())
        return normalized

    for column in df.columns:
        if normalize(column) in target_keys:
            return column
    return None


def generate_wordclouds(df, text_column: str) -> List[Path]:
    """Create wordclouds for each predicted category."""
    saved_paths: List[Path] = []
    if df.empty:
        return saved_paths
    df = df.copy()
    category_col = _resolve_category_column(df)
    if not category_col or category_col not in df.columns:
        print(
            'Wordclouds skipped: categoria column not found '
            f'(columns: {list(df.columns)[:10]})'
        )
        return saved_paths
    if 'Categoria' not in df.columns:
        df['Categoria'] = df[category_col]
        category_col = 'Categoria'

    for label in TARGET_LABELS:
        subset = df[df[category_col] == label]
        if subset.empty:
            continue
        text_blob = ' '.join(subset[text_column].dropna().astype(str))
        if not text_blob.strip():
            continue
        cloud = WordCloud(
            width=1200,
            height=700,
            background_color='white',
            stopwords=STOPWORDS,
            colormap='viridis',
        ).generate(text_blob)
        path = VISUAL_DIR / f'wordcloud_{label.lower()}.png'
        cloud.to_file(path)
        saved_paths.append(path)
    return saved_paths


def plot_category_distribution(df) -> Path | None:
    """Create a bar plot summarizing category counts."""
    if df.empty:
        return None
    df = df.copy()

    category_col = _resolve_category_column(df)
    if not category_col or category_col not in df.columns:
        print(
            'Category distribution skipped: categoria column not found '
            f'(columns: {list(df.columns)[:10]})'
        )
        return None
    if 'Categoria' not in df.columns:
        df['Categoria'] = df[category_col]
        category_col = 'Categoria'

    counts = df[category_col].value_counts().reset_index()
    counts.columns = ['Categoria', 'Quantidade']
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=counts,
        x='Categoria',
        y='Quantidade',
        hue='Categoria',
        palette='crest',
        legend=False,
    )
    plt.title('Distribuição das Categorias Previstas')
    plt.xlabel('Categoria')
    plt.ylabel('Quantidade')
    plt.tight_layout()
    path = VISUAL_DIR / 'categoria_distribuicao.png'
    plt.savefig(path)
    plt.close()
    return path
