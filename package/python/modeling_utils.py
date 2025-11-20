"""Training, evaluation, and inference helpers for NLP modeling."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

torch = None  # type: ignore[assignment]
F = None  # type: ignore[assignment]


def _ensure_torch_available() -> None:
    global torch, F  # noqa: PLW0603
    if torch is not None and F is not None:
        return
    try:  # pragma: no cover - optional dependency
        import torch as torch_module
        import torch.nn.functional as functional
    except OSError as exc:
        raise RuntimeError(
            'PyTorch is required for BERT-related operations but could not be loaded.'
        ) from exc
    torch = torch_module
    F = functional

from .config import (
    BERT_BATCH_SIZE,
    BERT_EPOCHS,
    BERT_LEARNING_RATE,
    BERT_MAX_LENGTH,
    BERT_MODEL_DIR,
    BERT_MODEL_NAME,
    BERT_WARMUP_RATIO,
    BERT_WEIGHT_DECAY,
    CATEGORY_PLAYBOOK,
    MAX_FEATURES,
    NGRAM_RANGE,
    RANDOM_STATE,
    TARGET_LABELS,
    TEST_SIZE,
)


def build_pipeline(vectorizer) -> Pipeline:
    """Return a vectorizer + LogisticRegression pipeline."""
    classifier = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        solver='liblinear',
        multi_class='ovr',
        random_state=RANDOM_STATE,
    )
    return Pipeline([('vectorizer', vectorizer), ('classifier', classifier)])


def train_and_evaluate(texts: Iterable[str], labels: Iterable[str]):
    """Train models with Bag-of-Words and TF-IDF features."""
    X_train, X_test, y_train, y_test = train_test_split(
        list(texts),
        list(labels),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=list(labels),
    )

    vectorizers = {
        'bow': CountVectorizer(
            max_features=MAX_FEATURES,
            ngram_range=NGRAM_RANGE,
        ),
        'tfidf': TfidfVectorizer(
            max_features=MAX_FEATURES,
            ngram_range=NGRAM_RANGE,
        ),
    }

    results: Dict[str, Dict[str, object]] = {}
    best_key = ''
    best_score = -1.0

    for key, vectorizer in vectorizers.items():
        pipeline = build_pipeline(vectorizer)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(
            y_test,
            y_pred,
            digits=3,
            output_dict=False,
            zero_division=0,
        )

        results[key] = {
            'pipeline': pipeline,
            'accuracy': accuracy,
            'report': report,
        }

        if accuracy > best_score:
            best_score = accuracy
            best_key = key

    return {
        'results': results,
        'best_key': best_key,
        'best_pipeline': results[best_key]['pipeline'],
    }


def _prepare_hf_dataset(
    texts: List[str],
    label_ids: List[int],
    tokenizer,
) -> Dataset:
    """Tokenize texts according to the configured BERT model."""
    dataset = Dataset.from_dict({'text': texts, 'labels': label_ids})

    def _tokenize(batch: Dict[str, List[str]]):
        return tokenizer(
            batch['text'],
            padding='max_length',
            truncation=True,
            max_length=BERT_MAX_LENGTH,
        )

    tokenized = dataset.map(
        _tokenize,
        batched=True,
        remove_columns=['text'],
    )
    tokenized.set_format(
        type='torch',
        columns=['input_ids', 'attention_mask', 'labels'],
    )
    return tokenized


def train_and_evaluate_bert(
    texts: Iterable[str],
    labels: Iterable[str],
):
    """Fine-tune BERTimbau for text classification."""
    _ensure_torch_available()
    try:
        from transformers import (  # noqa: WPS433 - local import to avoid heavy dependency on import time
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            'Transformers library is required for the BERT backend.'
        ) from exc
    texts_list = list(texts)
    labels_list = list(labels)
    if not texts_list or not labels_list:
        raise ValueError('No samples provided for BERT training.')

    unique_labels = sorted(set(labels_list))
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    X_train, X_test, y_train, y_test = train_test_split(
        texts_list,
        labels_list,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels_list,
    )

    y_train_ids = [label2id[label] for label in y_train]
    y_test_ids = [label2id[label] for label in y_test]

    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    tokenized_train = _prepare_hf_dataset(X_train, y_train_ids, tokenizer)
    tokenized_eval = _prepare_hf_dataset(X_test, y_test_ids, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        BERT_MODEL_NAME,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir=str(BERT_MODEL_DIR),
        num_train_epochs=BERT_EPOCHS,
        per_device_train_batch_size=BERT_BATCH_SIZE,
        per_device_eval_batch_size=BERT_BATCH_SIZE,
        evaluation_strategy='epoch',
        save_strategy='no',
        learning_rate=BERT_LEARNING_RATE,
        weight_decay=BERT_WEIGHT_DECAY,
        warmup_ratio=BERT_WARMUP_RATIO,
        logging_strategy='epoch',
        report_to=[],
        seed=RANDOM_STATE,
        load_best_model_at_end=False,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
    )
    trainer.train()

    predictions = trainer.predict(tokenized_eval)
    pred_ids = predictions.predictions.argmax(axis=-1)
    accuracy = accuracy_score(y_test_ids, pred_ids)
    y_pred = [id2label[int(idx)] for idx in pred_ids]
    report = classification_report(
        y_test,
        y_pred,
        digits=3,
        output_dict=False,
    )

    model.eval()
    model.to('cpu')

    artifacts = {
        'backend': 'bert',
        'model': model,
        'tokenizer': tokenizer,
        'label2id': label2id,
        'id2label': id2label,
        'batch_size': BERT_BATCH_SIZE,
        'max_length': BERT_MAX_LENGTH,
        'model_name': BERT_MODEL_NAME,
        'metrics_summary': {
            'backend': 'bert',
            'model_name': BERT_MODEL_NAME,
            'accuracy': accuracy,
            'report': report,
            'labels': unique_labels,
        },
    }
    return artifacts


def extract_feature_importance(
    pipeline: Pipeline,
    top_n: int = 15,
) -> Dict[str, List[Tuple[str, float]]]:
    """Return top-N features per class from the trained pipeline."""
    vectorizer = pipeline.named_steps['vectorizer']
    classifier = pipeline.named_steps['classifier']
    feature_names = vectorizer.get_feature_names_out()
    importances: Dict[str, List[Tuple[str, float]]] = {}

    coefs_matrix = classifier.coef_
    classes = classifier.classes_
    if coefs_matrix.shape[0] == 1 and len(classes) == 2:
        coefs_matrix = np.vstack([coefs_matrix, -coefs_matrix])

    for idx, label in enumerate(classes):
        coefs = coefs_matrix[idx]
        top_indices = np.argsort(coefs)[-top_n:][::-1]
        importances[label] = [
            (feature_names[i], round(float(coefs[i]), 4)) for i in top_indices
        ]

    return importances


def map_prediction_to_outputs(
    label: str,
    confidence: float,
) -> Tuple[str, str, float, int, str]:
    """Convert predicted label to the expected dashboard columns."""
    template = CATEGORY_PLAYBOOK.get(label, CATEGORY_PLAYBOOK['Avaliar'])
    mapped_label = label if label in TARGET_LABELS else 'Avaliar'

    impact = {'Sim': 0.9, 'Avaliar': 0.6, 'Não': 0.3}.get(mapped_label, 0.6)
    clarity = 1 if confidence >= 0.85 else 0.8 if confidence >= 0.65 else 0.6
    priority = round(impact * clarity * confidence * 100)

    justification = template['justification']
    return (
        mapped_label,
        template['action'],
        round(confidence, 2),
        priority,
        justification,
    )


def apply_best_model(df: pd.DataFrame, pipeline: Pipeline) -> pd.DataFrame:
    """Generate predictions, confidences e colunas finais."""
    if df.empty:
        return df

    texts = df['texto_limpo'].fillna('').tolist()
    predictions = pipeline.predict(texts)
    confidences = pipeline.predict_proba(texts).max(axis=1)

    mapped = [
        map_prediction_to_outputs(label, confidence)
        for label, confidence in zip(predictions, confidences, strict=False)
    ]

    df = df.copy()
    df['Categorização'] = predictions
    df['Categoria'] = predictions
    df['Previsão'] = predictions
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
    return df


def apply_bert_model(
    df: pd.DataFrame,
    artifacts: Dict[str, object],
    text_column: str = 'texto_bruto',
) -> pd.DataFrame:
    """Realiza inferência com o modelo BERT fine-tunado."""
    if df.empty:
        return df

    _ensure_torch_available()
    tokenizer = artifacts['tokenizer']
    model: AutoModelForSequenceClassification = artifacts['model']
    id2label = artifacts['id2label']
    batch_size = max(2, int(artifacts.get('batch_size', BERT_BATCH_SIZE)))
    max_length = int(artifacts.get('max_length', BERT_MAX_LENGTH))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    texts = df[text_column].fillna('').astype(str).tolist()
    predictions: List[str] = []
    confidences: List[float] = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt',
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits
            probs = F.softmax(logits, dim=-1)
        batch_conf, batch_idx = torch.max(probs, dim=-1)
        confidences.extend(batch_conf.cpu().numpy().tolist())
        predictions.extend(batch_idx.cpu().numpy().tolist())

    model.to('cpu')
    predicted_labels = [id2label[int(idx)] for idx in predictions]

    mapped = [
        map_prediction_to_outputs(label, confidence)
        for label, confidence in zip(predicted_labels, confidences, strict=False)
    ]

    df = df.copy()
    df['Categorização'] = predicted_labels
    df['Categoria'] = predicted_labels
    df['Previsão'] = predicted_labels
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
    df['fonte_classificacao'] = 'bert'
    df['modelo_classificacao'] = artifacts.get('model_name')
    return df
