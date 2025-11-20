"""Feasibility categorization orchestrator using modular pipeline."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from .config import (
    HF_API_NAME,
    HF_SPACE_ID,
    INPUT_DIR,
    LABEL_COLUMN,
    LLM_MODEL,
    LOCAL_MODEL_NAME,
    METRICS_FILE,
    MODEL_BACKEND,
    OUTPUT_DIR,
    OUTPUT_FILE,
)
from .local_model_utils import classify_dataframe_with_local_model
from .email_utils import send_result_email
from .hf_api_utils import classify_dataframe_with_hf_api
from .io_utils import ensure_directories, resolve_input_file
from .llm_utils import classify_dataframe_with_llm
from .preprocessing_utils import (
    ensure_nltk_resources,
    load_dataset,
    prepare_dataframe,
    validate_training_data,
)
from .reporting_utils import save_classified_data, save_metrics_summary
from .visualization_utils import (
    generate_wordclouds,
    plot_category_distribution,
)

DEFAULT_CHUNKS_DIR = INPUT_DIR / 'chunks'
DEFAULT_CHUNK_OUTPUT_DIR = OUTPUT_DIR / 'chunks'


def run_pipeline(input_path: Path | None = None) -> None:
    """Execute the full NLP workflow end-to-end."""
    ensure_directories()
    ensure_nltk_resources()

    dataset_path = Path(input_path) if input_path else resolve_input_file()
    print(f'Reading input file: {dataset_path}')
    df_raw = load_dataset(dataset_path)
    df_prepared = prepare_dataframe(df_raw)

    artifacts: dict | None = None
    feature_importance = None
    metrics_summary: dict | None = None
    run_metadata = {'backend': MODEL_BACKEND}

    if MODEL_BACKEND == 'llm':
        print('Using LLM backend; skipping traditional training.')
        df_classified = classify_dataframe_with_llm(df_prepared)
        metrics_summary = {'backend': 'llm', 'llm_model': LLM_MODEL}
        run_metadata.update(
            {
                'llm_model': LLM_MODEL,
                'records_classified': len(df_classified),
            }
        )
    elif MODEL_BACKEND == 'bert':
        from .modeling_utils import (  # Local import to avoid torch dependency when unused
            apply_bert_model,
            train_and_evaluate_bert,
        )

        training_df = validate_training_data(df_prepared)
        artifacts = train_and_evaluate_bert(
            training_df['texto_bruto'],
            training_df[LABEL_COLUMN],
        )
        metrics_summary = artifacts['metrics_summary']
        print(
            f"BERT ({artifacts['model_name']}) accuracy: "
            f"{metrics_summary['accuracy']:.3f}"
        )
        df_classified = apply_bert_model(
            df_prepared,
            artifacts,
            text_column='texto_bruto',
        )
        run_metadata.update(
            {
                'bert_model': artifacts['model_name'],
                'records_classified': len(df_classified),
            }
        )
    elif MODEL_BACKEND == 'hf':
        df_classified = classify_dataframe_with_hf_api(
            df_prepared,
            text_column='texto_bruto',
        )
        metrics_summary = {
            'backend': 'hf',
            'hf_space': HF_SPACE_ID,
            'hf_api_name': HF_API_NAME,
        }
        run_metadata.update(
            {
                'hf_space': HF_SPACE_ID,
                'hf_api_name': HF_API_NAME,
                'records_classified': len(df_classified),
            }
        )
    elif MODEL_BACKEND == 'local':
        print('Using local transformers pipeline backend.')
        df_classified = classify_dataframe_with_local_model(
            df_prepared,
            text_column='texto_bruto',
        )
        metrics_summary = {
            'backend': 'local',
            'local_model': LOCAL_MODEL_NAME,
        }
        run_metadata.update(
            {
                'local_model': LOCAL_MODEL_NAME,
                'records_classified': len(df_classified),
            }
        )
    else:
        from .modeling_utils import (
            apply_best_model,
            extract_feature_importance,
            train_and_evaluate,
        )

        training_df = validate_training_data(df_prepared)
        artifacts = train_and_evaluate(
            training_df['texto_limpo'],
            training_df[LABEL_COLUMN],
        )
        metrics_summary = artifacts
        best_pipeline = artifacts['best_pipeline']
        feature_importance = extract_feature_importance(best_pipeline)

        print(
            'Accuracies -> '
            + ', '.join(
                f'{name.upper()}: {result["accuracy"]:.3f}'
                for name, result in artifacts['results'].items()
            )
        )
        print('Best model:', artifacts['best_key'].upper())

        df_classified = apply_best_model(df_prepared, best_pipeline)
        run_metadata.update(
            {
                'best_model_key': artifacts['best_key'],
                'records_classified': len(df_classified),
            }
        )

    save_classified_data(df_classified, dataset_path)
    save_metrics_summary(metrics_summary, feature_importance, run_metadata)

    wordcloud_paths = generate_wordclouds(df_classified, 'texto_bruto')
    plot_category_distribution(df_classified)

    if wordcloud_paths:
        print('Word clouds saved at:')
        for path in wordcloud_paths:
            print(f'  - {path}')
    print(f'Metrics report saved to: {METRICS_FILE}')
    send_result_email(OUTPUT_FILE, df_classified, dataset_path)


def process_chunk_directory(
    chunk_dir: Path = DEFAULT_CHUNKS_DIR,
    destination_dir: Path = DEFAULT_CHUNK_OUTPUT_DIR,
) -> None:
    """Run the pipeline for every XLSX inside chunk_dir."""
    ensure_directories()
    files = sorted(chunk_dir.glob('*.xlsx'))
    if not files:
        raise FileNotFoundError(
            f'No XLSX files found in {chunk_dir}. '
            'Generate them with package.python.split_csv first.',
        )

    destination_dir.mkdir(parents=True, exist_ok=True)
    total = len(files)
    for index, chunk_path in enumerate(files, start=1):
        print(f'[{index}/{total}] Processing {chunk_path.name}...')
        run_pipeline(chunk_path)
        target_path = destination_dir / f'{chunk_path.stem}_tagged.xlsx'
        shutil.copy2(OUTPUT_FILE, target_path)
        print(f'    Saved result to {target_path}')


def clear_hf_cache(cache_dir: str | Path) -> None:
    """Remove Hugging Face cache directory after the run finishes."""
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        print(f'Cache path not found, nothing to remove: {cache_path}')
        return

    try:
        shutil.rmtree(cache_path)
        print(f'Removed Hugging Face cache at: {cache_path}')
    except OSError as exc:
        print(f'Failed to remove Hugging Face cache ({cache_path}): {exc}')


def get_hf_cache_dir() -> Path:
    """Return the Hugging Face cache directory using env hints or defaults."""
    env_cache = (
        os.environ.get('HF_HOME')
        or os.environ.get('HUGGINGFACE_HUB_CACHE')
        or os.environ.get('TRANSFORMERS_CACHE')
    )
    if env_cache:
        return Path(env_cache).expanduser()
    return Path.home() / '.cache' / 'huggingface'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Feasibility categorization pipeline.',
    )
    parser.add_argument(
        '--single-run',
        action='store_true',
        help='Process a single file (default behavior processes every chunk).',
    )
    parser.add_argument(
        '--chunk-dir',
        type=Path,
        default=DEFAULT_CHUNKS_DIR,
        help='Directory containing XLSX chunks. Default: inputs/chunks.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_CHUNK_OUTPUT_DIR,
        help='Directory where per-chunk tagged files are stored. Default: output/chunks.',
    )
    parser.add_argument(
        '--input-file',
        type=Path,
        help='Optional explicit XLSX path for a single run.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.single_run:
        run_pipeline(args.input_file)
    else:
        process_chunk_directory(args.chunk_dir, args.output_dir)
    clear_hf_cache(get_hf_cache_dir())


if __name__ == '__main__':
    main()
