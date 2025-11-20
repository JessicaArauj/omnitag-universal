<span align="justify">

## Project OmniTag Universal

> OmniTag is a reusable, end-to-end AI pipeline that ingests spreadsheet data, normalizes text, applies configurable ML/LLM backends, and emits enriched artifacts (Excel, JSON, dashboards) ready for downstream analytics. It is fully open source (MIT License), modular, and designed to plug into any classification workflow with minimal customization. The pipeline supports multiple backends:

- **Classic ML**: Bag-of-Words / TF-IDF + Logistic Regression with feature importance reports.
- **BERTimbau (Hugging Face)**: fine-tunes `carlosdelfino/eli5_clm-model` on your labeled samples.
- **Hugging Face API (default)**: each row is sent to a hosted Space/model via `gradio_client`/HF inference using `HUGGINGFACE_API_KEY`.
- **Local transformers backend**: run any `transformers.pipeline` model (e.g., `tiiuae/falcon-11B`) directly on your machine, without hitting external APIs.
- **LLM backend (optional)**: Cross-platform endpoint that interprets the `Procedencia da Coleta`, `Ponto de Coleta`, `Grupo de parametros`, `Parametro (demais parametros)`, `LD`, `LQ`, `Resultado` columns per row when explicitly enabled.

Every run produces an enriched Excel file (`output/tagged_file.xlsx`), visualizations, and a JSON report with metadata.

## Project Scope: A Successful Use Case

- **Description**: high-complexity laboratory analyses from Brazil's public health surveillance teams, used to decide whether each water source is potable.
- **Source**: SISAGUA - Surveillance - Other parameters ([open data](https://opendatasus.saude.gov.br/dataset/sisagua-vigilancia-demais-parametros/resource/974dee11-8f7b-4bfa-bf80-6393456eab10)).
- **Impact**: Nationwide coverage (all municipalities reporting to SISAGUA).

## Pre-Requisites

1. Python 3.11+
2. (Recommended) virtual environment: `python -m venv .venv andand .\.venv\Package\activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Place your spreadsheet in `inputs/` with the text columns `Procedencia da Coleta`, `Ponto de Coleta`, `Grupo de parametros`, `Parametro (demais parametros)`, `LD`, `LQ`, `Resultado` (default set) and the label column defined by `LABEL_COLUMN`.

Run the pipeline:

```bash
python -m package.python.aut
```

## Robots Helper Package

Automation helpers that live in `robots/` make repetitive tasks easier. Run them from the repository root using Git Bash, WSL, or any Unix-like shell:

- `bash robots/aut_setup.sh`: creates (or recreates) `.venv`, upgrades `pip`, purges the cache, and installs dependencies from `requirements.txt`. Set `PYTHON_BIN="py -3.11"` or similar if you need a custom interpreter.
- `bash robots/aut_extraction.sh`: downloads the latest SISAGUA vigilance ZIP, saves it under `backup/`, extracts the CSV, and immediately converts it to `inputs/vigilancia_demais_parametros.xlsx` so the rest of the pipeline (and the chunking helper below) can run without manual conversions. Override names or parsing details with `CSV_NAME`, `XLSX_NAME`, `CSV_SEPARATOR`, `CSV_ENCODING`, `CSV_FALLBACK_ENCODINGS`, or `PYTHON_BIN` if needed. When the CSV exceeds Excel's 1,048,576-row limit, the script automatically emits multiple files named `inputs/<basename>_part_XXX.xlsx`.
- `bash robots/aut_git.sh [--options]`: stages every change, prompts for a Conventional Commit-style message (unless `-m/--message` is provided), commits, and optionally pushes to the current branch. Pass `--no-push` to skip `git push`.

Each script is idempotent and prints its progress so you can confirm every step succeeded.

## Pipeline Steps

1. **Input resolution**: `aut.py` calls `resolve_input_file` to locate the Excel spreadsheet inside `inputs/` (or the override set via `INPUT_FILE`), creates `inputs/`, `output/`, and `output/nlp_visualizations`, and loads the file with `load_dataset`.
2. **Data preparation**: `prepare_dataframe` concatenates the configured `DEFAULT_TEXT_COLUMNS`, builds `texto_bruto`/`texto_limpo`, and normalizes the label (`LABEL_COLUMN`). For supervised backends (`ml`, `bert`) it also runs `validate_training_data` to ensure each class has enough samples.
3. **Backend selection**: according to `MODEL_BACKEND`, the pipeline either
   - trains/uses TF-IDF + logistic regression (`ml`);
   - fine-tunes and applies BERT (`bert`);
   - invokes the Hugging Face Space/API (`hf`);
   - calls an Cross-platform endpoint (`llm`);
   - or runs a local `transformers` pipeline (`local`).
4. **Classification and enrichment**: predictions populate columns such as `Categoria`, `Previsãoo`, `Possibilidade`, `Ação`, `Justificativa`, confidence, and priority score, plus any backend-specific explanations.
5. **Artifacts and metrics**: `reporting_utils` writes `output/tagged_file.xlsx`, `outputs/nlp_metrics.json`, and metadata about the chosen model, accuracy, label distribution, and feature importance when available.
6. **Visualizations and notifications**: `visualization_utils` generates word clouds and category charts; when `EMAIL_ENABLED` is true, `email_utils` emails the enriched Excel file.

## Testing and Observability

- **Unit tests**: run `python -m pytest --json-report --json-report-file output/test_results.json`. The JSON artifact feeds dashboards and CI checks with the exact counts of passed/failed tests.
- **Robots helper**: `bash robots/aut_tests.sh` orchestrates environment bootstrap, dependency installation, pytest (with JSON report), HTML report generation, and launches the Streamlit dashboard headlessly.
- **Dashboard (Streamlit)**: `python -m streamlit run package/python/test_metrics_dashboard.py`. Besides the classification metrics, it now surfaces the pytest summary, duration, and per-test outcomes straight from `output/test_results.json`.
- **Dark-mode report**: `python -m package.python.save_dashboard_report` produces `output/dashboard_report.html` (English, dark palette, Chart.js bar chart, summary cards). Use it to share results without spinning up Streamlit.

## Hugging Face API Backend in Default

1. `MODEL_BACKEND=hf` is the default setting.
2. Provide `HUGGINGFACE_API_KEY` (GitHub Secret or local env). Adjust `HF_SPACE_ID` (default `carlosdelfino/eli5_clm-model`) and `HF_API_NAME` if you point to a different Space.
3. Optional controls: `HF_API_MAX_RETRIES`, `HF_API_SLEEP_SECONDS`, `HF_INFERENCE_URL` (when pointing to the router or a private endpoint), `HF_API_PROMPT_TEMPLATE` (LLM-style prompt with `{text}` placeholder), and generation knobs such as `HF_API_MAX_NEW_TOKENS`, `HF_API_TEMPERATURE`, `HF_API_TOP_P`, `HF_API_RETURN_FULL_TEXT`.
4. Each record is sent to the Space/model (or inference endpoint). Use `HF_INFERENCE_URL=https://router.huggingface.co/hf-inference/models/tiiuae/falcon-11B`, for example, to leverage hosted models like Falcon2-11B or Mistral-7B. The Excel output shows `fonte_classificacao='hf_api'`, `modelo_classificacao` with the target id, and the textual justification returned by the API.

## LLM Backend in Optional

1. Set `MODEL_BACKEND=llm` only when you need an Cross-platform endpoint.
2. Configure `HUGGINGFACE_API_KEY` (or `LLM_API_KEY`) plus tunables such as `LLM_MODEL`, `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`, `LLM_MAX_RETRIES`, `LLM_SYSTEM_PROMPT`, `LLM_USER_PROMPT_TEMPLATE`.
3. When using a Hugging Face Inference endpoint, set `LLM_BASE_URL` to your Cross-platform URL (e.g., `https://api-inference.huggingface.co/v1`) so the `hf_xxx` token is accepted.
4. Run the pipeline; the Excel output and metrics JSON will reflect the LLM backend.

## Hosting the LLM Backend on Cross-Platform from AWS, GCP, Azure, Databricks, and Others

`llm_utils` uses the [`OpenAI`](https://github.com/openai/openai-python) client. Any provider (or router) that exposes an Cross-platform REST API works transparently as long as you set `LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODEL`. Typical setups:

| Provider | How to expose an OpenAI-style endpoint | Environment variables |
| --- | --- | --- |
| **AWS (Bedrock/SageMaker)** | Deploy an API Gateway + Lambda proxy, or run [litellm](https://github.com/BerriAI/litellm) / [bedrock-openai-proxy](https://github.com/explorers-lab/bedrock-openai-proxy) inside your VPC and route requests to Bedrock models (Claude, Llama, Titan). | `LLM_BASE_URL=https://<your-bedrock-proxy>/v1`<br>`LLM_API_KEY=<proxy-issued token>`<br>`LLM_MODEL=<bedrock-model-id e.g. anthropic.claude-3-sonnet>` |
| **Azure OpenAI** | Azure already exposes Cross-platform routes (`https://<resource>.openai.azure.com/openai`). Create a deployment and use its name as the `model`. | `LLM_BASE_URL=https://<resource>.openai.azure.com`<br>`LLM_API_KEY=<Azure OpenAI key>`<br>`LLM_MODEL=<deployment name>` |
| **GCP Vertex AI** | Serve the model via Vertex AI and front it with the [OpenAI proxy sample](https://github.com/GoogleCloudPlatform/generative-ai/tree/main/openai-proxy) or litellm on Cloud Run. | `LLM_BASE_URL=https://<cloud-run-proxy>/v1`<br>`LLM_API_KEY=<IAM token or custom key>`<br>`LLM_MODEL=<publisher/model id>` |
| **Databricks** | Enable the [Databricks Model Serving OpenAI endpoint](https://docs.databricks.com/en/machine-learning/foundation-models/openai-api.html) or host litellm on DBSQL. | `LLM_BASE_URL=https://<workspace-url>/serving-endpoints/openai/v1`<br>`LLM_API_KEY=<Databricks PAT>`<br>`LLM_MODEL=<served model, e.g. databricks-dbrx-instruct>` |
| **Other vendors / self-hosted** | Use litellm/OpenRouter/text-generation-inference (OpenAI bridge mode) to normalize the API. Point the proxy upstream to Anthropic, Cohere, Elastic, etc. | `LLM_BASE_URL=https://<router>/v1`<br>`LLM_API_KEY=<router token>`<br>`LLM_MODEL=<identifier required by the router>` |

Once the endpoint speaks the OpenAI schema, no code changes are required—the pipeline keeps producing the same Excel/JSON/email artifacts regardless of where the model actually runs.

## Local Transformers Backend

1. Set `MODEL_BACKEND=local`.
2. Configure `LOCAL_MODEL_NAME`, `LOCAL_PIPELINE_TASK`, and (optionally) generation knobs such as `LOCAL_MAX_NEW_TOKENS`, `LOCAL_TEMPERATURE`, `LOCAL_TOP_P`, `LOCAL_PROMPT_TEMPLATE`, `LOCAL_DEVICE_MAP`, `LOCAL_TORCH_DTYPE`.
3. The pipeline instantiates a `transformers.pipeline` (`trust_remote_code` defaults to true) and calls it with chat-style `messages`. This is ideal for self-hosting Falcon, Mistral, Kimi Linear, or other large models.
4. For `device_map=auto`, install `accelerate` (already listed in `requirements.txt`); otherwise keep `LOCAL_DEVICE_MAP` blank to avoid runtime errors.

## BERTimbau Backend Local Fine-Tuning

1. Set `MODEL_BACKEND=bert`.
2. Useful overrides: `BERT_MODEL_NAME`, `BERT_MAX_LENGTH`, `BERT_BATCH_SIZE`, `BERT_EPOCHS`, `BERT_LEARNING_RATE`, `BERT_WEIGHT_DECAY`, `BERT_WARMUP_RATIO`, `BERT_OUTPUT_SUBDIR`.
3. The pipeline uses the raw concatenated text (`texto_bruto`). GPU is recommended but optional.
4. Outputs include `fonte_classificacao='bert'`, `modelo_classificacao`, and accuracy metrics in the JSON report.

## Classic ML Backend

Keeping `MODEL_BACKEND=ml` preserves the original TF-IDF / BoW approach on the preprocessed `texto_limpo` column. It reports accuracy per vectorizer and the top features per class.

## Project Structure

- `package/python/aut.py`: orchestrates the input/output flow and selects the backend (`ml`, `bert`, `hf`, `llm`, `local`).
- `package/python/modeling_utils.py`: classic ML pipelines, BERT training/inference, and mapping predictions to the dashboard columns.
- `package/python/hf_api_utils.py`: inference client for Hugging Face Spaces/models.
- `package/python/llm_utils.py`: Cross-platform orchestration (prompting, retries, mapping).
- `package/python/reporting_utils.py`: exports metrics metadata to JSON and Excel.
- `inputs/`, `output/`: default folders for data ingress/egress.

## Architecture

The high-level dataflow is captured in `design/architecture.mmd` (Mermaid diagram). It starts with `.env.local` and `config.py` resolving runtime settings, then `io_utils` and `preprocessing_utils` load Excel files from `inputs/`, normalize text, and feed `aut.py`. The orchestrator evaluates `MODEL_BACKEND` to delegate to one of five engines: Hugging Face API (`hf_api_utils`), Cross-platform LLM (`llm_utils`), BERT fine-tuning or classic ML pipelines inside `modeling_utils`, and local `transformers` pipelines (`local_model_utils`). All backends return a classified dataframe that flows into `reporting_utils`, `visualization_utils`, and `email_utils` to produce Excel/JSON artifacts, visual dashboards, and optional stakeholder alerts.

```mermaid
%% OmniTag Universal architecture diagram (Mermaid)
flowchart LR
    subgraph Env["Environment and Configuration"]
        ENV[.env.local / Environment variables<br/>HUGGINGFACE_API_KEY, *]
        CFG[config.py<br/>Centralizes settings]
        ENV --> CFG
    end

    subgraph Ingestion["Data Ingestion and ETL"]
        INPUT[[Excel spreadsheet in inputs/]]
        IO[io_utils<br/>ensure_directories + resolve_input_file]
        PREP[preprocessing_utils<br/>load_dataset + clean text + ensure NLTK]
        INPUT --> IO --> PREP
    end

    ORCH[aut.py<br/>Pipeline Orchestrator]
    CFG --> ORCH
    PREP --> ORCH

    DECIDE{MODEL Backend}
    ORCH --> DECIDE

    subgraph HFBackend["HF API Backend Default"]
        HF[hf_api_utils<br/>gradio_client]
        HFF[Hugging Face Space / Inference Endpoint]
        HF --> HFF
    end
    DECIDE -->|HF| HF

    subgraph LLMBackend["LLM Backend Cross-Platform"]
        LLM[llm_utils<br/>OpenAI, Bedrock, Azure, Vertex, Databricks]
        LLMAPI[Proxy/router<br/>HF, AWS, Azure, GCP,  Databricks, Litellm]
        LLM --> LLMAPI
    end
    DECIDE -->|LLM| LLM

    subgraph BERTBackend["Local BERT Fine-Tuning"]
        BERTTRAIN[modeling_utils<br/>train_and_evaluate_bert]
        BERTINFER[apply_bert_model]
        BERTTRAIN --> BERTINFER
    end
    DECIDE -->|BERT| BERTTRAIN

    subgraph ClassicBackend["Classic ML Backend"]
        CLASSIC[modeling_utils<br/>TF-IDF/BoW + Logistic Regression]
    end
    DECIDE -->|ML| CLASSIC

    subgraph LocalBackend["Local Transformers Backend"]
        LOCAL[local_model_utils<br/>transformers.pipeline]
    end
    DECIDE -->|LOCAL| LOCAL

    HF & LLM & BERTINFER & CLASSIC & LOCAL --> CLASSIFIED["Classified Dataframe"]

    subgraph PostProcessing["Post-Processing and Delivery"]
        REPORT[reporting_utils<br/>save_classified_data + metrics JSON]
        VIS[visualization_utils<br/>word clouds + charts]
        EMAIL[email_utils<br/>send_result_email]
    end
    CLASSIFIED --> REPORT
    CLASSIFIED --> VIS
    CLASSIFIED --> EMAIL

    REPORT --> EXCEL[[output/tagged_file.xlsx]]
    REPORT --> JSON[[outputs/nlp_metrics.json]]
    VIS --> WORDCLOUDS[[output/nlp_visualizations/*]]
    EMAIL --> STAKEHOLDERS[Stakeholder Notification]

    subgraph QA["Quality Automation and Observability"]
        AUT_TESTS[robots/aut_tests.sh<br/>bootstrap venv + pytest + dashboard]
        PYTEST[tests/* + pytest.ini]
        DASHBOARD[package/python/test_metrics_dashboard.py<br/>Streamlit metrics]
        AUT_TESTS --> PYTEST --> ORCH
        AUT_TESTS --> DASHBOARD
        DASHBOARD --> JSON
    end
```

## Minimum Requirements

- **Operating system**: Windows 10 64-bit (x64). Install the Microsoft Visual C++ Redistributable 2015–2022 for PyTorch DLLs.
- **Processor**: At least 4 cores and 8 GB RAM (16 GB recommended for BERT/local transformers).
- **Disk space**: ~5 GB free (repository, `.venv`, inputs, outputs).
- **Python**: 3.11.x (64-bit) with the packages in `requirements.txt`.
- **Optional GPU**: If running the local backend on GPU, ensure CUDA/cuDNN matches your PyTorch build.

## Software Stack

- Python 3.11 with virtual environment
- pandas, numpy, scikit-learn, matplotlib, seaborn, wordcloud
- streamlit (dashboards)
- PyTorch (CPU build), transformers, accelerate, datasets, tqdm
- openpyxl, requests, openai, gradio_client
- pytest + unit tests under `tests/`

## Key Environment Variables

| Variable | Description |
| --- | --- |
| `TEXT_COLUMNS` | Source columns concatenated into `texto_bruto` (defaults to `Procedencia da Coleta`, `Ponto de Coleta`, `Grupo de parametros`, `Parametro (demais parametros)`, `LD`, `LQ`, `Resultado`). |
| `LABEL_COLUMN` | Ground-truth field containing `Sim/Nãoo/Avaliar`. |
| `MODEL_BACKEND` | `ml`, `bert`, `hf`, `llm`, or `local` (default `hf`). |
| `LLM_*` | Parameters for the optional LLM backend (requires `LLM_BASE_URL`). |
| `BERT_*` | Hyperparameters for fine-tuning BERT. |
| `HF_*` | Hugging Face Space/model configuration (id, endpoint, retries, optional prompt template, generation parameters). |
| `LOCAL_*` | Local backend settings for `transformers.pipeline`: model id, prompt template, generation tokens, temperature, `device_map`, etc. |
| `HUGGINGFACE_API_KEY` | Token consumed by Hugging Face inference and LLM calls. |
| `INPUT_FILE` | Optional explicit path to the Excel file. |
| `EMAIL_*` | Credentials for the optional email notification. |

## Large Spreadsheets

Files like `vigilancia_demais_parametros.xlsx` contain hundreds of thousands of rows. To prototype with smaller batches without modifying the original dataset:

```bash
## 1) Download + convert CSV -> XLSX (inputs/vigilancia_demais_parametros.xlsx*)
bash robots/aut_extraction.sh
## 2) Split the generated spreadsheet into manageable parts
python -m package.python.split_csv inputs/vigilancia_demais_parametros.xlsx --output-dir inputs/chunks --chunk-size 500
##    * If aut_extraction creates `inputs/vigilancia_demais_parametros_part_XXX.xlsx`
##      because the dataset exceeds Excel's row limit, point the command above to the
##      desired part file instead of the base name.

## Alternative: split the CSV directly into XLSX chunks (skips the intermediate file)
python -m package.python.split_csv backup/vigilancia_demais_parametros.csv \
    --output-dir inputs/chunks \
    --chunk-size 500 \
    --output-format xlsx \
    --csv-delimiter ';' \
    --fallback-encodings "latin-1,cp1252"

## 3) Execute the full pipeline for every chunk and save per-chunk outputs
python -m package.python.aut \
    --chunk-dir inputs/chunks \
    --output-dir output/chunks
```

`--output-format xlsx` tells the helper to read the CSV and emit Excel chunks directly (each named `*_part_XXX.xlsx`). `--csv-delimiter ';'` matches the SISAGUA export, and `--fallback-encodings` ensures legacy Latin encodings are tried if UTF-8 fails. Leave `--output-format` as the default (`auto`) to keep CSV chunks.

Running `python -m package.python.aut` without extra flags now defaults to chunk mode: it walks through every XLSX inside `inputs/chunks/`, runs the usual pipeline for each file, and stores the enriched result as `output/chunks/<chunk>_tagged.xlsx`. Override `--chunk-dir` or `--output-dir` if you prefer custom paths. For a single chunk, append `--single-run --input-file chunks/your_chunk.xlsx`.

Each chunk is saved as `*_part_XXX.xlsx`. To run the pipeline on a specific chunk, copy the generated file into `inputs/` (if necessary) and execute `INPUT_FILE=name_of_chunk.xlsx python -m package.python.aut`.

## Open Source and Community

This repository is released under the [MIT License](LICENSE), which allows reuse in commercial and government contexts provided attribution is preserved. Contributions from the community are welcomeâ€”review [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow and reference the [Code of Conduct](Code%20of%20Conduct.md) when engaging with other contributors.

</span>

