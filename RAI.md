<span align="justify">

# Responsible AI Guidelines

This open-source project classifies SISAGUA water-quality records to estimate whether a source is fit for human consumption. Treat the predictions as decision support only—**never** as the sole basis for regulatory, health, or enforcement actions.

## Data stewardship

- **Input boundaries:** only ingest SISAGUA-derived analytical data or datasets with equivalent public-usage rights. Remove personally identifiable information (PII) before loading spreadsheets.
- **Label provenance:** ensure the target column reflects ground truth from qualified lab technicians or environmental health authorities and that reuse complies with open-data terms. Keep an audit trail of the labeling process.
- **Retention:** delete temporary training files and intermediate exports from shared or public storage once the classification run is completed.

## Fairness and bias

- Monitor label distribution (`outputs/nlp_metrics.json`) to verify that every class has enough samples before retraining.
- When using LLM or Hugging Face backends, log the prompts/responses for spot checks and redact sensitive content before storing long-term artifacts.
- Review model outputs with a human in the loop; update the category playbook and retrain if biases or systematic errors are identified.

## Transparency

- The generated Excel file exposes the predicted category, score, priority, and a textual justification (standardized for ML/BERT or returned by the remote API/LLM).
- Keep the README and environment variables up to date so stakeholders understand which backend is deployed (`ml`, `bert`, `hf`, or `llm`).

## Safety controls

- Store API keys (`HUGGINGFACE_API_KEY`) in secure secret managers (GitHub Secrets, Azure Key Vault, etc.).
- Activate rate limits and retries (already configurable) to prevent overwhelming external services.
- Before deploying to production, run the pipeline against a curated validation spreadsheet to confirm there are no regressions in accuracy or business logic.

</span>
