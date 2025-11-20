#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"

echo "[aut-tests] Bootstrapping virtual environment..."
if [[ ! -d "${VENV_DIR}" ]]; then
  python -m venv "${VENV_DIR}"
fi

if [[ -d "${VENV_DIR}/Package" ]]; then
  source "${VENV_DIR}/Package/activate"
elif [[ -d "${VENV_DIR}/Scripts" ]]; then
  source "${VENV_DIR}/Scripts/activate"
else
  source "${VENV_DIR}/bin/activate"
fi

echo "[aut-tests] Installing dependencies..."
python -m pip install -q -U pip >/dev/null
python -m pip install -q -r "${PROJECT_ROOT}/requirements.txt"

echo "[aut-tests] Running pytest suite..."
cd "${PROJECT_ROOT}"
mkdir -p "${PROJECT_ROOT}/output"
python -m pytest --json-report --json-report-file "${PROJECT_ROOT}/output/test_results.json" "$@"
python -m package.python.test_report_utils --report "${PROJECT_ROOT}/output/test_results.json" || echo "[aut-tests] Failed to sanitize test report."

echo "[aut-tests] Saving Dashboard Report..."
python -m package.python.save_dashboard_report || echo "[aut-tests] Dashboard generation failed."

DASHBOARD_ENTRY="${PROJECT_ROOT}/package/python/test_metrics_dashboard.py"
if [[ -f "${DASHBOARD_ENTRY}" ]]; then
  echo "[aut-tests] Launching Streamlit dashboard (press Ctrl+C to stop)..."
  python -m streamlit run "${DASHBOARD_ENTRY}" --server.headless true
else
  echo "[aut-tests] Dashboard entry not found at ${DASHBOARD_ENTRY}; skipping Streamlit step."
  echo "           Create package/python/test_metrics_dashboard.py to enable"
fi
