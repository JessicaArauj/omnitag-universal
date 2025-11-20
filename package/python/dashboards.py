import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


class _ColumnStub:
    def metric(self, *args, **kwargs):
        return None


try:
    import streamlit as st
except Exception:  # pragma: no cover - fallback for tests

    class _StreamlitStub:
        def set_page_config(self, *args, **kwargs):
            return None

        def title(self, *args, **kwargs):
            return None

        def markdown(self, *args, **kwargs):
            return None

        def warning(self, *args, **kwargs):
            return None

        def error(self, *args, **kwargs):
            return None

        def subheader(self, *args, **kwargs):
            return None

        def columns(self, count):
            return tuple(_ColumnStub() for _ in range(count))

        def caption(self, *args, **kwargs):
            return None

        def dataframe(self, *args, **kwargs):
            return None

        def info(self, *args, **kwargs):
            return None

        def pyplot(self, *args, **kwargs):
            return None

        def bar_chart(self, *args, **kwargs):
            return None

        def success(self, *args, **kwargs):
            return None

        def stop(self, *args, **kwargs):
            return None

    st = _StreamlitStub()
else:  # pragma: no cover - ensure required APIs exist
    def _ensure_attr(name, factory=None):
        if hasattr(st, name):
            return
        if factory:
            setattr(st, name, factory)
        else:
            setattr(st, name, lambda *args, **kwargs: None)

    _ensure_attr(
        'columns',
        lambda count: tuple(_ColumnStub() for _ in range(count)),
    )
    for attr in (
        'info',
        'warning',
        'error',
        'subheader',
        'caption',
        'dataframe',
        'pyplot',
        'bar_chart',
        'success',
        'stop',
        'set_page_config',
        'title',
        'markdown',
    ):
        _ensure_attr(attr)

from config import EXCLUDED_RESULT_COLUMNS, TEST_REPORT_FILE

# === Home Page ===
st.set_page_config(
    page_title='Dashboard',
    layout='wide',
    initial_sidebar_state='expanded',
)

st.title('Painel de Categorização de Viabilidade Técnica')
st.markdown(
    'Visualize e explore os resultados da categorização de análise de '
    'viabilidade técnica'
)

# === Path Outputs  ===
BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_PATH = BASE_DIR / 'input' / 'tagged_file.xlsx'


def load_test_report():
    if not TEST_REPORT_FILE.exists():
        st.warning(
            f'Automated test report not found at "{TEST_REPORT_FILE}". '
            'Run robots/aut_tests.sh to generate it.'
        )
        return None
    try:
        return json.loads(TEST_REPORT_FILE.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        st.error(f'Unable to parse test report JSON: {exc}')
        return None


def render_test_summary(report: dict) -> None:
    st.subheader('Automated Test Results')
    summary = report.get('summary', {})
    passed = int(summary.get('passed', 0))
    failed = int(summary.get('failed', 0))
    error = int(summary.get('error', 0))
    skipped = int(summary.get('skipped', 0))
    xfailed = int(summary.get('xfailed', 0))
    xpassed = int(summary.get('xpassed', 0))
    total = (
        passed + failed + error + skipped + xfailed + xpassed
        + int(summary.get('rerun', 0))
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Total tests', total or 'n/a')
    col2.metric('Passed', passed)
    col3.metric('Failed/Error', failed + error)
    col4.metric('Skipped', skipped)

    duration = report.get('duration', 'n/a')
    exit_code = report.get('exitcode', 'n/a')
    st.caption(f'Duration: {duration}s | Exit code: {exit_code}')

    tests = report.get('tests') or []
    if tests:
        frame = pd.DataFrame(tests)
        keep_cols = [
            column
            for column in ('nodeid', 'outcome', 'duration', 'keywords')
            if column in frame.columns
        ]
        st.dataframe(
            frame[keep_cols].sort_values('outcome'),
            use_container_width=True,
        )
    else:
        st.info('Detailed test results are not available in the report.')


# === Pipeline CI/CD ===
def load_data():
    if OUTPUT_PATH.exists():
        return pd.read_excel(OUTPUT_PATH)

    st.error(
        'O arquivo de saída categorizado ainda não foi gerado '
        'pela esteira de processamento.'
    )
    return None


df = load_data()
test_report = load_test_report()
if test_report:
    render_test_summary(test_report)
st.markdown('---')

if df is not None:
    st.success(f'Arquivo carregado automaticamente: {OUTPUT_PATH}')

    category_col = 'Categoria'
    if category_col not in df.columns:
        category_col = 'Possibilidade' if 'Possibilidade' in df.columns else None

    drop_columns = [
        column
        for column in EXCLUDED_RESULT_COLUMNS
        if column in df.columns and column != category_col
    ]
    if drop_columns:
        df = df.drop(columns=drop_columns)

    if not category_col or category_col not in df.columns:
        st.error(
            'Nenhuma coluna de categoria foi encontrada no arquivo '
            'carregado. Execute novamente a esteira para gerar os dados.'
        )
        st.stop()

    # === KPI  ===
    col1, col2, col3 = st.columns(3)
    total = len(df)
    auto_yes = df[df[category_col] == 'Sim'].shape[0]
    avg_conf = df['confianca'].mean()

    col1.metric('Total de Registros', f'{total}')
    col2.metric('Automatizáveis (Sim)', f'{auto_yes}')
    col3.metric('Confiança Média', f'{avg_conf:.2f}')

    # === Graphs ===
    st.subheader('Distribuição das Categorias de Viabilidade')
    fig, ax = plt.subplots()
    df[category_col].value_counts().plot(
        kind='bar',
        ax=ax,
        color=['#4CAF50', '#FFC107', '#F44336'],
    )
    ax.set_title('Categorias de Viabilidade de Automação')
    ax.set_xlabel('Categoria')
    ax.set_ylabel('Quantidade')
    st.pyplot(fig)

    st.subheader('Confiança e Prioridade Média por Categoria')
    avg_metrics = (
        df.groupby(category_col)[
            ['confianca', 'Pontuação de Prioridade']
        ]
        .mean()
        .round(2)
    )
    st.bar_chart(avg_metrics)

    # === Table ===
    st.subheader('Resultados em Detalhes')
    st.dataframe(df, use_container_width=True)

    # === Automation Update ===
    st.caption(
        'O painel é atualizado automaticamente sempre que um novo arquivo '
        'categorizado é gerado.'
    )
else:
    st.warning(
        'Aguardando geração do arquivo categorizado na esteira de '
        'processamento...'
    )
