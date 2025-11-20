from pathlib import Path

import pandas as pd

from package.python import email_utils as email


def test_parse_recipients_supports_multiple_separators():
    recipients = email.parse_recipients('a@example.com; b@example.com, c@example.com')

    assert recipients == ['a@example.com', 'b@example.com', 'c@example.com']


def test_build_email_summary_includes_counts(tmp_path):
    df = pd.DataFrame({'Categoria': ['Sim', 'Não', 'Sim']})
    dataset_path = Path(tmp_path / 'dataset.xlsx')
    dataset_path.write_text('placeholder')

    summary = email.build_email_summary(df, dataset_path)

    assert 'Records processed: 3' in summary
    assert 'Sim: 2' in summary
    assert dataset_path.name in summary
