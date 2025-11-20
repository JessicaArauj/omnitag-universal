"""Email sending helpers for the pipeline."""

from __future__ import annotations

import re
import smtplib
from email.message import EmailMessage
from pathlib import Path
from typing import List

import pandas as pd

from .config import (
    EMAIL_ENABLED,
    EMAIL_FROM,
    EMAIL_SUBJECT,
    EMAIL_TO,
    SMTP_PASSWORD,
    SMTP_PORT,
    SMTP_SERVER,
    SMTP_USE_TLS,
    SMTP_USER,
    TARGET_LABELS,
)


def parse_recipients(raw: str) -> List[str]:
    """Convert comma/semicolon separated string into a list."""
    if not raw:
        return []
    parts = re.split(r'[;,]', raw)
    return [email.strip() for email in parts if email.strip()]


def build_email_summary(df: pd.DataFrame, dataset_path: Path) -> str:
    """Create a short textual summary for the email body."""
    total = len(df)
    category_col = 'Categoria'
    if category_col not in df.columns:
        category_col = 'Possibilidade' if 'Possibilidade' in df.columns else None
    counts = (
        df[category_col].value_counts().to_dict()
        if category_col and category_col in df.columns
        else {}
    )
    lines: list[str] = [
        'Hello,',
        '',
        'Please find attached the updated technical feasibility '
        'categorization file.',
        f'Dataset: {dataset_path.name}',
        f'Records processed: {total}',
        '',
        'Category distribution:',
    ]
    for label in sorted(TARGET_LABELS):
        lines.append(f' - {label}: {counts.get(label, 0)}')
    lines.extend(
        [
            '',
            'Key columns: Categoria, acao_sugerida, confianca, '
            'pontuacao_prioridade e justificativa.',
            '',
            'Best regards,',
            'Digital Pipeline',
        ]
    )
    return '\n'.join(lines)


def send_result_email(file_path: Path, df: pd.DataFrame, dataset_path: Path):
    """Send the generated Excel file via email if configured."""
    if not EMAIL_ENABLED:
        return

    recipients = parse_recipients(EMAIL_TO)
    if not EMAIL_FROM or not recipients:
        print(
            'Email skipped: set EMAIL_FROM and EMAIL_TO first.'
        )
        return

    if not file_path.exists():
        print(f'Email skipped: file not found ({file_path}).')
        return

    message = EmailMessage()
    message['Subject'] = EMAIL_SUBJECT
    message['From'] = EMAIL_FROM
    message['To'] = ', '.join(recipients)
    message.set_content(build_email_summary(df, dataset_path))

    attachment_data = file_path.read_bytes()
    message.add_attachment(
        attachment_data,
        maintype='application',
        subtype=(
            'vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        ),
        filename=file_path.name,
    )

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=30) as server:
            if SMTP_USE_TLS:
                server.starttls()
            if SMTP_USER and SMTP_PASSWORD:
                server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(message)
        print(
            'Email sent successfully to: '
            + ', '.join(recipients)
        )
    except Exception as exc:  # noqa: BLE001 - log only
        print(f'Failed to send email: {exc}')