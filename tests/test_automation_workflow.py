"""Basic validation for the GitHub workflow that runs the automation pipeline."""

from __future__ import annotations

from pathlib import Path


def test_automation_workflow_contains_expected_steps() -> None:
    """Sanity check the automation workflow structure."""
    workflow_path = Path('.github/workflows/automation.yml')
    assert workflow_path.exists(), 'automation workflow file is missing'
    content = workflow_path.read_text(encoding='utf-8')

    # Trigger and runner configuration
    assert 'workflow_dispatch' in content, 'workflow must allow manual trigger'
    assert 'runs-on: ubuntu-latest' in content, 'workflow must run on ubuntu-latest'

    # Key steps and commands
    assert 'actions/checkout@v4' in content, 'repository checkout step missing'
    assert 'actions/setup-python@v5' in content, 'Python setup action missing'
    assert 'pip install -r requirements.txt' in content, 'dependency installation step missing'
    assert (
        'python -m package.python.automation' in content
    ), 'automation pipeline invocation missing'
