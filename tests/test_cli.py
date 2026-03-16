# Author: gadwant
"""Tests for CLI argument parsing and command execution."""

import json

from bigocheck.cli import _parse_args, main


def test_parse_args_run_command():
    """CLI parser should return a namespace for run command."""
    args = _parse_args(
        [
            "run",
            "--target",
            "tests.targets:constant_sleep",
            "--sizes",
            "1",
            "2",
            "--trials",
            "1",
        ]
    )
    assert args.command == "run"
    assert args.target == "tests.targets:constant_sleep"
    assert args.sizes == [1, 2]
    assert args.trials == 1


def test_main_run_json_output(capsys):
    """Run command should emit valid JSON output."""
    main(
        [
            "run",
            "--target",
            "tests.targets:constant_sleep",
            "--sizes",
            "1",
            "2",
            "--trials",
            "1",
            "--json",
        ]
    )
    out = capsys.readouterr().out
    data = json.loads(out)
    assert "time_complexity" in data
    assert "measurements" in data
    assert len(data["measurements"]) == 2


def test_main_explain_command(capsys):
    """Explain command should print complexity details."""
    main(["explain", "O(n)"])
    out = capsys.readouterr().out
    assert "O(n)" in out

import pytest

def test_main_help(capsys):
    """Help command should print help and exit 0."""
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
