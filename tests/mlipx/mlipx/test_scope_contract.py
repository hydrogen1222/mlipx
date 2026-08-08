"""Public-surface regression tests for the MLMD-focused project scope."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from mlipx import api
from mlipx.cli import create_parser
from mlipx.config import get_schema


def test_analysis_v1_is_not_importable() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("mlipx.analysis")


def test_cli_has_no_analyze_command() -> None:
    parser = create_parser()
    subcommands = next(
        action.choices
        for action in parser._actions
        if getattr(action, "choices", None)
    )
    assert "analyze" not in subcommands


def test_adsorption_convenience_api_is_not_public() -> None:
    assert "calculate_adsorption_energy" not in api.__all__
    assert not hasattr(api, "calculate_adsorption_energy")


def test_equil_steps_is_an_unknown_option() -> None:
    errors = get_schema().validate_dict(
        {"EQUIL_STEPS": 10}, strict=True, context="INCAR"
    )
    assert len(errors) == 1
    assert "Unknown key 'EQUIL_STEPS' in INCAR" in errors[0]


def test_analysis_optional_dependencies_are_absent() -> None:
    pyproject = Path(__file__).parents[3] / "mlipx" / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    assert "kinisi" not in text
    assert "gemdat" not in text
    assert "matplotlib" not in text
