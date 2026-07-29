"""Tests for mlipx.config.defaults (single source of truth)."""

from __future__ import annotations

import pytest
from mlipx.config.defaults import (
    BUILTIN_DEFAULTS,
    DEFAULT_DEVICE_BY_CALC_TYPE,
    build_incar_default,
    get_default,
    get_default_config,
)

# ---------------------------------------------------------------------------
# Built-in structure
# ---------------------------------------------------------------------------

def test_all_expected_scopes_present() -> None:
    expected = {
        "general", "resources", "batch", "output", "safety",
        "sp", "opt", "md",
        "calculator", "calculator.uma", "calculator.mace",
        "calculator.dpa", "calculator.grace",
    }
    actual = set(BUILTIN_DEFAULTS)
    missing = expected - actual
    assert not missing, f"Missing scopes: {missing}"


def test_no_inference_mode_in_calculator_scope() -> None:
    """inference_mode is model-level, not in the generic calculator scope."""
    assert "inference_mode" not in BUILTIN_DEFAULTS["calculator"]


def test_inference_mode_in_calc_type_scopes() -> None:
    assert BUILTIN_DEFAULTS["sp"]["inference_mode"] == "default"
    assert BUILTIN_DEFAULTS["opt"]["inference_mode"] == "default"
    assert BUILTIN_DEFAULTS["md"]["inference_mode"] == "turbo"


def test_device_per_calc_type() -> None:
    assert DEFAULT_DEVICE_BY_CALC_TYPE["sp"] == "cpu"
    assert DEFAULT_DEVICE_BY_CALC_TYPE["opt"] == "cpu"
    assert DEFAULT_DEVICE_BY_CALC_TYPE["md"] == "cuda"


def test_mace_default_dtype_is_float32() -> None:
    """Plan section 12: MACE default dtype is float32."""
    assert BUILTIN_DEFAULTS["calculator.mace"]["default_dtype"] == "float32"


# ---------------------------------------------------------------------------
# get_default helper
# ---------------------------------------------------------------------------

def test_get_default_calc_type_scoped() -> None:
    assert get_default("opt", "fmax") == 0.05


def test_get_default_fallback_to_general() -> None:
    assert get_default("opt", "strict_config") is False
    assert get_default("md", "strict_config") is False


def test_get_default_unknown_returns_fallback() -> None:
    assert get_default("sp", "nonexistent", "fallback") == "fallback"


def test_get_default_no_calc_type() -> None:
    assert get_default(None, "strict_config") is False


# ---------------------------------------------------------------------------
# INCAR template generation
# ---------------------------------------------------------------------------

def test_build_incar_default_sp() -> None:
    text = build_incar_default("sp")
    assert "CALC_TYPE = SP" in text
    assert "DEVICE = cpu" in text
    assert "INFERENCE_MODE = default" in text


def test_build_incar_default_md() -> None:
    text = build_incar_default("md")
    assert "CALC_TYPE = MD" in text
    assert "DEVICE = cuda" in text
    assert "INFERENCE_MODE = turbo" in text


def test_build_incar_default_invalid_type() -> None:
    with pytest.raises(ValueError, match="Unknown calculation type"):
        build_incar_default("nonexistent")


def test_get_default_config_returns_incar() -> None:
    config = get_default_config("sp")
    assert config.get_str("CALC_TYPE", "").lower() == "sp"
