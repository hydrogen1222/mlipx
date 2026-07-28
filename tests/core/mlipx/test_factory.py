"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import pytest
from mlipx.base_calculator import BaseMLIPCalculator
from mlipx.calculator import UMACalculator
from mlipx.calculators.dpa_calc import DPACalculatorWrapper
from mlipx.calculators.factory import SUPPORTED_TYPES, CalculatorFactory
from mlipx.calculators.grace_calc import GRACECalculatorWrapper
from mlipx.calculators.mace_calc import MACECalculatorWrapper
from mlipx.engine import EngineConfig


class TestBaseMLIPCalculator:
    """Tests for the abstract calculator interface."""

    def test_is_abstract(self):
        """BaseMLIPCalculator cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseMLIPCalculator()  # type: ignore[abstract]

    def test_abstract_methods(self):
        """All required abstract methods are declared."""
        expected = {"get_calculator", "task", "has_stress", "info"}
        assert expected.issubset(BaseMLIPCalculator.__abstractmethods__)

    def test_inference_mode_default(self):
        """The default inference_mode property exists and returns 'default'."""
        getter = BaseMLIPCalculator.inference_mode.fget  # type: ignore[attr-defined]
        assert getter is not None


class TestCalculatorFactory:
    """Tests for the multi-engine calculator factory."""

    def test_supported_types(self):
        assert "uma" in SUPPORTED_TYPES
        assert {"mace", "dpa", "grace"}.issubset(SUPPORTED_TYPES)

    def test_uma_creates_umacalculator(self, tmp_path):
        """UMA engine produces a UMACalculator that is a BaseMLIPCalculator."""
        fake_model = tmp_path / "uma.pt"
        fake_model.write_text("x")
        with patch.object(UMACalculator, "_validate"):
            w = CalculatorFactory.create("uma", fake_model, task="omat", device="cpu")
        assert isinstance(w, UMACalculator)
        assert isinstance(w, BaseMLIPCalculator)
        assert w.task == "omat"

    def test_fairchem_alias(self, tmp_path):
        """'fairchem' is accepted as an alias for 'uma'."""
        fake_model = tmp_path / "uma.pt"
        fake_model.write_text("x")
        with patch.object(UMACalculator, "_validate"):
            w = CalculatorFactory.create("fairchem", fake_model, task="omat")
        assert isinstance(w, UMACalculator)

    def test_unsupported_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported model type"):
            CalculatorFactory.create("unknown", "x.model")

    def test_model_type_case_insensitive(self, tmp_path):
        """Model type matching is case-insensitive."""
        fake_model = tmp_path / "uma.pt"
        fake_model.write_text("x")
        with patch.object(UMACalculator, "_validate"):
            w = CalculatorFactory.create("UMA", fake_model, task="omat")
        assert w.task == "omat"


class TestGenericWrappers:
    """Tests for MACE/DPA/GRACE wrapper construction (lazy import)."""

    def test_mace_wrapper_construction(self, tmp_path):
        model = tmp_path / "mace.model"
        model.write_text("x")
        w = MACECalculatorWrapper(model, device="cpu", task="bulk")
        assert isinstance(w, BaseMLIPCalculator)
        assert w.task == "bulk"
        assert w.inference_mode == "default"  # non-UMA engines ignore inference_mode

    def test_mace_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            MACECalculatorWrapper(tmp_path / "nope.model")

    def test_mace_missing_package_raises_importerror(self, tmp_path):
        """get_calculator raises ImportError when mace-torch is absent (lazy)."""
        model = tmp_path / "mace.model"
        model.write_text("x")
        w = MACECalculatorWrapper(model)
        with pytest.raises(ImportError, match="mace-torch"):
            w.get_calculator()

    def test_dpa_wrapper_construction(self, tmp_path):
        model = tmp_path / "dpa.pth"
        model.write_text("x")
        w = DPACalculatorWrapper(model, task="molecule")
        assert isinstance(w, BaseMLIPCalculator)
        assert w.task == "molecule"
        assert w.inference_mode == "default"

    def test_grace_wrapper_construction(self, tmp_path):
        model = tmp_path / "grace"
        model.write_text("x")
        w = GRACECalculatorWrapper(model, task="bulk")
        assert isinstance(w, BaseMLIPCalculator)
        assert w.task == "bulk"

    def test_mace_info_with_mocked_calculator(self, tmp_path):
        """info() works once a calculator is available (mocked)."""
        model = tmp_path / "mace.model"
        model.write_text("x")
        w = MACECalculatorWrapper(model, default_dtype="float32")

        class FakeCalc:
            implemented_properties: ClassVar = ["energy", "forces", "stress"]

        w._calculator = FakeCalc()  # type: ignore[assignment]
        info = w.info()
        assert info["model_type"] == "mace"
        assert info["default_dtype"] == "float32"
        assert info["has_stress"] is True
        assert info["implemented_properties"] == ["energy", "forces", "stress"]

    def test_factory_creates_mace_wrapper(self, tmp_path):
        model = tmp_path / "mace.model"
        model.write_text("x")
        w = CalculatorFactory.create(
            "mace", model, task="bulk", default_dtype="float32"
        )
        assert isinstance(w, MACECalculatorWrapper)
        assert w._default_dtype == "float32"


class TestEngineModelType:
    """Tests that EngineConfig model_type flows to the factory."""

    def test_engine_config_has_model_type_default(self):
        cfg = EngineConfig(calc_type="sp", model_path=Path("uma-s-1.pt"))
        assert cfg.model_type == "uma"

    def test_engine_config_custom_model_type(self):
        cfg = EngineConfig(
            calc_type="sp", model_path=Path("x.model"), model_type="mace", task="bulk"
        )
        assert cfg.model_type == "mace"
        assert cfg.task == "bulk"
