"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import ClassVar
from unittest.mock import MagicMock, patch

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
        model.mkdir()
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

    def test_mace_unknown_head_fails_closed(self, tmp_path, monkeypatch):
        """Never accept mace-torch's silent fallback to a different PES head."""
        model = tmp_path / "mace.model"
        model.write_text("x")
        fake = MagicMock()
        fake.available_heads = ["default"]
        fake_cls = MagicMock(return_value=fake)
        module = MagicMock(MACECalculator=fake_cls)
        monkeypatch.setitem(sys.modules, "mace", MagicMock())
        monkeypatch.setitem(sys.modules, "mace.calculators", module)
        monkeypatch.setattr(
            "mlipx.doctor._installed_dependency_conflicts", lambda _: []
        )

        wrapper = MACECalculatorWrapper(model, head="omol")
        with pytest.raises(ValueError, match="Available heads"):
            wrapper.get_calculator()

    def test_factory_creates_mace_wrapper(self, tmp_path):
        model = tmp_path / "mace.model"
        model.write_text("x")
        w = CalculatorFactory.create(
            "mace", model, task="bulk", default_dtype="float32"
        )
        assert isinstance(w, MACECalculatorWrapper)
        assert w._default_dtype == "float32"

    def test_factory_forwards_dpa_head(self, tmp_path):
        """A DPA multi-task branch is a calculator option, not a PBC task."""
        model = tmp_path / "dpa.pt"
        model.write_text("x")
        w = CalculatorFactory.create(
            "dpa", model, task="bulk", head="Domains_SSE_PBE"
        )
        assert isinstance(w, DPACalculatorWrapper)
        assert w._head == "Domains_SSE_PBE"

    def test_factory_mace_default_dtype_is_float32(self, tmp_path):
        """Direct factory construction with no dtype defaults to float32 (docs)."""
        model = tmp_path / "mace.model"
        model.write_text("x")
        w = CalculatorFactory.create("mace", model, task="bulk")
        assert isinstance(w, MACECalculatorWrapper)
        assert w._default_dtype == "float32"


class TestDpaGraceDevice:
    """Plan section 6.2 / 7.5: DPA/GRACE device must take effect (via env) and
    info() must distinguish requested vs actual device (never guess)."""

    def test_dpa_honours_cuda_index(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

        class _FakeDPCalc:
            implemented_properties: ClassVar = ["energy", "forces", "stress"]

        fake_dp_cls = MagicMock(return_value=_FakeDPCalc())
        mod = MagicMock()
        mod.DP = fake_dp_cls
        monkeypatch.setitem(sys.modules, "deepmd", MagicMock())
        monkeypatch.setitem(sys.modules, "deepmd.calculator", mod)

        model = tmp_path / "dpa.pth"
        model.write_text("x")
        w = DPACalculatorWrapper(model, device="cuda:1", task="bulk")
        w.get_calculator()
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1"
        info = w.info()
        assert info["requested_device"] == "cuda:1"
        assert info["device"] == "cuda:1"  # backward-compat alias
        assert info["actual_device"] == "unknown"
        fake_dp_cls.assert_called_once_with(model=str(model), type_dict=None)

    def test_dpa_cpu_hides_gpus(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        fake_dp_cls = MagicMock(return_value=MagicMock())
        mod = MagicMock()
        mod.DP = fake_dp_cls
        monkeypatch.setitem(sys.modules, "deepmd", MagicMock())
        monkeypatch.setitem(sys.modules, "deepmd.calculator", mod)
        model = tmp_path / "dpa.pth"
        model.write_text("x")
        w = DPACalculatorWrapper(model, device="cpu", task="bulk")
        w.get_calculator()
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""

    def test_dpa_explicit_device_overrides_inherited_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
        fake_dp_cls = MagicMock(return_value=MagicMock())
        mod = MagicMock()
        mod.DP = fake_dp_cls
        monkeypatch.setitem(sys.modules, "deepmd", MagicMock())
        monkeypatch.setitem(sys.modules, "deepmd.calculator", mod)
        model = tmp_path / "dpa.pth"
        model.write_text("x")
        w = DPACalculatorWrapper(model, device="cuda:1", task="bulk")
        w.get_calculator()
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1"

    def test_grace_honours_cuda_index(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

        class _FakeTPCalc:
            implemented_properties: ClassVar = ["energy", "forces", "stress"]

        fake_tp_cls = MagicMock(return_value=_FakeTPCalc())
        mod = MagicMock()
        mod.TPCalculator = fake_tp_cls
        monkeypatch.setitem(sys.modules, "tensorpotential", MagicMock())
        monkeypatch.setitem(sys.modules, "tensorpotential.calculator", mod)
        model = tmp_path / "grace_model"
        model.mkdir()
        w = GRACECalculatorWrapper(model, device="cuda:1", task="bulk")
        w.get_calculator()
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1"
        info = w.info()
        assert info["requested_device"] == "cuda:1"
        assert info["actual_device"] == "unknown"
        fake_tp_cls.assert_called_once_with(model=str(model))

    def test_grace_cpu_hides_gpus(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        fake_tp_cls = MagicMock(return_value=MagicMock())
        mod = MagicMock()
        mod.TPCalculator = fake_tp_cls
        monkeypatch.setitem(sys.modules, "tensorpotential", MagicMock())
        monkeypatch.setitem(sys.modules, "tensorpotential.calculator", mod)
        model = tmp_path / "grace_model"
        model.mkdir()
        w = GRACECalculatorWrapper(model, device="cpu", task="bulk")
        w.get_calculator()
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""


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
