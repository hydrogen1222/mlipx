"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from mlipx.calculator import UMACalculator


class TestUMACalculatorValidation:
    """Tests for UMACalculator parameter validation (no model needed)."""

    def test_invalid_task_raises(self):
        with (
            patch.object(Path, "exists", return_value=True),
            pytest.raises(ValueError, match="Invalid task"),
        ):
            UMACalculator(model_path="fake.pt", task="invalid")

    def test_invalid_inference_mode_raises(self):
        with (
            patch.object(Path, "exists", return_value=True),
            pytest.raises(ValueError, match="Invalid inference mode"),
        ):
            UMACalculator(model_path="fake.pt", inference_mode="invalid")

    def test_resource_settings_stored(self):
        with patch.object(Path, "exists", return_value=True):
            calc = UMACalculator(
                model_path="fake.pt",
                torch_num_threads=4,
                activation_checkpointing=False,
            )
            assert calc.torch_num_threads == 4
            assert calc.activation_checkpointing is False

    def test_default_resource_settings(self):
        with patch.object(Path, "exists", return_value=True):
            calc = UMACalculator(model_path="fake.pt")
            assert calc.torch_num_threads is None
            assert calc.activation_checkpointing is None

    def test_task_case_insensitive(self):
        with patch.object(Path, "exists", return_value=True):
            calc = UMACalculator(model_path="fake.pt", task="OMAT")
            assert calc.task == "omat"
