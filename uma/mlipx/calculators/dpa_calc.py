# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Modified for the mlipx project: DPA (DeepMD-kit) engine wrapper.

"""
DPA (DeepMD-kit) engine wrapper.

Wraps ``deepmd.calculator.DP`` behind the ``BaseMLIPCalculator`` contract.
``deepmd-kit`` is imported lazily so mlipx works without it installed; users
only need it when selecting ``--model-type dpa``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from mlipx.base_calculator import BaseMLIPCalculator

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator


class DPACalculatorWrapper(BaseMLIPCalculator):
    """Wrapper for DeepMD-kit DPA ASE calculators."""

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cpu",
        task: str = "bulk",
    ):
        """
        Initialize DPA calculator wrapper.

        Args:
            model_path: Path to a DPA model (``.pth``/``.pt`` for PyTorch
                backend, or ``.pb`` for the legacy TensorFlow backend).
            device: Device for calculation (``cpu`` or ``cuda``).
            task: PBC hint (``bulk`` or ``molecule``); not consumed by DPA.
        """
        self.model_path = Path(model_path)
        self._device = device
        self._task = task
        self._calculator: Calculator | None = None

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

    def get_calculator(self) -> Calculator:
        """Return the cached DPA ASE calculator (lazy import)."""
        if self._calculator is None:
            try:
                from deepmd.calculator import DP  # noqa: PLC0415
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "DPA support requires the 'deepmd-kit' package.\n"
                    "Install with: pip install deepmd-kit>=3.0.0"
                ) from e
            self._calculator = DP(
                model=str(self.model_path),
                type_dict=None,
            )
        return self._calculator

    @property
    def task(self) -> str:
        """PBC hint (``bulk``/``molecule``); DPA itself is task-agnostic."""
        return self._task

    @property
    def has_stress(self) -> bool:
        """Whether stress is supported."""
        return "stress" in self.implemented_properties

    def info(self) -> dict:
        """Return model metadata."""
        return {
            "model_type": "dpa",
            "model_path": str(self.model_path),
            "device": self._device,
            "task": self._task,
            "implemented_properties": self.implemented_properties,
            "has_stress": self.has_stress,
        }
