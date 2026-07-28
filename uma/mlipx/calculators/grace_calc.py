# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Modified for the mlipx project: GRACE (tensorpotential) engine wrapper.

"""
GRACE engine wrapper.

Wraps ``tensorpotential.calculator.TPCalculator`` behind the
``BaseMLIPCalculator`` contract. ``tensorpotential`` is imported lazily so
mlipx works without it installed; users only need it when selecting
``--model-type grace``.

Note: GRACE uses a TensorFlow/XLA backend. It can coexist with PyTorch-based
engines (UMA/MACE) in the same environment, but they may compete for GPU
memory - consider isolating them with ``CUDA_VISIBLE_DEVICES``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from mlipx.base_calculator import BaseMLIPCalculator

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator


class GRACECalculatorWrapper(BaseMLIPCalculator):
    """Wrapper for GRACE (tensorpotential) ASE calculators."""

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cpu",
        task: str = "bulk",
    ):
        """
        Initialize GRACE calculator wrapper.

        Args:
            model_path: Path to a GRACE SavedModel directory or YAML config.
            device: Device for calculation (``cpu`` or ``cuda``).
            task: PBC hint (``bulk`` or ``molecule``); not consumed by GRACE.
        """
        self.model_path = Path(model_path)
        self._device = device
        self._task = task
        self._calculator: Calculator | None = None

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

    def get_calculator(self) -> Calculator:
        """Return the cached GRACE ASE calculator (lazy import)."""
        if self._calculator is None:
            try:
                from tensorpotential.calculator import TPCalculator  # noqa: PLC0415
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "GRACE support requires the 'tensorpotential' package.\n"
                    "Install with: pip install tensorpotential"
                ) from e
            self._calculator = TPCalculator(model=str(self.model_path))
        return self._calculator

    @property
    def task(self) -> str:
        """PBC hint (``bulk``/``molecule``); GRACE itself is task-agnostic."""
        return self._task

    @property
    def has_stress(self) -> bool:
        """Whether stress is supported."""
        return "stress" in self.implemented_properties

    def info(self) -> dict:
        """Return model metadata."""
        return {
            "model_type": "grace",
            "model_path": str(self.model_path),
            "device": self._device,
            "task": self._task,
            "implemented_properties": self.implemented_properties,
            "has_stress": self.has_stress,
        }
