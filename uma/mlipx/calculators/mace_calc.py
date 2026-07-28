# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Modified for the mlipx project: MACE engine wrapper.

"""
MACE engine wrapper.

Wraps ``mace.calculators.MACECalculator`` behind the ``BaseMLIPCalculator``
contract. The ``mace-torch`` package is imported lazily so mlipx works without
it installed; users only need it when selecting ``--model-type mace``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from mlipx.base_calculator import BaseMLIPCalculator

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator


class MACECalculatorWrapper(BaseMLIPCalculator):
    """Wrapper for MACE ASE calculators."""

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cpu",
        default_dtype: str = "float64",
        task: str = "bulk",
    ):
        """
        Initialize MACE calculator wrapper.

        Args:
            model_path: Path to a MACE model file (``.model`` / ``.pt``).
            device: Device for calculation (``cpu`` or ``cuda``).
            default_dtype: Model dtype (``float32`` or ``float64``).
            task: PBC hint (``bulk`` or ``molecule``); not consumed by MACE.
        """
        self.model_path = Path(model_path)
        self._device = device
        self._default_dtype = default_dtype
        self._task = task
        self._calculator: Calculator | None = None

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

    def get_calculator(self) -> Calculator:
        """Return the cached MACE ASE calculator (lazy import)."""
        if self._calculator is None:
            try:
                from mace.calculators import MACECalculator  # noqa: PLC0415
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "MACE support requires the 'mace-torch' package.\n"
                    "Install with: pip install mace-torch"
                ) from e
            self._calculator = MACECalculator(
                model_paths=str(self.model_path),
                device=self._device,
                default_dtype=self._default_dtype,
            )
        return self._calculator

    @property
    def task(self) -> str:
        """PBC hint (``bulk``/``molecule``); MACE itself is task-agnostic."""
        return self._task

    @property
    def has_stress(self) -> bool:
        """Whether stress is supported."""
        return "stress" in self.implemented_properties

    def info(self) -> dict:
        """Return model metadata."""
        return {
            "model_type": "mace",
            "model_path": str(self.model_path),
            "device": self._device,
            "default_dtype": self._default_dtype,
            "task": self._task,
            "implemented_properties": self.implemented_properties,
            "has_stress": self.has_stress,
        }
