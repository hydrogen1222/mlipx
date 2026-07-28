# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Modified for the mlipx project: multi-engine CalculatorFactory.

"""
Calculator factory.

Selects the right MLIP engine wrapper from a ``model_type`` string. Each
wrapper is imported lazily so only the chosen engine's backend is loaded.
Adding a new engine only requires one new wrapper module + one ``elif``
branch here; Runners/Writers/CLI/TUI stay untouched.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from mlipx.base_calculator import BaseMLIPCalculator

if TYPE_CHECKING:
    from typing import Any


# Engine type aliases. ``fairchem`` is accepted as a synonym for ``uma``.
UMA_ALIASES = {"uma", "fairchem"}
SUPPORTED_TYPES = {"uma", "fairchem", "mace", "dpa", "grace"}


class CalculatorFactory:
    """Create the appropriate MLIP Calculator wrapper from a model type."""

    @staticmethod
    def create(
        model_type: str,
        model_path: str | Path,
        device: str = "cpu",
        task: str = "bulk",
        **kwargs: Any,
    ) -> BaseMLIPCalculator:
        """
        Create a calculator wrapper for the requested engine.

        Args:
            model_type: Engine name (``uma``/``fairchem``, ``mace``,
                ``dpa``, ``grace``).
            model_path: Path to the model checkpoint/file.
            device: Device for calculation (``cpu`` or ``cuda``).
            task: Task type. UMA uses ``omat``/``omol``/...; other engines use
                ``bulk``/``molecule`` (PBC hint only).
            **kwargs: Engine-specific options. UMA accepts ``inference_mode``,
                ``torch_num_threads`` and ``activation_checkpointing``; MACE
                accepts ``default_dtype``.

        Returns:
            A ``BaseMLIPCalculator`` subclass instance.

        Raises:
            ValueError: If ``model_type`` is not supported.
        """
        m_type = (model_type or "uma").lower()

        if m_type in UMA_ALIASES:
            from mlipx.calculator import UMACalculator  # noqa: PLC0415

            return UMACalculator(
                model_path=model_path,
                task=task,
                device=device,
                inference_mode=kwargs.get("inference_mode", "default"),
                torch_num_threads=kwargs.get("torch_num_threads"),
                activation_checkpointing=kwargs.get("activation_checkpointing"),
            )
        elif m_type == "mace":
            from mlipx.calculators.mace_calc import MACECalculatorWrapper  # noqa: PLC0415

            return MACECalculatorWrapper(
                model_path=model_path,
                device=device,
                task=task,
                default_dtype=kwargs.get("default_dtype", "float64"),
            )
        elif m_type == "dpa":
            from mlipx.calculators.dpa_calc import DPACalculatorWrapper  # noqa: PLC0415

            return DPACalculatorWrapper(
                model_path=model_path,
                device=device,
                task=task,
            )
        elif m_type == "grace":
            from mlipx.calculators.grace_calc import GRACECalculatorWrapper  # noqa: PLC0415

            return GRACECalculatorWrapper(
                model_path=model_path,
                device=device,
                task=task,
            )
        else:
            raise ValueError(
                f"Unsupported model type: '{model_type}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_TYPES - {'fairchem'}))}"
            )
