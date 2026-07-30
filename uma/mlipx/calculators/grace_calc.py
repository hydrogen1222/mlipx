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
            model_path: Path to an exported GRACE SavedModel directory.
            device: Device for calculation (``cpu`` or ``cuda``).
            task: PBC hint (``bulk`` or ``molecule``); not consumed by GRACE.
        """
        self.model_path = Path(model_path)
        self._device = device
        self._task = task
        self._calculator: Calculator | None = None

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        if not self.model_path.is_dir():
            raise ValueError(
                "GRACE model_path must be an exported SavedModel directory, "
                f"got: {self.model_path}"
            )
        dev = str(device).lower()
        if dev not in {"cpu", "cuda", "gpu"} and not (
            dev.startswith("cuda:") and dev[5:].isdigit()
        ):
            raise ValueError(
                f"Invalid GRACE device {device!r}. Use cpu, cuda, gpu, or cuda:N."
            )

    def get_calculator(self) -> Calculator:
        """Return the cached GRACE ASE calculator (lazy import)."""
        if self._calculator is None:
            self._apply_device_env()
            try:
                from tensorpotential.calculator import TPCalculator  # noqa: PLC0415
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "GRACE support requires the 'tensorpotential' package.\n"
                    "Install with: pip install tensorpotential"
                ) from e
            self._calculator = TPCalculator(model=str(self.model_path))
        return self._calculator

    def _apply_device_env(self) -> None:
        """Honour a requested device for GRACE/TensorFlow (plan section 7.5).

        ``TPCalculator`` has no ``device`` parameter; TensorFlow device
        placement is governed by ``CUDA_VISIBLE_DEVICES``, which must be set
        *before* TensorFlow is imported. mlipx imports tensorpotential lazily
        here, so setting the env var just above makes a ``cuda:N`` / ``cpu``
        request actually take effect. An explicit mlipx device selection takes
        precedence over an inherited environment value.
        """
        import os  # noqa: PLC0415

        dev = str(self._device).lower()
        if dev.startswith("cuda:") and dev != "cuda:":
            idx = dev.split(":", 1)[1]
            os.environ["CUDA_VISIBLE_DEVICES"] = idx
        elif dev == "cpu":
            # Hide GPUs so TensorFlow runs on CPU.
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

    def _actual_device(self) -> str:
        """Best-effort actual device, else 'unknown' (plan section 7.5).

        TensorFlow device placement is not reliably queryable from the ASE
        calculator, so we report ``'unknown'`` rather than guessing.
        """
        calc = self._calculator
        if calc is None:
            return "unknown"
        for attr in ("device", "_device"):
            d = getattr(calc, attr, None)
            if d:
                return str(d)
        return "unknown"

    @property
    def task(self) -> str:
        """PBC hint (``bulk``/``molecule``); GRACE itself is task-agnostic."""
        return self._task

    @property
    def has_stress(self) -> bool:
        """Whether stress is supported."""
        return "stress" in self.implemented_properties

    def info(self) -> dict:
        """Return model metadata.

        Distinguishes ``requested_device`` from ``actual_device`` (``'unknown'``
        when TensorFlow placement cannot be read). The legacy ``device`` key is
        kept (== requested) for backward-compatible output writers (plan 7.5).
        """
        return {
            "model_type": "grace",
            "model_path": str(self.model_path),
            "requested_device": self._device,
            "actual_device": self._actual_device(),
            "device": self._device,
            "task": self._task,
            "implemented_properties": self.implemented_properties,
            "has_stress": self.has_stress,
        }
