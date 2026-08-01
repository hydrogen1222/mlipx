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
        head: str | None = None,
    ):
        """
        Initialize DPA calculator wrapper.

        Args:
            model_path: Path to a DPA model (``.pth``/``.pt`` for PyTorch
                backend, or ``.pb`` for the legacy TensorFlow backend).
            device: Device for calculation (``cpu`` or ``cuda``).
            task: PBC hint (``bulk`` or ``molecule``); not consumed by DPA.
            head: Optional branch/head for a multi-task DeepMD model.
        """
        self.model_path = Path(model_path)
        self._device = device
        self._task = task
        self._head = head
        self._calculator: Calculator | None = None

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        dev = str(device).lower()
        if dev not in {"cpu", "cuda", "gpu"} and not (
            dev.startswith("cuda:") and dev[5:].isdigit()
        ):
            raise ValueError(
                f"Invalid DPA device {device!r}. Use cpu, cuda, gpu, or cuda:N."
            )

    def get_calculator(self) -> Calculator:
        """Return the cached DPA ASE calculator (lazy import)."""
        if self._calculator is None:
            self._apply_device_env()
            self._preload_packaged_cuda_runtime()
            try:
                from deepmd.calculator import DP  # noqa: PLC0415
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "DPA support requires the 'deepmd-kit' package.\n"
                    "Install with: pip install deepmd-kit>=3.0.0"
                ) from e
            kwargs = {
                "model": str(self.model_path),
                "type_dict": None,
            }
            if self._head is not None:
                kwargs["head"] = self._head
            self._calculator = DP(**kwargs)
        return self._calculator

    def _apply_device_env(self) -> None:
        """Honour a requested device for DeepMD (plan section 6.2).

        The ASE ``DP`` calculator has no ``device`` parameter; DeepMD places
        the model through ``CUDA_VISIBLE_DEVICES`` / ``deepmd.env.DEVICE``.
        Setting the env var *before* deepmd is imported (mlipx imports it
        lazily above) makes a ``cuda:N`` (or ``cpu``) request actually take
        effect. An explicit mlipx device selection takes precedence over an
        inherited environment value.
        """
        import os  # noqa: PLC0415

        dev = str(self._device).lower()
        if dev.startswith("cuda:") and dev != "cuda:":
            idx = dev.split(":", 1)[1]
            os.environ["CUDA_VISIBLE_DEVICES"] = idx
        elif dev == "cpu":
            # Hide GPUs so the DeepMD backend falls back to CPU.
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

    @staticmethod
    def _preload_packaged_cuda_runtime() -> None:
        """Preload pip's optional CUDA runtime before importing DeepMD.

        The DeepMD wheel probes ``libcudart.so.12`` even for a CPU inference
        run. When ``nvidia-cuda-runtime-cu12`` is installed inside a venv its
        library directory is not necessarily on the dynamic-loader search
        path, producing a long but non-fatal error banner. Loading that exact
        packaged library globally avoids the false alarm. Absence of the
        optional package remains harmless.
        """
        try:
            import ctypes  # noqa: PLC0415
            from pathlib import Path  # noqa: PLC0415

            import nvidia.cuda_runtime  # type: ignore[import-not-found]  # noqa: PLC0415

            for root in nvidia.cuda_runtime.__path__:
                candidate = Path(root) / "lib" / "libcudart.so.12"
                if candidate.is_file():
                    ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
                    break
        except (ImportError, OSError):
            pass

    def _actual_device(self) -> str:
        """Best-effort actual device from the loaded DeepMD model, else 'unknown'.

        Per plan section 6.2 we never *guess* the device: if it cannot be read
        reliably from the backend we report ``'unknown'`` rather than echoing
        the requested value.
        """
        calc = self._calculator
        if calc is None:
            return "unknown"
        dp = getattr(calc, "dp", None)
        for attr in ("device", "_device"):
            d = getattr(dp, attr, None)
            if d:
                return str(d)
        return "unknown"

    @property
    def task(self) -> str:
        """PBC hint (``bulk``/``molecule``); DPA itself is task-agnostic."""
        return self._task

    @property
    def has_stress(self) -> bool:
        """Whether stress is supported."""
        return "stress" in self.implemented_properties

    def info(self) -> dict:
        """Return model metadata.

        Distinguishes ``requested_device`` (what the user asked for) from
        ``actual_device`` (what the backend reports, or ``'unknown'``). The
        legacy ``device`` key is kept (== requested) for backward-compatible
        output writers (plan section 6.2).
        """
        return {
            "model_type": "dpa",
            "model_path": str(self.model_path),
            "requested_device": self._device,
            "actual_device": self._actual_device(),
            "device": self._device,
            "task": self._task,
            "head": self._head,
            "implemented_properties": self.implemented_properties,
            "has_stress": self.has_stress,
        }
