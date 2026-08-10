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

Note: GRACE uses a TensorFlow/XLA backend. mlipx configures TensorFlow before
the first graph is built so it does not reserve the whole visible GPU by
default. A hard per-process limit can additionally be supplied when a GPU is
shared with another calculation.
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
        cpu_threads: int | None = None,
        gpu_memory_growth: bool = True,
        gpu_memory_limit_mb: int | None = None,
    ):
        """
        Initialize GRACE calculator wrapper.

        Args:
            model_path: Path to an exported GRACE SavedModel directory.
            device: Device for calculation (``cpu`` or ``cuda``).
            task: PBC hint (``bulk`` or ``molecule``); not consumed by GRACE.
            cpu_threads: TensorFlow intra-op CPU thread count. ``None`` keeps
                TensorFlow's default.
            gpu_memory_growth: Let TensorFlow grow its allocator on demand
                instead of reserving all visible GPU memory at startup.
            gpu_memory_limit_mb: Optional hard TensorFlow logical-device limit
                in MiB. When set it takes precedence over memory growth.
        """
        self.model_path = Path(model_path)
        self._device = device
        self._task = task
        self._cpu_threads = cpu_threads
        self._gpu_memory_growth = gpu_memory_growth
        self._gpu_memory_limit_mb = gpu_memory_limit_mb
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
        if cpu_threads is not None and (
            isinstance(cpu_threads, bool)
            or not isinstance(cpu_threads, int)
            or cpu_threads < 1
        ):
            raise ValueError("GRACE cpu_threads must be a positive integer.")
        if not isinstance(gpu_memory_growth, bool):
            raise ValueError("GRACE gpu_memory_growth must be a boolean.")
        if gpu_memory_limit_mb is not None and (
            isinstance(gpu_memory_limit_mb, bool)
            or not isinstance(gpu_memory_limit_mb, int)
            or gpu_memory_limit_mb < 1
        ):
            raise ValueError("GRACE gpu_memory_limit_mb must be a positive integer.")
        if dev == "cpu" and gpu_memory_limit_mb is not None:
            raise ValueError(
                "GRACE gpu_memory_limit_mb applies only to a CUDA/GPU device."
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
            tf = None
            if self._cpu_threads is not None or self._uses_gpu:
                # Import tensorpotential first: its package initializer must set
                # TF_USE_LEGACY_KERAS before TensorFlow is imported. Threading
                # and GPU allocation are still configured before TPCalculator
                # builds/executes a TensorFlow graph.
                try:
                    import tensorflow as tf  # noqa: PLC0415
                except ImportError as e:  # pragma: no cover
                    raise ImportError(
                        "GRACE support requires TensorFlow via tensorpotential."
                    ) from e
            if self._cpu_threads is not None:
                try:
                    tf.config.threading.set_intra_op_parallelism_threads(
                        self._cpu_threads
                    )
                except RuntimeError as e:
                    raise RuntimeError(
                        "Could not set GRACE CPU threads because TensorFlow was "
                        "already initialized. Start mlipx in a fresh process or "
                        "omit --cpu-threads."
                    ) from e
            if self._uses_gpu:
                assert tf is not None
                self._configure_tensorflow_gpu(tf)
            # UQ-capable GRACE exports otherwise select the full UQ signature,
            # including dsigma/dr, for an ordinary ASE energy/force request.
            # mlipx does not expose those UQ tensors, so retaining them wastes
            # substantial host/GPU memory without changing the requested
            # physical observables.
            self._calculator = TPCalculator(
                model=str(self.model_path),
                enable_uq_if_available=False,
            )
        return self._calculator

    @property
    def _uses_gpu(self) -> bool:
        return str(self._device).lower() != "cpu"

    def _configure_tensorflow_gpu(self, tf) -> None:
        """Apply a bounded TensorFlow GPU policy before runtime initialisation."""
        physical_gpus = list(tf.config.list_physical_devices("GPU"))
        if not physical_gpus:
            raise RuntimeError(
                f"GRACE device {self._device!r} requested a GPU, but TensorFlow "
                "reports no visible physical GPU."
            )
        try:
            if self._gpu_memory_limit_mb is not None:
                logical_config = tf.config.LogicalDeviceConfiguration(
                    memory_limit=self._gpu_memory_limit_mb
                )
                for gpu in physical_gpus:
                    tf.config.set_logical_device_configuration(
                        gpu, [logical_config]
                    )
            elif self._gpu_memory_growth:
                for gpu in physical_gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            policy = (
                f"a {self._gpu_memory_limit_mb} MiB hard limit"
                if self._gpu_memory_limit_mb is not None
                else "memory growth"
            )
            raise RuntimeError(
                f"Could not configure GRACE TensorFlow GPU {policy} because "
                "the TensorFlow runtime was already initialized. Start mlipx "
                "in a fresh process; refusing to run with an unbounded policy."
            ) from e

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
        if dev == "cpu":
            return
        if self._gpu_memory_limit_mb is not None:
            # A logical-device cap and TensorFlow memory growth are mutually
            # exclusive. Set this before importing TensorFlow.
            os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"
        elif self._gpu_memory_growth:
            # This environment-level guard is read by TensorFlow's BFC
            # allocator and complements the explicit config call below.
            os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

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
            "cpu_threads": self._cpu_threads,
            "gpu_memory_growth": (
                self._gpu_memory_growth
                if self._gpu_memory_limit_mb is None
                else False
            ),
            "gpu_memory_limit_mb": self._gpu_memory_limit_mb,
            "uq_enabled": False,
            "implemented_properties": self.implemented_properties,
            "has_stress": self.has_stress,
        }
