"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Modified for the mlipx project: multi-engine MLIP support (UMA/MACE/DPA/GRACE).
Unified execution engine for MLIP calculations.

Provides CalculationEngine as the single entry point for CLI, TUI, and API.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from mlipx.base_calculator import BaseMLIPCalculator
from mlipx.protocols import CancellationRequested, ProgressCallback, ProgressEvent

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from typing import Any, Literal

    from ase import Atoms


@dataclass
class EngineConfig:
    """Unified configuration for all calculation types and interfaces.

    Fields:
        calc_type: sp, opt, md, or batch.
        model_path: Path to model checkpoint/file.
        model_type: MLIP engine (uma, mace, dpa, grace).
        task: Task type. UMA: omat/omol/...; others: bulk/molecule.
        device: cpu or cuda.
        inference_mode: default or turbo (UMA only).
        output_dir: Directory for output files.
        job_name: Optional job name.
        options: Calc-type-specific parameters (fmax, temperature, etc.).
        torch_num_threads: CPU thread count for torch.
        activation_checkpointing: GPU memory saving (overrides inference_mode preset).
        detach: If True, submit as background job.
    """

    calc_type: Literal["sp", "opt", "md", "batch"]
    model_path: Path
    model_type: str = "uma"
    task: str = "omat"
    device: str = "cpu"
    inference_mode: str = "default"
    output_dir: Path = field(default_factory=lambda: Path("./results"))
    job_name: str | None = None
    options: dict = field(default_factory=dict)
    torch_num_threads: int | None = None
    activation_checkpointing: bool | None = None
    detach: bool = False


class CalculationEngine:
    """Unified execution engine for MLIP calculations.

    Use CalculationEngine.from_config() to create an instance, then
    call run(), run_async(), or run_batch().
    """

    VALID_CALC_TYPES: ClassVar[set[str]] = {"sp", "opt", "md", "batch"}

    def __init__(self, config: EngineConfig):
        self.config = config
        self._validate()

    @classmethod
    def from_config(cls, config: EngineConfig) -> CalculationEngine:
        return cls(config)

    def _validate(self) -> None:
        if self.config.calc_type not in self.VALID_CALC_TYPES:
            raise ValueError(
                f"Unknown calc_type '{self.config.calc_type}'. "
                f"Must be one of: {', '.join(self.VALID_CALC_TYPES)}"
            )
        # Warn about unknown option keys
        known_sp = set()
        known_opt = {"fmax", "max_steps", "optimizer", "cell_opt", "fix_symmetry"}
        known_md = {
            "ensemble",
            "temperature",
            "timestep",
            "steps",
            "friction",
            "save_interval",
            "pre_relax",
            "pre_relax_steps",
            "pre_relax_fmax",
        }
        known_batch = {"pattern", "sub_calc_type", "parallel", "max_workers"}
        known_all = known_sp | known_opt | known_md | known_batch
        unknown = set(self.config.options.keys()) - known_all
        for key in unknown:
            warnings.warn(
                f"Unknown option '{key}' for calc_type '{self.config.calc_type}'"
            )

    def _create_calculator(self) -> BaseMLIPCalculator:
        from mlipx.calculators.factory import CalculatorFactory  # noqa: PLC0415

        return CalculatorFactory.create(
            model_type=self.config.model_type,
            model_path=self.config.model_path,
            device=self.config.device,
            task=self.config.task,
            inference_mode=self.config.inference_mode,
            torch_num_threads=self.config.torch_num_threads,
            activation_checkpointing=self.config.activation_checkpointing,
        )

    def _create_runner(self, calculator, progress_callback=None, log_fn=None, cancel_event=None):
        from mlipx.runners.md import MDRunner  # noqa: PLC0415
        from mlipx.runners.optimization import OptimizationRunner  # noqa: PLC0415
        from mlipx.runners.singlepoint import SinglePointRunner  # noqa: PLC0415

        opts = self.config.options
        common = dict(
            calculator=calculator,
            output_dir=self.config.output_dir,
            verbose=False,
            job_name=self.config.job_name,
            log_fn=log_fn,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )

        if self.config.calc_type == "sp":
            return SinglePointRunner(**common)
        elif self.config.calc_type == "opt":
            return OptimizationRunner(
                fmax=opts.get("fmax", 0.05),
                max_steps=opts.get("max_steps", 500),
                optimizer=opts.get("optimizer", "FIRE"),
                cell_opt=opts.get("cell_opt", False),
                fix_symmetry=opts.get("fix_symmetry", False),
                **common,
            )
        elif self.config.calc_type == "md":
            return MDRunner(
                ensemble=opts.get("ensemble", "NVT"),
                temperature=opts.get("temperature", 300.0),
                timestep=opts.get("timestep", 1.0),
                steps=opts.get("steps", 1000),
                friction=opts.get("friction", 0.001),
                save_interval=opts.get("save_interval", 10),
                pre_relax=opts.get("pre_relax", True),
                pre_relax_steps=opts.get("pre_relax_steps", 50),
                pre_relax_fmax=opts.get("pre_relax_fmax", 0.1),
                **common,
            )
        else:
            raise ValueError(f"Unknown calc_type: {self.config.calc_type}")

    def run(
        self,
        atoms: Atoms,
        progress_callback: ProgressCallback | None = None,
        cancel_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        """Run calculation synchronously.

        Args:
            atoms: ASE Atoms object.
            progress_callback: Optional callback for progress events.
            cancel_event: Optional threading.Event for cooperative cancellation.

        Returns:
            Results dictionary.
        """
        calculator = self._create_calculator()
        runner = self._create_runner(
            calculator,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )
        return runner.run(atoms)
    async def run_async(
        self,
        atoms: Atoms,
    ) -> AsyncIterator[ProgressEvent]:
        """Run calculation asynchronously, yielding progress events.

        The actual computation runs in a thread pool; progress events
        are bridged back to the asyncio loop via a queue.

        Cooperative cancellation: when the async generator is cancelled,
        a threading.Event is set so the worker thread can stop early.
        The finally block waits for the thread with a timeout to avoid
        blocking indefinitely on long-running simulations.

        Args:
            atoms: ASE Atoms object.

        Yields:
            ProgressEvent at each phase transition.
        """
        queue: asyncio.Queue[ProgressEvent | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()
        cancel_event = threading.Event()

        def progress_callback(event: ProgressEvent) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, event)

        def blocking_work() -> dict[str, Any]:
            try:
                return self.run(
                    atoms,
                    progress_callback=progress_callback,
                    cancel_event=cancel_event,
                )
            except CancellationRequested as exc:
                # Normal cancellation — emit a "cancelled" event, not "error"
                event = ProgressEvent(
                    phase="cancelled",
                    message=str(exc),
                )
                loop.call_soon_threadsafe(queue.put_nowait, event)
                raise
            except Exception as exc:
                event = ProgressEvent(
                    phase="error",
                    message=f"Calculation failed: {exc}",
                )
                loop.call_soon_threadsafe(queue.put_nowait, event)
                raise

        task = loop.run_in_executor(None, blocking_work)
        _cancelled = False

        try:
            while True:
                event = await queue.get()
                yield event
                if event.phase in ("done", "error", "cancelled"):
                    break
        except asyncio.CancelledError:
            _cancelled = True
            # Signal the worker thread to stop cooperatively
            cancel_event.set()
            # Attempt to cancel the future (best-effort; won't stop a running thread)
            task.cancel()
            raise  # Re-raise so the caller's CancelledError handler runs
        finally:
            if _cancelled:
                # Wait for the worker thread with a generous timeout.
                # The cancel_event will cause runners to bail out at their next
                # check-point (every MD step or optimization step).
                with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError, Exception):
                    await asyncio.wait_for(task, timeout=30.0)
            else:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
    def run_batch(
        self,
        files: list[Path],
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        """Run batch calculation on multiple structure files.

        Args:
            files: List of structure file paths.
            progress_callback: Optional callback for progress events.

        Returns:
            Batch summary dictionary.
        """
        from mlipx.runners.batch import BatchRunner  # noqa: PLC0415

        calculator = self._create_calculator()
        opts = self.config.options
        runner = BatchRunner(
            calculator,
            calc_type=opts.get("sub_calc_type", "sp"),
            output_dir=self.config.output_dir,
            parallel=opts.get("parallel", False),
            max_workers=opts.get("max_workers", 1),
            verbose=False,
            job_name=self.config.job_name,
            progress_callback=progress_callback,
        )
        return runner.run_from_files(files)
