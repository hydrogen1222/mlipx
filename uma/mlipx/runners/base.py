# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Modified for the mlipx project: multi-engine MLIP support (UMA/MACE/DPA/GRACE).
"""
Base runner class for MLIP calculations.

Provides common functionality for all calculation runners,
including output directory management and logging.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING
from mlipx.protocols import ProgressEvent

if TYPE_CHECKING:
    from typing import Any

    from ase import Atoms
    from ase.calculators.calculator import Calculator

    from mlipx.base_calculator import BaseMLIPCalculator
    from mlipx.protocols import ProgressCallback


class BaseRunner(ABC):
    """Base class for all calculation runners.

    Provides common infrastructure for running calculations,
    managing output, and handling errors.

    Attributes:
        calculator: UMA calculator wrapper
        output_dir: Directory for output files
        verbose: Whether to print verbose output
        job_name: Optional job name for the calculation

    Example:
        >>> class MyRunner(BaseRunner):
        ...     def run(self, atoms):
        ...         # Implementation
        ...         pass
    """

    def __init__(
        self,
        calculator: BaseMLIPCalculator,
        output_dir: Path | str = ".",
        verbose: bool = True,
        job_name: str | None = None,
        log_fn: Any | None = None,
        progress_callback: ProgressCallback | None = None,
        cancel_event: threading.Event | None = None,
    ):
        """Initialize base runner.

        Args:
            calculator: UMA calculator wrapper
            output_dir: Directory for output files
            verbose: Whether to print progress messages
            job_name: Optional job name for organizing results
            log_fn: Optional callback function for custom log output
            progress_callback: Optional callback for progress events
        """
        self.calculator = calculator
        self.job_name = job_name
        self.verbose = verbose
        self.log_fn = log_fn
        self.progress_callback = progress_callback
        self._cancel_event = cancel_event  # threading.Event for cooperative cancellation

        # Build output directory path
        base_dir = Path(output_dir)
        if job_name:
            self.output_dir = base_dir / job_name
        else:
            self.output_dir = base_dir

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def log(self, message: str, level: str = "info") -> None:
        """Print log message if verbose mode or log callback is enabled.

        Args:
            message: Message to print
            level: Log level (info, warning, error)
        """
        if not self.verbose and not self.log_fn:
            return

        prefix = {
            "info": "  ",
            "warning": "! ",
            "error": "ERROR: ",
        }.get(level, "  ")

        formatted_msg = f"{prefix}{message}"

        if self.log_fn:
            self.log_fn(formatted_msg, level)
        if self.verbose:
            print(formatted_msg)

    def print_header(self, title: str) -> None:
        """Print section header.

        Args:
            title: Section title
        """
        if self.verbose or self.log_fn:
            msg = f"\n{'-' * 80}\n {title}\n{'-' * 80}"
            if self.log_fn:
                self.log_fn(msg, "info")
            if self.verbose:
                print(msg)

    def _emit_progress(
        self,
        phase: str,
        message: str,
        step: int | None = None,
        total_steps: int | None = None,
        extra: dict | None = None,
    ) -> None:
        """Emit a progress event to the callback if registered.

        Args:
            phase: Current phase (loading_model, running, writing_output, done, error)
            message: Human-readable description
            step: Current step number
            total_steps: Total expected steps
            extra: Optional phase-specific data
        """
        if self.progress_callback is None:
            return
        event = ProgressEvent(
            phase=phase,
            message=message,
            step=step,
            total_steps=total_steps,
            extra=extra,
        )
        self.progress_callback(event)

    def _is_cancelled(self) -> bool:
        """Check whether cooperative cancellation has been requested.

        Returns True if a cancel_event was provided and has been set,
        indicating the caller wants the calculation to stop early.
        """
        return self._cancel_event is not None and self._cancel_event.is_set()

    @abstractmethod
    def run(self, atoms: Atoms) -> dict[str, Any]:
        """Run calculation on structure.

        Args:
            atoms: ASE Atoms object

        Returns:
            Dictionary with calculation results
        """

    def _prepare_atoms(self, atoms: Atoms) -> Atoms:
        """Prepare atoms for calculation.

        Handles task-specific preparation like setting charge/spin for molecules
        and ensuring PBC is correctly set for periodic systems.

        Args:
            atoms: Input ASE Atoms object

        Returns:
            Prepared Atoms object
        """
        task = self.calculator.task

        # Periodic vs molecular task classification. UMA tasks (omat/oc20/...)
        # and the generic 'bulk' are periodic; 'omol'/'molecule' are not.
        PERIODIC_TASKS = {"omat", "oc20", "oc25", "odac", "omc", "bulk"}
        MOLECULAR_TASKS = {"omol", "molecule"}

        # Ensure PBC is set correctly for periodic systems
        if task in PERIODIC_TASKS:
            if not atoms.pbc.any():
                self.log("Setting PBC=True for periodic system", level="warning")
                atoms.pbc = True
            # Log cell info for debugging
            cell = atoms.cell
            if cell.volume > 0:
                self.log(
                    f"Cell: {cell.lengths()[0]:.4f} x {cell.lengths()[1]:.4f} x {cell.lengths()[2]:.4f} Å"
                )
            else:
                raise ValueError("Invalid cell: zero volume. Check input structure.")
        elif task in MOLECULAR_TASKS:
            # Molecules should not have PBC
            atoms.pbc = False
            if "charge" not in atoms.info:
                self.log("Setting default charge=0 for molecule", level="warning")
                atoms.info["charge"] = 0
            if "spin" not in atoms.info:
                self.log("Setting default spin=1 for molecule", level="warning")
                atoms.info["spin"] = 1
        else:
            # Unknown task type: preserve the input structure's PBC settings
            self.log(
                f"Unknown task '{task}', preserving input PBC settings", level="warning"
            )

        return atoms

    def _get_calculator(self) -> Calculator:
        """Get ASE calculator instance.

        Returns:
            ASE Calculator
        """
        return self.calculator.get_calculator()

    def _write_summary(self, results: dict[str, Any], atoms: Atoms) -> None:
        """Write calculation summary to stdout.

        Args:
            results: Results dictionary
            atoms: ASE Atoms object
        """
        if not self.verbose:
            return

        energy = results.get("energy")
        forces = results.get("forces")

        print()
        print("=" * 80)
        if self.job_name:
            print(f" SUMMARY - {self.job_name}")
        else:
            print(" SUMMARY")
        print("=" * 80)

        if energy is not None:
            print(f"Total energy:     {energy:16.8f} eV")
            print(f"Energy per atom:  {energy / len(atoms):16.8f} eV/atom")

        if forces is not None:
            import numpy as np

            force_mags = np.linalg.norm(forces, axis=1)
            print(f"Max force:        {np.max(force_mags):16.8f} eV/Å")
            print(f"RMS force:        {np.sqrt(np.mean(force_mags**2)):16.8f} eV/Å")

        if "stress" in results and results["stress"] is not None:
            import numpy as np

            stress = results["stress"]
            pressure = -(stress[0] + stress[1] + stress[2]) / 3.0 * 160.2177
            print(f"Pressure:         {pressure:16.8f} GPa")

        calc_time = results.get("time")
        if calc_time is not None:
            print(f"Calculation time: {calc_time:16.2f} s")

        print("=" * 80)
