"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Run screen for executing calculations with live output.
"""

from __future__ import annotations

import asyncio
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

from ase.io import read
from textual.containers import Container, Horizontal
from textual.screen import Screen
from textual.widgets import Button, Log, ProgressBar, Static

from umakit.engine import CalculationEngine, EngineConfig

if TYPE_CHECKING:
    from textual.app import ComposeResult


class RunScreen(Screen):
    """Screen for running calculations with live output."""

    _task: asyncio.Task | None = None

    def compose(self) -> ComposeResult:
        """Compose the run screen."""
        calc_type = self.app.get_config("calc_type", "sp")
        structure = self.app.get_config("structure_file", "Not set")

        yield Container(
            Static(f"Running: {calc_type.upper()}", id="title"),
            Static(f"Structure: {structure}", id="subtitle"),
            # Progress bar
            Static("Progress:"),
            ProgressBar(total=100, id="progress-bar"),
            Static("Initializing...", id="status-text"),
            # Log output
            Static("Log Output:", classes="section-header"),
            Log(id="run-log"),
            # Action buttons
            Horizontal(
                Button("◀ Cancel", variant="error", id="cancel-btn"),
                Button("🔄 Back", id="back-btn", disabled=True),
                id="action-buttons",
            ),
            id="run-container",
        )

    def on_mount(self) -> None:
        """Start calculation when screen mounts."""
        self.log_widget = self.query_one("#run-log", Log)
        self.progress = self.query_one("#progress-bar", ProgressBar)
        self.status = self.query_one("#status-text", Static)
        self._log_file = self._open_log_file()
        self._task = asyncio.create_task(self._run_calculation())

    def on_unmount(self) -> None:
        """Cancel any running calculation task when the screen is removed."""
        if self._task is not None and not self._task.done():
            self._task.cancel()

    def _open_log_file(self) -> Path | None:
        """Open a plain-text log file alongside the run for easy copying.

        The on-screen ``Log`` widget is hard to select/copy from; mirror every
        line to a ``run.log`` under the output dir so it can be grep/cat'd.
        """
        try:
            out_dir = Path(self.app.get_config("output_dir", "./results"))
            out_dir.mkdir(parents=True, exist_ok=True)
            log_path = out_dir / "run.log"
            # Truncate at the start of each run.
            log_path.write_text("", encoding="utf-8")
            return log_path
        except Exception:
            return None

    def _log(self, message: str) -> None:
        """Add message to log (on-screen + mirrored to file)."""
        if self.is_mounted and self.log_widget.is_mounted:
            self.log_widget.write_line(message)
        if self._log_file is not None:
            try:
                with open(self._log_file, "a", encoding="utf-8") as f:
                    f.write(message + "\n")
            except Exception:
                pass

    def _update_progress(self, value: float, status: str = "") -> None:
        """Update progress bar (determinate mode, 0-100).

        Restore a concrete total so percentage = value/100; total is set to
        None during indeterminate phases.
        """
        if self.is_mounted and self.progress.is_mounted:
            self.progress.update(total=100, progress=value)
        if status and self.is_mounted and self.status.is_mounted:
            self.status.update(status)

    def _update_indeterminate(self, status: str) -> None:
        """Update progress bar to indeterminate mode.

        In textual, indeterminate is signaled by ``total=None`` (NOT
        ``progress=None`` — that crashes ``_compute_percentage`` with
        ``None / total``).
        """
        if self.is_mounted and self.progress.is_mounted:
            self.progress.update(total=None, progress=0)
        if self.is_mounted and self.status.is_mounted:
            self.status.update(status)

    def _get_engine_config(self) -> EngineConfig:
        """Build EngineConfig from app state."""
        calc_type = self.app.get_config("calc_type")
        options = {}
        if calc_type == "opt":
            options.update(
                {
                    "fmax": self.app.get_config("fmax", 0.05),
                    "max_steps": self.app.get_config("max_steps", 500),
                    "optimizer": self.app.get_config("optimizer", "FIRE"),
                    "cell_opt": self.app.get_config("cell_opt", False),
                    "fix_symmetry": self.app.get_config("fix_symmetry", False),
                }
            )
        elif calc_type == "md":
            options.update(
                {
                    "ensemble": self.app.get_config("ensemble", "NVT"),
                    "temperature": self.app.get_config("temperature", 300.0),
                    "timestep": self.app.get_config("timestep", 1.0),
                    "steps": self.app.get_config("md_steps", 1000),
                    "save_interval": self.app.get_config("save_interval", 10),
                    "pre_relax": self.app.get_config("pre_relax", True),
                    "pre_relax_steps": self.app.get_config("pre_relax_steps", 50),
                    "pre_relax_fmax": self.app.get_config("pre_relax_fmax", 0.1),
                }
            )

        return EngineConfig(
            calc_type=calc_type,
            model_path=Path(self.app.get_config("model_file", "")),
            task=self.app.get_config("task", "omat"),
            device=self.app.get_config("device", "cpu"),
            output_dir=Path(self.app.get_config("output_dir", "./results")),
            job_name=self.app.get_config("job_name"),
            options=options,
            detach=self.app.get_config("detach", False),
        )

    async def _run_calculation(self) -> None:
        """Run the calculation asynchronously with progress events."""
        try:
            config = self._get_engine_config()

            self._log("Loading structure...")
            structure_file = self.app.get_config("structure_file")
            atoms = read(structure_file)
            self._log(f"Loaded: {atoms.get_chemical_formula()} ({len(atoms)} atoms)")

            engine = CalculationEngine.from_config(config)
            self._task = asyncio.current_task()

            async for event in engine.run_async(atoms):
                if event.phase == "loading_model":
                    self._update_indeterminate(event.message)
                elif event.phase == "running":
                    if event.total_steps is not None:
                        pct = (
                            (event.step / event.total_steps) * 100 if event.step else 0
                        )
                        self._update_progress(pct, event.message)
                    else:
                        self._update_indeterminate(event.message)
                elif event.phase == "writing_output":
                    self._update_indeterminate(event.message)
                elif event.phase == "done":
                    self._update_progress(100, "Complete")
                    self._log("\nCalculation complete!")
                    if event.extra and "energy" in event.extra:
                        self._log(f"Energy: {event.extra['energy']:.6f} eV")
                    self._log(f"Output: {self.app.get_config('output_dir')}")
                elif event.phase == "error":
                    self._log(f"\nERROR: {event.message}")
                    self._update_progress(0, "Failed")

        except asyncio.CancelledError:
            self._log("\nCalculation cancelled by user")
            self._update_progress(0, "Cancelled")
        except Exception as e:
            self._log(f"\nERROR: {e}")
            self._log(traceback.format_exc())
            self._update_progress(0, "Failed")
        finally:
            if self.is_mounted:
                back_btn = self.query_one("#back-btn", Button)
                back_btn.disabled = False

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id

        if button_id == "cancel-btn":
            if self._task and not self._task.done():
                self._task.cancel()
            self._log("\nCancel requested (finishing current step)...")

        elif button_id == "back-btn":
            self.app.pop_screen()
