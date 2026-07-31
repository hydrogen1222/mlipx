"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Run screen for persistent background calculations with live output.
"""

from __future__ import annotations

import shlex
import sys
from datetime import datetime
from typing import TYPE_CHECKING

from ase.io import read
from textual.containers import Container, Horizontal
from textual.screen import Screen
from textual.timer import Timer
from textual.widgets import Button, Log, ProgressBar, Static

from mlipx.jobs import JobManager

if TYPE_CHECKING:
    from textual.app import ComposeResult


class RunScreen(Screen):
    """Launch and monitor a calculation that survives screen/TUI exit."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._job_manager = JobManager()
        self._job_id: str | None = None
        self._refresh_timer: Timer | None = None
        self._displayed_log = ""

    def compose(self) -> ComposeResult:
        calc_type = self.app.get_config("calc_type", "sp")
        structure = self.app.get_config("structure_file", "Not set")
        yield Container(
            Static(f"Running: {calc_type.upper()}", id="title"),
            Static(f"Structure: {structure}", id="subtitle"),
            Static("Progress:"),
            ProgressBar(total=None, id="progress-bar"),
            Static("Starting background job...", id="status-text"),
            Static("Log Output:", classes="section-header"),
            Log(id="run-log"),
            Horizontal(
                Button("Cancel", variant="error", id="cancel-btn"),
                Button("◀ Back (Keep Running)", id="back-btn"),
                id="action-buttons",
            ),
            id="run-container",
        )

    def on_mount(self) -> None:
        self.log_widget = self.query_one("#run-log", Log)
        self.progress = self.query_one("#progress-bar", ProgressBar)
        self.status = self.query_one("#status-text", Static)
        try:
            atoms = read(self.app.get_config("structure_file"))
            self._job_id = self._make_job_id()
            command = self._build_command()
            self._job_manager.submit(
                job_id=self._job_id,
                calc_type=self.app.get_config("calc_type", "sp"),
                structure=self.app.get_config("structure_file"),
                formula=atoms.get_chemical_formula(),
                natoms=len(atoms),
                device=self.app.get_config("device", "cpu"),
                cmd=command,
            )
            log_path = self._job_manager._log_file(self._job_id).resolve()
            self.log_widget.write_line(f"Background job: {self._job_id}")
            self.log_widget.write_line(f"Live log: {log_path}")
            self.log_widget.write_line(
                f"Follow live output: tail -f {shlex.quote(str(log_path))}"
            )
            self.status.update("Running in background — safe to go Back or exit TUI")
            self._refresh_timer = self.set_interval(0.5, self._refresh_job)
        except Exception as exc:
            self.log_widget.write_line(f"ERROR: Could not start background job: {exc}")
            self.status.update("Failed to start")
            self.progress.update(total=100, progress=0)
            self.query_one("#cancel-btn", Button).disabled = True

    def on_unmount(self) -> None:
        """Stop only the UI refresh; the background process keeps running."""
        if self._refresh_timer is not None:
            self._refresh_timer.stop()
            self._refresh_timer = None

    def _make_job_id(self) -> str:
        configured = self.app.get_config("job_name")
        base = configured or (
            f"{self.app.get_config('calc_type', 'job')}-"
            f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        job_id = str(base).replace("/", "_").replace("\\", "_")
        candidate = job_id
        suffix = 2
        while self._job_manager.get_job(candidate) is not None:
            candidate = f"{job_id}-{suffix}"
            suffix += 1
        if configured is None:
            self.app.update_config("job_name", candidate)
        return candidate

    def _build_command(self) -> list[str]:
        calc_type = self.app.get_config("calc_type", "sp")
        model_type = self.app.get_config("model_type", "uma")
        command = [
            sys.executable,
            "-m",
            "mlipx.cli",
            calc_type,
            self.app.get_config("structure_file"),
            "--model",
            self.app.get_config("model_file"),
            "--model-type",
            model_type,
            "--task",
            self.app.get_config("task", "omat"),
            "--device",
            self.app.get_config("device", "cpu"),
            "--output",
            self.app.get_config("output_dir", "./results"),
            "--name",
            self._job_id,
        ]
        if model_type == "uma":
            command.extend(
                [
                    "--inference-mode",
                    self.app.get_config("inference_mode", "default"),
                ]
            )
            checkpointing = self.app.get_config("activation_checkpointing")
            if checkpointing is not None:
                command.append(
                    "--activation-checkpointing"
                    if checkpointing
                    else "--no-activation-checkpointing"
                )
        threads = self.app.get_config("torch_num_threads")
        if threads is not None:
            command.extend(["--cpu-threads", str(threads)])
        if model_type == "mace":
            command.extend(
                ["--dtype", self.app.get_config("default_dtype", "float32")]
            )
        if model_type in {"mace", "dpa"}:
            head = self.app.get_config("head")
            if head:
                command.extend(["--head", str(head)])

        if calc_type == "opt":
            command.extend(
                [
                    "--fmax",
                    str(self.app.get_config("fmax", 0.05)),
                    "--max-steps",
                    str(self.app.get_config("max_steps", 500)),
                    "--optimizer",
                    self.app.get_config("optimizer", "FIRE"),
                ]
            )
            if self.app.get_config("cell_opt", False):
                command.append("--cell-opt")
            if self.app.get_config("fix_symmetry", False):
                command.append("--fix-symmetry")
        elif calc_type == "md":
            command.extend(
                [
                    "--ensemble",
                    self.app.get_config("ensemble", "NVT"),
                    "--temp",
                    str(self.app.get_config("temperature", 300.0)),
                    "--timestep",
                    str(self.app.get_config("timestep", 1.0)),
                    "--steps",
                    str(self.app.get_config("md_steps", 1000)),
                    "--friction",
                    str(self.app.get_config("friction", 0.001)),
                    "--save-interval",
                    str(self.app.get_config("save_interval", 10)),
                    "--pre-relax-steps",
                    str(self.app.get_config("pre_relax_steps", 50)),
                    "--pre-relax-fmax",
                    str(self.app.get_config("pre_relax_fmax", 0.1)),
                    "--velocity-policy",
                    self.app.get_config("velocity_policy", "auto"),
                    "--fmax-abort",
                    str(self.app.get_config("fmax_abort", 20.0)),
                    (
                        "--pre-relax"
                        if self.app.get_config("pre_relax", True)
                        else "--no-pre-relax"
                    ),
                ]
            )
            seed = self.app.get_config("seed")
            if seed is not None:
                command.extend(["--seed", str(seed)])
        return command

    def _refresh_job(self) -> None:
        if self._job_id is None:
            return
        log_text = self._job_manager.tail_log(self._job_id, lines=500)
        if log_text != self._displayed_log:
            if log_text.startswith(self._displayed_log):
                new_text = log_text[len(self._displayed_log) :]
            else:
                self.log_widget.clear()
                new_text = log_text
            for line in new_text.splitlines():
                self.log_widget.write_line(line)
            self._displayed_log = log_text

        data = self._job_manager.get_job(self._job_id)
        if data is None:
            return
        status = data.get("status", "unknown")
        if status in {"done", "failed", "cancelled"}:
            self.progress.update(total=100, progress=100 if status == "done" else 0)
            self.status.update(status.capitalize())
            self.query_one("#cancel-btn", Button).disabled = True
            if self._refresh_timer is not None:
                self._refresh_timer.stop()
                self._refresh_timer = None

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back-btn":
            self.app.pop_screen()
        elif event.button.id == "cancel-btn" and self._job_id is not None:
            if self._job_manager.kill_job(self._job_id):
                self.status.update("Cancelled")
                event.button.disabled = True
            else:
                self.notify("Job is no longer running", severity="warning")
