# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Modified for the mlipx project: multi-engine MLIP support (UMA/MACE/DPA/GRACE).
"""Configuration screen for mlipx TUI."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import TYPE_CHECKING

from textual.containers import Container, Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    Button,
    Input,
    Label,
    RadioButton,
    RadioSet,
    Select,
    Static,
    Switch,
)

if TYPE_CHECKING:
    from typing import ClassVar

    from textual.app import ComposeResult

from mlipx.tui.run_screen import RunScreen


class ConfigScreen(Screen):
    """Configuration screen for setting up calculation parameters."""

    BINDINGS: ClassVar[list[tuple[str, str, str]]] = [
        ("escape", "back", "Back"),
        ("up", "scroll_up", "Scroll Up"),
        ("down", "scroll_down", "Scroll Down"),
        ("pageup", "page_up", "Page Up"),
        ("pagedown", "page_down", "Page Down"),
    ]

    CSS = """
    #structure-status, #model-status {
        margin-top: 0;
        margin-bottom: 1;
        padding-left: 1;
    }
    .status-ok {
        color: $success;
        text-style: bold;
    }
    .status-error {
        color: $error;
        text-style: italic;
    }
    .switch-row {
        height: 3;
    }
    """

    def compose(self) -> ComposeResult:
        """Compose the configuration screen."""
        calc_type = self.app.get_config("calc_type", "sp")

        # Main container fills the screen
        with Container(id="config-main"):
            # Header at top
            yield Container(
                Static(f"Configuration: {calc_type.upper()}", id="title"),
                id="config-header",
            )

            # Scrollable content area
            with VerticalScroll(id="config-scroll"):
                # File paths section
                yield Static("📁 File Paths", classes="section-title")

                yield Label("Structure File:")
                yield Input(
                    placeholder="e.g., structure.cif or POSCAR (relative paths supported)",
                    value=str(self.app.get_config("structure_file", "") or ""),
                    id="structure-input",
                )
                yield Static("", id="structure-status")

                yield Label("Model File:")
                yield Input(
                    placeholder="e.g., uma-s-1.pt (relative paths supported)",
                    value=str(self.app.get_config("model_file", "") or ""),
                    id="model-input",
                )
                yield Static("", id="model-status")

                yield Label("Output Directory:")
                yield Input(
                    value=self.app.get_config("output_dir", "./results"),
                    id="output-input",
                )

                yield Label("Job Name (optional):")
                yield Input(
                    placeholder="e.g., structure_01",
                    value=str(self.app.get_config("job_name", "") or ""),
                    id="job-name-input",
                )

                # Model engine + task selection
                yield Static("⚙️  Model Engine & Task", classes="section-title")

                yield Label("Model Engine:")
                yield Select(
                    options=[
                        ("UMA (FAIRChem)", "uma"),
                        ("MACE", "mace"),
                        ("DPA (DeepMD-kit)", "dpa"),
                        ("GRACE", "grace"),
                    ],
                    value=self.app.get_config("model_type", "uma"),
                    id="model-type-select",
                )

                yield Label("Task Type:")
                yield Select(
                    options=self._task_options(),
                    value=self._default_task_value(),
                    id="task-select",
                )

                yield Label("Device:")
                yield Input(
                    value=str(self.app.get_config("device", "cpu")),
                    placeholder="cpu, cuda, or cuda:N (for example cuda:1)",
                    id="device-input",
                )

                yield Static("🧠 Backend & Resource Options", classes="section-title")
                yield Static(
                    "UMA-only controls are disabled for other engines. CPU Threads "
                    "controls PyTorch (UMA/MACE/DPA) or TensorFlow (GRACE).",
                )

                yield Label("MACE Precision:")
                yield Select(
                    options=[("float32 (faster)", "float32"), ("float64", "float64")],
                    value=self.app.get_config("default_dtype", "float32"),
                    id="dtype-select",
                )

                yield Label("MACE/DPA Model Head or Branch (optional):")
                yield Input(
                    value=str(self.app.get_config("head", "") or ""),
                    placeholder="e.g., default or Domains_SSE_PBE",
                    id="head-input",
                )

                yield Label("UMA Inference Mode:")
                yield Select(
                    options=[("default", "default"), ("turbo", "turbo")],
                    value=self.app.get_config("inference_mode", "default"),
                    id="inference-mode-select",
                )

                threads = self.app.get_config("torch_num_threads")
                yield Label("CPU Threads (blank = backend/system default):")
                yield Input(
                    value="" if threads is None else str(threads),
                    placeholder="e.g., 4",
                    id="torch-threads-input",
                )

                checkpointing = self.app.get_config("activation_checkpointing")
                checkpointing_value = (
                    "auto"
                    if checkpointing is None
                    else ("on" if checkpointing else "off")
                )
                yield Label("UMA Activation Checkpointing:")
                yield Select(
                    options=[
                        ("Auto (from inference mode)", "auto"),
                        ("On (lower VRAM)", "on"),
                        ("Off (faster, more VRAM)", "off"),
                    ],
                    value=checkpointing_value,
                    id="activation-checkpointing-select",
                )

                # Calculation-specific options
                yield Static("🔧 Calculation Options", classes="section-title")
                yield from self._calc_options(calc_type)

                # Add some bottom padding
                yield Static("")

            # Action buttons fixed at bottom
            with Horizontal(id="button-bar"):
                yield Button("◀ Back", id="back-btn")
                yield Button("🚀 Run", variant="success", id="run-btn")

    # UMA-specific tasks vs generic bulk/molecule for other engines
    UMA_TASKS = [
        ("Inorganic Materials (omat)", "omat"),
        ("Molecules (omol)", "omol"),
        ("Catalysis OC20 (oc20)", "oc20"),
        ("Catalysis OC25 (oc25)", "oc25"),
        ("MOFs (odac)", "odac"),
        ("Molecular Crystals (omc)", "omc"),
    ]
    GENERIC_TASKS = [
        ("Bulk / Periodic (bulk)", "bulk"),
        ("Molecule (molecule)", "molecule"),
    ]

    def _current_model_type(self) -> str:
        """Model type from the select widget (or app config fallback)."""
        try:
            sel = self.query_one("#model-type-select", Select)
            if sel.value:
                return str(sel.value)
        except Exception:
            pass
        return str(self.app.get_config("model_type", "uma"))

    def _task_options(self):
        """Task options for the current model engine."""
        return (
            self.UMA_TASKS
            if self._current_model_type() == "uma"
            else self.GENERIC_TASKS
        )

    def _default_task_value(self) -> str:
        """Default task value for the current model engine."""
        current = str(self.app.get_config("task", "omat"))
        valid = {v for _, v in self._task_options()}
        return (
            current
            if current in valid
            else ("omat" if self._current_model_type() == "uma" else "bulk")
        )

    def _calc_options(self, calc_type: str):
        """Generate calculation-specific options."""
        if calc_type == "opt":
            yield Label("Force Threshold (eV/Å):")
            yield Input(value=str(self.app.get_config("fmax", 0.05)), id="fmax-input")

            yield Label("Max Steps:")
            yield Input(
                value=str(self.app.get_config("max_steps", 500)), id="max-steps-input"
            )

            yield Label("Optimizer:")
            yield Select(
                options=[("FIRE", "FIRE"), ("BFGS", "BFGS"), ("LBFGS", "LBFGS")],
                value=self.app.get_config("optimizer", "FIRE"),
                id="optimizer-select",
            )

            yield Horizontal(
                Label("Cell Optimization:"),
                Switch(value=self.app.get_config("cell_opt", False), id="cell-opt"),
                classes="switch-row",
            )
            yield Horizontal(
                Label("Preserve Crystal Symmetry:"),
                Switch(
                    value=self.app.get_config("fix_symmetry", False),
                    id="fix-symmetry",
                ),
                classes="switch-row",
            )

        elif calc_type == "md":
            current_ensemble = self.app.get_config("ensemble", "NVT").upper()
            yield Label("Ensemble:")
            yield RadioSet(
                RadioButton("NVT", id="nvt", value=current_ensemble == "NVT"),
                RadioButton("NVE", id="nve", value=current_ensemble == "NVE"),
                id="ensemble-radio",
            )

            yield Label("Temperature (K):")
            yield Input(
                value=str(self.app.get_config("temperature", 300.0)),
                id="temp-input",
            )

            yield Label("Time Step (fs):")
            yield Input(
                value=str(self.app.get_config("timestep", 1.0)),
                id="timestep-input",
            )

            yield Label("Steps:")
            yield Input(
                value=str(self.app.get_config("md_steps", 1000)),
                id="steps-input",
            )

            yield Label("Save Interval:")
            yield Input(
                value=str(self.app.get_config("save_interval", 10)),
                id="save-interval-input",
            )

            yield Label("NVT Friction (1/fs):")
            yield Input(
                value=str(self.app.get_config("friction", 0.001)),
                id="friction-input",
            )

            yield Horizontal(
                Label("Pre-relaxation:"),
                Switch(
                    value=self.app.get_config("pre_relax", True),
                    id="pre-relax",
                ),
                classes="switch-row",
            )

            yield Label("Pre-relaxation Max Steps:")
            yield Input(
                value=str(self.app.get_config("pre_relax_steps", 50)),
                id="pre-relax-steps-input",
            )

            yield Label("Pre-relaxation Force Threshold (eV/Å):")
            yield Input(
                value=str(self.app.get_config("pre_relax_fmax", 0.1)),
                id="pre-relax-fmax-input",
            )

            seed = self.app.get_config("seed")
            yield Label("Random Seed (blank = auto-generate and record):")
            yield Input(
                value="" if seed is None else str(seed),
                placeholder="e.g., 42",
                id="seed-input",
            )

            yield Label("Velocity Policy:")
            yield Select(
                options=[
                    ("Auto", "auto"),
                    ("Initialize / overwrite", "initialize"),
                    ("Preserve existing velocities", "preserve"),
                ],
                value=self.app.get_config("velocity_policy", "auto"),
                id="velocity-policy-select",
            )

            yield Label("Large-force Warning Threshold (eV/Å):")
            yield Input(
                value=str(self.app.get_config("fmax_abort", 20.0)),
                id="fmax-abort-input",
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id

        if button_id == "back-btn":
            self.app.pop_screen()

        elif button_id == "run-btn":
            self._save_and_run()

    def _validate_path(self, input_id: str, value: str) -> bool:
        """Validate input path and update its status label.

        Returns True if valid/exists, False otherwise.
        """
        status_id = f"#{input_id.replace('-input', '-status')}"
        try:
            status_widget = self.query_one(status_id, Static)
        except Exception:
            return False

        val_stripped = value.strip()
        if not val_stripped:
            status_widget.update(
                "[!] Please specify a path (relative paths supported, e.g. ./structure.cif)"
            )
            status_widget.set_classes("status-error")
            return False

        path = Path(val_stripped)
        resolved_path = path.resolve()

        if resolved_path.exists():
            status_widget.update(f"[OK] Found: {resolved_path}")
            status_widget.set_classes("status-ok")
            return True
        else:
            status_widget.update(f"[NOT FOUND] Checked: {resolved_path}")
            status_widget.set_classes("status-error")
            return False

    def on_select_changed(self, event: Select.Changed) -> None:
        """Refresh task options when the model engine changes."""
        if event.select.id == "model-type-select":
            self.app.update_config("model_type", event.value)
            task_select = self.query_one("#task-select", Select)
            task_select.set_options(self._task_options())
            task_select.value = self._default_task_value()
            self._update_engine_option_states()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle live path validation on input change."""
        if event.input.id in ("structure-input", "model-input"):
            self._validate_path(event.input.id, event.value)

    def on_mount(self) -> None:
        """Initialize and validate paths on mount."""
        # Initial validation of pre-filled values
        structure_input = self.query_one("#structure-input", Input)
        model_input = self.query_one("#model-input", Input)
        self._validate_path("structure-input", structure_input.value)
        self._validate_path("model-input", model_input.value)
        self._update_engine_option_states()

    def _update_engine_option_states(self) -> None:
        """Enable only controls consumed by the selected backend."""
        model_type = self._current_model_type()
        self.query_one("#dtype-select", Select).disabled = model_type != "mace"
        self.query_one("#head-input", Input).disabled = model_type not in {
            "mace",
            "dpa",
        }
        self.query_one("#inference-mode-select", Select).disabled = (
            model_type != "uma"
        )
        self.query_one("#activation-checkpointing-select", Select).disabled = (
            model_type != "uma"
        )
        self.query_one("#torch-threads-input", Input).disabled = False

    def _save_and_run(self) -> None:
        """Save configuration and run calculation."""
        run_started_at = time.perf_counter()

        # Get file paths
        structure = self.query_one("#structure-input", Input).value
        model = self.query_one("#model-input", Input).value
        output = self.query_one("#output-input", Input).value

        if not structure:
            self.notify("Please specify a structure file", severity="error")
            return

        if not model:
            self.notify("Please specify a model file", severity="error")
            return

        # Resolve paths to absolute paths
        structure_path = Path(structure.strip()).resolve()
        model_path = Path(model.strip()).resolve()
        output_path = Path(output.strip()).resolve()

        if not structure_path.exists():
            self.notify(f"Structure file not found: {structure_path}", severity="error")
            return

        if not model_path.exists():
            self.notify(f"Model file not found: {model_path}", severity="error")
            return

        # Save absolute paths to config
        self.app.update_config("structure_file", str(structure_path))
        self.app.update_config("model_file", str(model_path))
        self.app.update_config("output_dir", str(output_path))

        # Get job name
        job_name = self.query_one("#job-name-input", Input).value
        if job_name:
            self.app.update_config("job_name", job_name)
        else:
            self.app.update_config("job_name", None)

        # Get model engine, task and device
        model_type_select = self.query_one("#model-type-select", Select)
        if model_type_select.value:
            self.app.update_config("model_type", model_type_select.value)
        model_type = str(self.app.get_config("model_type", "uma"))

        task_select = self.query_one("#task-select", Select)
        if task_select.value:
            self.app.update_config("task", task_select.value)

        device = self.query_one("#device-input", Input).value.strip().lower()
        if not re.fullmatch(r"(?:cpu|gpu|cuda(?::\d+)?)", device):
            self.notify(
                "Device must be cpu, gpu, cuda, or cuda:N (for example cuda:1)",
                severity="error",
            )
            return
        self.app.update_config("device", device)

        # Backend/resource options. Disabled controls are intentionally not
        # forwarded to an incompatible backend.
        if model_type == "mace":
            dtype_select = self.query_one("#dtype-select", Select)
            self.app.update_config("default_dtype", str(dtype_select.value))
        if model_type in {"mace", "dpa"}:
            head = self.query_one("#head-input", Input).value.strip()
            self.app.update_config("head", head or None)
        if model_type == "uma":
            inference_select = self.query_one("#inference-mode-select", Select)
            self.app.update_config("inference_mode", str(inference_select.value))
            checkpoint_select = self.query_one(
                "#activation-checkpointing-select", Select
            )
            checkpoint_value = str(checkpoint_select.value)
            self.app.update_config(
                "activation_checkpointing",
                None if checkpoint_value == "auto" else checkpoint_value == "on",
            )
        threads_text = self.query_one("#torch-threads-input", Input).value.strip()
        if threads_text:
            try:
                threads = int(threads_text)
            except ValueError:
                self.notify("CPU threads must be a positive integer", severity="error")
                return
            if threads < 1:
                self.notify("CPU threads must be at least 1", severity="error")
                return
            self.app.update_config("torch_num_threads", threads)
        else:
            self.app.update_config("torch_num_threads", None)

        # Get calculation-specific options
        calc_type = self.app.get_config("calc_type")

        if calc_type == "opt":
            try:
                fmax = float(self.query_one("#fmax-input", Input).value)
                self.app.update_config("fmax", fmax)
            except ValueError:
                pass

            try:
                max_steps = int(self.query_one("#max-steps-input", Input).value)
                self.app.update_config("max_steps", max_steps)
            except ValueError:
                pass

            optimizer_select = self.query_one("#optimizer-select", Select)
            if optimizer_select.value:
                self.app.update_config("optimizer", optimizer_select.value)

            cell_opt = self.query_one("#cell-opt", Switch)
            self.app.update_config("cell_opt", cell_opt.value)
            fix_symmetry = self.query_one("#fix-symmetry", Switch)
            self.app.update_config("fix_symmetry", fix_symmetry.value)

        elif calc_type == "md":
            ensemble_radio = self.query_one("#ensemble-radio", RadioSet)
            pressed_ensemble = ensemble_radio.pressed_button
            self.app.update_config(
                "ensemble",
                pressed_ensemble.id.upper() if pressed_ensemble else "NVT",
            )

            try:
                temp = float(self.query_one("#temp-input", Input).value)
                self.app.update_config("temperature", temp)
            except ValueError:
                pass

            try:
                timestep = float(self.query_one("#timestep-input", Input).value)
                self.app.update_config("timestep", timestep)
            except ValueError:
                pass

            try:
                steps = int(self.query_one("#steps-input", Input).value)
                self.app.update_config("md_steps", steps)
            except ValueError:
                pass

            try:
                save_interval = int(self.query_one("#save-interval-input", Input).value)
                self.app.update_config("save_interval", save_interval)
            except ValueError:
                pass

            try:
                friction = float(self.query_one("#friction-input", Input).value)
                self.app.update_config("friction", friction)
            except ValueError:
                pass

            pre_relax = self.query_one("#pre-relax", Switch)
            self.app.update_config("pre_relax", pre_relax.value)

            try:
                pre_relax_steps = int(
                    self.query_one("#pre-relax-steps-input", Input).value
                )
                self.app.update_config("pre_relax_steps", pre_relax_steps)
            except ValueError:
                pass

            try:
                pre_relax_fmax = float(
                    self.query_one("#pre-relax-fmax-input", Input).value
                )
                self.app.update_config("pre_relax_fmax", pre_relax_fmax)
            except ValueError:
                pass

            seed_text = self.query_one("#seed-input", Input).value.strip()
            if seed_text:
                try:
                    seed = int(seed_text)
                except ValueError:
                    self.notify(
                        "Random seed must be a non-negative integer",
                        severity="error",
                    )
                    return
                if seed < 0:
                    self.notify(
                        "Random seed must be a non-negative integer",
                        severity="error",
                    )
                    return
                self.app.update_config("seed", seed)
            else:
                self.app.update_config("seed", None)

            velocity_select = self.query_one("#velocity-policy-select", Select)
            self.app.update_config("velocity_policy", str(velocity_select.value))

            try:
                fmax_abort = float(self.query_one("#fmax-abort-input", Input).value)
                self.app.update_config("fmax_abort", fmax_abort)
            except ValueError:
                pass

        # TUI calculations are launched as persistent background processes.
        self.app.update_config("detach", True)
        self.app.update_config("run_started_at", run_started_at)

        # Go to run screen. Push a fresh instance so on_mount/on_compose re-run
        # for this run's config (the named "run" screen is cached by Textual).
        self.app.push_screen(RunScreen())

    def action_back(self) -> None:
        """Go back to main screen."""
        self.app.pop_screen()

    def action_scroll_up(self) -> None:
        """Scroll up in the config panel."""
        scroll = self.query_one("#config-scroll", VerticalScroll)
        scroll.scroll_up()

    def action_scroll_down(self) -> None:
        """Scroll down in the config panel."""
        scroll = self.query_one("#config-scroll", VerticalScroll)
        scroll.scroll_down()

    def action_page_up(self) -> None:
        """Page up in the config panel."""
        scroll = self.query_one("#config-scroll", VerticalScroll)
        scroll.scroll_page_up()

    def action_page_down(self) -> None:
        """Page down in the config panel."""
        scroll = self.query_one("#config-scroll", VerticalScroll)
        scroll.scroll_page_down()
