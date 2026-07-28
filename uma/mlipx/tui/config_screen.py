# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Modified for the mlipx project: multi-engine MLIP support (UMA/MACE/DPA/GRACE).
"""Configuration screen for mlipx TUI."""

from __future__ import annotations

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
                yield RadioSet(
                    RadioButton(
                        "CPU", id="cpu", value=self.app.get_config("device") == "cpu"
                    ),
                    RadioButton(
                        "CUDA (GPU)",
                        id="cuda",
                        value=self.app.get_config("device") == "cuda",
                    ),
                    id="device-radio",
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

            yield Horizontal(
                Label("Pre-relaxation (recommended):"),
                Switch(
                    value=self.app.get_config("pre_relax", True),
                    id="pre-relax",
                ),
                classes="switch-row",
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

    def _save_and_run(self) -> None:
        """Save configuration and run calculation."""
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

        task_select = self.query_one("#task-select", Select)
        if task_select.value:
            self.app.update_config("task", task_select.value)

        # Get device from radio buttons
        device_radio = self.query_one("#device-radio", RadioSet)
        selected_device = device_radio.pressed_button
        if selected_device:
            self.app.update_config("device", selected_device.id)

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

            pre_relax = self.query_one("#pre-relax", Switch)
            self.app.update_config("pre_relax", pre_relax.value)

        # Background execution is not exposed in the TUI yet.
        self.app.update_config("detach", False)

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
