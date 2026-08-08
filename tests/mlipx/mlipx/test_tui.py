"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

if TYPE_CHECKING:
    from pathlib import Path
from mlipx.jobs import JobManager, JobStatus
from mlipx.tui.app import MlipxApp
from mlipx.tui.config_screen import ConfigScreen
from mlipx.tui.jobs_screen import JobDetailScreen, JobsScreen
from mlipx.tui.run_screen import RunScreen


@pytest.mark.asyncio
async def test_md_ensemble_and_options_persisted(tmp_path: Path) -> None:
    """MD ensemble, timestep and save interval are saved to app config."""
    structure = tmp_path / "structure.cif"
    model = tmp_path / "model.pt"
    structure.write_text("")
    model.write_text("")

    app = MlipxApp()
    app.update_config("calc_type", "md")

    async with app.run_test(size=(80, 80)) as pilot:
        config_screen = ConfigScreen()
        await app.push_screen(config_screen)
        await pilot.pause()

        config_screen.query_one("#structure-input").value = str(structure)
        config_screen.query_one("#model-input").value = str(model)
        config_screen.query_one("#device-input").value = "cuda:1"
        config_screen.query_one("#inference-mode-select").value = "turbo"
        config_screen.query_one("#torch-threads-input").value = "6"
        config_screen.query_one("#activation-checkpointing-select").value = "off"
        config_screen.query_one("#timestep-input").value = "2.5"
        config_screen.query_one("#save-interval-input").value = "25"
        config_screen.query_one("#friction-input").value = "0.004"
        config_screen.query_one("#pre-relax-steps-input").value = "12"
        config_screen.query_one("#pre-relax-fmax-input").value = "0.08"
        config_screen.query_one("#seed-input").value = "42"
        config_screen.query_one("#velocity-policy-select").value = "initialize"
        config_screen.query_one("#fmax-abort-input").value = "15"
        config_screen.query_one("#nve").value = True
        await pilot.pause()

        # Avoid actually mounting RunScreen in this unit test.
        app.push_screen = Mock()
        config_screen._save_and_run()

    assert app.get_config("ensemble") == "NVE"
    assert app.get_config("device") == "cuda:1"
    assert app.get_config("inference_mode") == "turbo"
    assert app.get_config("torch_num_threads") == 6
    assert app.get_config("activation_checkpointing") is False
    assert app.get_config("timestep") == 2.5
    assert app.get_config("save_interval") == 25
    # NVE hides and ignores thermostat-specific fields.
    assert app.get_config("friction") == 0.001
    assert app.get_config("pre_relax_steps") == 12
    assert app.get_config("pre_relax_fmax") == 0.08
    assert app.get_config("seed") == 42
    assert app.get_config("velocity_policy") == "initialize"
    assert app.get_config("fmax_abort") == 15.0
    assert isinstance(app.get_config("run_started_at"), float)


@pytest.mark.asyncio
async def test_opt_values_loaded_from_config() -> None:
    """ConfigScreen loads previously saved opt values when re-composed."""
    app = MlipxApp()
    app.update_config("calc_type", "opt")
    app.update_config("fmax", 0.02)
    app.update_config("max_steps", 100)
    app.update_config("optimizer", "BFGS")
    app.update_config("cell_opt", True)

    async with app.run_test(size=(80, 80)) as pilot:
        config_screen = ConfigScreen()
        await app.push_screen(config_screen)
        await pilot.pause()

        assert config_screen.query_one("#fmax-input").value == "0.02"
        assert config_screen.query_one("#max-steps-input").value == "100"
        assert config_screen.query_one("#optimizer-select").value == "BFGS"
        assert config_screen.query_one("#cell-opt").value is True


@pytest.mark.asyncio
async def test_md_values_loaded_from_config() -> None:
    """ConfigScreen loads previously saved md values when re-composed."""
    app = MlipxApp()
    app.update_config("calc_type", "md")
    app.update_config("ensemble", "NVE")
    app.update_config("temperature", 400.0)
    app.update_config("timestep", 2.0)
    app.update_config("steps", 2000)
    app.update_config("save_interval", 5)
    app.update_config("pre_relax", False)

    async with app.run_test(size=(80, 80)) as pilot:
        config_screen = ConfigScreen()
        await app.push_screen(config_screen)
        await pilot.pause()

        assert config_screen.query_one("#nve").value is True
        assert config_screen.query_one("#temp-input").value == "400.0"
        assert config_screen.query_one("#timestep-input").value == "2.0"
        assert config_screen.query_one("#steps-input").value == "2000"
        assert config_screen.query_one("#save-interval-input").value == "5"
        assert config_screen.query_one("#pre-relax").value is False


@pytest.mark.asyncio
async def test_backend_resource_controls_follow_selected_engine() -> None:
    """TUI exposes resource controls without forwarding invalid cross-engine options."""
    app = MlipxApp()

    async with app.run_test(size=(100, 100)) as pilot:
        config_screen = ConfigScreen()
        await app.push_screen(config_screen)
        await pilot.pause()

        assert config_screen.query_one("#device-input").value == "cpu"
        assert config_screen.query_one("#inference-mode-select").disabled is False
        assert config_screen.query_one("#activation-checkpointing-select").disabled is False
        assert config_screen.query_one("#dtype-select").disabled is True
        assert config_screen.query_one("#head-input").disabled is True
        assert config_screen.query_one("#inference-mode-select").display is True
        assert config_screen.query_one("#head-input").display is False
        assert config_screen.query_one("#dtype-select").display is False

        config_screen.query_one("#model-type-select").value = "dpa"
        await pilot.pause()
        assert config_screen.query_one("#inference-mode-select").disabled is True
        assert config_screen.query_one("#activation-checkpointing-select").disabled is True
        assert config_screen.query_one("#dtype-select").disabled is True
        assert config_screen.query_one("#head-input").disabled is False
        assert config_screen.query_one("#torch-threads-input").disabled is False
        assert config_screen.query_one("#inference-mode-select").display is False
        assert config_screen.query_one("#head-input").display is True
        assert config_screen.query_one("#dtype-select").display is False

        config_screen.query_one("#model-type-select").value = "grace"
        await pilot.pause()
        assert config_screen.query_one("#head-input").disabled is True
        assert config_screen.query_one("#head-input").display is False
        assert config_screen.query_one("#torch-threads-input").disabled is False


@pytest.mark.asyncio
async def test_molecular_charge_and_spin_controls_are_task_aware(tmp_path: Path) -> None:
    structure = tmp_path / "molecule.xyz"
    model = tmp_path / "uma.pt"
    structure.write_text("")
    model.write_text("")
    app = MlipxApp()

    async with app.run_test(size=(100, 100)) as pilot:
        config_screen = ConfigScreen()
        await app.push_screen(config_screen)
        await pilot.pause()

        assert config_screen.query_one("#charge-input").display is False
        assert config_screen.query_one("#spin-input").display is False

        config_screen.query_one("#task-select").value = "omol"
        await pilot.pause()
        assert config_screen.query_one("#charge-input").display is True
        assert config_screen.query_one("#spin-input").display is True
        assert "Multiplicity" in str(config_screen.query_one("#spin-label").render())

        config_screen.query_one("#structure-input").value = str(structure)
        config_screen.query_one("#model-input").value = str(model)
        config_screen.query_one("#charge-input").value = "-1"
        config_screen.query_one("#spin-input").value = "2"
        app.push_screen = Mock()
        config_screen._save_and_run()

    assert app.get_config("charge") == -1
    assert app.get_config("spin") == 2


@pytest.mark.asyncio
async def test_run_command_contains_tui_resource_and_md_options() -> None:
    """Every visible TUI option must reach the background CLI command."""
    app = MlipxApp()
    app.config.update(
        {
            "calc_type": "md",
            "structure_file": "/tmp/structure.vasp",
            "model_file": "/tmp/model.pt",
            "model_type": "uma",
            "task": "omat",
            "device": "cuda:1",
            "output_dir": "/tmp/out",
            "inference_mode": "turbo",
            "torch_num_threads": 6,
            "activation_checkpointing": False,
            "ensemble": "NVE",
            "temperature": 500.0,
            "timestep": 0.5,
            "steps": 10,
            "friction": 0.002,
            "save_interval": 2,
            "pre_relax": False,
            "pre_relax_steps": 7,
            "pre_relax_fmax": 0.07,
            "seed": 42,
            "velocity_policy": "initialize",
            "fmax_abort": 12.0,
        }
    )

    async with app.run_test(size=(80, 40)):
        screen = RunScreen()
        screen._job_id = "test-job"
        command = screen._build_command()

    for expected in (
        "--device",
        "cuda:1",
        "--inference-mode",
        "turbo",
        "--cpu-threads",
        "6",
        "--no-activation-checkpointing",
        "--velocity-policy",
        "initialize",
        "--fmax-abort",
        "12.0",
        "--seed",
        "42",
        "--no-pre-relax",
    ):
        assert expected in command
    assert "--thermostat" not in command
    assert "--friction" not in command


@pytest.mark.asyncio
async def test_run_command_contains_only_active_nhc_options() -> None:
    app = MlipxApp()
    app.config.update(
        {
            "calc_type": "md",
            "structure_file": "/tmp/structure.vasp",
            "model_file": "/tmp/model.pt",
            "model_type": "mace",
            "task": "bulk",
            "device": "cpu",
            "output_dir": "/tmp/out",
            "ensemble": "NVT",
            "thermostat": "NHC",
            "steps": 5,
            "friction": 0.003,
            "bussi_tau": 700.0,
            "nhc_tdamp": 120.0,
            "nhc_tchain": 4,
            "nhc_tloop": 2,
        }
    )

    async with app.run_test(size=(80, 40)):
        screen = RunScreen()
        screen._job_id = "test-nhc"
        command = screen._build_command()

    assert command[command.index("--thermostat") + 1] == "NHC"
    assert command[command.index("--nhc-tdamp") + 1] == "120.0"
    assert command[command.index("--nhc-tchain") + 1] == "4"
    assert command[command.index("--nhc-tloop") + 1] == "2"
    assert "--friction" not in command
    assert "--bussi-tau" not in command


@pytest.mark.asyncio
async def test_md_thermostat_controls_are_dynamic_and_task_label_is_accurate() -> None:
    app = MlipxApp()
    app.update_config("calc_type", "md")

    async with app.run_test(size=(100, 100)) as pilot:
        screen = ConfigScreen()
        await app.push_screen(screen)
        await pilot.pause()

        assert "Task Type" in str(screen.query_one("#task-type-label").render())
        assert screen.query_one("#friction-input").display is True
        assert screen.query_one("#bussi-tau-input").display is False
        assert screen.query_one("#nhc-tdamp-input").display is False

        screen.query_one("#thermostat-select").value = "BUSSI"
        await pilot.pause()
        assert screen.query_one("#friction-input").display is False
        assert screen.query_one("#bussi-tau-input").display is True

        screen.query_one("#thermostat-select").value = "NHC"
        await pilot.pause()
        assert screen.query_one("#bussi-tau-input").display is False
        assert screen.query_one("#nhc-tdamp-input").display is True
        assert screen.query_one("#nhc-tchain-input").display is True
        assert screen.query_one("#nhc-tloop-input").display is True

        screen.query_one("#nve").value = True
        await pilot.pause()
        assert screen.query_one("#thermostat-select").display is False
        assert screen.query_one("#nhc-tdamp-input").display is False

        screen.query_one("#model-type-select").value = "dpa"
        await pilot.pause()
        assert "System Type" in str(screen.query_one("#task-type-label").render())
        assert "TensorFlow" in str(
            screen.query_one("#engine-options-note").render()
        )


@pytest.mark.asyncio
async def test_md_invalid_numeric_inputs_block_submission(tmp_path: Path) -> None:
    structure = tmp_path / "structure.cif"
    model = tmp_path / "model.pt"
    structure.write_text("")
    model.write_text("")
    app = MlipxApp()
    app.update_config("calc_type", "md")

    async with app.run_test(size=(100, 100)) as pilot:
        screen = ConfigScreen()
        await app.push_screen(screen)
        await pilot.pause()
        screen.query_one("#structure-input").value = str(structure)
        screen.query_one("#model-input").value = str(model)
        screen.notify = Mock()
        app.push_screen = Mock()

        cases = [
            ("LANGEVIN", "#temp-input", "bad"),
            ("LANGEVIN", "#timestep-input", "0"),
            ("LANGEVIN", "#steps-input", "1.5"),
            ("LANGEVIN", "#save-interval-input", "0"),
            ("LANGEVIN", "#friction-input", "0"),
            ("BUSSI", "#bussi-tau-input", "bad"),
            ("NHC", "#nhc-tdamp-input", "0"),
            ("NHC", "#nhc-tchain-input", "0"),
            ("NHC", "#nhc-tloop-input", "bad"),
            ("LANGEVIN", "#pre-relax-steps-input", "bad"),
            ("LANGEVIN", "#pre-relax-fmax-input", "nan"),
            ("LANGEVIN", "#fmax-abort-input", "bad"),
        ]
        defaults = {
            "#temp-input": "300",
            "#timestep-input": "1",
            "#steps-input": "5",
            "#save-interval-input": "1",
            "#friction-input": "0.001",
            "#bussi-tau-input": "1000",
            "#nhc-tdamp-input": "100",
            "#nhc-tchain-input": "3",
            "#nhc-tloop-input": "1",
            "#pre-relax-steps-input": "50",
            "#pre-relax-fmax-input": "0.1",
            "#fmax-abort-input": "20",
        }
        for thermostat, selector, invalid in cases:
            for input_selector, value in defaults.items():
                screen.query_one(input_selector).value = value
            screen.query_one("#thermostat-select").value = thermostat
            screen.query_one(selector).value = invalid
            screen._save_and_run()
            app.push_screen.assert_not_called()
            assert screen.notify.called
            screen.notify.reset_mock()


@pytest.mark.asyncio
async def test_run_command_contains_tui_charge_and_spin() -> None:
    app = MlipxApp()
    app.config.update(
        {
            "calc_type": "sp",
            "structure_file": "/tmp/molecule.xyz",
            "model_file": "/tmp/uma.pt",
            "model_type": "uma",
            "task": "omol",
            "device": "cpu",
            "output_dir": "/tmp/out",
            "charge": -1,
            "spin": 2,
        }
    )

    async with app.run_test(size=(80, 40)):
        screen = RunScreen()
        screen._job_id = "test-molecule"
        command = screen._build_command()

    assert command[command.index("--charge") + 1] == "-1"
    assert command[command.index("--spin") + 1] == "2"


@pytest.mark.asyncio
async def test_md_switch_row_does_not_clip_or_overlap() -> None:
    """The pre-relax label and switch have a full row and no ghost control."""
    app = MlipxApp()
    app.update_config("calc_type", "md")

    async with app.run_test(size=(80, 40)) as pilot:
        config_screen = ConfigScreen()
        await app.push_screen(config_screen)
        await pilot.pause()

        pre_relax = config_screen.query_one("#pre-relax")
        row = pre_relax.parent
        label = row.query_one("Label")

        assert row.has_class("switch-row")
        assert row.region.height >= pre_relax.region.height
        assert row.region.y <= label.region.y < row.region.bottom
        assert len(config_screen.query("#detach-switch")) == 0

        before = pre_relax.value
        pre_relax.focus()
        await pilot.press("space")
        await pilot.pause()
        assert pre_relax.value is not before


@pytest.mark.asyncio
async def test_job_detail_screen_displays_log() -> None:
    """JobDetailScreen writes supplied log text into the Log widget."""
    app = MlipxApp()

    async with app.run_test(size=(80, 80)) as pilot:
        detail = JobDetailScreen("test-job", "line one\nline two")
        await app.push_screen(detail)
        await pilot.pause()

        log = detail.query_one("#job-detail-log")
        assert len(log.lines) == 2


@pytest.mark.asyncio
async def test_jobs_screen_handles_empty_store_and_remounts(tmp_path: Path) -> None:
    """An empty jobs screen mounts, refreshes, and remounts without crashing."""
    app = MlipxApp()
    jobs_screen = JobsScreen()
    jobs_screen._job_manager = JobManager(jobs_dir=tmp_path)

    async with app.run_test(size=(80, 40)) as pilot:
        await app.push_screen(jobs_screen)
        await pilot.pause()

        table = jobs_screen.query_one("#jobs-table")
        assert len(table.columns) == 6
        assert table.row_count == 0
        assert jobs_screen.query_one("#pause-job-btn")
        assert jobs_screen.query_one("#resume-job-btn")
        assert jobs_screen._jobs_refresh_timer is not None

        app.pop_screen()
        await pilot.pause()
        assert jobs_screen._jobs_refresh_timer is None

        await app.push_screen(jobs_screen)
        await pilot.pause()
        assert len(table.columns) == 6
        assert table.row_count == 0


@pytest.mark.asyncio
async def test_jobs_screen_uses_job_id_as_row_key(tmp_path: Path) -> None:
    """Selecting a table row resolves to the persisted job ID."""
    manager = JobManager(jobs_dir=tmp_path)
    manager._write_job_state(
        "job-123",
        status=JobStatus.RUNNING,
        calc_type="sp",
        structure="/tmp/POSCAR",
        formula="H2",
        natoms=2,
        pid=123,
        device="cpu",
    )

    app = MlipxApp()
    jobs_screen = JobsScreen()
    jobs_screen._job_manager = manager
    async with app.run_test(size=(80, 40)) as pilot:
        await app.push_screen(jobs_screen)
        await pilot.pause()

        table = jobs_screen.query_one("#jobs-table")
        assert [row_key.value for row_key in table.rows] == ["job-123"]


@pytest.mark.asyncio
async def test_jobs_screen_refresh_preserves_selected_job(tmp_path: Path) -> None:
    """Auto-refresh must not move the cursor back to the first running job."""
    manager = JobManager(jobs_dir=tmp_path)
    for job_id, status, pid in (
        ("a-running", JobStatus.RUNNING, 123),
        ("b-pending", JobStatus.PENDING, 0),
    ):
        manager._write_job_state(
            job_id,
            status=status,
            calc_type="sp",
            structure="/tmp/POSCAR",
            formula="H2",
            natoms=2,
            pid=pid,
            device="cpu",
        )

    app = MlipxApp()
    jobs_screen = JobsScreen()
    jobs_screen._job_manager = manager
    async with app.run_test(size=(80, 40)) as pilot:
        await app.push_screen(jobs_screen)
        await pilot.pause()

        table = jobs_screen.query_one("#jobs-table")
        table.move_cursor(row=1)
        assert table.get_row_at(table.cursor_row)[0] == "b-pending"
        jobs_screen._refresh_table()
        assert table.get_row_at(table.cursor_row)[0] == "b-pending"


def test_run_screen_unmount_keeps_background_job_running() -> None:
    """Leaving RunScreen stops UI refresh without cancelling the job."""
    screen = RunScreen()
    timer = Mock()
    manager = Mock()
    screen._refresh_timer = timer
    screen._job_manager = manager

    screen.on_unmount()

    timer.stop.assert_called_once()
    assert screen._refresh_timer is None
    manager.kill_job.assert_not_called()


def test_jobs_screen_unmount_cancels_timer() -> None:
    """JobsScreen cancels its refresh timer when the screen is unmounted."""
    screen = JobsScreen()
    timer = Mock()
    screen._jobs_refresh_timer = timer

    screen.on_unmount()

    timer.stop.assert_called_once()
    assert screen._jobs_refresh_timer is None
