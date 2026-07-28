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
        config_screen.query_one("#timestep-input").value = "2.5"
        config_screen.query_one("#save-interval-input").value = "25"
        config_screen.query_one("#nve").value = True
        await pilot.pause()

        # Avoid actually mounting RunScreen in this unit test.
        app.push_screen = Mock()
        config_screen._save_and_run()

    assert app.get_config("ensemble") == "NVE"
    assert app.get_config("timestep") == 2.5
    assert app.get_config("save_interval") == 25
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
    app.update_config("md_steps", 2000)
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
