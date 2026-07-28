"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Jobs screen for managing background calculations.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, ClassVar

from textual.containers import Container, Horizontal
from textual.screen import Screen
from textual.widgets import Button, DataTable, Log, Static

from mlipx.jobs import JobManager

if TYPE_CHECKING:
    from textual.app import ComposeResult


class JobsScreen(Screen):
    """Screen for viewing and managing background jobs."""

    BINDINGS: ClassVar[list] = [
        ("escape", "back", "Back"),
        ("c", "cancel_job", "Cancel Job"),
        ("d", "delete_job", "Delete"),
        ("r", "refresh", "Refresh"),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._job_manager = JobManager()
        self._refresh_timer: asyncio.Task | None = None

    def compose(self) -> ComposeResult:
        yield Container(
            Static("Background Jobs", id="title"),
            Static("Manage running and completed calculations", id="subtitle"),
            DataTable(id="jobs-table"),
            Horizontal(
                Button("Cancel Job", variant="error", id="cancel-job-btn"),
                Button("Delete", id="delete-btn"),
                Button("Refresh", id="refresh-btn"),
                Button("Back", id="back-btn"),
                id="jobs-button-bar",
            ),
            id="jobs-main",
        )

    def on_mount(self) -> None:
        table = self.query_one("#jobs-table", DataTable)
        table.add_columns("ID", "Status", "Type", "Formula", "Atoms", "Device")
        self._refresh_table()
        self._refresh_timer = asyncio.create_task(self._auto_refresh())

    def on_unmount(self) -> None:
        if self._refresh_timer is not None and not self._refresh_timer.done():
            self._refresh_timer.cancel()

    async def _auto_refresh(self) -> None:
        while True:
            await asyncio.sleep(2)
            self._refresh_table()

    def _refresh_table(self) -> None:
        table = self.query_one("#jobs-table", DataTable)
        table.clear()
        jobs = self._job_manager.list_jobs()
        status_icons = {
            "running": "●",
            "done": "✓",
            "failed": "✗",
            "cancelled": "⊘",
            "pending": "○",
        }
        for job in jobs:
            icon = status_icons.get(job["status"], "?")
            table.add_row(
                job["job_id"],
                f"{icon} {job['status']}",
                job.get("calc_type", ""),
                job.get("formula", ""),
                str(job.get("natoms", "")),
                job.get("device", ""),
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "back-btn":
            self.app.pop_screen()
        elif button_id == "refresh-btn":
            self._refresh_table()
        elif button_id == "cancel-job-btn":
            self._cancel_selected_job()
        elif button_id == "delete-btn":
            self._delete_selected_job()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Show job detail/log on selection."""
        row_key = event.row_key
        if row_key is not None:
            job_id = str(event.row_key.value)
            log_text = self._job_manager.tail_log(job_id, lines=200)
            self.app.push_screen(JobDetailScreen(job_id, log_text))

    def _cancel_selected_job(self) -> None:
        table = self.query_one("#jobs-table", DataTable)
        if table.cursor_row is not None and table.cursor_row < table.row_count:
            row = table.get_row_at(table.cursor_row)
            job_id = str(row[0])
            ok = self._job_manager.kill_job(job_id)
            if ok:
                self.app.notify(f"Cancelled job: {job_id}", title="OK")
            else:
                self.app.notify(
                    f"Failed to cancel: {job_id}", title="Error", severity="error"
                )
            self._refresh_table()

    def _delete_selected_job(self) -> None:
        table = self.query_one("#jobs-table", DataTable)
        if table.cursor_row is not None and table.cursor_row < table.row_count:
            row = table.get_row_at(table.cursor_row)
            job_id = str(row[0])
            data = self._job_manager.get_job(job_id)
            if data and data["status"] != "running":
                job_file = self._job_manager._job_file(job_id)
                job_file.unlink(missing_ok=True)
                self._refresh_table()

    def action_back(self) -> None:
        if self._refresh_timer:
            self._refresh_timer.cancel()
        self.app.pop_screen()

    def action_cancel_job(self) -> None:
        self._cancel_selected_job()

    def action_delete_job(self) -> None:
        self._delete_selected_job()

    def action_refresh(self) -> None:
        self._refresh_table()


class JobDetailScreen(Screen):
    """Screen for viewing job log output."""

    BINDINGS: ClassVar[list] = [("escape", "back", "Back")]

    def __init__(self, job_id: str, log_text: str, **kwargs):
        super().__init__(**kwargs)
        self.job_id = job_id
        self.log_text = log_text

    def compose(self) -> ComposeResult:
        yield Container(
            Static(f"Job: {self.job_id}", id="title"),
            Log(id="job-detail-log"),
            Button("Back", id="back-btn"),
            id="job-detail-container",
        )

    def on_mount(self) -> None:
        log = self.query_one("#job-detail-log", Log)
        text = self.log_text or "(no log output)"
        for line in text.splitlines():
            log.write_line(line)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back-btn":
            self.app.pop_screen()

    def action_back(self) -> None:
        self.app.pop_screen()
