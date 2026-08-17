"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Background job manager for MLIP calculations.

Manages calculation jobs as independent subprocesses with
disk-persisted state for attach/kill/clean operations.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import tempfile
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


class JobStatus(str, Enum):
    PENDING = "pending"
    PAUSED = "paused"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


def _default_jobs_dir() -> Path:
    """Get default jobs directory: ~/.mlipx/jobs/ (override via MLIPX_JOBS_DIR)."""
    override = os.environ.get("MLIPX_JOBS_DIR")
    if override:
        return Path(override)
    return Path.home() / ".mlipx" / "jobs"


class JobManager:
    """Manage background calculation jobs with disk-persisted state."""

    def __init__(self, jobs_dir: Path | None = None):
        self.jobs_dir = jobs_dir or _default_jobs_dir()
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._logs_dir = self.jobs_dir / "logs"
        self._logs_dir.mkdir(exist_ok=True)

    def _job_file(self, job_id: str) -> Path:
        return self.jobs_dir / f"{job_id}.json"

    def _log_file(self, job_id: str) -> Path:
        return self._logs_dir / f"{job_id}.log"

    def _write_job_state(
        self,
        job_id: str,
        status: JobStatus,
        calc_type: str,
        structure: str,
        formula: str,
        natoms: int,
        pid: int,
        device: str,
        progress: dict | None = None,
        results: dict | None = None,
        error: str | None = None,
        finished_at: str | None = None,
        cmd: list[str] | None = None,
        python: str | None = None,
    ) -> None:
        # Preserve original started_at if updating an existing job
        existing = self._read_job_state(job_id)
        original_started_at = existing["started_at"] if existing else None
        started_at_val = original_started_at or datetime.now().isoformat()

        data = {
            "job_id": job_id,
            "status": status.value,
            "calc_type": calc_type,
            "structure": structure,
            "formula": formula,
            "natoms": natoms,
            "pid": pid,
            "device": device,
            "started_at": started_at_val,
            "finished_at": finished_at,
            "log_file": str(self._log_file(job_id)),
            "progress": progress or {},
            "results": results,
            "error": error,
            "cmd": cmd,
            "python": python,
        }
        job_path = self._job_file(job_id)
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.jobs_dir,
                prefix=f".{job_id}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                json.dump(data, handle, indent=2)
                handle.flush()
                temp_path = Path(handle.name)
            temp_path.replace(job_path)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    def _read_job_state(self, job_id: str) -> dict[str, Any] | None:
        path = self._job_file(job_id)
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else None
        except (OSError, json.JSONDecodeError):
            return None

    def list_jobs(self) -> list[dict[str, Any]]:
        jobs = []
        for path in sorted(self.jobs_dir.glob("*.json")):
            data = self._read_job_state(path.stem)
            if data is not None:
                jobs.append(data)
        return jobs

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        return self._read_job_state(job_id)

    def update_status(
        self,
        job_id: str,
        status: JobStatus,
        *,
        error: str | None = None,
    ) -> bool:
        """Update a persisted job while preserving its identifying metadata."""
        data = self._read_job_state(job_id)
        if data is None:
            return False
        self._write_job_state(
            job_id=job_id,
            status=status,
            calc_type=data["calc_type"],
            structure=data["structure"],
            formula=data["formula"],
            natoms=data["natoms"],
            pid=data["pid"],
            device=data.get("device", "cpu"),
            progress=data.get("progress"),
            results=data.get("results"),
            error=error,
            cmd=data.get("cmd"),
            python=data.get("python"),
            finished_at=(
                datetime.now().isoformat()
                if status in {JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED}
                else None
            ),
        )
        return True

    def clean(self) -> list[str]:
        """Remove state files for done/failed/cancelled jobs. Returns list of removed IDs."""
        removed = []
        for path in self.jobs_dir.glob("*.json"):
            data = self._read_job_state(path.stem)
            if data is None:
                continue
            if data.get("status") in ("done", "failed", "cancelled"):
                path.unlink()
                removed.append(data["job_id"])
        # Also drop the log files of removed jobs (and any orphaned logs whose
        # state file is already gone), so clean() actually reclaims disk space.
        for log in self._logs_dir.glob("*.log"):
            if not self._job_file(log.stem).exists():
                log.unlink(missing_ok=True)
        return removed

    def _spawn_worker(self, job_id: str, cmd: list[str]) -> subprocess.Popen:
        """Start the worker subprocess that executes ``cmd`` and records output."""
        worker_cmd = [
            sys.executable,
            "-m",
            "mlipx.job_worker",
            str(self.jobs_dir.resolve()),
            job_id,
            *cmd,
        ]
        return subprocess.Popen(
            worker_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

    def submit(
        self,
        job_id: str,
        calc_type: str,
        structure: str,
        formula: str,
        natoms: int,
        device: str,
        cmd: list[str],
        python: str | None = None,
    ) -> subprocess.Popen:
        """Submit a calculation as a background subprocess (immediate start).

        See :meth:`enqueue` for the queued (PENDING) variant used by the
        scheduler; ``submit`` is the legacy immediate-run path.
        """
        self.enqueue(
            job_id,
            calc_type,
            structure,
            formula,
            natoms,
            device,
            cmd,
            python=python,
        )
        proc = self._spawn_worker(job_id, cmd)
        self.mark_running(job_id, proc.pid)
        return proc

    def enqueue(
        self,
        job_id: str,
        calc_type: str,
        structure: str,
        formula: str,
        natoms: int,
        device: str,
        cmd: list[str],
        python: str | None = None,
    ) -> None:
        """Add a job to the queue with status PENDING (Slurm-like).

        A scheduler process (``mlipx queue start``) promotes queued jobs to
        RUNNING one at a time (respecting the concurrency limit) and spawns
        the worker that executes ``cmd``.
        """
        self._log_file(job_id).write_text("", encoding="utf-8")
        self._write_job_state(
            job_id=job_id,
            status=JobStatus.PENDING,
            calc_type=calc_type,
            structure=structure,
            formula=formula,
            natoms=natoms,
            pid=0,
            device=device,
            cmd=list(cmd),
            python=python,
        )

    def mark_running(self, job_id: str, pid: int) -> bool:
        """Promote a queued (PENDING) job to RUNNING and record its PID."""
        data = self._read_job_state(job_id)
        if data is None:
            return False
        self._write_job_state(
            job_id=job_id,
            status=JobStatus.RUNNING,
            calc_type=data["calc_type"],
            structure=data["structure"],
            formula=data["formula"],
            natoms=data["natoms"],
            pid=pid,
            device=data.get("device", "cpu"),
            progress=data.get("progress"),
            results=data.get("results"),
            cmd=data.get("cmd"),
            python=data.get("python"),
        )
        return True

    def pause_pending(self, job_id: str) -> bool:
        """Move one PENDING job to PAUSED without touching its worker state."""
        data = self._read_job_state(job_id)
        if data is None or data.get("status") != JobStatus.PENDING.value:
            return False
        return self.update_status(job_id, JobStatus.PAUSED)

    def resume_paused(self, job_id: str) -> bool:
        """Move one PAUSED job back to PENDING so the scheduler can launch it."""
        data = self._read_job_state(job_id)
        if data is None or data.get("status") != JobStatus.PAUSED.value:
            return False
        return self.update_status(job_id, JobStatus.PENDING)

    def count_by_status(self, status: str) -> int:
        """Number of jobs currently in ``status`` (pending/running/done/...)."""
        return sum(1 for j in self.list_jobs() if j.get("status") == status)

    def next_pending(self) -> dict | None:
        """Oldest PENDING job (FIFO), or None when the queue is empty."""
        pending = [j for j in self.list_jobs() if j.get("status") == "pending"]
        if not pending:
            return None
        return min(pending, key=lambda j: j.get("started_at", ""))

    def queue_summary(self) -> dict[str, int]:
        """Counts per status, for `mlipx queue status`."""
        summary: dict[str, int] = {s.value: 0 for s in JobStatus}
        for job in self.list_jobs():
            status = job.get("status")
            if status in summary:
                summary[status] += 1
        return summary

    def kill_job(self, job_id: str) -> bool:
        """Kill a running job by PID. Returns True if successful."""
        data = self._read_job_state(job_id)
        if data is None:
            return False
        if data["status"] != "running":
            return False

        pid = data["pid"]

        try:
            self._kill_process(pid)
            self.update_status(job_id, JobStatus.CANCELLED)
            return True
        except Exception:
            return False

    def _kill_process(self, pid: int) -> None:
        """Kill a process by PID, platform-appropriate."""
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/PID", str(pid), "/F"], check=False)
        else:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pid, signal.SIGTERM)

    def tail_log(self, job_id: str, lines: int = 50) -> str:
        """Return the last N lines of the job log."""
        log_path = self._log_file(job_id)
        if not log_path.exists():
            return ""
        with open(log_path, encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
            return "".join(all_lines[-lines:])
