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
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


def _default_jobs_dir() -> Path:
    """Get default jobs directory: ~/.mlipx/jobs/"""
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
        return removed

    def submit(
        self,
        job_id: str,
        calc_type: str,
        structure: str,
        formula: str,
        natoms: int,
        device: str,
        cmd: list[str],
    ) -> subprocess.Popen:
        """Submit a calculation as a background subprocess.

        Args:
            job_id: Unique job identifier.
            calc_type: sp, opt, or md.
            structure: Path to structure file.
            formula: Chemical formula.
            natoms: Number of atoms.
            device: cpu or cuda.
            cmd: Full command to execute as subprocess (e.g., ['uv', 'run', 'mlipx', 'sp', ...]).

        Returns:
            Popen instance for the spawned process.
        """
        log_path = self._log_file(job_id)
        with open(log_path, "w") as log_f:
            proc = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

        self._write_job_state(
            job_id=job_id,
            status=JobStatus.RUNNING,
            calc_type=calc_type,
            structure=structure,
            formula=formula,
            natoms=natoms,
            pid=proc.pid,
            device=device,
        )
        return proc

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
            self._write_job_state(
                job_id=job_id,
                status=JobStatus.CANCELLED,
                calc_type=data["calc_type"],
                structure=data["structure"],
                formula=data["formula"],
                natoms=data["natoms"],
                pid=pid,
                device=data.get("device", "cpu"),
                finished_at=datetime.now().isoformat(),
            )
            return True
        except Exception:
            return False

    def _kill_process(self, pid: int) -> None:
        """Kill a process by PID, platform-appropriate."""
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/PID", str(pid), "/F"], check=False)
        else:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGTERM)

    def tail_log(self, job_id: str, lines: int = 50) -> str:
        """Return the last N lines of the job log."""
        log_path = self._log_file(job_id)
        if not log_path.exists():
            return ""
        with open(log_path, encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
            return "".join(all_lines[-lines:])
