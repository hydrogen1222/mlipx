"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import inspect
import tempfile
from pathlib import Path

from mlipx.jobs import JobManager, JobStatus


class TestJobManager:
    """Tests for job state management (no subprocess needed)."""

    def test_job_status_enum(self):
        assert JobStatus.RUNNING.value == "running"
        assert JobStatus.DONE.value == "done"
        assert JobStatus.FAILED.value == "failed"

    def test_job_manager_create_job_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = JobManager(jobs_dir=Path(tmpdir))
            assert mgr.jobs_dir.exists()

    def test_write_and_read_job_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = JobManager(jobs_dir=Path(tmpdir))
            mgr._write_job_state(
                job_id="test_job",
                status=JobStatus.RUNNING,
                calc_type="sp",
                structure="/path/to/POSCAR",
                formula="H2O",
                natoms=3,
                pid=12345,
                device="cpu",
            )
            data = mgr._read_job_state("test_job")
            assert data["status"] == "running"
            assert data["formula"] == "H2O"

    def test_list_jobs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = JobManager(jobs_dir=Path(tmpdir))
            mgr._write_job_state(
                "job1", JobStatus.RUNNING, "sp", "/a/b.cif", "H2O", 3, 100, "cpu"
            )
            mgr._write_job_state(
                "job2", JobStatus.DONE, "opt", "/a/c.cif", "Cu", 16, 200, "cuda"
            )
            jobs = mgr.list_jobs()
            assert len(jobs) == 2

    def test_clean_removes_done_and_failed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = JobManager(jobs_dir=Path(tmpdir))
            mgr._write_job_state(
                "done_job", JobStatus.DONE, "sp", "/a", "H2O", 3, 100, "cpu"
            )
            mgr._write_job_state(
                "running_job", JobStatus.RUNNING, "sp", "/a", "H2O", 3, 200, "cpu"
            )
            mgr._write_job_state(
                "failed_job", JobStatus.FAILED, "sp", "/a", "H2O", 3, 300, "cpu"
            )
            removed = mgr.clean()
            assert len(removed) == 2
            remaining = mgr.list_jobs()
            assert len(remaining) == 1
            assert remaining[0]["job_id"] == "running_job"

    def test_kill_process_method_exists(self):
        """Verify _kill_process method is callable and accepts int."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = JobManager(jobs_dir=Path(tmpdir))
            assert callable(mgr._kill_process)

            sig = inspect.signature(mgr._kill_process)
            assert "pid" in sig.parameters
