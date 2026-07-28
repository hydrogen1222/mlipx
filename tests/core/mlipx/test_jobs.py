"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import inspect
import tempfile
from pathlib import Path
from unittest.mock import patch

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

    def test_list_jobs_skips_partial_or_invalid_state_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = JobManager(jobs_dir=Path(tmpdir))
            mgr._write_job_state(
                "valid", JobStatus.RUNNING, "sp", "/a", "H2", 2, 100, "cpu"
            )
            (mgr.jobs_dir / "partial.json").write_text('{"job_id":', encoding="utf-8")
            (mgr.jobs_dir / "not-a-dict.json").write_text("[]", encoding="utf-8")

            jobs = mgr.list_jobs()

            assert [job["job_id"] for job in jobs] == ["valid"]
            assert mgr._read_job_state("partial") is None

    def test_job_state_write_is_atomic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = JobManager(jobs_dir=Path(tmpdir))
            mgr._write_job_state(
                "atomic", JobStatus.RUNNING, "sp", "/a", "H2", 2, 100, "cpu"
            )

            assert mgr._job_file("atomic").exists()
            assert list(mgr.jobs_dir.glob("*.tmp")) == []

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

    def test_submit_uses_persistent_worker_process(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = JobManager(jobs_dir=Path(tmpdir))
            with patch("mlipx.jobs.subprocess.Popen") as popen:
                popen.return_value.pid = 4321
                mgr.submit(
                    "md-job",
                    "md",
                    "/tmp/POSCAR",
                    "Li2",
                    2,
                    "cuda",
                    ["python", "-m", "mlipx.cli", "md", "/tmp/POSCAR"],
                )

            command = popen.call_args.args[0]
            assert command[1:3] == ["-m", "mlipx.job_worker"]
            assert command[-5:] == [
                "python",
                "-m",
                "mlipx.cli",
                "md",
                "/tmp/POSCAR",
            ]
            assert popen.call_args.kwargs["start_new_session"] is True
            assert mgr.get_job("md-job")["pid"] == 4321
            assert mgr.get_job("md-job")["status"] == "running"
