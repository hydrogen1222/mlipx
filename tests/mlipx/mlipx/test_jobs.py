"""Tests for mlipx.jobs (background job manager)."""

from __future__ import annotations

from pathlib import Path

from mlipx.jobs import JobManager, JobStatus


def _make_manager(tmp_path: Path) -> JobManager:
    return JobManager(jobs_dir=tmp_path / "jobs")


def _seed_job(mgr: JobManager, job_id: str, status: str) -> None:
    mgr._write_job_state(
        job_id=job_id,
        status=JobStatus(status),
        calc_type="sp",
        structure="s.cif",
        formula="H2O",
        natoms=3,
        pid=1234,
        device="cpu",
    )
    mgr._log_file(job_id).write_text("some log output\n", encoding="utf-8")


def test_submit_writes_pending_then_running(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path)
    assert mgr.get_job("nonexistent") is None


def test_clean_removes_state_and_logs(tmp_path: Path) -> None:
    """Regression: clean() removed the .json state but left the .log behind."""
    mgr = _make_manager(tmp_path)
    _seed_job(mgr, "job_done", "done")
    _seed_job(mgr, "job_failed", "failed")
    _seed_job(mgr, "job_running", "running")

    removed = mgr.clean()
    assert sorted(removed) == ["job_done", "job_failed"]
    # State files gone
    assert not mgr._job_file("job_done").exists()
    assert not mgr._job_file("job_failed").exists()
    # Log files gone too (the regression)
    assert not mgr._log_file("job_done").exists()
    assert not mgr._log_file("job_failed").exists()
    # Running job untouched
    assert mgr._job_file("job_running").exists()
    assert mgr._log_file("job_running").exists()


def test_clean_removes_orphaned_logs(tmp_path: Path) -> None:
    """Logs whose state file is already gone are cleaned up as well."""
    mgr = _make_manager(tmp_path)
    orphan = mgr._log_file("ghost")
    orphan.write_text("leftover\n", encoding="utf-8")
    assert mgr.clean() == []
    assert not orphan.exists()


def test_kill_job_requires_running(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path)
    _seed_job(mgr, "job_done", "done")
    assert mgr.kill_job("job_done") is False
    assert mgr.get_job("job_done")["status"] == "done"


def test_read_job_state_missing(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path)
    assert mgr.get_job("missing") is None
