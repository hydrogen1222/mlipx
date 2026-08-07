"""Slurm-like job queue for mlipx.

Background calculations can be *queued* instead of launched immediately:

* tasks are submitted with status ``PENDING`` (``mlipx queue submit tasks.json``
  or the TUI), each carrying its own interpreter (``python``), engine
  (``model_type``), model, structure, calc type and options;
* a scheduler process (``mlipx queue start``) promotes queued jobs to
  ``RUNNING`` one at a time -- by default at most one job runs concurrently
  (single GPU), which can be raised with ``--max-concurrent N`` for multi-GPU
  machines (Slurm-like);
* when a job finishes (DONE/FAILED), the scheduler automatically starts the
  next queued job.

This module also owns the task-file format and the shared
``build_mlipx_command`` helper used by both the CLI queue commands and the
TUI, so the two interfaces cannot drift apart.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from mlipx.jobs import JobManager, JobStatus

if TYPE_CHECKING:
    from typing import Any

#: Task file keys that are per-task run options vs. structural fields.
_STRUCTURAL_KEYS = {
    "name",
    "python",
    "calc_type",
    "structure",
    "model",
    "model_type",
    "task",
    "device",
    "output_dir",
    "options",
}
_VALID_CALC_TYPES = {"sp", "opt", "md"}
_VALID_ENGINES = {"uma", "fairchem", "mace", "dpa", "grace"}

#: option key -> CLI flag mapping for the shared command builder.
_OPT_FLAGS: dict[str, tuple[str, ...]] = {
    "charge": ("--charge",),
    "spin": ("--spin",),
    "inference_mode": ("--inference-mode",),
    "activation_checkpointing": ("--activation-checkpointing", "--no-activation-checkpointing"),
    "torch_num_threads": ("--cpu-threads",),
    "default_dtype": ("--dtype",),
    "head": ("--head",),
    # opt
    "fmax": ("--fmax",),
    "max_steps": ("--max-steps",),
    "optimizer": ("--optimizer",),
    "cell_opt": ("--cell-opt", "--no-cell-opt"),
    "fix_symmetry": ("--fix-symmetry", "--no-fix-symmetry"),
    # md
    "ensemble": ("--ensemble",),
    "temperature": ("--temp",),
    "timestep": ("--timestep",),
    "steps": ("--steps",),
    "friction": ("--friction",),
    "save_interval": ("--save-interval",),
    "pre_relax": ("--pre-relax", "--no-pre-relax"),
    "pre_relax_steps": ("--pre-relax-steps",),
    "pre_relax_fmax": ("--pre-relax-fmax",),
    "velocity_policy": ("--velocity-policy",),
    "fmax_abort": ("--fmax-abort",),
    "seed": ("--seed",),
}


def build_mlipx_command(
    calc_type: str,
    structure: str,
    model: str,
    model_type: str = "uma",
    task: str | None = None,
    device: str = "cpu",
    output_dir: str = "./results",
    job_name: str | None = None,
    options: dict[str, Any] | None = None,
    python: str | None = None,
) -> list[str]:
    """Build the argv for a ``mlipx <calc_type> ...`` calculation run.

    ``python`` selects the interpreter the calculation must run under (each
    engine has its own virtual environment, e.g. ``.venv`` for UMA,
    ``.venv-grace`` for GRACE). Defaults to the current interpreter.

    ``options`` maps mlipx option names (see :data:`_OPT_FLAGS`) to values;
    only keys known to the schema are forwarded, so a GRACE task never gets
    UMA-only flags and vice versa.
    """
    cmd = [
        python or sys.executable,
        "-m",
        "mlipx.cli",
        calc_type,
        structure,
        "--model",
        model,
        "--model-type",
        model_type,
        "--device",
        device,
        "--output",
        output_dir,
    ]
    if task:
        cmd.extend(["--task", task])
    if job_name:
        cmd.extend(["--name", job_name])

    engine = model_type.lower()
    for key, value in sorted((options or {}).items()):
        if key not in _OPT_FLAGS or value is None:
            continue
        # UMA-only options must not leak to other engines; MACE/DPA head and
        # dtype are engine-specific too. The CLI itself already ignores
        # inapplicable flags, but keeping the argv clean is friendlier.
        if key in {"inference_mode", "activation_checkpointing"} and engine not in {
            "uma",
            "fairchem",
        }:
            continue
        if key == "default_dtype" and engine != "mace":
            continue
        if key == "head" and engine not in {"mace", "dpa"}:
            continue
        flags = _OPT_FLAGS[key]
        if isinstance(value, bool):
            cmd.append(flags[0] if value else flags[1])
        else:
            cmd.extend([flags[0], str(value)])
    return cmd


def parse_task_file(path: str | Path) -> dict[str, Any]:
    """Parse and validate a queue task file (JSON).

    Schema::

        {
          "max_concurrent": 1,               // optional, >= 1
          "tasks": [
            {
              "name": "opt-1",               // optional, must be unique
              "python": "/abs/.venv/bin/python",  // optional, per-task env
              "calc_type": "opt",            // sp | opt | md
              "structure": "/abs/a.cif",     // required, must exist
              "model": "/abs/uma-s-1.pt",    // required, must exist
              "model_type": "uma",           // optional
              "task": "omat",                // optional
              "device": "cuda:0",            // optional
              "output_dir": "/abs/out",      // optional
              "options": {"fmax": 0.05}      // optional run options
            }
          ]
        }

    Raises:
        ValueError: with a human-readable message on any problem.
    """
    path = Path(path)
    if not path.exists():
        raise ValueError(f"Task file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Task file {path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Task file must be a JSON object with a 'tasks' list")

    max_concurrent = data.get("max_concurrent", 1)
    try:
        max_concurrent = int(max_concurrent)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"max_concurrent must be an integer, got {max_concurrent!r}") from exc
    if max_concurrent < 1:
        raise ValueError(f"max_concurrent must be >= 1, got {max_concurrent}")

    raw_tasks = data.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError("Task file must contain a non-empty 'tasks' list")

    names: set[str] = set()
    tasks: list[dict[str, Any]] = []
    for index, task in enumerate(raw_tasks):
        if not isinstance(task, dict):
            raise ValueError(f"tasks[{index}] must be an object")
        label = f"tasks[{index}]"
        calc_type = str(task.get("calc_type", "")).lower()
        if calc_type not in _VALID_CALC_TYPES:
            raise ValueError(
                f"{label}: calc_type must be one of "
                f"{sorted(_VALID_CALC_TYPES)}, got {task.get('calc_type')!r}"
            )
        structure = str(task.get("structure", ""))
        model = str(task.get("model", ""))
        for key, value in (("structure", structure), ("model", model)):
            if not value:
                raise ValueError(f"{label}: missing required '{key}'")
            if not Path(value).exists():
                raise ValueError(f"{label}: {key} not found: {value}")
        model_type = str(task.get("model_type", "uma")).lower()
        if model_type not in _VALID_ENGINES:
            raise ValueError(
                f"{label}: model_type must be one of {sorted(_VALID_ENGINES)}, "
                f"got {task.get('model_type')!r}"
            )
        python = task.get("python")
        if python is not None:
            if not Path(python).exists():
                raise ValueError(f"{label}: python not found: {python}")
        name = str(task.get("name") or f"{calc_type}-{index + 1}")
        if name in names:
            raise ValueError(f"{label}: duplicate task name {name!r}")
        names.add(name)

        options = task.get("options") or {}
        if not isinstance(options, dict):
            raise ValueError(f"{label}: 'options' must be an object")
        cleaned: dict[str, Any] = {}
        for key, value in options.items():
            if key in _STRUCTURAL_KEYS:
                raise ValueError(f"{label}: option {key!r} is a structural key, not an option")
            cleaned[key] = value

        tasks.append(
            {
                "name": name,
                "python": str(python) if python else None,
                "calc_type": calc_type,
                "structure": structure,
                "model": model,
                "model_type": model_type,
                "task": str(task.get("task", "")).lower() or None,
                "device": str(task.get("device", "cpu")),
                "output_dir": str(task.get("output_dir", "./results")),
                "options": cleaned,
            }
        )

    return {"max_concurrent": max_concurrent, "tasks": tasks}


def submit_task_file(
    mgr: JobManager, path: str | Path
) -> tuple[list[str], int]:
    """Parse ``path`` and enqueue every task as a PENDING job.

    Returns ``(job_ids, max_concurrent)``.
    """
    parsed = parse_task_file(path)
    job_ids: list[str] = []
    for task in parsed["tasks"]:
        job_id = task["name"]
        cmd = build_mlipx_command(
            calc_type=task["calc_type"],
            structure=task["structure"],
            model=task["model"],
            model_type=task["model_type"],
            task=task["task"],
            device=task["device"],
            output_dir=task["output_dir"],
            job_name=job_id,
            options=task["options"],
            python=task["python"],
        )
        formula, natoms = _probe_structure(task["structure"])
        mgr.enqueue(
            job_id=job_id,
            calc_type=task["calc_type"],
            structure=task["structure"],
            formula=formula,
            natoms=natoms,
            device=task["device"],
            cmd=cmd,
            python=task["python"],
        )
        job_ids.append(job_id)
    return job_ids, parsed["max_concurrent"]


def _probe_structure(structure: str) -> tuple[str, int]:
    """Best-effort formula/atom count for the jobs table (never raises)."""
    try:
        from ase.io import read

        atoms = read(structure)
        return atoms.get_chemical_formula(), len(atoms)
    except Exception:
        return "?", 0


class QueueScheduler:
    """Poll the job directory and launch queued (PENDING) jobs.

    Concurrency is capped at ``max_concurrent`` RUNNING jobs; each launched
    job is executed by :mod:`mlipx.job_worker` under the interpreter recorded
    on the job, so queued tasks can mix engines with different virtual
    environments.
    """

    def __init__(
        self,
        jobs_dir: str | Path | None = None,
        max_concurrent: int = 1,
        poll_interval: float = 5.0,
    ):
        self.mgr = JobManager(jobs_dir)
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
        self.max_concurrent = max_concurrent
        self.poll_interval = max(0.5, float(poll_interval))
        self._stop_requested = False

    def request_stop(self) -> None:
        """Ask the scheduler to exit after the current poll cycle."""
        self._stop_requested = True

    def _reap_stale_running(self) -> None:
        """Mark RUNNING jobs whose worker process is gone as FAILED.

        Without this, a worker that dies without updating its job state
        (kill -9, backend crash, OOM killer) leaves the job stuck in RUNNING
        forever and permanently occupies a concurrency slot, blocking the
        whole queue.

        Two cases are covered:
        * workers spawned by *this* scheduler process -- reaped via waitpid
          so they cannot linger as zombies;
        * workers spawned by another process (legacy JobManager.submit) --
          detected with a liveness probe on their recorded PID.
        """
        # 1) Reap our own exited worker children (zombies would otherwise
        #    keep the PID "alive" for the probe below).
        while True:
            try:
                reaped = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                break  # no children at all
            if reaped[0] == 0:
                break  # no child has exited yet
            for job in self.mgr.list_jobs():
                if (
                    job.get("status") == "running"
                    and job.get("pid") == reaped[0]
                ):
                    self.mgr.update_status(
                        job["job_id"],
                        JobStatus.FAILED,
                        error=("Worker process exited unexpectedly "
                               f"(status {reaped[1]})"),
                    )
                    break
        # 2) RUNNING jobs whose recorded PID no longer exists (workers that
        #    were spawned by a different process than this scheduler).
        for job in self.mgr.list_jobs():
            if job.get("status") != "running":
                continue
            pid = job.get("pid")
            if not isinstance(pid, int) or pid <= 0:
                continue
            if not _pid_alive(pid):
                self.mgr.update_status(
                    job["job_id"],
                    JobStatus.FAILED,
                    error="Worker process exited unexpectedly",
                )

    def run_once(self) -> int:
        """Launch as many queued jobs as the concurrency limit allows.

        Returns the number of jobs started in this pass.
        """
        self._reap_stale_running()
        # Pausing the queue is deliberately different from stopping the
        # scheduler: existing RUNNING workers continue, while PENDING jobs
        # remain untouched until the queue is resumed.
        if queue_paused(self.mgr.jobs_dir):
            return 0
        launched = 0
        running = self.mgr.count_by_status("running")
        while running < self.max_concurrent:
            # Re-check between launches so a pause request cannot cause a
            # second pending job to start after the first one in this pass.
            if queue_paused(self.mgr.jobs_dir):
                break
            job = self.mgr.next_pending()
            if job is None:
                break
            job_id = job["job_id"]
            cmd = job.get("cmd") or []
            if not cmd:
                self.mgr.update_status(
                    job_id, JobStatus.FAILED, error="Job has no command recorded"
                )
                continue
            try:
                proc = self.mgr._spawn_worker(job_id, cmd)
            except OSError as exc:
                self.mgr.update_status(
                    job_id, JobStatus.FAILED, error=f"Could not spawn worker: {exc}"
                )
                continue
            self.mgr.mark_running(job_id, proc.pid)
            running += 1
            launched += 1
        return launched

    def run_forever(self, stop_file: str | Path | None = None) -> None:
        """Poll until :meth:`request_stop` is called or ``stop_file`` exists."""
        stop_path = Path(stop_file) if stop_file else None
        while not self._stop_requested:
            if stop_path is not None and stop_path.exists():
                break
            self.run_once()
            time.sleep(self.poll_interval)


def scheduler_pid_file(jobs_dir: str | Path) -> Path:
    """PID file for a background scheduler, next to the jobs directory."""
    return Path(jobs_dir).parent / "scheduler.pid"


def scheduler_pause_file(jobs_dir: str | Path) -> Path:
    """Persistent control file that pauses only pending queue dispatch."""
    return Path(jobs_dir).parent / "scheduler.paused"


def queue_paused(jobs_dir: str | Path) -> bool:
    """Return whether pending jobs are currently prevented from launching."""
    return scheduler_pause_file(jobs_dir).exists()


def pause_scheduler(jobs_dir: str | Path) -> bool:
    """Pause dispatching PENDING jobs without affecting RUNNING workers."""
    path = scheduler_pause_file(jobs_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return False
    path.touch()
    return True


def resume_scheduler(jobs_dir: str | Path) -> bool:
    """Resume dispatching PENDING jobs. Returns whether it was paused."""
    path = scheduler_pause_file(jobs_dir)
    if not path.exists():
        return False
    path.unlink()
    return True


def pause_pending_job(jobs_dir: str | Path, job_id: str) -> bool:
    """Pause one queued job without affecting other pending or running jobs."""
    return JobManager(jobs_dir).pause_pending(job_id)


def resume_paused_job(jobs_dir: str | Path, job_id: str) -> bool:
    """Resume one paused job so it can re-enter normal FIFO dispatch."""
    return JobManager(jobs_dir).resume_paused(job_id)


def start_scheduler(
    jobs_dir: str | Path,
    max_concurrent: int = 1,
    poll_interval: float = 5.0,
) -> int:
    """Launch a detached background scheduler; returns its PID."""
    if max_concurrent < 1:
        raise ValueError("max_concurrent must be >= 1")
    pid_file = scheduler_pid_file(jobs_dir)
    if pid_file.exists():
        existing = pid_file.read_text(encoding="utf-8").strip()
        if existing.isdigit() and _pid_alive(int(existing)):
            raise RuntimeError(
                f"A scheduler is already running (PID {existing}); "
                f"stop it with 'mlipx queue stop' first."
            )
        pid_file.unlink(missing_ok=True)
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mlipx.queue_daemon",
            str(Path(jobs_dir).resolve()),
            "--max-concurrent",
            str(max_concurrent),
            "--poll",
            str(poll_interval),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    pid_file.write_text(str(proc.pid), encoding="utf-8")
    return proc.pid


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def stop_scheduler(jobs_dir: str | Path) -> bool:
    """Stop a background scheduler (SIGTERM). Returns True if one was running."""
    pid_file = scheduler_pid_file(jobs_dir)
    if not pid_file.exists():
        return False
    pid_text = pid_file.read_text(encoding="utf-8").strip()
    pid_file.unlink(missing_ok=True)
    if not pid_text.isdigit():
        return False
    try:
        os.kill(int(pid_text), signal.SIGTERM)
    except ProcessLookupError:
        return False
    return True


def scheduler_status(jobs_dir: str | Path) -> dict[str, Any]:
    """Whether a background scheduler is alive, paused, and its PID."""
    pid_file = scheduler_pid_file(jobs_dir)
    if not pid_file.exists():
        return {"running": False, "pid": None, "paused": queue_paused(jobs_dir)}
    pid_text = pid_file.read_text(encoding="utf-8").strip()
    if not pid_text.isdigit():
        return {"running": False, "pid": None, "paused": queue_paused(jobs_dir)}
    pid = int(pid_text)
    return {"running": _pid_alive(pid), "pid": pid, "paused": queue_paused(jobs_dir)}
