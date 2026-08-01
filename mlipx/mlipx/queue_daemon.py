"""Detached queue-scheduler entry point (``mlipx queue start``).

Runs :class:`mlipx.queue.QueueScheduler` in a loop, promoting PENDING jobs to
RUNNING up to ``--max-concurrent`` at a time. The daemon itself needs no MLIP
backend: each queued job records its own interpreter and is executed by
:mod:`mlipx.job_worker` under that interpreter.
"""

from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

from mlipx.queue import QueueScheduler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mlipx.queue_daemon", add_help=False)
    parser.add_argument("jobs_dir")
    parser.add_argument("--max-concurrent", type=int, default=1)
    parser.add_argument("--poll", type=float, default=5.0)
    args = parser.parse_args(argv)

    scheduler = QueueScheduler(
        jobs_dir=Path(args.jobs_dir),
        max_concurrent=args.max_concurrent,
        poll_interval=args.poll,
    )

    def _handle_sigterm(signum, frame):  # noqa: ARG001
        scheduler.request_stop()

    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    scheduler.run_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
