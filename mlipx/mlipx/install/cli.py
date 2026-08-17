# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Python-native mlipx installer entry point.

This is the thin-but-complete CLI behind ``scripts/install_mlipx.sh``.  It is
responsible for argument parsing, GPU detection, plan generation, dry-run
rendering, and (optionally) execution — with ``shell=False`` throughout.

The shell wrapper only ensures ``uv`` exists and selects a Python 3.10–3.12
interpreter for this module (so it does not depend on the system ``python3``).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from mlipx.install.hardware import detect_gpus
from mlipx.install.plan import (
    InstallPlanError,
    generate_plan,
    render_plan_shell,
)
from mlipx.install.sources import resolve_source


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mlipx-install",
        description="Install mlipx engines into isolated environments.",
    )
    p.add_argument(
        "--engines",
        default="uma,mace,dpa,grace",
        help="Comma-separated engines to install (default: all four).",
    )
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Target device (default: auto).",
    )
    p.add_argument(
        "--source",
        default="auto",
        choices=["auto", "official", "china", "offline", "custom"],
        help="Package source profile (default: auto -> official).",
    )
    p.add_argument(
        "--python",
        default="3.12",
        help="Python version for the isolated venvs (3.10-3.12).",
    )
    p.add_argument(
        "--clean",
        action="store_true",
        help="Remove each target venv before recreating it.",
    )
    p.add_argument(
        "--skip-doctor",
        action="store_true",
        help="Do not append doctor verify steps to the plan.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan without executing anything.",
    )
    return p


def _existing_venv_python_mismatch(venv: str, requested: str) -> bool:
    """Return True if an existing venv uses a different Python version."""
    py = Path(venv) / "bin" / "python"
    if not py.is_file():
        return False
    try:
        out = subprocess.run(
            [
                str(py),
                "-c",
                "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    if out.returncode != 0:
        return False
    return out.stdout.strip() != requested


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    # GPU detection is done in Python so multi-GPU is handled correctly.
    gpus = detect_gpus()
    if gpus:
        for g in gpus:
            print(
                f"[mlipx] GPU: {g.name} (CC {g.compute_capability}, {g.vram_mib} MiB)"
            )
    else:
        print("[mlipx] No NVIDIA GPU detected (nvidia-smi unavailable).")

    try:
        src = resolve_source(args.source)
        plan = generate_plan(
            gpus=gpus,
            engines=args.engines.split(","),
            source=args.source,
            python_version=args.python,
            device=args.device,
            clean=args.clean,
            verify=not args.skip_doctor,
        )
    except InstallPlanError as e:
        print(f"[mlipx] ERROR: {e}", file=sys.stderr)
        return 2

    for w in plan.warnings:
        print(f"[mlipx] WARNING: {w}", file=sys.stderr)

    # Existing-venv Python mismatch check (fail unless --clean).
    if not args.clean:
        for step in plan.steps:
            if step.stage != "venv":
                continue
            # The venv path is the last argument of `uv venv --python X <path>`.
            try:
                venv = step.argv[-1]
            except IndexError:
                continue
            if _existing_venv_python_mismatch(venv, args.python):
                print(
                    f"[mlipx] ERROR: existing {venv} uses a different Python than "
                    f"{args.python}. Re-run with --clean to recreate it.",
                    file=sys.stderr,
                )
                return 2

    if args.dry_run:
        print()
        print(render_plan_shell(plan))
        return 0

    # Execute
    cwd = Path.cwd()
    env = os.environ.copy()
    env["UV_NO_CONFIG"] = "1"
    env.update(src.env)

    failures = 0
    for step in plan.steps:
        print(f"[mlipx] [{step.stage}] {step.description}")
        step_env = env.copy()
        step_env.update(step.env)
        r = subprocess.run(
            step.argv,
            cwd=cwd,
            env=step_env,
            shell=False,
        )
        if r.returncode != 0:
            print(
                f"[mlipx] FAILED: {step.description} (exit {r.returncode})",
                file=sys.stderr,
            )
            failures += 1
            break

    if failures:
        return 1
    print(f"[mlipx] All {len(plan.steps)} steps completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
