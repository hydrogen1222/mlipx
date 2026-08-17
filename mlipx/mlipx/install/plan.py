# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Installation plan generation.

Given a set of detected GPUs, a list of requested engines, and a source
profile, this module produces a :class:`InstallPlan` — an ordered list of
shell commands that create the isolated venvs and install each backend.

The plan is serialisable to JSON so a thin shell wrapper can execute it
without re-implementing any logic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

from mlipx.install.compatibility import (
    BACKENDS,
    BackendSpec,
    get_backend_arch_profile,
)
from mlipx.install.hardware import GpuInfo, classify_gpu
from mlipx.install.sources import SourceProfile, resolve_source


@dataclass
class InstallStep:
    """One step in an installation plan.

    Attributes:
        stage: Human-readable stage name (``"venv"``, ``"pip"``, ``"verify"``).
        description: One-line description for logging.
        command: Shell command to execute (may be multi-line).
        env: Extra environment variables to set for this step.
    """

    stage: str
    description: str
    command: str
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class InstallPlan:
    """Complete installation plan.

    Attributes:
        gpu_arch: Detected architecture name (or ``"cpu"``).
        gpu_cc: Compute capability string (or ``""``).
        source: Resolved source profile name.
        python_version: Python version for the isolated venvs.
        steps: Ordered list of :class:`InstallStep`.
        warnings: Human-readable warnings to show before execution.
    """

    gpu_arch: str
    gpu_cc: str
    source: str
    python_version: str
    steps: list[InstallStep] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def generate_plan(
    gpus: Sequence[GpuInfo] | None,
    engines: Sequence[str] = ("uma", "mace", "dpa", "grace"),
    *,
    source: str = "auto",
    python_version: str = "3.12",
    device: str = "auto",
) -> InstallPlan:
    """Generate an installation plan for the given hardware and engines.

    Args:
        gpus: Detected GPUs (``None`` or empty = CPU-only).
        engines: Backend engines to install (``"uma"``, ``"mace"``,
            ``"dpa"``, ``"grace"``).
        source: Source profile name (``"auto"``, ``"official"``,
            ``"china"``, ``"offline"``, ``"custom"``).
        python_version: Python version for the isolated venvs.
        device: ``"auto"``, ``"cuda"``, or ``"cpu"``.

    Returns:
        An :class:`InstallPlan` ready for execution.
    """
    src = resolve_source(source)
    is_cpu = device == "cpu" or not gpus

    # Determine architecture
    if is_cpu or not gpus:
        arch_name = "cpu"
        cc_str = ""
        arch_label = "CPU"
    else:
        oldest = min(gpus, key=lambda g: (g.cc_major, g.cc_minor))
        arch = classify_gpu(oldest.cc_major, oldest.cc_minor)
        if arch is None:
            # Unsupported GPU (Kepler).  Fall back to CPU with a warning.
            arch_name = "cpu"
            cc_str = f"{oldest.cc_major}.{oldest.cc_minor}"
            arch_label = f"unsupported (CC {cc_str})"
        else:
            arch_name = arch.name
            cc_str = f"{oldest.cc_major}.{oldest.cc_minor}"
            arch_label = f"{arch.label} (CC {cc_str})"

    plan = InstallPlan(
        gpu_arch=arch_name,
        gpu_cc=cc_str,
        source=src.name,
        python_version=python_version,
    )

    if arch_name == "cpu" and not is_cpu:
        plan.warnings.append(
            f"GPU compute capability {cc_str} is unsupported by modern "
            f"PyTorch wheels.  Falling back to CPU-only installation."
        )

    # Build steps for each engine
    for engine_name in engines:
        engine_name = str(engine_name).strip().lower()
        if engine_name == "fairchem":
            engine_name = "uma"
        if engine_name not in BACKENDS:
            plan.warnings.append(f"Unknown engine '{engine_name}' — skipped.")
            continue

        backend = BACKENDS[engine_name]
        if arch_name == "cpu":
            _add_cpu_steps(plan, backend, src, python_version)
        else:
            _add_gpu_steps(plan, backend, arch_name, src, python_version, device)

    return plan


def _add_cpu_steps(
    plan: InstallPlan,
    backend: BackendSpec,
    src: SourceProfile,
    python_version: str,
) -> None:
    """Add CPU-only install steps for one backend."""
    name = backend.venv_name
    plan.steps.append(
        InstallStep(
            stage="venv",
            description=f"Create {name} for {backend.label}",
            command=f"uv venv --python {python_version} {name}",
        )
    )

    if backend.engine == "uma":
        # UMA: use uv sync --frozen (workspace default)
        plan.steps.append(
            InstallStep(
                stage="pip",
                description=f"Install {backend.label} (UMA workspace sync)",
                command="uv sync --frozen",
            )
        )
    elif backend.engine == "grace":
        # GRACE CPU: plain tensorflow (no [and-cuda])
        extra = " ".join(backend.install_extra)
        plan.steps.append(
            InstallStep(
                stage="pip",
                description=f"Install {backend.label} (CPU)",
                command=(
                    f"uv pip install --no-config --python {name}/bin/python "
                    f'-e ./mlipx "tensorflow=={backend.arch_profiles.get("volta", next(iter(backend.arch_profiles.values()))).framework_version}" '
                    f'"{extra}"'
                ),
            )
        )
    else:
        # MACE, DPA: install torch CPU + backend
        bp = next(iter(backend.arch_profiles.values()))
        plan.steps.append(
            InstallStep(
                stage="pip",
                description=f"Install torch (CPU) for {backend.label}",
                command=(
                    f"uv pip install --no-config --python {name}/bin/python "
                    f'"torch=={bp.framework_version}" '
                    f"--index-url https://download.pytorch.org/whl/cpu"
                ),
            )
        )
        extra = " ".join(backend.install_extra)
        plan.steps.append(
            InstallStep(
                stage="pip",
                description=f"Install {backend.label}",
                command=(
                    f"uv pip install --no-config --python {name}/bin/python "
                    f'-e ./mlipx {extra}'
                ),
            )
        )

    plan.steps.append(
        InstallStep(
            stage="verify",
            description=f"Verify {backend.label}",
            command=(
                f"{name}/bin/mlipx doctor --engine {backend.engine} --device cpu"
            ),
        )
    )


def _add_gpu_steps(
    plan: InstallPlan,
    backend: BackendSpec,
    arch_name: str,
    src: SourceProfile,
    python_version: str,
    device: str,
) -> None:
    """Add GPU install steps for one backend."""
    name = backend.venv_name
    bp = get_backend_arch_profile(backend.engine, arch_name)

    if bp is None:
        plan.warnings.append(
            f"No profile for {backend.label} on {arch_name}.  Skipping."
        )
        return

    if bp.status == "experimental":
        plan.warnings.append(
            f"{backend.label} on {arch_name}: EXPERIMENTAL — "
            f"upstream does not officially support this GPU. {bp.notes}"
        )
    elif bp.status == "needs_smoke_test":
        plan.warnings.append(
            f"{backend.label} on {arch_name}: needs smoke test — "
            f"upstream constraints are satisfied but mlipx has not yet "
            f"verified this combination on real hardware."
        )

    plan.steps.append(
        InstallStep(
            stage="venv",
            description=f"Create {name} for {backend.label}",
            command=f"uv venv --python {python_version} {name}",
        )
    )

    if backend.engine == "uma":
        # UMA: use uv sync --frozen
        plan.steps.append(
            InstallStep(
                stage="pip",
                description=f"Install {backend.label} (workspace sync)",
                command="uv sync --frozen",
            )
        )
    elif backend.engine == "grace":
        # GRACE: tensorflow[and-cuda] + tensorpotential + cuDNN
        tf_ver = bp.framework_version
        extra_pkgs = " ".join(
            f'"{p}"' for p in backend.install_extra + bp.extra_packages
        )
        plan.steps.append(
            InstallStep(
                stage="pip",
                description=f"Install {backend.label} (TF {tf_ver})",
                command=(
                    f"uv pip install --no-config --python {name}/bin/python "
                    f'-e ./mlipx "tensorflow[and-cuda]=={tf_ver}" '
                    f"{extra_pkgs}"
                ),
            )
        )
    else:
        # MACE, DPA: torch + backend
        torch_ver = bp.framework_version
        channel = bp.cuda_channel

        if src.pytorch_find_links:
            # Use find-links (Aliyun flat mirror)
            find_links = src.pytorch_find_links.format(cuda_tag=channel)
            torch_cmd = (
                f"uv pip install --no-config --python {name}/bin/python "
                f'"torch=={torch_ver}" '
                f"--find-links {find_links}"
            )
        elif src.pytorch_index:
            torch_cmd = (
                f"uv pip install --no-config --python {name}/bin/python "
                f'"torch=={torch_ver}" '
                f"--index-url {src.pytorch_index}"
            )
        else:
            # Official: use pytorch.org CUDA channel
            from mlipx.install.compatibility import CUDA_CHANNELS

            ch = CUDA_CHANNELS.get(channel)
            torch_url = ch.pytorch_url if ch else f"https://download.pytorch.org/whl/{channel}"
            torch_cmd = (
                f"uv pip install --no-config --python {name}/bin/python "
                f'"torch=={torch_ver}" '
                f"--index-url {torch_url}"
            )

        plan.steps.append(
            InstallStep(
                stage="pip",
                description=f"Install torch {torch_ver}+{channel} for {backend.label}",
                command=torch_cmd,
            )
        )

        extra = " ".join(backend.install_extra)
        plan.steps.append(
            InstallStep(
                stage="pip",
                description=f"Install {backend.label}",
                command=(
                    f"uv pip install --no-config --python {name}/bin/python "
                    f'-e ./mlipx {extra}'
                ),
            )
        )

    plan.steps.append(
        InstallStep(
            stage="verify",
            description=f"Verify {backend.label}",
            command=(
                f"{name}/bin/mlipx doctor --engine {backend.engine} "
                f"--device {device}"
            ),
        )
    )


def plan_to_json(plan: InstallPlan) -> str:
    """Serialize an :class:`InstallPlan` to JSON."""
    return json.dumps(
        {
            "gpu_arch": plan.gpu_arch,
            "gpu_cc": plan.gpu_cc,
            "source": plan.source,
            "python_version": plan.python_version,
            "warnings": plan.warnings,
            "steps": [
                {
                    "stage": s.stage,
                    "description": s.description,
                    "command": s.command,
                    "env": s.env,
                }
                for s in plan.steps
            ],
        },
        indent=2,
        ensure_ascii=False,
    )