# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Hardware GPU detection and PyTorch build recommendation (backward-compat).

This module is the **legacy public API** for ``mlipx setup`` and
``mlipx doctor``.  It delegates to :mod:`mlipx.install` internally.

New code should use :mod:`mlipx.install` directly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from mlipx.install.compatibility import (
    BACKENDS,
    CUDA_CHANNELS,
    BackendSpec,
    get_backend_arch_profile,
    select_cuda_channel,
)
from mlipx.install.hardware import (
    MIN_VRAM_MIB_WARN,
    GpuInfo,
    cc_arch_name,
    classify_gpu,
    detect_gpus,
    _pick_oldest,
)
from mlipx.install.sources import resolve_source

if TYPE_CHECKING:
    from collections.abc import Sequence

# ---------------------------------------------------------------------------
# Re-exports (backward-compatible)
# ---------------------------------------------------------------------------
__all__ = [
    "GpuInfo",
    "MIN_VRAM_MIB_WARN",
    "TorchRecommendation",
    "cc_arch_name",
    "detect_gpus",
    "engine_install_commands",
    "format_setup_report",
    "recommend_torch",
    "setup_report_json",
]

# ---------------------------------------------------------------------------
# Legacy TorchRecommendation (kept for doctor.py compatibility)
# ---------------------------------------------------------------------------


@dataclass
class TorchRecommendation:
    """Legacy PyTorch build recommendation for a given compute capability."""

    version: str
    index_url: str
    cu_tag: str
    supported: bool
    rationale: str
    install_commands: list[str] = field(default_factory=list)
    pyproject_snippet: str = ""


def recommend_torch(cc_major: int, cc_minor: int) -> TorchRecommendation:
    """Legacy: recommend a PyTorch build for the given compute capability.

    Kept for backward compatibility with ``mlipx doctor``.
    New code should use :func:`mlipx.install.compatibility.select_cuda_channel`.
    """
    arch = classify_gpu(cc_major, cc_minor)
    sm = f"sm_{cc_major}{cc_minor}"
    arch_name = cc_arch_name(cc_major, cc_minor)

    if arch is None:
        return TorchRecommendation(
            version="",
            index_url="",
            cu_tag="",
            supported=False,
            rationale=(
                f"{arch_name} ({sm}) has NO prebuilt PyTorch wheel. "
                f"Use --device cpu or upgrade to a Maxwell (GTX 900) GPU."
            ),
        )

    # Look up the UMA backend profile for this arch as the default recommendation
    bp = get_backend_arch_profile("uma", arch.name)
    if bp is None:
        return TorchRecommendation(
            version="",
            index_url="",
            cu_tag="",
            supported=False,
            rationale=f"No UMA profile for {arch_name} ({sm}).",
        )

    channel = CUDA_CHANNELS.get(bp.cuda_channel)
    url = channel.pytorch_url if channel else ""
    tag = bp.cuda_channel

    if arch.experimental:
        rationale = (
            f"{arch_name} ({sm}): EXPERIMENTAL.  torch {bp.framework_version}"
            f"+{tag} via legacy CUDA channel.  Upstream does not test this GPU."
        )
    elif bp.mlipx_verified:
        rationale = (
            f"{arch_name} ({sm}): mlipx-verified with torch "
            f"{bp.framework_version}+{tag}."
        )
    else:
        rationale = (
            f"{arch_name} ({sm}): torch {bp.framework_version}+{tag} "
            f"(needs smoke test)."
        )

    return TorchRecommendation(
        version=f"{bp.framework_version}+{tag}",
        index_url=url,
        cu_tag=tag,
        supported=True,
        rationale=rationale,
        install_commands=[
            "# Install / pin the recommended PyTorch build (run from repo root):",
            f"uv pip install torch=={bp.framework_version} --index-url {url}",
        ],
        pyproject_snippet="",
    )


# ---------------------------------------------------------------------------
# Per-engine install commands
# ---------------------------------------------------------------------------


def engine_install_commands(
    gpus: Sequence[GpuInfo] | None, engine: str
) -> list[str]:
    """Return copy-paste install commands for one mlipx engine.

    Delegates to the compatibility matrix in :mod:`mlipx.install.compatibility`.
    """
    name = str(engine).strip().lower()
    if name == "fairchem":
        name = "uma"
    if name not in BACKENDS:
        raise ValueError(f"Unknown engine '{name}'. Choose from: uma, mace, dpa, grace")

    backend = BACKENDS[name]
    has_gpu = gpus is not None and len(gpus) > 0

    if not has_gpu:
        return _cpu_commands(backend)
    else:
        oldest = _pick_oldest(gpus)
        arch = classify_gpu(oldest.cc_major, oldest.cc_minor)
        if arch is None:
            return _cpu_commands(backend)
        return _gpu_commands(backend, arch)


def _cpu_commands(backend: BackendSpec) -> list[str]:
    """CPU-only install commands for one backend."""
    name = backend.venv_name
    if backend.engine == "uma":
        return [
            "# UMA (default engine) — installed by the uv workspace:",
            "uv sync --frozen",
            "uv run mlipx doctor --engine uma --device cpu",
        ]
    if backend.engine == "grace":
        # Use the first arch profile's framework version as reference
        bp = next(iter(backend.arch_profiles.values()))
        extra = " ".join(f'"{p}"' for p in backend.install_extra)
        return [
            "# GRACE — dedicated venv (CPU):",
            f"uv venv --python 3.12 {name}",
            f"uv pip install --no-config --python {name}/bin/python "
            f'-e ./mlipx "tensorflow=={bp.framework_version}" {extra}',
            f"{name}/bin/mlipx doctor --engine grace --device cpu",
        ]

    bp = next(iter(backend.arch_profiles.values()))
    extra = " ".join(backend.install_extra)
    return [
        f"# {backend.label} — dedicated venv (CPU):",
        f"uv venv --python 3.12 {name}",
        f"uv pip install --no-config --python {name}/bin/python "
        f'"torch=={bp.framework_version}" '
        f"--index-url https://download.pytorch.org/whl/cpu",
        f"uv pip install --no-config --python {name}/bin/python "
        f"-e ./mlipx {extra}",
        f"{name}/bin/mlipx doctor --engine {backend.engine} --device cpu",
    ]


def _gpu_commands(backend: BackendSpec, arch) -> list[str]:
    """GPU install commands for one backend on a specific architecture."""
    name = backend.venv_name
    bp = get_backend_arch_profile(backend.engine, arch.name)
    if bp is None:
        return [f"# {backend.label}: no profile for {arch.label} — skipped."]

    status_note = ""
    if bp.status == "experimental":
        status_note = " [EXPERIMENTAL — upstream does not officially support this GPU]"
    elif bp.status == "needs_smoke_test":
        status_note = " [needs smoke test — upstream constraints satisfied, mlipx not yet verified]"

    if backend.engine == "uma":
        return [
            f"# {backend.label} — installed by the uv workspace{status_note}:",
            "uv sync --frozen",
            "uv run mlipx doctor --engine uma --device auto",
        ]

    if backend.engine == "grace":
        tf_ver = bp.framework_version
        extra = " ".join(
            f'"{p}"' for p in backend.install_extra + bp.extra_packages
        )
        return [
            f"# {backend.label} — dedicated venv (TF {tf_ver}){status_note}:",
            f"uv venv --python 3.12 {name}",
            f"uv pip install --no-config --python {name}/bin/python "
            f'-e ./mlipx "tensorflow[and-cuda]=={tf_ver}" {extra}',
            f"{name}/bin/mlipx doctor --engine grace --device auto",
        ]

    # MACE, DPA: torch + backend
    torch_ver = bp.framework_version
    channel = bp.cuda_channel
    ch = CUDA_CHANNELS.get(channel)
    torch_url = ch.pytorch_url if ch else f"https://download.pytorch.org/whl/{channel}"
    extra = " ".join(backend.install_extra)

    return [
        f"# {backend.label} — dedicated venv (torch {torch_ver}+{channel}){status_note}:",
        f"uv venv --python 3.12 {name}",
        f"uv pip install --no-config --python {name}/bin/python "
        f'"torch=={torch_ver}" --index-url {torch_url}',
        f"uv pip install --no-config --python {name}/bin/python "
        f"-e ./mlipx {extra}",
        f"{name}/bin/mlipx doctor --engine {backend.engine} --device auto",
    ]


# ---------------------------------------------------------------------------
# Setup report
# ---------------------------------------------------------------------------


def format_setup_report(gpus: list[GpuInfo] | None) -> str:
    """Format a human-readable GPU setup report.

    Delegates to the compatibility matrix for per-engine recommendations.
    """
    width = 68
    lines: list[str] = []
    lines.append("")
    lines.append("=" * width)
    lines.append(" mlipx GPU Setup — compatibility report")
    lines.append("=" * width)
    lines.append("")

    if not gpus:
        lines.append("  No NVIDIA GPU detected (nvidia-smi unavailable).")
        lines.append("  mlipx can still run with --device cpu.")
        lines.append("")
        lines.append("  Fastest path — one-command installer:")
        lines.append("    ./scripts/install_mlipx.sh --device cpu")
        lines.append("")
        _append_per_engine(lines, gpus)
        lines.append("=" * width)
        return "\n".join(lines)

    oldest = _pick_oldest(gpus)
    arch = classify_gpu(oldest.cc_major, oldest.cc_minor)

    for i, gpu in enumerate(gpus):
        a = classify_gpu(gpu.cc_major, gpu.cc_minor)
        arch_label = a.label if a else "unknown"
        vram_gb = gpu.vram_mib / 1024
        lines.append(f"  GPU {i}: {gpu.name}")
        lines.append(
            f"        Architecture : {arch_label} (CC {gpu.compute_capability}, {gpu.sm})"
        )
        lines.append(f"        VRAM         : {vram_gb:.1f} GB")
        lines.append(f"        Driver       : {gpu.driver_version}")
        if gpu.vram_mib < MIN_VRAM_MIB_WARN:
            lines.append(
                f"        ! VRAM below {MIN_VRAM_MIB_WARN // 1024} GB — small "
                f"systems only."
            )
        if a and a.experimental:
            lines.append(
                f"        ! {a.label} is EXPERIMENTAL — upstream frameworks "
                f"may not include official kernels."
            )
        lines.append("")

    lines.append("-" * width)
    if arch is None:
        lines.append("  NOT SUPPORTED — no prebuilt PyTorch wheel for this GPU.")
        lines.append("  Use --device cpu.")
        lines.append("=" * width)
        return "\n".join(lines)

    lines.append(f"  Architecture profile : {arch.label}")
    lines.append(f"  Compute capability   : {oldest.sm}")
    if arch.experimental:
        lines.append(f"  Status               : EXPERIMENTAL (best-effort only)")
    lines.append("")

    # Per-backend summary
    lines.append("  Compatibility matrix (backend × arch):")
    lines.append("")
    header = f"  {'Backend':<8} {'Version':<12} {'FW':<10} {'CUDA':<8} {'Status':<20}"
    lines.append(header)
    lines.append(f"  {'-'*8} {'-'*12} {'-'*10} {'-'*8} {'-'*20}")
    for engine_name in ("uma", "mace", "dpa", "grace"):
        backend = BACKENDS[engine_name]
        bp = get_backend_arch_profile(engine_name, arch.name)
        if bp is None:
            continue
        lines.append(
            f"  {backend.engine:<8} {backend.version:<12} "
            f"{bp.framework_version:<10} {bp.cuda_channel:<8} "
            f"{bp.status:<20}"
        )
    lines.append("")

    lines.append("  Fastest path — one-command installer:")
    lines.append("    ./scripts/install_mlipx.sh")
    lines.append("")
    _append_per_engine(lines, gpus)

    lines.append("  If the download fails, check your network / proxy settings.")
    lines.append("  Then verify:  uv run mlipx doctor")
    lines.append("=" * width)
    return "\n".join(lines)


def _append_per_engine(lines: list[str], gpus: list[GpuInfo] | None) -> None:
    """Append per-engine copy-paste commands."""
    lines.append("  Per-engine manual commands (copy-paste from repo root):")
    lines.append("")
    for engine_name, label in (
        ("uma", "UMA (FAIRChem)"),
        ("mace", "MACE"),
        ("dpa", "DPA / DeepMD"),
        ("grace", "GRACE"),
    ):
        lines.append(f"  --- {label} ---")
        for cmd in engine_install_commands(gpus, engine_name):
            lines.append(f"    {cmd}")
        lines.append("")


def setup_report_json(gpus: list[GpuInfo] | None) -> dict[str, Any]:
    """Build a JSON-serializable dict of the setup report (for --json)."""
    if not gpus:
        return {"gpus": [], "recommended": None, "has_gpu": False}

    oldest = _pick_oldest(gpus)
    arch = classify_gpu(oldest.cc_major, oldest.cc_minor)
    rec = recommend_torch(oldest.cc_major, oldest.cc_minor)

    result: dict[str, Any] = {
        "has_gpu": True,
        "gpus": [
            {
                "index": i,
                "name": g.name,
                "architecture": cc_arch_name(g.cc_major, g.cc_minor),
                "compute_capability": g.compute_capability,
                "sm": g.sm,
                "driver_version": g.driver_version,
                "vram_mib": g.vram_mib,
                "low_vram": g.vram_mib < MIN_VRAM_MIB_WARN,
            }
            for i, g in enumerate(gpus)
        ],
        "arch_profile": arch.name if arch else None,
        "recommended": {
            "version": rec.version,
            "index_url": rec.index_url,
            "cu_tag": rec.cu_tag,
            "supported": rec.supported,
            "rationale": rec.rationale,
            "install_commands": rec.install_commands,
            "pyproject_snippet": rec.pyproject_snippet,
        }
        if rec
        else None,
        "backends": {},
    }

    if arch:
        for engine_name in ("uma", "mace", "dpa", "grace"):
            backend = BACKENDS[engine_name]
            bp = get_backend_arch_profile(engine_name, arch.name)
            if bp:
                result["backends"][engine_name] = {
                    "version": backend.version,
                    "framework_version": bp.framework_version,
                    "cuda_channel": bp.cuda_channel,
                    "status": bp.status,
                    "upstream_supported": bp.upstream_supported,
                    "mlipx_verified": bp.mlipx_verified,
                    "notes": bp.notes,
                }

    return result
