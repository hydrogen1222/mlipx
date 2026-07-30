# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Modified for the mlipx project: multi-engine MLIP support (UMA/MACE/DPA/GRACE).
"""Environment diagnostic tool for mlipx.

Checks Python, PyTorch, CUDA, GPU compatibility, and model files.
Provides GPU-specific fix commands (works before PyTorch is installed via
``nvidia-smi``). See :mod:`mlipx.gpu_setup` for the detection/recommendation
logic shared with ``mlipx setup``.
"""

from __future__ import annotations

import sys
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING

from packaging.requirements import Requirement

from mlipx.gpu_compat import arch_supports_device
from mlipx.gpu_setup import (
    MIN_VRAM_MIB_WARN,
    cc_arch_name,
    detect_gpus,
    recommend_torch,
)

if TYPE_CHECKING:
    from typing import Any


def _installed_dependency_conflicts(
    distribution_names: tuple[str, ...],
) -> list[str]:
    """Return installed requirements that are not satisfied.

    Import-only checks are insufficient for plugin-style engines: MACE can be
    imported with a newer e3nn and then fail only while unpickling a model.
    Inspecting distribution metadata catches that invalid environment before a
    calculation is submitted.
    """
    conflicts: list[str] = []
    for distribution_name in distribution_names:
        try:
            requirements = metadata.requires(distribution_name) or []
        except metadata.PackageNotFoundError:
            continue
        for raw_requirement in requirements:
            requirement = Requirement(raw_requirement)
            # e3nn is the shared, mutually exclusive UMA/MACE dependency.
            # Other workspace pins (notably the intentional PyTorch override
            # for Pascal GPUs) are diagnosed by their dedicated checks.
            if requirement.name.lower() != "e3nn":
                continue
            if requirement.marker and not requirement.marker.evaluate():
                continue
            try:
                installed = metadata.version(requirement.name)
            except metadata.PackageNotFoundError:
                continue
            if requirement.specifier and installed not in requirement.specifier:
                conflicts.append(
                    f"{distribution_name} requires {requirement.name}"
                    f"{requirement.specifier}, but {installed} is installed"
                )
    return conflicts


def _distribution_installed(distribution_name: str) -> bool:
    """Return whether an installed distribution is visible to this interpreter."""
    try:
        metadata.version(distribution_name)
    except metadata.PackageNotFoundError:
        return False
    return True


def _should_warn_torch_mismatch(
    *,
    uma_installed: bool,
    installed_torch: str | None,
    recommended_torch: str,
) -> bool:
    """Apply the workspace Torch recommendation only to an UMA environment.

    MACE and DPA intentionally use their own Torch builds. Recommending UMA's
    Torch 2.6 inside a DPA environment would break DeepMD's Torch 2.10 ABI
    requirement even when the installed CUDA kernel already supports the GPU.
    """
    return bool(
        uma_installed
        and installed_torch
        and recommended_torch not in installed_torch
    )


def _recommendation_detail(rec, *, installed_torch: str | None = None) -> str:
    """Build a detail string for a GPU from a TorchRecommendation."""
    lines = [rec.rationale]
    if not rec.supported:
        lines.append("")
        lines.append("  Use --device cpu, or upgrade to a Maxwell (GTX 900) GPU.")
        return "\n  ".join(lines)

    lines.append("")
    if installed_torch is not None and installed_torch != rec.version:
        # Normalize for comparison (drop local-version suffix on installed).
        installed_base = installed_torch.split("+")[0]
        rec_base = rec.version.split("+")[0]
        if installed_base != rec_base:
            lines.append(
                f"  Installed torch ({installed_torch}) does NOT match the "
                f"recommended ({rec.version}) for this GPU."
            )
            lines.append("  Switch to the recommended build:")
    else:
        lines.append("  Install / pin the recommended build:")
    for cmd in rec.install_commands:
        lines.append(f"    {cmd}")
    lines.append("")
    lines.append("  Or run:  uv run mlipx setup")
    lines.append("  After fixing, re-run:  uv run mlipx doctor")
    return "\n  ".join(lines)


def run_diagnostics(
    model_path: str | Path | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Run all environment checks.

    Args:
        model_path: Optional path to model checkpoint to check.

    Returns:
        Tuple of (results list, number of failures).
    """
    checks: list[dict[str, Any]] = []
    failures = 0

    # Detect hardware GPUs via nvidia-smi (works without PyTorch).
    hw_gpus = detect_gpus()
    uma_installed = _distribution_installed("fairchem-core")

    # 1. Python version
    py_ver = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    if sys.version_info >= (3, 9):
        checks.append({"name": "Python", "value": py_ver, "status": "ok"})
    else:
        checks.append(
            {
                "name": "Python",
                "value": py_ver,
                "status": "fail",
                "detail": f"Python >= 3.9 required, found {py_ver}",
            }
        )
        failures += 1

    # 2. NVIDIA driver (via nvidia-smi) — checked even before PyTorch.
    if hw_gpus is None:
        checks.append(
            {
                "name": "NVIDIA driver",
                "value": "not found",
                "status": "warn",
                "detail": (
                    "nvidia-smi unavailable. No NVIDIA driver detected.\n"
                    "  Install the NVIDIA driver to use --device cuda,\n"
                    "  or proceed with --device cpu."
                ),
            }
        )
    else:
        driver = hw_gpus[0].driver_version
        checks.append({"name": "NVIDIA driver", "value": driver, "status": "ok"})

    # 3. PyTorch
    try:
        import torch

        torch_ver = torch.__version__
        checks.append({"name": "PyTorch", "value": torch_ver, "status": "ok"})
    except ImportError:
        torch_ver = None
        # GPU-specific install guidance instead of a generic "install torch".
        if hw_gpus:
            rec = recommend_torch(
                min((g.cc_major, g.cc_minor) for g in hw_gpus)[0],
                min((g.cc_major, g.cc_minor) for g in hw_gpus)[1],
            )
            detail = "PyTorch is not installed.\n  " + _recommendation_detail(rec)
        else:
            detail = (
                "PyTorch is not installed.\n"
                "  CPU:  uv pip install torch\n"
                "  CUDA: run `uv run mlipx setup` for GPU-specific guidance."
            )
        checks.append(
            {
                "name": "PyTorch",
                "value": "not installed",
                "status": "fail",
                "detail": detail,
            }
        )
        failures += 1
        # Can't continue without PyTorch
        if model_path:
            checks.append(
                {"name": "Model file", "value": str(model_path), "status": "skip"}
            )
        checks.append(
            {
                "name": "Verdict",
                "value": f"{failures} issue(s) to resolve",
                "status": "fail",
            }
        )
        return checks, failures

    # 4. CUDA
    cuda_available = torch.cuda.is_available()
    cuda_suffix = (
        f" (CUDA {torch.version.cuda})"
        if hasattr(torch.version, "cuda") and torch.version.cuda
        else ""
    )

    if cuda_available:
        checks.append(
            {"name": "CUDA available", "value": f"True{cuda_suffix}", "status": "ok"}
        )
    else:
        detail = (
            "No CUDA GPU available to PyTorch.\n"
            "  CUDA is required for --device cuda; use --device cpu otherwise."
        )
        if hw_gpus:
            detail += (
                "\n  An NVIDIA GPU was detected by nvidia-smi but PyTorch\n"
                "  cannot see it — your PyTorch build may be CPU-only.\n"
                "  Run:  uv run mlipx setup"
            )
        checks.append(
            {
                "name": "CUDA available",
                "value": "False",
                "status": "fail",
                "detail": detail,
            }
        )
        failures += 1

    # 5. GPU compatibility (per CUDA device)
    if cuda_available:
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            props = torch.cuda.get_device_properties(i)
            vram_gb = props.total_memory / (1024**3)
            major, minor = torch.cuda.get_device_capability(i)
            gpu_cc = f"sm_{major}{minor}"

            arch_list = torch.cuda.get_arch_list()
            arch_ok = arch_supports_device(gpu_cc, arch_list)
            arch_label = cc_arch_name(major, minor)
            rec = recommend_torch(major, minor)

            value_str = (
                f"{gpu_name} ({vram_gb:.0f} GB, CC {major}.{minor}, {arch_label})"
            )

            if arch_ok:
                # Kernel works. Still note if the installed torch differs from
                # the recommended one for this GPU.
                if rec.supported and _should_warn_torch_mismatch(
                    uma_installed=uma_installed,
                    installed_torch=torch_ver,
                    recommended_torch=rec.version,
                ):
                    detail = (
                        f"Kernel OK, but installed torch {torch_ver} differs "
                        f"from recommended {rec.version} for this GPU.\n  "
                        + _recommendation_detail(rec, installed_torch=torch_ver)
                    )
                    checks.append(
                        {
                            "name": f"GPU {i}",
                            "value": value_str,
                            "status": "warn",
                            "detail": detail,
                        }
                    )
                else:
                    checks.append(
                        {"name": f"GPU {i}", "value": value_str, "status": "ok"}
                    )
            else:
                detail = (
                    f"Architecture {gpu_cc} — NOT supported by this PyTorch "
                    f"build ({', '.join(arch_list)}).\n  "
                    + _recommendation_detail(rec, installed_torch=torch_ver)
                )
                checks.append(
                    {
                        "name": f"GPU {i}",
                        "value": value_str,
                        "status": "fail",
                        "detail": detail,
                    }
                )
                failures += 1

            # VRAM warning
            vram_mib = props.total_memory // (1024**2)
            if vram_mib < MIN_VRAM_MIB_WARN:
                checks.append(
                    {
                        "name": f"GPU {i} VRAM",
                        "value": f"{vram_gb:.1f} GB",
                        "status": "warn",
                        "detail": (
                            f"Below {MIN_VRAM_MIB_WARN // 1024} GB — small "
                            f"systems only. Use --inference-mode turbo and/or "
                            f"activation checkpointing for larger ones."
                        ),
                    }
                )

            # Triton/compile note for pre-Volta GPUs.
            if major < 7:
                checks.append(
                    {
                        "name": f"GPU {i} compile",
                        "value": "triton unsupported",
                        "status": "warn",
                        "detail": (
                            "Triton (torch.compile) needs CC >= 7.0; turbo MD "
                            "automatically disables compile on this GPU "
                            "(handled by UMACalculator)."
                        ),
                    }
                )
    elif hw_gpus:
        # CUDA not available but hardware detected — show the GPUs + guidance.
        for i, g in enumerate(hw_gpus):
            rec = recommend_torch(g.cc_major, g.cc_minor)
            value_str = f"{g.name} (CC {g.compute_capability}, {cc_arch_name(g.cc_major, g.cc_minor)})"
            checks.append(
                {
                    "name": f"GPU {i}",
                    "value": value_str,
                    "status": "fail",
                    "detail": "  "
                    + _recommendation_detail(rec, installed_torch=torch_ver),
                }
            )
            failures += 1

    # 6. UMA engine. It is optional in a deliberately isolated MACE
    # environment, just as MACE/DPA/GRACE are optional in the UMA environment.
    try:
        import importlib.util

        fc_spec = importlib.util.find_spec("fairchem.core")
        if fc_spec is not None:
            checks.append(
                {
                    "name": "UMA engine (fairchem-core)",
                    "value": "installed",
                    "status": "ok",
                }
            )
        else:
            raise ImportError("fairchem-core not found")
    except ImportError:
        checks.append(
            {
                "name": "UMA engine (fairchem-core)",
                "value": "not installed",
                "status": "skip",
                "detail": (
                    "Optional in a dedicated non-UMA environment. "
                    "For UMA, run `uv sync` from the repository root."
                ),
            }
        )

    # 6b. Optional MLIP engine backends (MACE / DPA / GRACE)
    import importlib.util  # noqa: PLC0415

    for engine_name, module_name, install_cmd in (
        (
            "MACE engine",
            "mace",
            "install in .venv-mace; see uma/docs/README_CN.md or README_EN.md",
        ),
        ("DPA engine", "deepmd", "pip install 'deepmd-kit>=3.0.0'"),
        ("GRACE engine", "tensorpotential", "pip install tensorpotential"),
    ):
        spec = importlib.util.find_spec(module_name)
        if spec is not None:
            try:
                mod = importlib.import_module(module_name)
                ver = getattr(mod, "__version__", "installed")
                checks.append({"name": engine_name, "value": ver, "status": "ok"})
            except Exception:
                checks.append(
                    {"name": engine_name, "value": "installed", "status": "ok"}
                )
        else:
            checks.append(
                {
                    "name": engine_name,
                    "value": "not installed (optional)",
                    "status": "skip",
                    "detail": f"Optional. Install with: {install_cmd}",
                }
            )

    dependency_conflicts = _installed_dependency_conflicts(
        ("mace-torch", "fairchem-core")
    )
    if dependency_conflicts:
        checks.append(
            {
                "name": "Engine dependencies",
                "value": "incompatible",
                "status": "fail",
                "detail": (
                    "\n".join(dependency_conflicts)
                    + "\n  UMA and MACE must use separate virtual environments."
                    "\n  See the Multi-Engine Guide in uma/docs/README_EN.md"
                    " or README_CN.md."
                ),
            }
        )
        failures += 1
    else:
        checks.append(
            {
                "name": "Engine dependencies",
                "value": "compatible",
                "status": "ok",
            }
        )
    # 7. mlipx
    try:
        from mlipx import __version__ as mlipx_ver

        checks.append({"name": "mlipx", "value": f"v{mlipx_ver}", "status": "ok"})
    except ImportError:
        checks.append(
            {
                "name": "mlipx",
                "value": "not installed",
                "status": "fail",
                "detail": "Install: cd uma && uv pip install -e .",
            }
        )
        failures += 1

    # 8. Model file (optional)
    if model_path:
        mp = Path(model_path)
        if mp.exists():
            size_gb = mp.stat().st_size / (1024**3)
            checks.append(
                {
                    "name": "Model file",
                    "value": f"{mp.name} ({size_gb:.1f} GB)",
                    "status": "ok",
                }
            )
        else:
            checks.append(
                {
                    "name": "Model file",
                    "value": f"{mp} — not found",
                    "status": "warn",
                    "detail": "Download from https://fair-chem.github.io/models/uma/",
                }
            )
    else:
        checks.append(
            {
                "name": "Model file",
                "value": "not specified",
                "status": "warn",
                "detail": "Pass --model <path> or specify MODEL_PATH in INCAR",
            }
        )

    # 9. Summary recommendation
    if failures == 0:
        checks.append(
            {
                "name": "Verdict",
                "value": "Ready for CUDA calculations",
                "status": "ok",
            }
        )
    else:
        checks.append(
            {
                "name": "Verdict",
                "value": f"{failures} issue(s) to resolve",
                "status": "fail",
            }
        )

    return checks, failures


def format_diagnostics(checks: list[dict[str, Any]]) -> str:
    """Format diagnostic results as a readable text table.

    Args:
        checks: List of check result dicts from run_diagnostics().

    Returns:
        Formatted string ready for display.
    """
    icons = {"ok": "✓", "fail": "✗", "warn": "!", "skip": "-"}

    lines = []
    lines.append("")
    lines.append("=" * 68)
    lines.append(" mlipx Environment Diagnostic")
    lines.append("=" * 68)
    lines.append("")

    max_name = max(len(c["name"]) for c in checks)

    for c in checks:
        icon = icons.get(c["status"], "?")
        name_padded = c["name"].ljust(max_name)
        value = c.get("value", "")

        if c["status"] == "fail":
            lines.append(f"  {name_padded}  {value:30s}  {icon} FAIL")
        elif c["status"] == "warn":
            lines.append(f"  {name_padded}  {value:30s}  {icon} WARN")
        elif c["status"] == "skip":
            lines.append(f"  {name_padded}  {value:30s}  {icon} SKIP")
        else:
            lines.append(f"  {name_padded}  {value:30s}  {icon}")

        if c.get("detail"):
            for detail_line in c["detail"].split("\n"):
                lines.append(f"    {detail_line}")

    lines.append("")
    lines.append("=" * 68)

    total_fails = sum(1 for c in checks if c["status"] == "fail")
    total_warns = sum(1 for c in checks if c["status"] == "warn")

    if total_fails == 0 and total_warns == 0:
        lines.append(" All checks passed. Ready to run calculations.")
    elif total_fails == 0:
        lines.append(f" All required checks passed ({total_warns} warning(s)).")
    else:
        lines.append(
            f" {total_fails} issue(s) found. Fix before running CUDA calculations."
        )
        lines.append(" After fixing, re-run: uv run mlipx doctor")

    lines.append("=" * 68)

    return "\n".join(lines)
