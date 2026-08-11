# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Modified for the mlipx project: multi-engine MLIP support (UMA/MACE/DPA/GRACE).
"""Target-aware environment diagnostics for mlipx.

The default command is a side-effect-free package and hardware inventory.
Runtime imports happen only when an engine is selected explicitly. This keeps
optional TensorFlow/DeepMD backends from initialising merely because ``doctor``
was asked to list the current environment.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

from packaging.requirements import Requirement

from mlipx.gpu_compat import arch_supports_device
from mlipx.gpu_setup import (
    MIN_VRAM_MIB_WARN,
    cc_arch_name,
    detect_gpus,
    recommend_torch,
)

_PROBE_SENTINEL = "__MLIPX_DOCTOR_JSON__="

_ENGINE_SPECS: dict[str, dict[str, str]] = {
    "uma": {
        "label": "UMA",
        "distribution": "fairchem-core",
        "module": "fairchem.core",
        "framework": "torch",
        "install": "Run `uv sync` from the repository root.",
    },
    "mace": {
        "label": "MACE",
        "distribution": "mace-torch",
        "module": "mace.calculators",
        "framework": "torch",
        "install": "Install MACE in the dedicated .venv-mace environment.",
    },
    "dpa": {
        "label": "DPA",
        "distribution": "deepmd-kit",
        "module": "deepmd.calculator",
        "framework": "torch",
        "install": "Install deepmd-kit[torch] in the dedicated .venv-dpa environment.",
    },
    "grace": {
        "label": "GRACE",
        "distribution": "tensorpotential",
        "module": "tensorpotential.calculator",
        "framework": "tensorflow",
        "install": "Install tensorpotential in the dedicated .venv-grace environment.",
    },
}


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


def _distribution_version(distribution_name: str) -> str | None:
    """Return an installed distribution version without importing its modules."""
    try:
        return metadata.version(distribution_name)
    except metadata.PackageNotFoundError:
        return None


def _normalize_device(device: str) -> str:
    """Validate and normalize a doctor device target."""
    value = str(device).strip().lower()
    if value == "gpu":
        return "cuda"
    if value in {"auto", "cpu", "cuda"}:
        return value
    if value.startswith("cuda:") and value[5:].isdigit():
        return value
    raise ValueError("device must be auto, cpu, cuda, gpu, or cuda:N")


def _probe_output_tail(stdout: str, stderr: str, *, limit: int = 8) -> str:
    """Return a compact diagnostic tail from a failed isolated probe."""
    lines = [
        line.strip()
        for line in (stderr.splitlines() + stdout.splitlines())
        if line.strip() and not line.startswith(_PROBE_SENTINEL)
    ]
    return "\n".join(lines[-limit:])


def _runtime_probe(
    engine: str,
    device: str,
    *,
    framework: str | None = None,
) -> dict[str, Any]:
    """Import one selected backend in an isolated subprocess.

    TensorFlow, tensorpotential and DeepMD may configure global runtime state
    during import. Capturing that import in a child process keeps ``doctor``
    itself quiet and prevents an inventory check from mutating the caller.
    """
    spec = _ENGINE_SPECS[engine]
    selected_framework = framework or spec["framework"]
    script = f"""
import importlib
import json

payload = {{"ok": False}}
try:
    importlib.import_module({spec['module']!r})
    payload["framework"] = {selected_framework!r}
    if payload["framework"] == "torch":
        import torch
        payload.update(
            torch_version=str(torch.__version__),
            cuda_runtime=str(torch.version.cuda) if torch.version.cuda else None,
            cuda_available=bool(torch.cuda.is_available()),
            device_count=int(torch.cuda.device_count()),
            arch_list=list(torch.cuda.get_arch_list()),
            gpus=[],
        )
        if payload["cuda_available"]:
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                major, minor = torch.cuda.get_device_capability(index)
                payload["gpus"].append({{
                    "name": str(torch.cuda.get_device_name(index)),
                    "major": int(major),
                    "minor": int(minor),
                    "total_memory": int(props.total_memory),
                }})
    else:
        import tensorflow as tf
        physical = list(tf.config.list_physical_devices("GPU"))
        payload.update(
            tensorflow_version=str(tf.__version__),
            cuda_available=bool(physical),
            device_count=len(physical),
            gpus=[{{"name": str(item.name)}} for item in physical],
        )
    payload["ok"] = True
except BaseException as exc:
    payload.update(
        error_type=type(exc).__name__,
        error=str(exc),
    )

print({_PROBE_SENTINEL!r} + json.dumps(payload, sort_keys=True))
raise SystemExit(0 if payload["ok"] else 1)
"""
    env = os.environ.copy()
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    elif device.startswith("cuda:"):
        env["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1]

    try:
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=90,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "error_type": "TimeoutExpired",
            "error": "Backend import did not finish within 90 seconds",
        }
    except OSError as exc:
        return {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    payload_line = next(
        (
            line[len(_PROBE_SENTINEL) :]
            for line in reversed(completed.stdout.splitlines())
            if line.startswith(_PROBE_SENTINEL)
        ),
        None,
    )
    if payload_line is None:
        return {
            "ok": False,
            "error_type": "ProbeProtocolError",
            "error": _probe_output_tail(completed.stdout, completed.stderr)
            or f"Backend probe exited with status {completed.returncode}",
        }
    try:
        payload = json.loads(payload_line)
    except json.JSONDecodeError as exc:
        return {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": f"Invalid backend probe response: {exc}",
        }
    if not payload.get("ok"):
        tail = _probe_output_tail(completed.stdout, completed.stderr)
        if tail:
            payload["detail"] = tail
    return payload


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
    lines.append("  After fixing, repeat the same doctor command.")
    return "\n  ".join(lines)


def run_diagnostics(
    model_path: str | Path | None = None,
    *,
    engine: str = "auto",
    device: str = "auto",
) -> tuple[list[dict[str, Any]], int]:
    """Run inventory checks and, when selected, one engine runtime probe.

    Args:
        model_path: Optional path to model checkpoint to check.
        engine: ``auto`` for package inventory, or one explicit MLIP engine.
        device: ``auto``, ``cpu``, ``cuda``, ``gpu``, or ``cuda:N``.

    Returns:
        Tuple of (results list, number of failures).
    """
    checks: list[dict[str, Any]] = []
    target_engine = str(engine).strip().lower()
    if target_engine == "fairchem":
        target_engine = "uma"
    if target_engine != "auto" and target_engine not in _ENGINE_SPECS:
        raise ValueError("engine must be auto, uma, mace, dpa, or grace")
    requested_device = _normalize_device(device)
    hw_gpus = list(detect_gpus() or [])

    if target_engine == "auto":
        checks.append(
            {
                "name": "Target engine",
                "value": "inventory only",
                "status": "skip",
                "detail": (
                    "No backend is imported in inventory mode. Choose --engine "
                    "uma, mace, dpa, or grace for a runtime check."
                ),
            }
        )
    else:
        checks.append(
            {
                "name": "Target engine",
                "value": _ENGINE_SPECS[target_engine]["label"],
                "status": "ok",
            }
        )

    if requested_device == "auto" and target_engine == "auto":
        resolved_device: str | None = None
        checks.append(
            {
                "name": "Target device",
                "value": "not selected",
                "status": "skip",
                "detail": "Select --engine to resolve auto to CUDA or CPU.",
            }
        )
    else:
        resolved_device = requested_device
        auto_detail = None
        if requested_device == "auto":
            resolved_device = "cuda" if hw_gpus else "cpu"
            auto_detail = (
                "auto selected CUDA because nvidia-smi found an NVIDIA GPU"
                if hw_gpus
                else "auto selected CPU because no NVIDIA GPU was detected"
            )
        checks.append(
            {
                "name": "Target device",
                "value": str(resolved_device).upper(),
                "status": "ok",
                **({"detail": auto_detail} if auto_detail else {}),
            }
        )

    # Python support must match the package metadata, including its upper bound.
    py_ver = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    if (3, 10) <= sys.version_info < (3, 13):
        checks.append({"name": "Python", "value": py_ver, "status": "ok"})
    else:
        checks.append(
            {
                "name": "Python",
                "value": py_ver,
                "status": "fail",
                "detail": f"mlipx requires Python >=3.10,<3.13; found {py_ver}",
            }
        )

    mlipx_version = _distribution_version("mlipx")
    if mlipx_version is None:
        checks.append(
            {
                "name": "mlipx",
                "value": "not installed",
                "status": "fail",
                "detail": "Install with: cd mlipx && python -m pip install -e .",
            }
        )
    else:
        checks.append({"name": "mlipx", "value": f"v{mlipx_version}", "status": "ok"})

    # nvidia-smi is optional for inventory and CPU targets, required for CUDA.
    cuda_requested = bool(resolved_device and resolved_device.startswith("cuda"))
    if hw_gpus:
        checks.append(
            {
                "name": "NVIDIA driver",
                "value": hw_gpus[0].driver_version,
                "status": "ok",
            }
        )
    elif cuda_requested:
        checks.append(
            {
                "name": "NVIDIA driver",
                "value": "not found",
                "status": "fail",
                "detail": (
                    "nvidia-smi found no usable NVIDIA driver or GPU. "
                    "Install a driver, or request --device cpu."
                ),
            }
        )
    else:
        reason = "CPU target" if resolved_device == "cpu" else "inventory mode"
        checks.append(
            {
                "name": "NVIDIA driver",
                "value": f"not required ({reason})",
                "status": "skip",
            }
        )

    cuda_selection_valid = bool(hw_gpus) if cuda_requested else True
    if cuda_requested and resolved_device and resolved_device.startswith("cuda:"):
        requested_index = int(resolved_device.split(":", 1)[1])
        if hw_gpus and requested_index >= len(hw_gpus):
            cuda_selection_valid = False
            checks.append(
                {
                    "name": "CUDA selection",
                    "value": resolved_device,
                    "status": "fail",
                    "detail": (
                        f"nvidia-smi reports {len(hw_gpus)} GPU(s); index "
                        f"{requested_index} is unavailable."
                    ),
                }
            )

    # Inventory uses distribution metadata only; it never imports an engine.
    installed_versions: dict[str, str | None] = {}
    for engine_name, spec in _ENGINE_SPECS.items():
        version = _distribution_version(spec["distribution"])
        installed_versions[engine_name] = version
        selected = engine_name == target_engine
        if version is not None:
            checks.append(
                {
                    "name": f"{spec['label']} package",
                    "value": f"{spec['distribution']} {version}",
                    "status": "ok",
                }
            )
        elif selected:
            checks.append(
                {
                    "name": f"{spec['label']} package",
                    "value": "not installed",
                    "status": "fail",
                    "detail": spec["install"],
                }
            )
        else:
            checks.append(
                {
                    "name": f"{spec['label']} package",
                    "value": "not installed (optional)",
                    "status": "skip",
                    "detail": spec["install"],
                }
            )

    installed_labels = [
        _ENGINE_SPECS[name]["label"]
        for name, version in installed_versions.items()
        if version is not None
    ]
    if len(installed_labels) > 1:
        checks.append(
            {
                "name": "Environment isolation",
                "value": f"multiple engines ({', '.join(installed_labels)})",
                "status": "warn",
                "detail": (
                    "Dedicated environments are recommended because backend "
                    "Torch/CUDA/TensorFlow requirements may conflict."
                ),
            }
        )
    else:
        checks.append(
            {
                "name": "Environment isolation",
                "value": "one or no engine installed",
                "status": "ok",
            }
        )
    # Detect the known mutually incompatible UMA/MACE e3nn combination before
    # importing either backend. It is fatal only when that environment is the
    # object of the check (or when inventory itself claims to be healthy).
    dependency_conflicts = _installed_dependency_conflicts(
        ("mace-torch", "fairchem-core")
    )
    if dependency_conflicts:
        conflict_status = "fail" if target_engine in {"auto", "uma", "mace"} else "warn"
        checks.append(
            {
                "name": "UMA/MACE dependencies",
                "value": "incompatible",
                "status": conflict_status,
                "detail": (
                    "\n".join(dependency_conflicts)
                    + "\n  Keep UMA and MACE in separate virtual environments."
                ),
            }
        )
    else:
        checks.append(
            {
                "name": "UMA/MACE dependencies",
                "value": "compatible",
                "status": "ok",
            }
        )

    torch_ver = _distribution_version("torch")
    if installed_versions["uma"] is not None and torch_ver is not None:
        fairchem_torch_requirement = None
        try:
            fairchem_requirements = metadata.requires("fairchem-core") or []
        except metadata.PackageNotFoundError:
            fairchem_requirements = []
        for raw_requirement in fairchem_requirements:
            requirement = Requirement(raw_requirement)
            if requirement.name.lower() == "torch" and (
                not requirement.marker or requirement.marker.evaluate()
            ):
                fairchem_torch_requirement = requirement
                break
        if (
            fairchem_torch_requirement is not None
            and fairchem_torch_requirement.specifier
            and torch_ver not in fairchem_torch_requirement.specifier
        ):
            checks.append(
                {
                    "name": "UMA dependency profile",
                    "value": "legacy GPU compatibility override",
                    "status": "warn",
                    "detail": (
                        f"Installed torch {torch_ver} does not satisfy FairChem's "
                        f"declared requirement {fairchem_torch_requirement}. The "
                        "workspace override supports legacy GPUs but is not the "
                        "FairChem reference environment. Runtime versions are "
                        "recorded in MD artifacts.json; do not hide this mismatch."
                    ),
                }
            )

    # Runtime imports are explicit and isolated. Package presence alone never
    # earns a green runtime check.
    runtime: dict[str, Any] | None = None
    if target_engine == "auto":
        checks.append(
            {
                "name": "Engine runtime",
                "value": "not imported",
                "status": "skip",
            }
        )
    elif installed_versions[target_engine] is None:
        checks.append(
            {
                "name": "Engine runtime",
                "value": "not checked",
                "status": "skip",
                "detail": "Install the selected engine package first.",
            }
        )
    else:
        assert resolved_device is not None
        probe_framework = _ENGINE_SPECS[target_engine]["framework"]
        if (
            target_engine == "dpa"
            and model_path is not None
            and Path(model_path).suffix.lower() == ".pb"
        ):
            probe_framework = "tensorflow"
        runtime = _runtime_probe(
            target_engine,
            resolved_device,
            framework=probe_framework,
        )
        if runtime.get("ok"):
            checks.append(
                {
                    "name": "Engine runtime",
                    "value": f"{_ENGINE_SPECS[target_engine]['module']} import succeeded",
                    "status": "ok",
                }
            )
            if runtime.get("framework") == "torch":
                framework_value = str(runtime.get("torch_version", "unknown"))
                if runtime.get("cuda_runtime"):
                    framework_value += f" (CUDA {runtime['cuda_runtime']})"
                checks.append(
                    {
                        "name": "PyTorch runtime",
                        "value": framework_value,
                        "status": "ok",
                    }
                )
            else:
                checks.append(
                    {
                        "name": "TensorFlow runtime",
                        "value": str(runtime.get("tensorflow_version", "unknown")),
                        "status": "ok",
                    }
                )
        else:
            error = f"{runtime.get('error_type', 'Error')}: {runtime.get('error', '')}".strip()
            detail = runtime.get("detail")
            if detail:
                error += f"\n{detail}"
            checks.append(
                {
                    "name": "Engine runtime",
                    "value": "import failed",
                    "status": "fail",
                    "detail": error,
                }
            )

    if target_engine != "auto" and resolved_device == "cpu" and runtime:
        if runtime.get("ok"):
            checks.append({"name": "Compute device", "value": "CPU", "status": "ok"})
    elif target_engine != "auto" and cuda_requested:
        if not cuda_selection_valid:
            checks.append(
                {
                    "name": "CUDA runtime",
                    "value": "not checked",
                    "status": "skip",
                    "detail": "Resolve the NVIDIA driver/device selection first.",
                }
            )
        elif runtime and runtime.get("ok"):
            if runtime.get("cuda_available"):
                checks.append(
                    {
                        "name": "CUDA runtime",
                        "value": f"available ({runtime.get('device_count', 0)} visible GPU(s))",
                        "status": "ok",
                    }
                )
            else:
                framework = runtime.get("framework", "selected backend")
                checks.append(
                    {
                        "name": "CUDA runtime",
                        "value": "unavailable",
                        "status": "fail",
                        "detail": (
                            f"{framework} cannot see a CUDA GPU. Check the installed "
                            "framework build and CUDA_VISIBLE_DEVICES, or use --device cpu."
                        ),
                    }
                )

    # PyTorch architecture checks apply only to the explicitly selected
    # PyTorch engine. TensorFlow does not expose an equivalent wheel arch list.
    if (
        target_engine != "auto"
        and cuda_requested
        and runtime
        and runtime.get("ok")
        and runtime.get("framework") == "torch"
        and runtime.get("cuda_available")
    ):
        arch_list = list(runtime.get("arch_list") or [])
        for index, gpu in enumerate(runtime.get("gpus") or []):
            major = int(gpu["major"])
            minor = int(gpu["minor"])
            gpu_cc = f"sm_{major}{minor}"
            vram_gb = int(gpu["total_memory"]) / (1024**3)
            value = (
                f"{gpu['name']} ({vram_gb:.1f} GB, CC {major}.{minor}, "
                f"{cc_arch_name(major, minor)})"
            )
            rec = recommend_torch(major, minor)
            if arch_supports_device(gpu_cc, arch_list):
                if rec.supported and _should_warn_torch_mismatch(
                    uma_installed=target_engine == "uma",
                    installed_torch=str(runtime.get("torch_version") or ""),
                    recommended_torch=rec.version,
                ):
                    checks.append(
                        {
                            "name": f"Runtime GPU {index}",
                            "value": value,
                            "status": "warn",
                            "detail": (
                                f"Kernel supports {gpu_cc}, but this UMA environment "
                                f"uses torch {runtime.get('torch_version')} instead of "
                                f"the workspace recommendation {rec.version}."
                            ),
                        }
                    )
                else:
                    checks.append(
                        {"name": f"Runtime GPU {index}", "value": value, "status": "ok"}
                    )
            else:
                checks.append(
                    {
                        "name": f"Runtime GPU {index}",
                        "value": value,
                        "status": "fail",
                        "detail": (
                            f"PyTorch wheel architectures {arch_list!r} do not support "
                            f"{gpu_cc}.\n  "
                            + _recommendation_detail(
                                rec,
                                installed_torch=str(runtime.get("torch_version") or ""),
                            )
                        ),
                    }
                )
            if int(gpu["total_memory"]) // (1024**2) < MIN_VRAM_MIB_WARN:
                checks.append(
                    {
                        "name": f"Runtime GPU {index} VRAM",
                        "value": f"{vram_gb:.1f} GB",
                        "status": "warn",
                        "detail": (
                            f"Below {MIN_VRAM_MIB_WARN // 1024} GB; use small systems "
                            "and conservative memory settings."
                        ),
                    }
                )

    # A model is optional for environment readiness, but an explicitly supplied
    # path is a required check and therefore fails closed.
    if model_path:
        mp = Path(model_path).expanduser()
        if not mp.exists():
            checks.append(
                {
                    "name": "Model file",
                    "value": f"{mp} — not found",
                    "status": "fail",
                    "detail": "Check the model path and selected engine.",
                }
            )
        else:
            wrong_kind = (target_engine == "grace" and not mp.is_dir()) or (
                target_engine in {"uma", "mace", "dpa"} and not mp.is_file()
            )
            if wrong_kind:
                expected = (
                    "SavedModel directory" if target_engine == "grace" else "model file"
                )
                checks.append(
                    {
                        "name": "Model file",
                        "value": str(mp),
                        "status": "fail",
                        "detail": f"{_ENGINE_SPECS[target_engine]['label']} expects a {expected}.",
                    }
                )
            else:
                value = (
                    f"{mp.name} (directory)"
                    if mp.is_dir()
                    else f"{mp.name} ({mp.stat().st_size / (1024**3):.2f} GB)"
                )
                checks.append({"name": "Model file", "value": value, "status": "ok"})
    else:
        checks.append(
            {
                "name": "Model file",
                "value": "not requested",
                "status": "skip",
                "detail": "Pass --model PATH to validate a model path and its basic type.",
            }
        )

    failures = sum(check["status"] == "fail" for check in checks)
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
    target = next(c for c in checks if c["name"] == "Target engine")
    target_device = next(c for c in checks if c["name"] == "Target device")
    inventory_only = target["value"] == "inventory only"
    warning_label = "warning" if total_warns == 1 else "warnings"

    if inventory_only and total_fails == 0:
        warning_text = f"; {total_warns} {warning_label}" if total_warns else ""
        lines.append(f" Inventory complete{warning_text}; no backend was imported.")
        lines.append(
            " For a runtime check, rerun with --engine ENGINE --device DEVICE "
            "(see --help)."
        )
    elif total_fails == 0:
        warning_text = f" ({total_warns} {warning_label})" if total_warns else ""
        lines.append(
            f" Environment checks passed for {target['value']} on "
            f"{target_device['value']}{warning_text}."
        )
    else:
        failure_label = "check" if total_fails == 1 else "checks"
        scope = (
            "the environment inventory"
            if inventory_only
            else f"{target['value']} on {target_device['value']}"
        )
        lines.append(f" {total_fails} required {failure_label} failed for {scope}.")
        lines.append(" Fix the failures above, then repeat the same doctor command.")

    lines.append("=" * 68)

    return "\n".join(lines)
