# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Hardware GPU detection and PyTorch build recommendation.

Unlike :mod:`mlipx.gpu_compat` (which does compute-capability *math* against an
already-installed PyTorch's ``arch_list``), this module detects the physical
hardware via ``nvidia-smi`` — so it works *before* PyTorch is installed — and
maps a GPU's compute capability to the right prebuilt PyTorch wheel.

Support floor: Maxwell (GTX 900 series, sm_50/52; GTX 960 works). Kepler
(GTX 700/600, sm_30/37) has no modern prebuilt PyTorch wheel and is rejected.

The recommendation rule:
  * sm 5.x–9.x (Maxwell → Hopper) → ``torch==2.6.0+cu124`` (torch 2.7+ dropped
    sm_50/sm_60; sm_50/sm_60 kernels are binary-compatible with sm_52/sm_61,
    and sm_86 covers Ada/RTX 40 sm_89).
  * sm 10.x/12.x (Blackwell, RTX 50) → ``torch==2.8.0+cu128`` (2.6 has no
    sm_100/120 kernel).
  * sm < 5 (Kepler) → unsupported (no prebuilt wheel; source build only).
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

# --- recommended PyTorch builds (per architecture family) -------------------

# Covers Maxwell(sm_50/52) through Hopper(sm_90). sm_86 in this wheel is
# binary-compatible with Ada/RTX 40 (sm_89). torch 2.7+ removed sm_50/sm_60,
# so old cards MUST stay on 2.6.x.
TORCH_2_6 = ("2.6.0+cu124", "https://download.pytorch.org/whl/cu124", "cu124")
# Blackwell (sm_100/120) needs torch 2.8+; 2.6 has no Blackwell kernel.
TORCH_2_8 = ("2.8.0+cu128", "https://download.pytorch.org/whl/cu128", "cu128")

# Minimum VRAM (MiB) to comfortably run the UMA-s model (~1.1 GB) on small
# systems. Below this we warn (still allowed).
MIN_VRAM_MIB_WARN = 2048


@dataclass
class GpuInfo:
    """One physical CUDA GPU as reported by ``nvidia-smi``."""

    name: str
    cc_major: int
    cc_minor: int
    driver_version: str
    vram_mib: int

    @property
    def compute_capability(self) -> str:
        """Compute capability as ``X.Y`` (e.g. ``"6.1"``)."""
        return f"{self.cc_major}.{self.cc_minor}"

    @property
    def sm(self) -> str:
        """Compute capability as ``sm_XY`` (e.g. ``"sm_61"``)."""
        return f"sm_{self.cc_major}{self.cc_minor}"


@dataclass
class TorchRecommendation:
    """The PyTorch build recommended for a given compute capability."""

    version: str
    index_url: str
    cu_tag: str
    supported: bool
    rationale: str
    install_commands: list[str] = field(default_factory=list)
    pyproject_snippet: str = ""


def cc_arch_name(major: int, minor: int) -> str:
    """Map a compute capability to a human architecture name.

    Args:
        major: Compute-capability major version.
        minor: Compute-capability minor version.

    Returns:
        Architecture family name (e.g. ``"Pascal"``).
    """
    if major == 3:
        return "Kepler"
    if major == 5:
        return "Maxwell"
    if major == 6:
        return "Pascal"
    if major == 7:
        return "Volta" if minor == 0 else "Turing"
    if major == 8:
        if minor == 9:
            return "Ada Lovelace"
        return "Ampere"
    if major == 9:
        return "Hopper"
    if major in (10, 12):
        return "Blackwell"
    return f"unknown (sm_{major}{minor})"


def recommend_torch(cc_major: int, cc_minor: int) -> TorchRecommendation:
    """Recommend a prebuilt PyTorch build for the given compute capability.

    Args:
        cc_major: Compute-capability major version.
        cc_minor: Compute-capability minor version.

    Returns:
        :class:`TorchRecommendation` with install commands and rationale.
    """
    arch = cc_arch_name(cc_major, cc_minor)
    sm = f"sm_{cc_major}{cc_minor}"

    # Kepler (GTX 700/600) — no modern prebuilt torch wheel.
    if cc_major < 5:
        return TorchRecommendation(
            version="",
            index_url="",
            cu_tag="",
            supported=False,
            rationale=(
                f"{arch} ({sm}, GTX 700/600 series) has NO prebuilt PyTorch "
                f"wheel — PyTorch dropped Kepler support years ago. Options: "
                f"build PyTorch from source (very hard), use --device cpu, or "
                f"upgrade to a Maxwell (GTX 900) or newer GPU."
            ),
            install_commands=[],
            pyproject_snippet="",
        )

    # Blackwell (RTX 50, sm_100/120) — torch 2.6 lacks the kernel.
    if cc_major >= 10:
        version, index_url, cu_tag = TORCH_2_8
        rationale = (
            f"{arch} ({sm}) requires torch 2.8+ — torch 2.6 has no "
            f"sm_100/sm_120 kernel. Note: this overrides the workspace default "
            f"torch==2.6.0+cu124 pin."
        )
    else:
        # Maxwell → Hopper (sm 5.x–9.x): torch 2.6.0+cu124.
        version, index_url, cu_tag = TORCH_2_6
        if cc_major in (5, 6):
            rationale = (
                f"{arch} ({sm}): torch 2.7+ dropped sm_50/sm_60 from prebuilt "
                f"wheels, so use torch 2.6.0+cu124 which still ships sm_50/sm_60 "
                f"(binary-compatible with sm_52/sm_61)."
            )
        else:
            rationale = (
                f"{arch} ({sm}): torch 2.6.0+cu124 ships matching kernels and "
                f"is the workspace default. Works out of the box with `uv sync`."
            )

    install_commands = [
        "# Install / pin the recommended PyTorch build (run from repo root):",
        f"uv pip install torch=={version} --index-url {index_url}",
    ]
    pyproject_snippet = (
        f"# [tool.uv] override in pyproject.toml (covers `uv sync`):\n"
        f'override-dependencies = ["torch=={version}"]\n\n'
        f"[[tool.uv.index]]\n"
        f'name = "pytorch-{cu_tag}"\n'
        f'url = "{index_url}"\n'
        f"explicit = true\n\n"
        f"[tool.uv.sources]\n"
        f'torch = {{ index = "pytorch-{cu_tag}" }}'
    )

    return TorchRecommendation(
        version=version,
        index_url=index_url,
        cu_tag=cu_tag,
        supported=True,
        rationale=rationale,
        install_commands=install_commands,
        pyproject_snippet=pyproject_snippet,
    )


def detect_gpus() -> list[GpuInfo] | None:
    """Detect CUDA GPUs via ``nvidia-smi`` (no PyTorch needed).

    Returns:
        List of :class:`GpuInfo` (one per GPU), or ``None`` if ``nvidia-smi``
        is unavailable (no NVIDIA driver installed). Never raises.
    """
    query = (
        "name,compute_cap,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    )
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={query[0]}",
                *query[1:],
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None

    if result.returncode != 0:
        return None

    gpus: list[GpuInfo] = []
    # Parse with the csv module: nvidia-smi quotes fields that contain commas
    # (some GPU names do), so a naive ``line.split(",")`` corrupts them.
    import csv as _csv
    import io as _io

    for parts in _csv.reader(_io.StringIO(result.stdout)):
        parts = [p.strip() for p in parts]
        if not parts or all(p == "" for p in parts):
            continue
        if len(parts) < 4:
            continue
        name, cc, driver, vram = parts[0], parts[1], parts[2], parts[3]
        try:
            cc_major_str, _, cc_minor_str = cc.partition(".")
            cc_major = int(cc_major_str)
            cc_minor = int(cc_minor_str) if cc_minor_str else 0
            vram_mib = int(vram)
        except ValueError:
            continue
        gpus.append(
            GpuInfo(
                name=name,
                cc_major=cc_major,
                cc_minor=cc_minor,
                driver_version=driver,
                vram_mib=vram_mib,
            )
        )

    return gpus or None


def _pick_recommendation(gpus: Sequence[GpuInfo]) -> TorchRecommendation | None:
    """Pick the recommendation for the oldest GPU (most conservative)."""
    if not gpus:
        return None
    # The GPU with the smallest compute capability dictates the torch build
    # (older GPUs need the older torch; a newer torch that drops old kernels
    # would break them).
    oldest = min(gpus, key=lambda g: (g.cc_major, g.cc_minor))
    return recommend_torch(oldest.cc_major, oldest.cc_minor)


def format_setup_report(gpus: list[GpuInfo] | None) -> str:
    """Format a human-readable GPU setup report.

    Args:
        gpus: Output of :func:`detect_gpus` (or ``None``).

    Returns:
        Multi-line report string with per-GPU info and install commands.
    """
    width = 68
    lines: list[str] = []
    lines.append("")
    lines.append("=" * width)
    lines.append(" mlipx GPU Setup — PyTorch install guidance")
    lines.append("=" * width)
    lines.append("")

    if not gpus:
        lines.append("  No NVIDIA GPU detected (nvidia-smi unavailable).")
        lines.append("  mlipx can still run with --device cpu.")
        lines.append("")
        lines.append("=" * width)
        return "\n".join(lines)

    for i, gpu in enumerate(gpus):
        arch = cc_arch_name(gpu.cc_major, gpu.cc_minor)
        vram_gb = gpu.vram_mib / 1024
        rec = recommend_torch(gpu.cc_major, gpu.cc_minor)
        lines.append(f"  GPU {i}: {gpu.name}")
        lines.append(
            f"        Architecture : {arch} (CC {gpu.compute_capability}, {gpu.sm})"
        )
        lines.append(f"        VRAM         : {vram_gb:.1f} GB")
        lines.append(f"        Driver       : {gpu.driver_version}")
        if gpu.vram_mib < MIN_VRAM_MIB_WARN:
            lines.append(
                f"        ! VRAM below {MIN_VRAM_MIB_WARN // 1024} GB — small "
                f"systems only; use --inference-mode turbo / activation "
                f"checkpointing for larger ones."
            )
        lines.append(
            f"        Recommended  : torch=={rec.version}"
            if rec.supported
            else "        Recommended  : (unsupported)"
        )
        lines.append("")

    rec = _pick_recommendation(gpus)
    lines.append("-" * width)
    if rec is None or not rec.supported:
        lines.append("  NOT SUPPORTED by any prebuilt PyTorch wheel.")
        if rec is not None:
            for ln in rec.rationale.split("\n"):
                lines.append(f"  {ln}")
        lines.append("")
        lines.append("  Use --device cpu, or upgrade to a Maxwell (GTX 900) GPU.")
        lines.append("=" * width)
        return "\n".join(lines)

    lines.append(f"  Recommended PyTorch: torch=={rec.version} ({rec.cu_tag})")
    for ln in rec.rationale.split("\n"):
        lines.append(f"  {ln}")
    lines.append("")
    lines.append("  Install / pin it (from repo root):")
    for cmd in rec.install_commands:
        lines.append(f"    {cmd}")
    lines.append("")
    lines.append("  Or paste this override into pyproject.toml [tool.uv]:")
    for ln in rec.pyproject_snippet.split("\n"):
        lines.append(f"    {ln}")
    lines.append("")
    lines.append("  If the download fails, enable a proxy first:  clashctl on")
    lines.append("  Then verify:  uv run mlipx doctor")
    lines.append("=" * width)

    return "\n".join(lines)


def setup_report_json(gpus: list[GpuInfo] | None) -> dict:
    """Build a JSON-serializable dict of the setup report (for --json)."""
    if not gpus:
        return {"gpus": [], "recommended": None, "has_gpu": False}
    rec = _pick_recommendation(gpus)
    return {
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
    }
