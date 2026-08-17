# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
GPU hardware detection via ``nvidia-smi`` — works *before* PyTorch is installed.

Unlike :mod:`mlipx.gpu_compat` (which does compute-capability *math* against an
already-installed PyTorch's ``arch_list``), this module detects the physical
hardware and maps a GPU's compute capability to an architecture profile.

Support floor: Maxwell (GTX 900 series, sm_50/52; GTX 960 works).
Kepler (GTX 700/600, sm_30/37) has no modern prebuilt PyTorch wheel and is
rejected.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

from mlipx.install.compatibility import ArchProfile, ARCH_PROFILES

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


def _cc_tuple(major: int, minor: int) -> tuple[int, int]:
    return (major, minor)


def classify_gpu(cc_major: int, cc_minor: int) -> ArchProfile | None:
    """Map a GPU compute capability to its architecture profile.

    Args:
        cc_major: Compute-capability major version.
        cc_minor: Compute-capability minor version.

    Returns:
        The matching :class:`ArchProfile`, or ``None`` if the GPU is
        unsupported (Kepler or older).
    """
    cc = (cc_major, cc_minor)
    for profile in ARCH_PROFILES.values():
        if profile.cc_min <= cc <= profile.cc_max:
            return profile
    return None


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


def _pick_oldest(gpus: Sequence[GpuInfo]) -> GpuInfo:
    """Return the GPU with the lowest compute capability (most conservative)."""
    return min(gpus, key=lambda g: (g.cc_major, g.cc_minor))