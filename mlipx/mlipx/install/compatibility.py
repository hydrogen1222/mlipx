# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Declarative compatibility matrix for mlipx engines.

**Architecture profiles** describe GPU families (Maxwell, Pascal, …).
**CUDA channels** describe PyTorch wheel distribution families (cu126, cu128, …).
**Backend specs** describe each MLIP engine's upstream version, constraints,
and per-architecture profile (framework version, CUDA channel, verification
status).

The two concepts are deliberately separate:

* An architecture profile determines *which CUDA channel* is appropriate
  for a given GPU (e.g. Pascal → cu126 Legacy).
* The CUDA channel determines *which wheel URL* to use.
* The backend spec determines *which framework version* to install, and
  whether that combination has been smoke-tested by mlipx.

Maintenance rule: when adding a new backend version or changing a torch/TF
pin, update this file.  The installer, ``mlipx setup``, and the docs all
derive their information from this single source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# ---------------------------------------------------------------------------
# Architecture profiles (GPU families)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArchProfile:
    """One GPU architecture family.

    Attributes:
        name: Machine-readable key (``"pascal"``, ``"volta"`` …).
        label: Human-readable name (``"Pascal"``).
        cc_min: Lowest compute capability in this family, inclusive.
        cc_max: Highest compute capability in this family, inclusive.
        experimental: If ``True``, mlipx considers this GPU best-effort;
            the upstream frameworks may not ship official kernels for it.
    """

    name: str
    label: str
    cc_min: tuple[int, int]
    cc_max: tuple[int, int]
    experimental: bool = False


ARCH_PROFILES: dict[str, ArchProfile] = {
    "maxwell": ArchProfile(
        name="maxwell",
        label="Maxwell",
        cc_min=(5, 0),
        cc_max=(5, 2),
        experimental=True,
    ),
    "pascal": ArchProfile(
        name="pascal",
        label="Pascal",
        cc_min=(6, 0),
        cc_max=(6, 1),
    ),
    "volta": ArchProfile(
        name="volta",
        label="Volta",
        cc_min=(7, 0),
        cc_max=(7, 0),
    ),
    "turing": ArchProfile(
        name="turing",
        label="Turing",
        cc_min=(7, 5),
        cc_max=(7, 5),
    ),
    "ampere": ArchProfile(
        name="ampere",
        label="Ampere",
        cc_min=(8, 0),
        cc_max=(8, 6),
    ),
    "ada": ArchProfile(
        name="ada",
        label="Ada Lovelace",
        cc_min=(8, 9),
        cc_max=(8, 9),
    ),
    "hopper": ArchProfile(
        name="hopper",
        label="Hopper",
        cc_min=(9, 0),
        cc_max=(9, 0),
    ),
    "blackwell": ArchProfile(
        name="blackwell",
        label="Blackwell",
        cc_min=(10, 0),
        cc_max=(12, 0),
    ),
}


# ---------------------------------------------------------------------------
# CUDA wheel channels (PyTorch distribution families)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CudaChannel:
    """One CUDA wheel distribution channel.

    A channel provides PyTorch wheels for a specific CUDA version.
    It has a known supported SM range (which GPUs can execute its kernels).

    Attributes:
        name: Machine-readable key (``"cu126"``, ``"cu128"`` …).
        label: Human-readable label (``"CUDA 12.6 (Legacy)"``).
        sm_min: Lowest SM the channel's kernels support.
        sm_max: Highest SM the channel's kernels support.
        pytorch_url: Official PyTorch index URL.
        aliyun_url: Aliyun flat mirror URL (for ``--find-links``), or ``""``.
    """

    name: str
    label: str
    sm_min: tuple[int, int]
    sm_max: tuple[int, int]
    pytorch_url: str
    aliyun_url: str = ""


CUDA_CHANNELS: dict[str, CudaChannel] = {
    "cu126": CudaChannel(
        name="cu126",
        label="CUDA 12.6 (Legacy)",
        sm_min=(5, 0),
        sm_max=(9, 0),
        pytorch_url="https://download.pytorch.org/whl/cu126",
        aliyun_url="https://mirrors.aliyun.com/pytorch-wheels/cu126/",
    ),
    "cu128": CudaChannel(
        name="cu128",
        label="CUDA 12.8",
        sm_min=(7, 5),
        sm_max=(12, 0),
        pytorch_url="https://download.pytorch.org/whl/cu128",
        aliyun_url="https://mirrors.aliyun.com/pytorch-wheels/cu128/",
    ),
    "cu130": CudaChannel(
        name="cu130",
        label="CUDA 13.0",
        sm_min=(7, 5),
        sm_max=(12, 0),
        pytorch_url="https://download.pytorch.org/whl/cu130",
        aliyun_url="https://mirrors.aliyun.com/pytorch-wheels/cu130/",
    ),
}


# ---------------------------------------------------------------------------
# Torch version → modern CUDA channel mapping
# ---------------------------------------------------------------------------
# For legacy GPUs (Maxwell/Pascal/Volta) the channel is always cu126.
# For modern GPUs (Turing+) the channel depends on the torch version:
#   torch 2.8.x – 2.10.x  → cu128
#   torch 2.12.x – 2.13.x  → cu130
# This table is exhaustive for the torch versions known to the matrix.


def _torch_modern_cuda(version: str) -> str:
    """Return the modern CUDA channel for a given torch version."""
    major_minor = ".".join(version.split(".")[:2])
    mapping = {
        "2.8": "cu128",
        "2.10": "cu128",
        "2.12": "cu130",
        "2.13": "cu130",
    }
    return mapping.get(major_minor, "cu128")


# ---------------------------------------------------------------------------
# Backend specs
# ---------------------------------------------------------------------------

Status = Literal["verified", "needs_smoke_test", "experimental"]


@dataclass(frozen=True)
class BackendArchProfile:
    """One backend × architecture combination.

    Attributes:
        framework_version: Torch or TF version string (e.g. ``"2.8.0"``).
        cuda_channel: CUDA channel name (``"cu126"``, ``"cu128"`` …).
        upstream_supported: The upstream project declares this combination
            as supported (dependency constraint satisfied, kernel present).
        mlipx_verified: mlipx has actually run a smoke test (SP+MD) on real
            hardware of this architecture family.
        extra_packages: Additional pip packages for this combination
            (e.g. ``("nvidia-cudnn-cu12==9.3.0.75",)``).
        notes: Human-readable rationale.
    """

    @property
    def status(self) -> str:
        """Derived status: verified / needs_smoke_test / experimental."""
        if not self.upstream_supported:
            return "experimental"
        if self.mlipx_verified:
            return "verified"
        return "needs_smoke_test"

    framework_version: str
    cuda_channel: str
    upstream_supported: bool = True
    mlipx_verified: bool = False
    extra_packages: tuple[str, ...] = ()
    notes: str = ""


@dataclass(frozen=True)
class BackendSpec:
    """One MLIP engine backend.

    Attributes:
        engine: Machine-readable key (``"uma"``, ``"mace"``, ``"dpa"``, ``"grace"``).
        label: Human-readable name.
        distribution: pip package name (``"fairchem-core"`` …).
        version: Upstream version string.
        framework: ``"torch"`` or ``"tensorflow"``.
        upstream_constraint: The framework version constraint declared by the
            upstream project (PEP 440 specifier, e.g. ``"torch~=2.8.0"``).
        arch_profiles: Per-architecture-profile entries.
        venv_name: Filesystem name for the isolated venv.
        install_extra: Extra pip packages always installed alongside this
            backend (e.g. ``("e3nn==0.4.4", "mace-torch==0.3.16")``).
    """

    engine: str
    label: str
    distribution: str
    version: str
    framework: str
    upstream_constraint: str
    arch_profiles: dict[str, BackendArchProfile] = field(default_factory=dict)
    venv_name: str = ""
    install_extra: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.venv_name:
            object.__setattr__(self, "venv_name", f".venv-{self.engine}")


# ---------------------------------------------------------------------------
# The matrix
# ---------------------------------------------------------------------------

BACKENDS: dict[str, BackendSpec] = {
    # ── UMA (FAIRChem) ────────────────────────────────────────────────
    "uma": BackendSpec(
        engine="uma",
        label="UMA (FAIRChem)",
        distribution="fairchem-core",
        version="2.21.0",
        framework="torch",
        upstream_constraint="torch~=2.8.0",
        venv_name=".venv",
        arch_profiles={
            "maxwell": BackendArchProfile(
                framework_version="2.8.0",
                cuda_channel="cu126",
                upstream_supported=False,
                mlipx_verified=False,
                notes="Experimental. fairchem-core 2.21.0 → torch~=2.8.0. "
                "Maxwell via cu126 legacy channel; upstream does not test this GPU.",
            ),
            "pascal": BackendArchProfile(
                framework_version="2.8.0",
                cuda_channel="cu126",
                upstream_supported=True,
                mlipx_verified=False,
                notes="torch 2.8.0+cu126 includes sm60 kernel. "
                "fairchem-core 2.21.0 constraint torch~=2.8.0 satisfied.",
            ),
            "volta": BackendArchProfile(
                framework_version="2.8.0",
                cuda_channel="cu126",
                upstream_supported=True,
                mlipx_verified=True,
                notes="Verified on V100 (sm_70). "
                "torch 2.8.0+cu126 includes sm70 kernel.",
            ),
            "turing": BackendArchProfile(
                framework_version="2.8.0",
                cuda_channel="cu128",
                upstream_supported=True,
                mlipx_verified=False,
                notes="Modern CUDA. torch 2.8.0+cu128 for Turing+.",
            ),
            "ampere": BackendArchProfile(
                framework_version="2.8.0",
                cuda_channel="cu128",
                upstream_supported=True,
                mlipx_verified=False,
                notes="Modern CUDA. torch 2.8.0+cu128 for Ampere.",
            ),
            "ada": BackendArchProfile(
                framework_version="2.8.0",
                cuda_channel="cu128",
                upstream_supported=True,
                mlipx_verified=False,
                notes="Modern CUDA. torch 2.8.0+cu128 for Ada (sm_89 via sm_86 compat).",
            ),
            "hopper": BackendArchProfile(
                framework_version="2.8.0",
                cuda_channel="cu128",
                upstream_supported=True,
                mlipx_verified=False,
                notes="Modern CUDA. torch 2.8.0+cu128 for Hopper.",
            ),
            "blackwell": BackendArchProfile(
                framework_version="2.8.0",
                cuda_channel="cu128",
                upstream_supported=True,
                mlipx_verified=False,
                notes="Modern CUDA. torch 2.8.0+cu128 for Blackwell.",
            ),
        },
    ),
    # ── MACE ──────────────────────────────────────────────────────────
    "mace": BackendSpec(
        engine="mace",
        label="MACE",
        distribution="mace-torch",
        version="0.3.16",
        framework="torch",
        upstream_constraint="torch>=1.12",
        install_extra=("e3nn==0.4.4", "mace-torch==0.3.16"),
        arch_profiles={
            "maxwell": BackendArchProfile(
                framework_version="2.8.0",
                cuda_channel="cu126",
                upstream_supported=False,
                mlipx_verified=False,
                notes="Experimental. MACE 0.3.16 has torch>=1.12, no upper bound. "
                "Maxwell via cu126; upstream does not test this GPU.",
            ),
            "pascal": BackendArchProfile(
                framework_version="2.8.0",
                cuda_channel="cu126",
                upstream_supported=True,
                mlipx_verified=False,
                notes="torch 2.8.0+cu126 includes sm60. "
                "MACE 0.3.16 explicitly fixed torch 2.8 compile issues. "
                "Existing 2.6.0+cu124 fallback is also available.",
            ),
            "volta": BackendArchProfile(
                framework_version="2.8.0",
                cuda_channel="cu126",
                upstream_supported=True,
                mlipx_verified=True,
                notes="Verified on V100 (sm_70). "
                "torch 2.8.0+cu126 includes sm70 kernel.",
            ),
            "turing": BackendArchProfile(
                framework_version="2.8.0",
                cuda_channel="cu128",
                upstream_supported=True,
                mlipx_verified=False,
                notes="Modern CUDA. torch 2.8.0+cu128.",
            ),
            "ampere": BackendArchProfile(
                framework_version="2.8.0",
                cuda_channel="cu128",
                upstream_supported=True,
                mlipx_verified=False,
                notes="Modern CUDA. torch 2.8.0+cu128.",
            ),
            "ada": BackendArchProfile(
                framework_version="2.8.0",
                cuda_channel="cu128",
                upstream_supported=True,
                mlipx_verified=False,
                notes="Modern CUDA. torch 2.8.0+cu128.",
            ),
            "hopper": BackendArchProfile(
                framework_version="2.8.0",
                cuda_channel="cu128",
                upstream_supported=True,
                mlipx_verified=False,
                notes="Modern CUDA. torch 2.8.0+cu128.",
            ),
            "blackwell": BackendArchProfile(
                framework_version="2.8.0",
                cuda_channel="cu128",
                upstream_supported=True,
                mlipx_verified=False,
                notes="Modern CUDA. torch 2.8.0+cu128.",
            ),
        },
    ),
    # ── DPA / DeepMD ──────────────────────────────────────────────────
    "dpa": BackendSpec(
        engine="dpa",
        label="DPA (DeepMD-kit)",
        distribution="deepmd-kit",
        version="3.1.3",
        framework="torch",
        upstream_constraint="torch==2.10.0",
        install_extra=("deepmd-kit==3.1.3",),
        arch_profiles={
            "maxwell": BackendArchProfile(
                framework_version="2.10.0",
                cuda_channel="cu126",
                upstream_supported=False,
                mlipx_verified=False,
                notes="Experimental. deepmd-kit 3.1.3 pins torch==2.10.0. "
                "Maxwell via cu126; must be smoke-tested on real hardware.",
            ),
            "pascal": BackendArchProfile(
                framework_version="2.10.0",
                cuda_channel="cu126",
                upstream_supported=True,
                mlipx_verified=False,
                notes="torch 2.10.0+cu126 includes sm60 kernel. "
                "deepmd-kit 3.1.3 pins torch==2.10.0 exactly.",
            ),
            "volta": BackendArchProfile(
                framework_version="2.10.0",
                cuda_channel="cu126",
                upstream_supported=True,
                mlipx_verified=True,
                notes="Verified on V100 (sm_70). "
                "torch 2.10.0+cu126 includes sm70 kernel.",
            ),
            "turing": BackendArchProfile(
                framework_version="2.10.0",
                cuda_channel="cu128",
                upstream_supported=True,
                mlipx_verified=False,
                notes="Modern CUDA. torch 2.10.0+cu128.",
            ),
            "ampere": BackendArchProfile(
                framework_version="2.10.0",
                cuda_channel="cu128",
                upstream_supported=True,
                mlipx_verified=False,
                notes="Modern CUDA. torch 2.10.0+cu128.",
            ),
            "ada": BackendArchProfile(
                framework_version="2.10.0",
                cuda_channel="cu128",
                upstream_supported=True,
                mlipx_verified=False,
                notes="Modern CUDA. torch 2.10.0+cu128.",
            ),
            "hopper": BackendArchProfile(
                framework_version="2.10.0",
                cuda_channel="cu128",
                upstream_supported=True,
                mlipx_verified=False,
                notes="Modern CUDA. torch 2.10.0+cu128.",
            ),
            "blackwell": BackendArchProfile(
                framework_version="2.10.0",
                cuda_channel="cu128",
                upstream_supported=True,
                mlipx_verified=False,
                notes="Modern CUDA. torch 2.10.0+cu128.",
            ),
        },
    ),
    # ── GRACE ─────────────────────────────────────────────────────────
    "grace": BackendSpec(
        engine="grace",
        label="GRACE",
        distribution="tensorpotential",
        version="0.6.0",
        framework="tensorflow",
        upstream_constraint="tensorflow<=2.20",
        install_extra=(
            "tensorpotential==0.6.0",
        ),
        arch_profiles={
            # All GPU archs use TF 2.20.0 because tensorpotential 0.6.0
            # pins tensorflow<=2.20. The only difference is Maxwell:
            # TF 2.20 official wheel build target starts at sm_60, so
            # Maxwell is experimental.
            "maxwell": BackendArchProfile(
                framework_version="2.20.0",
                cuda_channel="",
                upstream_supported=False,
                mlipx_verified=False,
                extra_packages=("nvidia-cudnn-cu12==9.3.0.75",),
                notes="Experimental. TF 2.20 official wheel build target "
                "starts at sm_60 (Pascal); Maxwell sm_50 not in the wheel. "
                "May need source-built TF.",
            ),
            "pascal": BackendArchProfile(
                framework_version="2.20.0",
                cuda_channel="",
                upstream_supported=True,
                mlipx_verified=False,
                extra_packages=("nvidia-cudnn-cu12==9.3.0.75",),
                notes="TF 2.20.0 + CUDA 12.5 + cuDNN 9.3. "
                "Official wheel includes sm_60 target.",
            ),
            "volta": BackendArchProfile(
                framework_version="2.20.0",
                cuda_channel="",
                upstream_supported=True,
                mlipx_verified=True,
                extra_packages=("nvidia-cudnn-cu12==9.3.0.75",),
                notes="Verified on V100 (sm_70). "
                "TF 2.20.0 + CUDA 12.5 + cuDNN 9.3.",
            ),
            "turing": BackendArchProfile(
                framework_version="2.20.0",
                cuda_channel="",
                upstream_supported=True,
                mlipx_verified=False,
                extra_packages=("nvidia-cudnn-cu12==9.3.0.75",),
                notes="TF 2.20.0 + CUDA 12.5 + cuDNN 9.3. "
                "sm75 via sm70 SASS forward-compat.",
            ),
            "ampere": BackendArchProfile(
                framework_version="2.20.0",
                cuda_channel="",
                upstream_supported=True,
                mlipx_verified=False,
                extra_packages=("nvidia-cudnn-cu12==9.3.0.75",),
                notes="TF 2.20.0 + CUDA 12.5 + cuDNN 9.3. "
                "sm86 via sm80 SASS forward-compat.",
            ),
            "ada": BackendArchProfile(
                framework_version="2.20.0",
                cuda_channel="",
                upstream_supported=True,
                mlipx_verified=False,
                extra_packages=("nvidia-cudnn-cu12==9.3.0.75",),
                notes="TF 2.20.0 + CUDA 12.5 + cuDNN 9.3. "
                "sm89 explicitly compiled.",
            ),
            "hopper": BackendArchProfile(
                framework_version="2.20.0",
                cuda_channel="",
                upstream_supported=True,
                mlipx_verified=False,
                extra_packages=("nvidia-cudnn-cu12==9.3.0.75",),
                notes="TF 2.20.0 + CUDA 12.5 + cuDNN 9.3. "
                "compute_90 via PTX JIT.",
            ),
            "blackwell": BackendArchProfile(
                framework_version="2.20.0",
                cuda_channel="",
                upstream_supported=True,
                mlipx_verified=False,
                extra_packages=("nvidia-cudnn-cu12==9.3.0.75",),
                notes="TF 2.20.0 + CUDA 12.5 + cuDNN 9.3. "
                "Blackwell not in TF 2.20 build target; PTX JIT may work.",
            ),
        },
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def select_cuda_channel(
    arch: ArchProfile,
    torch_version: str,
) -> CudaChannel | None:
    """Select the CUDA channel for a given arch profile and torch version.

    Legacy GPUs (Maxwell/Pascal/Volta) always use cu126.
    Modern GPUs (Turing+) use the torch version's modern CUDA channel.

    Args:
        arch: The GPU architecture profile.
        torch_version: Framework version string (e.g. ``"2.8.0"``).

    Returns:
        The matching :class:`CudaChannel`, or ``None`` if no channel
        supports this GPU (should not happen for supported profiles).
    """
    if arch.name in ("maxwell", "pascal", "volta"):
        channel_name = "cu126"
    else:
        channel_name = _torch_modern_cuda(torch_version)
    return CUDA_CHANNELS.get(channel_name)


def get_backend_arch_profile(
    engine: str,
    arch_name: str,
) -> BackendArchProfile | None:
    """Look up the backend×arch profile entry.

    Args:
        engine: ``"uma"``, ``"mace"``, ``"dpa"``, or ``"grace"``.
        arch_name: Architecture profile name (e.g. ``"pascal"``).

    Returns:
        The :class:`BackendArchProfile`, or ``None`` if not found.
    """
    backend = BACKENDS.get(engine)
    if backend is None:
        return None
    return backend.arch_profiles.get(arch_name)