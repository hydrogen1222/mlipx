# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Modified for the mlipx project: installation infrastructure.

"""
mlipx installation infrastructure — GPU detection, compatibility matrix,
source profiles, and installation plan generation.

This subpackage is deliberately kept free of torch/fairchem/tensorflow
imports so it can run *before* any MLIP backend is installed.
"""

from __future__ import annotations

from mlipx.install.hardware import (
    GpuInfo,
    cc_arch_name,
    classify_gpu,
    detect_gpus,
)
from mlipx.install.compatibility import (
    ArchProfile,
    BackendArchProfile,
    BackendSpec,
    CudaChannel,
    ARCH_PROFILES,
    BACKENDS,
    CUDA_CHANNELS,
    
    select_cuda_channel,
)
from mlipx.install.sources import (
    SourceProfile,
    SOURCE_PROFILES,
    resolve_source,
)
from mlipx.install.plan import (
    InstallPlan,
    InstallStep,
    generate_plan,
)

__all__ = [
    # hardware
    "GpuInfo",
    "cc_arch_name",
    "classify_gpu",
    "detect_gpus",
    # compatibility
    "ArchProfile",
    "BackendArchProfile",
    "BackendSpec",
    "CudaChannel",
    "ARCH_PROFILES",
    "BACKENDS",
    "CUDA_CHANNELS",
    "select_cuda_channel",
    # sources
    "SourceProfile",
    "SOURCE_PROFILES",
    "resolve_source",
    # plan
    "InstallPlan",
    "InstallStep",
    "generate_plan",
]