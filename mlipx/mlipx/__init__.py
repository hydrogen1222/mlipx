"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

mlipx - VASP-like interface for MLIP models (UMA, MACE, DPA, GRACE).
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _dist_version


def _package_version() -> str:
    """Package version from project metadata (single source of truth)."""
    try:
        return _dist_version("mlipx")
    except PackageNotFoundError:
        return "0.0.0"


__version__ = _package_version()

__all__ = [
    "BatchRunner",
    "BaseMLIPCalculator",
    "CalculationEngine",
    "CalculatorFactory",
    "IncarConfig",
    "JobManager",
    "MDRunner",
    "OptimizationRunner",
    "ProgressEvent",
    "SinglePointRunner",
    "SUPPORTED_TYPES",
    "UMACalculator",
    "calculate_energy",
    "run_md",
    "run_optimization",
    "run_single_point",
]


def __getattr__(name: str):
    """Lazy import to avoid loading torch/tensorflow backends at import time."""
    _imports = {
        "IncarConfig": ".config",
        "BaseMLIPCalculator": ".base_calculator",
        "UMACalculator": ".calculator",
        "CalculatorFactory": ".calculators.factory",
        "SUPPORTED_TYPES": ".calculators.factory",
        "OptimizationRunner": ".runners.optimization",
        "MDRunner": ".runners.md",
        "BatchRunner": ".runners.batch",
        "CalculationEngine": ".engine",
        "EngineConfig": ".engine",
        "ProgressEvent": ".protocols",
        "JobManager": ".jobs",
        "run_single_point": ".api",
        "run_optimization": ".api",
        "run_md": ".api",
        "calculate_energy": ".api",
    }
    if name in _imports:
        import importlib  # noqa: PLC0415

        mod = importlib.import_module(_imports[name], __package__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
