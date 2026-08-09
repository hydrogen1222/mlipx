"""Validated, calculator-independent trajectory analysis for mlipx.

Optional backends such as kinisi and GEMDAT are intentionally not imported at
package import time.
"""

from mlipx.analysis.dataset import TrajectoryDataset
from mlipx.analysis.validation import (
    AnalysisError,
    InvalidTrajectoryError,
    OptionalDependencyError,
    UnsupportedAnalysisError,
    ValidationReport,
    require_analysis,
    validate_trajectory,
)

__all__ = [
    "AnalysisError",
    "InvalidTrajectoryError",
    "OptionalDependencyError",
    "TrajectoryDataset",
    "UnsupportedAnalysisError",
    "ValidationReport",
    "require_analysis",
    "validate_trajectory",
]
