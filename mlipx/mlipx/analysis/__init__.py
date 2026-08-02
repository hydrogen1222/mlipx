"""Post-processing tools for mlipx molecular-dynamics trajectories.

The core layer depends only on NumPy and ASE.  Rigorous transport statistics
and solid-electrolyte workflows are exposed through optional kinisi and GEMDAT
adapters, respectively.
"""

from __future__ import annotations

from mlipx.analysis.dataset import TrajectoryDataset
from mlipx.analysis.runner import AnalysisRunner, analyze_run

__all__ = ["AnalysisRunner", "TrajectoryDataset", "analyze_run"]
