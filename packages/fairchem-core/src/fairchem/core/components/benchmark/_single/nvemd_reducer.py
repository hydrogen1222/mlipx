"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import os
from glob import glob

import numpy as np
import pandas as pd

from fairchem.core.components.benchmark._benchmark_reducer import BenchmarkReducer
from fairchem.core.components.calculate import NVEMDRunner
from fairchem.core.components.calculate._single.nve_md_runner import get_thermo


def moving_avg(x, window=20):
    all_x = []
    for i in range(window):
        all_x.append(x[i : len(x) - (window - i)])  # noqa: PERF401
    return np.stack(all_x).mean(axis=0)


def get_te_drift(filename):
    time, Et = get_thermo(filename)
    total_step = len(time)
    # remove 10% for equilibration
    equil_time = int(total_step * 0.1)
    Et = Et[equil_time:]
    total_time = time[-1] - time[equil_time]
    Et = moving_avg(Et, window=20)
    # The thermo log is written by ASE's MDLogger(peratom=True): energies are
    # already divided by the atom count (eV/atom) and the time column is
    # already in ps (MDLogger divides the simulation time by 1000*units.fs).
    # The drift rate is therefore 1000 * |dE| / dt with dE in eV/atom and dt in
    # ps, which yields meV/(atom*ps) -- no extra normalisation needed.
    #
    # Compare the FIRST smoothed value (Et[0], the start of the
    # post-equilibration window) with the LAST (Et[-1]). The previous code
    # indexed ``Et[equil_time]`` *after* the moving-average truncation, which
    # points at the wrong frame (around 0.2*N rather than 0.1*N) and returns
    # an incorrect drift magnitude.
    if Et.size == 0 or total_time <= 0:
        return 0.0
    drift = 1000 * np.abs(Et[-1] - Et[0]) / total_time
    return float(drift)


class NVEMDReducer(BenchmarkReducer):
    def __init__(
        self,
        benchmark_name: str,
    ):
        """
        Args:
            benchmark_name: Name of the benchmark, used for file naming
            target_data_path: Path to the target data JSON file
            index_name: Optional name of the column to use as index
            corrections: Optional correction class to apply to all entries
            max_error_threshold: Maximum allowed mean absolute formation energy per atom error threshold
        """
        self.benchmark_name = benchmark_name

    @property
    def runner_type(self) -> type[NVEMDRunner]:
        """The runner type this reducer is associated with."""
        return NVEMDRunner

    def join_results(self, results_dir: str, glob_pattern: str) -> pd.DataFrame:
        """Join results from multiple JSON files into a single DataFrame.

        Args:
            results_dir: Directory containing result files
            glob_pattern: Pattern to match result files

        Returns:
            Combined DataFrame containing all results
        """
        results = [
            get_te_drift(f) for f in glob(os.path.join(results_dir, glob_pattern))
        ]
        return results

    def save_results(self, results: list, results_dir: str) -> None:
        """Save joined results to a compressed json file

        Args:
            results:  results: Combined results from join_results
            results_dir: Directory containing result files
        """
        with open(
            os.path.join(results_dir, f"{self.benchmark_name}_results.txt"), "w"
        ) as f:
            f.write("\n".join([str(x) for x in results]))

    def compute_metrics(self, results: list, run_name: str) -> pd.DataFrame:
        """Compute Matbench discovery metrics for relaxed energy and structure predictions.

        Args:
            results: DataFrame containing prediction results with energy values
            run_name: Identifier for the current evaluation run

        Returns:
            DataFrame containing computed metrics for different material subsets
        """
        metrics = {
            "drift_mev_per_atom_ps": float(np.mean(results)),
        }
        return pd.DataFrame([metrics], index=[run_name])

    def save_metrics(self, metrics: pd.DataFrame, results_dir: str) -> None:
        """Save computed metrics to a compressed JSON file.

        Args:
            metrics: DataFrame containing the computed metrics
            results_dir: Directory where metrics will be saved
        """
        metrics.to_json(
            os.path.join(results_dir, f"{self.benchmark_name}_metrics.json")
        )

    def log_metrics(self, metrics: pd.DataFrame, run_name: str) -> None:
        """Log metrics to the configured logger if available.

        Args:
            metrics: DataFrame containing the computed metrics
            run_name: Name of the current run
        """
        if self.logger is not None:
            self.logger.log_dataframe(
                name=self.benchmark_name, dataframe=metrics.reset_index()
            )

    def save_state(self, checkpoint_location: str, is_preemption: bool = False) -> bool:
        pass

    def load_state(self, checkpoint_location: str | None) -> None:
        pass
