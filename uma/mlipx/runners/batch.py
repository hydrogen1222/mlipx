# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Batch calculation runner.

Processes multiple structures in batch mode with parallel execution support.
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from ase.io import read
from tqdm import tqdm

from mlipx.runners.optimization import OptimizationRunner
from mlipx.runners.singlepoint import SinglePointRunner
from mlipx.writers.json_writer import JsonWriter

if TYPE_CHECKING:
    from typing import Any

    from ase import Atoms


class BatchRunner:
    """Run calculations on multiple structures in batch mode.

    Processes a directory of structure files sequentially or in parallel.

    Example:
        >>> runner = BatchRunner(
        ...     calculator,
        ...     calc_type="sp",
        ...     output_dir="batch_results"
        ... )
        >>> results = runner.run_from_directory("structures/")
        >>> print(f"Processed: {results['success']}/{results['total']}")
    """

    def __init__(
        self,
        calculator,
        calc_type: str = "sp",
        output_dir: Path | str = ".",
        parallel: bool = False,
        max_workers: int = 1,
        verbose: bool = True,
        job_name: str | None = None,
        **calc_kwargs,
    ):
        """Initialize batch runner.

        Args:
            calculator: UMA calculator wrapper
            calc_type: Type of calculation (sp, opt, md)
            output_dir: Directory for output files
            parallel: Whether to use parallel processing
            max_workers: Number of parallel workers
            verbose: Whether to print progress messages
            job_name: Optional job name for organizing results
            **calc_kwargs: Additional arguments for calculation runner
        """
        self.calculator = calculator
        self.calc_type = calc_type.lower()
        self.job_name = job_name
        self.parallel = parallel
        self.max_workers = max_workers
        self.verbose = verbose
        self.calc_kwargs = calc_kwargs

        # Build output directory path
        base_dir = Path(output_dir)
        if job_name:
            self.output_dir = base_dir / job_name
        else:
            self.output_dir = base_dir

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Validate calc_type
        valid_types = {"sp", "opt", "md"}
        if self.calc_type not in valid_types:
            raise ValueError(
                f"Unknown calculation type: {calc_type}. "
                f"Use one of: {', '.join(valid_types)}"
            )

    def run_from_directory(
        self,
        input_dir: Path | str,
        pattern: str = "*.cif",
    ) -> dict[str, Any]:
        """Run calculations on all matching files in directory.

        Args:
            input_dir: Input directory containing structure files
            pattern: Glob pattern for matching files

        Returns:
            Dictionary with batch results summary
        """
        input_dir = Path(input_dir)

        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        # Find all matching files
        files = list(input_dir.glob(pattern))
        files.extend(input_dir.glob("*.xyz"))
        files.extend(input_dir.glob("*.vasp"))
        files.extend(input_dir.glob("POSCAR*"))

        # Remove duplicates and sort
        files = sorted(set(files))

        if not files:
            print(f"No structure files found in {input_dir}")
            return {"total": 0, "success": 0, "failed": 0}

        print(f"Found {len(files)} structure files")

        return self.run_from_files(files)

    def run_from_files(self, files: list[Path | str]) -> dict[str, Any]:
        """Run calculations on a list of files.

        Args:
            files: List of structure file paths

        Returns:
            Dictionary with batch results summary
        """
        results_list = []
        success_count = 0
        failed_count = 0

        if self.parallel and self.max_workers > 1:
            from concurrent.futures import (  # noqa: PLC0415
                ThreadPoolExecutor,
                as_completed,
            )

            def process_one(filepath):
                filepath = Path(filepath)
                atoms = read(filepath)
                sub_dir = self.output_dir / filepath.stem
                sub_dir.mkdir(exist_ok=True)
                result = self._run_single(atoms, sub_dir, filepath.stem)
                return {
                    "filename": filepath.name,
                    "formula": atoms.get_chemical_formula(),
                    "natoms": len(atoms),
                    "success": True,
                    **result,
                }

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(process_one, f): f for f in files}
                for future in as_completed(futures):
                    filepath = futures[future]
                    try:
                        result = future.result()
                        results_list.append(result)
                        success_count += 1
                    except Exception as e:
                        results_list.append(
                            {
                                "filename": Path(filepath).name,
                                "success": False,
                                "error": f"{type(e).__name__}: {e!s}",
                            }
                        )
                        failed_count += 1
        else:
            # Progress bar
            iterator = (
                tqdm(files, desc="Processing structures") if self.verbose else files
            )

            for fp in iterator:
                filepath = Path(fp)

                try:
                    # Read structure
                    atoms = read(filepath)

                    # Create sub-directory for this calculation
                    sub_dir = self.output_dir / filepath.stem
                    sub_dir.mkdir(exist_ok=True)

                    # Run calculation
                    result = self._run_single(atoms, sub_dir, filepath.stem)

                    results_list.append(
                        {
                            "filename": filepath.name,
                            "formula": atoms.get_chemical_formula(),
                            "natoms": len(atoms),
                            "success": True,
                            **result,
                        }
                    )
                    success_count += 1

                except Exception as e:
                    error_msg = f"{type(e).__name__}: {e!s}"
                    results_list.append(
                        {
                            "filename": filepath.name,
                            "success": False,
                            "error": error_msg,
                            "traceback": traceback.format_exc(),
                        }
                    )
                    failed_count += 1

                    if self.verbose:
                        print(f"\nError processing {filepath.name}: {error_msg}")

        # Write summary
        summary = {
            "total": len(files),
            "success": success_count,
            "failed": failed_count,
            "results": results_list,
        }

        self._write_summary(summary)

        return summary

    def _run_single(
        self,
        atoms: Atoms,
        sub_dir: Path,
        name: str,
    ) -> dict[str, Any]:
        """Run single calculation.

        Args:
            atoms: ASE Atoms object
            sub_dir: Sub-directory for output
            name: Structure name

        Returns:
            Dictionary with results
        """
        # Create appropriate runner
        if self.calc_type == "sp":
            runner = SinglePointRunner(
                self.calculator,
                output_dir=sub_dir,
                verbose=False,
                **self.calc_kwargs,
            )
        elif self.calc_type == "opt":
            runner = OptimizationRunner(
                self.calculator,
                output_dir=sub_dir,
                verbose=False,
                **self.calc_kwargs,
            )
        else:
            raise NotImplementedError(
                f"Batch mode for {self.calc_type} not yet implemented"
            )

        # Run calculation
        results = runner.execute(atoms)

        # Extract key results
        return {
            "energy": results.get("energy"),
            "fmax": (
                np.max(np.linalg.norm(results.get("forces"), axis=1))
                if results.get("forces") is not None
                else None
            ),
            "time": results.get("time"),
            "timing": results.get("timing"),
            "output_dir": str(sub_dir),
        }

    def _write_summary(self, summary: dict[str, Any]) -> None:
        """Write batch summary to file.

        Args:
            summary: Summary dictionary
        """
        # Write JSON summary
        json_path = self.output_dir / "batch_summary.json"
        writer = JsonWriter()
        metadata = {
            "calc_type": self.calc_type,
            "total": summary["total"],
            "success": summary["success"],
            "failed": summary["failed"],
        }
        if self.job_name:
            metadata["job_name"] = self.job_name
        writer.write_batch(
            summary["results"],
            json_path,
            metadata=metadata,
        )

        if self.verbose:
            print(f"\nBatch summary written to: {json_path}")
            print(
                f"Total: {summary['total']}, Success: {summary['success']}, Failed: {summary['failed']}"
            )
