# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
JSON output writer for machine-readable results.

Provides structured JSON output for programmatic analysis and data exchange.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np


def _mlipx_version() -> str:
    """Return the installed mlipx version (lazy to avoid circular import)."""
    try:
        from mlipx import __version__  # noqa: PLC0415

        return __version__
    except Exception:
        return "unknown"


if TYPE_CHECKING:
    from typing import Any

    from ase import Atoms


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy arrays and other special types."""

    def default(self, obj: Any) -> Any:
        """Convert numpy types to JSON-serializable types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


class JsonWriter:
    """Write calculation results to JSON format.

    Provides machine-readable output with all calculation details,
    suitable for programmatic analysis and data exchange.

    Example:
        >>> writer = JsonWriter()
        >>> writer.write(atoms, results, Path("results.json"), metadata={"model": "uma-s-1"})
    """

    def __init__(self, indent: int = 2):
        """Initialize JSON writer.

        Args:
            indent: Indentation level for pretty-printing
        """
        self.indent = indent

    def write(
        self,
        atoms: Atoms,
        results: dict[str, Any],
        output_path: Path | str,
        mode: str = "single_point",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Write results to JSON file.

        Args:
            atoms: ASE Atoms object
            results: Results dictionary from calculation
            output_path: Output file path
            mode: Calculation mode
            metadata: Additional metadata
        """
        output_path = Path(output_path)

        # Build JSON structure
        data = self._build_data(atoms, results, mode, metadata)

        # Write to file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, cls=NumpyEncoder, indent=self.indent)

    def _build_data(
        self,
        atoms: Atoms,
        results: dict[str, Any],
        mode: str,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Build JSON data structure.

        Args:
            atoms: ASE Atoms object
            results: Results dictionary
            mode: Calculation mode
            metadata: Additional metadata

        Returns:
            Dictionary ready for JSON serialization
        """
        # System information
        system_info = {
            "formula": atoms.get_chemical_formula(),
            "natoms": len(atoms),
            "symbols": atoms.get_chemical_symbols(),
            "cell": atoms.cell.tolist(),
            "cell_lengths": atoms.cell.lengths().tolist(),
            "cell_angles": atoms.cell.angles().tolist(),
            "volume": float(atoms.get_volume()),
            "pbc": list(atoms.pbc),
        }

        # Positions
        positions = atoms.positions.tolist()

        # Results
        output_results = {
            "energy": results.get("energy"),
            "energy_per_atom": (
                results.get("energy") / len(atoms) if results.get("energy") else None
            ),
            "forces": results.get("forces").tolist()
            if results.get("forces") is not None
            else None,
            "stress": results.get("stress").tolist()
            if results.get("stress") is not None
            else None,
        }

        # Remove None values
        output_results = {k: v for k, v in output_results.items() if v is not None}

        # Force statistics
        if results.get("forces") is not None:
            forces = results["forces"]
            force_mags = np.linalg.norm(forces, axis=1)
            output_results["force_statistics"] = {
                "fmax": float(np.max(force_mags)),
                "fmean": float(np.mean(force_mags)),
                "frms": float(np.sqrt(np.mean(force_mags**2))),
            }

        # Pressure from stress
        if results.get("stress") is not None:
            stress = results["stress"]
            pressure = -(stress[0] + stress[1] + stress[2]) / 3.0 * 160.2177
            output_results["pressure_gpa"] = float(pressure)

        # Timing
        timing = {"calculation_time_s": results.get("time")}

        # Build final structure
        data = {
            "mlipx_version": _mlipx_version(),
            "timestamp": datetime.now().isoformat(),
            "calculation": {
                "mode": mode,
                "system": system_info,
                "positions": positions,
                "results": output_results,
                "timing": timing,
            },
        }

        # Add metadata
        if metadata:
            data["metadata"] = metadata

        # Add mode-specific data
        if mode == "optimization":
            data["calculation"]["optimization"] = {
                "nsteps": results.get("nsteps"),
                "converged": results.get("converged"),
                "fmax_threshold": results.get("fmax"),
            }
        elif mode == "md":
            data["calculation"]["md"] = {
                "steps": results.get("md_steps"),
                "temperature": results.get("temperature"),
                "ensemble": results.get("ensemble"),
            }

        return data

    def write_batch(
        self,
        results_list: list[dict[str, Any]],
        output_path: Path | str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Write batch calculation results to JSON.

        Args:
            results_list: List of result dictionaries
            output_path: Output file path
            metadata: Additional metadata
        """
        output_path = Path(output_path)

        data = {
            "mlipx_version": _mlipx_version(),
            "timestamp": datetime.now().isoformat(),
            "batch_size": len(results_list),
            "results": results_list,
        }

        if metadata:
            data["metadata"] = metadata

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, cls=NumpyEncoder, indent=self.indent)
