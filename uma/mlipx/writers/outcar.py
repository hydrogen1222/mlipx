# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
VASP-style OUTCAR writer.

Generates detailed output files similar to VASP's OUTCAR format
for familiar visualization and analysis workflows.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import Any

    from ase import Atoms


class OutcarWriter:
    """Write calculation results in VASP OUTCAR-like format.

    Provides detailed output with structure, energies, forces, stress,
    and timing information in a format familiar to VASP users.

    Example:
        >>> writer = OutcarWriter()
        >>> writer.write(atoms, results, Path("OUTCAR"), mode="single_point")
    """

    def __init__(self):
        """Initialize OUTCAR writer."""
        self.lines: list[str] = []

    def write(
        self,
        atoms: Atoms,
        results: dict[str, Any],
        output_path: Path | str,
        mode: str = "single_point",
        task_name: str = "omat",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Write results to OUTCAR file.

        Args:
            atoms: ASE Atoms object
            results: Results dictionary from calculation
            output_path: Output file path
            mode: Calculation mode (single_point, optimization, md)
            task_name: Task name (omat, omol, etc.)
            metadata: Additional metadata to include
        """
        output_path = Path(output_path)
        self.lines = []

        # Generate content
        self._write_header()
        self._write_system_info(atoms, task_name, mode)
        self._write_model_info(metadata)
        self._write_input_structure(atoms)
        self._write_results(atoms, results)
        if mode == "optimization":
            self._write_optimization_info(results)
        elif mode == "md":
            self._write_md_info(results)
        self._write_timing(results)
        self._write_footer()

        # Write to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines))

    def _write_header(self) -> None:
        """Write file header."""
        self.lines.extend(
            [
                "=" * 80,
                " MLIP CALCULATION RESULTS".center(80),
                " (mlipx - MLIP eXtended)".center(80),
                "=" * 80,
                "",
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
            ]
        )

    def _write_system_info(self, atoms: Atoms, task_name: str, mode: str) -> None:
        """Write system information section."""
        formula = atoms.get_chemical_formula()
        symbols = atoms.get_chemical_symbols()
        atom_counts = Counter(symbols)
        atom_summary = ", ".join([f"{k}: {v}" for k, v in sorted(atom_counts.items())])

        self.lines.extend(
            [
                "-" * 80,
                " SYSTEM INFORMATION",
                "-" * 80,
                "",
                f"Formula:           {formula}",
                f"Number of atoms:   {len(atoms)}",
                f"Atom types:        {atom_summary}",
                f"Task:              {task_name}",
                f"Calculation mode:  {mode}",
                "",
            ]
        )

    def _write_model_info(self, metadata: dict[str, Any] | None) -> None:
        """Write model information section."""
        self.lines.extend(
            [
                "-" * 80,
                " MODEL INFORMATION",
                "-" * 80,
                "",
            ]
        )

        if metadata:
            if "model_path" in metadata:
                self.lines.append(f"Model path:        {metadata['model_path']}")
            if "device" in metadata:
                self.lines.append(f"Device:            {metadata['device']}")
            if "inference_mode" in metadata:
                self.lines.append(f"Inference mode:    {metadata['inference_mode']}")
            if "implemented_properties" in metadata:
                props = ", ".join(metadata["implemented_properties"])
                self.lines.append(f"Properties:        {props}")

        self.lines.append("")

    def _write_input_structure(self, atoms: Atoms) -> None:
        """Write input structure section."""
        cell = atoms.cell
        positions = atoms.positions
        symbols = atoms.get_chemical_symbols()

        self.lines.extend(
            [
                "-" * 80,
                " INPUT STRUCTURE",
                "-" * 80,
                "",
                "Lattice vectors (Å):",
            ]
        )

        for i in range(3):
            self.lines.append(
                f"  {cell[i][0]:12.6f}  {cell[i][1]:12.6f}  {cell[i][2]:12.6f}"
            )

        self.lines.extend(
            [
                "",
                f"Cell lengths (Å):    {cell.lengths()[0]:.6f}  {cell.lengths()[1]:.6f}  {cell.lengths()[2]:.6f}",
                f"Cell angles (°):     {cell.angles()[0]:.6f}  {cell.angles()[1]:.6f}  {cell.angles()[2]:.6f}",
                f"Volume (Å³):         {atoms.get_volume():.6f}",
                "",
                "Atomic positions (Cartesian, Å):",
                f"{'Atom':>6} {'Type':>6} {'x':>12} {'y':>12} {'z':>12}",
                "-" * 60,
            ]
        )

        for i, (symbol, pos) in enumerate(zip(symbols, positions)):
            self.lines.append(
                f"{i + 1:>6} {symbol:>6} {pos[0]:>12.6f} {pos[1]:>12.6f} {pos[2]:>12.6f}"
            )

        self.lines.append("")

    def _write_results(self, atoms: Atoms, results: dict[str, Any]) -> None:
        """Write calculation results section."""
        energy = results.get("energy")
        forces = results.get("forces")
        stress = results.get("stress")

        self.lines.extend(
            [
                "-" * 80,
                " ENERGY",
                "-" * 80,
                "",
            ]
        )

        if energy is not None:
            self.lines.extend(
                [
                    f"Total energy:       {energy:16.8f} eV",
                    f"Energy per atom:    {energy / len(atoms):16.8f} eV/atom",
                    "",
                ]
            )

        if forces is not None:
            self.lines.extend(
                [
                    "-" * 80,
                    " FORCES (eV/Å)",
                    "-" * 80,
                    "",
                    f"{'Atom':>6} {'Type':>6} {'Fx':>12} {'Fy':>12} {'Fz':>12} {'|F|':>12}",
                    "-" * 70,
                ]
            )

            symbols = atoms.get_chemical_symbols()
            max_force = 0.0
            max_force_idx = 0
            rms_force = 0.0

            for i in range(len(atoms)):
                fx, fy, fz = forces[i]
                force_mag = np.linalg.norm(forces[i])
                self.lines.append(
                    f"{i + 1:>6} {symbols[i]:>6} {fx:>12.6f} {fy:>12.6f} {fz:>12.6f} {force_mag:>12.6f}"
                )

                if force_mag > max_force:
                    max_force = force_mag
                    max_force_idx = i
                rms_force += force_mag**2

            rms_force = np.sqrt(rms_force / len(atoms))

            self.lines.extend(
                [
                    "",
                    f"Maximum force:      {max_force:12.6f} eV/Å on atom {max_force_idx + 1} ({symbols[max_force_idx]})",
                    f"RMS force:          {rms_force:12.6f} eV/Å",
                    "",
                ]
            )

        if stress is not None:
            self.lines.extend(
                [
                    "-" * 80,
                    " STRESS TENSOR",
                    "-" * 80,
                    "",
                    "Stress (eV/Å³):",
                    f"{'':>12} {'xx':>12} {'yy':>12} {'zz':>12} {'yz':>12} {'xz':>12} {'xy':>12}",
                    f"{'Voigt':>12} {stress[0]:>12.6f} {stress[1]:>12.6f} {stress[2]:>12.6f} "
                    f"{stress[3]:>12.6f} {stress[4]:>12.6f} {stress[5]:>12.6f}",
                    "",
                    "Stress (GPa):",
                ]
            )

            # Convert to GPa (1 eV/Å³ = 160.2177 GPa)
            stress_gpa = np.array(stress) * 160.2177
            self.lines.append(
                f"{'Voigt':>12} {stress_gpa[0]:>12.6f} {stress_gpa[1]:>12.6f} {stress_gpa[2]:>12.6f} "
                f"{stress_gpa[3]:>12.6f} {stress_gpa[4]:>12.6f} {stress_gpa[5]:>12.6f}"
            )

            # Pressure
            pressure = -(stress[0] + stress[1] + stress[2]) / 3.0 * 160.2177
            self.lines.extend(
                [
                    "",
                    f"Pressure:           {pressure:12.6f} GPa",
                    "",
                ]
            )

    def _write_optimization_info(self, results: dict[str, Any]) -> None:
        """Write optimization-specific information."""
        nsteps = results.get("nsteps")
        converged = results.get("converged")
        fmax = results.get("fmax")

        self.lines.extend(
            [
                "-" * 80,
                " OPTIMIZATION",
                "-" * 80,
                "",
            ]
        )

        if nsteps is not None:
            self.lines.append(f"Steps taken:        {nsteps}")
        if converged is not None:
            status = "Yes" if converged else "No"
            self.lines.append(f"Converged:          {status}")
        if fmax is not None:
            self.lines.append(f"Final fmax:         {fmax:.6f} eV/Å")

        self.lines.append("")

    def _write_md_info(self, results: dict[str, Any]) -> None:
        """Write MD-specific information."""
        steps = results.get("md_steps")
        temperature = results.get("temperature")

        self.lines.extend(
            [
                "-" * 80,
                " MOLECULAR DYNAMICS",
                "-" * 80,
                "",
            ]
        )

        if steps is not None:
            self.lines.append(f"MD steps:           {steps}")
        if temperature is not None:
            self.lines.append(f"Temperature:        {temperature:.2f} K")

        self.lines.append("")

    def _write_timing(self, results: dict[str, Any]) -> None:
        """Write timing information."""
        calc_time = results.get("time", 0.0)

        self.lines.extend(
            [
                "-" * 80,
                " TIMING",
                "-" * 80,
                "",
                f"Calculation time:   {calc_time:.2f} s",
                f"                     ({calc_time / 60:.2f} min)"
                if calc_time > 60
                else "",
            ]
        )

        # Remove empty line if no minutes
        if calc_time <= 60:
            self.lines.pop()

        self.lines.append("")

    def _write_footer(self) -> None:
        """Write file footer."""
        self.lines.extend(
            [
                "=" * 80,
                " END OF MLIP CALCULATION",
                "=" * 80,
            ]
        )
