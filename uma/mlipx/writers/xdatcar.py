# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
VASP-style XDATCAR writer.

Writes MD trajectories in VASP XDATCAR format for visualization.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

    from ase import Atoms


class XdatcarWriter:
    """Write MD trajectory in VASP XDATCAR format.

    XDATCAR is a simple format for storing MD trajectories,
    compatible with VASP visualization tools.

    Example:
        >>> writer = XdatcarWriter()
        >>> writer.write_header(atoms[0], Path("XDATCAR"))
        >>> for frame in trajectory:
        ...     writer.append_frame(Path("XDATCAR"), frame)
    """

    def __init__(self):
        """Initialize XDATCAR writer."""
        self.header_written = False

    def write_header(self, atoms: Atoms, output_path: Path | str) -> None:
        """Write XDATCAR header.

        Args:
            atoms: ASE Atoms object (template for structure)
            output_path: Output file path
        """
        output_path = Path(output_path)

        # Get chemical formula and counts
        symbols = atoms.get_chemical_symbols()
        from collections import Counter

        symbol_counts = Counter(symbols)

        # Build header
        lines = [
            f"{atoms.get_chemical_formula()}",
            "1.0",  # Scale factor
        ]

        # Lattice vectors
        cell = atoms.cell
        for i in range(3):
            lines.append(
                f"  {cell[i][0]:20.16f}  {cell[i][1]:20.16f}  {cell[i][2]:20.16f}"
            )

        # Element symbols
        element_line = "  ".join(symbol_counts.keys())
        lines.append(element_line)

        # Atom counts
        count_line = "  ".join(str(c) for c in symbol_counts.values())
        lines.append(count_line)

        # Direct coordinates marker
        lines.append("Direct")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        self.header_written = True

    def append_frame(
        self,
        output_path: Path | str,
        atoms: Atoms,
        step: int | None = None,
    ) -> None:
        """Append a trajectory frame to XDATCAR.

        Args:
            output_path: Output file path
            atoms: ASE Atoms object for this frame
            step: MD step number (optional)
        """
        output_path = Path(output_path)

        # Get scaled positions
        scaled_pos = atoms.get_scaled_positions()

        lines = []
        if step is not None:
            lines.append(f"\n# Step: {step}")

        for pos in scaled_pos:
            lines.append(f"  {pos[0]:20.16f}  {pos[1]:20.16f}  {pos[2]:20.16f}")

        with open(output_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def write(
        self,
        output_path: Path | str,
        trajectory: list[Atoms],
        step_interval: int = 1,
    ) -> None:
        """Write complete trajectory to XDATCAR.

        Args:
            output_path: Output file path
            trajectory: List of ASE Atoms objects
            step_interval: Interval between recorded steps
        """
        if not trajectory:
            return

        output_path = Path(output_path)

        # Write header from first frame
        self.write_header(trajectory[0], output_path)

        # Write all frames
        for i, atoms in enumerate(trajectory):
            step = i * step_interval
            self.append_frame(output_path, atoms, step=step)

    def write_from_md(
        self,
        output_path: Path | str,
        trajectory_data: list[dict[str, Any]],
        step_interval: int = 1,
    ) -> None:
        """Write trajectory from MD simulation data.

        Args:
            output_path: Output file path
            trajectory_data: List of frame dictionaries with 'atoms', 'step', 'energy'
            step_interval: Interval between recorded steps
        """
        if not trajectory_data:
            return

        output_path = Path(output_path)

        # Get initial structure for header
        first_frame = trajectory_data[0]
        if "atoms" in first_frame:
            atoms = first_frame["atoms"]
        elif "positions" in first_frame:
            # Reconstruct from positions
            from ase import Atoms as AtomsClass

            atoms = AtomsClass(
                symbols=first_frame.get(
                    "symbols", first_frame.get("atoms", {}).get_chemical_symbols()
                ),
                positions=first_frame["positions"],
                cell=first_frame["cell"],
                pbc=first_frame.get("pbc", True),
            )
        else:
            raise ValueError("Trajectory data must contain 'atoms' or 'positions'")

        self.write_header(atoms, output_path)

        # Write frames
        for frame in trajectory_data:
            if "atoms" in frame:
                frame_atoms = frame["atoms"]
            else:
                # Reconstruct
                from ase import Atoms as AtomsClass

                frame_atoms = AtomsClass(
                    symbols=frame.get("symbols", atoms.get_chemical_symbols()),
                    positions=frame["positions"],
                    cell=frame.get("cell", atoms.cell),
                    pbc=frame.get("pbc", atoms.pbc),
                )

            step = frame.get("step", 0)
            self.append_frame(output_path, frame_atoms, step=step)
