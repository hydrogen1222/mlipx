# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""VASP-syntax-compatible XDATCAR trajectory writer."""

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
        self.configuration_index = 0

    def write_header(self, atoms: Atoms, output_path: Path | str) -> None:
        """Write XDATCAR header.

        Args:
            atoms: ASE Atoms object (template for structure)
            output_path: Output file path
        """
        output_path = Path(output_path)

        # VASP associates each count with a contiguous block of coordinates.
        # Preserve the actual atom order and therefore repeated symbol blocks;
        # Counter-based grouping corrupts interleaved structures.
        symbols = atoms.get_chemical_symbols()
        symbol_counts: list[tuple[str, int]] = []
        for symbol in symbols:
            if symbol_counts and symbol_counts[-1][0] == symbol:
                previous, count = symbol_counts[-1]
                symbol_counts[-1] = (previous, count + 1)
            else:
                symbol_counts.append((symbol, 1))

        # Match VASP's own fixed-width XDATCAR layout.  Apart from making the
        # file familiar to users, keeping this grammar exact matters for
        # readers that are less permissive than ASE.
        lines = [
            f"{atoms.get_chemical_formula():<40s}",
            f"{1:12d}",  # absolute lattice vectors below
        ]

        # Lattice vectors
        cell = atoms.cell
        for i in range(3):
            lines.append(" " + "".join(f"{value:12.6f}" for value in cell[i]))

        # Element symbols
        element_line = "".join(f"{symbol:>5s}" for symbol, _ in symbol_counts)
        lines.append(element_line)

        # Atom counts
        count_line = " " + "".join(f"{count:>5d}" for _, count in symbol_counts)
        lines.append(count_line)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        self.header_written = True
        self.configuration_index = 0

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

        # VASP writes continuous (unwrapped) direct coordinates in XDATCAR.
        # This is essential for diffusion/MSD: wrapping into [0, 1) destroys
        # the image history whenever an atom crosses a periodic boundary.
        scaled_pos = atoms.get_scaled_positions(wrap=False)

        if not self.header_written:
            raise RuntimeError("write_header() must be called before append_frame()")

        self.configuration_index += 1
        # This exact marker is the VASP/ASE XDATCAR grammar.  The previous
        # custom "# Step:" comment made the streamed file unreadable as a
        # multi-frame XDATCAR.
        lines = [f"Direct configuration={self.configuration_index:12d}"]

        for pos in scaled_pos:
            lines.append(" " + "".join(f"{value:12.8f}" for value in pos))

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

            symbols = first_frame.get("symbols")
            if symbols is None and "atoms" in first_frame:
                symbols = first_frame["atoms"].get_chemical_symbols()
            if symbols is None:
                raise ValueError(
                    "Trajectory frame with 'positions' must also contain "
                    "'symbols' (or an 'atoms' object)."
                )
            atoms = AtomsClass(
                symbols=symbols,
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


def convert_to_vasp_xdatcar(
    input_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Convert an XDATCAR trajectory to the standard VASP XDATCAR layout.

    Current mlipx output already uses this layout.  The converter remains
    useful for older mlipx files, ASE trajectories (especially ``.traj``),
    and VASP-style files produced by other tools.  When recovering an old MD
    run, prefer ``trajectory.traj`` over an already-wrapped legacy XDATCAR:
    text reformatting cannot reconstruct image information that was lost.

    Args:
        input_path: Any ASE-readable trajectory, including ``trajectory.traj``
            and a real or mlipx XDATCAR (conversion is idempotent).
        output_path: Destination file. Defaults to ``<input>.vasp``.

    Returns:
        The path of the converted file.
    """
    from ase import Atoms as AtomsClass
    from ase.io import read

    frames = read(input_path, index=":")
    if isinstance(frames, AtomsClass):
        frames = [frames]
    if not frames:
        raise ValueError(f"No trajectory frames found in {input_path}")

    if output_path is None:
        source = Path(input_path)
        output_path = source.with_name(source.name + ".vasp")

    # Element blocks: VASP associates each count with a contiguous run of
    # coordinates, so group consecutive equal symbols (same rule as the
    # native writer's header).
    symbols = frames[0].get_chemical_symbols()
    groups: list[tuple[str, int]] = []
    for symbol in symbols:
        if groups and groups[-1][0] == symbol:
            groups[-1] = (symbol, groups[-1][1] + 1)
        else:
            groups.append((symbol, 1))

    lines: list[str] = []
    # Comment line: VASP writes an arbitrary label; use the formula.
    lines.append(f"{frames[0].get_chemical_formula():<40s}")
    # Scale factor: lattice vectors below are absolute, so the factor is 1.
    lines.append(f"{1:12d}")
    # Lattice vectors: 12-char fields, 6 decimals, one leading space per row
    # (matches VASP's ``1X,3F12.6`` layout byte-for-byte).
    for vector in frames[0].cell:
        lines.append(" " + "".join(f"{component:12.6f}" for component in vector))
    # Element symbols / counts: 5-char right-aligned fields; the count row
    # carries the same leading space as the numeric rows.
    lines.append("".join(f"{symbol:>5s}" for symbol, _ in groups))
    lines.append(" " + "".join(f"{count:>5d}" for _, count in groups))
    # Frames with UNWRAPPED scaled coordinates (VASP convention).
    for index, atoms in enumerate(frames, start=1):
        lines.append(f"Direct configuration={index:12d}")
        for position in atoms.get_scaled_positions(wrap=False):
            lines.append(" " + "".join(f"{component:12.8f}" for component in position))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path
