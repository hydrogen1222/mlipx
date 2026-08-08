"""Canonical, backend-independent trajectory data model."""

from __future__ import annotations

import csv
import json
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from ase.io import read
from ase.io.trajectory import Trajectory
from ase.stress import full_3x3_to_voigt_6_stress


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def _float_or_nan(value: str | None) -> float:
    if value in (None, ""):
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _legacy_log_options(run_dir: Path) -> dict[str, float | int]:
    """Recover explicitly printed MD settings from pre-contract run logs."""
    path = run_dir / "run.log"
    if not path.exists():
        return {}
    patterns = {
        "temperature": (r"Temperature:\s+([0-9.eE+-]+)\s+K", float),
        "timestep": (r"Time step:\s+([0-9.eE+-]+)\s+fs", float),
        "steps": (r"Steps:\s+(\d+)", int),
        "save_interval": (r"Save interval:\s+(\d+)", int),
    }
    values: dict[str, float | int] = {}
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            for key, (pattern, converter) in patterns.items():
                if key in values:
                    continue
                match = re.search(pattern, line)
                if match:
                    values[key] = converter(match.group(1))
            if len(values) == len(patterns):
                break
    return values


def _discover_run_and_source(source: str | Path) -> tuple[Path, Path]:
    path = Path(source).expanduser().resolve()
    if path.is_file():
        run_dir = (
            path.parent.parent if path.parent.name in {"raw", "vasp"} else path.parent
        )
        return run_dir, path
    if not path.is_dir():
        raise FileNotFoundError(f"Trajectory or run directory not found: {path}")

    for candidate in (
        path / "raw" / "trajectory.traj",
        path / "trajectory.traj",
        path / "vasp" / "XDATCAR",
        path / "XDATCAR",
    ):
        if candidate.exists():
            return path, candidate
    raise FileNotFoundError(
        f"No raw/trajectory.traj, trajectory.traj, or XDATCAR found in {path}"
    )


def _load_frames(path: Path):
    if path.suffix.lower() == ".traj":
        with Trajectory(path, mode="r") as trajectory:
            # Trajectory indexing already creates independent Atoms objects.
            # Do not call Atoms.copy(): ASE intentionally drops the attached
            # SinglePointCalculator, including stored energy/force/stress data.
            return list(trajectory)
    frames = read(path, index=":")
    return frames if isinstance(frames, list) else [frames]


def _unwrap_minimum_image(
    positions: np.ndarray,
    cells: np.ndarray,
    pbc: np.ndarray,
) -> np.ndarray:
    """Reconstruct continuous positions from consecutively wrapped frames."""
    n_frames = len(positions)
    if n_frames < 2 or not pbc.any():
        return positions.copy()
    fractional = np.empty_like(positions)
    for index in range(n_frames):
        fractional[index] = positions[index] @ np.linalg.inv(cells[index])
    continuous = np.empty_like(fractional)
    continuous[0] = fractional[0]
    for index in range(1, n_frames):
        delta = fractional[index] - fractional[index - 1]
        delta[:, pbc] -= np.round(delta[:, pbc])
        continuous[index] = continuous[index - 1] + delta
    return np.einsum("fai,fij->faj", continuous, cells)


@dataclass(slots=True)
class TrajectoryDataset:
    """Normalized trajectory arrays and their simulation metadata."""

    run_dir: Path
    source_path: Path
    positions: np.ndarray
    cells: np.ndarray
    symbols: tuple[str, ...]
    pbc: np.ndarray
    steps: np.ndarray
    time_fs: np.ndarray
    timestep_fs: float | None
    save_interval: int | None
    frame_interval_fs: float | None
    velocities: np.ndarray | None = None
    forces_eV_A: np.ndarray | None = None
    potential_energy_eV: np.ndarray | None = None
    stress_eV_A3: np.ndarray | None = None
    thermodynamics: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def load(
        cls,
        source: str | Path,
        *,
        frame_interval_fs: float | None = None,
        assume_wrapped: bool | None = None,
    ) -> "TrajectoryDataset":
        """Load a new or legacy mlipx run, or an ASE-readable trajectory."""
        run_dir, source_path = _discover_run_and_source(source)
        frames = _load_frames(source_path)
        if not frames:
            raise ValueError(f"Trajectory contains no frames: {source_path}")

        symbols = tuple(frames[0].get_chemical_symbols())
        natoms = len(symbols)
        positions = np.empty((len(frames), natoms, 3), dtype=float)
        cells = np.empty((len(frames), 3, 3), dtype=float)
        velocity_rows: list[np.ndarray] = []
        force_rows: list[np.ndarray] = []
        energy_rows: list[float] = []
        stress_rows: list[np.ndarray] = []
        kinetic_rows: list[float] = []
        temperature_rows: list[float] = []
        volume_rows: list[float] = []
        has_velocities = True
        has_forces = True
        has_energies = True
        has_stresses = True
        reference_pbc = np.asarray(frames[0].pbc, dtype=bool)
        for index, atoms in enumerate(frames):
            if len(atoms) != natoms:
                raise ValueError(f"Frame {index} has a different atom count")
            if tuple(atoms.get_chemical_symbols()) != symbols:
                raise ValueError(f"Frame {index} has a different atom order")
            if not np.array_equal(atoms.pbc, reference_pbc):
                raise ValueError(f"Frame {index} has different PBC flags")
            positions[index] = atoms.positions
            cells[index] = atoms.cell.array
            if atoms.has("momenta"):
                velocity_rows.append(atoms.get_velocities())
                kinetic_rows.append(float(atoms.get_kinetic_energy()))
                try:
                    temperature_rows.append(float(atoms.get_temperature()))
                except (ValueError, ZeroDivisionError, NotImplementedError):
                    temperature_rows.append(float("nan"))
            else:
                has_velocities = False
                kinetic_rows.append(float("nan"))
                temperature_rows.append(float("nan"))
            volume_rows.append(
                float(atoms.get_volume()) if atoms.cell.rank == 3 else 0.0
            )
            results = getattr(atoms.calc, "results", {}) if atoms.calc else {}
            if "forces" in results:
                force_rows.append(np.asarray(results["forces"], dtype=float))
            else:
                has_forces = False
            energy = results.get("energy", results.get("free_energy"))
            if energy is not None:
                energy_rows.append(float(energy))
            else:
                has_energies = False
            if "stress" in results:
                stress = np.asarray(results["stress"], dtype=float)
                if stress.shape == (3, 3):
                    stress = full_3x3_to_voigt_6_stress(stress)
                if stress.shape != (6,):
                    has_stresses = False
                else:
                    stress_rows.append(stress)
            else:
                has_stresses = False
        velocities = np.asarray(velocity_rows) if has_velocities else None
        forces = np.asarray(force_rows) if has_forces else None
        energies = np.asarray(energy_rows) if has_energies else None
        stresses = np.asarray(stress_rows) if has_stresses else None
        pbc = reference_pbc

        artifacts = _read_json(run_dir / "artifacts.json")
        resolved = _read_json(run_dir / "resolved_config.json")
        configured_options = resolved.get("run_options", {})
        legacy_options = _legacy_log_options(run_dir)
        run_options = {**legacy_options, **configured_options}
        trajectory_meta = artifacts.get("trajectory", {})
        timestep = trajectory_meta.get("timestep_fs", run_options.get("timestep"))
        save_interval = trajectory_meta.get(
            "save_interval_steps", run_options.get("save_interval")
        )
        inferred_interval = trajectory_meta.get("saved_interval_fs")
        if (
            inferred_interval is None
            and timestep is not None
            and save_interval is not None
        ):
            inferred_interval = float(timestep) * int(save_interval)
        if frame_interval_fs is None:
            frame_interval_fs = (
                float(inferred_interval) if inferred_interval is not None else None
            )

        thermo = cls._load_thermodynamics(run_dir)
        if energies is not None:
            thermo.setdefault("potential_energy_eV", energies)
        if has_velocities:
            kinetic = np.asarray(kinetic_rows)
            thermo.setdefault("kinetic_energy_eV", kinetic)
            if energies is not None:
                thermo.setdefault("total_energy_eV", energies + kinetic)
            thermo.setdefault("temperature_K", np.asarray(temperature_rows))
        thermo.setdefault("volume_A3", np.asarray(volume_rows))
        if stresses is not None:
            for component, values in zip(
                ("xx", "yy", "zz", "yz", "xz", "xy"), stresses.T, strict=True
            ):
                thermo.setdefault(f"stress_{component}_eV_A3", values)
            thermo.setdefault(
                "pressure_GPa", -np.mean(stresses[:, :3], axis=1) * 160.21766208
            )
        nframes = len(frames)
        if "step" in thermo and len(thermo["step"]) == nframes:
            steps = thermo["step"].astype(int)
            step_axis = "simulation_step"
        elif save_interval is not None:
            steps = np.arange(nframes, dtype=int) * int(save_interval)
            step_axis = "simulation_step"
        else:
            steps = np.arange(nframes, dtype=int)
            step_axis = "frame_index"
        if "time_fs" in thermo and len(thermo["time_fs"]) == nframes:
            time_fs = thermo["time_fs"]
            if frame_interval_fs is None and nframes > 1:
                frame_interval_fs = float(np.median(np.diff(time_fs)))
        elif frame_interval_fs is not None:
            time_fs = np.arange(nframes, dtype=float) * frame_interval_fs
        else:
            time_fs = np.arange(nframes, dtype=float)
        thermo.setdefault("step", steps.astype(float))
        thermo.setdefault("time_fs", time_fs)

        load_warnings: list[str] = []
        if legacy_options and not configured_options:
            load_warnings.append(
                "Simulation timing/temperature metadata were recovered from legacy "
                "run.log because resolved_config.json is absent."
            )
        if assume_wrapped is None:
            contract = str(trajectory_meta.get("positions", "")).lower()
            if "unwrapped" in contract or source_path.suffix.lower() == ".traj":
                assume_wrapped = False
            else:
                scaled = positions[0] @ np.linalg.inv(cells[0])
                assume_wrapped = bool(
                    np.all((scaled >= -1e-10) & (scaled < 1.0 + 1e-10))
                )
        if assume_wrapped:
            positions = _unwrap_minimum_image(positions, cells, pbc)
            load_warnings.append(
                "Positions were unwrapped using consecutive minimum-image "
                "displacements; this assumes no atom moves over half a cell "
                "between stored frames."
            )
        if frame_interval_fs is None:
            load_warnings.append(
                "Frame interval is unknown; pass --frame-interval-fs for "
                "time-dependent transport quantities."
            )

        temperature = run_options.get("temperature")
        if temperature is None:
            result_data = _read_json(run_dir / "raw" / "mlipx_results.json")
            if not result_data:
                result_data = _read_json(run_dir / "mlipx_results.json")
            temperature = (
                result_data.get("calculation", {}).get("md", {}).get("temperature")
            )
        metadata = {
            "artifacts": artifacts,
            "resolved_config": resolved,
            "legacy_log_options": legacy_options,
            "temperature_K": temperature,
            "formula": frames[0].get_chemical_formula(),
            "minimum_image_unwrapped": bool(assume_wrapped),
            "step_axis": step_axis,
        }
        return cls(
            run_dir=run_dir,
            source_path=source_path,
            positions=positions,
            cells=cells,
            symbols=symbols,
            pbc=pbc,
            steps=steps,
            time_fs=time_fs,
            timestep_fs=float(timestep) if timestep is not None else None,
            save_interval=int(save_interval) if save_interval is not None else None,
            frame_interval_fs=frame_interval_fs,
            velocities=velocities,
            forces_eV_A=forces,
            potential_energy_eV=energies,
            stress_eV_A3=stresses,
            thermodynamics=thermo,
            metadata=metadata,
            warnings=load_warnings,
        )

    @staticmethod
    def _load_thermodynamics(run_dir: Path) -> dict[str, np.ndarray]:
        for path in (run_dir / "raw" / "md.csv", run_dir / "md.csv"):
            if not path.exists():
                continue
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            if not rows:
                return {}
            return {
                key: np.asarray([_float_or_nan(row.get(key)) for row in rows])
                for key in rows[0]
            }
        return {}

    @property
    def nframes(self) -> int:
        return len(self.positions)

    @property
    def natoms(self) -> int:
        return len(self.symbols)

    @property
    def temperature_K(self) -> float | None:
        value = self.metadata.get("temperature_K")
        return float(value) if value is not None else None

    def require_time(self) -> float:
        if self.frame_interval_fs is None or self.frame_interval_fs <= 0:
            raise ValueError(
                "Frame interval is missing or non-positive. Supply "
                "--frame-interval-fs in femtoseconds."
            )
        return self.frame_interval_fs

    def select(self, species: str | list[str] | tuple[str, ...] | None) -> np.ndarray:
        """Return indices matching comma-separated element symbols."""
        if species is None:
            return np.arange(self.natoms, dtype=int)
        if isinstance(species, str):
            requested = {item.strip() for item in species.split(",") if item.strip()}
        else:
            requested = {str(item).strip() for item in species if str(item).strip()}
        if not requested:
            raise ValueError("Species selection is empty")
        indices = np.asarray(
            [index for index, symbol in enumerate(self.symbols) if symbol in requested],
            dtype=int,
        )
        if not len(indices):
            available = ", ".join(sorted(set(self.symbols)))
            raise ValueError(
                f"No atoms match {sorted(requested)}; available species: {available}"
            )
        return indices

    def corrected_positions(
        self,
        *,
        framework: str | list[str] | tuple[str, ...] | None = None,
        mass_weighted: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Subtract rigid framework drift and return positions plus drift."""
        if framework is None:
            return self.positions.copy(), np.zeros((self.nframes, 3))
        indices = self.select(framework)
        displacement = self.positions[:, indices] - self.positions[0, indices]
        if mass_weighted:
            from ase.data import atomic_masses, atomic_numbers

            weights = np.asarray(
                [
                    atomic_masses[atomic_numbers[self.symbols[index]]]
                    for index in indices
                ]
            )
            drift = np.average(displacement, axis=1, weights=weights)
        else:
            drift = displacement.mean(axis=1)
        return self.positions - drift[:, None, :], drift

    def validation_report(self) -> dict[str, Any]:
        issues: list[str] = []
        report_warnings = list(self.warnings)
        if self.nframes < 2:
            report_warnings.append("Trajectory has fewer than two frames")
        if not np.isfinite(self.positions).all():
            issues.append("Positions contain NaN or infinity")
        if not np.isfinite(self.cells).all():
            issues.append("Cells contain NaN or infinity")
        if self.forces_eV_A is not None and not np.isfinite(self.forces_eV_A).all():
            issues.append("Forces contain NaN or infinity")
        if (
            self.potential_energy_eV is not None
            and not np.isfinite(self.potential_energy_eV).all()
        ):
            issues.append("Potential energies contain NaN or infinity")
        if self.stress_eV_A3 is not None and not np.isfinite(self.stress_eV_A3).all():
            issues.append("Stress contains NaN or infinity")
        determinants = np.linalg.det(self.cells)
        if self.pbc.any() and np.any(np.abs(determinants) < 1e-12):
            issues.append("At least one frame has a singular cell")
        if self.frame_interval_fs is not None and self.frame_interval_fs <= 0:
            issues.append("Frame interval must be positive")
        if self.frame_interval_fs is not None and self.frame_interval_fs > 10:
            report_warnings.append(
                "Stored-frame interval exceeds 10 fs; VACF/VDOS and transition "
                "timing may be undersampled."
            )
        return {
            "valid": not issues,
            "issues": issues,
            "warnings": report_warnings,
            "source": str(self.source_path),
            "formula": self.metadata["formula"],
            "frames": self.nframes,
            "atoms": self.natoms,
            "available_frame_data": {
                "velocities": self.velocities is not None,
                "forces": self.forces_eV_A is not None,
                "potential_energy": self.potential_energy_eV is not None,
                "stress": self.stress_eV_A3 is not None,
            },
            "species": {
                symbol: self.symbols.count(symbol)
                for symbol in sorted(set(self.symbols))
            },
            "step_axis": self.metadata["step_axis"],
            "first_step": (
                int(self.steps[0])
                if self.metadata["step_axis"] == "simulation_step"
                else None
            ),
            "last_step": (
                int(self.steps[-1])
                if self.metadata["step_axis"] == "simulation_step"
                else None
            ),
            "frame_interval_fs": self.frame_interval_fs,
            "total_time_ps": (
                float(self.time_fs[-1] - self.time_fs[0]) / 1000
                if self.frame_interval_fs is not None
                else None
            ),
            "cell_volume_A3": {
                "min": float(np.min(np.abs(determinants))),
                "max": float(np.max(np.abs(determinants))),
            },
        }


def warn_dataset_messages(dataset: TrajectoryDataset) -> None:
    """Expose loading assumptions to API users who do not inspect metadata."""
    for message in dataset.warnings:
        warnings.warn(message, UserWarning, stacklevel=2)
