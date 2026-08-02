"""Optional GEMDAT-backed solid-state electrolyte analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from mlipx.analysis.dataset import TrajectoryDataset


@dataclass(slots=True)
class GemdatResult:
    summary: dict[str, Any]
    arrays: dict[str, np.ndarray]
    tables: dict[str, Any] = field(default_factory=dict)
    structures: dict[str, Any] = field(default_factory=dict)
    volumes: dict[str, Any] = field(default_factory=dict)
    paths: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _require_gemdat():
    try:
        from gemdat import Trajectory
        from pymatgen.core import Species, Structure
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "The electrolyte task requires GEMDAT; install mlipx[electrolyte]."
        ) from exc
    return Trajectory, Species, Structure


def _gemdat_trajectory(dataset: TrajectoryDataset):
    Trajectory, Species, _ = _require_gemdat()
    fractional = np.empty_like(dataset.positions)
    for index, cell in enumerate(dataset.cells):
        fractional[index] = dataset.positions[index] @ np.linalg.inv(cell)
    constant_lattice = bool(np.allclose(dataset.cells, dataset.cells[0]))
    return Trajectory(
        species=[Species(symbol) for symbol in dataset.symbols],
        coords=np.mod(fractional, 1.0),
        lattice=dataset.cells[0] if constant_lattice else dataset.cells,
        constant_lattice=constant_lattice,
        time_step=dataset.require_time() * 1e-15,
        metadata={"temperature": dataset.temperature_K},
    )


def _path_arrays(path: Any, lattice: Any) -> dict[str, np.ndarray]:
    voxels = np.asarray(path.sites, dtype=int)
    dims = np.asarray(path.dims, dtype=int)
    fractional = (np.mod(voxels, dims) + 0.5) / dims
    return {
        "voxel": voxels,
        "fractional": fractional,
        "cartesian_A": np.asarray(lattice.get_cartesian_coords(fractional)),
        "free_energy_eV": np.asarray(path.energy, dtype=float),
    }


def gemdat_electrolyte(
    dataset: TrajectoryDataset,
    *,
    species: str,
    temperature_K: float | None = None,
    resolution_A: float = 0.5,
    sites_path: str | Path | None = None,
    background_level: float = 0.1,
    site_radius_A: float | None = None,
    minimal_residence: int = 0,
    percolation: str = "xyz",
) -> GemdatResult:
    """Analyze mobile-ion density, sites, jumps and percolating paths."""
    if resolution_A <= 0:
        raise ValueError("resolution_A must be positive")
    if any(axis not in "xyz" for axis in percolation) or len(set(percolation)) != len(
        percolation
    ):
        raise ValueError("percolation must be a subset of 'xyz'")
    temperature_K = dataset.temperature_K if temperature_K is None else temperature_K
    if temperature_K is None:
        raise ValueError("GEMDAT free energy requires --temperature in kelvin")
    if temperature_K <= 0:
        raise ValueError("temperature_K must be positive")
    dataset.select(species)
    _, _, Structure = _require_gemdat()
    trajectory = _gemdat_trajectory(dataset)
    mobile = trajectory.filter(species)
    volume = mobile.to_volume(resolution=resolution_A)
    peaks = volume.find_peaks()
    warnings: list[str] = []

    if sites_path is not None:
        sites = Structure.from_file(str(sites_path))
        site_source = str(Path(sites_path).expanduser().resolve())
    else:
        sites = volume.to_structure(
            specie=species,
            background_level=background_level,
            peaks=peaks,
            return_occupancies=True,
            n_frames=dataset.nframes,
        )
        site_source = "automatic density peak segmentation"
    if not len(sites):
        warnings.append(
            "No sites were detected; adjust --background-level or provide --sites."
        )

    free_energy = volume.get_free_energy(temperature_K)
    result = GemdatResult(
        summary={
            "method": "GEMDAT density/site/transition analysis",
            "species": species,
            "temperature_K": temperature_K,
            "resolution_A": resolution_A,
            "density_grid_shape": list(volume.data.shape),
            "site_source": site_source,
            "number_of_density_peaks": int(len(peaks)),
            "number_of_sites": int(len(sites)),
        },
        arrays={
            "density_counts": np.asarray(volume.data),
            "density_probability": np.asarray(volume.probability()),
            "free_energy_eV": np.asarray(free_energy.data),
            "peak_voxels": np.asarray(peaks, dtype=int),
            "cell_A": np.asarray(dataset.cells[0]),
        },
        structures={"sites": sites, "reference": trajectory.get_structure(0)},
        volumes={"density": volume, "free_energy": free_energy},
        warnings=warnings,
    )

    for axis in percolation:
        if not len(peaks):
            break
        try:
            path = free_energy.optimal_percolating_path(peaks=peaks, percolate=axis)
            if path is None:
                warnings.append(f"No percolating path found along {axis}.")
                continue
            arrays = _path_arrays(path, free_energy.lattice)
            result.paths[axis] = arrays
            energy = arrays["free_energy_eV"]
            result.summary.setdefault("percolation", {})[axis] = {
                "steps": int(len(energy)),
                "barrier_eV": float(np.max(energy) - np.min(energy)),
                "path_integrated_energy_eV": float(np.sum(energy)),
            }
        except (ValueError, RuntimeError) as exc:
            warnings.append(f"Percolation analysis along {axis} failed: {exc}")

    if not len(sites):
        return result
    try:
        transitions = trajectory.transitions_between_sites(
            sites,
            species,
            site_radius=site_radius_A,
        )
        occupancy = transitions.occupancy()
        result.structures["occupancy"] = occupancy
        result.tables["transition_events"] = transitions.events
        result.arrays["transition_matrix"] = transitions.matrix()
        result.summary["transition_events"] = int(transitions.n_events)
        result.summary["occupancy_by_site_type"] = transitions.occupancy_by_site_type()
        result.summary["atom_locations"] = transitions.atom_locations()
        residence = transitions.residence_time()
        result.tables["residence_times"] = residence

        jumps = transitions.jumps(minimal_residence=minimal_residence)
        result.tables["jumps"] = jumps.data
        result.arrays["jump_matrix"] = jumps.matrix()
        result.summary["number_of_jumps"] = int(jumps.n_jumps)
        dimensions = len(percolation) if percolation else 3
        if jumps.n_jumps:
            result.summary["jump_diffusivity_m2_s"] = float(
                jumps.jump_diffusivity(dimensions)
            )
            n_parts = min(10, max(2, dataset.nframes // 4))
            if n_parts <= dataset.nframes:
                result.tables["jump_rates"] = jumps.rates(n_parts=n_parts)
            try:
                collective = jumps.collective()
                result.summary["solo_jump_fraction"] = float(jumps.solo_fraction)
                result.summary["collective_jump_count"] = int(collective.n_coll_jumps)
            except (ValueError, ZeroDivisionError, AssertionError) as exc:
                warnings.append(f"Collective-jump classification failed: {exc}")
    except (ValueError, RuntimeError, AssertionError) as exc:
        warnings.append(f"Site transition analysis failed: {exc}")
    return result
