"""Lazy GEMDAT adapter for explicit site/jump/pathway mechanism analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from mlipx.analysis.validation import (
    OptionalDependencyError,
    UnsupportedAnalysisError,
    require_analysis,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any

    from mlipx.analysis.dataset import TrajectoryDataset


@dataclass(slots=True)
class GemdatResult:
    summary: dict[str, Any]
    arrays: dict[str, np.ndarray]
    tables: dict[str, Any] = field(default_factory=dict)
    structures: dict[str, Any] = field(default_factory=dict)
    paths: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _require_gemdat():
    try:
        from gemdat import Trajectory
        from pymatgen.core import Species, Structure
    except ImportError as exc:  # pragma: no cover - optional environment
        raise OptionalDependencyError(
            "GEMDAT electrolyte analysis is unavailable. Install " "mlipx[electrolyte]."
        ) from exc
    try:
        gemdat_version = version("gemdat")
    except PackageNotFoundError:  # pragma: no cover
        gemdat_version = "unknown"
    if gemdat_version != "unknown" and not gemdat_version.startswith("1."):
        raise OptionalDependencyError(
            f"Unsupported GEMDAT version {gemdat_version}; this adapter is tested "
            "with GEMDAT>=1.7.3,<2."
        )
    return Trajectory, Species, Structure, gemdat_version


def _validate_mechanism_dimensions(
    *, jump_dimensions: int, percolation_axes: str
) -> None:
    if jump_dimensions not in {1, 2, 3}:
        raise ValueError("jump_dimensions must be 1, 2, or 3")
    if (
        not percolation_axes
        or any(axis not in "xyz" for axis in percolation_axes)
        or len(set(percolation_axes)) != len(percolation_axes)
    ):
        raise ValueError("percolation_axes must be a non-empty subset of xyz")


def jump_summary(
    jumps,
    *,
    jump_dimensions: int,
    percolation_axes: str,
) -> dict[str, Any]:
    """Serialize jump metrics while keeping dimensions independent of axes."""

    _validate_mechanism_dimensions(
        jump_dimensions=jump_dimensions, percolation_axes=percolation_axes
    )
    result = {
        "number_of_jumps": int(jumps.n_jumps),
        "jump_dimensions": int(jump_dimensions),
        "percolation_axes": percolation_axes,
    }
    if jumps.n_jumps:
        result["jump_diffusivity_m2_s"] = float(jumps.jump_diffusivity(jump_dimensions))
    return result


def _gemdat_trajectory(dataset: TrajectoryDataset, *, temperature_K: float | None):
    Trajectory, Species, _, _ = _require_gemdat()
    fractional = np.einsum(
        "fai,fij->faj", dataset.positions, np.linalg.inv(dataset.cells)
    )
    return Trajectory(
        species=[Species(symbol) for symbol in dataset.symbols],
        coords=np.mod(fractional, 1.0),
        lattice=dataset.cells[0],
        constant_lattice=True,
        time_step=dataset.frame_interval_fs * 1.0e-15,
        metadata={"temperature": temperature_K},
    )


def _temperature(
    dataset: TrajectoryDataset, explicit_temperature_K: float | None
) -> float:
    if explicit_temperature_K is not None:
        value = float(explicit_temperature_K)
    elif dataset.temperature_K is not None:
        finite = np.asarray(dataset.temperature_K, dtype=float)
        finite = finite[np.isfinite(finite)]
        if not len(finite):
            raise ValueError("GEMDAT free energy requires a finite temperature")
        value = float(np.mean(finite))
    else:
        raise ValueError("GEMDAT free energy/pathway analysis requires temperature_K")
    if not np.isfinite(value) or value <= 0:
        raise ValueError("temperature_K must be finite and positive")
    return value


def _path_arrays(path, lattice) -> dict[str, np.ndarray]:
    voxels = np.asarray(path.sites, dtype=int)
    dimensions = np.asarray(path.dims, dtype=int)
    fractional = (np.mod(voxels, dimensions) + 0.5) / dimensions
    return {
        "voxel": voxels,
        "fractional": fractional,
        "cartesian_A": np.asarray(lattice.get_cartesian_coords(fractional)),
        "free_energy_eV": np.asarray(path.energy, dtype=float),
    }


def gemdat_electrolyte(
    dataset: TrajectoryDataset,
    *,
    mobile_species: str,
    sites_path: str | Path | None = None,
    site_fractional_coordinates: Iterable[Iterable[float]] | None = None,
    discover_sites_from_density: bool = False,
    temperature_K: float | None = None,
    resolution_A: float = 0.5,
    background_level: float = 0.1,
    site_radius_A: float | None = None,
    minimal_residence: int = 0,
    jump_dimensions: int = 3,
    percolation_axes: str = "xyz",
) -> GemdatResult:
    """Run GEMDAT site mapping, transitions, jumps, and percolation."""

    _validate_mechanism_dimensions(
        jump_dimensions=jump_dimensions, percolation_axes=percolation_axes
    )
    report = require_analysis(dataset, "transport")
    if not report.fixed_cell:
        raise UnsupportedAnalysisError("GEMDAT jumps require a constant lattice")
    if resolution_A <= 0:
        raise ValueError("resolution_A must be positive")
    if minimal_residence < 0:
        raise ValueError("minimal_residence must be >= 0")
    explicit_sources = sum(
        (
            sites_path is not None,
            site_fractional_coordinates is not None,
            discover_sites_from_density,
        )
    )
    if explicit_sources != 1:
        raise ValueError(
            "Choose exactly one site source: sites_path, "
            "site_fractional_coordinates, or discover_sites_from_density=True"
        )
    view = dataset.analysis_view(include_equilibration=False)
    view.select(mobile_species)
    temperature = _temperature(view, temperature_K)
    _, _, Structure, gemdat_version = _require_gemdat()
    trajectory = _gemdat_trajectory(view, temperature_K=temperature)
    mobile_trajectory = trajectory.filter(mobile_species)
    volume = mobile_trajectory.to_volume(resolution=resolution_A)
    peaks = volume.find_peaks()
    if sites_path is not None:
        site_path = Path(sites_path).expanduser().resolve()
        if not site_path.is_file():
            raise FileNotFoundError(f"Site structure not found: {site_path}")
        sites = Structure.from_file(str(site_path))
        site_source = str(site_path)
    elif site_fractional_coordinates is not None:
        coordinates = np.asarray(list(site_fractional_coordinates), dtype=float)
        if coordinates.ndim != 2 or coordinates.shape[1] != 3 or not len(coordinates):
            raise ValueError("site_fractional_coordinates must have shape (N, 3)")
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("Site coordinates contain NaN or Inf")
        sites = Structure(
            lattice=view.cells[0],
            species=[mobile_species] * len(coordinates),
            coords=np.mod(coordinates, 1.0),
            coords_are_cartesian=False,
        )
        site_source = "explicit fractional coordinates"
    else:
        sites = volume.to_structure(
            specie=mobile_species,
            background_level=background_level,
            peaks=peaks,
            return_occupancies=True,
            n_frames=view.nframes,
        )
        site_source = "explicitly requested GEMDAT density peak segmentation"
    if not len(sites):
        raise ValueError("The selected site definition contains no sites")

    free_energy = volume.get_free_energy(temperature)
    result = GemdatResult(
        summary={
            "backend": "GEMDAT",
            "gemdat_version": gemdat_version,
            "mobile_species": mobile_species,
            "temperature_K": temperature,
            "site_source": site_source,
            "resolution_A": resolution_A,
            "number_of_sites": int(len(sites)),
            "number_of_density_peaks": int(len(peaks)),
            "jump_dimensions": jump_dimensions,
            "percolation_axes": percolation_axes,
            "gemdat_tracer_diffusivity_endpoint_promoted": False,
        },
        arrays={
            "density_counts": np.asarray(volume.data),
            "density_probability": np.asarray(volume.probability()),
            "free_energy_eV": np.asarray(free_energy.data),
            "peak_voxels": np.asarray(peaks, dtype=int),
            "cell_A": np.asarray(view.cells[0]),
        },
        structures={"sites": sites, "reference": trajectory.get_structure(0)},
    )
    for axis in percolation_axes:
        if not len(peaks):
            result.warnings.append(
                f"No density peaks were available for {axis}-axis percolation."
            )
            continue
        path = free_energy.optimal_percolating_path(peaks=peaks, percolate=axis)
        if path is None:
            result.warnings.append(f"No percolating path found along {axis}.")
            continue
        arrays = _path_arrays(path, free_energy.lattice)
        result.paths[axis] = arrays
        energy = arrays["free_energy_eV"]
        result.summary.setdefault("percolation", {})[axis] = {
            "steps": int(len(energy)),
            "barrier_eV": float(np.max(energy) - np.min(energy)),
        }

    transitions = trajectory.transitions_between_sites(
        sites, mobile_species, site_radius=site_radius_A
    )
    result.structures["occupancy"] = transitions.occupancy()
    result.tables["transition_events"] = transitions.events
    result.tables["residence_times"] = transitions.residence_time()
    result.arrays["transition_matrix"] = transitions.matrix()
    result.summary["transition_events"] = int(transitions.n_events)
    result.summary["occupancy_by_site_type"] = transitions.occupancy_by_site_type()
    result.summary["atom_locations"] = transitions.atom_locations()

    jumps = transitions.jumps(minimal_residence=minimal_residence)
    result.tables["jumps"] = jumps.data
    result.arrays["jump_matrix"] = jumps.matrix()
    result.summary.update(
        jump_summary(
            jumps,
            jump_dimensions=jump_dimensions,
            percolation_axes=percolation_axes,
        )
    )
    if jumps.n_jumps:
        n_parts = min(10, max(2, view.nframes // 4))
        if n_parts <= view.nframes:
            result.tables["jump_rates"] = jumps.rates(n_parts=n_parts)
        collective = jumps.collective()
        result.summary["solo_jump_fraction"] = float(jumps.solo_fraction)
        result.summary["collective_jump_count"] = int(collective.n_coll_jumps)
    return result
