"""Dependency-light trajectory analyses shared by every backend."""

from __future__ import annotations

from typing import Any

import numpy as np

from mlipx.analysis.dataset import TrajectoryDataset


def frame_slice(
    dataset: TrajectoryDataset,
    *,
    start: int = 0,
    stop: int | None = None,
    stride: int = 1,
) -> slice:
    if stride < 1:
        raise ValueError("stride must be >= 1")
    stop = dataset.nframes if stop is None else stop
    if start < 0 or stop > dataset.nframes or start >= stop:
        raise ValueError(
            f"Invalid frame range [{start}, {stop}) for {dataset.nframes} frames"
        )
    return slice(start, stop, stride)


def thermodynamics_summary(
    dataset: TrajectoryDataset,
    *,
    start: int = 0,
    stop: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Select thermodynamic columns and summarize finite values."""
    selection = frame_slice(dataset, start=start, stop=stop)
    columns = {
        key: values[selection]
        for key, values in dataset.thermodynamics.items()
        if len(values) == dataset.nframes
    }
    summary: dict[str, Any] = {
        "start_frame": start,
        "stop_frame": dataset.nframes if stop is None else stop,
        "columns": {},
    }
    for key, values in columns.items():
        finite = values[np.isfinite(values)]
        if not len(finite):
            continue
        summary["columns"][key] = {
            "mean": float(np.mean(finite)),
            "std": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
            "last": float(finite[-1]),
        }
    if "total_energy_eV" in columns and len(columns["total_energy_eV"]) > 1:
        time = dataset.time_fs[selection]
        energy = columns["total_energy_eV"]
        finite = np.isfinite(time) & np.isfinite(energy)
        if finite.sum() > 1 and np.ptp(time[finite]) > 0:
            slope = np.polyfit(time[finite], energy[finite], 1)[0]
            summary["total_energy_drift_eV_per_ps"] = float(slope * 1000)
    return columns, summary


def _kabsch(moving: np.ndarray, reference: np.ndarray) -> np.ndarray:
    covariance = moving.T @ reference
    left, _, right = np.linalg.svd(covariance)
    rotation = left @ right
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right
    return rotation


def rmsd_rmsf(
    dataset: TrajectoryDataset,
    *,
    species: str | None = None,
    framework: str | None = None,
    start: int = 0,
    stop: int | None = None,
    align: bool = False,
) -> dict[str, np.ndarray]:
    """Calculate PBC-aware RMSD and per-atom RMSF on continuous positions."""
    selection = frame_slice(dataset, start=start, stop=stop)
    atom_indices = dataset.select(species)
    positions, drift = dataset.corrected_positions(framework=framework)
    selected = positions[selection][:, atom_indices].copy()
    reference = selected[0].copy()
    if align:
        ref_center = reference.mean(axis=0)
        reference_centered = reference - ref_center
        for index, frame in enumerate(selected):
            frame_center = frame.mean(axis=0)
            centered = frame - frame_center
            selected[index] = (
                centered @ _kabsch(centered, reference_centered) + ref_center
            )
    displacement = selected - reference
    rmsd = np.sqrt(np.mean(np.sum(displacement**2, axis=2), axis=1))
    mean_position = selected.mean(axis=0)
    rmsf = np.sqrt(np.mean(np.sum((selected - mean_position) ** 2, axis=2), axis=0))
    return {
        "time_fs": dataset.time_fs[selection],
        "rmsd_A": rmsd,
        "atom_index": atom_indices,
        "rmsf_A": rmsf,
        "drift_A": drift[selection],
    }


def _pair_distances(
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
    indices_a: np.ndarray,
    indices_b: np.ndarray,
    same_selection: bool,
) -> np.ndarray:
    delta = positions[indices_b][None, :, :] - positions[indices_a][:, None, :]
    fractional = delta @ np.linalg.inv(cell)
    fractional[..., pbc] -= np.round(fractional[..., pbc])
    distances = np.linalg.norm(fractional @ cell, axis=-1)
    if same_selection:
        distances = distances[~np.eye(len(indices_a), dtype=bool)]
    return distances.reshape(-1)


def radial_distribution(
    dataset: TrajectoryDataset,
    *,
    species_a: str,
    species_b: str,
    r_max: float = 8.0,
    bins: int = 200,
    start: int = 0,
    stop: int | None = None,
    stride: int = 1,
) -> dict[str, np.ndarray]:
    """Compute partial RDF and direct cumulative coordination number."""
    if r_max <= 0 or bins < 2:
        raise ValueError("r_max must be positive and bins must be >= 2")
    if not dataset.pbc.all():
        raise ValueError("Bulk RDF normalization currently requires 3-D periodicity")
    frame_indices = np.arange(dataset.nframes)[
        frame_slice(dataset, start=start, stop=stop, stride=stride)
    ]
    safe_radius = float("inf")
    for cell in dataset.cells[frame_indices]:
        volume = abs(np.linalg.det(cell))
        face_areas = np.asarray(
            [
                np.linalg.norm(np.cross(cell[1], cell[2])),
                np.linalg.norm(np.cross(cell[0], cell[2])),
                np.linalg.norm(np.cross(cell[0], cell[1])),
            ]
        )
        safe_radius = min(safe_radius, 0.5 * float(np.min(volume / face_areas)))
    if r_max > safe_radius + 1e-10:
        raise ValueError(
            f"r_max={r_max:g} Å exceeds the minimum-image limit "
            f"({safe_radius:g} Å) for this cell"
        )
    indices_a = dataset.select(species_a)
    indices_b = dataset.select(species_b)
    same = np.array_equal(indices_a, indices_b)
    edges = np.linspace(0.0, r_max, bins + 1)
    counts = np.zeros(bins, dtype=float)
    normalization = np.zeros(bins, dtype=float)
    shell_volume = 4.0 * np.pi / 3.0 * (edges[1:] ** 3 - edges[:-1] ** 3)
    for frame in frame_indices:
        distances = _pair_distances(
            dataset.positions[frame],
            dataset.cells[frame],
            dataset.pbc,
            indices_a,
            indices_b,
            same,
        )
        counts += np.histogram(distances, bins=edges)[0]
        n_b_effective = len(indices_b) - (1 if same else 0)
        density_b = n_b_effective / abs(np.linalg.det(dataset.cells[frame]))
        normalization += len(indices_a) * density_b * shell_volume
    rdf = np.divide(
        counts,
        normalization,
        out=np.zeros_like(counts),
        where=normalization > 0,
    )
    coordination = np.cumsum(counts) / (len(frame_indices) * len(indices_a))
    return {
        "r_A": 0.5 * (edges[1:] + edges[:-1]),
        "g_r": rdf,
        "coordination_number": coordination,
        "raw_pair_counts": counts,
    }


def _msd_components_fft(positions: np.ndarray) -> np.ndarray:
    """Time-origin-averaged squared displacement for each particle/component."""
    n_frames = len(positions)
    transform = np.fft.rfft(positions, n=2 * n_frames, axis=0)
    autocorrelation = np.fft.irfft(
        transform * np.conjugate(transform), n=2 * n_frames, axis=0
    )[:n_frames]
    squared = positions**2
    prefix = np.concatenate(
        [np.zeros((1, *squared.shape[1:])), np.cumsum(squared, axis=0)], axis=0
    )
    lag = np.arange(n_frames)
    first = prefix[n_frames - lag]
    second = prefix[n_frames] - prefix[lag]
    denominator = (n_frames - lag)[:, None, None]
    result = np.maximum((first + second - 2.0 * autocorrelation) / denominator, 0.0)
    result[0] = 0.0
    return result


def mean_squared_displacement(
    dataset: TrajectoryDataset,
    *,
    species: str,
    framework: str | None = None,
    dimensions: str = "xyz",
    start: int = 0,
    stop: int | None = None,
) -> dict[str, np.ndarray]:
    """Compute time-origin-averaged tracer MSD using an FFT algorithm."""
    if (
        not dimensions
        or any(axis not in "xyz" for axis in dimensions)
        or len(set(dimensions)) != len(dimensions)
    ):
        raise ValueError("dimensions must be a non-empty subset of 'xyz'")
    dataset.require_time()
    selection = frame_slice(dataset, start=start, stop=stop)
    indices = dataset.select(species)
    corrected, drift = dataset.corrected_positions(framework=framework)
    positions = corrected[selection][:, indices]
    components = _msd_components_fft(positions)
    axes = np.asarray(["xyz".index(axis) for axis in dimensions], dtype=int)
    selected_components = components[:, :, axes]
    per_particle = selected_components.sum(axis=2)
    directional = components.mean(axis=1)
    total = per_particle.mean(axis=1)
    lag_time_fs = np.arange(len(total), dtype=float) * dataset.require_time()
    return {
        "lag_time_fs": lag_time_fs,
        "msd_A2": total,
        "msd_x_A2": directional[:, 0],
        "msd_y_A2": directional[:, 1],
        "msd_z_A2": directional[:, 2],
        "per_particle_msd_A2": per_particle,
        "atom_index": indices,
        "drift_A": drift[selection],
    }


def fit_diffusion(
    lag_time_fs: np.ndarray,
    msd_A2: np.ndarray,
    *,
    dimensions: int = 3,
    fit_start_fs: float | None = None,
    fit_stop_fs: float | None = None,
) -> dict[str, float]:
    """Fit the Einstein slope; kinisi should be preferred for final uncertainty."""
    if dimensions not in {1, 2, 3}:
        raise ValueError("dimensions must be 1, 2, or 3")
    start = 0.2 * float(lag_time_fs[-1]) if fit_start_fs is None else fit_start_fs
    stop = float(lag_time_fs[-1]) if fit_stop_fs is None else fit_stop_fs
    mask = (
        np.isfinite(lag_time_fs)
        & np.isfinite(msd_A2)
        & (lag_time_fs >= start)
        & (lag_time_fs <= stop)
    )
    x = lag_time_fs[mask]
    y = msd_A2[mask]
    if len(x) < 3 or np.ptp(x) <= 0:
        raise ValueError("Diffusion fit requires at least three finite lag times")
    design = np.column_stack([x, np.ones_like(x)])
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - (slope * x + intercept)
    dof = len(x) - 2
    variance = float(np.sum(residual**2) / dof) if dof > 0 else 0.0
    slope_stderr = float(np.sqrt(variance / np.sum((x - x.mean()) ** 2)))
    total = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - float(np.sum(residual**2)) / total if total > 0 else 1.0
    # 1 A^2/fs = 1e-5 m^2/s.
    conversion = 1e-5 / (2 * dimensions)
    return {
        "fit_start_fs": float(start),
        "fit_stop_fs": float(stop),
        "slope_A2_per_fs": float(slope),
        "slope_stderr_A2_per_fs": slope_stderr,
        "intercept_A2": float(intercept),
        "r_squared": r_squared,
        "diffusivity_m2_s": float(slope * conversion),
        "diffusivity_stderr_m2_s": float(slope_stderr * conversion),
        "note": "Ordinary least-squares uncertainty ignores MSD covariance; "
        "use the kinisi task for publication-grade uncertainty.",
    }


def density_grid(
    dataset: TrajectoryDataset,
    *,
    species: str,
    grid: tuple[int, int, int] = (40, 40, 40),
    start: int = 0,
    stop: int | None = None,
    stride: int = 1,
) -> dict[str, np.ndarray]:
    """Bin wrapped fractional positions into a normalized 3-D probability grid."""
    if any(size < 2 for size in grid):
        raise ValueError("Every density-grid dimension must be >= 2")
    if not dataset.pbc.all():
        raise ValueError("Fractional density grids currently require 3-D periodicity")
    selection = frame_slice(dataset, start=start, stop=stop, stride=stride)
    indices = dataset.select(species)
    positions = dataset.positions[selection][:, indices]
    cells = dataset.cells[selection]
    fractional = np.empty_like(positions)
    for index in range(len(positions)):
        fractional[index] = positions[index] @ np.linalg.inv(cells[index])
    fractional = np.mod(fractional, 1.0).reshape(-1, 3)
    counts, edges = np.histogramdd(
        fractional,
        bins=grid,
        range=((0, 1), (0, 1), (0, 1)),
    )
    probability = counts / counts.sum() if counts.sum() else counts
    return {
        "counts": counts,
        "probability": probability,
        "x_fractional": 0.5 * (edges[0][1:] + edges[0][:-1]),
        "y_fractional": 0.5 * (edges[1][1:] + edges[1][:-1]),
        "z_fractional": 0.5 * (edges[2][1:] + edges[2][:-1]),
        "cell_A": dataset.cells[selection][0],
    }


def vacf_vdos(
    dataset: TrajectoryDataset,
    *,
    species: str | None = None,
    start: int = 0,
    stop: int | None = None,
) -> dict[str, np.ndarray]:
    """Calculate time-origin-averaged VACF and its real-valued VDOS spectrum."""
    interval = dataset.require_time()
    if dataset.velocities is None:
        raise ValueError("Trajectory does not contain velocities")
    selection = frame_slice(dataset, start=start, stop=stop)
    indices = dataset.select(species)
    velocity = dataset.velocities[selection][:, indices]
    n_frames = len(velocity)
    transform = np.fft.rfft(velocity, n=2 * n_frames, axis=0)
    correlation = np.fft.irfft(
        transform * np.conjugate(transform), n=2 * n_frames, axis=0
    )[:n_frames]
    correlation /= (n_frames - np.arange(n_frames))[:, None, None]
    vacf = correlation.sum(axis=2).mean(axis=1)
    if vacf[0] != 0:
        vacf_normalized = vacf / vacf[0]
    else:
        vacf_normalized = vacf
    windowed = vacf_normalized * np.hanning(n_frames)
    spectrum = np.maximum(np.real(np.fft.rfft(windowed)), 0.0)
    frequency_thz = np.fft.rfftfreq(n_frames, d=interval * 1e-15) / 1e12
    return {
        "lag_time_fs": np.arange(n_frames) * interval,
        "vacf": vacf,
        "vacf_normalized": vacf_normalized,
        "frequency_THz": frequency_thz,
        "vdos_arb": spectrum,
    }


def arrhenius_fit(
    temperatures_K: np.ndarray,
    diffusivities_m2_s: np.ndarray,
) -> dict[str, Any]:
    """Fit ln(D)=ln(D0)-Ea/(kB*T) with ordinary least squares."""
    temperatures = np.asarray(temperatures_K, dtype=float)
    diffusivities = np.asarray(diffusivities_m2_s, dtype=float)
    valid = (
        np.isfinite(temperatures)
        & np.isfinite(diffusivities)
        & (temperatures > 0)
        & (diffusivities > 0)
    )
    if valid.sum() < 2:
        raise ValueError("Arrhenius fitting requires at least two positive data points")
    x = 1.0 / temperatures[valid]
    y = np.log(diffusivities[valid])
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    k_b_eV_K = 8.617333262145e-5
    return {
        "temperature_K": temperatures[valid],
        "diffusivity_m2_s": diffusivities[valid],
        "inverse_temperature_K-1": x,
        "ln_diffusivity": y,
        "ln_diffusivity_fit": predicted,
        "activation_energy_eV": float(-slope * k_b_eV_K),
        "preexponential_factor_m2_s": float(np.exp(intercept)),
        "r_squared": float(
            1.0 - np.sum((y - predicted) ** 2) / np.sum((y - y.mean()) ** 2)
        )
        if np.ptp(y) > 0
        else 1.0,
    }
