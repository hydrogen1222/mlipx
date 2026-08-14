"""PBC-correct, directional, windowed mean-squared displacement analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from ase.geometry import find_mic

from mlipx.analysis.structure import cell_heights_A
from mlipx.analysis.units import (
    diffusion_A2_fs_to_m2_s,
    diffusion_m2_s_to_cm2_s,
)
from mlipx.analysis.validation import require_analysis

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any

    from mlipx.analysis.dataset import TrajectoryDataset

_VALID_AXES = {"x", "y", "z", "xy", "xz", "yz", "xyz"}
_AXIS_COLUMNS = {"x": 0, "y": 1, "z": 2}


def _normalize_axes(axes: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(axes, str):
        values = tuple(value.strip().lower() for value in axes.split(","))
    else:
        values = tuple(str(value).strip().lower() for value in axes)
    if not values or any(value not in _VALID_AXES for value in values):
        raise ValueError("axes must contain only x, y, z, xy, xz, yz, or xyz")
    if len(set(values)) != len(values):
        raise ValueError("axes contains duplicate selections")
    return values


def unwrap_positions(dataset: TrajectoryDataset) -> tuple[np.ndarray, dict[str, Any]]:
    """Return continuous Cartesian positions and an ambiguity diagnostic."""

    require_analysis(dataset, "msd")
    cell = dataset.cells[0]
    inverse = np.linalg.inv(cell)
    minimum_height = float(np.min(cell_heights_A(cell)))
    raw_steps = np.diff(dataset.positions, axis=0)
    if dataset.positions_convention == "wrapped":
        flat_steps = raw_steps.reshape(-1, 3)
        mic_steps, _ = find_mic(flat_steps, cell, pbc=dataset.pbc)
        cartesian_steps = np.asarray(mic_steps, dtype=float).reshape(raw_steps.shape)
        continuous = np.empty_like(dataset.positions, dtype=float)
        continuous[0] = dataset.positions[0]
        continuous[1:] = dataset.positions[0] + np.cumsum(cartesian_steps, axis=0)
        step_fractional = cartesian_steps @ inverse
        reconstruction = "consecutive ASE general minimum-image"
        exact_images_available = False
    elif dataset.positions_convention == "unwrapped":
        cartesian_steps = raw_steps
        step_fractional = cartesian_steps @ inverse
        continuous = dataset.positions.copy()
        reconstruction = "source unwrapped Cartesian coordinates"
        exact_images_available = True
    else:  # protected by validation, retained for a precise direct-call error
        raise ValueError("Position convention must be wrapped or unwrapped")
    cartesian_norms = np.linalg.norm(cartesian_steps, axis=-1)
    maximum_cartesian = float(np.max(cartesian_norms)) if cartesian_norms.size else 0.0
    maximum_fractional = (
        float(np.max(np.abs(step_fractional))) if step_fractional.size else 0.0
    )
    ratio = maximum_cartesian / (0.5 * minimum_height)
    if exact_images_available:
        level = "not_applicable_exact_unwrapped_source"
    elif ratio < 0.5:
        level = "comfortably_safe"
    elif ratio < 0.8:
        level = "warning"
    else:
        level = "strong_warning"
    diagnostics = {
        "positions_convention": dataset.positions_convention,
        "reconstruction": reconstruction,
        "exact_image_information_available": exact_images_available,
        "max_fractional_step_mic": maximum_fractional,
        "max_cartesian_step_mic_A": maximum_cartesian,
        "minimum_cell_height_A": minimum_height,
        "unwrap_safety_ratio": ratio,
        "unwrap_safety_level": level,
        "interpretation": (
            "For wrapped sources this ratio is a heuristic, not proof that hidden "
            "multiple cell crossings did not occur between saved frames."
        ),
    }
    return continuous, diagnostics


def displacement_trajectory(
    dataset: TrajectoryDataset,
    *,
    mobile_indices: Iterable[int],
    drift_reference: str = "none",
    drift_indices: Iterable[int] | None = None,
) -> dict[str, Any]:
    """Build raw/corrected mobile displacements with explicit drift semantics."""

    mobile = dataset.select(indices=mobile_indices)
    continuous, diagnostics = unwrap_positions(dataset)
    mode = str(drift_reference).lower()
    if mode not in {"none", "indices", "nonmobile"}:
        raise ValueError("drift_reference must be none, indices, or nonmobile")
    if mode == "none":
        if drift_indices is not None:
            raise ValueError("drift_indices is only valid with drift_reference=indices")
        reference = np.asarray([], dtype=int)
        drift = np.zeros((dataset.nframes, 3), dtype=float)
    elif mode == "indices":
        if drift_indices is None:
            raise ValueError("drift_reference=indices requires drift_indices")
        reference = dataset.select(indices=drift_indices)
        if np.intersect1d(mobile, reference).size:
            raise ValueError("Drift reference indices overlap the mobile selection")
        reference_displacement = continuous[:, reference] - continuous[0, reference]
        weights = dataset.masses[reference]
        drift = np.average(reference_displacement, axis=1, weights=weights)
    else:
        reference = np.setdiff1d(np.arange(dataset.natoms), mobile)
        if len(reference) == 0:
            raise ValueError(
                "drift_reference=nonmobile requires at least one nonmobile atom"
            )
        reference_displacement = continuous[:, reference] - continuous[0, reference]
        weights = dataset.masses[reference]
        drift = np.average(reference_displacement, axis=1, weights=weights)

    raw_mobile_displacements = continuous[:, mobile] - continuous[0, mobile]
    corrected = raw_mobile_displacements - drift[:, None, :]
    reference_species = sorted({dataset.symbols[index] for index in reference})
    return {
        "continuous_positions_A": continuous,
        "raw_mobile_displacements_A": raw_mobile_displacements,
        "mobile_displacements_A": corrected,
        "framework_drift_A": drift,
        "mobile_indices": mobile,
        "drift_correction": {
            "mode": mode,
            "reference_indices": reference,
            "reference_species": reference_species,
            "center_definition": "mass-weighted center-of-mass translation",
        },
        "unwrap_diagnostics": diagnostics,
    }


def direct_windowed_msd_components(displacements_A: np.ndarray) -> np.ndarray:
    """Reference O(T^2) windowed MSD for x/y/z independently."""

    positions = np.asarray(displacements_A, dtype=float)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("displacements_A must have shape (frames, atoms, 3)")
    result = np.empty((len(positions), 3), dtype=float)
    result[0] = 0.0
    for lag in range(1, len(positions)):
        delta = positions[lag:] - positions[:-lag]
        result[lag] = np.mean(delta**2, axis=(0, 1))
    return result


def fft_windowed_msd_components(displacements_A: np.ndarray) -> np.ndarray:
    """FFT-accelerated windowed MSD exactly matching the direct estimator."""

    positions = np.asarray(displacements_A, dtype=float)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("displacements_A must have shape (frames, atoms, 3)")
    n_frames = len(positions)
    transform = np.fft.rfft(positions, n=2 * n_frames, axis=0)
    autocorrelation = np.fft.irfft(
        transform * np.conjugate(transform), n=2 * n_frames, axis=0
    )[:n_frames]
    squared = positions**2
    prefix = np.concatenate(
        (np.zeros((1, positions.shape[1], 3)), np.cumsum(squared, axis=0)),
        axis=0,
    )
    result = np.empty((n_frames, 3), dtype=float)
    for lag in range(n_frames):
        origins = n_frames - lag
        later_sum = prefix[n_frames] - prefix[lag]
        earlier_sum = prefix[n_frames - lag]
        squared_displacement_sum = later_sum + earlier_sum - 2.0 * autocorrelation[lag]
        result[lag] = np.mean(squared_displacement_sum / origins, axis=0)
    result[0] = 0.0
    # Roundoff can create values around -1e-14 at lag zero/very short lags.
    tolerance = np.finfo(float).eps * max(1.0, float(np.max(np.abs(result)))) * 100
    result[np.abs(result) < tolerance] = 0.0
    return result


def _axis_msd(components: np.ndarray, axes: str) -> np.ndarray:
    columns = [_AXIS_COLUMNS[axis] for axis in axes]
    return np.sum(components[:, columns], axis=1)


def _log_log_alpha(lag_ps: np.ndarray, msd_A2: np.ndarray) -> np.ndarray:
    result = np.full_like(msd_A2, np.nan, dtype=float)
    valid = (lag_ps > 0) & (msd_A2 > 0) & np.isfinite(msd_A2)
    if np.count_nonzero(valid) < 3:
        return result
    log_time = np.log(lag_ps[valid])
    log_msd = np.log(msd_A2[valid])
    result[valid] = np.gradient(log_msd, log_time)
    return result


def diagnostic_linear_diffusion_fit(
    lag_ps: np.ndarray,
    msd_A2: np.ndarray,
    *,
    axes: str,
    fit_start_ps: float,
    fit_stop_ps: float,
    fit_window_source: str = "explicit",
) -> dict[str, Any]:
    """Explicit-range OLS diagnostic; not a publication uncertainty model."""

    if axes not in _VALID_AXES:
        raise ValueError("Invalid diffusion axes")
    if fit_start_ps < 0 or fit_stop_ps <= fit_start_ps:
        raise ValueError("fit_stop_ps must be greater than fit_start_ps >= 0")
    maximum_lag_ps = float(np.max(lag_ps))
    tolerance = max(1.0e-12, abs(maximum_lag_ps) * 1.0e-12)
    if fit_stop_ps > maximum_lag_ps + tolerance:
        raise ValueError(
            f"fit_stop_ps {fit_stop_ps:g} exceeds the maximum available lag "
            f"time {maximum_lag_ps:g} ps"
        )
    mask = (lag_ps >= fit_start_ps) & (lag_ps <= fit_stop_ps) & np.isfinite(msd_A2)
    if np.count_nonzero(mask) < 3:
        raise ValueError("Diagnostic diffusion fit requires at least three lag points")
    x = lag_ps[mask]
    y = msd_A2[mask]
    slope_A2_ps, intercept_A2 = np.polyfit(x, y, 1)
    predicted = slope_A2_ps * x + intercept_A2
    residual = float(np.sum((y - predicted) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    dimensions = len(axes)
    diffusion_A2_ps = slope_A2_ps / (2.0 * dimensions)
    diffusion_m2_s = diffusion_A2_fs_to_m2_s(diffusion_A2_ps / 1000.0)
    alpha = _log_log_alpha(lag_ps, msd_A2)
    finite_alpha = alpha[mask & np.isfinite(alpha)]
    return {
        "estimator": "diagnostic_linear_diffusion_fit",
        "publication_grade": False,
        "axes": axes,
        "dimensions": dimensions,
        "fit_start_ps": float(fit_start_ps),
        "fit_stop_ps": float(fit_stop_ps),
        "fit_window_source": fit_window_source,
        "actual_fit_start_ps": float(x[0]),
        "actual_fit_stop_ps": float(x[-1]),
        "fit_points": int(np.count_nonzero(mask)),
        "slope_A2_ps": float(slope_A2_ps),
        "intercept_A2": float(intercept_A2),
        "r_squared": 1.0 - residual / total if total > 0 else 1.0,
        "D_diagnostic_m2_s": diffusion_m2_s,
        "D_diagnostic_cm2_s": diffusion_m2_s_to_cm2_s(diffusion_m2_s),
        "self_diffusion_coefficient_m2_s": diffusion_m2_s,
        "self_diffusion_coefficient_cm2_s": diffusion_m2_s_to_cm2_s(diffusion_m2_s),
        "mean_log_log_alpha_in_fit": (
            float(np.mean(finite_alpha)) if len(finite_alpha) else None
        ),
        "diffusive_regime_warning": bool(
            len(finite_alpha) and not 0.9 <= float(np.mean(finite_alpha)) <= 1.1
        ),
    }


def calculate_msd(
    dataset: TrajectoryDataset,
    *,
    mobile_species: str | None = None,
    mobile_indices: Iterable[int] | None = None,
    axes: str | Iterable[str] = "xyz",
    drift_reference: str = "none",
    drift_indices: Iterable[int] | None = None,
    method: str = "fft",
    include_equilibration: bool = False,
    start: int | None = None,
    stop: int | None = None,
    fit_start_ps: float | None = None,
    fit_stop_ps: float | None = None,
) -> dict[str, Any]:
    """Calculate directional MSD and, only when requested, a diagnostic fit."""

    selected_axes = _normalize_axes(axes)
    if (fit_start_ps is None) != (fit_stop_ps is None):
        raise ValueError(
            "Both fit_start_ps and fit_stop_ps are required for a diagnostic fit"
        )
    fit_window_source = "explicit" if fit_start_ps is not None else None
    view = dataset.analysis_view(
        include_equilibration=include_equilibration, start=start, stop=stop
    )
    mobile = view.select(mobile_species, indices=mobile_indices)
    prepared = displacement_trajectory(
        view,
        mobile_indices=mobile,
        drift_reference=drift_reference,
        drift_indices=drift_indices,
    )
    if method == "fft":
        components = fft_windowed_msd_components(prepared["mobile_displacements_A"])
    elif method == "direct":
        components = direct_windowed_msd_components(prepared["mobile_displacements_A"])
    else:
        raise ValueError("MSD method must be fft or direct")
    lag_ps = np.arange(view.nframes, dtype=float) * view.frame_interval_fs / 1000.0
    values = {axis: _axis_msd(components, axis) for axis in selected_axes}
    alpha = {axis: _log_log_alpha(lag_ps, values[axis]) for axis in selected_axes}
    if fit_start_ps is None:
        fit_window = None
        fits: dict[str, dict[str, Any]] = {}
    else:
        fit_window = {"start": float(fit_start_ps), "stop": float(fit_stop_ps)}
        fits = {
            axis: diagnostic_linear_diffusion_fit(
                lag_ps,
                values[axis],
                axes=axis,
                fit_start_ps=fit_start_ps,
                fit_stop_ps=fit_stop_ps,
                fit_window_source="explicit",
            )
            for axis in selected_axes
        }
    return {
        "lag_time_ps": lag_ps,
        "time_origin_counts": view.nframes - np.arange(view.nframes),
        "msd_x_A2": components[:, 0],
        "msd_y_A2": components[:, 1],
        "msd_z_A2": components[:, 2],
        "msd_by_axes_A2": values,
        "log_log_alpha_by_axes": alpha,
        "fit_window_ps": fit_window,
        "fit_window_source": fit_window_source,
        "diagnostic_linear_diffusion_fits": fits,
        "method": f"{method}_windowed_msd",
        "mobile_species": mobile_species,
        "mobile_indices": mobile,
        "selected_axes": selected_axes,
        "analysis_phase": "all" if include_equilibration else "production",
        "drift_correction": prepared["drift_correction"],
        "framework_drift_A": prepared["framework_drift_A"],
        "unwrap_diagnostics": prepared["unwrap_diagnostics"],
    }
