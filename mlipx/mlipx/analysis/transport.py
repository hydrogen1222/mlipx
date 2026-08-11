"""Tracer transport, kinisi integration, and explicitly named conductivity."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

import numpy as np
from ase import Atoms

from mlipx.analysis.msd import unwrap_positions
from mlipx.analysis.units import (
    BOLTZMANN_J_K,
    ELEMENTARY_CHARGE_C,
    S_M_TO_MS_CM,
    S_M_TO_S_CM,
)
from mlipx.analysis.validation import (
    OptionalDependencyError,
    UnsupportedAnalysisError,
    require_analysis,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any

    from mlipx.analysis.dataset import TrajectoryDataset


DEFAULT_MAX_NATIVE_KINISI_LAG_POINTS = 1000
_FRAME_OFFSET_ATOL = 1.0e-8
_FRAME_OFFSET_RTOL = 1.0e-9


def particle_number_density_m3(
    dataset: TrajectoryDataset, *, mobile_species: str
) -> float:
    """Return N_mobile/V for a fixed-cell bulk trajectory in m^-3."""

    report = require_analysis(dataset, "transport")
    if not report.fixed_cell or not report.three_dimensional_pbc:
        raise UnsupportedAnalysisError(
            "Particle number density for transport requires fixed-cell 3-D PBC"
        )
    count = len(dataset.select(mobile_species))
    volume_A3 = abs(float(np.linalg.det(dataset.cells[0])))
    if volume_A3 <= 0:
        raise ValueError("Cell volume must be positive")
    return count / (volume_A3 * 1.0e-30)


def nernst_einstein_tracer_conductivity(
    *,
    particle_density_m3: float,
    tracer_diffusion_m2_s: float,
    temperature_K: float,
    ionic_charge_e: float | None,
) -> dict[str, float]:
    """Calculate sigma_NE_tracer=n(z e)^2 D_tracer/(k_B T)."""

    values = (particle_density_m3, tracer_diffusion_m2_s, temperature_K)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("Nernst-Einstein inputs must be finite")
    if particle_density_m3 <= 0:
        raise ValueError("particle_density_m3 must be positive")
    if tracer_diffusion_m2_s < 0:
        raise ValueError("tracer_diffusion_m2_s must be non-negative")
    if temperature_K <= 0:
        raise ValueError("temperature_K must be positive")
    if ionic_charge_e is None:
        raise ValueError("ionic_charge_e is required for Nernst-Einstein conductivity")
    if not np.isfinite(ionic_charge_e) or ionic_charge_e == 0:
        raise ValueError("ionic_charge_e must be finite and non-zero")
    charge_C = float(ionic_charge_e) * ELEMENTARY_CHARGE_C
    sigma_S_m = (
        particle_density_m3
        * charge_C**2
        * tracer_diffusion_m2_s
        / (BOLTZMANN_J_K * temperature_K)
    )
    return {
        "sigma_NE_tracer_S_m": sigma_S_m,
        "sigma_NE_tracer_S_cm": sigma_S_m * S_M_TO_S_CM,
        "sigma_NE_tracer_mS_cm": sigma_S_m * S_M_TO_MS_CM,
    }


def _require_kinisi():
    try:
        import scipp as sc
        from kinisi.analyze import ConductivityAnalyzer, DiffusionAnalyzer
    except ImportError as exc:  # pragma: no cover - depends on optional env
        raise OptionalDependencyError(
            "kinisi transport is unavailable. Install mlipx[transport]."
        ) from exc
    try:
        kinisi_version = version("kinisi")
    except PackageNotFoundError:  # pragma: no cover
        kinisi_version = "unknown"
    if kinisi_version != "unknown" and not kinisi_version.startswith("2."):
        raise OptionalDependencyError(
            f"Unsupported kinisi version {kinisi_version}; Analysis v2 is tested "
            "with kinisi>=2.0.5,<3."
        )
    return sc, DiffusionAnalyzer, ConductivityAnalyzer, kinisi_version


def _frame_offset(
    value_ps: float, *, frame_interval_fs: float, label: str
) -> int:
    """Convert a ps time to an exact saved-frame offset."""

    offset = value_ps * 1000.0 / frame_interval_fs
    rounded = int(np.rint(offset))
    if not np.isclose(
        offset,
        rounded,
        rtol=_FRAME_OFFSET_RTOL,
        atol=_FRAME_OFFSET_ATOL,
    ):
        raise ValueError(
            f"{label}={value_ps:g} ps is incompatible with the trajectory "
            f"frame interval {frame_interval_fs:g} fs; it must map to an "
            "integer frame offset"
        )
    return rounded


def _resolve_kinisi_lag_grid(
    *,
    frame_interval_fs: float,
    total_duration_ps: float,
    fit_start_ps: float,
    lag_step_ps: float | None = None,
    lag_stop_ps: float | None = None,
    native_lag_guard: int = DEFAULT_MAX_NATIVE_KINISI_LAG_POINTS,
) -> dict[str, Any]:
    """Resolve a kinisi lag grid without importing kinisi or scipp.

    Custom grids are represented by integer offsets on the saved-frame time
    axis.  Omitting both custom parameters preserves kinisi's native ``dt``
    behavior, subject to the explicit resource guard.
    """

    if not np.isfinite(frame_interval_fs) or frame_interval_fs <= 0:
        raise ValueError("frame_interval_fs must be finite and positive")
    if not np.isfinite(total_duration_ps) or total_duration_ps <= 0:
        raise ValueError("total_duration_ps must be finite and positive")
    if not np.isfinite(fit_start_ps) or fit_start_ps < 0:
        raise ValueError("fit_start_ps must be finite and >= 0")
    if fit_start_ps >= total_duration_ps:
        raise ValueError(
            f"fit_start_ps must be less than production duration "
            f"{total_duration_ps:g} ps"
        )
    if native_lag_guard < 1:
        raise ValueError("native_lag_guard must be a positive integer")

    interval_ps = frame_interval_fs / 1000.0
    fit_start_frames = _frame_offset(
        fit_start_ps,
        frame_interval_fs=frame_interval_fs,
        label="fit_start_ps",
    )
    estimated_native_lag_points = max(
        0,
        int(np.floor(total_duration_ps / interval_ps + _FRAME_OFFSET_ATOL)),
    )
    if lag_step_ps is None and lag_stop_ps is None:
        if estimated_native_lag_points > native_lag_guard:
            raise ValueError(
                f"kinisi's default lag grid would contain ~"
                f"{estimated_native_lag_points} lag points for a "
                f"{total_duration_ps:g} ps trajectory sampled every "
                f"{interval_ps:g} ps.\n\n"
                "This is too large for mlipx's guarded transport path "
                "because kinisi performs covariance-aware analysis over the "
                "lag grid.\n\n"
                "Specify an explicit sparse grid, for example:\n"
                "  --lag-step-ps 1 --lag-stop-ps 200"
            )
        return {
            "mode": "kinisi_default",
            "lag_frame_indices": None,
            "lag_times_fs": None,
            "requested_step_ps": None,
            "requested_stop_ps": None,
            "actual_step_ps": None,
            "actual_stop_ps": None,
            "n_lag_points": None,
            "fit_start_frame_index": fit_start_frames,
            "estimated_n_lag_points": estimated_native_lag_points,
            "frame_interval_ps": interval_ps,
        }

    if (lag_step_ps is None) != (lag_stop_ps is None):
        raise ValueError(
            "--lag-step-ps and --lag-stop-ps must be provided together"
        )
    assert lag_step_ps is not None
    assert lag_stop_ps is not None
    if not np.isfinite(lag_step_ps) or lag_step_ps <= 0:
        raise ValueError("lag_step_ps must be finite and positive")
    if not np.isfinite(lag_stop_ps) or lag_stop_ps <= 0:
        raise ValueError("lag_stop_ps must be finite and positive")
    if lag_stop_ps <= fit_start_ps:
        raise ValueError("lag_stop_ps must be greater than fit_start_ps")
    if lag_stop_ps > total_duration_ps and not np.isclose(
        lag_stop_ps,
        total_duration_ps,
        rtol=_FRAME_OFFSET_RTOL,
        atol=interval_ps * _FRAME_OFFSET_ATOL,
    ):
        raise ValueError(
            f"lag_stop_ps={lag_stop_ps:g} ps exceeds production duration "
            f"{total_duration_ps:g} ps"
        )

    step_frames = _frame_offset(
        lag_step_ps,
        frame_interval_fs=frame_interval_fs,
        label="lag_step_ps",
    )
    stop_frames = _frame_offset(
        lag_stop_ps,
        frame_interval_fs=frame_interval_fs,
        label="lag_stop_ps",
    )
    if step_frames < 1:
        raise ValueError("lag_step_ps must be at least one frame interval")
    if stop_frames < 1:
        raise ValueError("lag_stop_ps must be at least one frame interval")
    if fit_start_frames < 0:
        raise ValueError("fit_start_ps must map to a non-negative frame offset")

    lag_indices = np.arange(
        step_frames,
        stop_frames + 1,
        step_frames,
        dtype=int,
    )
    if fit_start_frames > 0 and fit_start_frames not in lag_indices:
        lag_indices = np.concatenate((lag_indices, np.asarray([fit_start_frames])))
        lag_indices.sort()
    lag_indices = np.unique(lag_indices)
    if not len(lag_indices):
        raise ValueError("The custom lag grid contains no positive lag points")
    lag_times_fs = lag_indices.astype(float) * frame_interval_fs
    actual_stop_ps = float(lag_times_fs[-1] / 1000.0)
    return {
        "mode": "custom",
        "lag_frame_indices": lag_indices,
        "lag_times_fs": lag_times_fs,
        "requested_step_ps": float(lag_step_ps),
        "requested_stop_ps": float(lag_stop_ps),
        "actual_step_ps": float(lag_step_ps),
        "actual_stop_ps": actual_stop_ps,
        "n_lag_points": int(len(lag_indices)),
        "fit_start_frame_index": fit_start_frames,
        "estimated_n_lag_points": estimated_native_lag_points,
        "frame_interval_ps": interval_ps,
    }


def _kinisi_frames_and_indices(
    dataset: TrajectoryDataset,
    *,
    mobile: np.ndarray,
    drift_reference: str,
    drift_indices: Iterable[int] | None,
):
    mode = drift_reference.lower()
    if mode == "none":
        reference = np.asarray([], dtype=int)
    elif mode == "nonmobile":
        reference = np.setdiff1d(np.arange(dataset.natoms), mobile)
        if not len(reference):
            raise ValueError(
                "drift_reference=nonmobile requires at least one framework atom"
            )
    elif mode == "indices":
        if drift_indices is None:
            raise ValueError("drift_reference=indices requires drift_indices")
        reference = dataset.select(indices=drift_indices)
        if np.intersect1d(mobile, reference).size:
            raise ValueError("Drift reference indices overlap the mobile selection")
    else:
        raise ValueError("drift_reference must be none, indices, or nonmobile")
    retained = np.concatenate((mobile, reference))
    frames: list[Atoms] = []
    for positions, cell in zip(dataset.positions, dataset.cells, strict=True):
        atoms = Atoms(
            symbols=[dataset.symbols[index] for index in retained],
            positions=positions[retained],
            cell=cell,
            pbc=dataset.pbc,
        )
        atoms.wrap()
        frames.append(atoms)
    # Mobile atoms are deliberately first. kinisi's parser then applies its
    # supported drift correction exactly once using the retained complement.
    local_mobile = np.arange(len(mobile), dtype=int)
    return frames, local_mobile, reference


def _sample_summary(variable, *, target_unit: str) -> dict[str, Any]:
    converted = variable.to(unit=target_unit)
    values = np.asarray(converted.values, dtype=float).reshape(-1)
    return {
        "unit": target_unit,
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "median": float(np.median(values)),
        "credible_interval_95": [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ],
        "posterior_samples": int(len(values)),
    }


def _production_temperature(
    dataset: TrajectoryDataset, explicit_temperature_K: float | None
) -> tuple[float, str]:
    if explicit_temperature_K is not None:
        temperature = float(explicit_temperature_K)
        source = "explicit analysis request"
    elif dataset.temperature_K is not None:
        finite = np.asarray(dataset.temperature_K, dtype=float)
        finite = finite[np.isfinite(finite)]
        if not len(finite):
            raise ValueError(
                "Trajectory has no finite production temperature; provide temperature_K"
            )
        temperature = float(np.mean(finite))
        source = "production mean trajectory temperature"
    else:
        raise ValueError(
            "Temperature is required. Imported trajectories do not default to 300 K."
        )
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature_K must be finite and positive")
    return temperature, source


def kinisi_transport(
    dataset: TrajectoryDataset,
    *,
    mobile_species: str,
    ionic_charge_e: float | None,
    fit_start_ps: float,
    dimensions: str = "xyz",
    lag_step_ps: float | None = None,
    lag_stop_ps: float | None = None,
    drift_reference: str = "none",
    drift_indices: Iterable[int] | None = None,
    temperature_K: float | None = None,
    collective_conductivity: bool = False,
    random_seed: int = 0,
    n_samples: int = 1000,
    n_walkers: int = 32,
    n_burn: int = 500,
    n_thin: int = 10,
) -> dict[str, Any]:
    """Estimate publication-grade scalar tracer diffusion with kinisi."""

    if not dimensions or any(axis not in "xyz" for axis in dimensions):
        raise ValueError("dimensions must be a non-empty subset of xyz")
    if len(set(dimensions)) != len(dimensions):
        raise ValueError("dimensions contains duplicate axes")
    if fit_start_ps is None or fit_start_ps < 0:
        raise ValueError("fit_start_ps must be explicitly provided and >= 0")
    if n_samples < 1 or n_walkers < 2 or n_burn < 0 or n_thin < 1:
        raise ValueError("Invalid kinisi sampling parameters")
    require_analysis(dataset, "transport")
    view = dataset.analysis_view(include_equilibration=False)
    total_duration_ps = (view.times_fs[-1] - view.times_fs[0]) / 1000.0
    lag_grid = _resolve_kinisi_lag_grid(
        frame_interval_fs=view.frame_interval_fs,
        total_duration_ps=float(total_duration_ps),
        fit_start_ps=float(fit_start_ps),
        lag_step_ps=lag_step_ps,
        lag_stop_ps=lag_stop_ps,
    )
    mobile = view.select(mobile_species)
    _, unwrap_diagnostics = unwrap_positions(view)
    if (
        view.positions_convention == "wrapped"
        and unwrap_diagnostics["unwrap_safety_ratio"] > 0.8
    ):
        raise UnsupportedAnalysisError(
            "Publication transport refused: wrapped-frame unwrap safety ratio "
            "exceeds 0.8. Save frames more frequently or provide exact unwrapped "
            "positions/image counters."
        )
    temperature, temperature_source = _production_temperature(view, temperature_K)
    if ionic_charge_e is None:
        raise ValueError(
            "ionic_charge_e is required because transport reports sigma_NE_tracer"
        )
    sc, DiffusionAnalyzer, ConductivityAnalyzer, kinisi_version = _require_kinisi()
    frames, local_mobile, reference = _kinisi_frames_and_indices(
        view,
        mobile=mobile,
        drift_reference=drift_reference,
        drift_indices=drift_indices,
    )
    if (
        view.md_timestep_fs is not None
        and view.frame_stride_steps is not None
        and np.isclose(
            view.md_timestep_fs * view.frame_stride_steps,
            view.frame_interval_fs,
            rtol=1.0e-6,
            atol=max(1.0e-10, view.frame_interval_fs * 1.0e-10),
        )
    ):
        kinisi_time_step_fs = view.md_timestep_fs
        kinisi_step_skip = view.frame_stride_steps
        mapping_source = "MD timestep and saved frame stride"
    else:
        kinisi_time_step_fs = view.frame_interval_fs
        kinisi_step_skip = 1
        mapping_source = "explicit frame interval; original MD stride unavailable"
    common = {
        "trajectory": frames,
        "specie": None,
        "time_step": sc.scalar(kinisi_time_step_fs, unit="fs"),
        "step_skip": sc.scalar(kinisi_step_skip, unit="dimensionless"),
        "dimension": dimensions,
        "progress": False,
    }
    if lag_grid["mode"] == "custom":
        common["dt"] = sc.array(
            dims=["time interval"],
            values=lag_grid["lag_times_fs"],
            unit="fs",
        )
    indices_variable = sc.array(
        dims=["particle"], values=local_mobile, unit="dimensionless"
    )
    analyzer = DiffusionAnalyzer.from_ase(**common, specie_indices=indices_variable)
    start_dt = sc.scalar(fit_start_ps * 1000.0, unit="fs")
    mcmc = {
        "n_samples": n_samples,
        "n_walkers": n_walkers,
        "n_burn": n_burn,
        "n_thin": n_thin,
        "progress": False,
        "random_state": np.random.RandomState(random_seed),
    }
    analyzer.diffusion(start_dt, **mcmc)
    diffusion_m2_s = _sample_summary(analyzer.D, target_unit="m^2/s")
    diffusion_cm2_s = _sample_summary(analyzer.D, target_unit="cm^2/s")
    lag_time_ps = np.asarray(analyzer.dt.to(unit="ps").values, dtype=float)
    n_lag_points_total = int(len(lag_time_ps))
    fit_mask = lag_time_ps >= float(fit_start_ps)
    n_lag_points_in_fit = int(np.count_nonzero(fit_mask))
    if n_lag_points_in_fit == 0:
        raise ValueError(
            "kinisi lag grid contains no points at or above fit_start_ps"
        )
    actual_fit_stop_ps = float(lag_time_ps[-1])
    lag_grid_metadata = {
        "mode": lag_grid["mode"],
        "requested_step_ps": lag_grid["requested_step_ps"],
        "requested_stop_ps": lag_grid["requested_stop_ps"],
        "actual_step_ps": lag_grid["actual_step_ps"],
        "actual_min_ps": float(lag_time_ps[0]),
        "actual_max_ps": actual_fit_stop_ps,
        "n_lag_points_total": n_lag_points_total,
        "n_lag_points_in_fit": n_lag_points_in_fit,
        "frame_interval_ps": float(view.frame_interval_fs / 1000.0),
    }
    if lag_grid["mode"] == "kinisi_default":
        lag_grid_metadata["estimated_n_lag_points"] = lag_grid[
            "estimated_n_lag_points"
        ]
    number_density = particle_number_density_m3(view, mobile_species=mobile_species)
    nernst_einstein = nernst_einstein_tracer_conductivity(
        particle_density_m3=number_density,
        tracer_diffusion_m2_s=diffusion_m2_s["mean"],
        temperature_K=temperature,
        ionic_charge_e=ionic_charge_e,
    )
    result: dict[str, Any] = {
        "mobile_species": mobile_species,
        "analysis_phase": "production",
        "dimensions": dimensions,
        "temperature_mean_K": temperature,
        "temperature_source": temperature_source,
        "target_temperature_K": view.target_temperature_K,
        "particle_number_density_m3": number_density,
        "tracer_diffusion": {
            "backend": "kinisi",
            "kinisi_version": kinisi_version,
            "method": "covariance-aware Bayesian regression",
            "fit_start_ps": float(fit_start_ps),
            "fit_stop_ps": actual_fit_stop_ps,
            "lag_grid": lag_grid_metadata,
            "random_seed": int(random_seed),
            "D_posterior_m2_s": diffusion_m2_s,
            "D_posterior_cm2_s": diffusion_cm2_s,
        },
        "nernst_einstein": {
            "definition": "sigma_NE_tracer = n (z e)^2 D_tracer / (k_B T)",
            "ionic_charge_e": float(ionic_charge_e),
            **nernst_einstein,
        },
        "kinisi_time_mapping": {
            "time_step_fs": float(kinisi_time_step_fs),
            "step_skip": int(kinisi_step_skip),
            "resulting_frame_interval_fs": float(
                kinisi_time_step_fs * kinisi_step_skip
            ),
            "source": mapping_source,
        },
        "drift_correction": {
            "mode": drift_reference,
            "reference_indices": reference,
            "reference_species": sorted({view.symbols[index] for index in reference}),
            "backend_semantics": "kinisi mean framework displacement; applied once",
        },
        "unwrap_diagnostics": unwrap_diagnostics,
        "kinisi_resource_diagnostics": {
            "n_lag_points_total": n_lag_points_total,
            "n_lag_points_in_fit": n_lag_points_in_fit,
            "single_float64_square_matrix_lower_bound_bytes": int(
                8 * n_lag_points_in_fit**2
            ),
        },
        "lag_time_ps": lag_time_ps,
        "kinisi_msd_A2": np.asarray(
            analyzer.msd.to(unit="angstrom^2").values, dtype=float
        ),
        "kinisi_msd_variance_A4": (
            np.asarray(analyzer.msd.to(unit="angstrom^2").variances, dtype=float)
            if analyzer.msd.variances is not None
            else np.full(analyzer.msd.shape, np.nan)
        ),
        "quality": {
            "fixed_cell": True,
            "uniform_sampling": True,
            "unwrap_warning": bool(
                view.positions_convention == "wrapped"
                and unwrap_diagnostics["unwrap_safety_ratio"] >= 0.5
            ),
        },
    }
    if collective_conductivity:
        conductivity = ConductivityAnalyzer.from_ase(
            **common,
            ionic_charge=float(ionic_charge_e) * sc.Unit("e"),
            species_indices=indices_variable,
        )
        conductivity.conductivity(
            start_dt,
            temperature=sc.scalar(temperature, unit="K"),
            **{**mcmc, "random_state": np.random.RandomState(random_seed + 1)},
        )
        result["collective_conductivity"] = {
            "backend": "kinisi",
            "definition": "collective charge-displacement conductivity",
            "sigma_collective_mS_cm_posterior": _sample_summary(
                conductivity.sigma, target_unit="mS/cm"
            ),
        }
    return result
