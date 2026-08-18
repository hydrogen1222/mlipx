"""Tracer transport, kinisi integration, and explicitly named conductivity."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

import numpy as np
from ase import Atoms
from ase.geometry import find_mic

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
# An exact unwrapped displacement and its periodic minimum-image
# reconstruction agree to ~1e-12 A when no image crossing occurs; a real
# crossing differs by ~half a cell. 1e-6 A sits safely above roundoff and
# well below any physical image loss.
_KINISI_RECONSTRUCTION_ATOL_A = 1.0e-6


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
        from kinisi.analyze import (
            ConductivityAnalyzer,
            DiffusionAnalyzer,
            JumpDiffusionAnalyzer,
        )
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
    return sc, DiffusionAnalyzer, ConductivityAnalyzer, JumpDiffusionAnalyzer, kinisi_version


def _frame_offset(value_ps: float, *, frame_interval_fs: float, label: str) -> int:
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
            "nominal_step_ps": None,
            "actual_step_ps": None,
            "actual_stop_ps": None,
            "n_lag_points": None,
            "fit_start_frame_index": fit_start_frames,
            "fit_start_inserted": None,
            "is_uniform_grid": None,
            "estimated_n_lag_points": estimated_native_lag_points,
            "frame_interval_ps": interval_ps,
        }

    if (lag_step_ps is None) != (lag_stop_ps is None):
        raise ValueError("--lag-step-ps and --lag-stop-ps must be provided together")
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

    base_indices = np.arange(
        step_frames,
        stop_frames + 1,
        step_frames,
        dtype=int,
    )
    fit_start_inserted = bool(
        fit_start_frames > 0 and fit_start_frames not in base_indices
    )
    if fit_start_inserted:
        lag_indices = np.concatenate((base_indices, np.asarray([fit_start_frames])))
        lag_indices.sort()
    else:
        lag_indices = base_indices
    lag_indices = np.unique(lag_indices)
    if not len(lag_indices):
        raise ValueError("The custom lag grid contains no positive lag points")
    lag_times_fs = lag_indices.astype(float) * frame_interval_fs
    actual_stop_ps = float(lag_times_fs[-1] / 1000.0)
    is_uniform = not fit_start_inserted
    return {
        "mode": "custom",
        "lag_frame_indices": lag_indices,
        "lag_times_fs": lag_times_fs,
        "requested_step_ps": float(lag_step_ps),
        "requested_stop_ps": float(lag_stop_ps),
        "nominal_step_ps": float(lag_step_ps),
        "actual_step_ps": float(lag_step_ps) if is_uniform else None,
        "actual_stop_ps": actual_stop_ps,
        "n_lag_points": int(len(lag_indices)),
        "fit_start_frame_index": fit_start_frames,
        "fit_start_inserted": fit_start_inserted,
        "is_uniform_grid": is_uniform,
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
    """Build mobile-only frames with explicit unweighted drift correction.

    kinisi corrects drift by subtracting the unweighted mean displacement of
    every non-mobile atom.  Carrying that framework through its triclinic
    parser creates several ``time × atoms × 8`` arrays.  Apply the identical
    mean-displacement definition once, validate that the corrected mobile
    steps remain minimum-image reconstructible, and pass only mobile atoms to
    kinisi so its automatic complement is empty.
    """
    corrected, reference = _production_positions_with_drift(
        dataset,
        mobile=mobile,
        drift_reference=drift_reference,
        drift_indices=drift_indices,
    )
    frames: list[Atoms] = []
    symbols = [dataset.symbols[index] for index in mobile]
    for positions, cell in zip(corrected[:, mobile], dataset.cells, strict=True):
        atoms = Atoms(
            symbols=symbols,
            positions=positions,
            cell=cell,
            pbc=dataset.pbc,
        )
        atoms.wrap()
        frames.append(atoms)
    local_mobile = np.arange(len(mobile), dtype=int)
    return frames, local_mobile, reference


def _production_positions_with_drift(
    dataset: TrajectoryDataset,
    *,
    mobile: np.ndarray,
    drift_reference: str,
    drift_indices: Iterable[int] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return production positions after one explicit framework correction.

    The correction is intentionally the same unweighted reference displacement
    used by the kinisi adapter.  The returned array still contains every atom;
    this lets mechanism analysis construct a provenance-preserving GEMDAT
    trajectory without applying a second correction.
    """
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
    continuous, _ = unwrap_positions(dataset)
    if len(reference):
        reference_displacement = continuous[:, reference] - continuous[0, reference]
        drift = np.mean(reference_displacement, axis=1)
    else:
        drift = np.zeros((dataset.nframes, 3), dtype=float)
    corrected = continuous - drift[:, None, :]
    corrected_steps = np.diff(corrected[:, mobile], axis=0)
    if corrected_steps.size:
        mic_steps, _ = find_mic(
            corrected_steps.reshape(-1, 3),
            np.asarray(dataset.cells[0], dtype=float),
            pbc=np.asarray(dataset.pbc, dtype=bool),
        )
        difference = np.linalg.norm(
            corrected_steps - np.asarray(mic_steps).reshape(corrected_steps.shape),
            axis=-1,
        )
        maximum = float(np.max(difference))
        if maximum > _KINISI_RECONSTRUCTION_ATOL_A:
            raise UnsupportedAnalysisError(
                "Framework-corrected mobile displacements cannot be reconstructed "
                "from wrapped saved frames without losing an image crossing. Save "
                "the trajectory more frequently."
            )
    return corrected, reference


def _kinisi_parser_peak_bytes(*, nframes: int, natoms: int, triclinic: bool) -> int:
    """Conservative estimate of kinisi's trajectory-parser peak allocation."""
    points = max(0, int(nframes)) * max(0, int(natoms))
    if triclinic:
        # coordinates plus the integer images, Cartesian images, norms and
        # unavoidable result/temporary arrays in kinisi 2.x.
        return int(points * 512)
    # Orthorhombic parser has no eight-image expansion but still materialises
    # wrapped coordinates, differences, image corrections and cumulative sums.
    return int(points * 128)


def _validate_kinisi_periodic_reconstruction(
    dataset: TrajectoryDataset,
    unwrap_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    """Record kinisi backend position semantics; fail closed on image loss.

    kinisi's ASE backend reconstructs displacements from wrapped/scaled
    periodic coordinates, so exact unwrapped image counters are never
    consumed directly.  For an unwrapped source we verify that the periodic
    minimum-image reconstruction of every saved-frame displacement equals the
    exact displacement; otherwise exact image history would be lost and
    transport must be refused.  Wrapped sources carry no exact image counter,
    so only the heuristic unwrap safety ratio (applied by the caller) is
    available and the reconstruction equivalence is reported as null.
    """

    convention = dataset.positions_convention
    if convention == "unwrapped":
        positions = dataset.positions
        if positions.shape[0] < 2:
            raise UnsupportedAnalysisError(
                "Transport requires at least two saved frames to reconstruct "
                "displacements."
            )
        exact_steps = np.diff(positions, axis=0)
        cell = np.asarray(dataset.cells[0], dtype=float)
        pbc = np.asarray(dataset.pbc, dtype=bool)
        flat = np.asarray(exact_steps, dtype=float).reshape(-1, 3)
        mic_flat, _ = find_mic(flat, cell, pbc=pbc)
        mic_steps = mic_flat.reshape(exact_steps.shape)
        differences = np.linalg.norm(exact_steps - mic_steps, axis=-1)
        max_difference = float(np.max(differences)) if differences.size else 0.0
        n_intervals = int(exact_steps.shape[0])
        if max_difference > _KINISI_RECONSTRUCTION_ATOL_A:
            raise UnsupportedAnalysisError(
                "The source contains exact unwrapped image information, but "
                "kinisi's ASE backend reconstructs periodic displacements from "
                "wrapped/scaled coordinates. At least one saved-frame "
                "displacement is not equal to its minimum-image reconstruction, "
                "so exact image history would be lost.\n\n"
                "Use a denser saved trajectory or a future exact-displacement "
                "transport backend. Native mlipx MSD can still use the exact "
                "unwrapped coordinates."
            )
        return {
            "source_positions_convention": "unwrapped",
            "backend_input": "periodic ASE frames",
            "backend_reconstruction": "kinisi periodic displacement reconstruction",
            "exact_unwrapped_preserved_directly": False,
            "exact_unwrapped_reconstruction_equivalent": True,
            "checked_saved_intervals": n_intervals,
            "maximum_exact_vs_mic_difference_A": max_difference,
        }
    if convention != "wrapped":
        raise UnsupportedAnalysisError(
            "Kinisi transport requires an explicit wrapped or unwrapped "
            "position convention."
        )
    return {
        "source_positions_convention": "wrapped",
        "backend_input": "wrapped/scaled periodic coordinates",
        "backend_reconstruction": "kinisi periodic displacement reconstruction",
        "exact_unwrapped_preserved_directly": False,
        "exact_unwrapped_reconstruction_equivalent": None,
        "wrapped_source_safety": unwrap_diagnostics.get("unwrap_safety_level"),
    }


def _numeric_sample_summary(values: np.ndarray, *, unit: str) -> dict[str, Any]:
    """Posterior summary (mean/std/median/95% CrI) for a numeric sample array."""

    flat = np.asarray(values, dtype=float).reshape(-1)
    return {
        "unit": unit,
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat, ddof=1)) if len(flat) > 1 else 0.0,
        "median": float(np.median(flat)),
        "credible_interval_95": [
            float(np.quantile(flat, 0.025)),
            float(np.quantile(flat, 0.975)),
        ],
        "posterior_samples": int(len(flat)),
    }


def _sample_summary(variable, *, target_unit: str) -> dict[str, Any]:
    converted = variable.to(unit=target_unit)
    values = np.asarray(converted.values, dtype=float)
    return _numeric_sample_summary(values, unit=target_unit)


def _variable_values(variable, *, target_unit: str | None = None) -> np.ndarray:
    """Return a scipp variable's values as a flat float array."""

    converted = variable.to(unit=target_unit) if target_unit is not None else variable
    return np.asarray(converted.values, dtype=float).reshape(-1)


def _variable_variances(variable, *, target_unit: str | None = None) -> np.ndarray:
    """Return scipp variances, using NaN when a backend omits them."""

    converted = variable.to(unit=target_unit) if target_unit is not None else variable
    if converted.variances is None:
        return np.full(converted.shape, np.nan, dtype=float).reshape(-1)
    return np.asarray(converted.variances, dtype=float).reshape(-1)


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
    collective_system_particles: int = 1,
    jump_diffusion: bool = False,
    random_seed: int = 0,
    n_samples: int = 1000,
    n_walkers: int = 32,
    n_burn: int = 500,
    n_thin: int = 10,
    parser_memory_limit_gib: float = 4.0,
) -> dict[str, Any]:
    """Estimate covariance-aware scalar tracer diffusion with kinisi."""

    if not dimensions or any(axis not in "xyz" for axis in dimensions):
        raise ValueError("dimensions must be a non-empty subset of xyz")
    if len(set(dimensions)) != len(dimensions):
        raise ValueError("dimensions contains duplicate axes")
    if fit_start_ps is None or fit_start_ps < 0:
        raise ValueError("fit_start_ps must be explicitly provided and >= 0")
    if n_samples < 1 or n_walkers < 2 or n_burn < 0 or n_thin < 1:
        raise ValueError("Invalid kinisi sampling parameters")
    if not isinstance(collective_system_particles, (int, np.integer)) or isinstance(
        collective_system_particles, bool
    ) or int(collective_system_particles) < 1:
        raise ValueError("collective_system_particles must be a positive integer")
    collective_system_particles = int(collective_system_particles)
    if not np.isfinite(parser_memory_limit_gib) or parser_memory_limit_gib <= 0:
        raise ValueError("parser_memory_limit_gib must be finite and positive")
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
    cell_matrix = np.asarray(view.cells[0], dtype=float)
    # Match kinisi 2.x's actual parser dispatch: it calls a cell
    # "orthorhombic" only when exactly the six off-diagonal entries are zero.
    triclinic = np.count_nonzero(np.isclose(cell_matrix.reshape(9), 0.0)) != 6
    parser_peak_bytes = _kinisi_parser_peak_bytes(
        nframes=view.nframes + 1,
        natoms=len(mobile),
        triclinic=triclinic,
    )
    parser_limit_bytes = int(parser_memory_limit_gib * 1024**3)
    if parser_peak_bytes > parser_limit_bytes:
        raise UnsupportedAnalysisError(
            "Estimated kinisi trajectory-parser peak memory is "
            f"{parser_peak_bytes / 1024**3:.2f} GiB for {view.nframes} frames "
            f"and {len(mobile)} mobile atoms, above the configured "
            f"{parser_memory_limit_gib:.2f} GiB limit. Increase "
            "--parser-memory-limit-gib only after confirming available RAM, "
            "or analyze a shorter production window."
        )
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
    position_semantics = _validate_kinisi_periodic_reconstruction(
        view, unwrap_diagnostics
    )
    temperature, temperature_source = _production_temperature(view, temperature_K)
    if ionic_charge_e is None:
        raise ValueError(
            "ionic_charge_e is required because transport reports sigma_NE_tracer"
        )
    kinisi_backend = _require_kinisi()
    # Keep a small compatibility shim for downstream tests/plugins that
    # monkeypatch the pre-JumpDiffusionAnalyzer four-item return value.
    if len(kinisi_backend) == 4:
        sc, DiffusionAnalyzer, ConductivityAnalyzer, kinisi_version = kinisi_backend
        JumpDiffusionAnalyzer = None
    else:
        (
            sc,
            DiffusionAnalyzer,
            ConductivityAnalyzer,
            JumpDiffusionAnalyzer,
            kinisi_version,
        ) = kinisi_backend
    if jump_diffusion and JumpDiffusionAnalyzer is None:
        raise OptionalDependencyError(
            "The installed kinisi adapter does not provide JumpDiffusionAnalyzer"
        )
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
    d_samples_m2_s = np.asarray(
        analyzer.D.to(unit="m^2/s").values, dtype=float
    ).reshape(-1)
    lag_time_ps = np.asarray(analyzer.dt.to(unit="ps").values, dtype=float)
    n_lag_points_total = int(len(lag_time_ps))
    fit_mask = lag_time_ps >= float(fit_start_ps)
    n_lag_points_in_fit = int(np.count_nonzero(fit_mask))
    if n_lag_points_in_fit == 0:
        raise ValueError("kinisi lag grid contains no points at or above fit_start_ps")
    actual_fit_stop_ps = float(lag_time_ps[-1])
    lag_grid_metadata = {
        "mode": lag_grid["mode"],
        "requested_step_ps": lag_grid["requested_step_ps"],
        "requested_stop_ps": lag_grid["requested_stop_ps"],
        "nominal_step_ps": lag_grid["nominal_step_ps"],
        "actual_step_ps": lag_grid["actual_step_ps"],
        "actual_min_ps": float(lag_time_ps[0]),
        "actual_max_ps": actual_fit_stop_ps,
        "n_lag_points_total": n_lag_points_total,
        "n_lag_points_in_fit": n_lag_points_in_fit,
        "fit_start_inserted": lag_grid["fit_start_inserted"],
        "is_uniform_grid": lag_grid["is_uniform_grid"],
        "frame_interval_ps": float(view.frame_interval_fs / 1000.0),
    }
    if lag_grid["mode"] == "kinisi_default":
        lag_grid_metadata["estimated_n_lag_points"] = lag_grid["estimated_n_lag_points"]
    number_density = particle_number_density_m3(view, mobile_species=mobile_species)
    nernst_einstein = nernst_einstein_tracer_conductivity(
        particle_density_m3=number_density,
        tracer_diffusion_m2_s=diffusion_m2_s["mean"],
        temperature_K=temperature,
        ionic_charge_e=ionic_charge_e,
    )
    charge_C = float(ionic_charge_e) * ELEMENTARY_CHARGE_C
    ne_factor = number_density * charge_C**2 / (BOLTZMANN_J_K * temperature)
    sigma_samples_S_m = ne_factor * d_samples_m2_s
    sigma_ne_posterior_S_m = _numeric_sample_summary(sigma_samples_S_m, unit="S/m")
    sigma_ne_posterior_S_cm = _numeric_sample_summary(
        sigma_samples_S_m * S_M_TO_S_CM, unit="S/cm"
    )
    sigma_ne_posterior_mS_cm = _numeric_sample_summary(
        sigma_samples_S_m * S_M_TO_MS_CM, unit="mS/cm"
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
            "sigma_NE_tracer_posterior_S_m": sigma_ne_posterior_S_m,
            "sigma_NE_tracer_posterior_S_cm": sigma_ne_posterior_S_cm,
            "sigma_NE_tracer_posterior_mS_cm": sigma_ne_posterior_mS_cm,
            "uncertainty_semantics": (
                "Linear propagation of the kinisi tracer-D posterior; n, z, V "
                "and T are treated as fixed. This is not the total physical "
                "uncertainty: model, finite-size, replica, temperature, volume, "
                "Nernst-Einstein approximation and ion-ion correlation "
                "uncertainties are not included."
            ),
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
            "backend_semantics": (
                "mlipx unweighted mean framework displacement (kinisi "
                "definition), pre-applied once; kinisi receives mobile atoms only"
            ),
        },
        "unwrap_diagnostics": unwrap_diagnostics,
        "kinisi_position_semantics": position_semantics,
        "kinisi_resource_diagnostics": {
            "n_lag_points_total": n_lag_points_total,
            "n_lag_points_in_fit": n_lag_points_in_fit,
            "single_float64_square_matrix_lower_bound_bytes": int(
                8 * n_lag_points_in_fit**2
            ),
            "parser_atom_count": int(len(mobile)),
            "source_atom_count": int(view.natoms),
            "triclinic_eight_image_path": bool(triclinic),
            "estimated_parser_peak_bytes": parser_peak_bytes,
            "parser_memory_limit_bytes": parser_limit_bytes,
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
        "D_tracer_samples_m2_s": d_samples_m2_s,
        "sigma_NE_samples_S_m": sigma_samples_S_m,
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
            system_particles=collective_system_particles,
        )
        conductivity.conductivity(
            start_dt,
            temperature=sc.scalar(temperature, unit="K"),
            **{**mcmc, "random_state": np.random.RandomState(random_seed + 1)},
        )
        sigma_collective_samples_S_m = _variable_values(
            conductivity.sigma, target_unit="S/m"
        )
        sigma_collective_samples_S_cm = sigma_collective_samples_S_m * S_M_TO_S_CM
        sigma_collective_samples_mS_cm = sigma_collective_samples_S_m * S_M_TO_MS_CM
        sigma_collective_S_m = _numeric_sample_summary(
            sigma_collective_samples_S_m, unit="S/m"
        )
        sigma_collective_S_cm = _numeric_sample_summary(
            sigma_collective_samples_S_cm, unit="S/cm"
        )
        sigma_collective_mS_cm = _numeric_sample_summary(
            sigma_collective_samples_mS_cm, unit="mS/cm"
        )
        d_sigma_samples_m2_s = sigma_collective_samples_S_m * (
            BOLTZMANN_J_K * temperature / (number_density * charge_C**2)
        )
        d_sigma_m2_s = _numeric_sample_summary(d_sigma_samples_m2_s, unit="m^2/s")
        d_sigma_cm2_s = _numeric_sample_summary(
            d_sigma_samples_m2_s * 1.0e4, unit="cm^2/s"
        )
        mscd = getattr(conductivity, "mscd", None)
        if mscd is None:
            # Compatibility for lightweight downstream fakes written before
            # collective MSCD artifacts were exposed. Real kinisi 2.x always
            # provides this variable.
            mscd = sc.array(
                dims=["time interval"],
                values=np.full(len(lag_time_ps), np.nan),
                variances=np.full(len(lag_time_ps), np.nan),
                unit="angstrom^2",
            )
        try:
            mscd_target_unit = "C^2*m^2"
            mscd_values = _variable_values(mscd, target_unit=mscd_target_unit)
            mscd_variance = _variable_variances(mscd, target_unit=mscd_target_unit)
        except Exception:
            # Lightweight fakes may expose a dimension-only placeholder. Keep
            # its backend unit rather than inventing a charge conversion.
            mscd_target_unit = str(mscd.unit)
            mscd_values = _variable_values(mscd)
            mscd_variance = _variable_variances(mscd)
        result["collective_conductivity"] = {
            "backend": "kinisi",
            "definition": (
                "collective Einstein ionic conductivity within the analyzed "
                "classical MD trajectory / selected charge model"
            ),
            "ionic_charge_e": float(ionic_charge_e),
            "system_particles": collective_system_particles,
            "system_particles_semantics": (
                "index-ordered statistical groups in kinisi, not independent MD replicas"
            ),
            "sigma_collective_posterior_S_m": sigma_collective_S_m,
            "sigma_collective_posterior_S_cm": sigma_collective_S_cm,
            # Preserve the original public key for consumers of Analysis v2.
            "sigma_collective_mS_cm_posterior": sigma_collective_mS_cm,
            "sigma_collective_MScm_posterior": sigma_collective_mS_cm,
            "D_sigma_posterior_m2_s": d_sigma_m2_s,
            "D_sigma_posterior_cm2_s": d_sigma_cm2_s,
            "mscd_unit": mscd_target_unit,
            "mscd_variance_unit": f"({mscd_target_unit})^2",
            "uncertainty_semantics": (
                "Kinisi Bayesian posterior conditional on this trajectory and charge "
                "model; it is not model, finite-size, replica, or total physical uncertainty."
            ),
        }
        result["kinisi_mscd"] = mscd_values
        result["kinisi_mscd_variance"] = mscd_variance
        result["sigma_collective_samples_S_m"] = sigma_collective_samples_S_m
        result["D_sigma_samples_m2_s"] = d_sigma_samples_m2_s
        result["sigma_collective"] = sigma_collective_S_m
        result["D_sigma"] = d_sigma_m2_s

        # Do not present a marginal-ratio interval as a joint physical error
        # bar.  The interval below is only an explicitly labelled independent
        # posterior approximation, and is omitted for nonphysical posteriors.
        positive_collective = np.all(
            np.isfinite(sigma_collective_samples_S_m)
            & (sigma_collective_samples_S_m > 0)
        )
        mean_abs = max(
            abs(float(np.mean(sigma_collective_samples_S_m))),
            np.finfo(float).tiny,
        )
        near_zero = np.any(
            np.abs(sigma_collective_samples_S_m) <= mean_abs * 1.0e-6
        )
        positive_tracer = np.all(np.isfinite(d_samples_m2_s) & (d_samples_m2_s > 0))
        if positive_collective and not near_zero and positive_tracer:
            rng = np.random.RandomState(random_seed + 2)
            n_ratio = max(len(d_samples_m2_s), len(sigma_collective_samples_S_m))
            d_marginal = rng.choice(d_samples_m2_s, size=n_ratio, replace=True)
            sigma_marginal = rng.choice(
                sigma_collective_samples_S_m, size=n_ratio, replace=True
            )
            haven_samples = d_marginal / (
                sigma_marginal
                * BOLTZMANN_J_K
                * temperature
                / (number_density * charge_C**2)
            )
            correlation_samples = 1.0 / haven_samples
            ratio_semantics = (
                "independent-marginal-posterior approximation; covariance between "
                "tracer and collective estimators is not modeled; not total physical uncertainty"
            )
            haven = _numeric_sample_summary(haven_samples, unit="1")
            correlation = _numeric_sample_summary(correlation_samples, unit="1")
            haven["uncertainty_semantics"] = ratio_semantics
            correlation["uncertainty_semantics"] = ratio_semantics
            result["haven_ratio_samples"] = haven_samples
            result["correlation_factor_samples"] = correlation_samples
            result["haven_ratio"] = {
                "definition": "H_R = D_tracer / D_sigma = sigma_NE_tracer / sigma_collective",
                "point_estimate": float(diffusion_m2_s["mean"] / d_sigma_m2_s["mean"]),
                "posterior": haven,
                "uncertainty_semantics": ratio_semantics,
            }
            result["correlation_factor"] = {
                "definition": "sigma_collective / sigma_NE_tracer",
                "point_estimate": float(
                    sigma_collective_S_m["mean"] / sigma_ne_posterior_S_m["mean"]
                ),
                "posterior": correlation,
                "uncertainty_semantics": ratio_semantics,
            }
        else:
            warning = (
                "Haven/correlation posterior intervals omitted because the collective "
                "conductivity posterior contains non-finite, non-positive, or near-zero samples."
            )
            result.setdefault("warnings", []).append(warning)
            haven_point = None
            if np.isfinite(d_sigma_m2_s["mean"]) and d_sigma_m2_s["mean"] != 0:
                haven_point = float(diffusion_m2_s["mean"] / d_sigma_m2_s["mean"])
            correlation_point = None
            if np.isfinite(sigma_ne_posterior_S_m["mean"]) and sigma_ne_posterior_S_m[
                "mean"
            ] != 0:
                correlation_point = float(
                    sigma_collective_S_m["mean"] / sigma_ne_posterior_S_m["mean"]
                )
            result["haven_ratio"] = {
                "definition": "H_R = D_tracer / D_sigma = sigma_NE_tracer / sigma_collective",
                "point_estimate": haven_point,
                "posterior": None,
                "uncertainty_semantics": "omitted: invalid collective posterior",
            }
            result["correlation_factor"] = {
                "definition": "sigma_collective / sigma_NE_tracer",
                "point_estimate": correlation_point,
                "posterior": None,
                "uncertainty_semantics": "omitted: invalid collective posterior",
            }
    if jump_diffusion:
        jump = JumpDiffusionAnalyzer.from_ase(
            **common,
            specie_indices=indices_variable,
            system_particles=collective_system_particles,
        )
        jump.jump_diffusion(
            start_dt,
            **{**mcmc, "random_state": np.random.RandomState(random_seed + 3)},
        )
        d_j_m2_s = _variable_values(jump.D_J, target_unit="m^2/s")
        result["jump_diffusion"] = {
            "backend": "kinisi",
            "definition": "total/jump-displacement transport diagnostic; not tracer diffusion",
            "system_particles": collective_system_particles,
            "system_particles_semantics": (
                "index-ordered statistical groups in kinisi, not independent MD replicas"
            ),
            "D_J_posterior_m2_s": _numeric_sample_summary(d_j_m2_s, unit="m^2/s"),
            "D_J_posterior_cm2_s": _numeric_sample_summary(
                d_j_m2_s * 1.0e4, unit="cm^2/s"
            ),
            "mstd_unit": str(jump.mstd.unit),
        }
        result["kinisi_mstd"] = _variable_values(jump.mstd)
        result["kinisi_mstd_variance"] = _variable_variances(jump.mstd)
        result["D_J_samples_m2_s"] = d_j_m2_s
    return result
