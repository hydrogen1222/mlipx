"""Analysis task dispatch, result directories, status, and provenance."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import traceback
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from mlipx.analysis.dataset import TrajectoryDataset
from mlipx.analysis.schema import AnalysisRequest
from mlipx.analysis.validation import InvalidTrajectoryError, UnsupportedAnalysisError

if TYPE_CHECKING:
    from typing import Any


# Analysis output revisions are part of the request/cache identity. Bump the
# changed scientific tasks while deliberately leaving the native MSD revision
# untouched.
_TASK_OUTPUT_REVISIONS = {"msd": 5, "transport": 4, "electrolyte": 2}
_TASK_SCIENTIFIC_REVISIONS = {"transport": 4, "electrolyte": 2}


def _jsonable(value: Any, *, array_limit: int = 2000) -> Any:
    if isinstance(value, np.ndarray):
        if value.size <= array_limit:
            return _jsonable(value.tolist(), array_limit=array_limit)
        return {
            "stored_separately": True,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, np.generic):
        return _jsonable(value.item(), array_limit=array_limit)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _jsonable(item, array_limit=array_limit)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item, array_limit=array_limit) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def source_fingerprint(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if resolved.is_dir():
        for candidate in (
            resolved / "raw" / "trajectory.traj",
            resolved / "trajectory.traj",
            resolved / "vasp" / "XDATCAR",
            resolved / "XDATCAR",
        ):
            if candidate.is_file():
                resolved = candidate
                break
    if not resolved.exists():
        return {"path": str(resolved), "exists": False}
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "method": "path+size+mtime (no full-content scan)",
    }


def _versions() -> dict[str, str]:
    packages = {}
    for distribution in (
        "mlipx",
        "numpy",
        "scipy",
        "matplotlib",
        "ase",
        "kinisi",
        "gemdat",
    ):
        try:
            packages[distribution] = version(distribution)
        except PackageNotFoundError:
            continue
    return packages


def _analysis_id(request: dict[str, Any]) -> str:
    canonical = json.dumps(
        _jsonable(request, array_limit=100_000),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def _output_root(source: Path) -> Path:
    resolved = source.expanduser().resolve()
    if resolved.is_dir():
        return resolved / "analysis"
    if resolved.parent.name in {"raw", "vasp"}:
        return resolved.parent.parent / "analysis"
    return resolved.parent / "analysis"


def _update_index(root: Path, analysis_id: str, record: dict[str, Any]) -> None:
    index_path = root / "index.json"
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        index = {"schema": "mlipx.analysis-index/2", "jobs": {}}
    index.setdefault("jobs", {})[analysis_id] = _jsonable(record)
    _write_json(index_path, index)


def _write_columns(path: Path, columns: dict[str, Any]) -> None:
    usable = {
        key: np.asarray(value)
        for key, value in columns.items()
        if np.asarray(value).ndim == 1
    }
    if not usable:
        return
    lengths = {len(value) for value in usable.values()}
    if len(lengths) != 1:
        raise ValueError(f"CSV columns for {path.name} have unequal lengths")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(usable)
        writer.writerows(zip(*usable.values(), strict=True))


def _write_transport_summary(path: Path, result: dict[str, Any]) -> None:
    """Write a one-row human-readable transport summary CSV.

    Collective-conductivity columns are appended only when that analysis ran;
    absent fields are left blank rather than filled with fake zeros.
    """

    tracer = result["tracer_diffusion"]
    lag = tracer["lag_grid"]
    d_post = tracer["D_posterior_m2_s"]
    ne = result["nernst_einstein"]
    sigma = ne["sigma_NE_tracer_posterior_mS_cm"]
    semantics = result["kinisi_position_semantics"]
    fields: dict[str, Any] = {
        "mobile_species": result["mobile_species"],
        "dimensions": result["dimensions"],
        "temperature_K": result["temperature_mean_K"],
        "fit_start_ps": tracer["fit_start_ps"],
        "fit_stop_ps": tracer["fit_stop_ps"],
        "lag_grid_mode": lag["mode"],
        "lag_step_ps": lag["requested_step_ps"],
        "lag_stop_ps": lag["requested_stop_ps"],
        "n_lag_points_total": lag["n_lag_points_total"],
        "n_lag_points_in_fit": lag["n_lag_points_in_fit"],
        "D_mean_m2_s": d_post["mean"],
        "D_std_m2_s": d_post["std"],
        "D_median_m2_s": d_post["median"],
        "D_ci95_low_m2_s": d_post["credible_interval_95"][0],
        "D_ci95_high_m2_s": d_post["credible_interval_95"][1],
        "Dtr_mean_m2_s": d_post["mean"],
        "Dtr_ci95_low_m2_s": d_post["credible_interval_95"][0],
        "Dtr_ci95_high_m2_s": d_post["credible_interval_95"][1],
        "sigma_NE_mean_mS_cm": sigma["mean"],
        "sigma_NE_std_mS_cm": sigma["std"],
        "sigma_NE_median_mS_cm": sigma["median"],
        "sigma_NE_ci95_low_mS_cm": sigma["credible_interval_95"][0],
        "sigma_NE_ci95_high_mS_cm": sigma["credible_interval_95"][1],
        "sigma_NE_tracer_mean_S_m": ne["sigma_NE_tracer_posterior_S_m"]["mean"],
        "sigma_NE_tracer_ci95_low_S_m": ne["sigma_NE_tracer_posterior_S_m"]["credible_interval_95"][0],
        "sigma_NE_tracer_ci95_high_S_m": ne["sigma_NE_tracer_posterior_S_m"]["credible_interval_95"][1],
        "kinisi_version": tracer["kinisi_version"],
        "random_seed": tracer["random_seed"],
        "positions_convention": semantics["source_positions_convention"],
        "kinisi_backend_reconstruction": semantics["backend_reconstruction"],
    }
    if "collective_conductivity" in result:
        coll = result["collective_conductivity"].get(
            "sigma_collective_mS_cm_posterior",
            result["collective_conductivity"].get("sigma_collective_posterior_mS_cm", {}),
        )
        coll = coll or {}
        fields.update(
            {
                "sigma_collective_mean_mS_cm": coll["mean"],
                "sigma_collective_std_mS_cm": coll["std"],
                "sigma_collective_median_mS_cm": coll["median"],
                "sigma_collective_ci95_low_mS_cm": coll["credible_interval_95"][0],
                "sigma_collective_ci95_high_mS_cm": coll["credible_interval_95"][1],
            }
        )
        coll_s = result["collective_conductivity"].get(
            "sigma_collective_posterior_S_m", {}
        )
        dsigma = result["collective_conductivity"].get(
            "D_sigma_posterior_m2_s", {}
        )
        dsigma_cm = result["collective_conductivity"].get(
            "D_sigma_posterior_cm2_s", {}
        )
        haven = result.get("haven_ratio") or {}
        correlation = result.get("correlation_factor") or {}
        fields.update(
            {
                "sigma_collective_mean_S_m": coll_s.get("mean"),
                "sigma_collective_ci95_low_S_m": (coll_s.get("credible_interval_95") or [None, None])[0],
                "sigma_collective_ci95_high_S_m": (coll_s.get("credible_interval_95") or [None, None])[1],
                "D_sigma_mean_m2_s": dsigma.get("mean"),
                "Dsigma_mean_m2_s": dsigma.get("mean"),
                "D_sigma_ci95_low_m2_s": (dsigma.get("credible_interval_95") or [None, None])[0],
                "D_sigma_ci95_high_m2_s": (dsigma.get("credible_interval_95") or [None, None])[1],
                "Dsigma_mean_cm2_s": dsigma_cm.get("mean"),
                "Haven_point_estimate": haven.get("point_estimate"),
                "Haven": haven.get("point_estimate"),
                "Haven_ci95_low": (haven.get("posterior") or {}).get("credible_interval_95", [None, None])[0],
                "Haven_ci95_high": (haven.get("posterior") or {}).get("credible_interval_95", [None, None])[1],
                "correlation_factor_point_estimate": correlation.get("point_estimate"),
                "correlation_factor": correlation.get("point_estimate"),
                "correlation_factor_ci95_low": (correlation.get("posterior") or {}).get("credible_interval_95", [None, None])[0],
                "correlation_factor_ci95_high": (correlation.get("posterior") or {}).get("credible_interval_95", [None, None])[1],
                "ratio_uncertainty_semantics": (haven.get("uncertainty_semantics")),
            }
        )
    else:
        fields.update(
            {
                "sigma_collective_mean_mS_cm": None,
                "sigma_collective_mean_S_m": None,
                "D_sigma_mean_m2_s": None,
                "Dsigma_mean_m2_s": None,
                "Haven_point_estimate": None,
                "Haven": None,
                "correlation_factor_point_estimate": None,
                "correlation_factor": None,
            }
        )
    if "jump_diffusion" in result:
        jump = result["jump_diffusion"].get("D_J_posterior_m2_s", {})
        fields.update(
            {
                "D_J_mean_m2_s": jump.get("mean"),
                "D_J_ci95_low_m2_s": (jump.get("credible_interval_95") or [None, None])[0],
                "D_J_ci95_high_m2_s": (jump.get("credible_interval_95") or [None, None])[1],
            }
        )
    fields.update(
        {
            "temperature_source": result.get("temperature_source"),
            "drift_reference": (result.get("drift_correction") or {}).get("mode"),
            "dimensions": result.get("dimensions"),
            "fit_start_ps": tracer.get("fit_start_ps"),
        }
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields.keys())
        writer.writerow(["" if value is None else value for value in fields.values()])


def _load_dataset(request: AnalysisRequest) -> TrajectoryDataset:
    parameters = request.parameters
    return TrajectoryDataset.load(
        request.source,
        positions_convention=parameters.pop("positions_convention", None),
        frame_interval_fs=parameters.pop("frame_interval_fs", None),
    )


def _dispatch(request: AnalysisRequest, output_dir: Path) -> tuple[Any, list[str]]:
    task = request.task
    parameters = dict(request.parameters)
    request_copy = AnalysisRequest(task, request.source, parameters, request.force)
    artifacts: list[str] = []
    if task == "arrhenius":
        from mlipx.analysis.arrhenius import fit_arrhenius
        from mlipx.analysis.plots import plot_arrhenius

        result = fit_arrhenius(**parameters)
        _write_columns(
            output_dir / "arrhenius.csv",
            {
                "temperature_K": result["temperatures_K"],
                "inverse_temperature_K^-1": result["inverse_temperature_K^-1"],
                "diffusivity_m2_s": result["diffusivities_m2_s"],
                "ln_diffusivity": result["ln_diffusivity"],
                "ln_diffusivity_fit": result["ln_diffusivity_fit"],
            },
        )
        artifacts.append("arrhenius.csv")
        artifacts.extend(
            path.name for path in plot_arrhenius(result, output_dir / "arrhenius")
        )
        return result, artifacts

    dataset = _load_dataset(request_copy)
    if task == "validate":
        from mlipx.analysis.validation import validate_trajectory

        return validate_trajectory(dataset).to_dict(), artifacts
    if task == "thermo":
        from mlipx.analysis.plots import plot_thermo
        from mlipx.analysis.thermo import thermodynamic_diagnostics

        result = thermodynamic_diagnostics(dataset, **parameters)
        _write_columns(output_dir / "thermo.csv", result["columns"])
        artifacts.append("thermo.csv")
        artifacts.extend(
            path.name for path in plot_thermo(result, output_dir / "thermo")
        )
        return result, artifacts
    if task == "rdf":
        from mlipx.analysis.plots import plot_rdf
        from mlipx.analysis.structure import radial_distribution

        result = radial_distribution(dataset, **parameters)
        _write_columns(
            output_dir / "rdf.csv",
            {
                "r_A": result["r_A"],
                "g_center_neighbor": result["g_center_neighbor"],
                "coordination_number_center_neighbor": result[
                    "coordination_number_center_neighbor"
                ],
                "ordered_neighbor_counts": result["ordered_neighbor_counts"],
            },
        )
        artifacts.append("rdf.csv")
        artifacts.extend(path.name for path in plot_rdf(result, output_dir / "rdf"))
        return result, artifacts
    if task == "rmsd":
        from mlipx.analysis.structure import periodic_rmsd_rmsf

        result = periodic_rmsd_rmsf(dataset, **parameters)
        _write_columns(
            output_dir / "rmsd.csv",
            {
                "time_ps": result["time_ps"],
                "periodic_displacement_rmsd_A": result["periodic_displacement_rmsd_A"],
            },
        )
        _write_columns(
            output_dir / "rmsf.csv",
            {
                "atom_index": result["atom_indices"],
                "periodic_displacement_rmsf_A": result["periodic_displacement_rmsf_A"],
            },
        )
        artifacts.extend(("rmsd.csv", "rmsf.csv"))
        return result, artifacts
    if task == "msd":
        from mlipx.analysis.msd import calculate_msd
        from mlipx.analysis.plots import plot_msd, plot_msd_alpha

        result = calculate_msd(dataset, **parameters)
        columns = {
            "lag_time_ps": result["lag_time_ps"],
            "time_origin_counts": result["time_origin_counts"],
            "msd_x_A2": result["msd_x_A2"],
            "msd_y_A2": result["msd_y_A2"],
            "msd_z_A2": result["msd_z_A2"],
        }
        for axes, values in result["msd_by_axes_A2"].items():
            columns[f"msd_{axes}_A2"] = values
            columns[f"alpha_{axes}"] = result["log_log_alpha_by_axes"][axes]
        _write_columns(output_dir / "msd.csv", columns)
        fits = list(result["diagnostic_linear_diffusion_fits"].values())
        if fits:
            _write_columns(
                output_dir / "diffusion_fits.csv",
                {
                    "axes": [fit["axes"] for fit in fits],
                    "dimensions": [fit["dimensions"] for fit in fits],
                    "fit_start_ps": [fit["fit_start_ps"] for fit in fits],
                    "fit_stop_ps": [fit["fit_stop_ps"] for fit in fits],
                    "actual_fit_start_ps": [fit["actual_fit_start_ps"] for fit in fits],
                    "actual_fit_stop_ps": [fit["actual_fit_stop_ps"] for fit in fits],
                    "fit_points": [fit["fit_points"] for fit in fits],
                    "slope_A2_ps": [fit["slope_A2_ps"] for fit in fits],
                    "intercept_A2": [fit["intercept_A2"] for fit in fits],
                    "r_squared": [fit["r_squared"] for fit in fits],
                    "self_diffusion_coefficient_m2_s": [
                        fit["self_diffusion_coefficient_m2_s"] for fit in fits
                    ],
                    "self_diffusion_coefficient_cm2_s": [
                        fit["self_diffusion_coefficient_cm2_s"] for fit in fits
                    ],
                    "mean_log_log_alpha_in_fit": [
                        fit["mean_log_log_alpha_in_fit"] for fit in fits
                    ],
                    "diffusive_regime_warning": [
                        fit["diffusive_regime_warning"] for fit in fits
                    ],
                    "estimator": [fit["estimator"] for fit in fits],
                    "publication_grade": [fit["publication_grade"] for fit in fits],
                },
            )
        _write_json(
            output_dir / "diagnostics.json",
            {
                "unwrap": result["unwrap_diagnostics"],
                "fits": result["diagnostic_linear_diffusion_fits"],
            },
        )
        artifacts.extend(("msd.csv", "diagnostics.json"))
        if fits:
            artifacts.append("diffusion_fits.csv")
        artifacts.extend(path.name for path in plot_msd(result, output_dir / "msd"))
        artifacts.extend(
            path.name for path in plot_msd_alpha(result, output_dir / "alpha")
        )
        return result, artifacts
    if task == "density":
        from mlipx.analysis.structure import density_map

        result = density_map(dataset, **parameters)
        np.savez(
            output_dir / "density.npz",
            occupancy_probability=result["occupancy_probability"],
            number_density_A3=result["number_density_A^-3"],
            counts_per_voxel=result["counts_per_voxel"],
            cell_A=result["cell_A"],
        )
        artifacts.append("density.npz")
        return result, artifacts
    if task == "transport":
        from mlipx.analysis.plots import (
            plot_transport,
            plot_transport_mscd,
            plot_transport_mstd,
        )
        from mlipx.analysis.transport import kinisi_transport

        result = kinisi_transport(dataset, **parameters)
        arrays = {
            "lag_time_ps": result["lag_time_ps"],
            "msd_A2": result["kinisi_msd_A2"],
            "msd_variance_A4": result["kinisi_msd_variance_A4"],
            "D_tracer_samples_m2_s": result["D_tracer_samples_m2_s"],
            "sigma_NE_samples_S_m": result["sigma_NE_samples_S_m"],
        }
        if "kinisi_mscd" in result:
            arrays.update(
                {
                    "mscd": result["kinisi_mscd"],
                    "mscd_variance": result["kinisi_mscd_variance"],
                    "sigma_collective_samples_S_m": result[
                        "sigma_collective_samples_S_m"
                    ],
                    "D_sigma_samples_m2_s": result["D_sigma_samples_m2_s"],
                }
            )
            if "haven_ratio_samples" in result:
                arrays["haven_ratio_samples"] = result["haven_ratio_samples"]
        if "kinisi_mstd" in result:
            arrays.update(
                {
                    "mstd": result["kinisi_mstd"],
                    "mstd_variance": result["kinisi_mstd_variance"],
                    "D_J_samples_m2_s": result["D_J_samples_m2_s"],
                }
            )
        np.savez_compressed(
            output_dir / "kinisi_arrays.npz",
            **arrays,
        )
        artifacts.append("kinisi_arrays.npz")
        _write_transport_summary(output_dir / "transport_summary.csv", result)
        artifacts.append("transport_summary.csv")
        _write_json(
            output_dir / "diagnostics.json",
            {
                "quality": result.get("quality"),
                "warnings": result.get("warnings", []),
                "kinisi_position_semantics": result.get("kinisi_position_semantics"),
                "kinisi_time_mapping": result.get("kinisi_time_mapping"),
                "kinisi_resource_diagnostics": result.get(
                    "kinisi_resource_diagnostics"
                ),
                "drift_correction": result.get("drift_correction"),
                "uncertainty_semantics": {
                    "tracer": result["nernst_einstein"].get("uncertainty_semantics"),
                    "collective": (result.get("collective_conductivity") or {}).get(
                        "uncertainty_semantics"
                    ),
                    "ratios": (result.get("haven_ratio") or {}).get(
                        "uncertainty_semantics"
                    ),
                },
            },
        )
        artifacts.append("diagnostics.json")
        artifacts.extend(
            path.name for path in plot_transport(result, output_dir / "transport_msd")
        )
        if "kinisi_mscd" in result:
            artifacts.extend(
                path.name
                for path in plot_transport_mscd(result, output_dir / "transport_mscd")
            )
        if "kinisi_mstd" in result:
            artifacts.extend(
                path.name
                for path in plot_transport_mstd(result, output_dir / "transport_mstd")
            )
        return result, artifacts
    if task == "electrolyte":
        from mlipx.analysis.electrolyte import gemdat_electrolyte
        from mlipx.analysis.plots import (
            plot_electrolyte_density,
            plot_electrolyte_distribution,
            plot_electrolyte_paths,
        )

        result = gemdat_electrolyte(dataset, **parameters)
        np.savez_compressed(output_dir / "electrolyte_arrays.npz", **result.arrays)
        artifacts.append("electrolyte_arrays.npz")
        summary_fields = {
            key: value
            for key, value in result.summary.items()
            if np.isscalar(value) and not isinstance(value, (dict, list, tuple))
        }
        _write_columns(
            output_dir / "electrolyte_summary.csv",
            {key: [value] for key, value in summary_fields.items()},
        )
        artifacts.append("electrolyte_summary.csv")
        _write_json(
            output_dir / "diagnostics.json",
            {"warnings": result.warnings, "summary": result.summary},
        )
        artifacts.append("diagnostics.json")
        for matrix_name in ("transition_matrix", "jump_matrix"):
            if matrix_name in result.arrays:
                matrix_path = output_dir / f"{matrix_name}.csv"
                np.savetxt(matrix_path, np.asarray(result.arrays[matrix_name]), delimiter=",")
                artifacts.append(matrix_path.name)
        artifacts.extend(
            path.name
            for path in plot_electrolyte_density(
                result.arrays, output_dir / "density_projection"
            )
        )
        artifacts.extend(
            path.name
            for path in plot_electrolyte_paths(
                {"paths": result.paths}, output_dir / "free_energy_paths"
            )
        )
        artifacts.extend(
            path.name
            for path in plot_electrolyte_distribution(
                {"table": result.tables.get("residence_times")},
                output_dir / "residence_time_distribution",
                title="Residence-time distribution",
                xlabel="Residence time",
            )
        )
        artifacts.extend(
            path.name
            for path in plot_electrolyte_distribution(
                {"table": result.tables.get("jumps")},
                output_dir / "jump_distance_distribution",
                title="Jump-distance distribution",
                xlabel="Jump distance",
            )
        )
        artifacts.extend(
            path.name
            for path in plot_electrolyte_distribution(
                {"table": result.tables.get("jump_rates")},
                output_dir / "jump_rate_segments",
                title="Jump rate by trajectory segment",
                xlabel="Jump rate",
            )
        )
        for name, table in result.tables.items():
            path = output_dir / f"{name}.csv"
            if hasattr(table, "to_csv"):
                table.to_csv(path, index=False)
            else:
                values = np.asarray(table)
                if values.size == 0:
                    path.write_text("empty\n", encoding="utf-8")
                else:
                    np.savetxt(path, values, delimiter=",")
            artifacts.append(path.name)
        for name, structure in result.structures.items():
            path = output_dir / f"{name}.cif"
            structure.to(filename=str(path))
            artifacts.append(path.name)
        for axis, values in result.paths.items():
            path = output_dir / f"percolation_{axis}.csv"
            _write_columns(path, values)
            artifacts.append(path.name)
        # Explicit discovery artifacts have stable names in addition to the
        # generic structures, making exploratory site generation auditable.
        if "detected_sites" in result.structures:
            path = output_dir / "detected_sites.cif"
            result.structures["detected_sites"].to(filename=str(path))
            artifacts.append(path.name)
        if "occupancy_sites" in result.structures:
            path = output_dir / "occupancy_sites.cif"
            result.structures["occupancy_sites"].to(filename=str(path))
            artifacts.append(path.name)
        return {"summary": result.summary, "warnings": result.warnings}, artifacts
    if task in {"vacf", "spectrum"}:
        from mlipx.analysis.plots import plot_spectrum, plot_vacf
        from mlipx.analysis.spectral import calculate_vacf, velocity_spectrum

        spectrum_parameters = {}
        if task == "spectrum":
            spectrum_parameters = {
                key: parameters.pop(key)
                for key in ("taper", "normalization")
                if key in parameters
            }
        vacf = calculate_vacf(dataset, **parameters)
        _write_columns(
            output_dir / "vacf.csv",
            {
                "lag_time_fs": vacf["lag_time_fs"],
                "vacf_raw_A2_fs2": vacf["vacf_raw_A2_fs2"],
                "vacf_normalized": vacf["vacf_normalized"],
            },
        )
        artifacts.append("vacf.csv")
        artifacts.extend(path.name for path in plot_vacf(vacf, output_dir / "vacf"))
        if task == "vacf":
            return vacf, artifacts
        spectrum = velocity_spectrum(vacf, **spectrum_parameters)
        _write_columns(
            output_dir / "spectrum.csv",
            {
                "frequency_THz": spectrum["frequency_THz"],
                "frequency_cm^-1": spectrum["frequency_cm^-1"],
                "raw_spectrum": spectrum["raw_spectrum"],
                "spectrum": spectrum["spectrum"],
            },
        )
        artifacts.append("spectrum.csv")
        artifacts.extend(
            path.name for path in plot_spectrum(spectrum, output_dir / "spectrum")
        )
        return {"vacf": vacf, "spectrum": spectrum}, artifacts
    raise AssertionError(f"Unhandled analysis task: {task}")


def run_analysis(request: AnalysisRequest) -> dict[str, Any]:
    """Run one analysis job and persist its complete reproducibility record."""

    fingerprint = source_fingerprint(request.source_path)
    canonical_request = {
        "schema": "mlipx.analysis-request/2",
        "task": request.task,
        "source": str(request.source_path),
        "source_fingerprint": fingerprint,
        "parameters": request.parameters,
        "backend_versions": _versions(),
    }
    if request.task in _TASK_OUTPUT_REVISIONS:
        canonical_request["task_output_revision"] = _TASK_OUTPUT_REVISIONS[request.task]
    if request.task in _TASK_SCIENTIFIC_REVISIONS:
        canonical_request["task_scientific_revision"] = _TASK_SCIENTIFIC_REVISIONS[
            request.task
        ]
    analysis_id = _analysis_id(canonical_request)
    root = _output_root(request.source_path)
    output_dir = root / request.task / analysis_id
    root.mkdir(parents=True, exist_ok=True)
    existing = output_dir / "results.json"
    if existing.is_file() and not request.force:
        value = json.loads(existing.read_text(encoding="utf-8"))
        if value.get("status") == "success":
            return {
                "analysis_id": analysis_id,
                "output_dir": str(output_dir),
                "status": "success",
                "reused": True,
                "results": value.get("results"),
            }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "request.json", canonical_request)
    provenance = {
        "schema": "mlipx.analysis-provenance/2",
        "analysis_id": analysis_id,
        "request_hash": analysis_id,
        "analysis_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": _versions(),
        "parameters": request.parameters,
        "source_run": str(request.source_path),
        "source_trajectory": fingerprint,
    }
    if request.task in _TASK_SCIENTIFIC_REVISIONS:
        provenance["task_scientific_revision"] = _TASK_SCIENTIFIC_REVISIONS[request.task]
    _write_json(output_dir / "provenance.json", provenance)
    record = {
        "analysis_id": analysis_id,
        "task": request.task,
        "status": "running",
        "path": str(output_dir.relative_to(root)),
    }
    _update_index(root, analysis_id, record)
    try:
        result, artifacts = _dispatch(request, output_dir)
        if request.task == "transport":
            provenance["transport"] = {
                "fit_start_ps": result["tracer_diffusion"]["fit_start_ps"],
                "fit_stop_ps": result["tracer_diffusion"]["fit_stop_ps"],
                "lag_grid": result["tracer_diffusion"]["lag_grid"],
                "kinisi_position_semantics": result["kinisi_position_semantics"],
                "temperature_source": result.get("temperature_source"),
                "drift_correction": result.get("drift_correction"),
                "dimensions": result.get("dimensions"),
                "random_seed": result["tracer_diffusion"].get("random_seed"),
                "collective_system_particles": (
                    result.get("collective_conductivity") or {}
                ).get("system_particles"),
                "jump_diffusion": "jump_diffusion" in result,
            }
            _write_json(output_dir / "provenance.json", provenance)
        elif request.task == "electrolyte":
            summary = result.get("summary", {})
            provenance["electrolyte"] = {
                "analysis_phase": summary.get("analysis_phase", "production"),
                "time_source": summary.get("time_source"),
                "temperature_source": summary.get("temperature_source"),
                "position_convention": summary.get("position_convention"),
                "drift_correction": summary.get("drift_correction"),
                "site_source": summary.get("site_source"),
                "jump_dimensions": summary.get("jump_dimensions"),
                "percolation_axes": summary.get("percolation_axes"),
            }
            _write_json(output_dir / "provenance.json", provenance)
        payload = {
            "schema": "mlipx.analysis-results/2",
            "analysis_id": analysis_id,
            "status": "success",
            "task": request.task,
            "results": result,
            "artifacts": sorted(set(artifacts)),
        }
        _write_json(output_dir / "results.json", payload)
        record.update(status="success", artifacts=payload["artifacts"])
        _update_index(root, analysis_id, record)
        return {
            "analysis_id": analysis_id,
            "output_dir": str(output_dir),
            "status": "success",
            "reused": False,
            "results": _jsonable(result),
        }
    except Exception as exc:
        status = (
            "unsupported"
            if isinstance(exc, (InvalidTrajectoryError, UnsupportedAnalysisError))
            else "failed"
        )
        error = {
            "schema": "mlipx.analysis-error/2",
            "analysis_id": analysis_id,
            "status": status,
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_json(output_dir / "error.json", error)
        record.update(status=status, error=error["message"])
        _update_index(root, analysis_id, record)
        raise
