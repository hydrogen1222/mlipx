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


_TASK_OUTPUT_REVISIONS = {"msd": 5}


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
                    "fit_window_source": [
                        fit["fit_window_source"] for fit in fits
                    ],
                    "actual_fit_start_ps": [
                        fit["actual_fit_start_ps"] for fit in fits
                    ],
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
        from mlipx.analysis.transport import kinisi_transport

        result = kinisi_transport(dataset, **parameters)
        np.savez(
            output_dir / "kinisi_arrays.npz",
            lag_time_ps=result["lag_time_ps"],
            msd_A2=result["kinisi_msd_A2"],
            msd_variance_A4=result["kinisi_msd_variance_A4"],
        )
        artifacts.append("kinisi_arrays.npz")
        return result, artifacts
    if task == "electrolyte":
        from mlipx.analysis.electrolyte import gemdat_electrolyte

        result = gemdat_electrolyte(dataset, **parameters)
        np.savez(output_dir / "electrolyte_arrays.npz", **result.arrays)
        artifacts.append("electrolyte_arrays.npz")
        for name, table in result.tables.items():
            path = output_dir / f"{name}.csv"
            if hasattr(table, "to_csv"):
                table.to_csv(path, index=False)
            else:
                np.savetxt(path, np.asarray(table), delimiter=",")
            artifacts.append(path.name)
        for name, structure in result.structures.items():
            path = output_dir / f"{name}.cif"
            structure.to(filename=str(path))
            artifacts.append(path.name)
        for axis, values in result.paths.items():
            path = output_dir / f"percolation_{axis}.csv"
            _write_columns(path, values)
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
        "source_run": str(request.source_path),
        "source_trajectory": fingerprint,
    }
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
