"""High-level analysis orchestration and public API."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np

from mlipx.analysis.core import (
    density_grid,
    fit_diffusion,
    mean_squared_displacement,
    radial_distribution,
    rmsd_rmsf,
    thermodynamics_summary,
    vacf_vdos,
)
from mlipx.analysis.dataset import TrajectoryDataset
from mlipx.analysis.store import AnalysisStore, sha256_file


DEFAULT_TASKS = ("validate", "thermo", "rmsd", "rdf", "msd", "density")
ALL_TASKS = (*DEFAULT_TASKS, "vacf", "transport", "electrolyte")


class AnalysisRunner:
    """Run cached analyses against one canonical trajectory dataset."""

    def __init__(
        self,
        source: str | Path,
        *,
        frame_interval_fs: float | None = None,
        assume_wrapped: bool | None = None,
        plots: bool = True,
        force: bool = False,
    ) -> None:
        self.dataset = TrajectoryDataset.load(
            source,
            frame_interval_fs=frame_interval_fs,
            assume_wrapped=assume_wrapped,
        )
        self.source_sha256 = sha256_file(self.dataset.source_path)
        self.related_input_sha256 = {
            path.relative_to(self.dataset.run_dir).as_posix(): sha256_file(path)
            for path in (
                self.dataset.run_dir / "raw" / "md.csv",
                self.dataset.run_dir / "md.csv",
                self.dataset.run_dir / "artifacts.json",
                self.dataset.run_dir / "resolved_config.json",
                self.dataset.run_dir / "raw" / "mlipx_results.json",
                self.dataset.run_dir / "mlipx_results.json",
                self.dataset.run_dir / "run.log",
            )
            if path.exists()
        }
        self.plots = plots
        self.force = force

    def _store(self, task: str, parameters: dict[str, Any]) -> AnalysisStore:
        parameters = {
            **parameters,
            "requested_plots": self.plots,
            "related_input_sha256": self.related_input_sha256,
            "trajectory_interpretation": {
                "frame_interval_fs": self.dataset.frame_interval_fs,
                "minimum_image_unwrapped": self.dataset.metadata[
                    "minimum_image_unwrapped"
                ],
            },
        }
        return AnalysisStore(
            run_dir=self.dataset.run_dir,
            task=task,
            source_path=self.dataset.source_path,
            source_sha256=self.source_sha256,
            parameters=parameters,
        )

    def _plot(self, callback, warnings: list[str]) -> Path | None:
        if not self.plots:
            return None
        try:
            return callback()
        except ImportError as exc:
            warnings.append(str(exc))
            return None

    def run(
        self,
        tasks: Iterable[str] | None = None,
        **options: Any,
    ) -> dict[str, dict[str, Any]]:
        requested = tuple(tasks or DEFAULT_TASKS)
        unknown = sorted(set(requested) - set(ALL_TASKS))
        if unknown:
            raise ValueError(f"Unknown analysis tasks: {', '.join(unknown)}")
        results: dict[str, dict[str, Any]] = {}
        for task in requested:
            results[task] = getattr(self, f"run_{task}")(**options)
        return results

    def run_validate(self, **_: Any) -> dict[str, Any]:
        store = self._store("validate", {})
        if store.cached() and not self.force:
            return {"cached": True, "path": str(store.path)}
        report = self.dataset.validation_report()
        output = store.write_json("report.json", report)
        metadata = store.complete(
            outputs=[output], summary=report, warnings=report["warnings"]
        )
        return {"cached": False, "path": str(store.path), "metadata": metadata}

    def run_thermo(
        self,
        *,
        start: int = 0,
        stop: int | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        parameters = {"start": start, "stop": stop}
        store = self._store("thermo", parameters)
        if store.cached() and not self.force:
            return {"cached": True, "path": str(store.path)}
        columns, summary = thermodynamics_summary(self.dataset, start=start, stop=stop)
        if not columns:
            summary.update(
                {
                    "available": False,
                    "reason": "No raw/md.csv or legacy md.csv was found",
                }
            )
            output = store.write_json("summary.json", summary)
            metadata = store.complete(
                outputs=[output],
                summary=summary,
                warnings=[summary["reason"]],
            )
            return {
                "cached": False,
                "path": str(store.path),
                "metadata": metadata,
            }
        summary["available"] = True
        outputs = [store.write_csv("thermodynamics.csv", columns)]
        warnings: list[str] = []
        time = columns.get("time_fs")
        if time is not None:
            plot_columns = [
                (key, values)
                for key, values in columns.items()
                if key not in {"step", "time_fs"}
            ]
            plot = self._plot(
                lambda: __import__(
                    "mlipx.analysis.plots", fromlist=["line_plot"]
                ).line_plot(
                    store.prepare() / "thermodynamics.png",
                    time / 1000,
                    plot_columns,
                    xlabel="Time (ps)",
                    ylabel="Value (native CSV units)",
                    title="MD thermodynamics",
                ),
                warnings,
            )
            if plot:
                outputs.append(plot)
        outputs.append(store.write_json("summary.json", summary))
        metadata = store.complete(
            outputs=outputs, summary=summary, warnings=warnings, packages=["matplotlib"]
        )
        return {"cached": False, "path": str(store.path), "metadata": metadata}

    def run_rmsd(
        self,
        *,
        mobile: str | None = None,
        framework: str | None = None,
        start: int = 0,
        stop: int | None = None,
        align: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        parameters = {
            "species": mobile,
            "framework_drift": framework,
            "start": start,
            "stop": stop,
            "align": align,
        }
        store = self._store("rmsd", parameters)
        if store.cached() and not self.force:
            return {"cached": True, "path": str(store.path)}
        data = rmsd_rmsf(
            self.dataset,
            species=mobile,
            framework=framework,
            start=start,
            stop=stop,
            align=align,
        )
        outputs = [
            store.write_csv(
                "rmsd.csv",
                {"time_ps": data["time_fs"] / 1000, "rmsd_A": data["rmsd_A"]},
            ),
            store.write_csv(
                "rmsf.csv",
                {"atom_index": data["atom_index"], "rmsf_A": data["rmsf_A"]},
            ),
        ]
        warnings: list[str] = []
        plot = self._plot(
            lambda: __import__(
                "mlipx.analysis.plots", fromlist=["line_plot"]
            ).line_plot(
                store.prepare() / "rmsd.png",
                data["time_fs"] / 1000,
                [("RMSD", data["rmsd_A"])],
                xlabel="Time (ps)",
                ylabel="RMSD (Å)",
                title="Root-mean-square displacement",
            ),
            warnings,
        )
        if plot:
            outputs.append(plot)
        summary = {
            "mean_rmsd_A": float(np.mean(data["rmsd_A"])),
            "final_rmsd_A": float(data["rmsd_A"][-1]),
            "mean_rmsf_A": float(np.mean(data["rmsf_A"])),
        }
        outputs.append(store.write_json("summary.json", summary))
        metadata = store.complete(
            outputs=outputs, summary=summary, warnings=warnings, packages=["matplotlib"]
        )
        return {"cached": False, "path": str(store.path), "metadata": metadata}

    def _default_rdf_pairs(self, mobile: str) -> list[tuple[str, str]]:
        framework_species = sorted(set(self.dataset.symbols) - {mobile})
        return [(mobile, mobile), *((mobile, item) for item in framework_species)]

    def run_rdf(
        self,
        *,
        mobile: str | None = None,
        rdf_pairs: Iterable[tuple[str, str]] | None = None,
        rdf_rmax: float = 8.0,
        rdf_bins: int = 200,
        start: int = 0,
        stop: int | None = None,
        stride: int = 1,
        **_: Any,
    ) -> dict[str, Any]:
        mobile = mobile or self.dataset.symbols[0]
        pairs = list(rdf_pairs or self._default_rdf_pairs(mobile))
        parameters = {
            "pairs": pairs,
            "r_max_A": rdf_rmax,
            "bins": rdf_bins,
            "start": start,
            "stop": stop,
            "stride": stride,
        }
        store = self._store("rdf", parameters)
        if store.cached() and not self.force:
            return {"cached": True, "path": str(store.path)}
        outputs: list[Path] = []
        warnings: list[str] = []
        summary: dict[str, Any] = {"pairs": {}}
        for first, second in pairs:
            data = radial_distribution(
                self.dataset,
                species_a=first,
                species_b=second,
                r_max=rdf_rmax,
                bins=rdf_bins,
                start=start,
                stop=stop,
                stride=stride,
            )
            label = f"{first}-{second}"
            outputs.append(
                store.write_csv(
                    f"rdf_{label}.csv",
                    {
                        "r_A": data["r_A"],
                        "g_r": data["g_r"],
                        "coordination_number": data["coordination_number"],
                        "raw_pair_counts": data["raw_pair_counts"],
                    },
                )
            )
            summary["pairs"][label] = {
                "peak_r_A": float(data["r_A"][np.argmax(data["g_r"])]),
                "peak_g_r": float(np.max(data["g_r"])),
            }
            plot = self._plot(
                lambda data=data, label=label: __import__(
                    "mlipx.analysis.plots", fromlist=["line_plot"]
                ).line_plot(
                    store.prepare() / f"rdf_{label}.png",
                    data["r_A"],
                    [(label, data["g_r"])],
                    xlabel="r (Å)",
                    ylabel="g(r)",
                    title=f"Partial RDF: {label}",
                ),
                warnings,
            )
            if plot:
                outputs.append(plot)
        outputs.append(store.write_json("summary.json", summary))
        metadata = store.complete(
            outputs=outputs, summary=summary, warnings=warnings, packages=["matplotlib"]
        )
        return {"cached": False, "path": str(store.path), "metadata": metadata}

    def run_msd(
        self,
        *,
        mobile: str | None = None,
        framework: str | None = None,
        dimensions: str = "xyz",
        start: int = 0,
        stop: int | None = None,
        fit_start_ps: float | None = None,
        fit_stop_ps: float | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        mobile = mobile or self.dataset.symbols[0]
        parameters = {
            "species": mobile,
            "framework_drift": framework,
            "dimensions": dimensions,
            "start": start,
            "stop": stop,
            "fit_start_ps": fit_start_ps,
            "fit_stop_ps": fit_stop_ps,
        }
        store = self._store("msd", parameters)
        if store.cached() and not self.force:
            return {"cached": True, "path": str(store.path)}
        data = mean_squared_displacement(
            self.dataset,
            species=mobile,
            framework=framework,
            dimensions=dimensions,
            start=start,
            stop=stop,
        )
        outputs = [
            store.write_csv(
                "msd.csv",
                {
                    "lag_time_ps": data["lag_time_fs"] / 1000,
                    "msd_A2": data["msd_A2"],
                    "msd_x_A2": data["msd_x_A2"],
                    "msd_y_A2": data["msd_y_A2"],
                    "msd_z_A2": data["msd_z_A2"],
                },
            ),
            store.write_npz(
                "per_particle_msd.npz",
                atom_index=data["atom_index"],
                msd_A2=data["per_particle_msd_A2"],
            ),
        ]
        summary = fit_diffusion(
            data["lag_time_fs"],
            data["msd_A2"],
            dimensions=len(dimensions),
            fit_start_fs=None if fit_start_ps is None else fit_start_ps * 1000,
            fit_stop_fs=None if fit_stop_ps is None else fit_stop_ps * 1000,
        )
        summary.update(
            {"species": mobile, "dimensions": dimensions, "framework_drift": framework}
        )
        outputs.append(store.write_json("diffusion_ols.json", summary))
        warnings: list[str] = [summary["note"]]
        plot = self._plot(
            lambda: __import__(
                "mlipx.analysis.plots", fromlist=["line_plot"]
            ).line_plot(
                store.prepare() / "msd.png",
                data["lag_time_fs"] / 1000,
                [
                    ("total", data["msd_A2"]),
                    ("x", data["msd_x_A2"]),
                    ("y", data["msd_y_A2"]),
                    ("z", data["msd_z_A2"]),
                ],
                xlabel="Lag time (ps)",
                ylabel="MSD (Å²)",
                title=f"MSD: {mobile}",
            ),
            warnings,
        )
        if plot:
            outputs.append(plot)
        metadata = store.complete(
            outputs=outputs, summary=summary, warnings=warnings, packages=["matplotlib"]
        )
        return {"cached": False, "path": str(store.path), "metadata": metadata}

    def run_density(
        self,
        *,
        mobile: str | None = None,
        grid: tuple[int, int, int] = (40, 40, 40),
        start: int = 0,
        stop: int | None = None,
        stride: int = 1,
        **_: Any,
    ) -> dict[str, Any]:
        mobile = mobile or self.dataset.symbols[0]
        parameters = {
            "species": mobile,
            "grid": grid,
            "start": start,
            "stop": stop,
            "stride": stride,
        }
        store = self._store("density", parameters)
        if store.cached() and not self.force:
            return {"cached": True, "path": str(store.path)}
        data = density_grid(
            self.dataset,
            species=mobile,
            grid=grid,
            start=start,
            stop=stop,
            stride=stride,
        )
        outputs = [store.write_npz("density.npz", **data)]
        warnings: list[str] = []
        plot = self._plot(
            lambda: __import__(
                "mlipx.analysis.plots", fromlist=["density_projection"]
            ).density_projection(
                store.prepare() / "density_projection.png",
                data["probability"],
                title=f"{mobile} density",
            ),
            warnings,
        )
        if plot:
            outputs.append(plot)
        summary = {
            "species": mobile,
            "grid": list(grid),
            "samples": int(data["counts"].sum()),
        }
        outputs.append(store.write_json("summary.json", summary))
        metadata = store.complete(
            outputs=outputs, summary=summary, warnings=warnings, packages=["matplotlib"]
        )
        return {"cached": False, "path": str(store.path), "metadata": metadata}

    def run_vacf(
        self,
        *,
        mobile: str | None = None,
        start: int = 0,
        stop: int | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        parameters = {"species": mobile, "start": start, "stop": stop}
        store = self._store("vacf", parameters)
        if store.cached() and not self.force:
            return {"cached": True, "path": str(store.path)}
        data = vacf_vdos(self.dataset, species=mobile, start=start, stop=stop)
        outputs = [
            store.write_csv(
                "vacf.csv",
                {
                    "lag_time_fs": data["lag_time_fs"],
                    "vacf": data["vacf"],
                    "vacf_normalized": data["vacf_normalized"],
                },
            ),
            store.write_csv(
                "vdos.csv",
                {"frequency_THz": data["frequency_THz"], "vdos_arb": data["vdos_arb"]},
            ),
        ]
        warnings: list[str] = []
        for name, x, y, xlabel, ylabel in (
            (
                "vacf",
                data["lag_time_fs"],
                data["vacf_normalized"],
                "Lag time (fs)",
                "Normalized VACF",
            ),
            (
                "vdos",
                data["frequency_THz"],
                data["vdos_arb"],
                "Frequency (THz)",
                "VDOS (arb.)",
            ),
        ):
            plot = self._plot(
                lambda name=name, x=x, y=y, xlabel=xlabel, ylabel=ylabel: __import__(
                    "mlipx.analysis.plots", fromlist=["line_plot"]
                ).line_plot(
                    store.prepare() / f"{name}.png",
                    x,
                    [(name, y)],
                    xlabel=xlabel,
                    ylabel=ylabel,
                    title=name.upper(),
                ),
                warnings,
            )
            if plot:
                outputs.append(plot)
        summary = {"species": mobile, "frames": len(data["lag_time_fs"])}
        metadata = store.complete(
            outputs=outputs, summary=summary, warnings=warnings, packages=["matplotlib"]
        )
        return {"cached": False, "path": str(store.path), "metadata": metadata}

    def run_transport(
        self,
        *,
        mobile: str | None = None,
        framework: str | None = None,
        dimensions: str = "xyz",
        fit_start_ps: float | None = None,
        temperature: float | None = None,
        charge: float | None = 1.0,
        kinisi_samples: int = 1000,
        kinisi_walkers: int = 32,
        kinisi_burn: int = 500,
        kinisi_thin: int = 10,
        random_seed: int = 0,
        **_: Any,
    ) -> dict[str, Any]:
        from mlipx.analysis.transport import kinisi_transport

        mobile = mobile or self.dataset.symbols[0]
        temperature = self.dataset.temperature_K if temperature is None else temperature
        parameters = {
            "species": mobile,
            "framework_drift": framework,
            "dimensions": dimensions,
            "fit_start_ps": fit_start_ps,
            "temperature_K": temperature,
            "ionic_charge_e": charge,
            "n_samples": kinisi_samples,
            "n_walkers": kinisi_walkers,
            "n_burn": kinisi_burn,
            "n_thin": kinisi_thin,
            "random_seed": random_seed,
        }
        store = self._store("transport_kinisi", parameters)
        if store.cached() and not self.force:
            return {"cached": True, "path": str(store.path)}
        data = kinisi_transport(
            self.dataset,
            species=mobile,
            framework=framework,
            dimensions=dimensions,
            fit_start_ps=fit_start_ps,
            temperature_K=temperature,
            ionic_charge_e=charge,
            n_samples=kinisi_samples,
            n_walkers=kinisi_walkers,
            n_burn=kinisi_burn,
            n_thin=kinisi_thin,
            random_seed=random_seed,
        )
        outputs = [
            store.write_csv(
                "kinisi_msd.csv",
                {
                    "lag_time_ps": data["lag_time_ps"],
                    "msd_A2": data["msd_A2"],
                    "msd_variance_A4": data["msd_variance_A4"],
                },
            ),
            store.write_npz(
                "posterior_samples.npz",
                diffusivity_cm2_s=data["diffusivity_samples_cm2_s"],
                **(
                    {"conductivity_mS_cm": data["conductivity_samples_mS_cm"]}
                    if "conductivity_samples_mS_cm" in data
                    else {}
                ),
            ),
            store.write_json("summary.json", data["summary"]),
        ]
        if "mscd" in data:
            outputs.append(
                store.write_csv(
                    "kinisi_mscd.csv",
                    {
                        "lag_time_ps": data["mscd_lag_time_ps"],
                        "mscd": data["mscd"],
                        "mscd_variance": data["mscd_variance"],
                    },
                )
            )
        warnings: list[str] = []
        plot = self._plot(
            lambda: __import__(
                "mlipx.analysis.plots", fromlist=["line_plot"]
            ).line_plot(
                store.prepare() / "kinisi_msd.png",
                data["lag_time_ps"],
                [("kinisi MSD", data["msd_A2"])],
                xlabel="Lag time (ps)",
                ylabel="MSD (Å²)",
                title=f"kinisi transport: {mobile}",
            ),
            warnings,
        )
        if plot:
            outputs.append(plot)
        metadata = store.complete(
            outputs=outputs,
            summary=data["summary"],
            warnings=warnings,
            packages=["kinisi", "scipp", "matplotlib"],
        )
        return {"cached": False, "path": str(store.path), "metadata": metadata}

    def run_electrolyte(
        self,
        *,
        mobile: str | None = None,
        temperature: float | None = None,
        gemdat_resolution: float = 0.5,
        sites: str | Path | None = None,
        background_level: float = 0.1,
        site_radius: float | None = None,
        minimal_residence: int = 0,
        percolation: str = "xyz",
        **_: Any,
    ) -> dict[str, Any]:
        from mlipx.analysis.electrolyte import gemdat_electrolyte

        mobile = mobile or self.dataset.symbols[0]
        temperature = self.dataset.temperature_K if temperature is None else temperature
        resolved_sites = (
            Path(sites).expanduser().resolve() if sites is not None else None
        )
        parameters = {
            "species": mobile,
            "temperature_K": temperature,
            "resolution_A": gemdat_resolution,
            "sites": resolved_sites,
            "sites_sha256": (
                sha256_file(resolved_sites) if resolved_sites is not None else None
            ),
            "background_level": background_level,
            "site_radius_A": site_radius,
            "minimal_residence": minimal_residence,
            "percolation": percolation,
        }
        store = self._store("electrolyte_gemdat", parameters)
        if store.cached() and not self.force:
            return {"cached": True, "path": str(store.path)}
        result = gemdat_electrolyte(
            self.dataset,
            species=mobile,
            temperature_K=temperature,
            resolution_A=gemdat_resolution,
            sites_path=resolved_sites,
            background_level=background_level,
            site_radius_A=site_radius,
            minimal_residence=minimal_residence,
            percolation=percolation,
        )
        outputs: list[Path] = [store.write_npz("volumes.npz", **result.arrays)]
        for name, table in result.tables.items():
            path = store.prepare() / f"{name}.csv"
            table.to_csv(path, index=True)
            outputs.append(path)
        for name, structure in result.structures.items():
            if name == "reference":
                continue
            path = store.prepare() / f"{name}.cif"
            structure.to(filename=str(path))
            outputs.append(path)
        reference = result.structures["reference"]
        for name, volume in result.volumes.items():
            stem = store.prepare() / name
            volume.to_vasp_volume(reference, filename=str(stem))
            outputs.append(stem.with_suffix(".vasp"))
        for axis, arrays in result.paths.items():
            outputs.append(
                store.write_csv(
                    f"percolation_{axis}.csv",
                    {
                        "voxel_x": arrays["voxel"][:, 0],
                        "voxel_y": arrays["voxel"][:, 1],
                        "voxel_z": arrays["voxel"][:, 2],
                        "fractional_x": arrays["fractional"][:, 0],
                        "fractional_y": arrays["fractional"][:, 1],
                        "fractional_z": arrays["fractional"][:, 2],
                        "cartesian_x_A": arrays["cartesian_A"][:, 0],
                        "cartesian_y_A": arrays["cartesian_A"][:, 1],
                        "cartesian_z_A": arrays["cartesian_A"][:, 2],
                        "free_energy_eV": arrays["free_energy_eV"],
                    },
                )
            )
        plot = self._plot(
            lambda: __import__(
                "mlipx.analysis.plots", fromlist=["density_projection"]
            ).density_projection(
                store.prepare() / "density_projection.png",
                result.arrays["density_probability"],
                title=f"GEMDAT {mobile} density",
            ),
            result.warnings,
        )
        if plot:
            outputs.append(plot)
        outputs.append(store.write_json("summary.json", result.summary))
        metadata = store.complete(
            outputs=outputs,
            summary=result.summary,
            warnings=result.warnings,
            packages=["gemdat", "pymatgen", "kinisi"],
        )
        return {"cached": False, "path": str(store.path), "metadata": metadata}


def analyze_run(
    source: str | Path, tasks: Iterable[str] | None = None, **options: Any
) -> dict[str, dict[str, Any]]:
    """Public convenience API for running one or more analyses."""
    runner_keys = {"frame_interval_fs", "assume_wrapped", "plots", "force"}
    runner_options = {
        key: options.pop(key) for key in list(options) if key in runner_keys
    }
    return AnalysisRunner(source, **runner_options).run(tasks=tasks, **options)
