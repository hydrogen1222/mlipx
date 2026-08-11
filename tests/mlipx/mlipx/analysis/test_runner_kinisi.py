from __future__ import annotations

import csv
import json
from typing import ClassVar

import mlipx.analysis.runner as runner_module
import mlipx.analysis.transport as transport_module
import numpy as np
import pytest
from ase import Atoms
from ase.io.trajectory import Trajectory
from mlipx.analysis import TrajectoryDataset
from mlipx.analysis.runner import run_analysis
from mlipx.analysis.schema import AnalysisRequest
from mlipx.analysis.transport import (
    DEFAULT_MAX_NATIVE_KINISI_LAG_POINTS,
    _resolve_kinisi_lag_grid,
    kinisi_transport,
)
from mlipx.cli import main


def _write_short_run(path) -> None:
    raw = path / "raw"
    raw.mkdir(parents=True)
    positions = [9.4, 9.7, 0.0, 0.3, 0.6, 0.9]
    with Trajectory(raw / "trajectory.traj", "w") as writer:
        for index, x in enumerate(positions):
            atoms = Atoms(
                "LiS",
                positions=[[x, 1, 1], [5, 5, 5]],
                cell=[10, 10, 10],
                pbc=True,
            )
            atoms.info["mlipx_step"] = index
            atoms.info["mlipx_time_fs"] = float(index * 2)
            atoms.info["mlipx_phase"] = "equilibration" if index < 2 else "production"
            writer.write(atoms)
    with (raw / "md.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "step",
                "time_fs",
                "phase",
                "temperature_K",
                "potential_energy_eV",
                "kinetic_energy_eV",
                "total_energy_eV",
                "volume_A3",
            ]
        )
        for index in range(6):
            writer.writerow(
                [
                    index,
                    index * 2,
                    "equilibration" if index < 2 else "production",
                    600,
                    -2,
                    0.1,
                    -1.9,
                    1000,
                ]
            )
    (path / "artifacts.json").write_text(
        json.dumps(
            {
                "schema": "mlipx.md-artifacts/2",
                "status": "completed",
                "trajectory": {
                    "md_timestep_fs": 1.0,
                    "frame_stride_steps": 2,
                    "frame_interval_fs": 2.0,
                    "positions_convention": "wrapped",
                    "production_start_step": 2,
                },
            }
        )
    )
    (path / "resolved_config.json").write_text(
        json.dumps(
            {
                "run_options": {
                    "ensemble": "NVE",
                    "temperature": 600,
                }
            }
        )
    )


def test_analysis_runner_writes_provenance_results_and_reuses_id(tmp_path) -> None:
    run = tmp_path / "short-run"
    _write_short_run(run)
    first = run_analysis(AnalysisRequest("validate", str(run)))
    assert first["status"] == "success"
    assert first["results"]["production_frames"] == 4
    output = run / "analysis" / "validate" / first["analysis_id"]
    assert (output / "request.json").is_file()
    assert (output / "provenance.json").is_file()
    assert (output / "results.json").is_file()
    assert (run / "analysis" / "index.json").is_file()

    second = run_analysis(AnalysisRequest("validate", str(run)))
    assert second["analysis_id"] == first["analysis_id"]
    assert second["reused"] is True


def test_cli_analyze_validate_short_run(tmp_path, capsys) -> None:
    run = tmp_path / "short-run"
    _write_short_run(run)
    assert main(["analyze", str(run), "validate"]) == 0
    output = capsys.readouterr().out
    assert "Analysis success" in output
    assert "Production frames: 4" in output
    assert "MSD/transport eligible: True" in output


def test_analysis_runner_msd_defaults_to_production(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    run = tmp_path / "short-run"
    _write_short_run(run)
    outcome = run_analysis(
        AnalysisRequest(
            "msd",
            str(run),
            parameters={
                "mobile_species": "Li",
                "axes": "x,y,z,xy,xyz",
                "drift_reference": "none",
            },
        )
    )
    assert outcome["status"] == "success"
    assert len(outcome["results"]["lag_time_ps"]) == 4
    output = run / "analysis" / "msd" / outcome["analysis_id"]
    request = json.loads((output / "request.json").read_text(encoding="utf-8"))
    assert request["task_output_revision"] == 5
    assert (output / "msd.csv").is_file()
    assert (output / "msd.png").is_file()
    assert (output / "msd.svg").is_file()
    assert (output / "alpha.png").is_file()
    assert (output / "alpha.svg").is_file()
    assert (output / "diffusion_fits.csv").is_file()
    with (output / "msd.csv").open(newline="", encoding="utf-8") as handle:
        columns = next(csv.reader(handle))
    for axes in ("x", "y", "z", "xy", "xyz"):
        assert f"alpha_{axes}" in columns
    payload = json.loads((output / "results.json").read_text(encoding="utf-8"))
    assert {"alpha.png", "alpha.svg", "diffusion_fits.csv"} <= set(
        payload["artifacts"]
    )
    assert payload["results"]["fit_window_ps"] == {"start": 0.0, "stop": 0.006}
    assert payload["results"]["fit_window_source"] == "full_trajectory_default"
    with (output / "diffusion_fits.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        fit_rows = list(csv.DictReader(handle))
    assert [row["axes"] for row in fit_rows] == ["x", "y", "z", "xy", "xyz"]
    assert all("self_diffusion_coefficient_m2_s" in row for row in fit_rows)
    assert all(row["fit_window_source"] == "full_trajectory_default" for row in fit_rows)


def test_transport_analysis_id_includes_lag_parameters_and_revision(
    tmp_path, monkeypatch
) -> None:
    run = tmp_path / "short-run"
    _write_short_run(run)

    def fake_dispatch(_request, _output_dir):
        return (
            {
                "tracer_diffusion": {
                    "fit_start_ps": 40.0,
                    "fit_stop_ps": 200.0,
                    "lag_grid": {
                        "mode": "custom",
                        "requested_step_ps": 1.0,
                        "requested_stop_ps": 200.0,
                    },
                }
            },
            [],
        )

    monkeypatch.setattr(runner_module, "_dispatch", fake_dispatch)
    first = run_analysis(
        AnalysisRequest(
            "transport",
            str(run),
            parameters={"lag_step_ps": 1.0, "lag_stop_ps": 200.0},
        )
    )
    second = run_analysis(
        AnalysisRequest(
            "transport",
            str(run),
            parameters={"lag_step_ps": 2.0, "lag_stop_ps": 200.0},
        )
    )
    assert first["analysis_id"] != second["analysis_id"]
    first_output = run / "analysis" / "transport" / first["analysis_id"]
    request = json.loads((first_output / "request.json").read_text(encoding="utf-8"))
    provenance = json.loads(
        (first_output / "provenance.json").read_text(encoding="utf-8")
    )
    assert request["task_output_revision"] == 1
    assert provenance["parameters"]["lag_step_ps"] == 1.0
    assert provenance["transport"]["lag_grid"]["requested_step_ps"] == 1.0
    reused = run_analysis(
        AnalysisRequest(
            "transport",
            str(run),
            parameters={"lag_step_ps": 1.0, "lag_stop_ps": 200.0},
        )
    )
    assert reused["analysis_id"] == first["analysis_id"]
    assert reused["reused"] is True


@pytest.mark.parametrize(
    ("lag_step_ps", "expected_points"),
    [(1.0, 200), (0.5, 400), (2.0, 100)],
)
def test_resolve_custom_kinisi_lag_grid(lag_step_ps, expected_points) -> None:
    result = _resolve_kinisi_lag_grid(
        frame_interval_fs=10.0,
        total_duration_ps=400.0,
        fit_start_ps=40.0,
        lag_step_ps=lag_step_ps,
        lag_stop_ps=200.0,
    )
    assert result["mode"] == "custom"
    assert result["n_lag_points"] == expected_points
    assert result["lag_frame_indices"][0] == int(lag_step_ps * 100)
    assert result["lag_frame_indices"][-1] == 20000
    assert 4000 in result["lag_frame_indices"]
    assert result["actual_stop_ps"] == pytest.approx(200.0)


def test_resolve_custom_grid_inserts_fit_start() -> None:
    result = _resolve_kinisi_lag_grid(
        frame_interval_fs=10.0,
        total_duration_ps=400.0,
        fit_start_ps=40.0,
        lag_step_ps=3.0,
        lag_stop_ps=198.0,
    )
    assert 4000 in result["lag_frame_indices"]
    assert result["lag_frame_indices"][-1] == 19800


@pytest.mark.parametrize(
    ("lag_step_ps", "lag_stop_ps", "message"),
    [
        (1.0, None, "must be provided together"),
        (None, 200.0, "must be provided together"),
        (0.0, 200.0, "lag_step_ps must be finite and positive"),
        (-1.0, 200.0, "lag_step_ps must be finite and positive"),
        (1.0, 40.0, "lag_stop_ps must be greater than fit_start_ps"),
        (1.0, 401.0, "exceeds production duration"),
        (0.015, 200.0, "incompatible with the trajectory frame interval"),
        (1.0, 200.005, "incompatible with the trajectory frame interval"),
    ],
)
def test_resolve_custom_grid_rejects_invalid_parameters(
    lag_step_ps, lag_stop_ps, message
) -> None:
    with pytest.raises(ValueError, match=message):
        _resolve_kinisi_lag_grid(
            frame_interval_fs=10.0,
            total_duration_ps=400.0,
            fit_start_ps=40.0,
            lag_step_ps=lag_step_ps,
            lag_stop_ps=lag_stop_ps,
        )


def test_resolve_native_grid_guard() -> None:
    with pytest.raises(ValueError, match=r"default lag grid.*40000"):
        _resolve_kinisi_lag_grid(
            frame_interval_fs=10.0,
            total_duration_ps=400.0,
            fit_start_ps=40.0,
        )

    result = _resolve_kinisi_lag_grid(
        frame_interval_fs=10.0,
        total_duration_ps=10.0,
        fit_start_ps=1.0,
        native_lag_guard=DEFAULT_MAX_NATIVE_KINISI_LAG_POINTS,
    )
    assert result["mode"] == "kinisi_default"
    assert result["lag_times_fs"] is None
    assert result["estimated_n_lag_points"] == 1000


def _dense_dataset(tmp_path) -> TrajectoryDataset:
    n_frames = 40001
    cells = np.broadcast_to(
        np.diag([10.0, 10.0, 10.0]), (n_frames, 3, 3)
    ).copy()
    positions = np.zeros((n_frames, 2, 3), dtype=float)
    positions[:, 0, 0] = np.arange(n_frames, dtype=float) * 0.001
    return TrajectoryDataset(
        run_dir=tmp_path,
        source_path=tmp_path / "trajectory.traj",
        positions=positions,
        cells=cells,
        pbc=np.ones(3, dtype=bool),
        symbols=("Li", "S"),
        masses=np.asarray([7.0, 32.0]),
        times_fs=np.arange(n_frames, dtype=float) * 10.0,
        steps=None,
        positions_convention="unwrapped",
        frame_interval_fs=10.0,
    )


def test_dense_native_guard_runs_before_kinisi(tmp_path, monkeypatch) -> None:
    dataset = _dense_dataset(tmp_path)
    monkeypatch.setattr(
        transport_module,
        "_require_kinisi",
        lambda: pytest.fail("kinisi must not be imported for a guarded grid"),
    )
    with pytest.raises(ValueError, match=r"default lag grid.*40000"):
        kinisi_transport(
            dataset,
            mobile_species="Li",
            ionic_charge_e=1,
            fit_start_ps=40.0,
            temperature_K=700.0,
        )


def _synthetic_transport_dataset() -> TrajectoryDataset:
    n_frames = 140
    positions = np.zeros((n_frames, 2, 3), dtype=float)
    positions[:, 0, 0] = np.arange(n_frames, dtype=float) * 0.01
    positions[:, 0, 1] = np.arange(n_frames, dtype=float) * 0.005
    frames = [
        Atoms("LiS", positions=frame, cell=[100, 100, 100], pbc=True)
        for frame in positions
    ]
    return TrajectoryDataset.from_frames(
        frames,
        times_fs=np.arange(n_frames, dtype=float) * 2.0,
        positions_convention="unwrapped",
        md_timestep_fs=0.5,
        frame_stride_steps=4,
    )


def test_transport_tracer_and_collective_share_custom_dt(monkeypatch) -> None:
    sc = pytest.importorskip("scipp")
    dataset = _synthetic_transport_dataset()

    class FakeDiffusionAnalyzer:
        calls: ClassVar[list[dict[str, object]]] = []

        @classmethod
        def from_ase(cls, **kwargs):
            cls.calls.append(kwargs)
            analyzer = cls()
            analyzer.dt = kwargs.get("dt")
            if analyzer.dt is None:
                analyzer.dt = sc.array(
                    dims=["time interval"],
                    values=np.arange(1, 140, dtype=float) * 2.0,
                    unit="fs",
                )
            return analyzer

        def diffusion(self, *_args, **_kwargs):
            n_points = self.dt.values.size
            self.msd = sc.array(
                dims=["time interval"],
                values=np.arange(1, n_points + 1, dtype=float),
                variances=np.ones(n_points),
                unit="angstrom^2",
            )
            self.D = sc.array(
                dims=["sample"],
                values=np.full(16, 1.0e-10),
                unit="m^2/s",
            )

    class FakeConductivityAnalyzer:
        calls: ClassVar[list[dict[str, object]]] = []

        @classmethod
        def from_ase(cls, **kwargs):
            cls.calls.append(kwargs)
            analyzer = cls()
            analyzer.dt = kwargs.get("dt")
            return analyzer

        def conductivity(self, *_args, **_kwargs):
            self.sigma = sc.array(
                dims=["sample"],
                values=np.full(16, 1.0e-3),
                unit="mS/cm",
            )

    monkeypatch.setattr(
        transport_module,
        "_require_kinisi",
        lambda: (
            sc,
            FakeDiffusionAnalyzer,
            FakeConductivityAnalyzer,
            "2.1.0",
        ),
    )
    result = kinisi_transport(
        dataset,
        mobile_species="Li",
        ionic_charge_e=1,
        fit_start_ps=0.07,
        lag_step_ps=0.02,
        lag_stop_ps=0.2,
        temperature_K=600.0,
        collective_conductivity=True,
        n_samples=10,
        n_walkers=16,
        n_burn=10,
        n_thin=1,
    )
    tracer_dt = FakeDiffusionAnalyzer.calls[0]["dt"]
    collective_dt = FakeConductivityAnalyzer.calls[0]["dt"]
    assert tracer_dt is collective_dt
    np.testing.assert_array_equal(
        tracer_dt.values, np.asarray([20, 40, 60, 70, 80, 100, 120, 140, 160, 180, 200])
    )
    assert result["tracer_diffusion"]["fit_stop_ps"] == pytest.approx(0.2)
    assert result["tracer_diffusion"]["lag_grid"]["n_lag_points_total"] == 11
    assert result["tracer_diffusion"]["lag_grid"]["n_lag_points_in_fit"] == 8
    assert result["kinisi_resource_diagnostics"][
        "single_float64_square_matrix_lower_bound_bytes"
    ] == 8 * 8**2


def test_kinisi_adapter_matches_official_ase_api() -> None:
    sc = pytest.importorskip("scipp")
    kinisi = pytest.importorskip("kinisi.analyze")
    rng = np.random.default_rng(4)
    n_frames = 140
    n_particles = 10
    walk = np.concatenate(
        (
            np.zeros((1, n_particles, 3)),
            np.cumsum(
                rng.normal(scale=0.1, size=(n_frames - 1, n_particles, 3)),
                axis=0,
            ),
        ),
        axis=0,
    )
    frames = [
        Atoms(
            f"Li{n_particles}",
            positions=positions,
            cell=[100, 100, 100],
            pbc=True,
        )
        for positions in walk
    ]
    dataset = TrajectoryDataset.from_frames(
        frames,
        times_fs=np.arange(n_frames) * 2.0,
        positions_convention="unwrapped",
        md_timestep_fs=0.5,
        frame_stride_steps=4,
    )
    adapter = kinisi_transport(
        dataset,
        mobile_species="Li",
        ionic_charge_e=1,
        fit_start_ps=0.07,
        temperature_K=600,
        random_seed=5,
        n_samples=10,
        n_walkers=16,
        n_burn=10,
        n_thin=1,
    )
    official = kinisi.DiffusionAnalyzer.from_ase(
        trajectory=frames,
        specie="Li",
        time_step=sc.scalar(0.5, unit="fs"),
        step_skip=sc.scalar(4, unit="dimensionless"),
        dimension="xyz",
        progress=False,
    )
    official.diffusion(
        sc.scalar(70, unit="fs"),
        n_samples=10,
        n_walkers=16,
        n_burn=10,
        n_thin=1,
        progress=False,
        random_state=np.random.RandomState(5),
    )
    np.testing.assert_allclose(adapter["lag_time_ps"], official.dt.to(unit="ps").values)
    np.testing.assert_allclose(
        adapter["kinisi_msd_A2"], official.msd.to(unit="angstrom^2").values
    )
    assert adapter["kinisi_time_mapping"]["resulting_frame_interval_fs"] == 2.0
    assert adapter["tracer_diffusion"]["D_posterior_m2_s"]["mean"] == pytest.approx(
        float(np.mean(official.D.to(unit="m^2/s").values))
    )


def test_kinisi_adapter_custom_dt_smoke() -> None:
    pytest.importorskip("scipp")
    pytest.importorskip("kinisi.analyze")
    rng = np.random.default_rng(7)
    n_frames = 80
    n_particles = 4
    walk = np.concatenate(
        (
            np.zeros((1, n_particles, 3)),
            np.cumsum(
                rng.normal(scale=0.1, size=(n_frames - 1, n_particles, 3)),
                axis=0,
            ),
        ),
        axis=0,
    )
    frames = [
        Atoms(
            f"Li{n_particles}",
            positions=positions,
            cell=[100, 100, 100],
            pbc=True,
        )
        for positions in walk
    ]
    dataset = TrajectoryDataset.from_frames(
        frames,
        times_fs=np.arange(n_frames, dtype=float) * 2.0,
        positions_convention="unwrapped",
        md_timestep_fs=0.5,
        frame_stride_steps=4,
    )
    adapter = kinisi_transport(
        dataset,
        mobile_species="Li",
        ionic_charge_e=1,
        fit_start_ps=0.03,
        lag_step_ps=0.02,
        lag_stop_ps=0.12,
        temperature_K=600,
        random_seed=7,
        n_samples=10,
        n_walkers=16,
        n_burn=10,
        n_thin=1,
    )
    np.testing.assert_allclose(
        adapter["lag_time_ps"], np.asarray([0.02, 0.03, 0.04, 0.06, 0.08, 0.1, 0.12])
    )
    assert len(adapter["kinisi_msd_A2"]) == 7
    assert adapter["tracer_diffusion"]["fit_stop_ps"] == pytest.approx(0.12)
    assert adapter["tracer_diffusion"]["D_posterior_m2_s"]["posterior_samples"] > 0
