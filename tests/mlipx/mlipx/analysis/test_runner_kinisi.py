from __future__ import annotations

import csv
import json

import numpy as np
import pytest
from ase import Atoms
from ase.io.trajectory import Trajectory

from mlipx.analysis import TrajectoryDataset
from mlipx.analysis.runner import run_analysis
from mlipx.analysis.schema import AnalysisRequest
from mlipx.analysis.transport import kinisi_transport


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
    from mlipx.cli import main

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
                "fit_start_ps": 0.002,
                "fit_stop_ps": 0.006,
            },
        )
    )
    assert outcome["status"] == "success"
    assert len(outcome["results"]["lag_time_ps"]) == 4
    output = run / "analysis" / "msd" / outcome["analysis_id"]
    request = json.loads((output / "request.json").read_text(encoding="utf-8"))
    assert request["task_output_revision"] == 3
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
    assert payload["results"]["fit_window_ps"] == {"start": 0.002, "stop": 0.006}
    with (output / "diffusion_fits.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        fit_rows = list(csv.DictReader(handle))
    assert [row["axes"] for row in fit_rows] == ["x", "y", "z", "xy", "xyz"]
    assert all("self_diffusion_coefficient_m2_s" in row for row in fit_rows)


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
