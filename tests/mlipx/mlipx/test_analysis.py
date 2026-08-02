"""Tests for the layered MD post-processing API."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io.trajectory import Trajectory
from mlipx.analysis.core import (
    _msd_components_fft,
    arrhenius_fit,
    mean_squared_displacement,
    radial_distribution,
)
from mlipx.analysis.dataset import TrajectoryDataset
from mlipx.analysis.runner import AnalysisRunner
from mlipx.cli import create_parser, main


def _make_run(path: Path, nframes: int = 8) -> Path:
    raw = path / "raw"
    raw.mkdir(parents=True)
    with Trajectory(raw / "trajectory.traj", "w") as trajectory:
        for index in range(nframes):
            atoms = Atoms(
                "Li2O",
                positions=[
                    [0.2 + 0.15 * index, 0.2, 0.2],
                    [2.0 - 0.05 * index, 2.0, 2.0],
                    [3.5, 3.5, 3.5],
                ],
                cell=[5.0, 5.0, 5.0],
                pbc=True,
            )
            atoms.set_velocities(np.full((3, 3), 0.01 * (index + 1)))
            atoms.calc = SinglePointCalculator(
                atoms,
                energy=-10 + index * 0.01,
                forces=np.full((3, 3), 0.001 * index),
                stress=np.arange(6, dtype=float) * 1e-4,
            )
            trajectory.write(atoms)
    headers = [
        "step",
        "time_fs",
        "potential_energy_eV",
        "kinetic_energy_eV",
        "total_energy_eV",
        "temperature_K",
        "volume_A3",
    ]
    with (raw / "md.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for index in range(nframes):
            writer.writerow(
                [
                    index * 10,
                    index * 5.0,
                    -10 + index * 0.01,
                    0.5,
                    -9.5 + index * 0.01,
                    600,
                    125,
                ]
            )
    (path / "artifacts.json").write_text(
        json.dumps(
            {
                "trajectory": {
                    "timestep_fs": 0.5,
                    "save_interval_steps": 10,
                    "saved_interval_fs": 5.0,
                    "positions": "unwrapped Cartesian in trajectory.traj",
                },
            }
        ),
        encoding="utf-8",
    )
    (path / "resolved_config.json").write_text(
        json.dumps(
            {
                "run_options": {
                    "temperature": 600,
                    "timestep": 0.5,
                    "save_interval": 10,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_dataset_loads_run_contract(tmp_path: Path) -> None:
    dataset = TrajectoryDataset.load(_make_run(tmp_path / "run"))
    assert dataset.nframes == 8
    assert dataset.natoms == 3
    assert dataset.frame_interval_fs == 5.0
    assert dataset.temperature_K == 600
    assert dataset.select("Li").tolist() == [0, 1]
    assert dataset.velocities is not None
    assert dataset.forces_eV_A is not None
    assert dataset.potential_energy_eV is not None
    assert dataset.stress_eV_A3 is not None
    assert "kinetic_energy_eV" in dataset.thermodynamics
    assert dataset.validation_report()["valid"] is True


def test_dataset_recovers_legacy_timing_from_log(tmp_path: Path) -> None:
    run = _make_run(tmp_path / "legacy")
    (run / "artifacts.json").unlink()
    (run / "resolved_config.json").unlink()
    (run / "run.log").write_text(
        "Temperature:      700.0 K\n"
        "Time step:        0.5 fs\n"
        "Steps:            70\n"
        "Save interval:    10\n",
        encoding="utf-8",
    )
    dataset = TrajectoryDataset.load(run)
    assert dataset.frame_interval_fs == 5.0
    assert dataset.steps[-1] == 70
    assert dataset.temperature_K == 700
    assert "legacy run.log" in dataset.warnings[0]


def test_fft_msd_matches_direct_time_origin_average() -> None:
    rng = np.random.default_rng(4)
    positions = np.cumsum(rng.normal(size=(12, 3, 3)), axis=0)
    actual = _msd_components_fft(positions)
    expected = np.empty_like(actual)
    for lag in range(len(positions)):
        delta = positions[lag:] - positions[: len(positions) - lag]
        expected[lag] = np.mean(delta**2, axis=0)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_arrhenius_fit_recovers_activation_energy() -> None:
    temperature = np.asarray([500.0, 600.0, 700.0, 800.0])
    expected_ea = 0.32
    diffusivity = 2e-6 * np.exp(-expected_ea / (8.617333262145e-5 * temperature))
    result = arrhenius_fit(temperature, diffusivity)
    assert np.isclose(result["activation_energy_eV"], expected_ea)
    assert np.isclose(result["preexponential_factor_m2_s"], 2e-6)


def test_core_msd_and_partial_rdf(tmp_path: Path) -> None:
    dataset = TrajectoryDataset.load(_make_run(tmp_path / "run"))
    msd = mean_squared_displacement(dataset, species="Li")
    assert msd["msd_A2"][0] == 0
    assert msd["msd_A2"][-1] > 0
    assert msd["per_particle_msd_A2"].shape == (8, 2)
    rdf = radial_distribution(
        dataset, species_a="Li", species_b="O", r_max=2.4, bins=20
    )
    assert rdf["g_r"].shape == (20,)
    assert np.all(np.diff(rdf["coordination_number"]) >= 0)


def test_runner_writes_versioned_results_and_reuses_cache(tmp_path: Path) -> None:
    run = _make_run(tmp_path / "run")
    runner = AnalysisRunner(run, plots=False)
    result = runner.run(
        tasks=["validate", "rdf", "msd", "density"],
        mobile="Li",
        rdf_pairs=[("Li", "O")],
        rdf_rmax=2.4,
        rdf_bins=20,
        grid=(8, 8, 8),
    )
    for item in result.values():
        output = Path(item["path"])
        assert output.is_dir()
        assert (output / "metadata.json").exists()
        assert item["cached"] is False
    cached = runner.run(tasks=["msd"], mobile="Li")
    assert cached["msd"]["cached"] is True


def test_analysis_cli_parser_and_command(tmp_path: Path, capsys) -> None:
    parser = create_parser()
    args = parser.parse_args(
        [
            "analyze",
            "run",
            "--tasks",
            "msd",
            "rdf",
            "--mobile",
            "Li",
            "--rdf-pair",
            "Li-O",
            "--grid",
            "8",
            "8",
            "8",
            "--no-plots",
        ]
    )
    assert args.command == "analyze"
    assert args.rdf_pair == ["Li-O"]
    run = _make_run(tmp_path / "run")
    assert (
        main(
            [
                "analyze",
                str(run),
                "--tasks",
                "msd",
                "--mobile",
                "Li",
                "--no-plots",
                "--json",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert '"msd"' in output
