from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from mlipx.analysis import TrajectoryDataset, require_analysis, validate_trajectory
from mlipx.analysis.thermo import thermodynamic_diagnostics
from mlipx.analysis.validation import InvalidTrajectoryError


def _frames(n: int = 4) -> list[Atoms]:
    return [
        Atoms(
            "LiS",
            positions=[[0.1 + 0.1 * i, 0, 0], [2, 2, 2]],
            cell=[5, 5, 5],
            pbc=True,
        )
        for i in range(n)
    ]


def test_uniform_time_axis_is_explicit_and_eligible() -> None:
    dataset = TrajectoryDataset.from_frames(
        _frames(),
        times_fs=[0, 10, 20, 30],
        positions_convention="unwrapped",
    )
    report = validate_trajectory(dataset)
    assert report.time.uniform is True
    assert report.time.frame_interval_fs == 10
    assert report.eligible_for_msd is True


def test_nonuniform_time_is_not_replaced_by_median() -> None:
    dataset = TrajectoryDataset.from_frames(
        _frames(),
        times_fs=[0, 10, 25, 35],
        positions_convention="unwrapped",
    )
    report = validate_trajectory(dataset)
    assert report.time.uniform is False
    assert report.time.frame_interval_fs is None
    assert dataset.frame_interval_fs is None
    with pytest.raises(InvalidTrajectoryError, match="uniform time axis"):
        require_analysis(dataset, "msd")


def test_unknown_position_convention_fails_transport_closed() -> None:
    dataset = TrajectoryDataset.from_frames(
        _frames(), times_fs=[0, 10, 20, 30], positions_convention="unknown"
    )
    assert not validate_trajectory(dataset).eligible_for_transport
    with pytest.raises(InvalidTrajectoryError, match="Position convention"):
        require_analysis(dataset, "transport")


def test_default_analysis_view_excludes_equilibration() -> None:
    dataset = TrajectoryDataset.from_frames(
        _frames(6),
        times_fs=np.arange(6) * 10,
        positions_convention="unwrapped",
        phases=[
            "equilibration",
            "equilibration",
            "production",
            "production",
            "production",
            "production",
        ],
    )
    production = dataset.analysis_view()
    all_frames = dataset.analysis_view(include_equilibration=True)
    assert production.nframes == 4
    assert production.times_fs.tolist() == [20, 30, 40, 50]
    assert all_frames.nframes == 6


def test_external_ase_momenta_supply_temperature_and_kinetic_energy() -> None:
    frames = _frames(3)
    for atoms in frames:
        atoms.set_velocities([[0.02, 0.0, 0.0], [0.0, 0.01, 0.0]])
    dataset = TrajectoryDataset.from_frames(
        frames,
        times_fs=[0, 2, 4],
        positions_convention="wrapped",
    )
    assert dataset.velocities is not None
    assert dataset.kinetic_energy_eV is not None
    assert dataset.temperature_K is not None
    assert np.all(dataset.kinetic_energy_eV > 0)
    assert np.all(dataset.temperature_K > 0)


def test_nve_energy_drift_uses_production_per_atom_energy() -> None:
    dataset = TrajectoryDataset.from_frames(
        _frames(5),
        times_fs=np.arange(5) * 1000.0,
        positions_convention="unwrapped",
        phases=[
            "equilibration",
            "production",
            "production",
            "production",
            "production",
        ],
        metadata={"resolved_config": {"run_options": {"ensemble": "NVE"}}},
    )
    time_ps = dataset.times_fs / 1000.0
    dataset.total_energy_eV = 2 * (-5.0 + 0.002 * time_ps)
    result = thermodynamic_diagnostics(dataset)
    assert result["summary"]["n_frames"] == 4
    assert result["summary"]["nve_total_energy_drift_eV_atom_ps"] == pytest.approx(
        0.002
    )
