"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import csv
import gc
import tracemalloc

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.constraints import FixCom
from ase.io import Trajectory, read
from mlipx.config.defaults import BUILTIN_DEFAULTS
from mlipx.runners.md import MDRunner


class _CalculatorStub:
    inference_mode = "turbo"


@pytest.mark.parametrize("ensemble", ["NVT", "NVE"])
def test_md_initializes_velocities_at_target_temperature(tmp_path, ensemble):
    atoms = Atoms(
        "Ar4",
        positions=[
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
        ],
    )
    runner = MDRunner(
        _CalculatorStub(),
        ensemble=ensemble,
        temperature=900.0,
        output_dir=tmp_path,
        pre_relax=False,
        verbose=False,
    )

    runner._initialize_velocities(atoms)

    assert atoms.get_temperature() == pytest.approx(900.0)
    assert np.sum(atoms.get_momenta(), axis=0) == pytest.approx(np.zeros(3), abs=1e-12)


def _ar4():
    return Atoms(
        "Ar4",
        positions=[
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
        ],
    )


def test_md_seed_makes_velocities_reproducible(tmp_path):
    """Plan section 5.3: a recorded seed must actually drive velocity init."""
    a1, a2, a3 = _ar4(), _ar4(), _ar4()
    r42a = MDRunner(_CalculatorStub(), temperature=300.0, output_dir=tmp_path,
                    pre_relax=False, verbose=False, seed=42)
    r42b = MDRunner(_CalculatorStub(), temperature=300.0, output_dir=tmp_path,
                    pre_relax=False, verbose=False, seed=42)
    r7 = MDRunner(_CalculatorStub(), temperature=300.0, output_dir=tmp_path,
                  pre_relax=False, verbose=False, seed=7)
    r42a._initialize_velocities(a1)
    r42b._initialize_velocities(a2)
    r7._initialize_velocities(a3)
    # Same seed -> identical momenta; different seed -> different.
    assert np.allclose(a1.get_momenta(), a2.get_momenta())
    assert not np.allclose(a1.get_momenta(), a3.get_momenta())


def test_md_velocity_policy_preserve_requires_velocities(tmp_path):
    """preserve must fail loudly when there are no velocities (not silently init)."""
    runner = MDRunner(_CalculatorStub(), temperature=300.0, output_dir=tmp_path,
                      pre_relax=False, verbose=False, velocity_policy="preserve")
    with pytest.raises(ValueError, match="preserve"):
        runner._initialize_velocities(_ar4())


def test_md_velocity_policy_auto_preserves_existing(tmp_path):
    """auto keeps existing velocities instead of re-initializing them."""
    atoms = _ar4()
    atoms.set_momenta(np.full((4, 3), 0.5))
    runner = MDRunner(_CalculatorStub(), temperature=300.0, output_dir=tmp_path,
                      pre_relax=False, verbose=False, velocity_policy="auto")
    runner._initialize_velocities(atoms)
    assert np.allclose(atoms.get_momenta(), np.full((4, 3), 0.5))


def test_md_velocity_policy_preserves_explicit_zero_momenta(tmp_path):
    """An all-zero momenta array is a valid, explicitly supplied restart state."""
    atoms = _ar4()
    atoms.set_momenta(np.zeros((4, 3)))
    runner = MDRunner(
        _CalculatorStub(),
        temperature=300.0,
        output_dir=tmp_path,
        pre_relax=False,
        verbose=False,
        velocity_policy="preserve",
    )
    runner._initialize_velocities(atoms)
    assert np.array_equal(atoms.get_momenta(), np.zeros((4, 3)))


def test_md_velocity_policy_initialize_forces_reinit(tmp_path):
    """initialize re-initializes even when velocities already exist."""
    atoms = _ar4()
    atoms.set_momenta(np.full((4, 3), 0.5))
    runner = MDRunner(_CalculatorStub(), temperature=300.0, output_dir=tmp_path,
                      pre_relax=False, verbose=False, velocity_policy="initialize",
                      seed=11)
    runner._initialize_velocities(atoms)
    assert not np.allclose(atoms.get_momenta(), np.full((4, 3), 0.5))
    assert atoms.get_temperature() == pytest.approx(300.0)


@pytest.mark.parametrize("policy", ["auto", "preserve"])
def test_md_restart_momenta_reject_position_changing_pre_relax(tmp_path, policy):
    """A phase-space restart cannot silently alter positions before MD."""
    atoms = _bulk_atoms()
    atoms.set_momenta(np.ones((len(atoms), 3)))
    runner = MDRunner(
        _RunWrapper(_FiniteCalc()),
        ensemble="NVT",
        steps=1,
        output_dir=tmp_path,
        pre_relax=True,
        velocity_policy=policy,
        verbose=False,
    )
    with pytest.raises(ValueError, match="exact phase-space restart"):
        runner.run(atoms)


def test_md_rejects_equil_steps(tmp_path):
    """Plan section 5.2: equil_steps must not be silently ignored."""
    with pytest.raises(NotImplementedError, match="equil_steps"):
        MDRunner(_CalculatorStub(), output_dir=tmp_path, pre_relax=False,
                verbose=False, equil_steps=100)


def test_md_rejects_pre_relax_mode(tmp_path):
    """Plan section 5.2: pre_relax_mode must not be silently ignored."""
    with pytest.raises(NotImplementedError, match="pre_relax_mode"):
        MDRunner(_CalculatorStub(), output_dir=tmp_path, pre_relax=False,
                verbose=False, pre_relax_mode="positions")


def test_md_rejects_unknown_velocity_policy(tmp_path):
    with pytest.raises(ValueError, match="velocity_policy"):
        MDRunner(_CalculatorStub(), output_dir=tmp_path, pre_relax=False,
                verbose=False, velocity_policy="bogus")


# ---------------------------------------------------------------------------
# P1-1: streaming trajectory -- RAM must not grow with frame count (plan 5.5)
# ---------------------------------------------------------------------------
class _FiniteCalc(Calculator):
    """Trivial ASE calculator with optional balanced internal forces."""

    implemented_properties = ["energy", "forces", "stress"]  # noqa: RUF012

    def __init__(self, force_scale: float = 0.0):
        super().__init__()
        self._force_scale = force_scale

    def calculate(self, atoms, properties, system_changes):
        super().calculate(atoms, properties, system_changes)
        forces = np.zeros((len(atoms), 3))
        # Equal/opposite forces survive the explicit FixCom constraint. A
        # spatially uniform force is pure COM acceleration and is correctly
        # projected out, so it is not an internal "explosion" test.
        forces[::2, 0] = self._force_scale
        forces[1::2, 0] = -self._force_scale
        self.results = {
            "energy": -1.0,
            "forces": forces,
            "stress": np.zeros(6),
        }


class _RunWrapper:
    """Minimal BaseMLIPCalculator stub that returns a real ASE calculator."""

    def __init__(self, calc, task="bulk", has_stress=True):
        self._calc = calc
        self._task = task
        self._has_stress = has_stress

    def get_calculator(self):
        return self._calc

    @property
    def task(self):
        return self._task

    @property
    def has_stress(self):
        return self._has_stress

    @property
    def implemented_properties(self):
        return list(self._calc.implemented_properties)

    def info(self):
        return {"model_type": "stub", "task": self._task}


def _bulk_atoms(n=8):
    """A simple periodic noble-gas cell for MD."""
    return Atoms(
        f"Ar{n}",
        positions=np.linspace(0, 5, n * 3).reshape(n, 3),
        cell=[10, 10, 10],
        pbc=True,
    )


def test_md_streams_trajectory_to_disk(tmp_path):
    """Plan 5.5: full frames go to disk (trajectory.traj + XDATCAR), scalars
    to md.csv, and only scalar summaries stay in RAM."""
    wrapper = _RunWrapper(_FiniteCalc())
    runner = MDRunner(
        wrapper, ensemble="NVE", temperature=300.0, steps=10, save_interval=2,
        output_dir=tmp_path, pre_relax=False, verbose=False, seed=42,
    )
    results = runner.run(_bulk_atoms())

    frames = results["trajectory"]
    nframes = len(frames)
    assert nframes == 10 // 2 + 1  # step 0 + every 2nd step (ASE fires at step 0)
    # In-memory trajectory is scalars-only: NO per-frame atoms.copy().
    for f in frames:
        assert "atoms" not in f
        assert set(f) == {
            "step", "energy", "kinetic_energy", "total_energy", "temperature"
        }
    # Full frames are on disk in trajectory.traj.
    assert results["trajectory_path"] == str(tmp_path / "trajectory.traj")
    assert len(list(Trajectory(results["trajectory_path"]))) == nframes
    # XDATCAR was streamed using the standard VASP configuration marker and
    # is readable by ASE as a multi-frame trajectory.
    xdatcar = (tmp_path / "XDATCAR").read_text()
    assert xdatcar.count("Direct configuration=") == nframes
    assert "# Step:" not in xdatcar
    assert len(read(tmp_path / "XDATCAR", index=":", format="vasp-xdatcar")) == nframes
    # md.csv was streamed: header + nframes rows.
    with open(tmp_path / "md.csv", newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == [
        "step", "potential_energy_eV", "kinetic_energy_eV",
        "total_energy_eV", "temperature_K",
    ]
    assert len(rows) == nframes + 1
    assert results["md_csv_path"] == str(tmp_path / "md.csv")


def test_md_long_trajectory_keeps_only_scalars_in_memory(tmp_path):
    """Plan 5.5: a long run must not accumulate full atoms frames in RAM.

    Measures the marginal per-frame memory between a short (20-frame) and a
    long (200-frame) run of the *same* 256-atom system. With the streaming
    fix each frame adds only a small scalar dict (~hundreds of bytes); the old
    per-frame ``atoms.copy()`` would add the full positions array (~6 KB for
    256 atoms) per frame, so a 2 KB/frame cap cleanly separates the two.
    """
    def _peak_for(steps, out_dir):
        gc.collect()
        tracemalloc.start()
        runner = MDRunner(
            _RunWrapper(_FiniteCalc()), ensemble="NVE", temperature=300.0,
            steps=steps, save_interval=1, output_dir=out_dir,
            pre_relax=False, verbose=False, seed=1,
        )
        res = runner.run(_bulk_atoms(256))
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return peak, res

    peak_short, res_short = _peak_for(20, tmp_path / "short")
    peak_long, res_long = _peak_for(200, tmp_path / "long")

    # Structural guarantee: no atoms objects retained per frame in either run.
    assert all("atoms" not in f for f in res_short["trajectory"])
    assert all("atoms" not in f for f in res_long["trajectory"])
    assert len(res_long["trajectory"]) == 200 + 1
    assert len(res_short["trajectory"]) == 20 + 1
    # Marginal per-frame memory stays small (not a full atoms copy).
    marginal_per_frame = (peak_long - peak_short) / (200 - 20)
    assert marginal_per_frame < 2000, (
        f"marginal per-frame memory {marginal_per_frame:.0f} B/frame exceeds "
        "the scalar-only budget; full frames may be accumulating in RAM"
    )


# ---------------------------------------------------------------------------
# P1-2: finite-large-force warning threshold sourced from safety (plan 5.7)
# ---------------------------------------------------------------------------
def test_md_fmax_abort_defaults_to_safety():
    """The explosion-guard threshold defaults to safety.fmax_abort, not a
    magic literal (plan section 5.7)."""
    runner = MDRunner(
        _CalculatorStub(), output_dir=".", pre_relax=False, verbose=False
    )
    assert runner.fmax_abort == BUILTIN_DEFAULTS["safety"]["fmax_abort"]
    assert runner.fmax_abort == 20.0


def test_md_fmax_abort_warning_uses_configured_threshold(tmp_path):
    """A configurable fmax_abort drives the large-force warning."""
    logs: list[tuple[str, str]] = []
    wrapper = _RunWrapper(_FiniteCalc(force_scale=6.0))  # 6 eV/Å > threshold
    runner = MDRunner(
        wrapper, ensemble="NVE", temperature=300.0, steps=100,
        save_interval=1000, output_dir=tmp_path, pre_relax=False,
        verbose=False, seed=1, fmax_abort=5.0,
        log_fn=lambda msg, lvl: logs.append((msg, lvl)),
    )
    runner.run(_bulk_atoms())
    assert any("Large forces" in msg for msg, _ in logs)


def test_nvt_seed_reproduces_entire_stochastic_trajectory(tmp_path):
    """The seed must drive Langevin kicks, not only initial velocities."""
    def run(seed, name):
        out = tmp_path / name
        runner = MDRunner(
            _RunWrapper(_FiniteCalc()),
            ensemble="NVT",
            temperature=300.0,
            timestep=0.5,
            friction=0.01,
            steps=5,
            save_interval=1,
            output_dir=out,
            pre_relax=False,
            verbose=False,
            seed=seed,
        )
        runner.run(_bulk_atoms())
        return np.asarray([a.positions for a in Trajectory(out / "trajectory.traj")])

    same_a = run(42, "same-a")
    same_b = run(42, "same-b")
    different = run(7, "different")
    assert np.array_equal(same_a, same_b)
    assert not np.array_equal(same_a, different)


def test_md_uses_explicit_com_constraint_for_consistent_temperature_dof(tmp_path):
    atoms = _bulk_atoms()
    runner = MDRunner(
        _RunWrapper(_FiniteCalc()),
        ensemble="NVT",
        temperature=300,
        steps=0,
        output_dir=tmp_path,
        pre_relax=False,
        verbose=False,
        seed=3,
    )
    runner._ensure_com_constraint(atoms)
    runner._initialize_velocities(atoms)
    assert any(isinstance(c, FixCom) for c in atoms.constraints)
    assert atoms.get_number_of_degrees_of_freedom() == 3 * len(atoms) - 3
    assert atoms.get_temperature() == pytest.approx(300.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"timestep": 0.0}, "timestep"),
        ({"steps": -1}, "steps"),
        ({"save_interval": 0}, "save_interval"),
        ({"ensemble": "NVT", "friction": 0.0}, "friction"),
    ],
)
def test_md_rejects_nonphysical_parameters(tmp_path, kwargs, message):
    with pytest.raises(ValueError, match=message):
        MDRunner(
            _CalculatorStub(),
            output_dir=tmp_path,
            pre_relax=False,
            verbose=False,
            **kwargs,
        )
