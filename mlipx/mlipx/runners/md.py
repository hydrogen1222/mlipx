"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Modified for the mlipx project: multi-engine MLIP support (UMA/MACE/DPA/GRACE).
Molecular dynamics runner.

Runs MD simulations using ASE's integrators:
- NVT ensemble (Langevin, Bussi/CSVR, or Nose-Hoover-chain dynamics)
- NVE ensemble (Velocity Verlet)

Outputs trajectories in multiple formats.
"""

from __future__ import annotations

import json
import threading
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from ase import units
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixCom
from ase.md.bussi import Bussi
from ase.md.langevin import Langevin
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.verlet import VelocityVerlet
from ase.optimize import FIRE

from mlipx.config.defaults import BUILTIN_DEFAULTS
from mlipx.protocols import CancellationRequested
from mlipx.runners.base import BaseRunner
from mlipx.runners.md_output import (
    AsyncMDOutputWriter,
    MDFrameSnapshot,
    MDFrameStats,
    MDTrajectorySummary,
)
from mlipx.writers.contcar import ContcarWriter
from mlipx.writers.outcar import MDOutcarWriter
from mlipx.writers.xdatcar import XdatcarWriter
from mlipx.writers.json_writer import JsonWriter

if TYPE_CHECKING:
    from typing import Any

    from ase import Atoms
    from mlipx.protocols import ProgressCallback


class ForceSafetyAbort(RuntimeError):
    """Raised after checkpointing an MD frame that exceeds ``fmax_abort``."""

    def __init__(
        self, *, step: int, max_force: float, atom_index: int, threshold: float
    ):
        self.step = step
        self.max_force = max_force
        self.atom_index = atom_index
        self.threshold = threshold
        super().__init__(
            f"Force safety abort at MD step {step}: atom {atom_index} has "
            f"|F|={max_force:.6g} eV/Angstrom, exceeding "
            f"fmax_abort={threshold:.6g} eV/Angstrom"
        )


class MDRunner(BaseRunner):
    """Run molecular dynamics simulations.

    Supports NVT (Langevin, Bussi/CSVR, or Nose-Hoover chain) and NVE
    (Velocity Verlet) ensembles.
    Can optionally reduce large initial atomic forces with a positions-only
    pre-relaxation before MD.

    Example:
        >>> runner = MDRunner(
        ...     calculator,
        ...     ensemble="NVT",
        ...     temperature=300,
        ...     timestep=1.0,
        ...     steps=10000
        ... )
        >>> results = runner.run(atoms)
        >>> print(f"Final temperature: {results['temperature']:.1f} K")
    """

    VALID_ENSEMBLES = {"nvt", "nve"}
    VALID_THERMOSTATS = {"langevin", "bussi", "nhc"}
    # Every MD step is retained in run.log independently of trajectory saving.
    # LiveRunLogger coalesces the actual filesystem flushes and suppresses these
    # high-frequency records from UI callbacks.
    LOG_INTERVAL_STEPS = 1
    PROGRESS_INTERVAL_STEPS = 100

    def __init__(
        self,
        calculator,
        ensemble: str = "NVT",
        temperature: float = 300.0,
        timestep: float = 1.0,
        steps: int = 1000,
        equilibration_steps: int = 0,
        thermostat: str = "LANGEVIN",
        friction: float = 0.001,
        bussi_tau: float = 1000.0,
        nhc_tdamp: float = 100.0,
        nhc_tchain: int = 3,
        nhc_tloop: int = 1,
        save_interval: int = 10,
        output_dir: Path | str = ".",
        write_outcar: bool = True,
        write_forces: bool = True,
        write_stress: bool = True,
        write_xdatcar: bool = True,
        write_trajectory: bool = True,
        write_json: bool = True,
        verbose: bool = True,
        job_name: str | None = None,
        # NEW: Pre-relaxation options
        pre_relax: bool = True,
        pre_relax_steps: int = 50,
        pre_relax_fmax: float = 0.1,
        # Reproducibility / velocity policy (plan section 5.2 / 5.3).
        seed: int | None = None,
        velocity_policy: str = "auto",
        pre_relax_mode: str = "none",
        # Explosion-guard threshold for finite (but unsafe) forces.
        fmax_abort: float = BUILTIN_DEFAULTS["safety"]["fmax_abort"],
        log_fn: Any | None = None,
        progress_callback: ProgressCallback | None = None,
        cancel_event: threading.Event | None = None,
        charge: int | None = None,
        spin: int | None = None,
    ):
        """Initialize MD runner.

        Args:
            calculator: UMA calculator wrapper
            ensemble: MD ensemble (NVT or NVE)
            temperature: Temperature in Kelvin
            timestep: Time step in femtoseconds
            steps: Number of production MD steps
            equilibration_steps: Same-ensemble MD steps before production
            thermostat: NVT thermostat (LANGEVIN, BUSSI, or NHC)
            friction: Langevin friction coefficient (1/fs)
            bussi_tau: Bussi/CSVR coupling time in femtoseconds
            nhc_tdamp: Nose-Hoover-chain damping time in femtoseconds
            nhc_tchain: Nose-Hoover chain length
            nhc_tloop: Nose-Hoover thermostat integration substeps
            save_interval: Interval for saving trajectory frames
            output_dir: Directory for output files
            write_outcar: Whether to write OUTCAR file
            write_forces: Whether OUTCAR includes the final force table
            write_stress: Whether OUTCAR includes the final stress tensor
            write_xdatcar: Whether to write XDATCAR file
            write_trajectory: Whether to write ASE trajectory
            write_json: Whether to write JSON results
            verbose: Whether to print progress messages
            job_name: Optional job name for organizing results
            pre_relax: Whether to perform pre-relaxation before MD
            pre_relax_steps: Maximum steps for pre-relaxation
            pre_relax_fmax: Force threshold for pre-relaxation
            log_fn: Optional callback function for custom log output
            progress_callback: Optional callback for progress events
        """
        super().__init__(
            calculator,
            output_dir,
            verbose,
            job_name,
            log_fn,
            progress_callback,
            cancel_event=cancel_event,
            charge=charge,
            spin=spin,
        )
        self.ensemble = ensemble.lower()
        self.temperature = temperature
        self.timestep = timestep * units.fs  # Convert to ASE units
        self.steps = int(steps)
        self.equilibration_steps = int(equilibration_steps)
        self.total_steps = self.equilibration_steps + self.steps
        self.thermostat = str(thermostat).lower()
        self.friction = friction / units.fs  # Convert to ASE units
        self.bussi_tau = bussi_tau * units.fs
        self.nhc_tdamp = nhc_tdamp * units.fs
        self.nhc_tchain = int(nhc_tchain)
        self.nhc_tloop = int(nhc_tloop)
        self.save_interval = save_interval
        self.write_outcar = write_outcar
        self.write_forces = write_forces
        self.write_stress = write_stress
        self.write_xdatcar = write_xdatcar
        self.write_trajectory = write_trajectory
        self.write_json = write_json

        # MD output contract: ``raw`` is the lossless internal source of truth
        # and ``vasp`` is a syntax-compatible interoperability view.
        self.raw_dir = self.output_dir / "raw"
        self.vasp_dir = self.output_dir / "vasp"
        for directory in (self.raw_dir, self.vasp_dir):
            directory.mkdir(parents=True, exist_ok=True)

        # NEW: Pre-relaxation settings
        self.pre_relax = pre_relax
        self.pre_relax_steps = pre_relax_steps
        self.pre_relax_fmax = pre_relax_fmax

        # Reproducibility / velocity policy (plan section 5.2 / 5.3).
        self.seed = seed
        # One generator drives *both* the initial Maxwell-Boltzmann draw and
        # every stochastic Langevin kick.  Seeding only the initial velocities
        # does not make an NVT trajectory reproducible.
        self.rng = np.random.default_rng(seed)
        self.velocity_policy = str(velocity_policy).lower()
        self.pre_relax_mode = str(pre_relax_mode).lower()
        self.fmax_abort = float(fmax_abort)

        if self.velocity_policy not in {"auto", "initialize", "preserve"}:
            raise ValueError(
                f"Unknown velocity_policy {velocity_policy!r}. "
                "Use one of: auto, initialize, preserve."
            )
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0 K")
        if self.timestep <= 0:
            raise ValueError("timestep must be > 0 fs")
        if self.steps < 0:
            raise ValueError("steps must be >= 0")
        if self.equilibration_steps < 0:
            raise ValueError("equilibration_steps must be >= 0")
        if self.save_interval <= 0:
            raise ValueError("save_interval must be > 0")
        if self.fmax_abort <= 0:
            raise ValueError("fmax_abort must be > 0 eV/Angstrom")
        if self.ensemble == "nvt":
            if self.thermostat not in self.VALID_THERMOSTATS:
                raise ValueError(
                    f"Unknown thermostat: {thermostat}. Use one of: "
                    f"{', '.join(sorted(self.VALID_THERMOSTATS))}"
                )
            if self.thermostat == "langevin" and self.friction <= 0:
                raise ValueError("friction must be > 0 fs^-1 for NVT Langevin dynamics")
            if self.thermostat == "bussi" and self.bussi_tau <= 0:
                raise ValueError("bussi_tau must be > 0 fs for NVT Bussi dynamics")
            if self.thermostat == "nhc":
                if self.nhc_tdamp <= 0:
                    raise ValueError("nhc_tdamp must be > 0 fs for NVT NHC dynamics")
                if self.nhc_tchain < 1:
                    raise ValueError("nhc_tchain must be >= 1 for NVT NHC dynamics")
                if self.nhc_tloop < 1:
                    raise ValueError("nhc_tloop must be >= 1 for NVT NHC dynamics")
        # pre_relax_mode is future vocabulary and must not be silently ignored.
        if self.pre_relax_mode != "none":
            raise NotImplementedError(
                f"pre_relax_mode={self.pre_relax_mode!r} is not yet implemented "
                "(Phase 3). Use the legacy pre_relax=True/False (positions-only) "
                "for now."
            )
        # Validate ensemble
        if self.ensemble not in self.VALID_ENSEMBLES:
            raise ValueError(
                f"Unknown ensemble: {ensemble}. "
                f"Use one of: {', '.join(self.VALID_ENSEMBLES)}"
            )

        # turbo-mode recommendation only applies to engines that support it
        # (currently only UMA). Other engines return 'default' and ignore it.
        if getattr(calculator, "inference_mode", "default") != "turbo" and hasattr(
            calculator, "VALID_INFERENCE_MODES"
        ):
            self.log(
                "Consider using inference_mode='turbo' for better MD performance",
                level="warning",
            )

    def _stress_observables(self, atoms: Atoms) -> dict[str, Any]:
        """Return explicitly named 3D-bulk stress and pressure observables.

        Calculator/configurational stress excludes the kinetic ideal-gas term;
        total MD stress includes it via ASE. Scalar bulk pressure is only
        physically exposed for fully periodic systems. Molecules and partial-PBC
        systems return unavailable values instead of vacuum-dependent numbers.
        """
        unavailable = {
            "configurational_stress": None,
            "total_stress": None,
            "configurational_pressure_gpa": None,
            "total_pressure_gpa": None,
        }
        if (
            not self.write_stress
            or not getattr(self.calculator, "has_stress", False)
            or not bool(np.asarray(atoms.pbc, dtype=bool).all())
        ):
            return unavailable

        configurational = np.asarray(
            atoms.get_stress(voigt=True, include_ideal_gas=False), dtype=float
        )
        total = np.asarray(
            atoms.get_stress(voigt=True, include_ideal_gas=True), dtype=float
        )
        factor = MDOutcarWriter.EV_A3_TO_GPA
        return {
            "configurational_stress": configurational,
            "total_stress": total,
            "configurational_pressure_gpa": (
                -float(np.sum(configurational[:3])) / 3.0 * factor
            ),
            "total_pressure_gpa": -float(np.sum(total[:3])) / 3.0 * factor,
        }

    def _write_artifacts_manifest(
        self,
        *,
        status: str,
        frame_stats: MDFrameStats,
        error: dict[str, Any] | None = None,
    ) -> None:
        """Describe the versioned MD output contract without parsing files."""
        artifacts: dict[str, dict[str, Any]] = {}
        candidates = {
            "resolved_config": (
                self.output_dir / "resolved_config.json",
                "mlipx-resolved-config",
            ),
            "trajectory": (self.raw_dir / "trajectory.traj", "ase-traj"),
            "thermodynamics": (self.raw_dir / "md.csv", "csv"),
            "results": (self.raw_dir / "mlipx_results.json", "mlipx-json"),
            "xdatcar": (self.vasp_dir / "XDATCAR", "vasp-xdatcar"),
            "contcar": (self.vasp_dir / "CONTCAR", "vasp-poscar"),
            "outcar": (self.vasp_dir / "OUTCAR", "mlipx-vasp-like-outcar"),
        }
        for name, (path, file_format) in candidates.items():
            if path.exists():
                artifacts[name] = {
                    "path": path.relative_to(self.output_dir).as_posix(),
                    "format": file_format,
                    "bytes": path.stat().st_size,
                }

        timestep_fs = float(self.timestep / units.fs)
        try:
            mlipx_version = version("mlipx")
        except PackageNotFoundError:
            mlipx_version = "unknown"

        dependency_versions = {}
        for distribution in (
            "ase",
            "numpy",
            "torch",
            "fairchem-core",
            "mace-torch",
            "deepmd-kit",
            "tensorpotential",
        ):
            try:
                dependency_versions[distribution] = version(distribution)
            except PackageNotFoundError:
                dependency_versions[distribution] = "unknown"

        manifest = {
            "schema": "mlipx.md-artifacts/2",
            "status": status,
            "producer": {"name": "mlipx", "version": mlipx_version},
            "runtime": {"packages": dependency_versions},
            "model": self.calculator.info(),
            "layout": {
                "raw": "Lossless canonical trajectory and machine-readable data",
                "vasp": "VASP-syntax-compatible interoperability exports",
            },
            "trajectory": {
                "frames": frame_stats.count,
                "first_step": frame_stats.first_step,
                "last_step": frame_stats.last_step,
                "md_timestep_fs": timestep_fs,
                "frame_stride_steps": self.save_interval,
                "frame_interval_fs": timestep_fs * self.save_interval,
                # Legacy aliases remain readable by older consumers. The
                # explicitly named fields above are authoritative.
                "timestep_fs": timestep_fs,
                "save_interval_steps": self.save_interval,
                "saved_interval_fs": timestep_fs * self.save_interval,
                "positions_convention": "unwrapped",
                "positions": "unwrapped Cartesian in trajectory.traj; "
                "unwrapped direct in XDATCAR",
                "equilibration_steps": self.equilibration_steps,
                "production_steps": self.steps,
                "total_steps": self.total_steps,
                "production_start_step": self.equilibration_steps,
                "production_start_frame": frame_stats.production_start_frame,
            },
            "units": {
                "time": "fs",
                "length": "angstrom",
                "energy": "eV",
                "force": "eV/angstrom",
                "stress": "eV/angstrom^3",
                "temperature": "K",
            },
            "observables": {
                "configurational_stress": "ASE calculator stress; kinetic term excluded",
                "total_stress": "configurational stress plus ASE ideal-gas kinetic term",
                "configurational_pressure_gpa": "-trace(configurational_stress)/3",
                "total_pressure_gpa": "-trace(total_stress)/3; reported only for 3D PBC",
                "non_3d_pbc_policy": "stress and scalar bulk pressure unavailable",
            },
            "artifacts": artifacts,
        }
        if error is not None:
            manifest["error"] = error
        (self.output_dir / "artifacts.json").write_text(
            json.dumps(manifest, indent=2, default=str) + "\n",
            encoding="utf-8",
        )

    def _calculate_temperature(self, atoms: Atoms) -> float:
        """Instantaneous temperature for logging / trajectory.

        Delegates to ASE's canonical ``atoms.get_temperature()``, which uses
        ``get_number_of_degrees_of_freedom() = 3N - sum(constraint removed_dof)``.
        This is the *same* convention that ``MaxwellBoltzmannDistribution`` /
        ``force_temperature`` (used in ``_initialize_velocities``) and the NVT
        thermostat operate on, so the reported value matches the requested
        target temperature instead of being offset by an extra centre-of-mass
        correction. It also correctly accounts for every constraint that
        reports its removed DOF (FixAtoms, FixCom, FixBondLengths, ...) without
        double-counting the COM translation.

        A few constraints -- notably ``FixSymmetry`` -- do not report a
        removed-DOF count and raise ``NotImplementedError``. For those we fall
        back to the standard MD estimate ``3N - 3`` (the COM translation that
        ``Stationary`` zeroes at init) and warn; the previous code hard-coded
        an arbitrary ``ndof -= 3`` for FixSymmetry, which is wrong because the
        real count depends on the space group / number of atoms.

        Args:
            atoms: ASE Atoms object

        Returns:
            Temperature in Kelvin
        """
        try:
            # ASE's own temperature: 2*KE / (ndof*kB) with
            # ndof = 3N - sum(constraint.get_removed_dof()). Self-consistent
            # with velocity initialisation and the thermostat.
            return atoms.get_temperature()
        except NotImplementedError:
            # A constraint (e.g. FixSymmetry) cannot report its removed DOF.
            self.log(
                "Temperature DOF is approximate: a constraint does not "
                "report removed DOF; falling back to 3N-3 (COM removed).",
                level="warning",
            )
            ke = atoms.get_kinetic_energy()
            ndof = max(3 * len(atoms) - 3, 1)  # COM translation removed
            return 2 * ke / (ndof * units.kB)

    def _initialize_velocities(self, atoms: Atoms) -> None:
        """Initialize velocities at the requested MD temperature.

        Honours ``velocity_policy`` (plan section 5.3):

        * ``auto``       -- initialize only when the structure has no momenta;
                            otherwise preserve existing velocities (restart).
        * ``initialize`` -- always (re-)initialize a Maxwell-Boltzmann distribution.
        * ``preserve``   -- keep existing velocities; raise if there are none.

        When initializing, a seeded numpy Generator (``self.seed``) is forwarded
        to ASE's ``MaxwellBoltzmannDistribution`` so the run is reproducible when
        a seed is recorded (the resolver auto-generates one for MD).
        """
        # Presence, not magnitude, defines a restart.  An explicitly stored
        # all-zero momenta array is still a deliberate initial condition.
        has_momenta = atoms.has("momenta")
        policy = self.velocity_policy
        if policy == "preserve":
            if not has_momenta:
                raise ValueError(
                    "velocity_policy='preserve' requires existing velocities, but "
                    "the structure has none. Use velocity_policy='initialize' or "
                    "'auto'."
                )
            self.log("\nPreserving existing velocities (velocity_policy=preserve)")
            return
        if policy == "auto" and has_momenta:
            self.log("\nPreserving existing velocities (velocity_policy=auto)")
            return
        # policy == "initialize", or auto with no velocities.
        seed_msg = f" (seed={self.seed})" if self.seed is not None else ""
        self.log(
            f"\nInitializing Maxwell-Boltzmann distribution at "
            f"{self.temperature} K{seed_msg}"
        )
        MaxwellBoltzmannDistribution(
            atoms,
            temperature_K=self.temperature,
            force_temp=True,
            rng=self.rng,
        )
        Stationary(atoms, preserve_temperature=True)

    @staticmethod
    def _ensure_com_constraint(atoms: Atoms) -> None:
        """Remove centre-of-mass translation with an explicit ASE constraint.

        ASE's legacy ``Langevin(fixcm=True)`` does not strictly sample the
        canonical distribution.  An explicit ``FixCom`` constraint together
        with ``fixcm=False`` is the recommended formulation and also makes the
        removed three degrees of freedom visible to ``Atoms.get_temperature``.
        """
        if len(atoms) <= 1 or any(isinstance(c, FixCom) for c in atoms.constraints):
            return
        atoms.set_constraint([*atoms.constraints, FixCom()])

    def _build_dynamics(self, atoms: Atoms):
        """Build the ASE integrator without coupling it to calculator options."""
        if self.ensemble == "nve":
            self.log("Setting up NVE (Velocity Verlet)")
            return VelocityVerlet(atoms, timestep=self.timestep)

        if self.thermostat == "langevin":
            friction_fs = self.friction * units.fs
            self.log(
                "Setting up NVT Langevin dynamics "
                f"(friction={friction_fs:.4f} fs^-1)"
            )
            self.log(
                "Approx. velocity damping time: " f"{1.0 / friction_fs / 1000.0:.6g} ps"
            )
            return Langevin(
                atoms,
                timestep=self.timestep,
                temperature_K=self.temperature,
                friction=self.friction,
                rng=self.rng,
                fixcm=False,
            )

        if self.thermostat == "bussi":
            if np.isclose(atoms.get_kinetic_energy(), 0.0, rtol=0.0, atol=1e-12):
                raise ValueError(
                    "Bussi/CSVR requires non-zero initial kinetic energy. "
                    "Use a positive temperature with velocity_policy='initialize', "
                    "or provide non-zero velocities."
                )
            self.log(
                "Setting up NVT Bussi/CSVR dynamics "
                f"(coupling time={self.bussi_tau / units.fs:.6g} fs)"
            )
            return Bussi(
                atoms,
                timestep=self.timestep,
                temperature_K=self.temperature,
                taut=self.bussi_tau,
                rng=self.rng,
            )

        if self.thermostat == "nhc":
            self.log(
                "Setting up NVT Nose-Hoover-chain dynamics "
                f"(tdamp={self.nhc_tdamp / units.fs:.6g} fs, "
                f"chain={self.nhc_tchain}, substeps={self.nhc_tloop})"
            )
            return NoseHooverChainNVT(
                atoms,
                timestep=self.timestep,
                temperature_K=self.temperature,
                tdamp=self.nhc_tdamp,
                tchain=self.nhc_tchain,
                tloop=self.nhc_tloop,
            )

        raise ValueError(f"Unknown thermostat: {self.thermostat}")

    def _md_provenance(self) -> dict[str, Any]:
        """Return canonical MD settings, including only the active coupling."""
        provenance: dict[str, Any] = {
            "ensemble": self.ensemble.upper(),
            "thermostat": self.thermostat.upper() if self.ensemble == "nvt" else None,
            "temperature": self.temperature,
            "timestep": float(self.timestep / units.fs),
            "steps": self.steps,
            "equilibration_steps": self.equilibration_steps,
            "production_steps": self.steps,
            "total_steps": self.total_steps,
            "seed": self.seed,
            "velocity_policy": self.velocity_policy,
        }
        if self.ensemble == "nvt" and self.thermostat == "langevin":
            friction_fs = float(self.friction * units.fs)
            provenance.update(
                {
                    "friction_fs^-1": friction_fs,
                    "approx_velocity_damping_time_ps": 1.0 / friction_fs / 1000.0,
                }
            )
        elif self.ensemble == "nvt" and self.thermostat == "bussi":
            provenance["bussi_tau_fs"] = float(self.bussi_tau / units.fs)
        elif self.ensemble == "nvt" and self.thermostat == "nhc":
            provenance.update(
                {
                    "nhc_tdamp_fs": float(self.nhc_tdamp / units.fs),
                    "nhc_tchain": self.nhc_tchain,
                    "nhc_tloop": self.nhc_tloop,
                }
            )
        return provenance

    def _outcar_md_settings(self) -> dict[str, Any]:
        """Translate canonical provenance to human-readable OUTCAR labels."""
        provenance = self._md_provenance()
        settings: dict[str, Any] = {
            "Ensemble": provenance["ensemble"],
            "Thermostat": provenance["thermostat"],
            "Target temperature (K)": provenance["temperature"],
            "Time step (fs)": provenance["timestep"],
            "MD steps": provenance["steps"],
            "Equilibration steps": provenance["equilibration_steps"],
            "Production steps": provenance["production_steps"],
            "Total integrated steps": provenance["total_steps"],
            "Save interval (steps)": self.save_interval,
            "Random seed": provenance["seed"],
            "Velocity policy": provenance["velocity_policy"],
            "Pre-relaxed": self.pre_relax,
        }
        if "friction_fs^-1" in provenance:
            settings["Friction (1/fs)"] = provenance["friction_fs^-1"]
            settings["Approx. velocity damping time (ps)"] = provenance[
                "approx_velocity_damping_time_ps"
            ]
        elif "bussi_tau_fs" in provenance:
            settings["Bussi coupling time (fs)"] = provenance["bussi_tau_fs"]
        elif "nhc_tdamp_fs" in provenance:
            settings.update(
                {
                    "NHC damping time (fs)": provenance["nhc_tdamp_fs"],
                    "NHC chain length": provenance["nhc_tchain"],
                    "NHC thermostat substeps": provenance["nhc_tloop"],
                }
            )
        return settings

    def _pre_relax_structure(self, atoms: Atoms) -> Atoms:
        """Perform a short positions-only relaxation to reduce large forces.

        This does not optimize the cell and therefore does not in general
        remove cell stress or guarantee a local minimum.

        This can lower risky initial forces, but a short capped run does not
        ensure that the structure reaches a local minimum.

        Args:
            atoms: ASE Atoms object

        Returns:
            Relaxed Atoms object
        """
        self._emit_progress(
            "running",
            "Pre-relaxing structure...",
            step=0,
            total_steps=self.pre_relax_steps,
        )
        self.log("\n" + "=" * 60)
        self.log("PRE-RELAXATION PHASE")
        self.log("=" * 60)
        self.log("Reducing large initial atomic forces before MD...")
        self.log(f"Target fmax: {self.pre_relax_fmax} eV/Å")
        self.log(f"Max steps: {self.pre_relax_steps}")

        # Setup calculator
        atoms.calc = self._get_calculator()

        # Use FIRE optimizer for robust relaxation
        optimizer = FIRE(atoms, logfile=None)

        # Attach cancellation check to the optimizer
        def _check_cancel():
            if self._is_cancelled():
                self.log("\nCancellation requested during pre-relaxation")
                raise CancellationRequested("Pre-relaxation cancelled by user")

        optimizer.attach(_check_cancel, interval=1)

        # Track initial energy
        e_init = atoms.get_potential_energy()
        self.log(f"Initial energy: {e_init:.6f} eV")

        # Run optimization.  A backend/neighbor-list failure here is not a
        # recoverable "partially relaxed" result: continuing into MD would
        # merely hide the original error and can launch from an unsafe state.
        # Fail closed, matching safety.pre_relax_failure=abort.
        try:
            optimizer.run(fmax=self.pre_relax_fmax, steps=self.pre_relax_steps)

            e_final = atoms.get_potential_energy()
            delta_e = e_final - e_init

            self.log(f"Final energy: {e_final:.6f} eV")
            self.log(
                f"Energy change: {delta_e:.6f} eV ({delta_e / len(atoms):.6f} eV/atom)"
            )

            if optimizer.converged():
                self.log("✓ Pre-relaxation converged")
            else:
                forces = atoms.get_forces()
                final_fmax = float(np.max(np.linalg.norm(forces, axis=1)))
                raise RuntimeError(
                    "Pre-relaxation did not converge after "
                    f"{optimizer.nsteps} steps (final fmax={final_fmax:.6g} "
                    "eV/Angstrom); production MD was not started. Increase "
                    "pre_relax_steps, loosen pre_relax_fmax with scientific "
                    "justification, or disable pre_relax explicitly."
                )

        except CancellationRequested:
            self.log("Pre-relaxation cancelled by user")
            raise  # Re-raise to stop the entire MD run
        except RuntimeError as e:
            if "Pre-relaxation did not converge" in str(e):
                raise
            raise RuntimeError(
                "Pre-relaxation failed at step "
                f"{optimizer.nsteps}; MD was not started because the structure "
                "may be unsafe."
            ) from e
        except Exception as e:
            raise RuntimeError(
                "Pre-relaxation failed at step "
                f"{optimizer.nsteps}; MD was not started because the structure "
                "may be unsafe."
            ) from e

        self.log("=" * 60)

        return atoms

    def run(self, atoms: Atoms) -> dict[str, Any]:
        """Run MD simulation.

        Args:
            atoms: ASE Atoms object

        Returns:
            Dictionary with results including final temperature and trajectory
        """
        if (
            self.pre_relax
            and atoms.has("momenta")
            and self.velocity_policy in {"auto", "preserve"}
        ):
            raise ValueError(
                "Pre-relaxation changes positions and is incompatible with "
                f"velocity_policy={self.velocity_policy!r} on a structure that "
                "already contains momenta. For an exact phase-space restart use "
                "pre_relax=False; to start a new trajectory use "
                "velocity_policy='initialize'."
            )

        self.print_header("MOLECULAR DYNAMICS")
        self._emit_progress("loading_model", "Loading model and preparing structure...")

        # Print settings
        self.log(f"Ensemble:         {self.ensemble.upper()}")
        self.log(
            "Thermostat:       "
            + (self.thermostat.upper() if self.ensemble == "nvt" else "none")
        )
        self.log(f"Temperature:      {self.temperature} K")
        self.log(f"Time step:        {self.timestep / units.fs} fs")
        self.log(f"Equilibration:    {self.equilibration_steps} steps")
        self.log(f"Production:       {self.steps} steps")
        self.log(f"Total MD steps:   {self.total_steps}")
        self.log(f"Save interval:    {self.save_interval} steps (trajectory frames)")
        log_step_unit = "step" if self.LOG_INTERVAL_STEPS == 1 else "steps"
        self.log(
            f"Thermodynamic log interval: {self.LOG_INTERVAL_STEPS} "
            f"{log_step_unit} (buffered disk flush)"
        )
        self.log(f"Pre-relaxation:   {'Yes' if self.pre_relax else 'No'}")

        # Copy atoms to prevent mutating the caller's object. Pre-relaxation
        # (FIRE) modifies positions and velocity initialisation modifies
        # momenta on this same object. (_prepare_atoms now copies internally
        # as well, but this top-level copy is still required to shield the
        # pre-relax / velocity phases that run afterwards.)
        atoms = atoms.copy()

        # Prepare atoms
        atoms = self._prepare_atoms(atoms)

        # Setup calculator
        calc = self._get_calculator()
        atoms.calc = calc

        # NEW: Pre-relaxation step
        if self.pre_relax:
            atoms = self._pre_relax_structure(atoms)

        # Make COM removal explicit so temperature DOF and the Langevin
        # invariant distribution use the same constraint-aware convention.
        self._ensure_com_constraint(atoms)

        # Initialize a thermal velocity distribution for both NVT and NVE.
        self._initialize_velocities(atoms)

        # Setup integrator. Thermostat settings stay in MDRunner and are never
        # forwarded to the backend calculator.
        if self.ensemble == "nvt":
            self.log(
                "Note: under NVT the total energy E = PE + KE is NOT conserved "
                "(the thermostat exchanges heat); monitor T instead."
            )
        dyn = self._build_dynamics(atoms)

        # The live calculator and mutable Atoms object stay on the MD thread.
        # Saved frames are detached into CPU-only snapshots and submitted to a
        # small bounded queue; the writer thread owns every output file handle.
        traj_path = self.raw_dir / "trajectory.traj"
        md_csv_path = self.raw_dir / "md.csv"
        xdatcar_writer = XdatcarWriter() if self.write_xdatcar else None
        xdatcar_path = self.vasp_dir / "XDATCAR"
        md_outcar_writer = MDOutcarWriter() if self.write_outcar else None
        if md_outcar_writer is not None:
            md_outcar_writer.write_header(
                atoms,
                self.vasp_dir / "OUTCAR",
                task_name=self.calculator.task,
                metadata=self.calculator.info(),
                settings=self._outcar_md_settings(),
            )
        output_stream = AsyncMDOutputWriter(
            trajectory_path=traj_path,
            csv_path=md_csv_path,
            write_trajectory=self.write_trajectory,
            xdatcar_writer=xdatcar_writer,
            xdatcar_path=xdatcar_path,
            outcar_writer=md_outcar_writer,
        )
        # Run MD
        self.log("\nStarting MD simulation...")
        self._emit_progress(
            "running",
            f"Starting {self.ensemble.upper()} MD simulation...",
            step=0,
            total_steps=self.total_steps,
        )
        start_time = time.time()

        def print_progress():
            """Print progress and save trajectory."""
            # Surface writer failures promptly even on steps that do not save
            # a frame; continuing an MD run with broken output is not allowed.
            output_stream.raise_if_failed()
            # Check for cooperative cancellation first
            if self._is_cancelled():
                raise CancellationRequested("MD simulation cancelled by user")

            step = dyn.nsteps
            phase = "equilibration" if step < self.equilibration_steps else "production"

            # Calculate temperature with proper DOF handling
            temp = self._calculate_temperature(atoms)

            # Calculate energies
            pe = atoms.get_potential_energy()
            ke = atoms.get_kinetic_energy()
            total_e = pe + ke

            # Abort on NaN/inf energy so it never propagates or is written as
            # a "successful" result.
            self._check_finite(atoms, pe, context=f"MD step {step}")

            # Check forces for NaN/inf EVERY step. NaN forces corrupt positions
            # and velocities via the integrator and must be caught immediately.
            forces_chk = atoms.get_forces()
            self._check_finite(atoms, pe, forces=forces_chk, context=f"MD step {step}")

            force_magnitudes = np.linalg.norm(forces_chk, axis=1)
            atom_index = int(np.argmax(force_magnitudes))
            max_force = float(force_magnitudes[atom_index])
            force_abort = max_force > self.fmax_abort

            # Save regular trajectory frames and always checkpoint the unsafe
            # frame before raising a force-safety abort.
            if step % self.save_interval == 0 or force_abort:
                time_fs = float(step * self.timestep / units.fs)
                stress_data = self._stress_observables(atoms)
                configurational_stress = stress_data["configurational_stress"]
                total_stress = stress_data["total_stress"]
                volume = float(atoms.get_volume())
                frame_data = {
                    "step": int(step),
                    "time_fs": time_fs,
                    "phase": phase,
                    "energy": float(pe),
                    "kinetic_energy": float(ke),
                    "total_energy": float(total_e),
                    "temperature": float(temp),
                    "volume": volume,
                    "configurational_stress": (
                        np.array(configurational_stress, dtype=float, copy=True)
                        if configurational_stress is not None
                        else None
                    ),
                    "total_stress": (
                        np.array(total_stress, dtype=float, copy=True)
                        if total_stress is not None
                        else None
                    ),
                    "configurational_pressure_gpa": stress_data[
                        "configurational_pressure_gpa"
                    ],
                    "total_pressure_gpa": stress_data["total_pressure_gpa"],
                }
                snapshot = atoms.copy()
                snapshot.info["mlipx_step"] = int(step)
                snapshot.info["mlipx_time_fs"] = time_fs
                snapshot.info["mlipx_phase"] = phase
                single_point_results: dict[str, Any] = {
                    "energy": float(pe),
                    "forces": np.array(forces_chk, dtype=float, copy=True),
                }
                if configurational_stress is not None:
                    single_point_results["stress"] = np.array(
                        configurational_stress, dtype=float, copy=True
                    )
                snapshot.calc = SinglePointCalculator(snapshot, **single_point_results)
                output_stream.submit(
                    MDFrameSnapshot(
                        atoms=snapshot,
                        summary=frame_data,
                        outcar_forces=(
                            np.array(forces_chk, dtype=float, copy=True)
                            if self.write_forces
                            else None
                        ),
                    )
                )

            if force_abort:
                raise ForceSafetyAbort(
                    step=step,
                    max_force=max_force,
                    atom_index=atom_index,
                    threshold=self.fmax_abort,
                )
            # Retain one thermodynamic record per step, independently of the
            # trajectory save interval. The live run logger batches filesystem
            # flushes and does not emit these records as UI render events.
            if step % self.LOG_INTERVAL_STEPS == 0 or step == self.total_steps:
                self.log_buffered(
                    f"Step {step:6d}/{self.total_steps}: "
                    f"E = {total_e:12.4f} eV, T = {temp:6.1f} K "
                    f"[{phase}]"
                )

            # Keep UI/progress-callback updates throttled; emitting hundreds
            # of thousands of render events would not add useful information.
            if step % self.PROGRESS_INTERVAL_STEPS == 0 or step == self.total_steps:
                self._emit_progress(
                    "running",
                    f"Step {step:6d}/{self.total_steps}: "
                    f"E = {total_e:12.4f} eV, T = {temp:6.1f} K "
                    f"[{phase}]",
                    step=step,
                    total_steps=self.total_steps,
                    extra={
                        "energy": float(pe),
                        "temperature": float(temp),
                        "total_energy": float(total_e),
                    },
                )

        dyn.attach(print_progress, interval=1)

        def close_output_stream(primary_error: BaseException | None = None) -> None:
            """Drain requested frames, reporting output failure as fatal."""
            try:
                output_stream.close()
            except Exception as output_error:
                self.log(
                    f"\nMD output failed while draining buffered frames: "
                    f"{output_error}",
                    level="error",
                )
                if md_outcar_writer is not None:
                    md_outcar_writer.finalize(status="failed")
                self._write_artifacts_manifest(
                    status="failed",
                    frame_stats=output_stream.stats,
                    error={
                        "type": "output_error",
                        "message": str(output_error),
                    },
                )
                if primary_error is not None:
                    raise output_error from primary_error
                raise

        try:
            dyn.run(self.total_steps)
        except ForceSafetyAbort as exc:
            self.log(f"\nMD aborted by force safety threshold: {exc}", level="error")
            close_output_stream(exc)
            ContcarWriter().write_with_energy(
                atoms,
                self.vasp_dir / "CONTCAR",
                energy=float(atoms.get_potential_energy()),
                forces=atoms.get_forces(),
            )
            if md_outcar_writer is not None:
                md_outcar_writer.finalize(status="aborted")
            self._write_artifacts_manifest(
                status="aborted",
                frame_stats=output_stream.stats,
                error={
                    "type": "force_safety_abort",
                    "message": str(exc),
                    "step": exc.step,
                    "atom_index": exc.atom_index,
                    "max_force_eV_A": exc.max_force,
                    "fmax_abort_eV_A": exc.threshold,
                },
            )
            raise
        except CancellationRequested as exc:
            self.log("\n⚠️ MD simulation cancelled by user")
            close_output_stream(exc)
            if md_outcar_writer is not None:
                md_outcar_writer.finalize(status="cancelled")
            self._write_artifacts_manifest(
                status="cancelled", frame_stats=output_stream.stats
            )
            raise  # Re-raise to propagate cancellation
        except Exception as e:
            self.log(f"\n❌ MD simulation failed: {e}", level="error")
            close_output_stream(e)
            if md_outcar_writer is not None:
                md_outcar_writer.finalize(status="failed")
            self._write_artifacts_manifest(
                status="failed", frame_stats=output_stream.stats
            )
            raise

        close_output_stream()
        md_time = time.time() - start_time

        # Final temperature with proper calculation
        final_temp = self._calculate_temperature(atoms)

        # Final energy
        final_energy = atoms.get_potential_energy()
        final_forces = atoms.get_forces()
        final_stress_data = self._stress_observables(atoms)
        self._check_finite(
            atoms, final_energy, final_forces, context="MD final structure"
        )

        self.log(f"\nMD simulation completed in {md_time:.2f} s")
        self.log(f"Final temperature: {final_temp:.1f} K")
        self.log(f"Final energy: {final_energy:.6f} eV")

        # Build results
        results = {
            "energy": final_energy,
            "forces": final_forces,
            **final_stress_data,
            "temperature": final_temp,
            "md_steps": self.total_steps,
            "equilibration_steps": self.equilibration_steps,
            "production_steps": self.steps,
            "production_start_step": self.equilibration_steps,
            "production_start_frame": output_stream.stats.production_start_frame,
            "timestep_fs": float(self.timestep / units.fs),
            "save_interval": self.save_interval,
            "ensemble": self.ensemble.upper(),
            "thermostat": self.thermostat.upper() if self.ensemble == "nvt" else None,
            "target_temperature": self.temperature,
            "seed": self.seed,
            "velocity_policy": self.velocity_policy,
            "md_provenance": self._md_provenance(),
            "time": md_time,
            "trajectory": MDTrajectorySummary(md_csv_path, output_stream.stats.count),
            "trajectory_frame_count": output_stream.stats.count,
            "trajectory_path": str(traj_path) if self.write_trajectory else None,
            "md_csv_path": str(md_csv_path),
            "pre_relaxed": self.pre_relax,
        }

        # Write outputs (XDATCAR was already streamed during the run)
        self._emit_progress("writing_output", "Writing trajectory and output files...")
        if md_outcar_writer is not None:
            md_outcar_writer.finalize(
                status="completed",
                md_time_s=md_time,
                final_energy=float(final_energy),
                final_temperature=float(final_temp),
            )
        self._write_outputs(atoms, results)
        self._write_artifacts_manifest(
            status="completed", frame_stats=output_stream.stats
        )

        # Print summary
        self._write_summary(results, atoms)
        self._emit_progress(
            "done",
            f"MD complete. Final T = {final_temp:.1f} K",
            extra={"energy": float(final_energy), "temperature": float(final_temp)},
        )

        return results

    def _write_outputs(
        self,
        atoms: Atoms,
        results: dict[str, Any],
    ) -> None:
        """Write output files.

        XDATCAR, OUTCAR, trajectory.traj, and md.csv are streamed during the
        run (see :meth:`run`), so this writes the JSON summary and final
        VASP-syntax-compatible CONTCAR.

        Args:
            atoms: ASE Atoms object
            results: Results dictionary
        """
        metadata = self.calculator.info()
        metadata["pre_relaxed"] = self.pre_relax

        if self.write_outcar and (self.vasp_dir / "OUTCAR").exists():
            self.log(f"VASP-like OUTCAR written to: {self.vasp_dir / 'OUTCAR'}")

        # XDATCAR + md.csv were streamed frame-by-frame during the run;
        # log their paths if anything was written.
        if self.write_xdatcar and (self.vasp_dir / "XDATCAR").exists():
            self.log(f"XDATCAR written to: {self.vasp_dir / 'XDATCAR'}")
        if (self.raw_dir / "md.csv").exists():
            self.log(f"md.csv written to: {self.raw_dir / 'md.csv'}")
        if (self.raw_dir / "trajectory.traj").exists():
            self.log(
                f"Canonical trajectory written to: "
                f"{self.raw_dir / 'trajectory.traj'}"
            )

        # Write JSON
        if self.write_json:
            json_path = self.raw_dir / "mlipx_results.json"
            writer = JsonWriter()
            json_metadata = metadata.copy() if metadata else {}
            if self.job_name:
                json_metadata["job_name"] = self.job_name
            writer.write(
                atoms,
                results,
                json_path,
                mode="md",
                metadata=json_metadata,
            )
            self.log(f"JSON results written to: {json_path}")

        # Write final structure
        contcar_path = self.vasp_dir / "CONTCAR"
        export_atoms = atoms.copy()
        # FixCom is an integrator detail, not a VASP selective-dynamics flag.
        # Keeping it makes ASE emit a redundant "Selective dynamics" section
        # with T/T/T on every ion. Preserve any real user constraints.
        export_atoms.set_constraint(
            [
                constraint
                for constraint in export_atoms.constraints
                if not isinstance(constraint, FixCom)
            ]
        )
        ContcarWriter().write(
            export_atoms,
            contcar_path,
            comment="mlipx MD final structure",
            direct=True,
        )
        self.log(f"CONTCAR written to: {contcar_path}")
