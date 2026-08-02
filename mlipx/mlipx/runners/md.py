"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Modified for the mlipx project: multi-engine MLIP support (UMA/MACE/DPA/GRACE).
Molecular dynamics runner.

Runs MD simulations using ASE's integrators:
- NVT ensemble (Langevin dynamics)
- NVE ensemble (Velocity Verlet)

Outputs trajectories in multiple formats.
"""

from __future__ import annotations

import csv
import json
import threading
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from ase import units
from ase.constraints import FixCom
from ase.io.trajectory import TrajectoryWriter as AseTrajectoryWriter
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.verlet import VelocityVerlet
from ase.optimize import FIRE

from mlipx.config.defaults import BUILTIN_DEFAULTS
from mlipx.protocols import CancellationRequested
from mlipx.runners.base import BaseRunner
from mlipx.writers.contcar import ContcarWriter
from mlipx.writers.outcar import MDOutcarWriter
from mlipx.writers.xdatcar import XdatcarWriter
from mlipx.writers.json_writer import JsonWriter

if TYPE_CHECKING:
    from typing import Any

    from ase import Atoms
    from mlipx.protocols import ProgressCallback


class MDRunner(BaseRunner):
    """Run molecular dynamics simulations.

    Supports NVT (Langevin) and NVE (Velocity Verlet) ensembles.
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

    def __init__(
        self,
        calculator,
        ensemble: str = "NVT",
        temperature: float = 300.0,
        timestep: float = 1.0,
        steps: int = 1000,
        friction: float = 0.001,
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
        equil_steps: int = 0,
        pre_relax_mode: str = "none",
        # Explosion-guard threshold for finite (but suspiciously large) forces.
        # Sourced from the safety defaults so it is not a magic literal
        # (plan section 5.7). Warnings never abort the run.
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
            steps: Number of MD steps
            friction: Friction coefficient for NVT (1/fs)
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
        self.steps = steps
        self.friction = friction / units.fs  # Convert to ASE units
        self.save_interval = save_interval
        self.write_outcar = write_outcar
        self.write_forces = write_forces
        self.write_stress = write_stress
        self.write_xdatcar = write_xdatcar
        self.write_trajectory = write_trajectory
        self.write_json = write_json

        # MD output contract.  ``raw`` is the lossless internal source of
        # truth, ``vasp`` is a syntax-compatible interoperability view, and
        # ``analysis`` is reserved for derived data.  Keeping these separate
        # prevents a formatted export from silently becoming the analysis
        # input in future workflows.
        self.raw_dir = self.output_dir / "raw"
        self.vasp_dir = self.output_dir / "vasp"
        self.analysis_dir = self.output_dir / "analysis"
        for directory in (self.raw_dir, self.vasp_dir, self.analysis_dir):
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
        self.equil_steps = int(equil_steps)
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
        if self.save_interval <= 0:
            raise ValueError("save_interval must be > 0")
        if self.ensemble == "nvt" and self.friction <= 0:
            raise ValueError(
                "friction must be > 0 fs^-1 for NVT Langevin dynamics"
            )
        # pre_relax_mode / equil_steps are declared in the schema as Phase-3
        # vocabulary. They must NOT be silently ignored (plan section 5.2): a
        # non-default value raises an explicit, loud "not yet implemented"
        # error instead of a silent no-op.
        if self.pre_relax_mode != "none":
            raise NotImplementedError(
                f"pre_relax_mode={self.pre_relax_mode!r} is not yet implemented "
                "(Phase 3). Use the legacy pre_relax=True/False (positions-only) "
                "for now."
            )
        if self.equil_steps > 0:
            raise NotImplementedError(
                f"equil_steps={self.equil_steps} is not yet implemented "
                "(Phase 3). Run equilibration as a separate short MD and restart "
                "from it."
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

    def _stress_for_saved_frame(self, atoms: Atoms) -> np.ndarray | None:
        """Return ASE-Voigt stress when requested and supported."""
        if not self.write_stress or not getattr(self.calculator, "has_stress", False):
            return None
        return np.asarray(atoms.get_stress(voigt=True), dtype=float)

    def _write_artifacts_manifest(
        self,
        *,
        status: str,
        frames: list[dict[str, Any]],
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
        for distribution in ("ase", "numpy"):
            try:
                dependency_versions[distribution] = version(distribution)
            except PackageNotFoundError:
                dependency_versions[distribution] = "unknown"

        manifest = {
            "schema": "mlipx.md-artifacts/1",
            "status": status,
            "producer": {"name": "mlipx", "version": mlipx_version},
            "runtime": {"packages": dependency_versions},
            "model": self.calculator.info(),
            "layout": {
                "raw": "Lossless canonical trajectory and machine-readable data",
                "vasp": "VASP-syntax-compatible interoperability exports",
                "analysis": "Reserved for derived post-processing results",
            },
            "trajectory": {
                "frames": len(frames),
                "first_step": frames[0]["step"] if frames else None,
                "last_step": frames[-1]["step"] if frames else None,
                "timestep_fs": timestep_fs,
                "save_interval_steps": self.save_interval,
                "saved_interval_fs": timestep_fs * self.save_interval,
                "positions": "unwrapped Cartesian in trajectory.traj; "
                "unwrapped direct in XDATCAR",
            },
            "units": {
                "time": "fs",
                "length": "angstrom",
                "energy": "eV",
                "force": "eV/angstrom",
                "stress": "eV/angstrom^3",
                "temperature": "K",
            },
            "artifacts": artifacts,
        }
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
                self.log("! Pre-relaxation did not fully converge, but continuing...")

        except CancellationRequested:
            self.log("Pre-relaxation cancelled by user")
            raise  # Re-raise to stop the entire MD run
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
        self.log(f"Temperature:      {self.temperature} K")
        self.log(f"Time step:        {self.timestep / units.fs} fs")
        self.log(f"Steps:            {self.steps}")
        self.log(f"Save interval:    {self.save_interval}")
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

        # Setup integrator
        if self.ensemble == "nvt":
            self.log(
                f"Setting up Langevin dynamics (friction={self.friction * units.fs:.4f} fs^-1)"
            )
            self.log(
                "Note: under NVT the total energy E = PE + KE is NOT conserved "
                "(the thermostat exchanges heat); monitor T instead."
            )
            dyn = Langevin(
                atoms,
                timestep=self.timestep,
                temperature_K=self.temperature,
                friction=self.friction,
                rng=self.rng,
                fixcm=False,
            )
        else:  # nve
            self.log("Setting up NVE (Velocity Verlet)")
            dyn = VelocityVerlet(atoms, timestep=self.timestep)

        # Streaming trajectory outputs (plan section 5.5):
        #   * full frames  -> raw/trajectory.traj (lossless ASE source)
        #   * full frames  -> vasp/XDATCAR (VASP syntax, unwrapped direct)
        #   * rich text    -> vasp/OUTCAR (documented VASP-like subset)
        #   * scalars      -> raw/md.csv (streamed row-by-row)
        # Only the scalar summary is kept in memory, so RAM stays flat
        # regardless of run length (no per-frame atoms.copy() list).
        trajectory_summary: list[dict[str, Any]] = []
        traj_path = self.raw_dir / "trajectory.traj"

        # File writers
        traj_writer = None
        if self.write_trajectory:
            traj_writer = AseTrajectoryWriter(traj_path, mode="w")

        md_csv_file = None
        md_csv_writer = None
        xdatcar_writer = XdatcarWriter() if self.write_xdatcar else None
        xdatcar_header_written = False
        xdatcar_path = self.vasp_dir / "XDATCAR"
        md_outcar_writer = MDOutcarWriter() if self.write_outcar else None
        if md_outcar_writer is not None:
            md_outcar_writer.write_header(
                atoms,
                self.vasp_dir / "OUTCAR",
                task_name=self.calculator.task,
                metadata=self.calculator.info(),
                settings={
                    "Ensemble": self.ensemble.upper(),
                    "Target temperature (K)": self.temperature,
                    "Time step (fs)": self.timestep / units.fs,
                    "MD steps": self.steps,
                    "Save interval (steps)": self.save_interval,
                    "Friction (1/fs)": self.friction * units.fs
                    if self.ensemble == "nvt"
                    else "n/a",
                    "Random seed": self.seed,
                    "Pre-relaxed": self.pre_relax,
                },
            )
        # Run MD
        self.log("\nStarting MD simulation...")
        self._emit_progress(
            "running",
            f"Starting {self.ensemble.upper()} MD simulation...",
            step=0,
            total_steps=self.steps,
        )
        start_time = time.time()

        def print_progress():
            """Print progress and save trajectory."""
            nonlocal md_csv_file, md_csv_writer, xdatcar_header_written
            # Check for cooperative cancellation first
            if self._is_cancelled():
                raise CancellationRequested("MD simulation cancelled by user")

            step = dyn.nsteps

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

            # Explosion guard: warn on suspiciously large (but finite) forces.
            # _check_finite above already aborts on NaN/inf; this catches a run
            # that is diverging while still producing finite numbers. The
            # threshold comes from the safety default (fmax_abort, 20 eV/Å by
            # default) -- configurable via the ``fmax_abort`` parameter (plan
            # section 5.7) -- and never aborts, only warns. max_force is
            # computed every step because forces_chk is already available
            # (free), but the warning is only logged every 100 steps to avoid
            # flooding the log.
            max_force = float(np.max(np.linalg.norm(forces_chk, axis=1)))
            if max_force > self.fmax_abort and (
                step > 0 and (step % 100 == 0 or step == self.steps)
            ):
                self.log(
                    f"⚠️ WARNING: Large forces detected ({max_force:.1f} eV/Å)",
                    level="warning",
                )
                self.log("   Structure may be unstable. Consider:")
                self.log("   - Lowering temperature")
                self.log("   - Relaxing the positions more carefully before MD")
                self.log("   - Checking initial structure")

            # Save trajectory frame (streamed to disk; only scalars in RAM)
            if step % self.save_interval == 0:
                time_fs = float(step * self.timestep / units.fs)
                stress = self._stress_for_saved_frame(atoms)
                volume = float(atoms.get_volume())
                pressure_gpa = None
                if stress is not None:
                    pressure_gpa = (
                        -float(np.sum(stress[:3]))
                        / 3.0
                        * MDOutcarWriter.EV_A3_TO_GPA
                    )
                frame_data = {
                    "step": step,
                    "time_fs": time_fs,
                    "energy": pe,
                    "kinetic_energy": ke,
                    "total_energy": total_e,
                    "temperature": temp,
                    "volume": volume,
                    "stress": stress,
                    "pressure_gpa": pressure_gpa,
                }
                trajectory_summary.append(frame_data)

                # md.csv: create lazily on the first saved frame.
                if md_csv_writer is None:
                    md_csv_file = open(
                        self.raw_dir / "md.csv",
                        "w",
                        newline="",
                        encoding="utf-8",
                    )
                    md_csv_writer = csv.writer(md_csv_file)
                    md_csv_writer.writerow(
                        [
                            "step",
                            "time_fs",
                            "potential_energy_eV",
                            "kinetic_energy_eV",
                            "total_energy_eV",
                            "temperature_K",
                            "volume_A3",
                            "stress_xx_eV_A3",
                            "stress_yy_eV_A3",
                            "stress_zz_eV_A3",
                            "stress_yz_eV_A3",
                            "stress_xz_eV_A3",
                            "stress_xy_eV_A3",
                            "pressure_GPa",
                        ]
                    )
                stress_values = stress.tolist() if stress is not None else [""] * 6
                md_csv_writer.writerow(
                    [
                        step,
                        time_fs,
                        pe,
                        ke,
                        total_e,
                        temp,
                        volume,
                        *stress_values,
                        pressure_gpa if pressure_gpa is not None else "",
                    ]
                )
                md_csv_file.flush()

                # XDATCAR: header from the first frame, then stream frames.
                if xdatcar_writer is not None:
                    if not xdatcar_header_written:
                        xdatcar_writer.write_header(atoms, xdatcar_path)
                        xdatcar_header_written = True
                    xdatcar_writer.append_frame(xdatcar_path, atoms, step=step)

                if md_outcar_writer is not None:
                    md_outcar_writer.append_frame(
                        atoms,
                        step=step,
                        time_fs=time_fs,
                        potential_energy=float(pe),
                        kinetic_energy=float(ke),
                        total_energy=float(total_e),
                        temperature=float(temp),
                        forces=forces_chk if self.write_forces else None,
                        stress=stress,
                    )

                if traj_writer is not None:
                    traj_writer.write(atoms)
            # Print progress every 100 steps
            if step % 100 == 0 or step == self.steps:
                self._emit_progress(
                    "running",
                    f"Step {step:6d}/{self.steps}: E = {total_e:12.4f} eV, T = {temp:6.1f} K",
                    step=step,
                    total_steps=self.steps,
                    extra={
                        "energy": float(pe),
                        "temperature": float(temp),
                        "total_energy": float(total_e),
                    },
                )
                self.log(
                    f"Step {step:6d}/{self.steps}: "
                    f"E = {total_e:12.4f} eV, T = {temp:6.1f} K"
                )

        dyn.attach(print_progress, interval=1)

        try:
            dyn.run(self.steps)
        except CancellationRequested:
            self.log("\n⚠️ MD simulation cancelled by user")
            # Save what we have before returning
            if traj_writer is not None:
                traj_writer.close()
            if md_csv_file is not None:
                md_csv_file.close()
            if md_outcar_writer is not None:
                md_outcar_writer.finalize(status="cancelled")
            self._write_artifacts_manifest(
                status="cancelled", frames=trajectory_summary
            )
            raise  # Re-raise to propagate cancellation
        except Exception as e:
            self.log(f"\n❌ MD simulation failed: {e}", level="error")
            # Save what we have
            if traj_writer is not None:
                traj_writer.close()
            if md_csv_file is not None:
                md_csv_file.close()
            if md_outcar_writer is not None:
                md_outcar_writer.finalize(status="failed")
            self._write_artifacts_manifest(status="failed", frames=trajectory_summary)
            raise

        md_time = time.time() - start_time

        # Close trajectory writer + csv
        if traj_writer is not None:
            traj_writer.close()
        if md_csv_file is not None:
            md_csv_file.close()

        # Final temperature with proper calculation
        final_temp = self._calculate_temperature(atoms)

        # Final energy
        final_energy = atoms.get_potential_energy()
        final_forces = atoms.get_forces()
        final_stress = self._stress_for_saved_frame(atoms)
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
            "stress": final_stress,
            "temperature": final_temp,
            "md_steps": self.steps,
            "timestep_fs": float(self.timestep / units.fs),
            "save_interval": self.save_interval,
            "ensemble": self.ensemble.upper(),
            "time": md_time,
            "trajectory": trajectory_summary,
            "trajectory_path": str(traj_path) if self.write_trajectory else None,
            "md_csv_path": str(self.raw_dir / "md.csv")
            if md_csv_writer is not None
            else None,
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
        self._write_artifacts_manifest(status="completed", frames=trajectory_summary)

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
