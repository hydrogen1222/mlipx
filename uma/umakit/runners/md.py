"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Molecular dynamics runner.

Runs MD simulations using ASE's integrators:
- NVT ensemble (Langevin dynamics)
- NVE ensemble (Velocity Verlet)

Outputs trajectories in multiple formats.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from ase import units
from ase.constraints import FixAtoms, FixSymmetry
from ase.io.trajectory import TrajectoryWriter as AseTrajectoryWriter
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.optimize import FIRE

from umakit.runners.base import BaseRunner
from umakit.writers.json_writer import JsonWriter
from umakit.writers.outcar import OutcarWriter
from umakit.writers.trajectory import TrajectoryWriter
from umakit.writers.xdatcar import XdatcarWriter

if TYPE_CHECKING:
    from typing import Any

    from ase import Atoms
    from umakit.protocols import ProgressCallback


class MDRunner(BaseRunner):
    """Run molecular dynamics simulations.

    Supports NVT (Langevin) and NVE (Velocity Verlet) ensembles.
    Includes pre-relaxation step to eliminate internal stress before MD.

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
        write_xdatcar: bool = True,
        write_trajectory: bool = True,
        write_json: bool = True,
        verbose: bool = True,
        job_name: str | None = None,
        # NEW: Pre-relaxation options
        pre_relax: bool = True,
        pre_relax_steps: int = 50,
        pre_relax_fmax: float = 0.1,
        log_fn: Any | None = None,
        progress_callback: ProgressCallback | None = None,
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
            calculator, output_dir, verbose, job_name, log_fn, progress_callback
        )
        self.ensemble = ensemble.lower()
        self.temperature = temperature
        self.timestep = timestep * units.fs  # Convert to ASE units
        self.steps = steps
        self.friction = friction / units.fs  # Convert to ASE units
        self.save_interval = save_interval
        self.write_outcar = write_outcar
        self.write_xdatcar = write_xdatcar
        self.write_trajectory = write_trajectory
        self.write_json = write_json

        # NEW: Pre-relaxation settings
        self.pre_relax = pre_relax
        self.pre_relax_steps = pre_relax_steps
        self.pre_relax_fmax = pre_relax_fmax

        # Validate ensemble
        if self.ensemble not in self.VALID_ENSEMBLES:
            raise ValueError(
                f"Unknown ensemble: {ensemble}. "
                f"Use one of: {', '.join(self.VALID_ENSEMBLES)}"
            )

        # Use turbo mode recommendation for MD
        if calculator.inference_mode != "turbo":
            self.log(
                "Consider using inference_mode='turbo' for better MD performance",
                level="warning",
            )

    def _calculate_temperature(self, atoms: Atoms) -> float:
        """Calculate temperature accounting for constraints.

        Uses proper degrees of freedom calculation considering constraints.

        Args:
            atoms: ASE Atoms object

        Returns:
            Temperature in Kelvin
        """
        ke = atoms.get_kinetic_energy()
        natoms = len(atoms)

        # Calculate degrees of freedom
        # Start with 3N - 3 (remove center of mass translation)
        ndof = 3 * natoms - 3

        # Account for constraints
        for constraint in atoms.constraints:
            if isinstance(constraint, FixAtoms):
                # Fixed atoms have 0 DOF
                # constraint.index can be a boolean mask or an array of indices
                index = constraint.index
                num_fixed = np.sum(index) if index.dtype == bool else len(index)
                ndof -= 3 * num_fixed
            elif isinstance(constraint, FixSymmetry):
                # Symmetry constraints reduce DOF
                # This is complex; for now use conservative estimate
                ndof -= 3

        # Ensure at least 1 DOF to avoid division by zero
        ndof = max(ndof, 1)

        # T = 2E_k / (N_dof * k_B)
        return 2 * ke / (ndof * units.kB)

    def _pre_relax_structure(self, atoms: Atoms) -> Atoms:
        """Perform quick relaxation to eliminate internal stress.

        This prevents atom explosion during MD by ensuring the
        structure is at a local minimum before adding thermal energy.

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
        self.log("Eliminating internal stress before MD...")
        self.log(f"Target fmax: {self.pre_relax_fmax} eV/Å")
        self.log(f"Max steps: {self.pre_relax_steps}")

        # Setup calculator
        atoms.calc = self._get_calculator()

        # Use FIRE optimizer for robust relaxation
        optimizer = FIRE(atoms, logfile=None)

        # Track initial energy
        e_init = atoms.get_potential_energy()
        self.log(f"Initial energy: {e_init:.6f} eV")

        # Run optimization
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

        except Exception as e:
            self.log(f"! Pre-relaxation warning: {e}", level="warning")
            self.log("! Continuing with original structure...")

        self.log("=" * 60)

        return atoms

    def run(self, atoms: Atoms) -> dict[str, Any]:
        """Run MD simulation.

        Args:
            atoms: ASE Atoms object

        Returns:
            Dictionary with results including final temperature and trajectory
        """
        self.print_header("MOLECULAR DYNAMICS")
        self._emit_progress("loading_model", "Loading model and preparing structure...")

        # Print settings
        self.log(f"Ensemble:         {self.ensemble.upper()}")
        self.log(f"Temperature:      {self.temperature} K")
        self.log(f"Time step:        {self.timestep / units.fs} fs")
        self.log(f"Steps:            {self.steps}")
        self.log(f"Save interval:    {self.save_interval}")
        self.log(f"Pre-relaxation:   {'Yes' if self.pre_relax else 'No'}")

        # Prepare atoms
        atoms = self._prepare_atoms(atoms)

        # Setup calculator
        calc = self._get_calculator()
        atoms.calc = calc

        # NEW: Pre-relaxation step
        if self.pre_relax:
            atoms = self._pre_relax_structure(atoms)

        # Initialize velocities for NVE
        if self.ensemble == "nve":
            self.log(
                f"\nInitializing Maxwell-Boltzmann distribution at {self.temperature} K"
            )
            MaxwellBoltzmannDistribution(atoms, temperature_K=self.temperature)

        # Setup integrator
        if self.ensemble == "nvt":
            self.log(
                f"Setting up Langevin dynamics (friction={self.friction * units.fs:.4f} fs^-1)"
            )
            dyn = Langevin(
                atoms,
                timestep=self.timestep,
                temperature_K=self.temperature,
                friction=self.friction,
            )
        else:  # nve
            self.log("Setting up NVE (Velocity Verlet)")
            dyn = VelocityVerlet(atoms, timestep=self.timestep)

        # Setup trajectory tracking
        trajectory = []

        # Setup file writers
        traj_writer = None
        if self.write_trajectory:
            traj_path = self.output_dir / "trajectory.traj"
            traj_writer = AseTrajectoryWriter(traj_path, mode="w")

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
            step = dyn.nsteps

            # Calculate temperature with proper DOF handling
            temp = self._calculate_temperature(atoms)

            # Calculate energies
            pe = atoms.get_potential_energy()
            ke = atoms.get_kinetic_energy()
            total_e = pe + ke

            # Check for explosion (simple check)
            if step > 0 and step % 100 == 0:
                max_force = np.max(np.linalg.norm(atoms.get_forces(), axis=1))
                if max_force > 50:  # eV/Å - suspiciously high
                    self.log(f"⚠️ WARNING: Large forces detected ({max_force:.1f} eV/Å)")
                    self.log("   Structure may be unstable. Consider:")
                    self.log("   - Lowering temperature")
                    self.log("   - Increasing pre-relaxation steps")
                    self.log("   - Checking initial structure")

            # Save trajectory frame
            if step % self.save_interval == 0:
                frame_data = {
                    "step": step,
                    "atoms": atoms.copy(),
                    "energy": pe,
                    "kinetic_energy": ke,
                    "total_energy": total_e,
                    "temperature": temp,
                }
                trajectory.append(frame_data)

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
        except Exception as e:
            self.log(f"\n❌ MD simulation failed: {e}", level="error")
            # Save what we have
            if traj_writer is not None:
                traj_writer.close()
            raise

        md_time = time.time() - start_time

        # Close trajectory writer
        if traj_writer is not None:
            traj_writer.close()

        # Final temperature with proper calculation
        final_temp = self._calculate_temperature(atoms)

        # Final energy
        final_energy = atoms.get_potential_energy()
        final_forces = atoms.get_forces()

        self.log(f"\nMD simulation completed in {md_time:.2f} s")
        self.log(f"Final temperature: {final_temp:.1f} K")
        self.log(f"Final energy: {final_energy:.6f} eV")

        # Build results
        results = {
            "energy": final_energy,
            "forces": final_forces,
            "temperature": final_temp,
            "md_steps": self.steps,
            "ensemble": self.ensemble.upper(),
            "time": md_time,
            "trajectory": trajectory,
            "pre_relaxed": self.pre_relax,
        }

        # Write outputs
        self._emit_progress("writing_output", "Writing trajectory and output files...")
        self._write_outputs(atoms, results, trajectory)

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
        trajectory: list,
    ) -> None:
        """Write output files.

        Args:
            atoms: ASE Atoms object
            results: Results dictionary
            trajectory: MD trajectory
        """
        metadata = self.calculator.info()
        metadata["pre_relaxed"] = self.pre_relax

        # Write OUTCAR
        if self.write_outcar:
            outcar_path = self.output_dir / "OUTCAR"
            writer = OutcarWriter()
            writer.write(
                atoms,
                results,
                outcar_path,
                mode="md",
                task_name=self.calculator.task,
                metadata=metadata,
            )
            self.log(f"OUTCAR written to: {outcar_path}")

        # Write XDATCAR
        if self.write_xdatcar and trajectory:
            xdatcar_path = self.output_dir / "XDATCAR"
            writer = XdatcarWriter()
            writer.write_from_md(
                xdatcar_path,
                trajectory,
                step_interval=self.save_interval,
            )
            self.log(f"XDATCAR written to: {xdatcar_path}")

        # Write JSON
        if self.write_json:
            json_path = self.output_dir / "uma_results.json"
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
        contcar_path = self.output_dir / "CONTCAR"
        writer = TrajectoryWriter()
        writer.write_single(atoms, contcar_path, format="vasp")
        self.log(f"CONTCAR written to: {contcar_path}")
