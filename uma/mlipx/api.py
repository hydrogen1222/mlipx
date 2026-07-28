"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Modified for the mlipx project: multi-engine MLIP support (UMA/MACE/DPA/GRACE).
Programmatic API for mlipx.

Provides high-level functions for running calculations from Python scripts.
This module is designed for external scripts that need to integrate UMA
calculations into complex workflows.

Example:
    >>> from mlipx.api import run_single_point, calculate_energy
    >>>
    >>> # Run a single point calculation
    >>> results = run_single_point(
    ...     structure="structure.cif",
    ...     model_path="uma-s-1.pt",
    ...     task="omat",
    ...     job_name="my_calculation"
    ... )
    >>> print(f"Energy: {results['energy']:.4f} eV")
    >>>
    >>> # Just get the energy
    >>> energy = calculate_energy("structure.cif", "uma-s-1.pt", task="omat")
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ase import Atoms
from ase.io import read

from mlipx.engine import CalculationEngine, EngineConfig

if TYPE_CHECKING:
    from typing import Any


def _load_structure(structure: Atoms | str | Path) -> Atoms:
    """Load structure from various input types.

    Args:
        structure: ASE Atoms object or path to structure file

    Returns:
        ASE Atoms object

    Raises:
        ValueError: If structure cannot be loaded
    """
    if isinstance(structure, Atoms):
        return structure

    structure_path = Path(structure)
    if not structure_path.exists():
        raise ValueError(f"Structure file not found: {structure_path}")

    try:
        return read(structure_path)
    except Exception as e:
        raise ValueError(f"Error reading structure: {e}") from e


def run_single_point(
    structure: Atoms | str | Path,
    model_path: str,
    model_type: str = "uma",
    task: str = "omat",
    device: str = "cpu",
    job_name: str | None = None,
    output_dir: str = "./results",
    verbose: bool = True,
    **kwargs,
) -> dict[str, Any]:
    """Run single point calculation.

    Calculates energy, forces, and stress for a given structure.

    Args:
        structure: ASE Atoms object or path to structure file
        model_path: Path to model checkpoint
        model_type: MLIP engine (uma, mace, dpa, grace) (omat, omol, oc20, oc25, odac, omc)
        device: Device for calculation (cpu or cuda)
        job_name: Optional job name for organizing results
        output_dir: Base directory for output files
        verbose: Whether to print progress messages
        **kwargs: Additional arguments passed to SinglePointRunner

    Returns:
        Dictionary with results (energy, forces, stress, time)

    Example:
        >>> results = run_single_point(
        ...     structure="structure.cif",
        ...     model_path="uma-s-1.pt",
        ...     task="omat",
        ...     job_name="sp_calc"
        ... )
        >>> print(f"Energy: {results['energy']:.4f} eV")
    """
    atoms = _load_structure(structure)
    if verbose:
        print(f"System: {atoms.get_chemical_formula()}")
        print(f"Atoms: {len(atoms)}")
        print(f"Loading model: {model_path}")

    config = EngineConfig(
        calc_type="sp",
        model_path=Path(model_path),
        model_type=model_type,
        task=task,
        device=device,
        output_dir=Path(output_dir),
        job_name=job_name,
    )
    engine = CalculationEngine.from_config(config)
    return engine.run(atoms)


def run_optimization(
    structure: Atoms | str | Path,
    model_path: str,
    model_type: str = "uma",
    task: str = "omat",
    device: str = "cpu",
    job_name: str | None = None,
    output_dir: str = "./results",
    fmax: float = 0.05,
    max_steps: int = 500,
    optimizer: str = "FIRE",
    cell_opt: bool = False,
    fix_symmetry: bool = False,
    verbose: bool = True,
    **kwargs,
) -> dict[str, Any]:
    """Run geometry optimization.

    Optimizes atomic positions and optionally cell parameters
    until forces converge below threshold.

    Args:
        structure: ASE Atoms object or path to structure file
        model_path: Path to model checkpoint
        model_type: MLIP engine (uma, mace, dpa, grace) (omat, omol, oc20, oc25, odac, omc)
        device: Device for calculation (cpu or cuda)
        job_name: Optional job name for organizing results
        output_dir: Base directory for output files
        fmax: Force convergence threshold in eV/Å
        max_steps: Maximum optimization steps
        optimizer: Optimization algorithm (FIRE, BFGS, LBFGS)
        cell_opt: Whether to optimize cell parameters
        fix_symmetry: Whether to preserve symmetry
        verbose: Whether to print progress messages
        **kwargs: Additional arguments passed to OptimizationRunner

    Returns:
        Dictionary with results (energy, converged, nsteps, etc.)

    Example:
        >>> results = run_optimization(
        ...     structure="structure.cif",
        ...     model_path="uma-s-1.pt",
        ...     fmax=0.02,
        ...     cell_opt=True,
        ...     job_name="opt_calc"
        ... )
        >>> print(f"Converged: {results['converged']}")
        >>> print(f"Final energy: {results['energy']:.4f} eV")
    """
    atoms = _load_structure(structure)
    if verbose:
        print(f"System: {atoms.get_chemical_formula()}")
        print(f"Atoms: {len(atoms)}")
        print(f"Loading model: {model_path}")

    config = EngineConfig(
        calc_type="opt",
        model_path=Path(model_path),
        model_type=model_type,
        task=task,
        device=device,
        output_dir=Path(output_dir),
        job_name=job_name,
        options={
            "fmax": fmax,
            "max_steps": max_steps,
            "optimizer": optimizer,
            "cell_opt": cell_opt,
            "fix_symmetry": fix_symmetry,
        },
    )
    engine = CalculationEngine.from_config(config)
    return engine.run(atoms)


def run_md(
    structure: Atoms | str | Path,
    model_path: str,
    model_type: str = "uma",
    task: str = "omat",
    device: str = "cuda",
    job_name: str | None = None,
    output_dir: str = "./results",
    ensemble: str = "NVT",
    temperature: float = 300.0,
    timestep: float = 1.0,
    steps: int = 1000,
    friction: float = 0.001,
    save_interval: int = 10,
    pre_relax: bool = True,
    verbose: bool = True,
    **kwargs,
) -> dict[str, Any]:
    """Run molecular dynamics simulation.

    Runs MD simulation using NVT (Langevin) or NVE (Velocity Verlet) ensemble.
    Includes pre-relaxation step to eliminate internal stress.

    Args:
        structure: ASE Atoms object or path to structure file
        model_path: Path to model checkpoint
        model_type: MLIP engine (uma, mace, dpa, grace) (omat, omol, oc20, oc25, odac, omc)
        device: Device for calculation (cpu or cuda)
        job_name: Optional job name for organizing results
        output_dir: Base directory for output files
        ensemble: MD ensemble (NVT or NVE)
        temperature: Temperature in Kelvin
        timestep: Time step in femtoseconds
        steps: Number of MD steps
        friction: Friction coefficient for NVT (1/fs)
        save_interval: Interval for saving trajectory frames
        pre_relax: Whether to perform pre-relaxation before MD
        verbose: Whether to print progress messages
        **kwargs: Additional arguments passed to MDRunner

    Returns:
        Dictionary with results (temperature, energy, etc.)

    Example:
        >>> results = run_md(
        ...     structure="structure.cif",
        ...     model_path="uma-s-1.pt",
        ...     ensemble="NVT",
        ...     temperature=300,
        ...     steps=10000,
        ...     job_name="md_calc"
        ... )
        >>> print(f"Final temperature: {results['temperature']:.1f} K")
    """
    atoms = _load_structure(structure)
    if verbose:
        print(f"System: {atoms.get_chemical_formula()}")
        print(f"Atoms: {len(atoms)}")
        print(f"Loading model: {model_path}")

    config = EngineConfig(
        calc_type="md",
        model_path=Path(model_path),
        model_type=model_type,
        task=task,
        device=device,
        inference_mode="turbo",
        output_dir=Path(output_dir),
        job_name=job_name,
        options={
            "ensemble": ensemble,
            "temperature": temperature,
            "timestep": timestep,
            "steps": steps,
            "friction": friction,
            "save_interval": save_interval,
            "pre_relax": pre_relax,
        },
    )
    engine = CalculationEngine.from_config(config)
    return engine.run(atoms)


def calculate_energy(
    structure: Atoms | str | Path,
    model_path: str,
    model_type: str = "uma",
    task: str = "omat",
    device: str = "cpu",
    relax: bool = False,
    fmax: float = 0.05,
    max_steps: int = 100,
    verbose: bool = False,
    **kwargs,
) -> float:
    """Calculate energy of a structure.

    Simple interface to get just the energy value. Optionally performs
    a quick geometry optimization before calculating energy.

    Args:
        structure: ASE Atoms object or path to structure file
        model_path: Path to model checkpoint
        model_type: MLIP engine (uma, mace, dpa, grace) (omat, omol, oc20, oc25, odac, omc)
        device: Device for calculation (cpu or cuda)
        relax: Whether to pre-relax structure before energy calculation
        fmax: Force convergence threshold for relaxation (if relax=True)
        max_steps: Maximum steps for relaxation (if relax=True)
        verbose: Whether to print progress messages
        **kwargs: Additional arguments

    Returns:
        Energy in eV

    Example:
        >>> # Direct energy calculation
        >>> energy = calculate_energy("structure.cif", "uma-s-1.pt")
        >>> print(f"Energy: {energy:.4f} eV")
        >>>
        >>> # With pre-relaxation
        >>> energy = calculate_energy("structure.cif", "uma-s-1.pt", relax=True)
    """
    if relax:
        results = run_optimization(
            structure=structure,
            model_path=model_path,
            model_type=model_type,
            task=task,
            device=device,
            fmax=fmax,
            max_steps=max_steps,
            verbose=verbose,
            **kwargs,
        )
    else:
        results = run_single_point(
            structure=structure,
            model_path=model_path,
            model_type=model_type,
            task=task,
            device=device,
            verbose=verbose,
            **kwargs,
        )

    return results["energy"]


def calculate_adsorption_energy(
    adsorbed_structure: Atoms | str | Path,
    gas_structure: Atoms | str | Path,
    surface_structure: Atoms | str | Path,
    model_path: str,
    model_type: str = "uma",
    task: str = "omat",
    device: str = "cpu",
    relax: bool = False,
    verbose: bool = True,
    **kwargs,
) -> dict[str, float]:
    """Calculate adsorption energy.

    Adsorption energy is calculated as:
    E_adsorption = E(adsorbed) - E(gas) - E(surface)

    Args:
        adsorbed_structure: Structure with adsorbed molecule
        gas_structure: Isolated gas molecule structure
        surface_structure: Clean surface structure
        model_path: Path to model checkpoint
        model_type: MLIP engine (uma, mace, dpa, grace) (omat, omol, oc20, oc25, odac, omc)
        device: Device for calculation (cpu or cuda)
        relax: Whether to relax structures before calculation
        verbose: Whether to print progress messages
        **kwargs: Additional arguments passed to calculate_energy

    Returns:
        Dictionary with energies and adsorption energy:
        {
            "adsorbed_energy": float,
            "gas_energy": float,
            "surface_energy": float,
            "adsorption_energy": float
        }

    Example:
        >>> results = calculate_adsorption_energy(
        ...     adsorbed_structure="adsorbed.cif",
        ...     gas_structure="co2.xyz",
        ...     surface_structure="slab.cif",
        ...     model_path="uma-s-1.pt",
        ...     task="oc20"
        ... )
        >>> print(f"Adsorption energy: {results['adsorption_energy']:.4f} eV")
    """
    if verbose:
        print("=" * 60)
        print(" ADSORPTION ENERGY CALCULATION")
        print("=" * 60)
        print()

    # Calculate energies
    if verbose:
        print("1. Calculating adsorbed system energy...")
    E_adsorbed = calculate_energy(
        adsorbed_structure,
        model_path,
        model_type=model_type,
        task=task,
        device=device,
        relax=relax,
        verbose=verbose,
        **kwargs,
    )

    if verbose:
        print(f"   Energy: {E_adsorbed:.6f} eV")
        print()
        print("2. Calculating gas molecule energy...")
    E_gas = calculate_energy(
        gas_structure,
        model_path,
        model_type=model_type,
        task=task,
        device=device,
        relax=relax,
        verbose=verbose,
        **kwargs,
    )

    if verbose:
        print(f"   Energy: {E_gas:.6f} eV")
        print()
        print("3. Calculating clean surface energy...")
    E_surface = calculate_energy(
        surface_structure,
        model_path,
        model_type=model_type,
        task=task,
        device=device,
        relax=relax,
        verbose=verbose,
        **kwargs,
    )

    if verbose:
        print(f"   Energy: {E_surface:.6f} eV")
        print()

    # Calculate adsorption energy
    adsorption_energy = E_adsorbed - E_gas - E_surface

    if verbose:
        print("-" * 60)
        print(f" Adsorption Energy: {adsorption_energy:.6f} eV")
        print("=" * 60)

    return {
        "adsorbed_energy": E_adsorbed,
        "gas_energy": E_gas,
        "surface_energy": E_surface,
        "adsorption_energy": adsorption_energy,
    }


__all__ = [
    "calculate_adsorption_energy",
    "calculate_energy",
    "run_md",
    "run_optimization",
    "run_single_point",
]
