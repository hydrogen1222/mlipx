#!/usr/bin/env python3
"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

吸附能计算示例脚本

计算吸附能的公式：
E_adsorption = E(吸附结构) - E(气体分子) - E(清洁表面)

Usage:
    python adsorption_energy.py \\
        --adsorbed adsorbed.cif \\
        --gas gas.xyz \\
        --surface surface.cif \\
        --model uma-s-1.pt \\
        --task oc20

Or use as a module:
    from examples.adsorption_energy import calculate_adsorption_energy

    results = calculate_adsorption_energy(
        adsorbed_structure="adsorbed.cif",
        gas_structure="gas.xyz",
        surface_structure="surface.cif",
        model_path="uma-s-1.pt",
        task="oc20"
    )
    print(f"Adsorption energy: {results['adsorption_energy']:.4f} eV")
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add parent directory to path for mlipx imports
script_dir = Path(__file__).parent.resolve()
mlipx_dir = script_dir.parent
if str(mlipx_dir) not in sys.path:
    sys.path.insert(0, str(mlipx_dir))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Calculate adsorption energy using UMA models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python adsorption_energy.py \\
      --adsorbed adsorbed.cif \\
      --gas co2.xyz \\
      --surface clean_surface.cif \\
      --model uma-s-1.pt

  # With relaxation
  python adsorption_energy.py \\
      --adsorbed adsorbed.cif \\
      --gas co2.xyz \\
      --surface clean_surface.cif \\
      --model uma-s-1.pt \\
      --relax

  # Catalysis task (oc20)
  python adsorption_energy.py \\
      --adsorbed adsorbed.cif \\
      --gas co2.xyz \\
      --surface clean_surface.cif \\
      --model uma-s-1.pt \\
      --task oc20 \\
      --device cuda
        """,
    )

    # Required arguments
    parser.add_argument(
        "--adsorbed",
        type=str,
        required=True,
        help="Structure file with adsorbed molecule (required)",
    )
    parser.add_argument(
        "--gas",
        type=str,
        required=True,
        help="Isolated gas molecule structure file (required)",
    )
    parser.add_argument(
        "--surface",
        type=str,
        required=True,
        help="Clean surface structure file (required)",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to UMA model checkpoint (required)",
    )

    # Optional arguments
    parser.add_argument(
        "--task",
        type=str,
        default="omat",
        choices=["omat", "omol", "oc20", "oc25", "odac", "omc"],
        help="Task type (default: omat)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for calculation (default: cpu)",
    )
    parser.add_argument(
        "--relax",
        action="store_true",
        help="Pre-relax structures before energy calculation",
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=0.05,
        help="Force convergence threshold for relaxation (default: 0.05 eV/Å)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Maximum optimization steps for relaxation (default: 100)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output directory for detailed results (optional)",
    )
    parser.add_argument(
        "--name",
        "-n",
        type=str,
        default="adsorption_calc",
        help="Job name for output organization (default: adsorption_calc)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    return parser.parse_args()


def validate_inputs(args) -> bool:
    """Validate input files exist.

    Args:
        args: Parsed arguments

    Returns:
        True if all inputs valid, False otherwise
    """
    valid = True

    for name, path in [
        ("Adsorbed structure", args.adsorbed),
        ("Gas structure", args.gas),
        ("Surface structure", args.surface),
        ("Model", args.model),
    ]:
        if not Path(path).exists():
            print(f"Error: {name} file not found: {path}", file=sys.stderr)
            valid = False

    return valid


def calculate_adsorption_energy(
    adsorbed_structure: str,
    gas_structure: str,
    surface_structure: str,
    model_path: str,
    task: str = "omat",
    device: str = "cpu",
    relax: bool = False,
    fmax: float = 0.05,
    max_steps: int = 100,
    output_dir: str | None = None,
    job_name: str = "adsorption_calc",
    verbose: bool = True,
) -> dict[str, float]:
    """Calculate adsorption energy.

    Args:
        adsorbed_structure: Path to structure file with adsorbed molecule
        gas_structure: Path to isolated gas molecule structure
        surface_structure: Path to clean surface structure
        model_path: Path to UMA model checkpoint
        task: Task type (omat, omol, oc20, etc.)
        device: Device for calculation (cpu or cuda)
        relax: Whether to pre-relax structures
        fmax: Force convergence threshold for relaxation
        max_steps: Maximum optimization steps
        output_dir: Directory for detailed output (optional)
        job_name: Job name for organization
        verbose: Whether to print progress

    Returns:
        Dictionary with energy components and adsorption energy
    """
    # Import here to avoid slow import if just checking help
    from mlipx.api import calculate_adsorption_energy as api_calc_adsorption

    return api_calc_adsorption(
        adsorbed_structure=adsorbed_structure,
        gas_structure=gas_structure,
        surface_structure=surface_structure,
        model_path=model_path,
        task=task,
        device=device,
        relax=relax,
        verbose=verbose,
        fmax=fmax,
        max_steps=max_steps,
    )


def main():
    """Main entry point."""
    args = parse_args()

    # Validate inputs
    if not validate_inputs(args):
        return 1

    # Calculate adsorption energy
    try:
        results = calculate_adsorption_energy(
            adsorbed_structure=args.adsorbed,
            gas_structure=args.gas,
            surface_structure=args.surface,
            model_path=args.model,
            task=args.task,
            device=args.device,
            relax=args.relax,
            fmax=args.fmax,
            max_steps=args.max_steps,
            output_dir=args.output,
            job_name=args.name,
            verbose=not args.quiet,
        )

        # Print results summary
        if not args.quiet:
            print()
            print("=" * 60)
            print(" CALCULATION COMPLETE")
            print("=" * 60)

        print("\nEnergy Components:")
        print(f"  Adsorbed system:  {results['adsorbed_energy']:12.6f} eV")
        print(f"  Gas molecule:     {results['gas_energy']:12.6f} eV")
        print(f"  Clean surface:    {results['surface_energy']:12.6f} eV")
        print("\nAdsorption Energy:")
        print(f"  E_ads = {results['adsorption_energy']:12.6f} eV")
        print(f"  E_ads = {results['adsorption_energy'] * 23.0605:12.4f} kcal/mol")
        print(f"  E_ads = {results['adsorption_energy'] * 96.485:12.4f} kJ/mol")

        return 0

    except Exception as e:
        print(f"\nError during calculation: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
