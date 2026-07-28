# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Modified for the mlipx project: multi-engine MLIP support (UMA/MACE/DPA/GRACE).
"""
Command-line interface for mlipx.

Provides subcommands for different calculation types:
- run: Run from INCAR configuration file
- sp: Single point calculation
- opt: Geometry optimization
- md: Molecular dynamics
- batch: Batch processing
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from ase.io import read

from mlipx.config import IncarConfig, get_default_config
from mlipx.engine import CalculationEngine, EngineConfig


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="mlipx",
        description="mlipx - VASP-like CLI for MLIP models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run from INCAR file
  mlipx run

  # Single point calculation
  mlipx sp structure.cif --model uma-s-1.pt --task omat

  # Geometry optimization with cell relaxation
  mlipx opt structure.cif --cell-opt --fmax 0.02

  # Molecular dynamics (NVT)
  mlipx md structure.cif --ensemble NVT --temp 300 --steps 10000

  # Batch processing
  mlipx batch structures/ --pattern "*.cif" --output results/

  # Run environment diagnostic (recommended after install!)
  mlipx doctor

  # Generate template INCAR
  mlipx template sp
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # run command
    run_parser = subparsers.add_parser(
        "run",
        help="Run calculation from INCAR file",
        description="Read configuration from INCAR file and run calculation",
    )
    run_parser.add_argument(
        "-i",
        "--incar",
        type=str,
        default="INCAR.mlipx",
        help="Path to INCAR configuration file (default: INCAR.mlipx)",
    )
    run_parser.add_argument(
        "-s",
        "--structure",
        type=str,
        default=None,
        help="Structure file (default: POSCAR, CONTCAR, or from INCAR)",
    )
    run_parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=".",
        help="Output directory (default: current directory)",
    )

    # sp command
    sp_parser = subparsers.add_parser(
        "sp",
        help="Single point calculation",
        description="Calculate energy, forces, and stress",
    )
    sp_parser.add_argument(
        "structure",
        type=str,
        help="Input structure file (CIF, XYZ, POSCAR, etc.)",
    )
    sp_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    sp_parser.add_argument(
        "--model-type",
        type=str,
        default="uma",
        choices=["uma", "mace", "dpa", "grace"],
        help="MLIP engine type (default: uma)",
    )
    sp_parser.add_argument(
        "--task",
        type=str,
        default="omat",
        help="Task type. UMA: omat/omol/oc20/oc25/odac/omc. Others: bulk/molecule (default: omat)",
    )
    sp_parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for calculation (default: cpu)",
    )
    sp_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=".",
        help="Output directory",
    )
    sp_parser.add_argument(
        "--name",
        "-n",
        type=str,
        default=None,
        help="Job name (output will be in OUTPUT/NAME)",
    )

    # opt command
    opt_parser = subparsers.add_parser(
        "opt",
        help="Geometry optimization",
        description="Optimize atomic positions and optionally cell parameters",
    )
    opt_parser.add_argument(
        "structure",
        type=str,
        help="Input structure file",
    )
    opt_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    opt_parser.add_argument(
        "--model-type",
        type=str,
        default="uma",
        choices=["uma", "mace", "dpa", "grace"],
        help="MLIP engine type (default: uma)",
    )
    opt_parser.add_argument(
        "--task",
        type=str,
        default="omat",
        help="Task type. UMA: omat/omol/oc20/oc25/odac/omc. Others: bulk/molecule (default: omat)",
    )
    opt_parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for calculation (default: cpu)",
    )
    opt_parser.add_argument(
        "--fmax",
        type=float,
        default=0.05,
        help="Force convergence threshold in eV/Å (default: 0.05)",
    )
    opt_parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Maximum optimization steps (default: 500)",
    )
    opt_parser.add_argument(
        "--optimizer",
        type=str,
        default="FIRE",
        choices=["FIRE", "BFGS", "LBFGS"],
        help="Optimization algorithm (default: FIRE)",
    )
    opt_parser.add_argument(
        "--cell-opt",
        action="store_true",
        help="Optimize cell parameters (requires stress support)",
    )
    opt_parser.add_argument(
        "--fix-symmetry",
        action="store_true",
        help="Preserve crystal symmetry during optimization",
    )
    opt_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=".",
        help="Output directory",
    )
    opt_parser.add_argument(
        "--name",
        "-n",
        type=str,
        default=None,
        help="Job name (output will be in OUTPUT/NAME)",
    )

    # md command
    md_parser = subparsers.add_parser(
        "md",
        help="Molecular dynamics",
        description="Run MD simulation (NVT or NVE ensemble)",
    )
    md_parser.add_argument(
        "structure",
        type=str,
        help="Input structure file",
    )
    md_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    md_parser.add_argument(
        "--model-type",
        type=str,
        default="uma",
        choices=["uma", "mace", "dpa", "grace"],
        help="MLIP engine type (default: uma)",
    )
    md_parser.add_argument(
        "--task",
        type=str,
        default="omat",
        help="Task type. UMA: omat/omol/oc20/oc25/odac/omc. Others: bulk/molecule (default: omat)",
    )
    md_parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Device for calculation (default: cuda)",
    )
    md_parser.add_argument(
        "--ensemble",
        type=str,
        default="NVT",
        choices=["NVT", "NVE"],
        help="MD ensemble (default: NVT)",
    )
    md_parser.add_argument(
        "--temp",
        type=float,
        default=300.0,
        help="Temperature in Kelvin (default: 300)",
    )
    md_parser.add_argument(
        "--timestep",
        type=float,
        default=1.0,
        help="Time step in femtoseconds (default: 1.0)",
    )
    md_parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Number of MD steps (default: 1000)",
    )
    md_parser.add_argument(
        "--friction",
        type=float,
        default=0.001,
        help="Friction coefficient for NVT (default: 0.001)",
    )
    md_parser.add_argument(
        "--save-interval",
        type=int,
        default=10,
        help="Interval for saving trajectory frames (default: 10)",
    )
    md_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=".",
        help="Output directory",
    )
    md_parser.add_argument(
        "--name",
        "-n",
        type=str,
        default=None,
        help="Job name (output will be in OUTPUT/NAME)",
    )

    # batch command
    batch_parser = subparsers.add_parser(
        "batch",
        help="Batch processing",
        description="Process multiple structures in batch mode",
    )
    batch_parser.add_argument(
        "input_dir",
        type=str,
        help="Input directory containing structure files",
    )
    batch_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    batch_parser.add_argument(
        "--model-type",
        type=str,
        default="uma",
        choices=["uma", "mace", "dpa", "grace"],
        help="MLIP engine type (default: uma)",
    )
    batch_parser.add_argument(
        "--task",
        type=str,
        default="omat",
        help="Task type. UMA: omat/omol/oc20/oc25/odac/omc. Others: bulk/molecule (default: omat)",
    )
    batch_parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for calculation (default: cpu)",
    )
    batch_parser.add_argument(
        "--calc-type",
        type=str,
        default="sp",
        choices=["sp", "opt"],
        help="Type of calculation (default: sp)",
    )
    batch_parser.add_argument(
        "--pattern",
        type=str,
        default="*.cif",
        help="File pattern to match (default: *.cif)",
    )
    batch_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="batch_results",
        help="Output directory (default: batch_results)",
    )
    batch_parser.add_argument(
        "--name",
        "-n",
        type=str,
        default=None,
        help="Job name (output will be in OUTPUT/NAME)",
    )

    # template command
    template_parser = subparsers.add_parser(
        "template",
        help="Generate template INCAR files",
        description="Generate template configuration files",
    )
    template_parser.add_argument(
        "type",
        choices=["sp", "opt", "md"],
        help="Type of template to generate",
    )
    template_parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output file name (default: INCAR.{type})",
    )

    # tui command
    subparsers.add_parser(
        "tui",
        help="Launch interactive TUI mode",
        description="Launch interactive terminal UI for visual configuration",
    )

    # doctor command
    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Run environment diagnostic checks",
        description="Check Python, PyTorch, CUDA, GPU compatibility, and model file",
    )
    doctor_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to model checkpoint to verify",
    )

    # setup command
    setup_parser = subparsers.add_parser(
        "setup",
        help="Detect your GPU and get the matching PyTorch install command",
        description=(
            "Detect NVIDIA GPUs via nvidia-smi (works even before PyTorch is "
            "installed) and print the exact torch version + install commands "
            "for your card. Supports Maxwell (GTX 900 series) through Hopper; "
            "Blackwell (RTX 50) gets torch 2.8+; Kepler (GTX 700) is flagged "
            "unsupported."
        ),
    )
    setup_parser.add_argument(
        "--json",
        action="store_true",
        help="Output machine-readable JSON instead of the text report",
    )

    # jobs command
    jobs_parser = subparsers.add_parser("jobs", help="List background jobs")
    jobs_parser.add_argument(
        "--refresh", type=int, default=0, help="Auto-refresh interval in seconds"
    )

    # kill command
    kill_parser = subparsers.add_parser("kill", help="Kill a background job")
    kill_parser.add_argument("job_id", help="Job ID to kill")

    # clean command
    subparsers.add_parser("clean", help="Remove completed/failed job records")

    return parser


def print_header():
    """Print mlipx header."""
    print("=" * 80)
    print(" MLIPX".center(80))
    print(" (mlipx - MLIP eXtended)".center(80))
    print("=" * 80)
    print()


def _run_started_at(args: argparse.Namespace) -> float:
    """Return the command-entry timestamp, including parsing and input loading."""
    return getattr(args, "_run_started_at", time.perf_counter())


def _console_log(message: str, level: str = "info") -> None:
    """Print a live engine log message immediately."""
    print(message, flush=True)


def cmd_run(args: argparse.Namespace) -> int:
    """Execute 'run' command."""
    started_at = _run_started_at(args)
    # Load configuration
    incar_path = Path(args.incar)
    # Backward-compat: fall back to legacy INCAR.uma if the default is missing.
    if not incar_path.exists() and args.incar == "INCAR.mlipx":
        legacy = Path("INCAR.uma")
        if legacy.exists():
            incar_path = legacy
    if not incar_path.exists():
        print(f"Error: INCAR file not found: {incar_path}")
        return 1

    config = IncarConfig.from_file(incar_path)

    # Validate configuration
    errors = config.validate()
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        return 1

    # Determine structure file
    structure_file = args.structure
    if structure_file is None:
        # Try common defaults
        for default in ["POSCAR", "CONTCAR", "structure.cif", "structure.xyz"]:
            if Path(default).exists():
                structure_file = default
                break

    if structure_file is None:
        print("Error: No structure file specified and no default found")
        return 1

    structure_path = Path(structure_file)
    if not structure_path.exists():
        print(f"Error: Structure file not found: {structure_path}")
        return 1

    # Read structure
    print(f"Reading structure from: {structure_path}")
    try:
        atoms = read(structure_path)
    except Exception as e:
        print(f"Error reading structure: {e}")
        return 1

    print(f"System: {atoms.get_chemical_formula()}")
    print(f"Atoms: {len(atoms)}")
    print()

    # Determine calculation type
    calc_type = config.get_str("CALC_TYPE", "sp").lower()
    output_dir = Path(args.output)
    job_name = config.get_str("JOB_NAME", None)

    engine_config = EngineConfig(
        calc_type=calc_type,
        model_path=Path(config.get_str("MODEL_PATH", "uma-s-1.pt")),
        model_type=config.get_str("MODEL_TYPE", "uma"),
        task=config.get_str("TASK", "omat"),
        device=config.get_str("DEVICE", "cpu"),
        inference_mode=config.get_str("INFERENCE_MODE", "default"),
        output_dir=output_dir,
        job_name=job_name,
        options={
            "fmax": config.get_float("FMAX", 0.05),
            "max_steps": config.get_int("MAX_STEPS", 500),
            "optimizer": config.get_str("OPT_ALGO", "FIRE"),
            "cell_opt": config.get_bool("CELL_OPT", False),
            "fix_symmetry": config.get_bool("FIX_SYMMETRY", False),
            "ensemble": config.get_str("MD_ENSEMBLE", "NVT"),
            "temperature": config.get_float("TEMPERATURE", 300.0),
            "timestep": config.get_float("TIMESTEP", 1.0),
            "steps": config.get_int("STEPS", 1000),
            "friction": config.get_float("FRICTION", 0.001),
            "save_interval": config.get_int("SAVE_INTERVAL", 10),
        },
    )

    try:
        engine = CalculationEngine.from_config(engine_config)
        engine.run(atoms, log_fn=_console_log, started_at=started_at)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


def cmd_sp(args: argparse.Namespace) -> int:
    """Execute 'sp' command."""
    started_at = _run_started_at(args)

    structure_path = Path(args.structure)
    if not structure_path.exists():
        print(f"Error: Structure file not found: {structure_path}")
        return 1

    config = EngineConfig(
        calc_type="sp",
        model_path=Path(args.model),
        model_type=args.model_type,
        task=args.task,
        device=args.device,
        output_dir=Path(args.output),
        job_name=args.name,
    )

    print_header()
    print(f"System: reading from {structure_path}")

    try:
        atoms = read(structure_path)
        print(f"System: {atoms.get_chemical_formula()}")
        print(f"Atoms: {len(atoms)}")

        engine = CalculationEngine.from_config(config)
        engine.run(atoms, log_fn=_console_log, started_at=started_at)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


def cmd_opt(args: argparse.Namespace) -> int:
    """Execute 'opt' command."""
    started_at = _run_started_at(args)

    structure_path = Path(args.structure)
    if not structure_path.exists():
        print(f"Error: Structure file not found: {structure_path}")
        return 1

    config = EngineConfig(
        calc_type="opt",
        model_path=Path(args.model),
        model_type=args.model_type,
        task=args.task,
        device=args.device,
        output_dir=Path(args.output),
        job_name=args.name,
        options={
            "fmax": args.fmax,
            "max_steps": args.max_steps,
            "optimizer": args.optimizer,
            "cell_opt": args.cell_opt,
            "fix_symmetry": args.fix_symmetry,
        },
    )

    print_header()
    print(f"Reading structure from: {structure_path}")

    try:
        atoms = read(structure_path)
        print(f"System: {atoms.get_chemical_formula()}")
        print(f"Atoms: {len(atoms)}")

        engine = CalculationEngine.from_config(config)
        engine.run(atoms, log_fn=_console_log, started_at=started_at)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


def cmd_md(args: argparse.Namespace) -> int:
    """Execute 'md' command."""
    started_at = _run_started_at(args)

    structure_path = Path(args.structure)
    if not structure_path.exists():
        print(f"Error: Structure file not found: {structure_path}")
        return 1

    config = EngineConfig(
        calc_type="md",
        model_path=Path(args.model),
        model_type=args.model_type,
        task=args.task,
        device=args.device,
        inference_mode="turbo",
        output_dir=Path(args.output),
        job_name=args.name,
        options={
            "ensemble": args.ensemble,
            "temperature": args.temp,
            "timestep": args.timestep,
            "steps": args.steps,
            "friction": args.friction,
            "save_interval": args.save_interval,
        },
    )

    print_header()
    print(f"Reading structure from: {structure_path}")

    try:
        atoms = read(structure_path)
        print(f"System: {atoms.get_chemical_formula()}")
        print(f"Atoms: {len(atoms)}")

        engine = CalculationEngine.from_config(config)
        engine.run(atoms, log_fn=_console_log, started_at=started_at)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


def cmd_batch(args: argparse.Namespace) -> int:
    """Execute 'batch' command."""
    started_at = _run_started_at(args)

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return 1

    config = EngineConfig(
        calc_type="batch",
        model_path=Path(args.model),
        model_type=args.model_type,
        task=args.task,
        device=args.device,
        output_dir=Path(args.output),
        job_name=args.name,
        options={
            "sub_calc_type": args.calc_type,
            "pattern": args.pattern,
        },
    )

    print_header()

    try:
        engine = CalculationEngine.from_config(config)
        files = list(input_dir.glob(args.pattern))
        if not files:
            print(f"No files matching '{args.pattern}' found in {input_dir}")
            return 1
        print(f"Found {len(files)} structure files")
        summary = engine.run_batch(
            files,
            log_fn=_console_log,
            started_at=started_at,
        )
        if summary["failed"] > 0:
            return 1
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


def cmd_template(args: argparse.Namespace) -> int:
    """Execute 'template' command."""
    config = get_default_config(args.type)

    output_file = args.output
    if output_file is None:
        output_file = f"INCAR.{args.type}"

    config.write(output_file)
    print(f"Template written to: {output_file}")

    return 0


def cmd_tui(args: argparse.Namespace) -> int:
    """Launch interactive TUI mode."""
    try:
        from mlipx.tui import MlipxApp  # noqa: PLC0415
    except ImportError as e:
        print("Error: TUI mode requires textual.")
        print("Install with: pip install textual")
        print(f"Import error: {e}")
        return 1

    app = MlipxApp()
    return app.run()


def cmd_doctor(args: argparse.Namespace) -> int:
    """Run environment diagnostic checks."""
    from mlipx.doctor import run_diagnostics, format_diagnostics  # noqa: PLC0415

    checks, failures = run_diagnostics(model_path=args.model)
    print(format_diagnostics(checks))
    return 0 if failures == 0 else 1


def cmd_setup(args: argparse.Namespace) -> int:
    """Detect the GPU and print the matching PyTorch install commands."""
    import json  # noqa: PLC0415

    from mlipx.gpu_setup import (  # noqa: PLC0415
        detect_gpus,
        format_setup_report,
        setup_report_json,
    )

    gpus = detect_gpus()
    if args.json:
        print(json.dumps(setup_report_json(gpus), indent=2, ensure_ascii=False))
        return 0 if gpus else 1
    print(format_setup_report(gpus))
    return 0 if gpus else 1


def cmd_jobs(args: argparse.Namespace) -> int:
    """List background jobs."""
    from mlipx.jobs import JobManager  # noqa: PLC0415

    mgr = JobManager()
    jobs = mgr.list_jobs()
    if not jobs:
        print("No jobs found.")
        return 0
    print(f"{'ID':<40} {'Status':<12} {'Type':<6} {'Formula':<12} {'Device'}")
    print("-" * 90)
    for j in jobs:
        print(
            f"{j['job_id']:<40} {j['status']:<12} {j.get('calc_type', ''):<6} {j.get('formula', ''):<12} {j.get('device', '')}"
        )
    return 0


def cmd_kill(args: argparse.Namespace) -> int:
    """Kill a background job."""
    from mlipx.jobs import JobManager  # noqa: PLC0415

    mgr = JobManager()
    ok = mgr.kill_job(args.job_id)
    if ok:
        print(f"Killed: {args.job_id}")
        return 0
    else:
        print(f"Failed to kill: {args.job_id}")
        return 1


def cmd_clean(args: argparse.Namespace) -> int:
    """Remove completed/failed job records."""
    from mlipx.jobs import JobManager  # noqa: PLC0415

    mgr = JobManager()
    removed = mgr.clean()
    if removed:
        print(f"Removed {len(removed)} completed/failed job records.")
    else:
        print("No completed/failed jobs to clean.")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    run_started_at = time.perf_counter()
    # Check if running in TUI mode (no command, or explicit 'tui' command)
    if argv is None:
        argv = sys.argv[1:]

    # If no arguments provided, launch TUI by default
    if len(argv) == 0:
        try:
            from mlipx.tui import MlipxApp  # noqa: PLC0415

            print("Launching interactive TUI mode...")
            print("(Use --help for command-line interface)")
            time.sleep(0.5)
            app = MlipxApp()
            return app.run()
        except ImportError:
            # Fall back to CLI help if textual not installed
            pass

    parser = create_parser()
    args = parser.parse_args(argv)
    args._run_started_at = run_started_at

    # Handle TUI command
    if args.command == "tui":
        return cmd_tui(args)

    if args.command is None:
        parser.print_help()
        return 1

    # --json output must be clean (no banner) for scripting.
    if not (args.command == "setup" and getattr(args, "json", False)):
        print_header()

    # Dispatch to appropriate command handler
    commands = {
        "run": cmd_run,
        "sp": cmd_sp,
        "opt": cmd_opt,
        "md": cmd_md,
        "batch": cmd_batch,
        "template": cmd_template,
        "doctor": cmd_doctor,
        "setup": cmd_setup,
        "jobs": cmd_jobs,
        "kill": cmd_kill,
        "clean": cmd_clean,
    }

    handler = commands.get(args.command)
    if handler is None:
        print(f"Error: Unknown command: {args.command}")
        return 1

    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
