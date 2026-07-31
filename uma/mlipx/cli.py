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

from mlipx.config import (
    IncarConfig,
    get_default_config,
    get_schema,
    resolve_config,
)
from mlipx.config.settings import init_settings_file
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

    # Global option: explicit settings.ini (plan section 4.2). Must precede
    # the subcommand, e.g. `mlipx --settings path.ini sp ...`.
    parser.add_argument(
        "--settings",
        type=str,
        default=None,
        help="Path to a settings.ini file (overrides MLIPX_SETTINGS env / "
        "./settings.ini / ~/.config/mlipx/settings.ini).",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    def _add_resolver_args(p: argparse.ArgumentParser) -> None:
        """Add config-resolver args shared by sp/opt/md/batch."""
        p.add_argument(
            "--inference-mode",
            type=str,
            default=None,
            choices=["default", "turbo"],
            help="UMA inference mode (default: task-specific; ignored by other engines).",
        )
        p.add_argument(
            "--cpu-threads",
            "--torch-num-threads",
            dest="torch_num_threads",
            type=int,
            metavar="N",
            default=None,
            help="CPU intra-op threads (PyTorch for UMA/MACE/DPA, TensorFlow "
            "for GRACE; default: backend/system).",
        )
        p.add_argument(
            "--activation-checkpointing",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="UMA GPU memory-saving activation checkpointing.",
        )
        p.add_argument(
            "--dtype",
            "--default-dtype",
            dest="default_dtype",
            type=str,
            default=None,
            choices=["float32", "float64"],
            help="MACE model dtype (default: float32 for every calculation type).",
        )
        p.add_argument(
            "--head",
            type=str,
            default=None,
            help="MACE head or DeepMD/DPA branch name.",
        )
        p.add_argument(
            "--model-alias",
            type=str,
            default=None,
            help="Model alias defined in settings.ini [model:NAME].",
        )
        p.add_argument(
            "--profile",
            type=str,
            default=None,
            help="Reusable profile from settings.ini [profile:NAME].",
        )

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
        default=None,
        help="Path to model checkpoint (or a model-alias name; see --model-alias).",
    )
    sp_parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        choices=["uma", "mace", "dpa", "grace"],
        help="MLIP engine type (default: uma; resolved from settings).",
    )
    sp_parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task type. UMA: omat/omol/...; others: bulk/molecule (default: engine-specific).",
    )
    sp_parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for calculation: cpu, cuda, gpu or cuda:N (default: cpu).",
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
    _add_resolver_args(sp_parser)
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
        default=None,
        help="Path to model checkpoint (or a model-alias name; see --model-alias).",
    )
    opt_parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        choices=["uma", "mace", "dpa", "grace"],
        help="MLIP engine type (default: uma; resolved from settings).",
    )
    opt_parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task type. UMA: omat/omol/...; others: bulk/molecule (default: engine-specific).",
    )
    opt_parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for calculation: cpu, cuda, gpu or cuda:N (default: cpu).",
    )
    opt_parser.add_argument(
        "--fmax",
        type=float,
        default=None,
        help="Force convergence threshold in eV/Angstrom (default: 0.05).",
    )
    opt_parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum optimization steps (default: 500).",
    )
    opt_parser.add_argument(
        "--optimizer",
        type=str,
        default=None,
        choices=["FIRE", "BFGS", "LBFGS"],
        help="Optimization algorithm (default: FIRE).",
    )
    opt_parser.add_argument(
        "--cell-opt",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Optimize cell parameters (requires stress support).",
    )
    opt_parser.add_argument(
        "--fix-symmetry",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Preserve crystal symmetry during optimization.",
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
    _add_resolver_args(opt_parser)

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
        default=None,
        help="Path to model checkpoint (or a model-alias name; see --model-alias).",
    )
    md_parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        choices=["uma", "mace", "dpa", "grace"],
        help="MLIP engine type (default: uma; resolved from settings).",
    )
    md_parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task type. UMA: omat/omol/...; others: bulk/molecule (default: engine-specific).",
    )
    md_parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for calculation: cpu, cuda, gpu or cuda:N (default: cuda).",
    )
    md_parser.add_argument(
        "--ensemble",
        type=str,
        default=None,
        choices=["NVT", "NVE"],
        help="MD ensemble (default: NVT).",
    )
    md_parser.add_argument(
        "--temp",
        type=float,
        default=None,
        help="Temperature in Kelvin (default: 300).",
    )
    md_parser.add_argument(
        "--timestep",
        type=float,
        default=None,
        help="Time step in femtoseconds (default: 1.0).",
    )
    md_parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of MD steps (default: 1000).",
    )
    md_parser.add_argument(
        "--friction",
        type=float,
        default=None,
        help="Friction coefficient for NVT (default: 0.001).",
    )
    md_parser.add_argument(
        "--save-interval",
        type=int,
        default=None,
        help="Interval for saving trajectory frames (default: 10).",
    )
    md_parser.add_argument(
        "--pre-relax",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Pre-relax the structure before MD (default: enabled for NVT, disabled for NVE).",
    )
    md_parser.add_argument(
        "--pre-relax-steps",
        type=int,
        default=None,
        help="Maximum pre-relaxation steps (default: 50).",
    )
    md_parser.add_argument(
        "--pre-relax-fmax",
        type=float,
        default=None,
        help="Pre-relaxation force threshold in eV/Angstrom (default: 0.1).",
    )
    md_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible MD (default: auto-generated and recorded).",
    )
    md_parser.add_argument(
        "--velocity-policy",
        choices=["auto", "initialize", "preserve"],
        default=None,
        help="Velocity initialization policy (default: auto).",
    )
    md_parser.add_argument(
        "--fmax-abort",
        type=float,
        default=None,
        help="Large-force warning threshold in eV/Angstrom (default: 20).",
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
    _add_resolver_args(md_parser)

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
        default=None,
        help="Path to model checkpoint (or a model-alias name; see --model-alias).",
    )
    batch_parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        choices=["uma", "mace", "dpa", "grace"],
        help="MLIP engine type (default: uma; resolved from settings).",
    )
    batch_parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task type. UMA: omat/omol/...; others: bulk/molecule (default: engine-specific).",
    )
    batch_parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for calculation: cpu, cuda, gpu or cuda:N (default: cpu).",
    )
    batch_parser.add_argument(
        "--calc-type",
        type=str,
        default=None,
        choices=["sp", "opt"],
        help="Sub-calculation type for the sweep (default: sp).",
    )
    batch_parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help=(
            "File glob to match. If omitted, discovers *.cif, *.xyz, *.vasp "
            "and POSCAR*."
        ),
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
    _add_resolver_args(batch_parser)

    # config command (plan section 4.2 / 9 / 17.6 / 24)
    config_parser = subparsers.add_parser(
        "config",
        help="Inspect and manage mlipx configuration",
        description="Show resolved config, settings search paths, validate "
        "settings.ini, or explain where a parameter value comes from.",
    )
    config_sub = config_parser.add_subparsers(
        dest="config_command", required=True
    )
    config_sub.add_parser("show", help="Show the resolved configuration")
    config_sub.add_parser("paths", help="List settings.ini search paths")
    config_init = config_sub.add_parser("init", help="Create a settings.ini")
    config_init.add_argument(
        "--project",
        action="store_true",
        help="Write ./settings.ini",
    )
    config_init.add_argument(
        "--user",
        action="store_true",
        help="Write the user-level settings.ini",
    )
    config_init.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Explicit output path (overrides --project/--user)",
    )
    config_init.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing file",
    )
    config_validate = config_sub.add_parser(
        "validate", help="Validate a settings.ini file"
    )
    config_validate.add_argument(
        "path",
        type=str,
        nargs="?",
        default=None,
        help="settings.ini path (default: resolved search)",
    )
    config_explain = config_sub.add_parser(
        "explain",
        help="Explain why a parameter has its resolved value",
    )
    config_explain.add_argument("key", type=str, help="Parameter name")
    config_show = config_sub.add_parser(
        "schema", help="List recognised option keys"
    )
    config_show.add_argument(
        "--strict",
        action="store_true",
        help="Only show keys valid in strict mode",
    )

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


def _load_settings(args: argparse.Namespace):
    """Load settings.ini honouring --settings / MLIPX_SETTINGS / cwd / user."""
    from mlipx.config.settings import load_settings  # noqa: PLC0415

    return load_settings(explicit=getattr(args, "settings", None))


def _build_cli_opts(args: argparse.Namespace, calc_type: str) -> dict:
    """Collect non-None CLI args into a canonical option dict.

    Only explicitly-provided values are included so that unspecified
    parameters fall through to settings/profile/built-in defaults
    (plan section 17.6: argparse defaults -> None, resolved by ConfigResolver).
    """
    opts: dict = {}
    for key in (
        "model_type",
        "task",
        "device",
        "inference_mode",
        "torch_num_threads",
        "activation_checkpointing",
        "default_dtype",
        "head",
    ):
        value = getattr(args, key, None)
        if value is not None:
            opts[key] = value
    if calc_type == "opt":
        for key in ("fmax", "max_steps", "optimizer", "cell_opt", "fix_symmetry"):
            value = getattr(args, key, None)
            if value is not None:
                opts[key] = value
    elif calc_type == "md":
        if getattr(args, "temp", None) is not None:
            opts["temperature"] = args.temp
        for key in (
            "ensemble",
            "timestep",
            "steps",
            "friction",
            "save_interval",
            "pre_relax",
            "pre_relax_steps",
            "pre_relax_fmax",
            "seed",
            "velocity_policy",
            "fmax_abort",
        ):
            value = getattr(args, key, None)
            if value is not None:
                opts[key] = value
    elif calc_type == "batch":
        # batch `--calc-type` selects the sub-calculation (sp/opt).
        if getattr(args, "calc_type", None) is not None:
            opts["sub_calc_type"] = args.calc_type
        for key in ("pattern",):
            value = getattr(args, key, None)
            if value is not None:
                opts[key] = value
    return opts


def _resolve_engine_config(
    args: argparse.Namespace,
    calc_type: str,
    *,
    model_path: str | None = None,
    incar_layer: dict | None = None,
    output_dir: str | None = None,
    job_name: str | None = None,
) -> tuple:
    """Resolve a full config and build an EngineConfig from it.

    Returns ``(EngineConfig, ResolvedConfig, MlipxSettings)``.
    """
    from mlipx.config.aliases import parse_model_aliases  # noqa: PLC0415

    settings = _load_settings(args)
    aliases = parse_model_aliases(settings.parser)

    model_alias = getattr(args, "model_alias", None)
    cli = _build_cli_opts(args, calc_type)

    # `--model` may be a filesystem path OR a model-alias name (plan section
    # 5.1: `--model mace_mpa0`). An explicit --model-alias wins; otherwise a
    # value matching a known alias is treated as the alias.
    if model_path is not None:
        cli.setdefault("model_path", model_path)
    else:
        model_arg = getattr(args, "model", None)
        if model_arg is not None:
            if model_alias is None and model_arg in aliases:
                model_alias = model_arg
            else:
                cli.setdefault("model_path", model_arg)
        elif model_alias is None and not incar_layer:
            raise SystemExit(
                "Error: one of --model PATH or --model-alias NAME is required."
            )

    resolved = resolve_config(
        calc_type=calc_type,
        settings=settings,
        model_alias_name=model_alias,
        profile_name=getattr(args, "profile", None),
        incar=incar_layer,
        cli=cli,
    )
    engine_config = EngineConfig.from_resolved(resolved)
    engine_config.output_dir = Path(output_dir if output_dir else getattr(args, "output", "."))
    engine_config.job_name = job_name if job_name is not None else getattr(args, "name", None)
    return engine_config, resolved, settings


def _emit_resolved_config(resolved, output_dir: Path) -> None:
    """Write resolved_config.json when enabled (plan section 15 / 4.5)."""
    import json  # noqa: PLC0415

    if not resolved.settings.get("write_resolved_config", True):
        return
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "resolved_config.json").write_text(
            json.dumps(resolved.as_dict(), indent=2, default=str),
            encoding="utf-8",
        )
    except OSError:
        pass


def cmd_run(args: argparse.Namespace) -> int:
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

    # Determine calculation type from INCAR (authoritative for the `run` flow).
    calc_type = config.get_str("CALC_TYPE", "sp").lower()
    job_name = config.get_str("JOB_NAME", None)
    # The INCAR dict (UPPER keys) is passed as a resolver layer; aliases are
    # canonicalised automatically (MODEL_TYPE -> model_type, FMAX -> fmax ...).
    incar_layer = {str(k): v for k, v in config.items()}

    engine_config, resolved, _settings = _resolve_engine_config(
        args,
        calc_type,
        incar_layer=incar_layer,
        output_dir=args.output,
        job_name=job_name,
    )

    try:
        _emit_resolved_config(resolved, engine_config.output_dir)
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

    config, resolved, _settings = _resolve_engine_config(args, "sp")

    print_header()
    print(f"System: reading from {structure_path}")

    try:
        atoms = read(structure_path)
        print(f"System: {atoms.get_chemical_formula()}")
        print(f"Atoms: {len(atoms)}")

        _emit_resolved_config(resolved, config.output_dir / (config.job_name or "") if config.job_name else config.output_dir)
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

    config, resolved, _settings = _resolve_engine_config(args, "opt")

    print_header()
    print(f"Reading structure from: {structure_path}")

    try:
        atoms = read(structure_path)
        print(f"System: {atoms.get_chemical_formula()}")
        print(f"Atoms: {len(atoms)}")

        _emit_resolved_config(resolved, config.output_dir / (config.job_name or "") if config.job_name else config.output_dir)
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

    config, resolved, _settings = _resolve_engine_config(args, "md")

    print_header()
    print(f"Reading structure from: {structure_path}")

    try:
        atoms = read(structure_path)
        print(f"System: {atoms.get_chemical_formula()}")
        print(f"Atoms: {len(atoms)}")

        _emit_resolved_config(resolved, config.output_dir / (config.job_name or "") if config.job_name else config.output_dir)
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

    config, resolved, _settings = _resolve_engine_config(args, "batch")
    pattern = resolved.run_options.get("pattern")

    print_header()

    try:
        _emit_resolved_config(resolved, config.output_dir / (config.job_name or "") if config.job_name else config.output_dir)
        engine = CalculationEngine.from_config(config)
        if "pattern" in resolved.run_options:
            files = sorted(input_dir.glob(pattern))
        else:
            # Match the formats supported by BatchRunner/read(), not only CIF.
            # The old CLI duplicated discovery but forgot XYZ/VASP/POSCAR, so
            # valid directories were reported as empty.
            files = sorted(
                {
                    *input_dir.glob("*.cif"),
                    *input_dir.glob("*.xyz"),
                    *input_dir.glob("*.vasp"),
                    *input_dir.glob("POSCAR*"),
                }
            )
        if not files:
            if "pattern" in resolved.run_options:
                print(f"No files matching '{pattern}' found in {input_dir}")
            else:
                print(f"No supported structure files found in {input_dir}")
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

def cmd_config(args: argparse.Namespace) -> int:
    """Execute 'config' subcommands (plan section 4.2 / 9 / 17.6)."""
    sub = args.config_command

    if sub == "paths":
        settings = _load_settings(args)
        print("settings.ini search paths (high priority first):")
        for path in settings.searched:
            marker = "  (loaded)" if path in settings.loaded_paths else ""
            print(f"  {path}{marker}")
        if not settings.loaded_paths:
            print("  (no settings.ini found; using built-in defaults)")
        return 0

    if sub == "init":
        if args.output:
            target = args.output
        elif args.user:
            target = "user"
        else:
            # default to project-level when neither flag is given
            target = "project"
        try:
            path = init_settings_file(target, force=args.force)
        except FileExistsError as exc:
            print(f"Error: {exc} (use --force to overwrite)")
            return 1
        print(f"Wrote settings.ini: {path}")
        return 0

    if sub == "validate":
        explicit = args.path
        settings = _load_settings(argparse.Namespace(settings=explicit)) if explicit else _load_settings(args)
        # Re-parse strictly to surface parser errors.
        import configparser  # noqa: PLC0415
        parser = configparser.ConfigParser(interpolation=None)
        paths_to_check = [Path(explicit)] if explicit else settings.loaded_paths
        if not paths_to_check:
            print("No settings.ini found to validate.")
            return 1
        errors: list[str] = []
        for p in paths_to_check:
            try:
                parser.read(p, encoding="utf-8")
            except (configparser.Error, OSError) as exc:
                errors.append(f"{p}: {exc}")
        # Schema-validate every section's keys.
        schema = get_schema()
        known = schema.known_names()
        for section in parser.sections():
            for key, _value in parser.items(section):
                if key.lower() not in known and not section.startswith(("engine:", "model:", "profile:")):
                    suggestion = schema.suggest(key)
                    hint = f" Did you mean {suggestion[0]!r}?" if suggestion else ""
                    errors.append(f"[{section}] unknown key {key!r}.{hint}")
        if errors:
            print("settings.ini validation errors:")
            for err in errors:
                print(f"  - {err}")
            return 1
        print(f"settings.ini OK ({len(paths_to_check)} file(s)).")
        return 0

    if sub == "explain":
        # Explain uses a representative md resolve; the source trace is what matters.
        settings = _load_settings(args)
        resolved = resolve_config(calc_type="md", settings=settings)
        print(resolved.explain(args.key))
        return 0

    if sub == "schema":
        schema = get_schema()
        print(f"{'name':24s} {'type':8s} {'scopes':24s} aliases")
        print("-" * 80)
        for spec in schema.specs:
            scopes = ",".join(sorted(spec.scopes))
            aliases = ",".join(sorted(spec.aliases)) if spec.aliases else ""
            print(f"{spec.name:24s} {spec.type.__name__:8s} {scopes:24s} {aliases}")
        return 0

    # sub == "show"
    settings = _load_settings(args)
    resolved = resolve_config(calc_type="md", settings=settings)
    print("Resolved configuration (calc_type=md, no CLI overrides):")
    print(f"  settings.ini: {resolved.settings_path or '(built-in defaults)'}")
    print(f"  model_type  : {resolved.model_type}")
    print(f"  task        : {resolved.task}")
    print(f"  device      : {resolved.device}")
    print(f"  inference_mode: {resolved.inference_mode}")
    print(f"  calculator_options: {resolved.calculator_options}")
    print(f"  run_options: {resolved.run_options}")
    return 0


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

    # --json output must be clean (no banner) for scripting. The `config`
    # subcommands also produce machine/script-oriented output, so suppress the
    # banner for them too.
    suppress_banner = (
        args.command == "setup"
        and getattr(args, "json", False)
    ) or args.command == "config"
    if not suppress_banner:
        print_header()

    # Dispatch to appropriate command handler
    commands = {
        "run": cmd_run,
        "sp": cmd_sp,
        "opt": cmd_opt,
        "md": cmd_md,
        "batch": cmd_batch,
        "config": cmd_config,
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
