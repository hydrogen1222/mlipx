"""Single source of truth for mlipx built-in defaults.

Per the improvement plan (section 4.5 / 17.7) there must be exactly *one* place
that defines built-in default values. Every other interface (CLI argparse
defaults, the INCAR ``template`` generator, the Python API, the engine fallbacks)
must read from here rather than re-hard-coding its own copy.

The values below deliberately match the historical mlipx behaviour so that
existing CLI/API/INCAR flows keep producing the same result. New vocabulary
introduced by the plan (``production_steps``, ``pre_relax_mode``,
``trajectory_interval`` ...) is registered in :mod:`mlipx.config.schema` as
recognised aliases, but is *not* wired to change scientific defaults until the
MD rewrite (Phase 3). This honours the rule "do not change scientific defaults
without recording it".
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

    from mlipx.config.incar import IncarConfig


# ---------------------------------------------------------------------------
# Built-in defaults, grouped by scope.
#
# Scopes mirror the settings.ini sections from the plan (section 4.4):
#   general / resources / batch / output / safety / sp / opt / md
# plus the per-engine calculator scopes ``calculator.uma`` / ``calculator.mace``.
#
# Keys are lowercase internal names. The schema (see schema.py) records the
# allowed type, scope and aliases for each key.
# ---------------------------------------------------------------------------
BUILTIN_DEFAULTS: dict[str, dict[str, Any]] = {
    "general": {
        "output_root": "./results",
        "log_level": "INFO",
        # Backward compatible: unknown keys warn (existing behaviour). Users opt
        # into hard errors via settings.ini ``strict_config = true``.
        "strict_config": False,
        "write_resolved_config": True,
        "write_manifest": True,
        # error = refuse to overwrite an existing output directory (safest).
        "output_collision": "error",
        # None = generate a seed and record it (see resolver).
        "default_seed": None,
    },
    "resources": {
        "max_gpu_jobs": 1,
        "max_cpu_jobs": 1,
        "gpu_devices": "0",
        "cpu_threads": 1,
        "device_lock_dir": "./.mlipx/locks",
    },
    "batch": {
        "mode": "serial",
        "continue_on_error": True,
        "resume": True,
        "retry_failed": 0,
        "state_flush_interval": 1,
        "preflight_all_jobs": True,
        "stop_when_disk_free_gb_below": 5,
        "stop_when_gpu_memory_free_mb_below": 1000,
    },
    "output": {
        "trajectory_interval": 20,
        "log_interval": 20,
        "checkpoint_interval": 1000,
        "write_xdatcar": True,
        "write_traj": True,
        "write_json": True,
        "compress_finished_trajectory": False,
    },
    "safety": {
        "abort_on_nan": True,
        "guard_interval": 10,
        "fmax_warn": 5.0,
        "fmax_abort": 20.0,
        "minimum_distance_warn": 0.8,
        "minimum_distance_abort": 0.5,
        "temperature_factor_warn": 2.0,
        "temperature_factor_abort": 3.0,
        "temperature_violation_count": 5,
        # Fail-closed pre-relaxation (plan section 2.4 / 13.2).
        "pre_relax_failure": "abort",
    },
    "sp": {
        "device": "cpu",
        "inference_mode": "default",
    },
    "opt": {
        "inference_mode": "default",
        "optimizer": "FIRE",
        "fmax": 0.05,
        "max_steps": 500,
        "cell_opt": False,
        "fix_symmetry": False,
    },
    "md": {
        "inference_mode": "turbo",
        "ensemble": "NVT",
        "temperature": 300.0,
        "timestep": 1.0,
        "steps": 1000,
        "friction": 0.001,
        "save_interval": 10,
        # ``pre_relax`` has NO built-in default here: its default is
        # ensemble-aware (off for NVE, on for NVT) and is applied by the
        # engine (CalculationEngine._create_runner). Putting a blanket
        # True here would override that logic for every config-layer path.
        "pre_relax_steps": 50,
        "pre_relax_fmax": 0.1,
    },
    "calculator": {
        "torch_num_threads": None,
        "activation_checkpointing": None,
    },
    # Engine-specific calculator options. These flow all the way to the
    # underlying ASE calculator (plan section 11.1 / 12).
    "calculator.uma": {
        "torch_num_threads": None,
        "activation_checkpointing": None,
    },
    "calculator.mace": {
        # MACE dtype default is calc-type aware (applied by the engine, not
        # the resolver): float32 for MD (speed), float64 for sp/opt (accuracy
        # for energy differences). Direct factory construction defaults to
        # float64; users override via --default-dtype / DEFAULT_DTYPE.
        "default_dtype": "float64",
        "head": None,
    },
    "calculator.dpa": {},
    "calculator.grace": {},
}


# Map calc_type -> default device, used by the CLI resolver when nothing else
# specifies a device. Matches the historical CLI defaults (sp/opt: cpu, md: cuda).
DEFAULT_DEVICE_BY_CALC_TYPE: dict[str, str] = {
    "sp": "cpu",
    "opt": "cpu",
    "md": "cuda",
    "batch": "cpu",
}


def get_default(calc_type: str | None, key: str, default: Any = None) -> Any:
    """Return a built-in default value for ``key`` scoped to ``calc_type``.

    Looks up ``<calc_type>`` scope first, then falls back to ``general`` /
    ``calculator``. ``calc_type`` may be ``None`` to only consult the
    non-calc-type scopes.
    """
    scopes: list[str] = []
    if calc_type:
        scopes.append(calc_type)
    scopes.extend(["general", "calculator"])
    for scope in scopes:
        block = BUILTIN_DEFAULTS.get(scope)
        if block and key in block:
            return block[key]
    return default


def build_incar_default(calc_type: str) -> str:
    """Build the INCAR template text for ``calc_type`` from the single source.

    Replaces the previously hand-written ``DEFAULT_SP_CONFIG`` /
    ``DEFAULT_OPT_CONFIG`` / ``DEFAULT_MD_CONFIG`` strings so there is only one
    place that owns the values (plan section 17.7).
    """
    calc_type = calc_type.lower()
    if calc_type not in {"sp", "opt", "md"}:
        raise ValueError(f"Unknown calculation type: {calc_type}")

    lines: list[str] = []
    lines.append("# mlipx Calculation Settings")
    lines.append(f"# Generated from the single source of defaults (calc_type={calc_type}).")
    lines.append("")
    lines.append(f"CALC_TYPE = {calc_type.upper()}")
    lines.append("TASK = omat")
    lines.append("")
    lines.append("# Model Settings")
    lines.append("MODEL_TYPE = UMA")
    lines.append("MODEL_PATH = uma-s-1.pt")
    device = DEFAULT_DEVICE_BY_CALC_TYPE.get(calc_type, "cpu")
    lines.append(f"DEVICE = {device}")
    if calc_type == "md":
        lines.append("INFERENCE_MODE = turbo")
    else:
        lines.append("INFERENCE_MODE = default")
    lines.append("")

    if calc_type == "opt":
        opt = BUILTIN_DEFAULTS["opt"]
        lines.append("# Optimization Settings")
        lines.append(f"OPT_ALGO = {opt['optimizer']}")
        lines.append(f"FMAX = {opt['fmax']}")
        lines.append(f"MAX_STEPS = {opt['max_steps']}")
        lines.append(f"CELL_OPT = {_bool(opt['cell_opt'])}")
        lines.append(f"FIX_SYMMETRY = {_bool(opt['fix_symmetry'])}")
        lines.append("")
    elif calc_type == "md":
        md = BUILTIN_DEFAULTS["md"]
        lines.append("# MD Settings")
        lines.append(f"MD_ENSEMBLE = {md['ensemble']}")
        lines.append(f"TEMPERATURE = {md['temperature']}")
        lines.append(f"TIMESTEP = {md['timestep']}")
        lines.append(f"STEPS = {md['steps']}")
        lines.append(f"FRICTION = {md['friction']}")
        lines.append(f"SAVE_INTERVAL = {md['save_interval']}")
        lines.append("")

    lines.append("# Output Control")
    lines.append("WRITE_FORCES = .TRUE.")
    lines.append("WRITE_STRESS = .TRUE.")
    lines.append("OUTPUT_FORMAT = VASP")
    lines.append("")
    return "".join(line + "\n" for line in lines)


def _bool(value: bool) -> str:
    return ".TRUE." if value else ".FALSE."


def get_default_config(calc_type: str) -> IncarConfig:
    """Return an :class:`IncarConfig` built from the single source of defaults."""
    from mlipx.config.incar import IncarConfig  # noqa: PLC0415

    return IncarConfig.from_string(build_incar_default(calc_type))
