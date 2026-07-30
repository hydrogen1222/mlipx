"""Configuration schema and strict validation (plan section 10).

The schema records, for every recognised option:

* its canonical (internal) name,
* the Python type,
* the scopes it belongs to (``sp``/``opt``/``md``/``batch``/``calculator``/
  ``general`` ...),
* any aliases (e.g. ``TEMPERATURE`` / ``--temp`` / ``temperature``),
* allowed ``choices`` and numeric bounds.

In strict mode (``strict_config = true``) an unknown key is a hard error and the
schema suggests the closest legal key via :func:`difflib.get_close_matches`, so a
typo such as ``DEFAULT_DTPE`` is reported as::

    Unknown key DEFAULT_DTPE in [job:01_sp]. Did you mean DEFAULT_DTYPE?

instead of being silently ignored.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


@dataclass(frozen=True)
class OptionSpec:
    """Specification of a single recognised option."""

    name: str
    type: type = str
    scopes: frozenset[str] = field(default_factory=frozenset)
    aliases: frozenset[str] = field(default_factory=frozenset)
    choices: tuple[Any, ...] | None = None
    minimum: float | None = None
    maximum: float | None = None
    default: Any | None = None
    description: str = ""

    def matches(self, key: str) -> bool:
        """True if ``key`` (case-insensitive) is this option's name or an alias."""
        k = key.lower()
        return k == self.name.lower() or k in {a.lower() for a in self.aliases}

    def coerce(self, value: Any) -> Any:
        """Coerce ``value`` to this spec's type, raising on failure."""
        if self.type is bool:
            return _coerce_bool(value)
        if self.type is int and not isinstance(value, bool):
            return int(value)
        if self.type is float and not isinstance(value, bool):
            return float(value)
        try:
            return self.type(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Option '{self.name}' expects {self.type.__name__}, got {value!r}"
            ) from exc

    def validate_value(self, value: Any) -> list[str]:
        """Return a list of human-readable validation errors for ``value``."""
        errors: list[str] = []
        if self.choices is not None and value not in self.choices:
            errors.append(
                f"Option '{self.name}' must be one of "
                f"{', '.join(repr(c) for c in self.choices)}, got {value!r}"
            )
        if self.minimum is not None and isinstance(value, (int, float)):
            if value < self.minimum:
                errors.append(
                    f"Option '{self.name}'={value} is below minimum {self.minimum}"
                )
        if self.maximum is not None and isinstance(value, (int, float)):
            if value > self.maximum:
                errors.append(
                    f"Option '{self.name}'={value} is above maximum {self.maximum}"
                )
        return errors


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "t", "yes", "y", "1", ".true.", ".t."}:
            return True
        if v in {"false", "f", "no", "n", "0", ".false.", ".f."}:
            return False
    raise ValueError(f"Cannot convert {value!r} to boolean")


# ---------------------------------------------------------------------------
# Registry of recognised options.
#
# Names are lowercase internal names. Aliases include the INCAR (UPPER) form and
# common CLI short forms so the same schema validates every interface.
# ---------------------------------------------------------------------------
_SPECS: list[OptionSpec] = [
    # --- global configuration ---
    OptionSpec(
        "strict_config", bool, frozenset({"general"}),
        aliases={"STRICT_CONFIG"},
        description="Treat unknown configuration keys as errors.",
    ),
    OptionSpec(
        "write_resolved_config", bool, frozenset({"general"}),
        aliases={"WRITE_RESOLVED_CONFIG"},
        description="Write resolved_config.json before a calculation.",
    ),
    OptionSpec(
        "fmax_abort", float, frozenset({"safety"}),
        aliases={"FMAX_ABORT"},
        minimum=0.0,
        description="MD large-force warning threshold in eV/Angstrom.",
    ),
    # --- model / device (calculator scope) ---
    OptionSpec(
        "model_type", str, frozenset({"calculator"}),
        aliases={"MODEL_TYPE", "model"},
        choices=("uma", "fairchem", "mace", "dpa", "grace"),
        description="MLIP engine.",
    ),
    OptionSpec(
        "model_path", str, frozenset({"calculator"}),
        aliases={"MODEL_PATH", "model"},
        description="Path to the model checkpoint/file.",
    ),
    OptionSpec(
        "task", str, frozenset({"calculator"}),
        aliases={"TASK"},
        description="UMA task (omat/omol/...) or bulk/molecule for other engines.",
    ),
    OptionSpec(
        "device", str, frozenset({"calculator", "general"}),
        aliases={"DEVICE"},
        description="Compute device: cpu, cuda, gpu or cuda:N.",
    ),
    OptionSpec(
        "inference_mode", str, frozenset({"calculator"}),
        aliases={"INFERENCE_MODE"},
        choices=("default", "turbo"),
        description="UMA inference mode (other engines ignore this).",
    ),
    # --- MACE calculator options (plan section 11.1 / 12) ---
    OptionSpec(
        "default_dtype", str, frozenset({"calculator.mace"}),
        aliases={"DEFAULT_DTYPE", "dtype"},
        choices=("float32", "float64"),
        description="MACE model dtype (float32 recommended for long MD).",
    ),
    OptionSpec(
        "head", str, frozenset({"calculator.mace", "calculator.dpa"}),
        aliases={"HEAD"},
        description="MACE or DeepMD model head/branch (multi-task models).",
    ),
    OptionSpec(
        "torch_num_threads", int, frozenset({"calculator"}),
        aliases={"TORCH_NUM_THREADS"},
        minimum=1,
        description="CPU thread count for torch.",
    ),
    OptionSpec(
        "activation_checkpointing", bool, frozenset({"calculator"}),
        aliases={"ACTIVATION_CHECKPOINTING"},
        description="GPU memory saving (UMA, overrides inference_mode preset).",
    ),
    # --- OPT run options ---
    OptionSpec(
        "optimizer", str, frozenset({"opt"}),
        aliases={"OPT_ALGO", "opt_algo"},
        choices=("FIRE", "BFGS", "LBFGS"),
        description="Geometry optimization algorithm.",
    ),
    OptionSpec(
        "fmax", float, frozenset({"opt"}),
        aliases={"FMAX"},
        minimum=0.0,
        description="Force convergence threshold in eV/Angstrom.",
    ),
    OptionSpec(
        "max_steps", int, frozenset({"opt"}),
        aliases={"MAX_STEPS"},
        minimum=0,
        description="Maximum optimization steps.",
    ),
    OptionSpec(
        "cell_opt", bool, frozenset({"opt"}),
        aliases={"CELL_OPT"},
        description="Optimize cell parameters (requires stress support).",
    ),
    OptionSpec(
        "fix_symmetry", bool, frozenset({"opt"}),
        aliases={"FIX_SYMMETRY"},
        description="Preserve crystal symmetry during optimization.",
    ),
    # --- MD run options ---
    OptionSpec(
        "ensemble", str, frozenset({"md"}),
        aliases={"MD_ENSEMBLE"},
        choices=("NVT", "NVE"),
        description="MD ensemble.",
    ),
    OptionSpec(
        "temperature", float, frozenset({"md"}),
        aliases={"TEMPERATURE", "temp"},
        minimum=0.0,
        description="Temperature in Kelvin.",
    ),
    OptionSpec(
        "timestep", float, frozenset({"md"}),
        aliases={"TIMESTEP", "timestep_fs"},
        minimum=0.0,
        description="Time step in femtoseconds.",
    ),
    OptionSpec(
        "steps", int, frozenset({"md"}),
        aliases={"STEPS", "production_steps", "PRODUCTION_STEPS"},
        minimum=0,
        description="Number of MD steps.",
    ),
    OptionSpec(
        "friction", float, frozenset({"md"}),
        aliases={"FRICTION", "friction_per_fs"},
        minimum=0.0,
        description="Friction coefficient for NVT (1/fs).",
    ),
    OptionSpec(
        "save_interval", int, frozenset({"md"}),
        aliases={"SAVE_INTERVAL", "trajectory_interval", "TRAJECTORY_INTERVAL"},
        minimum=1,
        description="Interval for saving trajectory frames.",
    ),
    OptionSpec(
        "pre_relax", bool, frozenset({"md"}),
        aliases={"PRE_RELAX"},
        description="Pre-relax the structure before MD (legacy).",
    ),
    OptionSpec(
        "pre_relax_steps", int, frozenset({"md"}),
        aliases={"PRE_RELAX_STEPS"},
        minimum=0,
        description="Maximum pre-relaxation steps (legacy).",
    ),
    OptionSpec(
        "pre_relax_fmax", float, frozenset({"md"}),
        aliases={"PRE_RELAX_FMAX"},
        minimum=0.0,
        description="Pre-relaxation force threshold (legacy).",
    ),
    # --- plan vocabulary recognised as aliases / future keys (Phase 3+) ---
    OptionSpec(
        "pre_relax_mode", str, frozenset({"md"}),
        aliases={"PRE_RELAX_MODE"},
        choices=("none", "positions", "cell"),
        default="none",
        description="Pre-relaxation mode (plan section 13.2; Phase 3).",
    ),
    OptionSpec(
        "velocity_policy", str, frozenset({"md"}),
        aliases={"VELOCITY_POLICY"},
        choices=("auto", "initialize", "preserve"),
        default="auto",
        description="Velocity initialisation policy (plan section 13.5; Phase 3).",
    ),
    OptionSpec(
        "equil_steps", int, frozenset({"md"}),
        aliases={"EQUIL_STEPS"},
        minimum=0,
        default=0,
        description="Equilibration steps (plan section 13.7; Phase 3).",
    ),
    OptionSpec(
        "seed", int, frozenset({"md", "general"}),
        aliases={"SEED", "default_seed", "DEFAULT_SEED"},
        minimum=0,
        description="Random seed for reproducible MD.",
    ),
    # --- batch run options ---
    OptionSpec(
        "sub_calc_type", str, frozenset({"batch"}),
        aliases={"sub-calc-type"},
        choices=("sp", "opt"),
        description="Sub-calculation type for a sweep batch (sp/opt).",
    ),
    # ``calc_type`` is a meta key: it selects the runner (sp/opt/md/batch).
    # The resolver pops it from every layer; it never becomes a run option.
    OptionSpec(
        "calc_type", str, frozenset({"meta"}),
        aliases={"CALC_TYPE"},
        choices=("sp", "opt", "md", "batch", "analyze"),
        description="Top-level calculation type (selects the runner).",
    ),
    OptionSpec(
        "pattern", str, frozenset({"batch"}),
        aliases={"PATTERN"},
        description="Glob pattern for batch structure discovery.",
    ),
    # --- INCAR output / meta keys (recognised so INCAR flows don't warn).
    # These are consumed by writers, not runners; they land in the settings bag.
    OptionSpec(
        "write_forces", bool, frozenset({"output"}),
        aliases={"WRITE_FORCES"},
        description="Write forces to OUTCAR.",
    ),
    OptionSpec(
        "write_stress", bool, frozenset({"output"}),
        aliases={"WRITE_STRESS"},
        description="Write stress to OUTCAR.",
    ),
    OptionSpec(
        "write_trajectory", bool, frozenset({"output"}),
        aliases={"WRITE_TRAJECTORY"},
        description="Write ASE trajectory.",
    ),
    OptionSpec(
        "write_json", bool, frozenset({"output"}),
        aliases={"WRITE_JSON"},
        description="Write JSON results.",
    ),
    OptionSpec(
        "output_format", str, frozenset({"output"}),
        aliases={"OUTPUT_FORMAT"},
        description="Output format (VASP).",
    ),
    OptionSpec(
        "job_name", str, frozenset({"meta"}),
        aliases={"JOB_NAME"},
        description="Optional job name.",
    ),
]



def get_schema() -> Schema:
    """Return the singleton :class:`Schema` instance."""
    global _SCHEMA  # noqa: PLW0603
    if _SCHEMA is None:
        _SCHEMA = Schema(_SPECS)
    return _SCHEMA


_SCHEMA: Schema | None = None


class Schema:
    """Registry of recognised :class:`OptionSpec` objects."""

    def __init__(self, specs: list[OptionSpec]):
        self._specs: list[OptionSpec] = list(specs)
        self._by_name: dict[str, OptionSpec] = {}
        for spec in self._specs:
            self._by_name[spec.name.lower()] = spec
            for alias in spec.aliases:
                self._by_name[alias.lower()] = spec

    @property
    def specs(self) -> list[OptionSpec]:
        return list(self._specs)

    def known_names(self) -> set[str]:
        """All recognised keys (canonical names + aliases), lowercased."""
        return set(self._by_name.keys())

    def canonical_names(self) -> set[str]:
        """Only canonical option names (lowercased)."""
        return {s.name.lower() for s in self._specs}

    def resolve(self, key: str) -> OptionSpec | None:
        """Return the spec matching ``key`` (case-insensitive) or ``None``."""
        return self._by_name.get(key.lower())

    def canonical_name(self, key: str) -> str | None:
        """Return the canonical name for ``key`` or ``None`` if unknown."""
        spec = self.resolve(key)
        return spec.name if spec is not None else None

    def suggest(self, key: str, n: int = 3) -> list[str]:
        """Return the closest recognised keys to ``key`` (for typo hints)."""
        candidates = sorted(self._by_name.keys())
        return difflib.get_close_matches(key.lower(), candidates, n=n, cutoff=0.6)

    def validate_dict(
        self,
        values: dict[str, Any],
        *,
        strict: bool,
        context: str = "",
        known_extra: set[str] | None = None,
    ) -> list[str]:
        """Validate a flat ``{key: value}`` mapping.

        Args:
            values: Options to validate (keys may be canonical or alias forms).
            strict: When True, unknown keys are returned as errors (with a typo
                suggestion); when False, unknown keys are ignored by this method
                and the caller may choose to warn.
            context: Human-readable location included in error messages, e.g.
                ``"[job:01_sp]"`` or ``"CLI --opt"``.
            known_extra: Additional keys to treat as known (e.g. keys consumed
                by the resolver itself rather than a runner/calculator).

        Returns:
            A list of error messages. Empty when everything is valid. When
            ``strict`` is False, unknown-key errors are *not* included so the
            caller can emit warnings instead.
        """
        errors: list[str] = []
        known_extra = known_extra or set()
        loc = f" in {context}" if context else ""
        for key, value in values.items():
            spec = self.resolve(key)
            if spec is None:
                if key.lower() in {k.lower() for k in known_extra}:
                    continue
                if strict:
                    suggestion = self.suggest(key)
                    hint = (
                        f" Did you mean {suggestion[0]!r}?"
                        if suggestion
                        else ""
                    )
                    errors.append(f"Unknown key {key!r}{loc}.{hint}")
                continue
            # Coerce + validate the value.
            try:
                coerced = spec.coerce(value)
            except ValueError as exc:
                errors.append(str(exc))
                continue
            errors.extend(spec.validate_value(coerced))
        return errors
