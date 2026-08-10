"""Layered configuration resolver (plan section 4.3 / 17.6 / 17.8).

The resolver merges, from lowest to highest priority::

    built-in defaults  <  settings.ini  <  model alias  <  profile
                        <  INCAR / job   <  CLI / kwargs

and produces:

* the model-level fields (``model_type`` / ``model_path`` / ``task`` / ``device``
  / ``inference_mode``),
* ``calculator_options`` (engine-specific, e.g. MACE ``default_dtype``/``head``),
* ``run_options`` (calc-type-specific, e.g. ``fmax`` / ``temperature``),
* a per-key ``source`` trace so ``mlipx config explain TEMPERATURE`` can say
  *why* a parameter ended up with a given value.

The output is engine-agnostic: callers (CLI / API / engine) consume
``calculator_options`` and ``run_options`` directly. Path resolution follows the
plan (section 18.1): model paths are resolved relative to the loaded
settings.ini, structure paths relative to the BATCH file, output paths relative
to the current working directory.
"""

from __future__ import annotations

import random
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from mlipx.config.aliases import (
    resolve_model_alias,
    resolve_profile,
)
from mlipx.config.defaults import (
    BUILTIN_DEFAULTS,
    DEFAULT_DEVICE_BY_CALC_TYPE,
)
from mlipx.config.schema import Schema, get_schema

if TYPE_CHECKING:
    from typing import Any

    from mlipx.config.aliases import ModelAlias, Profile
    from mlipx.config.settings import MlipxSettings


@dataclass
class ResolvedValue:
    """A single resolved parameter with provenance."""

    value: Any
    source: str

    def __repr__(self) -> str:
        return f"ResolvedValue({self.value!r}, source={self.source!r})"


@dataclass
class ResolvedConfig:
    """Fully resolved configuration produced by :func:`resolve_config`."""

    calc_type: str
    model_type: str
    model_path: str
    task: str
    device: str
    inference_mode: str
    calculator_options: dict[str, Any] = field(default_factory=dict)
    run_options: dict[str, Any] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)
    sources: dict[str, ResolvedValue] = field(default_factory=dict)
    strict: bool = False
    settings_path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Flat dict suitable for ``resolved_config.json`` output."""
        return {
            "calc_type": self.calc_type,
            "model_type": self.model_type,
            "model_path": self.model_path,
            "task": self.task,
            "device": self.device,
            "inference_mode": self.inference_mode,
            "calculator_options": dict(self.calculator_options),
            "run_options": dict(self.run_options),
            "settings": dict(self.settings),
            "sources": {k: {"value": v.value, "source": v.source} for k, v in self.sources.items()},
            "strict": self.strict,
            "settings_path": self.settings_path,
        }

    def explain(self, key: str) -> str:
        """Human-readable explanation of why ``key`` has its resolved value."""
        canon = get_schema().canonical_name(key)
        lookup = canon or key
        rv = self.sources.get(lookup)
        if rv is None:
            return f"{key}: not set"
        return f"{key} = {rv.value!r}  (source: {rv.source})"


# Keys that are model-level (not calculator/run options) and are consumed by
# the engine directly rather than passed through **calculator_options.
_MODEL_KEYS = {"model_type", "model_path", "task", "device", "inference_mode"}

# Calculator-option keys, per engine. Anything not here and not a model key is
# treated as a run option (scoped to the calc_type by the schema).
_CALCULATOR_KEYS_BY_ENGINE: dict[str, set[str]] = {
    "uma": {"inference_mode", "torch_num_threads", "activation_checkpointing"},
    "fairchem": {"inference_mode", "torch_num_threads", "activation_checkpointing"},
    "mace": {"default_dtype", "head"},
    "dpa": {"head"},
    "grace": {"gpu_memory_growth", "gpu_memory_limit_mb"},
}


def _is_calculator_key(key: str, model_type: str) -> bool:
    engine = (model_type or "uma").lower()
    return key in _CALCULATOR_KEYS_BY_ENGINE.get(engine, set())


def _settings_layer(
    settings: MlipxSettings | None, calc_type: str | None = None
) -> dict[str, Any]:
    """Flatten the relevant settings.ini sections into a single dict."""
    if settings is None:
        return {}
    layer: dict[str, Any] = {}
    # Global sections feed the settings bag. Calculation sections are scoped:
    # loading [opt] into an MD run (and vice versa) previously leaked irrelevant
    # keys into run_options and produced misleading resolved configurations.
    # ``sp`` was historically missing here, so a ``[sp]`` section in
    # settings.ini (device / inference_mode) was silently ignored.
    for section_name in ("general", "resources", "output", "safety"):
        layer.update(settings.section(section_name))
    if calc_type in {"sp", "md", "opt", "batch"}:
        layer.update(settings.section(calc_type))
    return layer


def _canonicalize_layer(
    raw: dict[str, Any], schema: Schema
) -> dict[str, Any]:
    """Map alias keys to canonical names and coerce types where possible."""
    out: dict[str, Any] = {}
    for key, value in raw.items():
        canon = schema.canonical_name(key)
        name = canon if canon is not None else key.lower()
        # Avoid clobbering a canonical key set by an earlier alias in the same
        # layer only when the value is None/empty (e.g. default_seed="").
        if name in out and (value is None or value == ""):
            continue
        out[name] = value
    return out


def _merge_layer(
    base: dict[str, ResolvedValue],
    layer: dict[str, Any],
    source: str,
) -> None:
    """Merge ``layer`` into ``base`` recording ``source`` for each new key."""
    for key, value in layer.items():
        if value is None or value == "":
            continue
        base[key] = ResolvedValue(value=value, source=source)


def resolve_config(
    *,
    calc_type: str,
    settings: MlipxSettings | None = None,
    model_aliases: dict[str, ModelAlias] | None = None,
    profiles: dict[str, Profile] | None = None,
    model_alias_name: str | None = None,
    profile_name: str | None = None,
    incar: dict[str, Any] | None = None,
    cli: dict[str, Any] | None = None,
    schema: Schema | None = None,
) -> ResolvedConfig:
    """Resolve a full configuration from all layers.

    Args:
        calc_type: ``sp``/``opt``/``md``/``batch``.
        settings: Loaded :class:`MlipxSettings` (may be None).
        model_aliases/profiles: Parsed alias/profile maps (usually extracted
            from ``settings``). If None they are derived from ``settings``.
        model_alias_name: Name of a ``[model:...]`` alias to apply.
        profile_name: Name of a ``[profile:...]`` profile to apply.
        incar: INCAR/job-level overrides (already parsed, keys may be aliases).
        cli: Highest-priority overrides (CLI args or API kwargs).
        schema: Optional schema override (defaults to the global one).

    Returns:
        A :class:`ResolvedConfig` with calculator/run options split out and a
        per-key ``sources`` trace.
    """
    schema = schema or get_schema()
    calc_type = calc_type.lower()
    model_aliases = model_aliases if model_aliases is not None else _aliases_from_settings(settings)
    profiles = profiles if profiles is not None else _profiles_from_settings(settings)

    sources: dict[str, ResolvedValue] = {}

    # Layer 1: built-in defaults.
    defaults_layer: dict[str, Any] = {}
    defaults_layer.update(BUILTIN_DEFAULTS.get("general", {}))
    defaults_layer.update(BUILTIN_DEFAULTS.get(calc_type, {}))
    defaults_layer.update(BUILTIN_DEFAULTS.get("calculator", {}))
    if calc_type == "md":
        # Implemented MD guards are resolved like user [safety] overrides, so
        # EngineConfig.from_resolved can route them to MDRunner.
        defaults_layer.update(BUILTIN_DEFAULTS.get("safety", {}))
    defaults_layer["device"] = DEFAULT_DEVICE_BY_CALC_TYPE.get(calc_type, "cpu")
    _merge_layer(sources, _canonicalize_layer(defaults_layer, schema), "built-in defaults")

    # Canonicalise the higher layers once. Besides avoiding repeated parsing,
    # this lets us select the right [engine:<name>] section even when the
    # engine is chosen by a CLI flag, profile, INCAR or model alias.
    alias_layer = _canonicalize_layer(
        resolve_model_alias(model_alias_name, model_aliases), schema
    )
    profile_layer = _canonicalize_layer(
        resolve_profile(profile_name, profiles), schema
    )
    profile_layer.pop("calc_type", None)
    incar_layer = _canonicalize_layer(incar or {}, schema)
    incar_layer.pop("calc_type", None)
    cli_layer = _canonicalize_layer(cli or {}, schema)
    cli_layer.pop("calc_type", None)

    # Layer 2: settings.ini (calculation section + selected engine defaults).
    settings_layer = _canonicalize_layer(
        _settings_layer(settings, calc_type), schema
    )
    if settings is not None:
        engine_name = "uma"
        for candidate in (
            settings_layer,
            alias_layer,
            profile_layer,
            incar_layer,
            cli_layer,
        ):
            if "model_type" in candidate:
                engine_name = str(candidate["model_type"]).lower()
        engine_defaults = settings.engine_section(str(engine_name))
        settings_layer.update(_canonicalize_layer(engine_defaults, schema))
    _merge_layer(sources, settings_layer, "settings.ini")

    # Layer 3: model alias.
    _merge_layer(sources, alias_layer, f"model alias {model_alias_name!r}")

    # Layer 4: profile.
    # ``calc_type`` is authoritative from the caller (CLI subcommand / API);
    # a profile's calc_type is declarative only and is not allowed to override
    # it. (Plan section 4.3: CLI > profile.)
    profile_layer.pop("calc_type", None)
    _merge_layer(sources, profile_layer, f"profile {profile_name!r}")

    # Layer 5: INCAR / job.
    _merge_layer(sources, incar_layer, "INCAR/job")

    # Layer 6: CLI / kwargs (highest priority).
    _merge_layer(sources, cli_layer, "CLI")

    # ---- Finalise model-level fields. ----
    model_type = str(sources.get("model_type", ResolvedValue("uma", "built-in defaults")).value).lower()
    model_path = str(sources.get("model_path", ResolvedValue("", "built-in defaults")).value)
    task = str(sources.get("task", ResolvedValue("omat" if model_type in {"uma", "fairchem"} else "bulk", "built-in defaults")).value).lower()
    device = str(sources.get("device", ResolvedValue(DEFAULT_DEVICE_BY_CALC_TYPE.get(calc_type, "cpu"), "built-in defaults")).value)
    # inference_mode defaults: 'turbo' for MD (historical behaviour),
    # 'default' for everything else.
    _default_inference = "turbo" if calc_type == "md" else "default"
    inference_mode = str(
        sources.get("inference_mode", ResolvedValue(_default_inference, "built-in defaults")).value
)
    # Resolve model path relative to settings.ini when it is not absolute.
    settings_path = str(settings.path) if settings and settings.path else None
    if settings_path and model_path and not Path(model_path).is_absolute():
        candidate = (Path(settings.path).parent / model_path).resolve()  # type: ignore[union-attr]
        if candidate.exists():
            model_path = str(candidate)

    # ---- Split calculator vs run options. ----
    calculator_options: dict[str, Any] = {}
    run_options: dict[str, Any] = {}
    settings_bag: dict[str, Any] = {}
    settings_scope_keys = (
        set(BUILTIN_DEFAULTS.get("general", {}))
        | set(BUILTIN_DEFAULTS.get("resources", {}))
        | set(BUILTIN_DEFAULTS.get("batch", {}))
        | set(BUILTIN_DEFAULTS.get("output", {}))
        | set(BUILTIN_DEFAULTS.get("safety", {}))
    )

    for key, rv in sources.items():
        if key in _MODEL_KEYS:
            continue
        if key in settings_scope_keys:
            settings_bag[key] = rv.value
            continue
        if _is_calculator_key(key, model_type):
            calculator_options[key] = rv.value
            continue
        # Classify by schema scope: keys scoped to a calc type become run
        # options; everything else (output/meta/general/...) lands in the
        # settings bag so it neither pollutes run_options nor triggers
        # unknown-key warnings.
        spec = schema.resolve(key)
        calc_scopes = {"sp", "opt", "md", "batch"}
        if spec is not None and (set(spec.scopes) & calc_scopes):
            run_options[key] = rv.value
        else:
            settings_bag[key] = rv.value
    # Seed handling: auto-generate when not set, and record it.
    if "seed" not in sources and calc_type == "md":
        seed = random.randint(0, 2**31 - 1)
        sources["seed"] = ResolvedValue(seed, "auto-generated")
        run_options["seed"] = seed

    strict = bool(settings_bag.get("strict_config", False)) if settings_bag else bool(
        _settings_layer(settings, calc_type).get("strict_config", False)
    )

    resolved = ResolvedConfig(
        calc_type=calc_type,
        model_type=model_type,
        model_path=model_path,
        task=task,
        device=device,
        inference_mode=inference_mode,
        calculator_options=calculator_options,
        run_options=run_options,
        settings=settings_bag,
        sources=sources,
        strict=strict,
        settings_path=settings_path,
    )

    _validate_resolved(resolved, schema)
    return resolved


def _validate_resolved(resolved: ResolvedConfig, schema: Schema) -> None:
    """Run schema validation; warn or raise depending on ``strict``."""
    output_format = resolved.settings.get("output_format")
    if output_format is not None and str(output_format).lower() != "vasp":
        raise ValueError(
            f"Unsupported OUTPUT_FORMAT {output_format!r}; only VASP is "
            "currently implemented."
        )

    all_opts: dict[str, Any] = {}
    all_opts.update(resolved.calculator_options)
    all_opts.update(resolved.run_options)

    errors = schema.validate_dict(
        all_opts,
        strict=resolved.strict,
        context=f"calc_type={resolved.calc_type}",
        known_extra=set(resolved.settings.keys()),
    )
    if resolved.strict and errors:
        raise ValueError("Strict config validation failed:\n  - " + "\n  - ".join(errors))
    if errors and not resolved.strict:
        for err in errors:
            warnings.warn(err, stacklevel=2)


def _aliases_from_settings(settings: MlipxSettings | None) -> dict[str, ModelAlias]:
    if settings is None:
        return {}
    from mlipx.config.aliases import parse_model_aliases  # noqa: PLC0415

    return parse_model_aliases(settings.parser)


def _profiles_from_settings(settings: MlipxSettings | None) -> dict[str, Profile]:
    if settings is None:
        return {}
    from mlipx.config.aliases import parse_profiles  # noqa: PLC0415

    return parse_profiles(settings.parser)
