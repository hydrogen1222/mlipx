# mlipx Configuration System (Phase 1)

> **Status:** Phase 1 · **Since:** mlipx v1.x

This document describes the mlipx configuration system introduced in Phase 1 of the
configuration refactor. It covers the unified defaults model, `settings.ini` file,
the layered config resolver, model aliases, reusable profiles, MACE backend support,
and the `mlipx config` CLI.

> **Environment convention:** use `uv run mlipx ...` for UMA and
> `.venv-mace/bin/mlipx ...` for MACE. The two engines cannot share a virtual
> environment.

---

## 1. Quick start

### 1.1 See your current config

```bash
mlipx config show
```

Output:
```
Resolved configuration (calc_type=md, no CLI overrides):
  settings.ini: (built-in defaults)
  model_type  : uma
  task        : omat
  device      : cuda
  inference_mode: turbo
  calculator_options: {}
  run_options: {ensemble: NVT, temperature: 300.0, steps: 1000, ...}
```

### 1.2 Create a settings.ini

```bash
mlipx config init --project    # writes ./settings.ini
mlipx config init --user       # writes ~/.config/mlipx/settings.ini
```

### 1.3 Use a model alias

```ini
; settings.ini
[model:mace_mpa0]
engine = mace
path = ./mace-mpa-0.model
task = bulk
dtype = float32
```

```bash
.venv-mace/bin/mlipx sp structure.cif --model-alias mace_mpa0 --dtype float64
```

The `--dtype float64` overrides the alias default of `float32`.
Check the resolved value with `mlipx config explain default_dtype`.

---

## 2. Where defaults come from (single source of truth)

All built-in defaults live in **one place** (`mlipx/config/defaults.py`).
Every interface — CLI, Python API, INCAR template — reads the same defaults.

| Scope           | Examples                              |
|-----------------|---------------------------------------|
| `general`       | `output_root`, `strict_config`        |
| `sp`            | `device=cpu`, `inference_mode=default`|
| `opt`           | `fmax=0.05`, `optimizer=FIRE`         |
| `md`            | `temperature=300`, `inference_mode=turbo` |
| `calculator.mace` | `default_dtype=float32`, `head=None` |

Run `mlipx config schema` to see every recognised option.

---

## 3. The layered resolution order

The resolver merges seven layers in increasing priority:

```
1. BUILT-IN DEFAULTS   (always present, lowest priority)
2. settings.ini        (user-level)
3. settings.ini        (project-level, overrides user)
4. Model alias         ([model:NAME] section)
5. Profile             ([profile:NAME] section)
6. INCAR / job config  (CALC_TYPE, FMAX … keys)
7. CLI / API kwargs    (--fmax 0.02, --dtype float64, etc.)
```

Each layer *adds or overrides* earlier values — it never removes them.
Every resolved key carries a **source trace** recording which layer provided it.

### 3.1 Example: trace a value

```bash
mlipx config explain temperature
# temperature = 300.0  (source: built-in defaults)

# After `--temp 500`:
mlipx --settings prod.ini sp s.xyz --model-alias mace_mpa0 --temp 500 \
  config explain temperature
# temperature = 500.0  (source: CLI)
```

### 3.2 Source tracking in code

```python
from mlipx.config import resolve_config

rc = resolve_config(calc_type="md", cli={"temperature": 500})
print(rc.sources["temperature"])
# ResolvedValue(value=500.0, source='CLI')
```

---

## 4. settings.ini format

### 4.1 Search order

```
1. --settings PATH        (CLI flag, highest priority)
2. MLIPX_SETTINGS env var
3. ./settings.ini         (project-level)
4. ~/.config/mlipx/settings.ini  (user-level)
```

Multiple files are *merged* — the project file can override the user file.
Use `mlipx config paths` to see every candidate path.

### 4.2 Section reference

```ini
; ── Global settings ─────────────────────────────────────
[general]
output_root = ./results
strict_config = false
write_resolved_config = true

; ── Per‑calculation defaults ─────────────────────────────
[sp]
device = cpu

[opt]
fmax = 0.05
optimizer = BFGS

[md]
temperature = 400
steps = 5000

; ── Model aliases ────────────────────────────────────────
[model:mace_mpa0]
engine = mace
path = ./models/mace-mpa-0.model
task = bulk
dtype = float64

[model:uma_prod]
engine = uma
path = /opt/models/uma-s-1.pt
task = omat

; ── Reusable profiles ────────────────────────────────────
[profile:gpu_2x]
device = cuda:1
inference_mode = turbo

[profile:high_precision]
fmax = 0.01
max_steps = 2000
```

---

## 5. Model aliases

A **model alias** is a named `[model:NAME]` section that bundles

| Key      | Required | Description                              |
|----------|----------|------------------------------------------|
| `engine` | yes      | `uma` / `mace` / `dpa` / `grace`         |
| `path`   | yes      | Model checkpoint (relative to settings.ini) |
| `task`   | yes      | Model head (`omat`, `bulk`, …)           |
| `dtype`  | no       | MACE dtype (`float32` or `float64`)     |
| `head`   | no       | MACE foundation-model head name          |

**Usage:**
```bash
# Via --model-alias flag
.venv-mace/bin/mlipx sp s.xyz --model-alias mace_mpa0

# Via --model (shorthand when the name matches an alias)
.venv-mace/bin/mlipx sp s.xyz --model mace_mpa0

# Via Python API
from mlipx.api import run_single_point
results = run_single_point("s.xyz", model_alias="mace_mpa0")
```

---

## 6. Profiles

A **profile** is a `[profile:NAME]` section that provides reusable overrides.
Profiles apply *after* the model alias layer and *before* CLI flags.

```ini
[profile:gpu_production]
device = cuda:1
inference_mode = turbo
max_steps = 5000
```

```bash
mlipx opt s.xyz --model-alias uma_prod --profile gpu_production
```

You can stack a profile with a model alias; the profile wins when both define
the same key.

---

## 7. MACE dtype and head

### 7.1 The full propagation chain

```
settings.ini [model:NAME] dtype
    ↓
resolve_config()  built-in layer → alias layer → CLI layer
    ↓
EngineConfig.calculator_options  {"default_dtype": "float64"}
    ↓
factory.build_calculator(..., default_dtype="float64")
    ↓
MACECalculatorWrapper(default_dtype="float64")
    ↓
mace-torch Calculator
```

### 7.2 Overriding dtype

```bash
# Override the alias default
.venv-mace/bin/mlipx sp s.xyz --model-alias mace_mpa0 --dtype float64

# Set it in a profile
[profile:mace_f64]
dtype = float64
```

### 7.3 Head (foundation model selector)

```bash
.venv-mace/bin/mlipx sp s.xyz --model-alias mace_fm --head "some_head_name"
```

The `--head` flag is passed to MACE's foundation model loading.

---

## 8. CLI reference

### 8.1 New global flags

| Flag                | Description                                   |
|---------------------|-----------------------------------------------|
| `--settings PATH`   | Explicit settings.ini (must precede subcommand) |

### 8.2 New per‑subcommand flags (sp / opt / md / batch)

| Flag                  | Description                        |
|-----------------------|------------------------------------|
| `--dtype {float32,float64}` | MACE dtype override           |
| `--head HEAD`         | MACE foundation head name          |
| `--model-alias NAME`  | Named model from settings.ini      |
| `--profile NAME`      | Reusable profile from settings.ini |

### 8.3 config subcommands

| Command                          | Description                             |
|----------------------------------|-----------------------------------------|
| `mlipx config show`              | Resolved configuration summary          |
| `mlipx config paths`             | Settings.ini search path list           |
| `mlipx config init [--project,--user]` | Create a settings.ini template  |
| `mlipx config validate [PATH]`   | Validate a settings.ini file            |
| `mlipx config explain KEY`       | Trace a parameter's source              |
| `mlipx config schema`            | List all recognised option keys         |

### 8.4 Backward compatibility

All legacy flags remain supported:
`--model-type`, `--task`, `--device`, `--fmax`, `--max-steps`,
`--optimizer`, `--cell-opt`, `--fix-symmetry`, `--ensemble`, `--temp`,
`--timestep`, `--steps`, `--friction`, `--save-interval`, `--pre-relax`,
`--continue-on-error`, `--pattern`, `--job-name`, `--output`.

The INCAR `run` flow also works as before.

---

## 9. Python API

All API functions now accept the new configuration parameters:

```python
from mlipx.api import run_single_point, run_optimization, run_md

# With model alias and dtype override
results = run_single_point(
    "structure.cif",
    model_alias="mace_mpa0",
    default_dtype="float64",
    job_name="sp_test",
)

# With settings.ini and profile
results = run_optimization(
    "structure.cif",
    model_alias="uma_prod",
    profile="gpu_production",
    settings_path="./prod-settings.ini",
    fmax=0.02,
)
```

The resolver merges API kwargs, the named profile, the model alias, and any
settings.ini files, so you only specify what you need to override.

---

## 10. Strict mode

Enable `strict_config = true` in settings.ini (or `--strict-config` on the
CLI while in strict mode) to turn unknown-key warnings into hard errors.
Without strict mode, every unrecognised key is reported as a warning with a
"Did you mean …?" suggestion using difflib fuzzy matching.

---

## 11. Resolved config artifact

When `write_resolved_config` is true (the default), every calculation writes a
`resolved_config.json` file into the output directory. This is useful for
auditing and debugging:

```json
{
  "calc_type": "md",
  "model_type": "mace",
  "device": "cuda",
  "inference_mode": "turbo",
  "calculator_options": {"default_dtype": "float64"},
  "run_options": {"temperature": 300.0, "steps": 1000, "seed": 312904314},
  "sources": {
    "default_dtype": {"value": "float64", "source": "CLI"},
    "temperature": {"value": 300.0, "source": "built-in defaults"},
    "seed": {"value": 312904314, "source": "auto-generated"}
  }
}
```

---

*Generated by Phase 1 config refactor. For details see the `mlipx_稳定性_默认配置_批量队列_MD改进方案.md` plan document.*
