# mlipx — User Manual

> **MLIP eXtended**
> A multi-engine machine-learning interatomic-potential interface for VASP-oriented workflows. Supports UMA (FAIRChem, default), MACE, DPA (DeepMD-kit), and GRACE behind one interface.

---

## Table of Contents

- [1. Introduction](#1-introduction)
- [2. Installation](#2-installation)
- [3. Quick Start](#3-quick-start)
- [4. Architecture Overview](#4-architecture-overview)
- [Multi-Engine Guide](#multi-engine-guide)
- [5. Calculation Types](#5-calculation-types)
  - [5.1 Single Point (SP)](#51-single-point-sp)
  - [5.2 Geometry Optimization (OPT)](#52-geometry-optimization-opt)
  - [5.3 Molecular Dynamics (MD)](#53-molecular-dynamics-md)
  - [5.4 Built-in Trajectory Analysis](#54-built-in-trajectory-analysis-analysis-v2)
  - [5.5 Batch Processing](#55-batch-processing)
- [6. User Interfaces](#6-user-interfaces)
  - [6.1 CLI — Command Line Interface](#61-cli--command-line-interface)
  - [6.2 TUI — Terminal User Interface](#62-tui--terminal-user-interface)
  - [6.3 Python API](#63-python-api)
- [7. INCAR Configuration Reference](#7-incar-configuration-reference)
- [8. Output Files Reference](#8-output-files-reference)
- [9. Task Types Reference](#9-task-types-reference)
- [10. Background Jobs](#10-background-jobs)
- [11. Resource Control](#11-resource-control)
- [12. Troubleshooting & FAQ](#12-troubleshooting--faq)
- [13. Performance Guide](#13-performance-guide)
- [14. Examples](#14-examples)
- [15. License](#15-license)

---

## 1. Introduction

mlipx (MLIP eXtended) is a multi-engine machine-learning interatomic potential (MLIP) computation tool that provides a VASP-like user experience. It supports multiple MLIP backends behind one unified interface:

Analysis v2 provides validated, calculator-independent trajectory analysis for
solid-state ion transport. It uses explicit time/PBC/phase/unit contracts and
does not import the archived Analysis v1. See
[Section 5.4](#54-built-in-trajectory-analysis-analysis-v2) for usage; the
focused documents retain deeper algorithm definitions and references.

| Engine | `MODEL_TYPE` | Backend package |
|-------|--------------|-----------------|
| **UMA (FAIRChem)** (default) | `uma` | `fairchem-core` |
| MACE | `mace` | `mace-torch` |
| DPA (DeepMD-kit) | `dpa` | `deepmd-kit` |
| GRACE | `grace` | `tensorpotential` |

All engines plug in through a unified ASE Calculator interface, so the higher-level logic (single point, optimization, MD, batch, output) is completely engine-agnostic. The default is UMA; existing workflows keep working unchanged.

**What mlipx does:**

- Compute energy, forces, and stress for crystal structures and molecules
- Optimize atomic positions and cell parameters (geometry relaxation)
- Run molecular dynamics simulations (NVT / NVE ensembles)
- Validate and analyze MD trajectories, including thermo, RDF, MSD, transport, and VACF
- Process hundreds of structures in batch mode
- Write VASP-syntax CONTCAR/XDATCAR, a documented VASP-like OUTCAR, and OSZICAR

**How mlipx works:**

Unlike VASP, which solves the Kohn-Sham equations self-consistently, mlipx uses a pre-trained neural network to predict energies and forces in a single forward pass. There are no electronic steps, no SCF cycles, and no k-points. The cost scales roughly linearly with the number of atoms.

```
                          ┌──────────────────────┐
  structure.cif  ────────▶│  MLIP Neural Network  │───────▶  energy, forces, stress
  (atomic positions)      │  (UMA/MACE/DPA/GRACE) │         (single forward pass)
                          └──────────────────────┘
```

**Key features at a glance:**

| Feature | Description |
|---------|-------------|
| Multi-engine | UMA / MACE / DPA / GRACE via one interface; switch with `--model-type` |
| CLI mode | Full command-line interface (sp/opt/md/batch/run/config/queue/...) |
| TUI mode | Interactive terminal UI with live progress |
| Python API | Programmatic access for scripting and workflows |
| Trajectory analysis | Backend-independent validation, thermo, RDF, MSD, transport, and VACF |
| Background jobs | Submit, detach, re-attach, and kill long-running calculations |
| Batch processing | Process many structures sequentially with one model load |
| CPU & CUDA | Runs on CPU or GPU, auto-detected |
| VASP ecosystem output | Syntax-compatible CONTCAR/XDATCAR, VASP-like OUTCAR, OSZICAR |
| Cross-platform | Windows, Linux, macOS |

---

## 2. Installation

### 2.1 Start here: four engines need four environments

Do not install all four backends into one Python environment. The reliable,
tested layout is:

| Engine | Command prefix | Environment | Why it is separate |
|---|---|---|---|
| UMA | `uv run mlipx` | `.venv` | Default project environment with fairchem-core |
| MACE | `.venv-mace/bin/mlipx` | `.venv-mace` | MACE and UMA require different e3nn versions |
| DPA | `.venv-dpa/bin/mlipx` | `.venv-dpa` | DeepMD requires PyTorch 2.10 with the CXX11 ABI |
| GRACE | `.venv-grace/bin/mlipx` | `.venv-grace` | TensorFlow CUDA/cuDNN must not replace PyTorch libraries |

The versions below are the repository's **known-good profile as of
2026-08-14**, not a promise that one "latest" stack fits every GPU. Before
changing them, check the upstream
[uv installation guide](https://docs.astral.sh/uv/getting-started/installation/),
[DeePMD installation guide](https://docs.deepmodeling.com/projects/deepmd/en/latest/getting-started/install.html),
and [GRACE/TensorPotential installation guide](https://gracemaker.readthedocs.io/en/latest/gracemaker/install/),
then rerun doctor's runtime and model probes.

An “environment” is only an isolated directory of Python packages, not another
copy of the source code. All four share this repository, structures, and model
files. Run the commands below from the repository root. If you need only one
optional engine, install only its section.

#### Step 0: install uv and enter the repository

Skip the uv installation if `uv --version` already works:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/hydrogen1222/mlipx.git
cd mlipx
```

#### Environment 1: UMA (default; recommended first)

```bash
uv sync --frozen
uv run mlipx doctor --engine uma --device auto
```

Always use the `uv run mlipx` prefix for UMA:

```bash
uv run mlipx sp structure.vasp \
  --model models/uma/uma-s-1.pt --model-type uma \
  --task omat --device cuda:0
```

#### Environment 2: MACE

```bash
uv venv --python 3.12 .venv-mace
uv pip install --no-config --python .venv-mace/bin/python \
  "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124
uv pip install --no-config --python .venv-mace/bin/python \
  -e ./mlipx "e3nn==0.4.4" "mace-torch==0.3.16"
.venv-mace/bin/mlipx doctor --engine mace --device auto
```

Example:

```bash
.venv-mace/bin/mlipx sp structure.vasp \
  --model models/mace/mace-mpa-0-medium.model --model-type mace \
  --task bulk --head default --device cuda:0
```

#### Environment 3: DPA / DeepMD (GPU recommended)

```bash
uv venv --python 3.12 .venv-dpa
uv pip install --no-config --python .venv-dpa/bin/python \
  "torch==2.10.0+cu126" --index-url https://download.pytorch.org/whl/cu126
uv pip install --no-config --python .venv-dpa/bin/python \
  -e ./mlipx "deepmd-kit[torch]==3.1.3"
.venv-dpa/bin/mlipx doctor --engine dpa --device auto
```

Do not omit the appropriate model branch for LGPS and related solid
electrolytes:

```bash
.venv-dpa/bin/mlipx sp structure.vasp \
  --model models/dpa/DPA-3.2-5M.pt --model-type dpa \
  --task bulk --head Domains_SSE_PBE --device cuda:0
```

On a machine without an NVIDIA GPU, only the first Torch command changes:

```bash
uv pip install --no-config --python .venv-dpa/bin/python \
  "torch==2.10.0" --index-url https://download.pytorch.org/whl/cpu
```

#### Environment 4: GRACE (GPU)

```bash
uv venv --python 3.12 .venv-grace
uv pip install --no-config --python .venv-grace/bin/python \
  -e ./mlipx "tensorflow[and-cuda]==2.20.0" "tensorpotential==0.6.0"

# Use the validated cuDNN version on V100/Volta
uv pip install --no-config --python .venv-grace/bin/python \
  "nvidia-cudnn-cu12==9.3.0.75"
.venv-grace/bin/python -c \
  "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"
.venv-grace/bin/mlipx doctor --engine grace --device auto
```

For GRACE, `--model` must point to the complete SavedModel directory:

```bash
.venv-grace/bin/mlipx sp structure.vasp \
  --model models/grace/GRACE-2L-SMAX-large --model-type grace \
  --task bulk --device cuda:0
```

#### How do I know that I selected the right environment?

Check the beginning of the command:

```text
UMA    -> uv run mlipx ...
MACE   -> .venv-mace/bin/mlipx ...
DPA    -> .venv-dpa/bin/mlipx ...
GRACE  -> .venv-grace/bin/mlipx ...
```

The TUI follows the same rule. For example, launch MACE with
`.venv-mace/bin/mlipx tui`; changing `MODEL_TYPE` inside the TUI does not
switch Python environments. Models are not downloaded with the environments;
place them under `models/` or another known path yourself.

### 2.2 Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.10–3.12 | 3.12 |
| RAM | 8 GB | 32 GB |
| Disk | 15 GB (one environment and a small model) | 60+ GB (all four environments and several models) |
| GPU (optional) | NVIDIA driver compatible with the selected PyTorch/TensorFlow wheel | Validated CUDA 12.x combination, 8+ GB VRAM |

### 2.3 UMA environment details

Although `fairchem-core` is published on
[PyPI](https://pypi.org/project/fairchem-core/), the root workspace explicitly
uses the local member at `packages/fairchem-core/`. Therefore Section 2.1's
`uv sync --frozen` installs the lockfile/local-source combination. Do not
overwrite `.venv` with a bare `uv pip install`. Run `uv run mlipx setup` when
you need the GPU architecture and recommended-wheel report.

### 2.4 CUDA GPU vs CPU

The root workspace installs and pins PyTorch through `uv.lock`. Do not replace
it with a bare `uv pip install torch`.

| Scenario | PyTorch | mlipx device flag |
|----------|---------|--------------------|
| **CUDA GPU machine** | `fairchem-core` installed in CUDA Python env | `--device cuda` |
| **CPU-only machine** | `fairchem-core` installed in standard Python env | `--device cpu` (default) |

**Verify CUDA availability after installation:**

```bash
uv run python -c "import torch; print('CUDA available' if torch.cuda.is_available() else 'CPU only')"
```

**If CUDA is not available but you have a GPU:**
- Run `uv run mlipx setup` for the GPU architecture recommendation
- Verify that the recommended CUDA wheel, rather than a CPU wheel, is installed

### 2.4.1 Verified Setups

The following configuration has been tested and confirmed working:

| Component | Detail |
|-----------|--------|
| **GPU** | NVIDIA Tesla V100-SXM2-16GB |
| **Architecture** | Volta, Compute Capability 7.0 (sm_70) |
| **VRAM** | 16 GB |
| **Driver** | 580.173.02 |
| **UMA** | PyTorch 2.6.0+cu124 |
| **MACE** | mace-torch 0.3.16, PyTorch 2.6.0+cu124 |
| **DPA** | deepmd-kit 3.1.3, PyTorch 2.10.0+cu126 |
| **GRACE** | TensorFlow 2.20.0, cuDNN 9.3.0.75 |
| **Python** | 3.12.13 |
| **OS** | Linux |
| **Install note** | Models are not downloaded automatically by `uv sync` |

*This is a known-good Linux/NVIDIA GPU reference rather than a minimum
requirement. Lower-spec GPUs (for example GTX 10-series Pascal, sm_61) and
CPU-only systems can also work, but their installation commands may differ.*

### 2.5 Command Prefixes and Activation

These two forms are equivalent for UMA's `.venv` only:

```bash
# Method A: uv run (recommended — auto-detects .venv, works everywhere)
uv run mlipx --help
uv run mlipx tui
uv run mlipx sp structure.cif --model uma-s-1.pt

# Method B: Activate venv first
source .venv/bin/activate      # Linux/Mac
# .venv\Scripts\activate       # Windows
mlipx --help
mlipx tui
```

For another backend, replace `.venv` with `.venv-mace`, `.venv-dpa`, or
`.venv-grace`. Calling `ENV/bin/mlipx` explicitly is the least ambiguous form
for scripts and the TUI.

### 2.6 Model Checkpoint

**UMA (default engine):** Checkpoints are hosted in the gated
[Hugging Face `facebook/UMA`](https://huggingface.co/facebook/UMA) repository.
They are not on GitHub and are not bundled with `fairchem-core` or mlipx:

```bash
# 1. Request access in the browser and create a token that can read gated repos
hf auth login

# 2. After approval, download the current recommended small checkpoint
hf download facebook/UMA checkpoints/uma-s-1p2p1.pt --local-dir models/uma
```

Access is manually approved by Hugging Face/Meta. The form requires a full
legal name, date of birth, country and affiliation, plus acceptance of the
license. The official model page currently says UMA is unavailable in China,
Russia, Belarus and comprehensively sanctioned jurisdictions. Check the model
page before applying because these conditions may change.

> The official repository archives `uma-s-1.pt` and notes a known extensivity
> bug. Do not use it as the recommended starting point for new runs. mlipx takes
> any local checkpoint path, for example
> `--model models/uma/checkpoints/uma-s-1p2p1.pt`.

**Other engines (optional):** each project publishes its own model. MACE uses
`.model`/`.pt`; DPA requires a frozen/exported `.pt`/`.pth`, not a training
checkpoint; GRACE requires the complete TensorFlow SavedModel directory.
Section 2.1 remains the single installation authority.

> **Do not run `uv pip install mace-torch`.** The default target of `uv pip` is
> the project `.venv`. Doctor may then find both engines but will report
> `UMA/MACE dependencies: incompatible`, and MACE checkpoints cannot load.

The model path is specified with `--model` (CLI), in the TUI config screen, or via the `MODEL_PATH` key in INCAR files.

### 2.7 Verify Installation

```bash
uv run mlipx doctor
```

With no options, `doctor` reads package metadata and the hardware inventory
without importing a model backend, so it does not initialise DeepMD,
TensorFlow, or CUDA as a side effect. To test whether an environment can
actually start the backend you intend to use, name both the engine and device:

```bash
# auto checks CUDA when nvidia-smi finds a GPU, otherwise CPU
uv run mlipx doctor --engine uma --device auto

# Isolated environments
.venv-mace/bin/mlipx doctor --engine mace --device cuda:0
.venv-dpa/bin/mlipx doctor --engine dpa --device cuda:0
.venv-grace/bin/mlipx doctor --engine grace --device cpu
```

With an explicit backend, doctor imports it in an isolated subprocess and runs
a real tensor operation on the target CPU/GPU. Import errors, invisible CUDA
devices, failed operations, and unsupported PyTorch wheel architectures
produce a nonzero exit status. A CPU target does not require CUDA.

Use the full probe to validate a concrete model/task/head/structure. It loads
the checkpoint and evaluates energy and forces; for periodic models that
support stress it checks stress too. It writes no OUTCAR, trajectory, or result
directory:

```bash
.venv-dpa/bin/mlipx doctor --engine dpa --device cuda:0 \
  --model models/dpa/DPA-3.2-5M.pt --task bulk \
  --head Domains_SSE_PBE --structure structure.vasp
```

`--model` requires both `--engine` and `--task`. Multi-head MACE/DPA models
also require an explicit `--head`; doctor never guesses a potential-energy
surface.

For a full command list:
```bash
uv run mlipx --help
```

---

## 3. Quick Start

### 3.1 Your First Calculation (CLI)

```bash
# Single-point energy of a crystal structure
uv run mlipx sp structure.cif --model uma-s-1.pt --task omat

# Output:
# ================================================================================
#                             UMA CALCULATOR
#                (Universal Material Application - FAIRChem)
# ================================================================================
#
# Reading structure from: structure.cif
# System: Li3PS4
# Atoms: 28
#
# Loading model: uma-s-1.pt
#   Calculating energy and forces...
#   Energy: -123.456789 eV
#   Calculation completed in 2.34 s
#
# ================================================================================
#  SUMMARY
# ================================================================================
# Total energy:       -123.45678900 eV
# Energy per atom:      -4.40917068 eV/atom
# Max force:             0.12345678 eV/Å
# RMS force:             0.05678901 eV/Å
# ================================================================================
```

### 3.2 Your First Calculation (TUI)

```bash
# Launch the interactive terminal UI
uv run mlipx tui
```

Navigate with arrow keys, Tab to switch fields, Enter to select.

```
┌─ UMA Calculator ───────────────────────────────────────────────────────────────┐
│ Select Calculation Type                                                        │
│                                                                                │
│   Single Point (SP)                                                            │
│     Calculate energy, forces, and stress                                       │
│                                                                                │
│   Geometry Optimization (OPT)                                                  │
│     Optimize atomic positions                                                  │
│                                                                                │
│   Molecular Dynamics (MD)                                                      │
│     Run NVT/NVE simulations                                                    │
│                                                                                │
│   Batch Processing                                                             │
│     Process multiple structures                                                │
│                                                                                │
│   Background Jobs                                                              │
│     View/manage running calculations                                           │
│                                                                                │
│   Generate Template                                                            │
│     Create INCAR template file                                                 │
│                                                                                │
│   Exit                                                                         │
│     Quit the application                                                       │
└────────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Using INCAR Files (VASP-style)

```bash
# Generate a template
mlipx template sp -o INCAR.mlipx

# Edit it:
#   CALC_TYPE   = SP
#   MODEL_TYPE  = UMA            # default; or MACE/DPA/GRACE
#   MODEL_PATH  = uma-s-1.pt
#   TASK        = omat           # UMA: omat/omol/...; others: bulk/molecule
#   DEVICE      = cpu

# Run from INCAR
mlipx run
```

---

## 4. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE LAYER                          │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────┐          │
│  │ CLI      │    │ TUI          │    │ Python API       │          │
│  │ (argparse)│   │ (Textual)    │    │ (mlipx.api)     │          │
│  └────┬─────┘    └──────┬───────┘    └────────┬─────────┘          │
│       │                 │                     │                     │
│       └─────────────────┼─────────────────────┘                     │
│                         ▼                                           │
│              ┌─────────────────────┐                                │
│              │   EngineConfig      │  ← Unified configuration       │
│              │   (dataclass)       │                                │
│              └─────────┬───────────┘                                │
│                        ▼                                            │
│              ┌─────────────────────┐                                │
│              │ CalculationEngine   │  ← Single execution entry      │
│              │ .run() / .run_async()│                                │
│              └─────────┬───────────┘                                │
└────────────────────────┼────────────────────────────────────────────┘
                         │
┌────────────────────────┼────────────────────────────────────────────┐
│              CALCULATION LAYER                                       │
│              ┌─────────────────────┐                                │
│              │   BaseRunner        │  ← Progress events, logging     │
│              └─────────┬───────────┘                                │
│         ┌──────────────┼──────────────────┐                         │
│         ▼              ▼                  ▼                          │
│  ┌────────────┐ ┌────────────┐ ┌──────────────┐                     │
│  │SinglePoint │ │Optimization│ │Molecular     │                     │
│  │Runner      │ │Runner      │ │DynamicsRunner│                     │
│  └─────┬──────┘ └─────┬──────┘ └──────┬───────┘                     │
│        │              │               │                              │
│        └──────────────┼───────────────┘                              │
│                       ▼                                              │
│              ┌─────────────────────┐                                │
│              │   UMACalculator     │  ← Wraps FAIRChem ASE calc     │
│              └─────────┬───────────┘                                │
└────────────────────────┼────────────────────────────────────────────┘
                         │
┌────────────────────────┼────────────────────────────────────────────┐
│              MODEL LAYER                                             │
│              ┌─────────────────────┐                                │
│              │  FAIRChem UMA Model │  ← SO(3)-equivariant NN        │
│              │  InferenceSettings  │     tf32, compile, threads      │
│              └─────────────────────┘                                │
└─────────────────────────────────────────────────────────────────────┘
```

The `CalculationEngine` is the central orchestrator: all three interfaces (CLI, TUI, API) construct an `EngineConfig` and call the same `CalculationEngine` methods. This eliminates code duplication and ensures consistent behavior.

---

## Multi-Engine Guide

mlipx is designed to be **engine-agnostic**: every MLIP backend implements the unified `BaseMLIPCalculator` abstract interface, so the higher-level single-point, optimization, MD, batch, and output logic is fully shared. Switching engines only requires setting `MODEL_TYPE` - no changes to your calculation scripts or INCAR structure are needed.

### Engine Capability Comparison

| Engine | `MODEL_TYPE` | Backend package | Energy/Forces | Stress | Task values | Inference modes |
|--------|--------------|-----------------|---------------|--------|-------------|-----------------|
| UMA (FAIRChem) | `uma` (default, alias `fairchem`) | `fairchem-core` | ✓ | ✓ | `omat`/`omol`/`oc20`/`oc25`/`odac`/`omc` | `default`/`turbo` |
| MACE | `mace` | `mace-torch` | ✓ | ✓ | `bulk`/`molecule` | - |
| DPA (DeepMD-kit) | `dpa` | `deepmd-kit` | ✓ | ✓ | `bulk`/`molecule` | - |
| GRACE | `grace` | `tensorpotential` | ✓ | ✓ | `bulk`/`molecule` | - |

> `INFERENCE_MODE = turbo` is UMA-only (requires the `VALID_INFERENCE_MODES` attribute). Other engines ignore it.

### Four Ways to Switch Engines

**Option 1: CLI flag `--model-type`** (applies to sp/opt/md/batch)

```bash
.venv-mace/bin/mlipx sp structure.cif --model mace.model --model-type mace --task bulk
```

**Option 2: INCAR file `MODEL_TYPE` key**

```ini
CALC_TYPE   = OPT
MODEL_TYPE  = MACE
MODEL_PATH  = mace.model
TASK        = bulk
FMAX        = 0.05
```

**Option 3: TUI dropdown**

The TUI uses the Python environment that launched it. Run `uv run mlipx tui`
for UMA and `.venv-mace/bin/mlipx tui` for MACE. The Model Engine dropdown
does not switch virtual environments automatically; do not select MACE inside
the UMA TUI.

**Option 4: Python API `model_type` parameter**

```python
from mlipx.api import run_single_point

result = run_single_point(
    structure="structure.cif",
    model_path="mace.model",
    model_type="mace",     # default 'uma'
    task="bulk",
    device="cpu",
)
```

### Environments, model task/head, and pre-run validation

Installation commands are maintained only in Section 2; this section no
longer duplicates a second, easily stale setup guide. Always use the command
prefix for the corresponding environment. If MACE was accidentally installed
into UMA's `.venv`, run `uv sync --frozen` to restore the locked UMA
environment, then rebuild `.venv-mace` from Section 2.1.

`task` and model `head` are different decisions. For non-UMA engines,
`bulk|molecule` explicitly declares periodic semantics; a MACE head or DPA
branch selects the potential-energy surface. mlipx no longer rewrites input
PBC: `bulk` requires full 3-D PBC and `molecule` requires a fully nonperiodic
structure. A multi-head MACE/DPA model without explicit `--head` fails closed.
Use `dp show MODEL model-branch` for canonical DPA branch names and aliases.

Before production MD, run a no-output single-structure doctor probe:

```bash
.venv-mace/bin/mlipx doctor --engine mace --device cuda:0 \
  --model mace.model --task bulk --head default --structure test.vasp

.venv-dpa/bin/mlipx doctor --engine dpa --device cuda:0 \
  --model models/dpa/DPA-3.2-5M.pt --task bulk \
  --head Domains_SSE_PBE --structure test.vasp
```

**GRACE neighbour caching (enabled by default)** uses a Verlet candidate table
at `cutoff + NEIGHBOR_SKIN` and still applies the model's exact cutoff every
step. It rebuilds when atom count, species, cell, or PBC changes, or when any
atom's general minimum-image displacement exceeds `skin/2`; skew cells and
repeated periodic images are retained. Cached and uncached paths use the same
canonical neighbour order and are cross-checked by unit tests plus a short
200-frame GPU random trajectory. If a TensorPotential export cannot safely
accept the cache, mlipx errors instead of silently falling back. Disable it
explicitly with INCAR `NEIGHBOR_CACHE=False` or CLI `--no-neighbor-cache`.
A historical V100/400-atom LGPS run measured about 51 ms/step versus 66
ms/step uncached; that is a hardware/model-specific reference, not a general
performance guarantee.

### Task Mapping & Periodic Boundaries (PBC)

UMA has a well-defined task system (each maps to a training dataset); other engines are task-unaware, and `task` only controls the PBC strategy:

| `task` | PBC | Equivalent UMA task | Suitable systems |
|--------|------|---------------------|-------------------|
| `bulk` | True | `omat` | Periodic crystals, surface slabs, MOFs |
| `molecule` | False | `omol` | Isolated molecules (auto charge=0) |

> **Charge/spin defaults (task-dependent):**
> - UMA `omol`: when `atoms.info` is unset, auto-fills `charge=0` and `spin=1` (spin **multiplicity**, 1 = singlet).
> - Non-UMA `molecule` (MACE/DPA/GRACE): auto-fills only `charge=0`; **no** `spin` is injected. MACE reads `atoms.info["spin"]` as **total spin S** (0 = singlet), a different semantic from UMA's multiplicity -- blindly injecting `spin=1` would force spin-sensitive MACE models into a doublet radical. Set spin explicitly if needed.
> - For molecular tasks, set these in the TUI, with CLI `--charge` / `--spin`, or with INCAR `CHARGE` / `SPIN`. These explicit values override existing input `atoms.info`; editing the metadata in Python remains supported.
> - Molecular systems (`pbc=False`) do not compute a stress tensor.

### Complete Examples per Engine

```bash
# -- UMA (default) -- bulk material optimization
mlipx opt Li2O.cif --model uma-s-1.pt --model-type uma --task omat --cell-opt --fmax 0.02

# -- MACE -- periodic single point
.venv-mace/bin/mlipx sp bulk.cif --model mace.model --model-type mace --task bulk --device cuda

# -- MACE head selection: use a head actually listed by the loaded model
.venv-mace/bin/mlipx sp bulk.cif --model mace-mpa-0-medium.model --model-type mace --task bulk --head default

# -- MACE dtype (float64 by default; opt into float32 for speed)
.venv-mace/bin/mlipx opt bulk.cif --model mace.model --model-type mace --task bulk --dtype float32

# -- DPA (DeepMD) -- periodic optimization in its isolated environment
.venv-dpa/bin/mlipx opt bulk.vasp --model dpa.pt --model-type dpa --task bulk --fmax 0.01 --optimizer LBFGS

# -- GRACE -- high-throughput screening (batch)
mlipx batch structures/ --model grace_model/ --model-type grace --task bulk --calc-type sp --output results/
```

### MACE-specific options

| Option | Description |
|--------|-------------|
| `--head HEAD` | Select a head that is actually present in the model. Omit it for a single-head model. Head names are model/version-specific. |
| `--dtype float32\|float64` | Compute precision. The accuracy-first default is **`float64` for SP, optimization, and MD**. Opt into `float32` explicitly for speed. |

> ⚠️ `--head` takes one string. Although mace-torch itself may warn and fall
> back for an unknown name, mlipx rejects that case: silently changing heads
> changes the potential-energy surface. The bundled
> `models/mace/mace-mpa-0-medium.model` currently exposes only `default`.
> The default `float64` setting is accuracy-first. Use `--dtype float32` only
> when the precision/performance tradeoff is explicitly acceptable.

### Engine Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| `Backend mace_torch not installed` | The current interpreter is not the MACE environment | Follow the isolation procedure and launch `.venv-mace/bin/mlipx tui` |
| `ImportError: e3nn` version conflict | mace + fairchem in same env | Create a separate venv for MACE |
| `too many values to unpack (expected 2)` | A model saved with e3nn 0.4.4 is loaded by e3nn 0.5/0.6 | Use `.venv-mace`; verify `e3nn.__version__ == 0.4.4` |
| `Object of type Tensor is not JSON serializable` | An older mlipx wrote MACE tensor metadata directly to JSON | Update mlipx; the current writer converts tensors to scalars/lists |
| `Model file not found` | Wrong path | Use an absolute path for `--model` |
| UMA works but MACE errors | e3nn conflict in same env | See environment isolation above |
| task `bulk` invalid under UMA | UMA does not recognize bulk | Use omat for UMA; bulk is for non-UMA engines |

---

## 5. Calculation Types

### 5.1 Single Point (SP)

A single-point calculation computes the potential energy, atomic forces, and (if supported) stress tensor for a fixed atomic configuration. This is the simplest and fastest calculation type.

**What it produces:**
- Total energy (eV)
- Energy per atom (eV/atom)
- Force on each atom (eV/Å), with max and RMS force magnitudes
- Stress tensor (Voigt notation, eV/Å³) — if supported by the model/task
- Pressure (GPa) — derived from stress trace

**CLI usage:**

```bash
mlipx sp <structure> --model <model.pt> [options]

# Basic
mlipx sp POSCAR --model uma-s-1.pt --task omat

# With output directory and job name
mlipx sp structure.cif \
    --model uma-s-1.pt \
    --task omat \
    --device cuda \
    --output ./results \
    --name my_calculation
```

**Output files:** `OUTCAR`, `CONTCAR`, `mlipx_results.json`

### 5.2 Geometry Optimization (OPT)

Optimizes atomic positions (and optionally cell parameters) to find a local energy minimum. The calculation stops when the maximum force on any atom falls below the convergence threshold (`fmax`), or when the maximum number of steps is reached.

**Algorithms:**

| Optimizer | Description | Best for |
|-----------|-------------|----------|
| `FIRE` | Fast Inertial Relaxation Engine (default) | Most systems, robust |
| `BFGS` | Broyden-Fletcher-Goldfarb-Shanno | Small systems, fast convergence |
| `LBFGS` | Limited-memory BFGS | Larger systems |

**CLI usage:**

```bash
mlipx opt <structure> --model <model.pt> [options]

# Basic optimization
mlipx opt POSCAR --model uma-s-1.pt

# Tight convergence with cell relaxation
mlipx opt POSCAR \
    --model uma-s-1.pt \
    --fmax 0.02 \
    --max-steps 1000 \
    --cell-opt \
    --optimizer BFGS

# Preserve crystal symmetry
mlipx opt structure.cif \
    --model uma-s-1.pt \
    --fix-symmetry
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--fmax` | 0.05 | Force convergence threshold (eV/Å) |
| `--max-steps` | 500 | Maximum optimization steps |
| `--optimizer` | FIRE | FIRE / BFGS / LBFGS |
| `--cell-opt` | off | Enable cell parameter optimization |
| `--fix-symmetry` | off | Preserve crystal symmetry |

**Output files:** `OUTCAR`, `CONTCAR` (optimized structure), `OSZICAR` (step-by-step progress), `mlipx_results.json`

### 5.3 Molecular Dynamics (MD)

Simulates the time evolution of atoms at a given temperature. Supports these methods:

| Ensemble | Integrator | Description |
|----------|-----------|-------------|
| NVT | Langevin | Stochastic Langevin thermostat |
| NVT | Bussi / CSVR | Stochastic velocity rescaling |
| NVT | Nosé-Hoover Chain | Deterministic chain thermostat |
| NVE | Velocity Verlet | Constant particle number, volume, energy (microcanonical) |

**Pre-relaxation:** Before starting MD, mlipx can perform a short,
positions-only FIRE optimization (default: 50 steps, fmax=0.1 eV/Å) to reduce
large atomic forces. It does not relax the cell or guarantee a local minimum,
and therefore does not in general eliminate cell stress. The default depends
on the ensemble:
- **NVT**: pre-relaxation is **on** by default (avoids explosions from high initial forces).
- **NVE**: pre-relaxation is **off** by default -- changing positions first
  changes the conserved-energy baseline, so the trajectory no longer
  corresponds to the input structure. Enable it explicitly only when that
  change of initial condition is intended.

Override the default with `--pre-relax` / `--no-pre-relax` (CLI) or `PRE_RELAX = .TRUE./.FALSE.` (INCAR).
For a restart structure that already stores momenta, use `--no-pre-relax` with
`--velocity-policy auto` or `preserve`: changing positions while preserving
velocities is not an exact phase-space restart and is rejected.

**CLI usage:**

```bash
mlipx md <structure> --model <model.pt> [options]

# NVT at 300K for 10 ps
mlipx md POSCAR \
    --model uma-s-1.pt \
    --ensemble NVT \
    --temp 300 \
    --timestep 1.0 \
    --steps 10000 \
    --save-interval 10

# NVE ensemble
mlipx md CONTCAR \
    --model uma-s-1.pt \
    --ensemble NVE \
    --temp 300 \
    --steps 5000
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--ensemble` | NVT | NVT or NVE |
| `--temp` | 300 | Temperature (K) |
| `--timestep` | 1.0 | Time step (fs) |
| `--steps` | 1000 | Number of production MD steps |
| `--equilibration-steps` | 0 | Same-ensemble equilibration steps before production |
| `--thermostat` | LANGEVIN | `LANGEVIN`, `BUSSI`, or `NHC` for NVT |
| `--friction` | 0.001 | Langevin friction (fs⁻¹) |
| `--bussi-tau` | 1000.0 | Bussi/CSVR coupling time (fs) |
| `--nhc-tdamp` | 100.0 | NHC damping time (fs) |
| `--nhc-tchain` | 3 | NHC chain length |
| `--nhc-tloop` | 1 | NHC thermostat substeps |
| `--save-interval` | 10 | Save trajectory every N steps |
| `--seed` | generated and recorded | Random seed for reproducible velocity initialization and NVT noise |
| `--velocity-policy` | auto | `auto`, `initialize`, or `preserve` stored momenta |
| `--pre-relax` | NVT on / NVE off | Pre-relax before MD (`--no-pre-relax` to disable) |
| `--pre-relax-steps` | 50 | Max pre-relaxation steps |
| `--pre-relax-fmax` | 0.1 | Pre-relaxation force threshold (eV/Å) |

Thermostat choice and coupling strength may influence dynamical and transport
properties. For transport-oriented calculations, thermostat sensitivity should
be checked. `--equilibration-steps` runs an initial phase with the same ensemble,
thermostat, and timestep. Switching from NVT equilibration to NVE production
still requires separate runs and explicit handling of restart velocities.

The real-backend interface smoke test uses four atoms, CPU by default, and
5 steps per method. Run it with each isolated environment, for example:

```bash
.venv/bin/python mlipx/examples/smoke_md_backends.py --backend uma --model UMA.pt
.venv-mace/bin/python mlipx/examples/smoke_md_backends.py --backend mace --model MACE.model
.venv-dpa/bin/python mlipx/examples/smoke_md_backends.py --backend dpa --model DPA.pt
.venv-grace/bin/python mlipx/examples/smoke_md_backends.py --backend grace --model GRACE_SAVEDMODEL
```

**MD output layout:**

```text
<run-directory>/
├── raw/
│   ├── trajectory.traj       # lossless ASE trajectory; internal source of truth
│   ├── md.csv                # time, energies, temperature, volume, stress, pressure
│   └── mlipx_results.json    # machine-readable result summary
├── vasp/
│   ├── XDATCAR               # VASP syntax, unwrapped direct coordinates
│   ├── CONTCAR               # VASP POSCAR/CONTCAR syntax
│   └── OUTCAR                # explicitly labelled VASP-like MD subset
├── artifacts.json            # output contract, units, frame metadata, file index
├── resolved_config.json
└── run.log
```

`raw/trajectory.traj` is the trajectory source of truth;
`vasp/XDATCAR` primarily targets OVITO, ASE, and other VASP ecosystem tools.
For analysis, pass the whole run directory when possible instead of only the
trajectory file. This also gives mlipx the time, phase, unit, and coordinate
contracts in `raw/md.csv`, `artifacts.json`, and `resolved_config.json`.

### 5.4 Built-in Trajectory Analysis (Analysis v2)

The current release directly supports thermodynamic statistics, RDF,
RMSD/RMSF, directional MSD, diffusion, Nernst–Einstein conductivity, 3-D
occupancy density, Arrhenius fitting, ionic sites and jumps, VACF, and velocity
spectra. The analysis layer does not load a model, so trajectories from DPA,
MACE, GRACE, and UMA can all be processed in one analysis environment.

Install the required dependencies from the repository root:

```bash
# MSD, RDF, density, and plotting
python -m pip install -e './mlipx[analysis]'

# Include both kinisi and GEMDAT
python -m pip install -e './mlipx[analysis-all]'
```

`RUN` may be an mlipx MD run directory, an ASE `.traj`, or an XDATCAR. If an
external trajectory lacks time or coordinate metadata, declare it only when
the values are genuinely known, for example:

```bash
mlipx analyze trajectory.traj validate \
    --positions-convention wrapped \
    --frame-interval-fs 10
```

Do not infer wrapped versus unwrapped coordinates from appearance, and do not
confuse the MD timestep with the interval between saved frames.

#### Validate before calculating

```bash
mlipx analyze RUN validate
```

`validate` reports the time axis, MD timestep and saved-frame interval, PBC,
fixed or variable cell, coordinate convention, velocities,
equilibration/production phases, Nyquist frequency, run status, and eligibility
for each analysis. An mlipx run marked failed, aborted, or cancelled is
rejected. Imported trajectories have no mlipx run status, so the user remains
responsible for confirming that they are complete.

MD can record equilibration and production separately with
`--equilibration-steps N` and `--steps M`, using the same ensemble, integrator,
thermostat, timestep, and temperature. Analyses use production frames by
default. `thermo`, RDF, RMSD, MSD, density, VACF, and spectrum accept
`--include-equilibration` for diagnostics; `transport` and `electrolyte` always
use production frames. mlipx does not decide when equilibration ends or choose
the linear diffusive window. Legacy trajectories without phase metadata treat
every frame as production. That is a compatibility rule, not a claim that the
entire run is equilibrated. Inspect `thermo`, then use `--start-frame` and
`--stop-frame` on tasks that support a frame range. These are saved-frame
indices, not MD step numbers.

#### Common analyses

```bash
# Temperature, energy, pressure, and NVE energy-drift diagnostics
mlipx analyze RUN thermo

# Li-S partial RDF; the coordination cutoff is always explicit
mlipx analyze RUN rdf --center Li --neighbor S --rmax 6 --cn-cutoff 3

# Periodic-displacement RMSD/RMSF for a crystal
mlipx analyze RUN rmsd --species Li

# Directional, multiple-time-origin MSD
mlipx analyze RUN msd --mobile Li --axes x,y,z,xy,xyz \
    --drift-reference nonmobile

# 3-D occupancy density, VACF, and a VACF-derived velocity spectrum
mlipx analyze RUN density --mobile Li --spacing 0.25
mlipx analyze RUN vacf --species Li
mlipx analyze RUN spectrum --species Li --taper one-sided-cosine
```

RDF `center` and `neighbor` are ordered, and self pairs are excluded when the
two selections are identical. `rmax` may not exceed half the minimum face
height of the triclinic cell. A coordination number is calculated only when
`--cn-cutoff` is explicit. Density output contains both an occupancy
probability summing to one and a number density in Å⁻³ whose cell integral is
the number of selected ions.

MSD uses multiple time origins and defaults to an FFT implementation;
`--method direct` is available as a cross-check. It operates on continuous
coordinates: unwrapped trajectories are used directly, while wrapped,
fixed-cell trajectories are reconstructed from consecutive minimum-image
displacements and checked for half-cell ambiguity. Variable-cell trajectories
are rejected because the current implementation cannot reliably separate
affine cell deformation from migration. Drift removal is never enabled
silently: choose `none`, `nonmobile`, or explicit `indices`; the choice is
recorded in provenance. An ordinary-least-squares diagnostic slope is produced
only when both `--fit-start-ps` and `--fit-stop-ps` are explicit. Without a
window mlipx reports MSD and the local log-log exponent but does not guess a
diffusion coefficient. OLS is not a substitute for covariance-aware transport
and uncertainty analysis.

#### Diffusion and conductivity

Covariance-aware transport fitting uses kinisi v2:

```bash
mlipx analyze RUN transport \
    --mobile Li \
    --charge 1 \
    --drift-reference nonmobile \
    --fit-start-ps 20 \
    --lag-step-ps 1 \
    --lag-stop-ps 200 \
    --random-seed 0
```

The 20 ps value is only a command example. Choose the start of an equilibrated,
diffusive regime from `thermo`, the MSD curve, the sampling length, and the
physics of the system. Transport requires a fixed cell, three-dimensional PBC,
at least four production frames, uniform sampling, and a known coordinate
convention. A wrapped trajectory is rejected when consecutive displacements
approach the unsafe range for minimum-image reconstruction; mlipx will not
return a plausible-looking diffusion coefficient from an unverifiable unwrap.

For long, frequently saved trajectories, use `--lag-step-ps` together with
`--lag-stop-ps` to provide an explicit kinisi lag-time grid. These options
reduce the lag times evaluated by kinisi, not the trajectory frames, so the
complete trajectory and its time origins remain available. `--fit-start-ps`
has a separate meaning: it chooses the first lag used by diffusion regression,
whereas the lag grid chooses the available lag values. Select the grid from the
observed diffusive regime and compare spacings such as 0.5, 1, and 2 ps as a
convergence check. kinisi's documentation notes that evaluating every possible
time interval can be excessive for full covariance analysis. Without an
explicit grid, mlipx retains kinisi's native behavior for small grids but
rejects pathological dense defaults and asks for explicit parameters.

kinisi 2.x's triclinic parser creates several `frames × atoms × 8` arrays.
mlipx first applies kinisi's unweighted mean-framework drift definition once,
then passes only mobile atoms to the parser and estimates peak allocation
before creating frames. The default guard is 4 GiB and can be changed with
`--parser-memory-limit-gib`; raise it only after checking available RAM.
Results record source/parser atom counts, triclinic dispatch, estimated bytes,
and the limit. This guards peak allocation; it is not an exact prediction of
total process RSS.

kinisi reports posterior mean, standard deviation, and 95% interval in both
m²/s and cm²/s. Directional diffusion follows `D = slope / (2d)`, where `d` is
the number of selected directions. `--charge` is required and is never guessed
from the element. Temperature is the production-frame mean; if unavailable,
it must be supplied with `--temperature-K`, with no 300 K fallback. The
Nernst–Einstein tracer conductivity uses `σ = n(ze)²D/(kBT)` and should not be
treated as an experimental or correlation-aware conductivity. A collective
quantity is calculated only with `--collective-conductivity`; mlipx does not
silently report a Haven ratio.

The Nernst–Einstein output now includes posterior summaries in S/m, S/cm, and
mS/cm. They are the linear propagation of the kinisi tracer-D posterior with
number density, charge, volume, and temperature held fixed—not total physical
uncertainty. Model, finite-size, replica, temperature, volume, approximation,
and ion-ion-correlation uncertainty are not included. Legacy scalar
`sigma_NE_tracer_*` fields remain and equal the corresponding posterior means.

kinisi's ASE backend reconstructs periodic displacements from wrapped/scaled
coordinates. For an exact unwrapped source, mlipx compares every saved-frame
displacement against ASE's periodic minimum-image reconstruction before
calling kinisi and refuses the analysis if image history would be lost. A
successful result records that the reconstruction was equivalent, while
making clear that kinisi did not consume exact image counters directly.

Successful transport also writes `transport_summary.csv`,
`transport_msd.png`, and `transport_msd.svg` alongside `kinisi_arrays.npz`,
and the CLI prints the posterior summary, 95% credible interval, fit window,
lag grid, and Nernst–Einstein conductivity. Prefer `mlipx analyze RUN
transport ...` so the run metadata supplies time, save stride, coordinate
convention, phase, and temperature. For external trajectories, provide
`--positions-convention`, `--frame-interval-fs`, and `--temperature-K` only
when their values are known. The `2 ps` lag spacing used in current LGPS work
is a sensitivity-checked working parameter, not a universal default.

Independent diffusion results at several temperatures can be fitted with the
`arrhenius` task using `ln D = ln D₀ - Eₐ/(kBT)`. At least three temperatures
are recommended; a two-point fit warns because linearity cannot be tested. If
`--diffusivity-std` values are supplied, the fit uses the approximation
`σ(ln D) = σ(D)/D` as weights, and extrapolated values are clearly labelled.
See all parameters with:

```bash
mlipx analyze RUN arrhenius --help
```

#### Electrolyte sites, jumps, and percolation

This layer uses GEMDAT. Either provide reference sites or explicitly request
site discovery from the density:

```bash
mlipx analyze RUN electrolyte \
    --mobile Li \
    --sites Li_sites.cif \
    --jump-dimensions 3 \
    --percolation-axes xyz

# Use only when no reference sites are available, then inspect the result.
mlipx analyze RUN electrolyte --mobile Li --discover-sites-from-density
```

The two site sources are mutually exclusive and mlipx never chooses one
silently. `--jump-dimensions` controls the dimensional factor in the jump
diffusivity. `--percolation-axes` only constrains pathway connectivity; it does
not change the jump dimension, kinisi tracer diffusion, or conductivity
definition. Outputs include reference or discovered sites, transition and
jump tables and matrices, density and free-energy arrays, pathway coordinates,
and barriers. GEMDAT's endpoint tracer estimate does not replace the kinisi
transport result; the two use different methods and serve different purposes.

VACF and `spectrum` use velocities actually stored at uniform sampling; they do
not numerically differentiate positions. The default one-sided cosine taper
has unit weight at time zero, and negative spectral values are retained and
reported. The result is a “VACF-derived velocity spectrum,” not automatically
a harmonic phonon DOS.

#### Output and reproducibility

Results are written to `RUN/analysis/<task>/<request-hash>/`, including
`request.json`, `provenance.json`, `results.json`, task-specific CSV/NPZ data,
PNG/SVG figures, and diagnostics. The hash includes the source fingerprint,
selection, frame range, directions, drift rule, scientific parameters, and
backend versions. Identical requests reuse an existing result unless `--force`
is given. Failure details are written to `error.json` without creating a
`results.json`; an incomplete task is never marked as successful.

This README is sufficient to run the workflows above. Detailed algorithm
definitions, unit tables, and references remain in the
[Analysis v2 notes](ANALYSIS.md), [transport definitions](TRANSPORT.md), and
[electrolyte mechanism guide](ELECTROLYTE.md).

### 5.5 Batch Processing

Run the same calculation type on many structures in a directory. Batch
supports `sp` and `opt`, is currently CLI-only, and processes files
sequentially while loading the model only once.

**CLI usage:**

```bash
uv run mlipx batch <input_dir> --model <model.pt> [options]

# SP calculation on all CIF files
uv run mlipx batch structures/ \
    --model uma-s-1.pt \
    --calc-type sp \
    --pattern "*.cif" \
    --output batch_results

# MACE batch in the isolated environment
.venv-mace/bin/mlipx batch structures/ \
    --model mace.model \
    --model-type mace \
    --task bulk \
    --device cuda \
    --calc-type sp \
    --pattern "*.cif" \
    --output mace_batch_results
```

**Output:** Each structure gets its own subdirectory and `batch_summary.json`
contains the aggregate result. The current CLI does not accept `--parallel` or
`--workers`, and the TUI has no Batch menu.

---

## 6. User Interfaces

### 6.1 CLI — Command Line Interface

The CLI is invoked via `mlipx <command> [options]`. Running `mlipx` without arguments launches the TUI by default.

#### Complete Command Reference

##### `mlipx sp` — Single Point

```
mlipx sp STRUCTURE --model MODEL [--model-type TYPE] [--task TASK] [--device DEVICE]
                       [--charge INTEGER] [--spin INTEGER]
                       [--cpu-threads N] [--inference-mode MODE]
                       [--activation-checkpointing | --no-activation-checkpointing]
                       [--dtype DTYPE] [--head HEAD] [--output DIR] [--name NAME]

  STRUCTURE             Input structure file (CIF, XYZ, POSCAR, VASP, etc.)
  --model MODEL         Path to model checkpoint [required]
  --model-type TYPE     MLIP engine: uma|mace|dpa|grace [default: uma]
  --task TASK           UMA: omat|omol|oc20|oc25|odac|omc; others: bulk|molecule [default: omat]
  --device DEVICE       cpu|gpu|cuda|cuda:N [default: cpu]
  --charge INTEGER      Total molecular charge (UMA omol default: 0)
  --spin INTEGER        Molecular spin metadata; spin multiplicity for UMA omol (default: 1)
  --cpu-threads N       CPU intra-op threads (all engines; backend default if omitted)
  --inference-mode MODE UMA inference preset: default|turbo
  --[no-]activation-checkpointing
                        UMA memory/speed override; omitted: follow inference preset
  --dtype DTYPE         MACE dtype float32|float64 [default: float64]
  --head HEAD           MACE head or DeepMD/DPA multi-task branch
  --output DIR, -o DIR  Output directory [default: .]
  --name NAME, -n NAME  Job name (output goes to DIR/NAME)
```

> `--model-type` applies to all calculation commands (sp/opt/md/batch); the default remains `uma`.

##### `mlipx opt` — Geometry Optimization

```
mlipx opt STRUCTURE --model MODEL [options]

  --fmax FMAX           Force convergence threshold eV/Å [default: 0.05]
  --max-steps N         Maximum optimization steps [default: 500]
  --optimizer ALGO      FIRE|BFGS|LBFGS [default: FIRE]
  --cell-opt            Enable cell parameter optimization
  --fix-symmetry        Preserve crystal symmetry
```

##### `mlipx md` — Molecular Dynamics

```
mlipx md STRUCTURE --model MODEL [options]

  --ensemble ENSEMBLE   NVT|NVE [default: NVT]
  --temp TEMP           Temperature in Kelvin [default: 300]
  --timestep DT         Time step in fs [default: 1.0]
  --steps N             Number of production MD steps [default: 1000]
  --equilibration-steps N
                        Same-ensemble equilibration steps [default: 0]
  --thermostat TYPE     LANGEVIN|BUSSI|NHC [default: LANGEVIN]
  --friction VALUE      Langevin friction in fs^-1 [default: 0.001]
  --bussi-tau FS        Bussi/CSVR coupling time [default: 1000.0]
  --nhc-tdamp FS        NHC damping time [default: 100.0]
  --nhc-tchain N        NHC chain length [default: 3]
  --nhc-tloop N         NHC thermostat substeps [default: 1]
  --save-interval N     Save trajectory every N steps [default: 10]
  --seed N              Reproducible MD seed [default: generated and recorded]
  --velocity-policy P   auto|initialize|preserve [default: auto]
  --pre-relax           Pre-relax before MD [default: on for NVT, off for NVE]
  --pre-relax-steps N   Max pre-relaxation steps [default: 50]
  --pre-relax-fmax F    Pre-relaxation force threshold eV/Å [default: 0.1]
  --fmax-abort F        Force-safety abort threshold eV/Å [default: 20.0]
```

##### `mlipx analyze` — Trajectory Analysis

```text
mlipx analyze RUN validate
mlipx analyze RUN thermo
mlipx analyze RUN msd --mobile Li --axes x,y,z,xyz \
    --drift-reference nonmobile
```

`RUN` may be an mlipx run directory, an ASE `.traj`, or an XDATCAR. Passing
the run directory preserves the most complete time, PBC, phase, and unit
metadata. Available tasks are `validate`, `thermo`, `rdf`, `rmsd`, `msd`,
`transport`, `density`, `arrhenius`, `electrolyte`, `vacf`, and `spectrum`.
Run `validate` first and use its eligibility report before further analysis.

Dependency installation, complete examples, and scientific limits are in
Section 5.4. Use `mlipx analyze RUN <task> --help` for the current arguments of
each subcommand.

##### `mlipx batch` — Batch Processing

```
mlipx batch INPUT_DIR --model MODEL [options]

  --calc-type TYPE      sp|opt [default: sp]
  --pattern PATTERN     Explicit file glob; omitted: CIF/XYZ/VASP/POSCAR*
  --output DIR          Output directory [default: batch_results]
```

Use the correct executable: `uv run mlipx batch ...` for UMA and
`.venv-mace/bin/mlipx batch ...` for MACE.

##### `mlipx run` — Run from INCAR File

```
mlipx run [-i INCAR] [-s STRUCTURE] [-o OUTPUT]

  -i, --incar INCAR     Path to INCAR file [default: INCAR.mlipx]
  -s, --structure FILE  Structure file (auto-detected: POSCAR, CONTCAR, *.cif, *.xyz)
  -o, --output DIR      Output directory [default: .]
```

##### `mlipx template` — Generate INCAR Template

```
mlipx template TYPE [-o OUTPUT]

  TYPE                  sp|opt|md
  -o, --output FILE     Output file name [default: INCAR.<type>]
```

##### `mlipx jobs` — List Background Jobs

```
mlipx jobs
```

Shows all background jobs with their ID, status, type, formula, and device.

##### `mlipx kill` — Kill a Background Job

```
mlipx kill JOB_ID
```

Terminates the specified job (cross-platform: `taskkill` on Windows, `SIGTERM` on Unix).

##### `mlipx clean` — Clean Completed/Failed Jobs

```
mlipx clean
```

Removes state files for jobs that are done, failed, or cancelled. Running jobs are preserved.

##### `mlipx queue pause/resume` — Pause/Resume an Individual Pending Job

```bash
mlipx queue pause <job-id>     # pause only the selected PENDING job
mlipx queue status
mlipx queue resume <job-id>    # resume that job
```

The selected job becomes `paused`, so the scheduler skips it and can continue
with other `pending` jobs. The current RUNNING job and other pending jobs are
unaffected. After resuming, the job returns to the queue and participates in
FIFO dispatch by its original submission time. Omitting `<job-id>` still pauses
or resumes the whole pending queue; this differs from `mlipx queue stop`, which
stops the scheduler process itself.

##### `mlipx tui` — Launch TUI

```
mlipx tui
```

Starts the interactive Terminal User Interface.

### 6.2 TUI — Terminal User Interface

The TUI is built on [Textual](https://textual.textualize.io/) and provides an interactive, keyboard-driven experience.

#### Navigation

| Key | Action |
|-----|--------|
| `↑` / `↓` | Navigate menu items / scroll |
| `Tab` | Move to next input field |
| `Shift+Tab` | Move to previous input field |
| `Enter` | Select / confirm |
| `Esc` | Go back to previous screen |
| `Q` | Quit application |
| `PgUp` / `PgDn` | Page up/down in scrollable areas |
| `C` | Cancel selected job (Jobs screen) |
| `D` | Delete job record (Jobs screen) |
| `P` | Pause selected pending job (Jobs screen) |
| `R` | Refresh job list (Jobs screen) |
| `U` | Resume selected paused job (Jobs screen) |

#### Screens

**Main Menu** — Select SP, OPT, MD, Jobs, Template, or Exit. Batch is currently
CLI-only.

**Configuration** — Fill in paths, engine/task and a device string (`cpu`,
`cuda`, or `cuda:N`). The **Backend & Resource Options** section exposes:

- CPU threads for every engine (PyTorch for UMA/MACE and DPA `.pt` models,
  TensorFlow for GRACE; DPA `.pb` uses its DeepMD TensorFlow settings)
- UMA inference mode and activation checkpointing
- MACE precision and MACE/DPA model head or branch
- Total charge and spin for `omol` / `molecule`; UMA labels spin as multiplicity

Only controls used by the selected backend are shown; for example, selecting
UMA no longer leaves a disabled DPA branch field on screen. Charge/spin appear
only for molecular tasks and are cleared when switching back to a periodic
task. OPT also exposes cell optimization and symmetry preservation. MD exposes
the ensemble, temperature, timestep, equilibration/production step counts,
save interval, NVT thermostat and its active coupling parameters,
pre-relaxation controls, random seed, velocity policy, and force-safety abort
threshold. Paths support live validation with visual feedback:

```
📁 Structure File: [structure.cif                ]
   ✅ Found: /home/user/mlipx/structure.cif
   💡 Tip: Relative paths are supported (e.g., ./data/structure.cif)
```

**Run** — Starts the calculation as an independent background process:
- Back returns immediately without cancelling the calculation
- The calculation continues after exiting the TUI or disconnecting SSH
- Cancel terminates the background calculation; Back only closes the monitor
- The absolute live-log path and a copy-paste `tail -f` command at startup

**Jobs** — DataTable listing all background jobs with status icons:
- ● Running | ✓ Done | ✗ Failed | ⊘ Cancelled
- Press Enter on a job to view its log output
- Auto-refreshes every 2 seconds
- Select a job in the Jobs table and use **Pause Job** / **Resume Job**, or press `P` / `U`; only that pending/paused job is affected.
- **Pause Queue** / **Resume Queue** remain available for pausing/resuming the entire pending queue. Running jobs are unaffected.

### 6.3 Python API

For scripting and workflow integration, import `mlipx.api`:

```python
from mlipx.api import run_single_point, run_optimization, run_md
from mlipx.api import calculate_energy

# Single point energy
results = run_single_point(
    structure="structure.cif",
    model_path="uma-s-1.pt",
    task="omat",
    device="cuda",
    job_name="my_calc",
)
print(f"Energy: {results['energy']:.4f} eV")
print(f"Forces: {results['forces']}")

# Geometry optimization
results = run_optimization(
    structure="POSCAR",
    model_path="uma-s-1.pt",
    fmax=0.02,
    cell_opt=True,
)
print(f"Converged: {results['converged']} in {results['nsteps']} steps")

# Molecular dynamics
results = run_md(
    structure="CONTCAR",
    model_path="uma-s-1.pt",
    ensemble="NVT",
    temperature=300,
    steps=10000,
    save_interval=10,
)
print(f"Final temperature: {results['temperature']:.1f} K")

# Quick energy calculation
energy = calculate_energy("structure.cif", "uma-s-1.pt")
print(f"Energy: {energy:.4f} eV")

```

Full API reference:

| Function | Returns | Description |
|----------|---------|-------------|
| `run_single_point(structure, model_path, ...)` | `dict` | SP energy, forces, stress |
| `run_optimization(structure, model_path, ...)` | `dict` | OPT with convergence info |
| `run_md(structure, model_path, ...)` | `dict` | MD with trajectory and temperature |
| `calculate_energy(structure, model_path, ...)` | `float` | Quick energy value |

---

## 7. INCAR Configuration Reference

INCAR files use a VASP-style `KEY = VALUE` format. Lines starting with `#` or `!` are comments.

### 7.1 All INCAR Keywords

#### Calculation Control

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `CALC_TYPE` | string | `SP` | Calculation type: `SP`, `OPT`, `MD`, `BATCH` |

#### Model Settings

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `MODEL_PATH` | string | `uma-s-1p2p1.pt` (template placeholder) | Path to model checkpoint (.pt file) |
| `MODEL_TYPE` | string | `uma` | MLIP engine: `uma`, `mace`, `dpa`, `grace` (`fairchem` = `uma`) |
| `TASK` | string | `omat` | Task type. UMA: `omat`/`omol`/`oc20`/`oc25`/`odac`/`omc`; others: `bulk`/`molecule` |
| `DEVICE` | string | `cpu` | Compute device: `cpu`, `cuda`, `gpu`, or `cuda:N` |
| `CHARGE` | int | unset (UMA omol uses `0`) | Total molecular charge; overrides `atoms.info["charge"]` |
| `SPIN` | int | unset (UMA omol uses `1`) | Molecular spin metadata; spin multiplicity for UMA omol |
| `INFERENCE_MODE` | string | `default` | Inference mode: `default`, `turbo` |
| `DEFAULT_DTYPE` | string | `float64` | MACE dtype for every calculation type. MACE only. |
| `HEAD` | string | - | MACE head or DeepMD/DPA multi-task branch. |
| `NEIGHBOR_CACHE` | boolean | `.TRUE.` | GRACE Verlet neighbour cache preserving the exact cutoff and complete periodic-image semantics. Floating bond vectors may differ by about 1e-14 Å rounding. GRACE only. |
| `NEIGHBOR_SKIN` | float | `1.5` | GRACE neighbour-cache skin in Å; the neighbour table is rebuilt when any atom moves more than `NEIGHBOR_SKIN/2`. GRACE only. |

#### Output Control

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `WRITE_FORCES` | bool | `.TRUE.` | Write forces to OUTCAR |
| `WRITE_STRESS` | bool | `.TRUE.` | Write stress to OUTCAR |
| `WRITE_TRAJECTORY` | bool | `.TRUE.` | Write trajectory for MD |
| `OUTPUT_FORMAT` | string | `VASP` | Output format: `VASP` |

#### Optimization Settings

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `OPT_ALGO` | string | `FIRE` | Optimizer: `FIRE`, `BFGS`, `LBFGS` |
| `FMAX` | float | `0.05` | Force convergence (eV/Å) |
| `MAX_STEPS` | int | `500` | Max optimization steps |
| `CELL_OPT` | bool | `.FALSE.` | Optimize cell parameters |
| `FIX_SYMMETRY` | bool | `.FALSE.` | Preserve symmetry |

#### MD Settings

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `MD_ENSEMBLE` | string | `NVT` | Ensemble: `NVT`, `NVE` |
| `TEMPERATURE` | float | `300.0` | Temperature (K) |
| `TIMESTEP` | float | `1.0` | Time step (fs) |
| `STEPS` | int | `10000` | Number of production MD steps |
| `EQUILIBRATION_STEPS` | int | `0` | Same-ensemble equilibration steps before production; `EQUIL_STEPS` is an alias |
| `THERMOSTAT` | string | `LANGEVIN` | NVT thermostat: `LANGEVIN`, `BUSSI`, `NHC` |
| `FRICTION` | float | `0.001` | Langevin friction (fs⁻¹) |
| `BUSSI_TAU` | float | `1000.0` | Bussi/CSVR coupling time (fs) |
| `NHC_TDAMP` | float | `100.0` | NHC damping time (fs) |
| `NHC_TCHAIN` | int | `3` | NHC chain length |
| `NHC_TLOOP` | int | `1` | NHC thermostat substeps |
| `SAVE_INTERVAL` | int | `10` | Trajectory save interval |
| `PRE_RELAX` | bool | NVT=`.TRUE.`/NVE=`.FALSE.` | Pre-relax before MD (ensemble-aware default) |
| `PRE_RELAX_STEPS` | int | `50` | Max pre-relaxation steps |
| `PRE_RELAX_FMAX` | float | `0.1` | Pre-relaxation force threshold (eV/Å) |

`SAVE_INTERVAL` controls only full trajectory frames containing positions,
velocities, forces, and related frame data. Energy and temperature lines in
`run.log` are lightweight thermodynamic records and are currently emitted at
every MD step. For example, 400000 steps produce 400001 step log records, while
`SAVE_INTERVAL=200` still produces only 2001 trajectory frames. Both intervals
are printed separately at the start of a run.

### 7.2 Template Examples

**INCAR.sp** (Single Point):
```bash
CALC_TYPE = SP
TASK = omat
MODEL_PATH = uma-s-1.pt
DEVICE = cpu
INFERENCE_MODE = default
WRITE_FORCES = .TRUE.
WRITE_STRESS = .TRUE.
```

**INCAR.opt** (Geometry Optimization):
```bash
CALC_TYPE = OPT
TASK = omat
MODEL_PATH = uma-s-1.pt
DEVICE = cuda
OPT_ALGO = FIRE
FMAX = 0.05
MAX_STEPS = 500
CELL_OPT = .FALSE.
FIX_SYMMETRY = .FALSE.
```

**INCAR.md** (Molecular Dynamics):
```bash
CALC_TYPE = MD
TASK = omat
MODEL_PATH = uma-s-1.pt
DEVICE = cuda
INFERENCE_MODE = turbo
MD_ENSEMBLE = NVT
TEMPERATURE = 300.0
TIMESTEP = 1.0
STEPS = 10000
EQUILIBRATION_STEPS = 0
THERMOSTAT = LANGEVIN
FRICTION = 0.001
BUSSI_TAU = 1000.0
NHC_TDAMP = 100.0
NHC_TCHAIN = 3
NHC_TLOOP = 1
SAVE_INTERVAL = 10
```

### 7.3 Boolean Values

The following are all recognized as `TRUE` and `FALSE` (case-insensitive):

- **TRUE**: `.TRUE.`, `.T.`, `TRUE`, `T`, `YES`, `Y`, `1`
- **FALSE**: `.FALSE.`, `.F.`, `FALSE`, `F`, `NO`, `N`, `0`

---

## 8. Output Files Reference

### 8.1 File Inventory

| File | Created By | Format | Description |
|------|-----------|--------|-------------|
| `OUTCAR` | SP, OPT | Text | VASP-style detailed output with energies, forces, stress, timing |
| `vasp/OUTCAR` | MD | Text | Labelled VASP-like subset with per-frame positions, forces, energies, temperature, cell, and stress |
| `CONTCAR` | SP, OPT | Text | Current/final atomic structure in VASP POSCAR format |
| `vasp/CONTCAR` | MD | Text | MD final structure in VASP POSCAR/CONTCAR syntax |
| `OSZICAR` | OPT | Text | Step-by-step optimization progress with energy and force |
| `vasp/XDATCAR` | MD | Text | VASP-syntax trajectory with unwrapped direct coordinates |
| `mlipx_results.json` | SP, OPT | JSON | Machine-readable results with all computed quantities |
| `raw/mlipx_results.json` | MD | JSON | MD result summary and canonical data paths |
| `raw/trajectory.traj` | MD | Binary | Lossless ASE trajectory retaining positions, velocities, forces, and more |
| `raw/md.csv` | MD | CSV | Per-frame thermodynamic scalars plus configurational/total stress and pressure; stress/pressure only for full 3D PBC |
| `artifacts.json` | MD | JSON | Versioned output contract, units, frame interval, and file index |
| `optimization.log` | OPT | Text | ASE optimizer log |
| `run.log` | All | Text | Continuously flushed live log; its path is shown by CLI and TUI |
| `batch_summary.json` | BATCH | JSON | Summary of all structures processed in batch |

Every run prints a hint like:

```text
Live log: /absolute/path/to/results/run.log
Follow live output: tail -f /absolute/path/to/results/run.log
```

`run.log` is flushed after every message, so the displayed command can be used from another terminal. On completion, the terminal log, `run.log`, the end of the corresponding `OUTCAR`, and JSON output record:

- **Total elapsed time:** from the user's run request until all standard output files are generated.
- **Compute elapsed time:** from the first compute phase after the model is ready until output writing starts; model loading and final output writing are excluded.

### 8.2 OUTCAR Format

The OUTCAR file contains:

SP/OPT retain the existing summary format. MD writes the versioned
`mlipx.vasp-like-outcar.md/1` subset to `vasp/OUTCAR`. Its header explicitly
states that it is not a native VASP OUTCAR. Every saved frame records the cell,
`POSITION / TOTAL-FORCE`, potential/kinetic/total energies, temperature,
volume, ASE stress, and pressure without inventing SCF, POTCAR, or electronic
structure data.

```
================================================================================
                         UMA CALCULATION RESULTS
                (Universal Material Application - FAIRChem)
================================================================================

Generated: 2026-06-19 14:30:15

--------------------------------------------------------------------------------
 SYSTEM INFORMATION
--------------------------------------------------------------------------------

Formula:           Li3PS4
Number of atoms:   28
Atom types:        Li: 12, P: 4, S: 12
Task:              omat
Calculation mode:  single_point

--------------------------------------------------------------------------------
 MODEL INFORMATION
--------------------------------------------------------------------------------

Model path:        uma-s-1.pt
Device:            cuda
Inference mode:    default
Properties:        energy, forces, stress

--------------------------------------------------------------------------------
 INPUT STRUCTURE
--------------------------------------------------------------------------------

Lattice vectors (Å):
      8.500000      0.000000      0.000000
      0.000000      8.500000      0.000000
      0.000000      0.000000      8.500000

Cell lengths (Å):    8.500000  8.500000  8.500000
Cell angles (°):    90.000000  90.000000  90.000000
Volume (Å³):         614.125000

Atomic positions (Cartesian, Å):
  Atom   Type            x            y            z
--------------------------------------------------------------
     1     Li      0.000000     0.000000     0.000000
     2     Li      4.250000     4.250000     0.000000
   ...

--------------------------------------------------------------------------------
 ENERGY
--------------------------------------------------------------------------------

Total energy:         -123.45678900 eV
Energy per atom:        -4.40917068 eV/atom

--------------------------------------------------------------------------------
 FORCES (eV/Å)
--------------------------------------------------------------------------------

  Atom   Type           Fx           Fy           Fz          |F|
----------------------------------------------------------------------
     1     Li      0.012345    -0.023456     0.034567     0.043210
   ...

Maximum force:           0.123457 eV/Å on atom 7 (S)
RMS force:               0.056789 eV/Å

--------------------------------------------------------------------------------
 STRESS TENSOR
--------------------------------------------------------------------------------

Stress (eV/Å³):
        Voigt           xx           yy           zz           yz           xz           xy
        Voigt     0.001234    -0.000567     0.000890     0.000000     0.000000     0.000000

Stress (GPa):
        Voigt     0.197734    -0.090856     0.142644     0.000000     0.000000     0.000000

Pressure:               -0.083174 GPa

--------------------------------------------------------------------------------
 TIMING
--------------------------------------------------------------------------------

Calculation time:     2.34 s

================================================================================
 END OF UMA CALCULATION
================================================================================
```

### 8.3 JSON Output (mlipx_results.json)

```json
{
  "uma_version": "1.0.0",
  "timestamp": "2026-06-19T14:30:15",
  "calculation": {
    "mode": "single_point",
    "system": {
      "formula": "Li3PS4",
      "natoms": 28,
      "symbols": ["Li", "Li", ...],
      "cell": [[8.5, 0.0, 0.0], [0.0, 8.5, 0.0], [0.0, 0.0, 8.5]],
      "cell_lengths": [8.5, 8.5, 8.5],
      "cell_angles": [90.0, 90.0, 90.0],
      "volume": 614.125,
      "pbc": [true, true, true]
    },
    "positions": [[0.0, 0.0, 0.0], ...],
    "results": {
      "energy": -123.456789,
      "energy_per_atom": -4.409171,
      "forces": [[0.012345, -0.023456, 0.034567], ...],
      "stress": [0.001234, -0.000567, 0.000890, 0.0, 0.0, 0.0],
      "force_statistics": {
        "fmax": 0.123457,
        "fmean": 0.045678,
        "frms": 0.056789
      },
      "pressure_gpa": -0.083174
    },
    "timing": {
      "calculation_time_s": 2.34,
      "total_elapsed_time_s": 8.91,
      "compute_time_s": 2.34
    }
  }
}
```

---

## 9. Task Types Reference

### 9.1 UMA Tasks (`MODEL_TYPE = UMA`)

UMA models are trained on different datasets; each task corresponds to a specific domain:

| Task | Domain | Systems | Charge/Spin | Stress | Typical Use |
|------|--------|---------|-------------|--------|-------------|
| `omat` | Inorganic Materials | Bulk crystals | Optional | ✓ | Battery materials, solid electrolytes, oxides |
| `omol` | Molecules | Isolated molecules | Default 0/1 | ✗ | Organic chemistry, drug-like molecules |
| `oc20` | Catalysis (OC20) | Surface slabs | Optional | ✓ | Heterogeneous catalysis, adsorption |
| `oc25` | Catalysis (OC25) | Surface slabs | Optional | ✓ | Extended catalysis benchmark |
| `odac` | MOFs | Metal-organic frameworks | Optional | ✓ | Gas storage, separation |
| `omc` | Molecular Crystals | Organic crystals | Optional | ✓ | Pharmaceuticals, organic electronics |

This installation follows `fairchem-core 0.1.dev1316`'s `UMATask` API, which
does not expose OC22. mlipx therefore rejects `--task oc22` instead of showing
a TUI choice that the installed calculator cannot accept. Checkpoint-specific
task availability is also validated when the checkpoint is loaded.

**Important for molecular systems (omol):** UMA's `omol` task needs total charge
and spin multiplicity. When unset, mlipx fills `charge=0` and `spin=1`
(singlet). The TUI shows both fields after selecting `omol`; CLI/INCAR can set
them directly:

```bash
mlipx sp molecule.xyz --model uma.pt --task omol --charge -1 --spin 2
# INCAR.mlipx: CHARGE = -1, SPIN = 2
```

The input structure metadata can still be edited in Python:

```python
from ase.io import read, write

atoms = read("molecule.xyz")
atoms.info["charge"] = 0    # Net charge
atoms.info["spin"] = 1      # Spin multiplicity = 2S+1
write("molecule.xyz", atoms)
```

mlipx validates but never rewrites PBC: periodic tasks (omat, oc20, oc25, odac, omc) require full 3-D PBC and a nondegenerate cell; omol requires a fully nonperiodic input.

### 9.2 Generic Tasks (MACE / DPA / GRACE)

Non-UMA engines are not task-aware; `task` only controls the periodic-boundary (PBC) strategy:
| Task | PBC | Equivalent UMA task | Suitable for |
|------|------|---------------------|--------------|
| `bulk` | True | `omat` | Periodic crystals, surfaces, MOFs |
| `molecule` | False | `omol` | Isolated molecules |

```bash
# MACE periodic material
.venv-mace/bin/mlipx sp bulk.cif --model mace.model --model-type mace --task bulk

# DPA isolated molecule
mlipx sp molecule.xyz --model dpa2.pth --model-type dpa --task molecule
```

mlipx validates but never rewrites PBC: periodic tasks require full 3-D PBC and a nondegenerate cell, while molecular tasks require a fully nonperiodic input.

---

## 10. Background Jobs

Long-running calculations (large-system MD, batch processing) can be submitted as background jobs. Jobs run as independent subprocesses and survive terminal disconnection.

### 10.1 Submitting

**CLI:** Not yet exposed via a `--detach` flag (use TUI).

**TUI:** SP, OPT, and MD calculations are launched in the background
automatically. Use **Back (Keep Running)** to leave the run screen without
stopping the job. The job also survives exiting the TUI or disconnecting SSH.

### 10.2 Managing Jobs

```bash
# List all jobs
mlipx jobs

# Output:
# ID                                       Status       Type   Formula      Device
# -----------------------------------------------------------------------------------------
# 2026-06-19_14-30-15_Li3PS4_sp            ● running    sp     Li3PS4       cuda
# 2026-06-19_15-00-22_Cu_slab_opt          ✓ done       opt    Cu16         cpu
# 2026-06-19_13-10-00_H2O_md               ✗ failed     md     H2O          cuda

# View a job's log (TUI: press Enter on the job row)

# Kill a running job
mlipx kill 2026-06-19_14-30-15_Li3PS4_sp

# Clean up completed/failed job records
mlipx clean
```

### 10.3 Job Lifecycle

```
pending ──→ running ──→ done
                │
                ├── cancelled (user kill)
                └── failed   (runtime error)
```

Job state files are stored at `~/.mlipx/jobs/`. Each job has a JSON state file and a log file:

```
~/.mlipx/jobs/
├── 2026-06-19_14-30-15_Li3PS4_sp.json       # State (status, PID, progress)
├── 2026-06-19_15-00-22_Cu_slab_opt.json
├── 2026-06-19_13-10-00_H2O_md.json
└── logs/
    ├── 2026-06-19_14-30-15_Li3PS4_sp.log    # Full calculation output
    ├── 2026-06-19_15-00-22_Cu_slab_opt.log
    └── 2026-06-19_13-10-00_H2O_md.log
```

---

## 11. Resource Control

The common controls are available directly in both the TUI and CLI; environment
variables are optional alternatives, not a requirement for TUI use.

| Control | TUI | CLI | Applies to |
|---|---|---|---|
| Device / GPU index | **Device** | `--device cpu`, `cuda`, or `cuda:N` | All engines |
| CPU thread count | **CPU Threads** | `--cpu-threads N` | All engines |
| Inference preset | **UMA Inference Mode** | `--inference-mode default|turbo` | UMA only |
| Activation checkpointing | **UMA Activation Checkpointing** | `--activation-checkpointing` / `--no-activation-checkpointing` | UMA only |

### 11.1 CPU Threads

Set the backend's intra-op CPU thread count directly:

```bash
mlipx sp structure.cif --model uma-s-1.pt --cpu-threads 4
```

The equivalent TUI field is **CPU Threads**. Leaving it blank lets the
backend/system choose. mlipx maps the value to PyTorch intra-op threads for
UMA/MACE and DPA PyTorch models, and TensorFlow intra-op threads for GRACE.
Legacy DPA TensorFlow `.pb` models use their DeepMD TensorFlow backend settings. The historical
`--torch-num-threads` spelling remains a compatible CLI alias.
`OMP_NUM_THREADS=4` remains an environment-level alternative for PyTorch
backends.

**Python API / EngineConfig:**

```python
config = EngineConfig(
    ...,
    torch_num_threads=4,
)
```

### 11.2 GPU Memory

UMA's `activation_checkpointing` trades compute for memory: enabling it reduces
GPU memory use at the cost of some speed. It is available in the TUI and CLI:

```bash
mlipx sp structure.cif --model uma-s-1.pt --device cuda \
    --activation-checkpointing
```

Omit the flag (TUI: **Auto**) to follow the selected inference preset. Use
`--no-activation-checkpointing` only when the structure fits in VRAM and speed
is more important.

```python
config = EngineConfig(
    ...,
    activation_checkpointing=True,  # Lower GPU memory
)
```

### 11.3 GPU Selection

Select a logical GPU directly with `--device cuda:N`, or enter the same value in
the TUI **Device** field:

```bash
mlipx sp structure.cif --model uma-s-1.pt --device cuda:0
```

For DPA and GRACE, mlipx maps this selection to the environment mechanism used
by DeepMD/TensorFlow before constructing the calculator. An explicitly set
`CUDA_VISIBLE_DEVICES` remains useful for process-level GPU isolation, but it
is not required for ordinary TUI or CLI selection.

### 11.4 Inference Modes

| Mode | tf32 | compile | merge_mole | activation_ckpt | Best for |
|------|------|---------|------------|-----------------|----------|
| `default` | No | No | No | Yes | General use, SP, OPT |
| `turbo` | Yes | Yes | Yes | No | MD, large systems, production |

UMA MD defaults to `turbo`; SP and OPT default to `default`. Choose another
preset explicitly with the TUI selector or `--inference-mode`. MACE, DPA, and
GRACE do not use this UMA-specific setting.

---

## 12. Troubleshooting & FAQ

### 12.1 Common Errors

#### "No edges found in structure"

**Cause:** The model cannot build a neighbor graph for your structure. This happens when atoms are too far apart (> 6 Å cutoff), the cell is invalid, or PBC settings are wrong.

**Solutions:**
1. Check the input structure file — ensure atomic positions are reasonable
2. For bulk materials, ensure the cell is not too large (atoms should be within ~6 Å of each other)
3. Try the original POSCAR format instead of CIF
4. Check that PBC is set correctly (use `omat` task for periodic, `omol` for molecules)

#### "no kernel image is available for execution on the device"

**Cause:** Your PyTorch build has no CUDA kernel compatible with your GPU. PyTorch 2.7+ removed `sm_50`/`sm_60` from its pre-built CUDA wheels, so Maxwell/Pascal GPUs (compute capability 5.x/6.x) lost their matching kernels.

| GPU family | CC | Recommended torch |
|-----|-----|--------|
| Maxwell (GTX 750/9xx, incl. GTX 960) | sm_50/52 | `torch==2.6.0+cu124` |
| Pascal (GTX 10xx, P104-100) | sm_60/61 | `torch==2.6.0+cu124` |
| Volta–Hopper (V100…H100, RTX 20/30/40) | sm_70–90 | `torch==2.6.0+cu124` |
| Blackwell (RTX 50) | sm_100/120 | `torch==2.8.0+cu128` |
| Kepler (GTX 700/600) | sm_30/37 | not supported |

> **`sm_50`/`sm_60` kernels are binary-compatible with `sm_52`/`sm_61` devices** (minor-revision compatibility within the same major). So a PyTorch build that still ships `sm_50`/`sm_60` — e.g. the `torch 2.6.0+cu124` wheel — drives GTX 960 (sm_52) and P104-100 (sm_61) correctly. Only torch 2.7+ is the problem, not the GPU itself.

**Diagnose (works even before torch is installed):**
```bash
uv run mlipx setup      # detect GPU + print the exact torch command
uv run python -c "import torch; print(torch.__version__, torch.cuda.get_arch_list())"
uv run mlipx doctor --engine uma --device cuda:0  # check CC and wheel support
```

**Solutions (preferred first):**
1. Install the recommended PyTorch build for your GPU (see table above):
   ```bash
   clashctl on  # enable proxy if needed
   uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124   # Maxwell–Hopper
   # uv pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128  # Blackwell
   ```
2. Build PyTorch from source with the needed kernels: `TORCH_CUDA_ARCH_LIST="6.0;6.1" python setup.py develop`
3. Fall back to CPU: `--device cpu`


#### CUDA Out of Memory

**Cause:** The model + structure requires more GPU memory than available.

**Solutions:**
1. Switch to CPU: `--device cpu`
2. Use a smaller model (e.g., `uma-s-1.pt` instead of a larger variant)
3. For large systems (> 100 atoms), CPU may be the only option

#### "No structure file found"

**Cause:** mlipx couldn't find a structure file.

**Solutions:**
1. Specify the structure explicitly: `mlipx sp POSCAR --model ...`
2. Place `POSCAR`, `CONTCAR`, or `*.cif` in the current directory
3. Use absolute paths: `mlipx sp /path/to/structure.cif --model ...`

#### TUI Import Error

**Cause:** The `textual` package is not installed.

**Solution:**
```bash
uv pip install textual
```

#### Atom Explosion in MD

**Cause:** High initial forces cause atoms to fly apart in early MD steps.

**Solution:**
- Pre-relaxation is on by default for NVT (up to 50 positions-only FIRE steps before MD); for NVE it is off by default. Enable it only when changing the input positions to reduce large forces is intended.
- If it still fails, run a full geometry optimization first:
  ```bash
  mlipx opt POSCAR --model uma-s-1.pt --fmax 0.02
  mlipx md CONTCAR --model uma-s-1.pt --temp 100
  ```
- Lower the initial temperature: `--temp 100`
- Check that your initial structure is physically reasonable
- If energy/forces become NaN or inf, the run aborts automatically (no NaN results are written) -- check for atoms too close or outside the training distribution

### 12.2 FAQ

**Q: How accurate is the UMA model compared to DFT?**

A: UMA models are trained on DFT (PBE/SCAN) reference data. For systems within the training domain, energy errors are typically < 10 meV/atom and force errors < 50 meV/Å. Accuracy outside the training domain is not guaranteed.

**Q: Can I use this for chemical reactions?**

A: For reactions where bonds break/form, you need molecular dynamics. Use the `omol` task for molecules. Note that UMA is an MLIP — it approximates the PES but does not describe electronic transitions.

**Q: How many atoms can I simulate?**

A: Depends on memory. With 32 GB CPU RAM, systems up to ~500 atoms are feasible. With GPU, 100-200 atoms depending on VRAM. The model scales roughly O(N) with system size.

**Q: Does the TUI work over SSH?**

A: Yes, if your terminal supports Unicode and 256 colors. Most modern terminals (Windows Terminal, iTerm2, GNOME Terminal, Alacritty) work. Ensure your terminal window is at least 80 columns × 24 rows.

**Q: Can I run multiple calculations simultaneously?**

A: Yes. TUI SP/OPT/MD jobs run as background processes. Batch processing is
currently sequential and CLI-only; use `uv run mlipx batch ...` for UMA or
`.venv-mace/bin/mlipx batch ...` for MACE.

**Q: What file formats are supported for input structures?**

A: All formats supported by ASE's `read()` function: CIF, POSCAR, CONTCAR, XYZ, VASP, XSD, and many others.

**Q: Is the UMA model suitable for metals? Semiconductors? Insulators?**

A: UMA models support all three. The accuracy depends on whether the specific material is within the training distribution. The `omat` task covers a broad range of inorganic materials.

---

## 13. Performance Guide

### 13.1 Typical Timings

Timings for `uma-s-1.pt` on a single structure (approximate, varies by system):

| System | Atoms | CPU (8 cores) | GPU (RTX 3080) |
|--------|-------|---------------|-----------------|
| Li₃PS₄ (bulk) | 28 | 2-3 s | 0.5-1 s |
| Cu slab (3×3×4) | 144 | 15-20 s | 3-5 s |
| MOF-5 | 424 | 60-90 s | 15-20 s |

MD: ~1000 steps/minute for a 28-atom system on GPU.

### 13.2 Optimization Tips

1. **GPU for large systems, CPU for small ones:** GPU has overhead; for < 20 atoms, CPU may be faster
2. **turbo mode for MD/production:** `--inference-mode turbo` (auto for MD via CLI)
3. **Batch for throughput:** Process many structures in one invocation to avoid model reloading overhead
4. **Pre-relax before MD:** On by default for NVT (positions only); for NVE enable explicitly only if changing the initial structure is intended

### 13.3 Memory Scaling

```
Atoms  ~RAM (CPU)  ~VRAM (GPU)
  10      2 GB       1 GB
  50      4 GB       2 GB
 100      6 GB       4 GB
 200     10 GB       7 GB
 500     24 GB      16 GB
```

---

## 14. Examples

### 14.1 Battery Material Energy

```bash
# Calculate the energy of an LLZO electrolyte structure
mlipx sp LLZO.cif --model uma-s-1.pt --task omat --device cuda
# Output: OUTCAR, CONTCAR, mlipx_results.json
```

### 14.2 Surface Relaxation

```bash
# Optimize a Pt(111) slab with cell fixed
mlipx opt Pt111_slab.cif \
    --model uma-s-1.pt \
    --task oc20 \
    --fmax 0.02 \
    --max-steps 300 \
    --optimizer FIRE \
    --device cuda
```

### 14.3 NVT MD Simulation

```bash
# Run 100 ps NVT at 400 K on an optimized structure
mlipx md CONTCAR \
    --model uma-s-1.pt \
    --ensemble NVT \
    --temp 400 \
    --timestep 1.0 \
    --steps 100000 \
    --save-interval 100 \
    --device cuda
```

### 14.4 Batch Screening

```bash
# Single-point energy on 100 CIF files
uv run mlipx batch candidates/ \
    --model uma-s-1.pt \
    --calc-type sp \
    --pattern "*.cif" \
    --output screening_results

# Analyze with Python
import json
with open("screening_results/batch_summary.json") as f:
    data = json.load(f)
for r in data["results"]:
    if r["success"]:
        print(f"{r['filename']}: {r['energy']:.4f} eV")
```

### 14.5 Workflow with INCAR File

```bash
# 1. Generate template
mlipx template opt -o INCAR.opt

# 2. Edit INCAR.opt:
#    CALC_TYPE = OPT
#    TASK = omat
#    MODEL_PATH = /home/user/models/uma-s-1.pt
#    DEVICE = cuda
#    FMAX = 0.01
#    MAX_STEPS = 1000
#    CELL_OPT = .TRUE.

# 3. Run
cat relax_results/OUTCAR
```

### 14.7 Equation of State (EOS)

Compute an energy-vs-volume curve:

```python
# eos_workflow.py
from ase.io import read, write
import numpy as np
import subprocess

atoms = read("Li2O.cif")
for i, scale in enumerate(np.linspace(0.9, 1.1, 11)):
    a = atoms.copy()
    a.set_cell(atoms.cell * scale, scale_atoms=True)
    write(f"eos_{i:02d}.cif", a)
    subprocess.run([
        "mlipx", "sp", f"eos_{i:02d}.cif",
        "--model", "uma-s-1.pt", "--task", "omat",
        "--output", f"eos_{i:02d}_results",
    ])
# Then collect energies from mlipx_results.json to fit the EOS
```

### 14.8 NEB Transition-State Preparation

Generate NEB intermediate images and optimize each:

```python
from ase.io import read, write
from ase.neb import NEB
import subprocess

initial, final = read("initial.cif"), read("final.cif")
images = [initial] + [initial.copy() for _ in range(3)] + [final]
NEB(images).interpolate()
for i, img in enumerate(images):
    write(f"neb_{i:02d}.cif", img)
    subprocess.run([
        "mlipx", "opt", f"neb_{i:02d}.cif",
        "--model", "uma-s-1.pt", "--output", f"neb_{i:02d}_opt",
    ])
```

### 14.9 Phonon Calculation (with phonopy)

```python
from ase.io import read, write
import subprocess, json
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

atoms = read("structure.cif")
ph_atoms = PhonopyAtoms(symbols=atoms.get_chemical_symbols(),
                      positions=atoms.positions, cell=atoms.cell)
phonopy = Phonopy(ph_atoms, [[2,0,0],[0,2,0],[0,0,2]])
phonopy.generate_displacements(distance=0.03)

forces = []
for i, sc in enumerate(phonopy.supercells_with_displacements):
    write(f"disp_{i:03d}.cif", sc)
    subprocess.run(["mlipx","sp",f"disp_{i:03d}.cif","--model","uma-s-1.pt",
                    "--output",f"disp_{i:03d}_results"])
    with open(f"disp_{i:03d}_results/mlipx_results.json") as f:
        forces.append(json.load(f)["calculation"]["results"]["forces"])
phonopy.forces = forces
phonopy.produce_force_constants()
phonopy.run_mesh([20,20,20]); phonopy.run_total_dos()
```

### 14.10 Formation Energy Calculation

```python
from ase.io import read
from collections import Counter
import subprocess, json

compound = read("Li2O.cif")
subprocess.run(["mlipx","sp","Li2O.cif","--model","uma-s-1.pt",
                "--task","omat","--output","Li2O_results"])
with open("Li2O_results/mlipx_results.json") as f:
    e_total = json.load(f)["calculation"]["results"]["energy"]

e_ref = {"Li": -1.9, "O": -4.9}  # elemental reference energies (calculate separately)
e_form = e_total - sum(c * e_ref[el] for el, c in
                       Counter(compound.get_chemical_symbols()).items())
print(f"Formation energy: {e_form/len(compound):.4f} eV/atom")
```

### 14.11 Recommended Settings Cheat Sheet

| System type | Recommended settings |
|-------------|----------------------|
| Small molecules (1-50 atoms) | `--optimizer LBFGS --fmax 0.01` |
| Bulk materials | `--optimizer FIRE --fmax 0.05` |
| Surfaces | `--optimizer FIRE --fmax 0.03` |
| MD equilibration | `--ensemble NVT --timestep 1.0` |
| MD production | `--ensemble NVE --timestep 1.0` |
| High-temperature MD | `--timestep 0.5 --friction 0.002` |

---

## 15. License

This project is open source under the MIT License.

Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the LICENSE file in the root directory of this source tree.
