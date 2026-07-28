# mlipx — User Manual

> **MLIP eXtended**
> A VASP-compatible CLI/TUI/API for machine-learning interatomic potentials. Supports UMA (FAIRChem, default), MACE, DPA (DeepMD-kit), and GRACE behind one interface.

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
  - [5.4 Batch Processing](#54-batch-processing)
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
- Process hundreds of structures in batch mode
- Output results in VASP-compatible formats (OUTCAR, CONTCAR, XDATCAR, OSZICAR)

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
| CLI mode | Full command-line interface with 10 subcommands |
| TUI mode | Interactive terminal UI with live progress |
| Python API | Programmatic access for scripting and workflows |
| Background jobs | Submit, detach, re-attach, and kill long-running calculations |
| Batch processing | Process many structures in parallel |
| CPU & CUDA | Runs on CPU or GPU, auto-detected |
| VASP output | OUTCAR, CONTCAR, XDATCAR, OSZICAR formats |
| Cross-platform | Windows, Linux, macOS |

---

## 2. Installation

### 2.1 Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.9+ | 3.11+ |
| RAM | 8 GB | 32 GB |
| Disk | 2 GB (model: 1.2 GB) | 15 GB (larger models) |
| GPU (optional) | CUDA 11.8+ | CUDA 12.x, 8+ GB VRAM |

### 2.2 Installing with uv (recommended)

> **Important:** `fairchem-core` is NOT published on PyPI. It must be installed from the local `packages/fairchem-core/` directory in this repository.

```bash
# Clone the repository
git clone https://github.com/FAIR-Chem/fairchem.git
cd fairchem

# Step 1: Detect your GPU and get the matching PyTorch command (CPU-only users
#         can skip). Uses nvidia-smi, works before PyTorch is installed.
uv run mlipx setup

# Step 2: Create a pinned venv (Python 3.12 via .python-version) and install
#         everything from the lockfile. uv auto-creates .venv.
#         Do NOT use `pip install -r requirements.txt` — that is the upstream CI
#         test snapshot (torch 2.8) and conflicts with this fork's GPU setup.
uv sync
```

### 2.3 CUDA GPU vs CPU Installation

mlipx does not ship its own PyTorch — it inherits the PyTorch installation provided by `fairchem-core`.

| Scenario | PyTorch | mlipx device flag |
|----------|---------|--------------------|
| **CUDA GPU machine** | `fairchem-core` installed in CUDA Python env | `--device cuda` |
| **CPU-only machine** | `fairchem-core` installed in standard Python env | `--device cpu` (default) |

**Verify CUDA availability after installation:**

```bash
uv run python -c "import torch; print('CUDA available' if torch.cuda.is_available() else 'CPU only')"
```

**If CUDA is not available but you have a GPU:**
- Ensure your PyTorch build includes CUDA (`pip install torch` with CUDA index)
- Or install `fairchem-core` in a conda/venv environment that already has CUDA PyTorch

### 2.4 How to Run Commands

Two equivalent methods:

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

### 2.5 Model Checkpoint

**UMA (default engine):** Download the checkpoint from FAIRChem:

```bash
# UMA Small (recommended starting point, ~1.2 GB)
# Download from: https://fair-chem.github.io/models/uma/
# Place the .pt file in your working directory or a known path
```

**Other engines (optional):** MACE / DPA / GRACE models are released by each project. Install the backend package, then use it:

| Engine | Backend install | Model format |
|--------|-----------------|--------------|
| MACE | `pip install mace-torch` | `.model` / `.pt` |
| DPA | `pip install 'deepmd-kit>=3.0.0'` | `.pth` / `.pt` (PyTorch backend) |
| GRACE | `pip install tensorpotential` | SavedModel dir / YAML |

> Note: `mace-torch` depends on `e3nn==0.4.4`, which conflicts with `fairchem-core` (`e3nn>=0.5`) - they cannot coexist in one environment. Run `mlipx doctor` to check which engine backends are ready.

The model path is specified with `--model` (CLI), in the TUI config screen, or via the `MODEL_PATH` key in INCAR files.

### 2.6 Verify Installation

```bash
uv run mlipx doctor
```

This runs a comprehensive diagnostic: Python, PyTorch, CUDA, GPU compatibility, fairchem-core, mlipx, and model file. If any check fails, it prints exact fix commands.

For a full command list:
```bash
uv run mlipx --help

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
mlipx sp structure.cif --model mace.model --model-type mace --task bulk
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

In the `mlipx tui` config screen, the Model Engine selector switches between UMA/MACE/DPA/GRACE, and the task options update dynamically (UMA shows omat/omol/...; others show bulk/molecule).

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

### Backend Installation & Environment Isolation

`fairchem-core` ships with mlipx, so UMA works out of the box. Other engines need their backend package installed separately:

| Engine | Install command | Model format |
|--------|-----------------|--------------|
| MACE | `pip install mace-torch` | `.model` / `.pt` |
| DPA | `pip install 'deepmd-kit>=3.0.0'` | `.pth` (PyTorch backend) |
| GRACE | `pip install tensorpotential` | SavedModel dir / YAML |

> ⚠️ **Environment isolation warning:** `mace-torch` pins `e3nn==0.4.4`, which **fundamentally conflicts** with `fairchem-core` (`e3nn>=0.5`) - they cannot coexist in one Python environment.
>
> **Recommended approach:** create a separate environment for MACE/DPA/GRACE:
> ```bash
> # MACE-dedicated environment
> uv venv .venv-mace && source .venv-mace/bin/activate
> pip install mace-torch
> pip install -e uma/        # install mlipx (can still call MACE without fairchem-core)
> ```
>
> Run `mlipx doctor` to check whether each engine backend is ready (uninstalled engines show a warn status).

### Task Mapping & Periodic Boundaries (PBC)

UMA has a well-defined task system (each maps to a training dataset); other engines are task-unaware, and `task` only controls the PBC strategy:

| `task` | PBC | Equivalent UMA task | Suitable systems |
|--------|------|---------------------|-------------------|
| `bulk` | True | `omat` | Periodic crystals, surface slabs, MOFs |
| `molecule` | False | `omol` | Isolated molecules (auto charge=0/spin=1) |

For UMA, the `omol` task **requires** charge and spin to be set; for non-UMA engines, the `molecule` task auto-fills defaults.

### Complete Examples per Engine

```bash
# -- UMA (default) -- bulk material optimization
mlipx opt Li2O.cif --model uma-s-1.pt --model-type uma --task omat --cell-opt --fmax 0.02

# -- MACE -- periodic single point
mlipx sp bulk.cif --model mace.model --model-type mace --task bulk --device cuda

# -- DPA (DeepMD) -- molecule optimization
mlipx opt drug.xyz --model dpa2.pth --model-type dpa --task molecule --fmax 0.01 --optimizer LBFGS

# -- GRACE -- high-throughput screening (batch)
mlipx batch structures/ --model grace_model/ --model-type grace --task bulk --calc-type sp --output results/
```

### Engine Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| `Backend mace_torch not installed` | Backend package missing | `pip install mace-torch` (in isolated env) |
| `ImportError: e3nn` version conflict | mace + fairchem in same env | Create a separate venv for MACE |
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

Simulates the time evolution of atoms at a given temperature. Supports two ensembles:

| Ensemble | Integrator | Description |
|----------|-----------|-------------|
| NVT | Langevin | Constant particle number, volume, temperature (canonical) |
| NVE | Velocity Verlet | Constant particle number, volume, energy (microcanonical) |

**Pre-relaxation:** Before starting MD, mlipx automatically performs a quick FIRE optimization (default: 50 steps, fmax=0.1 eV/Å) to eliminate internal stress. This prevents "atom explosion" — a common failure mode where high initial forces cause atoms to fly apart. You can disable this with `--no-pre-relax` (not yet exposed in CLI; use INCAR or TUI).

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
| `--steps` | 1000 | Number of MD steps |
| `--friction` | 0.001 | Friction coefficient (NVT only, fs⁻¹) |
| `--save-interval` | 10 | Save trajectory every N steps |

**Output files:** `OUTCAR`, `CONTCAR` (final structure), `XDATCAR` (trajectory), `trajectory.traj` (ASE format), `mlipx_results.json`

### 5.4 Batch Processing

Run the same calculation type on many structures in a directory. Supports `sp` and `opt` calculation types. Parallel execution is available via `--parallel`.

**CLI usage:**

```bash
mlipx batch <input_dir> --model <model.pt> [options]

# SP calculation on all CIF files
mlipx batch structures/ \
    --model uma-s-1.pt \
    --calc-type sp \
    --pattern "*.cif" \
    --output batch_results

# OPT on all POSCAR files in parallel
mlipx batch poscars/ \
    --model uma-s-1.pt \
    --calc-type opt \
    --pattern "POSCAR*" \
    --parallel \
    --workers 4
```

**Output:** Each structure gets its own sub-directory under the output directory. A `batch_summary.json` file lists all results.

---

## 6. User Interfaces

### 6.1 CLI — Command Line Interface

The CLI is invoked via `mlipx <command> [options]`. Running `mlipx` without arguments launches the TUI by default.

#### Complete Command Reference

##### `mlipx sp` — Single Point

```
mlipx sp STRUCTURE --model MODEL [--model-type TYPE] [--task TASK] [--device DEVICE]
                       [--output DIR] [--name NAME]

  STRUCTURE             Input structure file (CIF, XYZ, POSCAR, VASP, etc.)
  --model MODEL         Path to model checkpoint [required]
  --model-type TYPE     MLIP engine: uma|mace|dpa|grace [default: uma]
  --task TASK           UMA: omat|omol|oc20|oc25|odac|omc; others: bulk|molecule [default: omat]
  --device DEVICE       cpu|cuda [default: cpu]
  --output DIR, -o DIR  Output directory [default: .]
  --name NAME, -n NAME  Job name (output goes to DIR/NAME)
```

> `--model-type` applies to all calculation commands (sp/opt/md/batch). Default `uma`, fully backward compatible.

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
  --steps N             Number of MD steps [default: 1000]
  --friction FRICTION   Friction coefficient for NVT [default: 0.001]
  --save-interval N     Save trajectory every N steps [default: 10]
```

##### `mlipx batch` — Batch Processing

```
mlipx batch INPUT_DIR --model MODEL [options]

  --calc-type TYPE      sp|opt [default: sp]
  --pattern PATTERN     File glob pattern [default: *.cif]
  --output DIR          Output directory [default: batch_results]
```

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
| `R` | Refresh job list (Jobs screen) |

#### Screens

**Main Menu** — Select calculation type (SP, OPT, MD, Batch, Jobs, Template, Exit).

**Configuration** — Fill in paths, task type, device, and calculation-specific parameters. Paths support live validation with visual feedback:

```
📁 Structure File: [structure.cif                ]
   ✅ Found: /home/user/fairchem/uma/structure.cif
   💡 Tip: Relative paths are supported (e.g., ./data/structure.cif)
```

**Run** — Shows live progress during calculation:
- Indeterminate spinner for SP calculations
- Step counter for OPT (Step 5/500) and MD (Step 100/10000)
- Real-time energy, forces, and temperature during MD

**Jobs** — DataTable listing all background jobs with status icons:
- ● Running | ✓ Done | ✗ Failed | ⊘ Cancelled
- Press Enter on a job to view its log output
- Auto-refreshes every 2 seconds

### 6.3 Python API

For scripting and workflow integration, import `mlipx.api`:

```python
from mlipx.api import run_single_point, run_optimization, run_md
from mlipx.api import calculate_energy, calculate_adsorption_energy

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

# Adsorption energy
ads_results = calculate_adsorption_energy(
    adsorbed_structure="adsorbed.cif",
    gas_structure="co2.xyz",
    surface_structure="slab.cif",
    model_path="uma-s-1.pt",
    task="oc20",
)
print(f"Adsorption energy: {ads_results['adsorption_energy']:.4f} eV")
```

Full API reference:

| Function | Returns | Description |
|----------|---------|-------------|
| `run_single_point(structure, model_path, ...)` | `dict` | SP energy, forces, stress |
| `run_optimization(structure, model_path, ...)` | `dict` | OPT with convergence info |
| `run_md(structure, model_path, ...)` | `dict` | MD with trajectory and temperature |
| `calculate_energy(structure, model_path, ...)` | `float` | Quick energy value |
| `calculate_adsorption_energy(ads, gas, surf, ...)` | `dict` | E_ads = E_adsorbed - E_gas - E_surface |

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
| `MODEL_PATH` | string | `uma-s-1.pt` | Path to model checkpoint (.pt file) |
| `MODEL_TYPE` | string | `uma` | MLIP engine: `uma`, `mace`, `dpa`, `grace` (`fairchem` = `uma`) |
| `TASK` | string | `omat` | Task type. UMA: `omat`/`omol`/`oc20`/`oc25`/`odac`/`omc`; others: `bulk`/`molecule` |
| `DEVICE` | string | `cpu` | Compute device: `cpu`, `cuda` |
| `INFERENCE_MODE` | string | `default` | Inference mode: `default`, `turbo` |

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
| `STEPS` | int | `10000` | Number of MD steps |
| `FRICTION` | float | `0.001` | Friction coefficient |
| `SAVE_INTERVAL` | int | `10` | Trajectory save interval |

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
FRICTION = 0.001
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
| `OUTCAR` | SP, OPT, MD | Text | VASP-style detailed output with energies, forces, stress, timing |
| `CONTCAR` | SP, OPT, MD | Text | Current/final atomic structure in VASP POSCAR format |
| `OSZICAR` | OPT | Text | Step-by-step optimization progress with energy and force |
| `XDATCAR` | MD | Text | Trajectory in VASP format (concatenated POSCARs) |
| `mlipx_results.json` | SP, OPT, MD | JSON | Machine-readable results with all computed quantities |
| `trajectory.traj` | MD | Binary | ASE trajectory file for analysis |
| `optimization.log` | OPT | Text | ASE optimizer log |
| `calculation.log` | All | Text | Structured calculation log |
| `batch_summary.json` | BATCH | JSON | Summary of all structures processed in batch |

### 8.2 OUTCAR Format

The OUTCAR file contains:

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
      "calculation_time_s": 2.34
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
| `omol` | Molecules | Isolated molecules | **Required** | ✗ | Organic chemistry, drug-like molecules |
| `oc20` | Catalysis (OC20) | Surface slabs | Optional | ✓ | Heterogeneous catalysis, adsorption |
| `oc25` | Catalysis (OC25) | Surface slabs | Optional | ✓ | Extended catalysis benchmark |
| `odac` | MOFs | Metal-organic frameworks | Optional | ✓ | Gas storage, separation |
| `omc` | Molecular Crystals | Organic crystals | Optional | ✓ | Pharmaceuticals, organic electronics |

**Important for molecular systems (omol):** Molecules must have `charge` and `spin` set in the Atoms info:

```python
from ase.io import read, write
For periodic systems (omat, oc20, oc25, odac, omc), PBC is automatically set to `True` and the cell is validated. For molecules (omol), PBC is set to `False`.

### 9.2 Generic Tasks (MACE / DPA / GRACE)

Non-UMA engines are not task-aware; `task` only controls the periodic-boundary (PBC) strategy:

| Task | PBC | Equivalent UMA task | Suitable for |
|------|------|---------------------|--------------|
| `bulk` | True | `omat` | Periodic crystals, surfaces, MOFs |
| `molecule` | False | `omol` | Isolated molecules |

```bash
# MACE periodic material
mlipx sp bulk.cif --model mace.model --model-type mace --task bulk

# DPA isolated molecule
mlipx sp molecule.xyz --model dpa2.pth --model-type dpa --task molecule
```
atoms = read("molecule.xyz")
atoms.info["charge"] = 0    # Net charge
atoms.info["spin"] = 1      # Spin multiplicity = 2S+1
write("molecule.xyz", atoms)
```

For periodic systems (omat, oc20, oc25, odac, omc), PBC is automatically set to `True` and the cell is validated. For molecules (omol), PBC is set to `False`.

---

## 10. Background Jobs

Long-running calculations (large-system MD, batch processing) can be submitted as background jobs. Jobs run as independent subprocesses and survive terminal disconnection.

### 10.1 Submitting

**CLI:** Not yet exposed via `--detach` flag (use TUI).

**TUI:** Enable the "Run in background (detach)" switch on the Configuration screen.

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

### 11.1 CPU Threads

Control the number of CPU threads used by PyTorch:

```bash
# CLI: set via environment variable
export OMP_NUM_THREADS=4
mlipx sp structure.cif --model uma-s-1.pt
```

**Python API / EngineConfig:**
```python
config = EngineConfig(
    ...,
    torch_num_threads=4,
)
```

### 11.2 GPU Memory

`activation_checkpointing` trades compute for memory — enabling it reduces GPU memory usage at the cost of a slight slowdown:

```python
config = EngineConfig(
    ...,
    activation_checkpointing=True,  # Lower GPU memory
)
```

For systems with limited VRAM (< 8 GB), keep this enabled (default).

### 11.3 GPU Selection

Use the standard `CUDA_VISIBLE_DEVICES` environment variable:

```bash
# Use GPU 0 only
CUDA_VISIBLE_DEVICES=0 mlipx sp structure.cif --model uma-s-1.pt --device cuda

# Use GPUs 0 and 1
CUDA_VISIBLE_DEVICES=0,1 mlipx sp structure.cif --model uma-s-1.pt --device cuda
```

### 11.4 Inference Modes

| Mode | tf32 | compile | merge_mole | activation_ckpt | Best for |
|------|------|---------|------------|-----------------|----------|
| `default` | No | No | No | Yes | General use, SP, OPT |
| `turbo` | Yes | Yes | Yes | No | MD, large systems, production |

`turbo` mode is automatically used for MD calculations through the CLI and API.

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
uv run mlipx doctor     # shows your GPU's CC and whether PyTorch supports it
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
- Pre-relaxation is enabled by default (50 FIRE steps before MD)
- If it still fails, run a full geometry optimization first:
  ```bash
  mlipx opt POSCAR --model uma-s-1.pt --fmax 0.02
  mlipx md CONTCAR --model uma-s-1.pt --temp 100
  ```
- Lower the initial temperature: `--temp 100`
- Check that your initial structure is physically reasonable

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

A: Yes. Use background jobs (`--detach` in TUI or `mlipx jobs`) for multiple independent calculations. For batch processing of many structures, use `mlipx batch --parallel --workers N`.

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
4. **Pre-relax before MD:** Enabled by default, saves wasted MD steps on high-force structures

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
mlipx batch candidates/ \
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

### 14.5 Adsorption Energy via Python API

```python
from mlipx.api import calculate_adsorption_energy

result = calculate_adsorption_energy(
    adsorbed_structure="CO_on_Pt.cif",
    gas_structure="CO.xyz",
    surface_structure="Pt_slab.cif",
    model_path="uma-s-1.pt",
    task="oc20",
    device="cuda",
)
print(f"Adsorption energy: {result['adsorption_energy']:.4f} eV")
# E_ads = E(CO+Pt) - E(CO) - E(Pt)
# Negative = favorable adsorption
```

### 14.6 Workflow with INCAR File

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
