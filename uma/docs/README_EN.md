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
| Batch processing | Process many structures sequentially with one model load |
| CPU & CUDA | Runs on CPU or GPU, auto-detected |
| VASP output | OUTCAR, CONTCAR, XDATCAR, OSZICAR formats |
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
uv sync
uv run mlipx doctor
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
  -e ./uma "e3nn==0.4.4" "mace-torch==0.3.16"
.venv-mace/bin/mlipx doctor
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
  -e ./uma "deepmd-kit[torch]==3.1.3"
.venv-dpa/bin/mlipx doctor
```

Do not omit the appropriate model branch for LGPS and related solid
electrolytes:

```bash
.venv-dpa/bin/mlipx sp structure.vasp \
  --model models/dma/DPA-3.2-5M.pt --model-type dpa \
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
  -e ./uma "tensorflow[and-cuda]==2.20.0" "tensorpotential==0.6.0"

# Use the validated cuDNN version on V100/Volta
uv pip install --no-config --python .venv-grace/bin/python \
  "nvidia-cudnn-cu12==9.3.0.75"
.venv-grace/bin/python -c \
  "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"
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
| GPU (optional) | CUDA 11.8+ | CUDA 12.x, 8+ GB VRAM |

### 2.3 UMA environment details

> **Note:** `fairchem-core` is not published on PyPI. It is a workspace member in this repository under `packages/fairchem-core/` and will be installed automatically by `uv sync`.

```bash
# Clone the repository
git clone https://github.com/hydrogen1222/mlipx.git
cd mlipx

# Step 1: Create a pinned venv (Python 3.12 via .python-version) and install
#         everything from the lockfile. uv auto-creates .venv.
uv sync

# Step 2 (optional): Detect your GPU and verify the PyTorch CUDA match.
#         Uses nvidia-smi and the installed PyTorch.
uv run mlipx setup
```

> Running `uv run mlipx doctor` immediately after cloning also performs an
> implicit sync and creates `.venv`. That environment is still UMA-only.
> Running `uv sync` explicitly first makes this boundary clearer.

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

### 2.5 How to Run Commands

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

### 2.6 Model Checkpoint

**UMA (default engine):** Download the checkpoint from FAIRChem:

```bash
# UMA Small (recommended starting point, ~1.2 GB)
# Download from: https://fair-chem.github.io/models/uma/
# Place the .pt file in your working directory or a known path
```

**Other engines (optional):** MACE / DPA / GRACE models are released by each
project. To prevent e3nn, PyTorch ABI, and CUDA/cuDNN binaries from overwriting
one another, use one environment per backend: UMA `.venv`, MACE `.venv-mace`,
DPA `.venv-dpa`, and GRACE `.venv-grace`. Do not install the other backends
into the UMA `.venv`.

| Engine | Backend install | Model format |
|--------|-----------------|--------------|
| MACE | Use the `.venv-mace` commands in Section 2.1 | `.model` / `.pt` |
| DPA | Use the `.venv-dpa` commands in Section 2.1 | frozen `.pt` / `.pth` |
| GRACE | Use the `.venv-grace` commands in Section 2.1 | TensorFlow SavedModel directory |

> **Do not run `uv pip install mace-torch`.** The default target of `uv pip` is
> the project `.venv`. Doctor may then find both engines but will report
> `Engine dependencies: incompatible`, and MACE checkpoints cannot load.

The model path is specified with `--model` (CLI), in the TUI config screen, or via the `MODEL_PATH` key in INCAR files.

### 2.7 Verify Installation

```bash
uv run mlipx doctor
```

This runs a comprehensive diagnostic: Python, PyTorch, CUDA, GPU compatibility, fairchem-core, mlipx, and model file. If any check fails, it prints exact fix commands.

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

### Backend Installation & Environment Isolation

`fairchem-core` ships with mlipx, so UMA works out of the box. Other engines need their backend package installed separately:

| Engine | Install command | Model format |
|--------|-----------------|--------------|
| MACE | Install only into `.venv-mace` using the complete commands below | `.model` / `.pt` |
| DPA | Install into `.venv-dpa` using the commands below | frozen `.pt` / `.pth` |
| GRACE | Install into `.venv-grace` with the commands below | TensorFlow SavedModel directory |

> ⚠️ **Environment isolation warning:** `mace-torch` pins `e3nn==0.4.4`, which **fundamentally conflicts** with `fairchem-core` (`e3nn>=0.5`) - they cannot coexist in one Python environment.
>
> **Recommended approach:** keep the `.venv` created by `uv sync` for UMA and
> create a separate `.venv-mace`. Never install `mace-torch` into the UMA
> environment:
> ```bash
> # Run from the repository root; the project does not support Python 3.13+
> uv venv --python 3.12 .venv-mace
> uv pip install --python .venv-mace/bin/python \
>   torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
> uv pip install --python .venv-mace/bin/python -e ./uma
> uv pip install --python .venv-mace/bin/python "e3nn==0.4.4" mace-torch
>
> # UMA: always use the repository's uv environment
> uv run mlipx doctor
> uv run mlipx tui
>
> # MACE: explicitly use the MACE environment
> .venv-mace/bin/mlipx doctor
> .venv-mace/bin/mlipx tui
> ```

The root workspace deliberately pins UMA to PyTorch 2.6.0+cu124 to retain the
`sm_60` kernels required by Pascal GPUs, while current fairchem-core metadata
declares `torch~=2.8.0`. Consequently,
`uv pip check --python .venv/bin/python` reports this known override. The mlipx
UMA inference path is validated with real checkpoints; this does not imply
support for every other fairchem-core feature under the non-upstream version
combination.

Recent DeePMD-kit wheels are compiled against PyTorch's CXX11 ABI, while the
PyTorch 2.6 build required by the UMA environment uses the older ABI. Keep DPA
isolated as well; an ABI mismatch may allow `import deepmd` but fails as soon as
a PyTorch model is loaded:

```bash
uv venv --python 3.12 .venv-dpa
uv pip install --no-config --python .venv-dpa/bin/python \
  "torch==2.10.0+cu126" --index-url https://download.pytorch.org/whl/cu126
uv pip install --no-config --python .venv-dpa/bin/python \
  -e ./uma "deepmd-kit[torch]==3.1.3"

.venv-dpa/bin/mlipx sp structure.vasp \
  --model models/dma/DPA-3.2-5M.pt --model-type dpa --task bulk \
  --head Domains_SSE_PBE --device cuda:0
```

Use a DeePMD-kit/PyTorch pair whose declared versions and CXX11 ABI match if
you choose other releases. On a machine without an NVIDIA GPU, use the CPU
Torch command in Section 2.1 and run with `--device cpu`. DPA accepts an
exported/frozen inference model, not a raw training checkpoint.

`DPA-3.2-5M.pt` is a multi-task model. Its `head` selects the learned
potential-energy surface, while mlipx `task=bulk|molecule` only controls PBC.
For LGPS use `--head Domains_SSE_PBE` (the branch explicitly trained on
Li–Si/Ge/Sn–P–S solid electrolytes); for general molecules use
`--head OMol25`. Run `dp show MODEL model-branch` to list branches. Never rely
on the default branch merely because a calculation completes.

GRACE uses TensorFlow. CPU inference can technically run in the UMA
environment, but use a dedicated `.venv-grace` for consistent CPU/GPU commands
and to prevent TensorFlow from replacing UMA/PyTorch CUDA libraries:

```bash
uv venv --python 3.12 .venv-grace
uv pip install --no-config --python .venv-grace/bin/python \
  -e ./uma "tensorflow[and-cuda]==2.20.0" "tensorpotential==0.6.0"

# Required by the tested V100/Volta setup; newer cuDNN 9.24 fails its
# convolution-algorithm probe on this GPU.
uv pip install --no-config --python .venv-grace/bin/python \
  "nvidia-cudnn-cu12==9.3.0.75"

.venv-grace/bin/mlipx sp structure.vasp \
  --model models/grace/GRACE-2L-SMAX-large --model-type grace \
  --task bulk --device cuda:0
```

A successful TensorFlow import or a listed GPU does not validate GPU graph
execution. Run a real TensorFlow GPU operation and a one-structure GRACE SP
smoke test before GRACE MD.

If you already ran `uv pip install mace-torch`, you do not need to delete the
repository. Restore the UMA environment and then create `.venv-mace`:

```bash
# uv removes mace-torch because it is not present in uv.lock
uv sync
uv run mlipx doctor       # MACE should be absent here; e3nn should be 0.6.x

# Now run the three --python .venv-mace/bin/python commands above
```
>
> `doctor` now checks installed distribution constraints. If `mace-torch`,
> `fairchem-core`, and an incompatible e3nn coexist, it reports
> `Engine dependencies: incompatible`. An import check still cannot prove that
> a checkpoint can be loaded, so run the model smoke test below after setup.

**MACE CUDA smoke test (required before a long MD run):**

```bash
nvidia-smi
.venv-mace/bin/python -c \
  "import torch,e3nn; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), e3nn.__version__)"

.venv-mace/bin/mlipx md test.vasp --model mace.model \
  --model-type mace --task bulk --device cuda \
  --steps 1 --save-interval 1 --no-pre-relax \
  --output /tmp/mlipx-mace-smoke --name smoke
```

Success means step 1 completes, the process exits with code 0, and complete
OUTCAR, XDATCAR, and `mlipx_results.json` files are produced. Only then submit
the long job with `.venv-mace/bin/mlipx tui`.

### Task Mapping & Periodic Boundaries (PBC)

UMA has a well-defined task system (each maps to a training dataset); other engines are task-unaware, and `task` only controls the PBC strategy:

| `task` | PBC | Equivalent UMA task | Suitable systems |
|--------|------|---------------------|-------------------|
| `bulk` | True | `omat` | Periodic crystals, surface slabs, MOFs |
| `molecule` | False | `omol` | Isolated molecules (auto charge=0) |

> **Charge/spin defaults (task-dependent):**
> - UMA `omol`: when `atoms.info` is unset, auto-fills `charge=0` and `spin=1` (spin **multiplicity**, 1 = singlet).
> - Non-UMA `molecule` (MACE/DPA/GRACE): auto-fills only `charge=0`; **no** `spin` is injected. MACE reads `atoms.info["spin"]` as **total spin S** (0 = singlet), a different semantic from UMA's multiplicity -- blindly injecting `spin=1` would force spin-sensitive MACE models into a doublet radical. Set spin explicitly if needed.
> - Either can be overridden by setting `charge`/`spin` explicitly in the input structure's `atoms.info`.
> - Molecular systems (`pbc=False`) do not compute a stress tensor.

### Complete Examples per Engine

```bash
# -- UMA (default) -- bulk material optimization
mlipx opt Li2O.cif --model uma-s-1.pt --model-type uma --task omat --cell-opt --fmax 0.02

# -- MACE -- periodic single point
.venv-mace/bin/mlipx sp bulk.cif --model mace.model --model-type mace --task bulk --device cuda

# -- MACE head selection: use a head actually listed by the loaded model
.venv-mace/bin/mlipx sp bulk.cif --model mace-mpa-0-medium.model --model-type mace --task bulk --head default

# -- MACE dtype (float32 by default for every calculation type)
.venv-mace/bin/mlipx opt bulk.cif --model mace.model --model-type mace --task bulk --dtype float64

# -- DPA (DeepMD) -- periodic optimization in its isolated environment
.venv-dpa/bin/mlipx opt bulk.vasp --model dpa.pt --model-type dpa --task bulk --fmax 0.01 --optimizer LBFGS

# -- GRACE -- high-throughput screening (batch)
mlipx batch structures/ --model grace_model/ --model-type grace --task bulk --calc-type sp --output results/
```

### MACE-specific options

| Option | Description |
|--------|-------------|
| `--head HEAD` | Select a head that is actually present in the model. Omit it for a single-head model. Head names are model/version-specific. |
| `--dtype float32\|float64` | Compute precision. The default is **`float32` for SP, optimization, and MD**. Opt into `float64` when a compatible model and high-precision energy differences require it. |

> ⚠️ `--head` takes one string. Although mace-torch itself may warn and fall
> back for an unknown name, mlipx rejects that case: silently changing heads
> changes the potential-energy surface. The bundled
> `models/mace/mace-mpa-0-medium.model` currently exposes only `default`.
> That local file stores float64 weights; the default `float32` setting makes
> mace-torch convert it at load time (and emit a warning). Use
> `--dtype float64` when preserving the stored precision matters more than
> memory and throughput.

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

Simulates the time evolution of atoms at a given temperature. Supports two ensembles:

| Ensemble | Integrator | Description |
|----------|-----------|-------------|
| NVT | Langevin | Constant particle number, volume, temperature (canonical) |
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
| `--steps` | 1000 | Number of MD steps |
| `--friction` | 0.001 | Friction coefficient (NVT only, fs⁻¹) |
| `--save-interval` | 10 | Save trajectory every N steps |
| `--seed` | generated and recorded | Random seed for reproducible velocity initialization and NVT noise |
| `--velocity-policy` | auto | `auto`, `initialize`, or `preserve` stored momenta |
| `--pre-relax` | NVT on / NVE off | Pre-relax before MD (`--no-pre-relax` to disable) |
| `--pre-relax-steps` | 50 | Max pre-relaxation steps |
| `--pre-relax-fmax` | 0.1 | Pre-relaxation force threshold (eV/Å) |

**Output files:** `OUTCAR`, `CONTCAR` (final structure), `XDATCAR` (trajectory), `trajectory.traj` (ASE format), `mlipx_results.json`

### 5.4 Batch Processing

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
                       [--cpu-threads N] [--inference-mode MODE]
                       [--activation-checkpointing | --no-activation-checkpointing]
                       [--dtype DTYPE] [--head HEAD] [--output DIR] [--name NAME]

  STRUCTURE             Input structure file (CIF, XYZ, POSCAR, VASP, etc.)
  --model MODEL         Path to model checkpoint [required]
  --model-type TYPE     MLIP engine: uma|mace|dpa|grace [default: uma]
  --task TASK           UMA: omat|omol|oc20|oc25|odac|omc; others: bulk|molecule [default: omat]
  --device DEVICE       cpu|gpu|cuda|cuda:N [default: cpu]
  --cpu-threads N       CPU intra-op threads (all engines; backend default if omitted)
  --inference-mode MODE UMA inference preset: default|turbo
  --[no-]activation-checkpointing
                        UMA memory/speed override; omitted: follow inference preset
  --dtype DTYPE         MACE dtype float32|float64 [default: float32]
  --head HEAD           MACE head or DeepMD/DPA multi-task branch
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
  --seed N              Reproducible MD seed [default: generated and recorded]
  --velocity-policy P   auto|initialize|preserve [default: auto]
  --pre-relax           Pre-relax before MD [default: on for NVT, off for NVE]
  --pre-relax-steps N   Max pre-relaxation steps [default: 50]
  --pre-relax-fmax F    Pre-relaxation force threshold eV/Å [default: 0.1]
  --fmax-abort F        Large-force warning threshold eV/Å [default: 20.0]
```

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

**Main Menu** — Select SP, OPT, MD, Jobs, Template, or Exit. Batch is currently
CLI-only.

**Configuration** — Fill in paths, engine/task and a device string (`cpu`,
`cuda`, or `cuda:N`). The **Backend & Resource Options** section exposes:

- CPU threads for every engine (PyTorch for UMA/MACE/DPA, TensorFlow for GRACE)
- UMA inference mode and activation checkpointing
- MACE precision and MACE/DPA model head or branch

Controls that do not apply to the selected engine are disabled instead of being
silently accepted. OPT also exposes cell optimization and symmetry
preservation. MD exposes the ensemble, temperature, timestep, step/save counts,
NVT friction, pre-relaxation controls, random seed, velocity policy, and
large-force warning threshold. Paths support live validation with visual
feedback:

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
# `task` sets the task for the periodic slab/adsorbed system; the isolated
# gas molecule is scored automatically with a molecular task (UMA periodic
# task -> omol; generic engine -> molecule) -- no need to set it separately.
ads_results = calculate_adsorption_energy(
    adsorbed_structure="adsorbed.cif",
    gas_structure="co2.xyz",
    surface_structure="slab.cif",
    model_path="uma-s-1.pt",
    task="oc20",          # periodic task for slab/adsorbed system
    # gas_task="omol",    # optional: override the gas molecule's task
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
| `calculate_adsorption_energy(ads, gas, surf, ...)` | `dict` | E_ads = E_adsorbed - E_gas - E_surface; gas scored with a molecular task automatically (`gas_task` overrides) |

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
| `DEVICE` | string | `cpu` | Compute device: `cpu`, `cuda`, `gpu`, or `cuda:N` |
| `INFERENCE_MODE` | string | `default` | Inference mode: `default`, `turbo` |
| `DEFAULT_DTYPE` | string | `float32` | MACE dtype for every calculation type. MACE only. |
| `HEAD` | string | - | MACE head or DeepMD/DPA multi-task branch. |

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
| `PRE_RELAX` | bool | NVT=`.TRUE.`/NVE=`.FALSE.` | Pre-relax before MD (ensemble-aware default) |
| `PRE_RELAX_STEPS` | int | `50` | Max pre-relaxation steps |
| `PRE_RELAX_FMAX` | float | `0.1` | Pre-relaxation force threshold (eV/Å) |

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
| `run.log` | All | Text | Continuously flushed live log; its path is shown by CLI and TUI |
| `batch_summary.json` | BATCH | JSON | Summary of all structures processed in batch |

Every run prints a hint like:

```text
Live log: /absolute/path/to/results/run.log
Follow live output: tail -f /absolute/path/to/results/run.log
```

`run.log` is flushed after every message, so the displayed command can be used from another terminal. On completion, the terminal log, `run.log`, the end of `OUTCAR`, and JSON output record:

- **Total elapsed time:** from the user's run request until all standard output files are generated.
- **Compute elapsed time:** from the first compute phase after the model is ready until output writing starts; model loading and final output writing are excluded.

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

**Important for molecular systems (omol):** UMA's `omol` task needs the total charge and spin multiplicity. If `atoms.info` does not set them, mlipx auto-fills `charge=0` and `spin=1` (singlet); to change this (e.g. ions, radicals, triplets) set them explicitly:

```python
from ase.io import read, write

atoms = read("molecule.xyz")
atoms.info["charge"] = 0    # Net charge
atoms.info["spin"] = 1      # Spin multiplicity = 2S+1
write("molecule.xyz", atoms)
```

For periodic systems (omat, oc20, oc25, odac, omc), PBC is automatically set to `True` and the cell is validated. For molecules (omol), PBC is set to `False`.

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

For periodic systems (omat, oc20, oc25, odac, omc), PBC is automatically set to `True` and the cell is validated. For molecules (omol), PBC is set to `False`.

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
UMA/MACE/DPA and TensorFlow intra-op threads for GRACE. The historical
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
