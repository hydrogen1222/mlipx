# mlipx — MLIP eXtended

**A VASP-style CLI / TUI / Python API for machine-learning interatomic potentials (MLIPs).**

mlipx wraps multiple MLIP engines behind one unified interface — **UMA (FAIRChem)** (default), **MACE**, **DPA (DeepMD-kit)**, and **GRACE** — so you can run single-point, optimization, molecular-dynamics, and batch calculations with VASP-compatible output (OUTCAR, CONTCAR, XDATCAR, OSZICAR) regardless of which model you choose.

Switch engines with a single flag: `--model-type` (CLI) or the `MODEL_TYPE` key (INCAR). Default is `uma`; existing UMA workflows keep working unchanged.

> ℹ️ This repository contains the full source of [mlipx](mlipx/) — a multi-engine MLIP CLI/TUI/API tool — built on top of [fairchem-core](packages/fairchem-core/). The underlying fairchem library (UMA model, training, datasets) is preserved under [`src/`](src/), [`packages/`](packages/), and [`docs/`](docs/). Originally forked from [FAIRChem](https://github.com/FAIR-Chem/fairchem).

---

## 📖 Documentation

| Language | Manual |
|----------|--------|
| 🇨🇳 中文 | [mlipx/docs/README_CN.md](mlipx/docs/README_CN.md) — 完整中文手册 |
| 🇬🇧 English | [mlipx/docs/README_EN.md](mlipx/docs/README_EN.md) — Complete reference |

Both manuals are wiki-level references covering: installation, quick start, architecture, all calculation types, CLI/TUI/API, every INCAR keyword, output-file formats, task types, the **multi-engine guide**, background jobs, resource control, worked examples, troubleshooting, and performance.

---

## 🚀 Quick Start

### Install: four engines, four isolated environments

```bash
# Clone the repository
git clone https://github.com/hydrogen1222/mlipx.git
cd mlipx

# Environment 1: UMA
uv sync
uv run mlipx doctor --engine uma --device auto
```

Each optional engine needs its own Python environment. These directories share
the repository, structures, and models; they only isolate incompatible Python
and CUDA dependencies.

| Engine | Environment | Always run it as |
|--------|-------------|------------------|
| UMA | `.venv` | `uv run mlipx ...` |
| MACE | `.venv-mace` | `.venv-mace/bin/mlipx ...` |
| DPA | `.venv-dpa` | `.venv-dpa/bin/mlipx ...` |
| GRACE | `.venv-grace` | `.venv-grace/bin/mlipx ...` |

Copy-ready installation commands for all four environments are at the very
start of the [English installation guide](mlipx/docs/README_EN.md#21-start-here-four-engines-need-four-environments)
and [中文安装教程](mlipx/docs/README_CN.md#21-小白先看四个引擎要用四个环境).
Do not install MACE, DPA, or GRACE into the UMA `.venv`.

UMA weights are not bundled with either mlipx or `fairchem-core`. They are
hosted in the manually gated
[`facebook/UMA` Hugging Face repository](https://huggingface.co/facebook/UMA),
which currently excludes China, Russia, Belarus, and comprehensively sanctioned
jurisdictions. Request access and authenticate with a gated-repository token
before downloading a checkpoint; see the language manuals for exact steps.

### Run a calculation

```bash
# Single-point energy with the default UMA engine
uv run mlipx sp structure.cif --model uma-s-1.pt --task omat --device cpu

# Geometry optimization (with cell relaxation)
uv run mlipx opt structure.cif --model uma-s-1.pt --task omat --cell-opt --fmax 0.02

# Molecular dynamics on GPU with turbo inference
uv run mlipx md structure.cif --model uma-s-1.pt --task omat --device cuda --steps 10000
```

### Use a different MLIP engine

```bash
# MACE
.venv-mace/bin/mlipx sp bulk.cif --model mace.model \
  --model-type mace --task bulk --head default --device cuda:0

# DPA / DeepMD
.venv-dpa/bin/mlipx opt bulk.cif --model dpa.pt \
  --model-type dpa --task bulk --head Domains_SSE_PBE \
  --device cuda:0 --fmax 0.05

# GRACE: --model points to the complete SavedModel directory
.venv-grace/bin/mlipx sp bulk.cif --model grace_model/ \
  --model-type grace --task bulk --device cuda:0 \
  --gpu-memory-limit-mb 6144
```

GRACE uses TensorFlow memory growth by default, so it no longer reserves the
whole visible GPU at startup. When sharing a GPU, set an explicit hard limit
with `--gpu-memory-limit-mb`; choose the value from the other job's measured
peak plus safety headroom. The example value above is not a universal default.

For MD, `raw/trajectory.traj` and `raw/md.csv` are the reproducible canonical
outputs. If VASP interoperability files are not needed during a high-throughput
run, use `--no-write-outcar --no-write-xdatcar` to avoid their duplicate text
I/O; the canonical trajectory remains enabled.

### Analyze an MD trajectory

Analysis is calculator-independent: a DPA, MACE, or GRACE trajectory can be
analyzed from the UMA `.venv` without loading the model backend. Validate the
trajectory contract before calculating MSD or transport properties:

```bash
uv run mlipx analyze results/LGPS-800K validate
uv run mlipx analyze results/LGPS-800K thermo
uv run mlipx analyze results/LGPS-800K msd \
  --mobile Li --axes x,y,z,xyz --drift-reference nonmobile
```

MSD and transport require a fixed-cell, three-dimensionally periodic trajectory
with a uniform time axis and a known wrapped/unwrapped coordinate convention;
an mlipx run marked failed, aborted, or cancelled is rejected. See the
[Analysis v2 guide](mlipx/docs/ANALYSIS.md) and
[transport definitions](mlipx/docs/TRANSPORT.md) before fitting a diffusion
coefficient; mlipx does not choose an equilibration cutoff or diffusive fitting
window automatically.

### Or use a VASP-style INCAR file

```ini
CALC_TYPE   = SP
MODEL_TYPE  = UMA            # default; or MACE / DPA / GRACE
MODEL_PATH  = uma-s-1.pt
TASK        = omat           # UMA: omat/omol/oc20/...  others: bulk/molecule
DEVICE      = cpu
```

```bash
uv run mlipx run -i INCAR.mlipx -s structure.cif
```

### Batch calculations

Batch mode is currently CLI-only and processes matching structures
sequentially while reusing one loaded model:

```bash
# UMA batch single-point calculations
uv run mlipx batch structures/ --model uma-s-1.pt \
  --model-type uma --task omat --device cuda \
  --calc-type sp --pattern "*.cif" --output batch_results

# MACE batch single-point calculations
.venv-mace/bin/mlipx batch structures/ --model mace.model \
  --model-type mace --task bulk --device cuda \
  --calc-type sp --pattern "*.cif" --output mace_batch_results
```

Each input gets its own output subdirectory and the root output directory gets
`batch_summary.json`. The current CLI does not expose `--parallel` or
`--workers`; do not include them.

### 🎟️ Job queue (Slurm-like, background)

For long or many calculations, submit **queued jobs** instead of running them
in the foreground. Queued jobs start one at a time by default (single GPU);
a scheduler promotes `PENDING` jobs to `RUNNING` and automatically starts the
next job when one finishes. Each task can use its own Python environment
(venv), engine (UMA/MACE/DPA/GRACE), model, structure and calc type, so a UMA
OPT task and a GRACE MD task can share one queue.

```bash
# 1. Describe the tasks in a JSON file
cat > tasks.json <<'JSON'
{
  "max_concurrent": 1,
  "tasks": [
    {
      "name": "opt-uma-1",
      "calc_type": "opt",
      "structure": "/path/a.cif",
      "model": "/path/uma-s-1.pt",
      "model_type": "uma",
      "device": "cuda:0",
      "options": {"fmax": 0.05}
    },
    {
      "name": "md-grace-1",
      "python": "/path/.venv-grace/bin/python",
      "calc_type": "md",
      "structure": "/path/b.cif",
      "model": "/path/grace_model/",
      "model_type": "grace",
      "options": {"ensemble": "NVE", "steps": 1000}
    }
  ]
}
JSON

# 2. Enqueue the tasks (status PENDING)
uv run mlipx queue submit tasks.json

# 3. Run the scheduler (background; max_concurrent from the task file)
uv run mlipx queue start            # stop with: mlipx queue stop
uv run mlipx queue status           # queued / running / finished counts
uv run mlipx queue pause <job-id>   # hold one pending job
uv run mlipx queue resume <job-id>  # resume one paused job

# Or run the scheduler in the foreground of a terminal
uv run mlipx queue start --foreground
```

The TUI queues every calculation it submits (visible as `pending` in the
Jobs screen) and provides **Start/Stop Scheduler** controls plus a concurrency
setting for multi-GPU machines. Select a pending row and use **Pause Job** or
**Resume Job** (also `P`/`U`) to control one task; **Pause Queue** and
**Resume Queue** remain available for the whole pending queue. Running jobs are
never affected. `mlipx jobs` / `mlipx kill` / `mlipx clean`
manage individual jobs; `mlipx convert-xdatcar` re-emits a trajectory in the
exact standard VASP XDATCAR layout (unwrapped coordinates).

---

## 🧰 Interfaces

| Interface | Command | Best for |
|-----------|---------|----------|
| **CLI** | Use the command prefix for the selected engine in the table above | Scripts, HPC jobs, automation |
| **TUI** | `<engine-command-prefix> tui` | Interactive SP/OPT/MD, live progress |
| **Python API** | `from mlipx.api import ...` | Workflows, custom analysis |
| **INCAR** | `mlipx run -i INCAR` | VASP-style batch configuration |

Run `mlipx doctor` for a side-effect-free package inventory. For an actual
readiness check, select the same engine and device you will use, for example
`uv run mlipx doctor --engine uma --device cuda:0`.

---

## 🔌 Supported Engines

| `MODEL_TYPE` / `--model-type` | Engine | Backend package | Task values |
|-------------------------------|--------|-----------------|-------------|
| `uma` (default, alias `fairchem`) | UMA — FAIRChem | `fairchem-core` | `omat` / `omol` / `oc20` / `oc25` / `odac` / `omc` |
| `mace` | MACE | `mace-torch` | `bulk` / `molecule` |
| `dpa` | DPA — DeepMD-kit | `deepmd-kit` | `bulk` / `molecule` |
| `grace` | GRACE | `tensorpotential` | `bulk` / `molecule` |

> ⚠️ **Environment isolation is required.** MACE conflicts with UMA's e3nn;
> DPA requires a different PyTorch ABI/version; GRACE uses TensorFlow and a
> separate cuDNN runtime. Use the four command prefixes shown above rather than
> a globally installed `mlipx`. See the copy-ready
> [English](mlipx/docs/README_EN.md#21-start-here-four-engines-need-four-environments)
> or [Chinese](mlipx/docs/README_CN.md#21-小白先看四个引擎要用四个环境)
> installation guide.

---

## ✨ Features

- **Multi-engine:** UMA / MACE / DPA / GRACE via one ASE Calculator interface
- **Calculation types:** Single-point (SP), geometry optimization (OPT, FIRE/BFGS/LBFGS), molecular dynamics (MD, NVT/NVE), batch processing
- **Trajectory analysis:** validation, thermodynamics, RDF/coordination,
  directional MSD, diffusion/transport, density maps, VACF, and velocity spectra
- **VASP-compatible output:** OUTCAR, CONTCAR, XDATCAR, OSZICAR, JSON
- **Background jobs:** submit, detach, re-attach, kill long-running calculations
- **INCAR files:** VASP-style `KEY = VALUE` configuration
- **Cross-platform:** Windows, Linux, macOS | CPU & CUDA
- **Resource control in TUI and CLI:** indexed GPU selection, backend CPU
  threads, GRACE TensorFlow GPU limits, and UMA activation
  checkpointing/inference mode
- **Live progress:** structured progress events, indeterminate spinner for SP, step counter for OPT/MD

---

## 📦 Package Layout

```
mlipx/
├── README.md                  # Package README (links to full manuals)
├── docs/
│   ├── README_CN.md           # 中文完整手册
│   ├── ANALYSIS.md            # Trajectory analysis contract and commands
│   ├── README_EN.md           # English complete manual
│   └── TRANSPORT.md           # Diffusion and conductivity definitions
├── mlipx/                     # The Python package
│   ├── analysis/              # Calculator-independent trajectory analysis
│   ├── engine.py              # CalculationEngine (unified execution)
│   ├── base_calculator.py     # BaseMLIPCalculator abstract interface
│   ├── calculator.py          # UMACalculator wrapper
│   ├── calculators/           # MACE/DPA/GRACE wrappers + Factory
│   ├── config.py              # INCAR config parser
│   ├── api.py                 # Python API functions
│   ├── cli.py                 # CLI (argparse, sp/opt/md/batch/run/config/queue/...)
│   ├── runners/               # SinglePoint, Optimization, MD, Batch
│   ├── tui/                   # Textual TUI (app, screens)
│   └── writers/               # OUTCAR, CONTCAR, XDATCAR, OSZICAR, JSON
├── templates/                 # INCAR template files
└── examples/                  # Example scripts
```

---

## 📄 License

MIT License. mlipx is built upon code from [FAIRChem](https://github.com/FAIR-Chem/fairchem) (Copyright © Meta Platforms, Inc. and affiliates), licensed under MIT. See [`mlipx/LICENSE`](mlipx/LICENSE) and [`LICENSE.md`](LICENSE.md).

## 🙏 Acknowledgements

mlipx builds on [FAIRChem](https://github.com/FAIR-Chem/fairchem) and its UMA foundation model. Multi-engine integration supports [MACE](https://github.com/ACEsuit/mace), [DeepMD-kit](https://github.com/deepmodeling/deepmd-kit), and [GRACE](https://github.com/IBM/grace).
