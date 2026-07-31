# mlipx — MLIP eXtended

**A VASP-style CLI / TUI / Python API for machine-learning interatomic potentials (MLIPs).**

mlipx wraps multiple MLIP engines behind one unified interface — **UMA (FAIRChem)** (default), **MACE**, **DPA (DeepMD-kit)**, and **GRACE** — so you can run single-point, optimization, molecular-dynamics, and batch calculations with VASP-compatible output (OUTCAR, CONTCAR, XDATCAR, OSZICAR) regardless of which model you choose.

Switch engines with a single flag: `--model-type` (CLI) or the `MODEL_TYPE` key (INCAR). Default is `uma`; existing UMA workflows keep working unchanged.

> ℹ️ This repository contains the full source of [mlipx](uma/) — a multi-engine MLIP CLI/TUI/API tool — built on top of [fairchem-core](packages/fairchem-core/). The underlying fairchem library (UMA model, training, datasets) is preserved under [`src/`](src/), [`packages/`](packages/), and [`docs/`](docs/). Originally forked from [FAIRChem](https://github.com/FAIR-Chem/fairchem).

---

## 📖 Documentation

| Language | Manual |
|----------|--------|
| 🇨🇳 中文 | [uma/docs/README_CN.md](uma/docs/README_CN.md) — 完整中文手册 |
| 🇬🇧 English | [uma/docs/README_EN.md](uma/docs/README_EN.md) — Complete reference |

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
uv run mlipx doctor
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
start of the [English installation guide](uma/docs/README_EN.md#21-start-here-four-engines-need-four-environments)
and [中文安装教程](uma/docs/README_CN.md#21-小白先看四个引擎要用四个环境).
Do not install MACE, DPA, or GRACE into the UMA `.venv`.

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
  --model-type grace --task bulk --device cuda:0
```

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

---

## 🧰 Interfaces

| Interface | Command | Best for |
|-----------|---------|----------|
| **CLI** | Use the command prefix for the selected engine in the table above | Scripts, HPC jobs, automation |
| **TUI** | `<engine-command-prefix> tui` | Interactive SP/OPT/MD, live progress |
| **Python API** | `from mlipx.api import ...` | Workflows, custom analysis |
| **INCAR** | `mlipx run -i INCAR` | VASP-style batch configuration |

Run `mlipx doctor` to diagnose your Python/PyTorch/CUDA setup and check which MLIP engine backends are installed.

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
> [English](uma/docs/README_EN.md#21-start-here-four-engines-need-four-environments)
> or [Chinese](uma/docs/README_CN.md#21-小白先看四个引擎要用四个环境)
> installation guide.

---

## ✨ Features

- **Multi-engine:** UMA / MACE / DPA / GRACE via one ASE Calculator interface
- **Calculation types:** Single-point (SP), geometry optimization (OPT, FIRE/BFGS/LBFGS), molecular dynamics (MD, NVT/NVE), batch processing
- **VASP-compatible output:** OUTCAR, CONTCAR, XDATCAR, OSZICAR, JSON
- **Background jobs:** submit, detach, re-attach, kill long-running calculations
- **INCAR files:** VASP-style `KEY = VALUE` configuration
- **Cross-platform:** Windows, Linux, macOS | CPU & CUDA
- **Resource control in TUI and CLI:** indexed GPU selection, backend CPU
  threads, and UMA activation checkpointing/inference mode
- **Live progress:** structured progress events, indeterminate spinner for SP, step counter for OPT/MD

---

## 📦 Package Layout

```
uma/
├── README.md                  # Package README (links to full manuals)
├── docs/
│   ├── README_CN.md           # 中文完整手册
│   └── README_EN.md           # English complete manual
├── mlipx/                     # The Python package
│   ├── engine.py              # CalculationEngine (unified execution)
│   ├── base_calculator.py     # BaseMLIPCalculator abstract interface
│   ├── calculator.py          # UMACalculator wrapper
│   ├── calculators/           # MACE/DPA/GRACE wrappers + Factory
│   ├── config.py              # INCAR config parser
│   ├── api.py                 # Python API functions
│   ├── cli.py                 # CLI (argparse, 10 subcommands)
│   ├── runners/               # SinglePoint, Optimization, MD, Batch
│   ├── tui/                   # Textual TUI (app, screens)
│   └── writers/               # OUTCAR, CONTCAR, XDATCAR, OSZICAR, JSON
├── templates/                 # INCAR template files
└── examples/                  # Example scripts
```

---

## 📄 License

MIT License. mlipx is built upon code from [FAIRChem](https://github.com/FAIR-Chem/fairchem) (Copyright © Meta Platforms, Inc. and affiliates), licensed under MIT. See [`uma/LICENSE`](uma/LICENSE) and [`LICENSE.md`](LICENSE.md).

## 🙏 Acknowledgements

mlipx builds on [FAIRChem](https://github.com/FAIR-Chem/fairchem) and its UMA foundation model. Multi-engine integration supports [MACE](https://github.com/ACEsuit/mace), [DeepMD-kit](https://github.com/deepmodeling/deepmd-kit), and [GRACE](https://github.com/IBM/grace).
