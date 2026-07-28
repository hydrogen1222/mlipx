# mlipx — MLIP eXtended

**A VASP-style CLI / TUI / Python API for machine-learning interatomic potentials (MLIPs).**

mlipx wraps multiple MLIP engines behind one unified interface — **UMA (FAIRChem)** (default), **MACE**, **DPA (DeepMD-kit)**, and **GRACE** — so you can run single-point, optimization, molecular-dynamics, and batch calculations with VASP-compatible output (OUTCAR, CONTCAR, XDATCAR, OSZICAR) regardless of which model you choose.

Switch engines with a single flag: `--model-type` (CLI) or the `MODEL_TYPE` key (INCAR). Default is `uma`; existing UMA workflows keep working unchanged.

> ℹ️ This repository is a fork of [FAIRChem](https://github.com/FAIR-Chem/fairchem) and contains its full source. The mlipx package lives in [`uma/`](uma/) and is built on top of `fairchem-core`. FAIRChem's own documentation is preserved under [`docs/`](docs/) and [`packages/`](packages/).

---

## 📖 Documentation

| Language | Manual |
|----------|--------|
| 🇨🇳 中文 | [uma/docs/README_CN.md](uma/docs/README_CN.md) — 完整中文手册 |
| 🇬🇧 English | [uma/docs/README_EN.md](uma/docs/README_EN.md) — Complete reference |

Both manuals are wiki-level references covering: installation, quick start, architecture, all calculation types, CLI/TUI/API, every INCAR keyword, output-file formats, task types, the **multi-engine guide**, background jobs, resource control, worked examples, troubleshooting, and performance.

---

## 🚀 Quick Start

### Install

```bash
# From the repository root (Python 3.12, managed by uv)
uv sync                       # creates .venv and installs all locked deps
uv pip install -e uma/        # install the mlipx package (editable)
```

GPU users: see the [installation section](uma/docs/README_EN.md#2-installation) of the manual for CUDA/PyTorch matching by GPU architecture.

### Run a calculation

```bash
# Single-point energy with the default UMA engine
mlipx sp structure.cif --model uma-s-1.pt --task omat --device cpu

# Geometry optimization (with cell relaxation)
mlipx opt structure.cif --model uma-s-1.pt --task omat --cell-opt --fmax 0.02

# Molecular dynamics on GPU with turbo inference
mlipx md structure.cif --model uma-s-1.pt --task omat --device cuda --steps 10000
```

### Use a different MLIP engine

```bash
# MACE (install backend first: pip install mace-torch)
mlipx sp bulk.cif --model mace.model --model-type mace --task bulk

# DPA / DeepMD (pip install 'deepmd-kit>=3.0.0')
mlipx opt bulk.cif --model dpa2.pth --model-type dpa --task bulk --fmax 0.05

# GRACE (pip install tensorpotential)
mlipx sp bulk.cif --model grace_model/ --model-type grace --task bulk
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
mlipx run -i INCAR.mlipx -s structure.cif
```

---

## 🧰 Interfaces

| Interface | Command | Best for |
|-----------|---------|----------|
| **CLI** | `mlipx <command>` | Scripts, HPC jobs, automation |
| **TUI** | `mlipx tui` | Interactive exploration, live progress |
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

> ⚠️ **Engine isolation note:** `mace-torch` pins `e3nn==0.4.4`, which conflicts with `fairchem-core` (`e3nn>=0.5`). MACE/DPA/GRACE backends must be installed in a **separate environment** from `fairchem-core`. See the [Multi-Engine Guide](uma/docs/README_EN.md#multi-engine-guide) for details.

---

## ✨ Features

- **Multi-engine:** UMA / MACE / DPA / GRACE via one ASE Calculator interface
- **Calculation types:** Single-point (SP), geometry optimization (OPT, FIRE/BFGS/LBFGS), molecular dynamics (MD, NVT/NVE), batch processing
- **VASP-compatible output:** OUTCAR, CONTCAR, XDATCAR, OSZICAR, JSON
- **Background jobs:** submit, detach, re-attach, kill long-running calculations
- **INCAR files:** VASP-style `KEY = VALUE` configuration
- **Cross-platform:** Windows, Linux, macOS | CPU & CUDA
- **Resource control:** CPU threads, GPU memory (activation checkpointing), inference mode (default/turbo)
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
