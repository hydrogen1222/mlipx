# mlipx - MLIP eXtended

A VASP-compatible CLI / TUI / Python API for machine-learning interatomic
potentials (UMA, MACE, DPA, GRACE). Built upon code originally developed by
Meta FAIR as part of the FAIRChem project.

## Quick Start

```bash
# 1. Detect your GPU and get the matching PyTorch command (CPU-only users can skip).
uv run mlipx setup

# 2. Create a pinned venv (Python 3.12) + install everything from the lockfile.
#    uv auto-creates .venv. Do NOT use `pip install -r requirements.txt`
#    (that's the upstream CI snapshot, torch 2.8 - conflicts with this fork).
uv sync

# 3. Verify & run
uv run mlipx doctor              # comprehensive environment diagnostic
uv run mlipx --help              # show all commands
uv run mlipx tui                 # launch interactive TUI
uv run mlipx template sp         # generate INCAR template
```

> **Note:** All commands use `uv run` prefix which auto-detects the `.venv`. Alternatively, activate the venv first: `source .venv/bin/activate` (Linux/Mac) or `.venv\Scripts\activate` (Windows), then run `mlipx` directly.

### CUDA vs CPU Installation

| Machine | engine install | mlipx usage |
|---------|----------------------|--------------|
| **CUDA GPU** | Install engine (e.g. fairchem-core) with CUDA PyTorch | `--device cuda` (auto-detected) |
| **CPU only** | engine uses CPU PyTorch by default | `--device cpu` (default) |

mlipx itself does not ship PyTorch - it inherits whatever PyTorch the engine backend provides. Verify CUDA availability:
```bash
uv run python -c "import torch; print('CUDA OK' if torch.cuda.is_available() else 'CPU only')"
```

> **GPU install guidance (supports GTX 900 series and newer):** The right PyTorch build depends on your GPU's compute capability. Run this **before** installing torch - it uses `nvidia-smi` and works even with no PyTorch installed yet:
> ```bash
> uv run mlipx setup      # detect GPU + print the exact torch install command
> uv run mlipx doctor     # verify after install
> ```
> Supported floor is **Maxwell (GTX 900 series, e.g. GTX 960)**; Kepler (GTX 700/600) is not supported (no prebuilt PyTorch wheel). The recommendation table:
>
> | GPU family | CC | Recommended torch |
> |---|---|---|
> | Maxwell (GTX 750/9xx) | sm_50/52 | `torch==2.6.0+cu124` |
> | Pascal (GTX 10xx, P104-100) | sm_60/61 | `torch==2.6.0+cu124` |
> | Volta–Hopper (V100…H100, RTX 20/30/40) | sm_70–90 | `torch==2.6.0+cu124` |
> | Blackwell (RTX 50) | sm_100/120 | `torch==2.8.0+cu128` |
> | Kepler (GTX 700/600) | sm_30/37 | not supported |
>
> Why: PyTorch 2.7+ dropped `sm_50`/`sm_60` from its prebuilt CUDA wheels, so old cards (Maxwell/Pascal) must stay on `torch 2.6.0+cu124`; its `sm_50`/`sm_60` kernels are binary-compatible with `sm_52`/`sm_61`. The workspace already pins `torch==2.6.0+cu124` by default, so `uv sync` works out of the box for Maxwell–Hopper; only Blackwell needs an override. If the download fails, enable a proxy first: `clashctl on`.

## 使用方法 / Usage

mlipx 默认使用 **UMA (FAIRChem)** 引擎，现有用户无需修改任何配置即可继续使用。
要切换到其他 MLIP 引擎（MACE / DPA / GRACE），只需指定 `--model-type` 或在 INCAR 中设置 `MODEL_TYPE`。

### 引擎类型

| `--model-type` / `MODEL_TYPE` | 引擎 | 后端包 | task 取值 |
|------------------------------|------|--------|-----------|
| `uma`（默认） | UMA (FAIRChem) | `fairchem-core` | `omat`/`omol`/`oc20`/`oc25`/`odac`/`omc` |
| `mace` | MACE | `mace-torch` | `bulk`/`molecule` |
| `dpa` | DPA (DeepMD-kit) | `deepmd-kit` | `bulk`/`molecule` |
| `grace` | GRACE | `tensorpotential` | `bulk`/`molecule` |

> 非 UMA 引擎没有 "task" 概念，`task` 仅作为周期性提示：`bulk`=周期性体系（PBC=True），`molecule`=分子（PBC=False）。

### 1. 默认 UMA 计算（与旧版完全兼容）

```bash
# 单点能
mlipx sp structure.cif --model uma-s-1.pt --task omat --device cpu

# 几何优化（含晶胞弛豫）
mlipx opt structure.cif --model uma-s-1.pt --task omat --cell-opt --fmax 0.02

# 分子动力学（GPU + turbo 模式）
mlipx md structure.cif --model uma-s-1.pt --task omat --device cuda --steps 10000
```

### 2. 使用其他引擎（MACE / DPA / GRACE）

```bash
# MACE 单点能（需先 pip install mace-torch）
mlipx sp structure.cif --model mace.model --model-type mace --task bulk --device cpu

# DPA 几何优化（需先 pip install 'deepmd-kit>=3.0.0'）
mlipx opt structure.cif --model dpa2.pth --model-type dpa --task bulk --fmax 0.05

# GRACE 单点能（需先 pip install tensorpotential）
mlipx sp structure.cif --model grace_model/ --model-type grace --task bulk
```

### 3. 通过 INCAR 文件运行（VASP 风格）

```bash
mlipx run -i INCAR.mlipx -s structure.cif
```

INCAR 中用 `MODEL_TYPE` 指定引擎：

```ini
CALC_TYPE   = SP
MODEL_TYPE  = MACE          # 默认 UMA；可选 MACE/DPA/GRACE
MODEL_PATH  = mace.model
TASK        = bulk          # UMA 用 omat/omol/...；其他引擎用 bulk/molecule
DEVICE      = cpu
```

### 4. 检查环境与已安装的引擎

```bash
mlipx doctor                # 诊断 Python/PyTorch/CUDA/各引擎后端
mlipx template sp           # 生成 INCAR 模板
mlipx tui                   # 交互式界面（可下拉选择引擎）
```


## Interfaces

| Interface | How to use | Best for |
|-----------|-----------|----------|
| **CLI** | `mlipx <command>` in terminal | Scripts, HPC jobs, automation |
| **TUI** | `mlipx tui` | Interactive exploration, live progress |
| **Python API** | `from mlipx.api import ...` | Workflows, custom analysis |

## Documentation

- 📖 **[English User Manual](docs/README_EN.md)** - Complete wiki-level reference: installation, quick start, architecture, calculation types, CLI/TUI/API, all INCAR keywords, output files, task types, **multi-engine guide**, background jobs, resource control, worked examples, troubleshooting, performance
- 📖 **[中文用户手册](docs/README_CN.md)** - 完整中文 wiki 级手册：安装、快速开始、架构、计算类型、CLI/TUI/API、全部 INCAR 关键字、输出文件、任务类型、**多引擎指南**、后台任务、资源控制、示例、故障排除、性能指南
- 📚 Examples are integrated into both manuals (§14) - covering SP, OPT, MD, batch, EOS, NEB, phonons, formation energy

## Features

- **Multi-engine support:** UMA (FAIRChem), MACE, DPA (DeepMD-kit), GRACE via a unified ASE Calculator interface
- **Calculation types:** Single-point (SP), Geometry optimization (OPT, FIRE/BFGS/LBFGS), Molecular dynamics (MD, NVT/NVE), Batch processing
- **VASP-compatible output:** OUTCAR, CONTCAR, XDATCAR, OSZICAR formats
- **Background jobs:** Submit, detach, re-attach, kill long-running calculations
- **INCAR files:** VASP-style `KEY = VALUE` configuration format
- **Cross-platform:** Windows, Linux, macOS | CPU & CUDA
- **Resource control:** CPU threads, GPU memory (activation checkpointing), inference mode (default/turbo)
- **Live progress:** Structured progress events, indeterminate spinner for SP, step counter for OPT/MD

## Package Structure

```
uma/
├── mlipx -> mlipx/cli.py:main       # CLI entry point
├── mlipx/
│   ├── engine.py                     # CalculationEngine (unified execution)
│   ├── protocols.py                  # ProgressEvent protocol
│   ├── jobs.py                       # JobManager (background tasks)
│   ├── base_calculator.py            # BaseMLIPCalculator abstract interface
│   ├── calculator.py                 # UMACalculator wrapper
│   ├── calculators/                  # MACE/DPA/GRACE wrappers + Factory
│   ├── config.py                     # INCAR config parser
│   ├── api.py                        # Python API functions
│   ├── cli.py                        # CLI (argparse, 10 subcommands)
│   ├── runners/                      # SinglePoint, Optimization, MD, Batch
│   ├── tui/                          # Textual TUI (app, screens)
│   └── writers/                      # OUTCAR, CONTCAR, XDATCAR, OSZICAR, JSON
├── docs/                             # Manuals and examples
├── templates/                        # INCAR template files
└── examples/                         # Example scripts
```

## License

MIT License - Copyright (c) Meta Platforms, Inc. and affiliates.
