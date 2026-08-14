# mlipx — 用户手册

> **通用材料应用计算器**
> 面向 VASP 工作流的多引擎机器学习原子间势函数接口

---

## 目录

- [1. 简介](#1-简介)
- [2. 安装](#2-安装)
- [3. 快速开始](#3-快速开始)
- [4. 架构概览](#4-架构概览)
- [多引擎指南](#多引擎指南)
- [5. 计算类型](#5-计算类型)
  - [5.1 单点能计算 (SP)](#51-单点能计算-sp)
  - [5.2 几何优化 (OPT)](#52-几何优化-opt)
  - [5.3 分子动力学 (MD)](#53-分子动力学-md)
  - [5.4 内置轨迹后处理](#54-内置轨迹后处理analysis-v2)
  - [5.5 批量处理](#55-批量处理)
- [6. 用户界面](#6-用户界面)
  - [6.1 CLI — 命令行界面](#61-cli--命令行界面)
  - [6.2 TUI — 终端交互界面](#62-tui--终端交互界面)
  - [6.3 Python API](#63-python-api)
- [7. INCAR 配置文件参考](#7-incar-配置文件参考)
- [8. 输出文件参考](#8-输出文件参考)
- [9. 任务类型参考](#9-任务类型参考)
- [10. 后台任务管理](#10-后台任务管理)
- [11. 资源控制](#11-资源控制)
- [12. 故障排除与常见问题](#12-故障排除与常见问题)
- [13. 性能指南](#13-性能指南)
- [14. 使用示例](#14-使用示例)
- [15. 许可证](#15-许可证)

---

## 1. 简介

mlipx (MLIP eXtended) 是一个多引擎机器学习原子间势函数（MLIP）计算工具，提供类似 VASP 的用户体验。它支持多种 MLIP 后端：

Analysis v2 已恢复面向固态离子输运的、与计算器后端解耦的轨迹分析。
它使用显式的时间、PBC、阶段、单位与 provenance 数据契约，且不会导入
archive 中的 Analysis v1。使用方法见本手册
[第 5.4 节](#54-内置轨迹后处理analysis-v2)；专题文档保留更细的算法定义和
参考文献。

| 引擎 | `MODEL_TYPE` | 后端包 |
|------|--------------|--------|
| **UMA (FAIRChem)**（默认） | `uma` | `fairchem-core` |
| MACE | `mace` | `mace-torch` |
| DPA (DeepMD-kit) | `dpa` | `deepmd-kit` |
| GRACE | `grace` | `tensorpotential` |

所有引擎都通过统一的 ASE Calculator 接口接入，上层计算逻辑（单点、优化、MD、批处理、输出）完全引擎无关。默认使用 UMA，现有用户无需修改任何配置。

**mlipx 能够做什么：**

- 计算晶体结构和分子的能量、力和应力
- 优化原子位置和晶胞参数（几何弛豫）
- 运行分子动力学模拟（NVT / NVE 系综）
- 验证并分析 MD 轨迹，包括热力学、RDF、MSD、扩散和 VACF
- 批量处理数百个结构
- 输出 VASP 语法兼容的 CONTCAR/XDATCAR、VASP-like OUTCAR 和 OSZICAR

**工作原理：**

与 VASP 自洽求解 Kohn-Sham 方程不同，mlipx 使用预训练的神经网络在单次前向传播中预测能量和力。没有电子步，没有 SCF 循环，也没有 k 点。计算成本大致与原子数成线性关系。

```
                          ┌──────────────────────┐
  structure.cif  ────────▶│  MLIP 神经网络        │───────▶  能量、力、应力
  (原子坐标)              │  (UMA/MACE/DPA/GRACE) │         (单次前向传播)
                          └──────────────────────┘
```

**核心特性一览：**

| 特性 | 说明 |
|------|------|
| 多引擎支持 | UMA / MACE / DPA / GRACE，统一接口，`--model-type` 切换 |
| CLI 模式 | 完整的命令行界面（sp/opt/md/batch/run/config/queue/...） |
| TUI 模式 | 交互式终端界面，实时进度显示 |
| Python API | 面向脚本和工作流的编程接口 |
| 轨迹后处理 | 与模型后端解耦的验证、热力学、RDF、MSD、输运和 VACF |
| 后台任务 | 提交、分离、重连、终止长时间计算 |
| 批量处理 | 模型只加载一次，顺序处理多个结构 |
| CPU & CUDA | 支持 CPU 和 GPU，自动检测 |
| VASP 生态输出 | 语法兼容的 CONTCAR/XDATCAR、VASP-like OUTCAR、OSZICAR |
| 跨平台 | Windows、Linux、macOS |

---

## 2. 安装

### 2.1 小白先看：四个引擎要用四个环境

不要把四个后端装进同一个 Python 环境。最稳妥、也是本项目实际验证过的布局是：

| 你要使用的引擎 | 运行命令前缀 | 独立环境 | 主要原因 |
|---|---|---|---|
| UMA | `uv run mlipx` | `.venv` | 项目默认环境，包含 fairchem-core |
| MACE | `.venv-mace/bin/mlipx` | `.venv-mace` | MACE 与 UMA 需要不同版本的 e3nn |
| DPA | `.venv-dpa/bin/mlipx` | `.venv-dpa` | DeepMD 需要 PyTorch 2.10 和 CXX11 ABI |
| GRACE | `.venv-grace/bin/mlipx` | `.venv-grace` | TensorFlow 的 CUDA/cuDNN 不能覆盖 PyTorch 环境 |

下面的版本组合是本仓库截至 2026-08-14 的**已验证配置**，不是对所有硬件的
“永远最新版”承诺。换版本前应先核对
[uv 安装文档](https://docs.astral.sh/uv/getting-started/installation/)、
[DeePMD 安装文档](https://docs.deepmodeling.com/projects/deepmd/en/latest/getting-started/install.html)
和 [GRACE/TensorPotential 安装文档](https://gracemaker.readthedocs.io/en/latest/gracemaker/install/)，
然后重新执行 doctor 的运行时与模型探针。

这里的“环境”只是四个互不干扰的 Python 软件目录，不是四份源代码。四套环境
共用当前仓库、结构文件和模型文件。请在仓库根目录依次执行下面的命令；只使用
某一个引擎时，只安装对应的一段即可。

#### 第 0 步：安装 uv 并进入仓库

如果终端里执行 `uv --version` 已有输出，可以跳过安装 uv：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/hydrogen1222/mlipx.git
cd mlipx
```

#### 第 1 套：UMA（默认，建议先装）

```bash
uv sync --frozen
uv run mlipx doctor --engine uma --device auto
```

以后运行 UMA 时始终使用 `uv run mlipx`：

```bash
uv run mlipx sp structure.vasp \
  --model models/uma/uma-s-1.pt --model-type uma \
  --task omat --device cuda:0
```

#### 第 2 套：MACE

```bash
uv venv --python 3.12 .venv-mace
uv pip install --no-config --python .venv-mace/bin/python \
  "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124
uv pip install --no-config --python .venv-mace/bin/python \
  -e ./mlipx "e3nn==0.4.4" "mace-torch==0.3.16"
.venv-mace/bin/mlipx doctor --engine mace --device auto
```

运行示例：

```bash
.venv-mace/bin/mlipx sp structure.vasp \
  --model models/mace/mace-mpa-0-medium.model --model-type mace \
  --task bulk --head default --device cuda:0
```

#### 第 3 套：DPA / DeepMD（GPU 推荐）

```bash
uv venv --python 3.12 .venv-dpa
uv pip install --no-config --python .venv-dpa/bin/python \
  "torch==2.10.0+cu126" --index-url https://download.pytorch.org/whl/cu126
uv pip install --no-config --python .venv-dpa/bin/python \
  -e ./mlipx "deepmd-kit[torch]==3.1.3"
.venv-dpa/bin/mlipx doctor --engine dpa --device auto
```

运行 LGPS 等固态电解质时，不要省略正确的模型分支：

```bash
.venv-dpa/bin/mlipx sp structure.vasp \
  --model models/dpa/DPA-3.2-5M.pt --model-type dpa \
  --task bulk --head Domains_SSE_PBE --device cuda:0
```

没有 NVIDIA GPU 时，只有第一条 Torch 安装命令不同；其余命令保持不变：

```bash
uv pip install --no-config --python .venv-dpa/bin/python \
  "torch==2.10.0" --index-url https://download.pytorch.org/whl/cpu
```

#### 第 4 套：GRACE（GPU）

```bash
uv venv --python 3.12 .venv-grace
uv pip install --no-config --python .venv-grace/bin/python \
  -e ./mlipx "tensorflow[and-cuda]==2.20.0" "tensorpotential==0.6.0"

# V100/Volta 使用已验证的 cuDNN 版本
uv pip install --no-config --python .venv-grace/bin/python \
  "nvidia-cudnn-cu12==9.3.0.75"
.venv-grace/bin/python -c \
  "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"
.venv-grace/bin/mlipx doctor --engine grace --device auto
```

GRACE 的 `--model` 必须指向整个 SavedModel 目录：

```bash
.venv-grace/bin/mlipx sp structure.vasp \
  --model models/grace/GRACE-2L-SMAX-large --model-type grace \
  --task bulk --device cuda:0
```

#### 安装完成后怎么判断自己用对了环境？

看命令开头即可：

```text
UMA    -> uv run mlipx ...
MACE   -> .venv-mace/bin/mlipx ...
DPA    -> .venv-dpa/bin/mlipx ...
GRACE  -> .venv-grace/bin/mlipx ...
```

TUI 也遵循同一规则。例如 MACE 必须运行
`.venv-mace/bin/mlipx tui`；TUI 内切换 `MODEL_TYPE` 不会自动切换 Python
环境。模型不会随环境自动下载，需要自行放到 `models/` 或其他已知路径。

### 2.2 环境要求

| 需求 | 最低配置 | 推荐配置 |
|------|----------|----------|
| Python | 3.10–3.12 | 3.12 |
| 内存 | 8 GB | 32 GB |
| 磁盘 | 15 GB（单个环境与小模型） | 60+ GB（四个环境与多个模型） |
| GPU（可选） | 与所选 PyTorch/TensorFlow wheel 匹配的 NVIDIA 驱动 | 已验证的 CUDA 12.x 组合，8+ GB 显存 |

### 2.3 UMA 环境说明

`fairchem-core` 虽已发布到
[PyPI](https://pypi.org/project/fairchem-core/)，本仓库根 workspace 明确使用
`packages/fairchem-core/` 的本地成员。因此 §2.1 的 `uv sync --frozen` 安装的是
锁文件与本地源码组合；不要再用裸 `uv pip install` 覆盖 `.venv`。需要查看 GPU
架构和推荐 wheel 时运行 `uv run mlipx setup`。

### 2.4 CUDA GPU 与 CPU

根 workspace 通过 `uv.lock` 安装并固定 PyTorch；不要再用裸
`uv pip install torch` 覆盖它。

| 场景 | PyTorch | mlipx 设备参数 |
|------|---------|-----------------|
| **CUDA GPU 机器** | 在 CUDA Python 环境中安装 fairchem-core | `--device cuda` |
| **纯 CPU 机器** | 在标准 Python 环境中安装 fairchem-core | `--device cpu`（默认） |

**安装后验证 CUDA：**

```bash
uv run python -c "import torch; print('CUDA 可用' if torch.cuda.is_available() else '仅 CPU')"
```

**如果有 GPU 但 CUDA 不可用：**
- 运行 `uv run mlipx setup` 查看 GPU 架构建议
- 确认安装的是文档推荐的 CUDA wheel，而不是 CPU wheel

### 2.4.1 已验证配置

以下配置已经过测试确认可用：

| 组件 | 详情 |
|------|------|
| **GPU** | NVIDIA Tesla V100-SXM2-16GB |
| **架构** | Volta，计算能力 7.0（sm_70）|
| **显存** | 16 GB |
| **驱动** | 580.173.02 |
| **UMA** | PyTorch 2.6.0+cu124 |
| **MACE** | mace-torch 0.3.16，PyTorch 2.6.0+cu124 |
| **DPA** | deepmd-kit 3.1.3，PyTorch 2.10.0+cu126 |
| **GRACE** | TensorFlow 2.20.0，cuDNN 9.3.0.75 |
| **Python** | 3.12.13 |
| **操作系统** | Linux |
| **安装说明** | 模型不会由 `uv sync` 自动下载 |

*这不是最低配置要求，而是一套已知可用的 Linux/NVIDIA GPU 参考配置。更低规格
的 GPU（如 GTX 10 系列 Pascal，sm_61）和纯 CPU 环境也可使用，但安装命令
可能不同。*

### 2.5 命令前缀与环境激活

以下两种写法只针对 UMA `.venv`，彼此等价：

```bash
# 方式 A：uv run（推荐——自动检测 .venv，跨平台通用）
uv run mlipx --help
uv run mlipx tui
uv run mlipx sp structure.cif --model uma-s-1.pt

# 方式 B：先激活 venv
source .venv/bin/activate      # Linux/Mac
# .venv\Scripts\activate       # Windows
mlipx --help
mlipx tui
```

其他后端把 `.venv` 换成 `.venv-mace`、`.venv-dpa` 或 `.venv-grace`。直接写
`环境/bin/mlipx` 最不容易误用解释器，也适合脚本和 TUI。

### 2.6 模型检查点

**UMA（默认引擎）：** 检查点托管在 gated 的
[Hugging Face `facebook/UMA`](https://huggingface.co/facebook/UMA) 仓库，
不在 GitHub，也不随 `fairchem-core` 或 mlipx 安装：

```bash
# 1. 先在网页申请访问权限，并创建可读取 gated 仓库的 Hugging Face token
hf auth login

# 2. 审批通过后下载当前推荐的小模型（保留仓库内 checkpoints/ 子目录）
hf download facebook/UMA checkpoints/uma-s-1p2p1.pt --local-dir models/uma
```

访问申请由 Hugging Face/Meta 人工审批，需要提供完整法定姓名、出生日期、国家和
单位等信息并接受许可。官方模型页目前明确说明 UMA 不在中国、俄罗斯、白俄罗斯及
受全面制裁的司法辖区提供，因此中国大陆地区用户也不在可申请使用范围内。该条件
可能变化，申请前请以模型页的最新说明为准。

> `uma-s-1.pt` 已被官方列为归档检查点并注明存在 extensivity bug；新任务不要再把
> 它作为推荐起点。mlipx 接受任意本地检查点路径，例如
> `--model models/uma/checkpoints/uma-s-1p2p1.pt`。

**其他引擎（可选）：** MACE / DPA / GRACE 的模型由各项目单独发布。MACE 使用
`.model`/`.pt`，DPA 使用冻结或导出的 `.pt`/`.pth`（不是训练 checkpoint），
GRACE 的路径必须是完整 TensorFlow SavedModel 目录。安装环境仍以 §2.1 为唯一
说明。

> **不要执行 `uv pip install mace-torch`。** `uv pip` 默认目标就是项目的
> `.venv`；此时 doctor 虽能看到两个引擎，却会报告
> `UMA/MACE dependencies: incompatible`，MACE 模型也无法加载。

模型路径通过 `--model`（CLI）、TUI 配置界面或 INCAR 文件中的 `MODEL_PATH` 键指定。

### 2.7 验证安装

```bash
uv run mlipx doctor
```

不带参数时，`doctor` 只读取包元数据和硬件清单，不导入任何模型后端，因此不会
顺手初始化 DeepMD、TensorFlow 或 CUDA。要判断一个环境能不能实际启动所选
后端，必须把准备使用的引擎和设备写清楚：

```bash
# auto：nvidia-smi 能发现 GPU 时检查 CUDA，否则检查 CPU
uv run mlipx doctor --engine uma --device auto

# 各隔离环境
.venv-mace/bin/mlipx doctor --engine mace --device cuda:0
.venv-dpa/bin/mlipx doctor --engine dpa --device cuda:0
.venv-grace/bin/mlipx doctor --engine grace --device cpu
```

指定后端后，doctor 会在隔离子进程中导入后端，并在目标 CPU/GPU 上执行一个
真实张量运算；导入错误、CUDA 不可见、算子执行失败和 PyTorch 架构不匹配都会
返回非零退出码。纯 CPU 目标不要求 CUDA。

要验证“这个模型、这个 task/head、这个结构”能否真正计算，使用完整探针。此命令
会加载模型并执行一次能量/力计算；周期模型支持应力时也会检查应力，但不会写入
OUTCAR、轨迹或结果目录：

```bash
.venv-dpa/bin/mlipx doctor --engine dpa --device cuda:0 \
  --model models/dpa/DPA-3.2-5M.pt --task bulk \
  --head Domains_SSE_PBE --structure structure.vasp
```

`--model` 必须同时给出 `--engine` 和 `--task`。多 head 的 MACE/DPA 模型还必须
显式给出 `--head`；doctor 不会猜测势能面。

查看完整命令列表：
```bash
uv run mlipx --help
```

---

## 3. 快速开始

### 3.1 第一次计算（CLI）

```bash
# 晶体结构的单点能计算
uv run mlipx sp structure.cif --model uma-s-1.pt --task omat

# 输出：
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

### 3.2 第一次计算（TUI）

```bash
# 启动交互式终端界面
uv run mlipx tui
```

使用方向键导航，Tab 切换输入字段，Enter 确认选择。

```
┌─ UMA Calculator ───────────────────────────────────────────────────────────────┐
│ Select Calculation Type                                                        │
│                                                                                │
│   Single Point (SP)                                                            │
│     计算能量、力和应力                                                         │
│                                                                                │
│   Geometry Optimization (OPT)                                                  │
│     优化原子位置                                                               │
│                                                                                │
│   Molecular Dynamics (MD)                                                      │
│     运行 NVT/NVE 模拟                                                          │
│                                                                                │
│   Batch Processing                                                             │
│     处理多个结构                                                               │
│                                                                                │
│   Background Jobs                                                              │
│     查看/管理运行中的计算                                                      │
│                                                                                │
│   Generate Template                                                            │
│     创建 INCAR 模板文件                                                        │
│                                                                                │
│   Exit                                                                         │
│     退出程序                                                                   │
└────────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 使用 INCAR 文件（VASP 风格）

```bash
# 生成模板
mlipx template sp -o INCAR.mlipx

# 编辑：
#   CALC_TYPE   = SP
#   MODEL_TYPE  = UMA            # 默认；可选 MACE/DPA/GRACE
#   MODEL_PATH  = uma-s-1.pt
#   TASK        = omat           # UMA 用 omat/omol/...；其他引擎用 bulk/molecule
#   DEVICE      = cpu

# 从 INCAR 运行
mlipx run
```

---

## 4. 架构概览

```
┌─────────────────────────────────────────────────────────────────────┐
│                        用户界面层                                    │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────┐          │
│  │ CLI      │    │ TUI          │    │ Python API       │          │
│  │ (argparse)│   │ (Textual)    │    │ (mlipx.api)     │          │
│  └────┬─────┘    └──────┬───────┘    └────────┬─────────┘          │
│       │                 │                     │                     │
│       └─────────────────┼─────────────────────┘                     │
│                         ▼                                           │
│              ┌─────────────────────┐                                │
│              │   EngineConfig      │  ← 统一配置                    │
│              │   (dataclass)       │                                │
│              └─────────┬───────────┘                                │
│                        ▼                                            │
│              ┌─────────────────────┐                                │
│              │ CalculationEngine   │  ← 唯一执行入口                │
│              │ .run() / .run_async()│                               │
│              └─────────┬───────────┘                                │
└────────────────────────┼────────────────────────────────────────────┘
                         │
┌────────────────────────┼────────────────────────────────────────────┐
│              计算层                                                  │
│              ┌─────────────────────┐                                │
│              │   BaseRunner        │  ← 进度事件、日志              │
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
│              │   UMACalculator     │  ← 封装 FAIRChem ASE 计算器    │
│              └─────────┬───────────┘                                │
└────────────────────────┼────────────────────────────────────────────┘
                         │
┌────────────────────────┼────────────────────────────────────────────┐
│              模型层                                                  │
│              ┌─────────────────────┐                                │
│              │  FAIRChem UMA 模型  │  ← SO(3) 等变神经网络          │
│              │  InferenceSettings  │     tf32, compile, threads     │
│              └─────────────────────┘                                │
└─────────────────────────────────────────────────────────────────────┘
```

`CalculationEngine` 是核心编排器：所有三种界面（CLI、TUI、API）都构建一个 `EngineConfig` 并调用相同的 `CalculationEngine` 方法。这消除了代码重复，确保行为一致。

---

## 多引擎指南

mlipx 的核心设计是**引擎无关**：所有 MLIP 后端都实现统一的 `BaseMLIPCalculator` 抽象接口，上层的单点、优化、MD、批处理、输出逻辑完全共用。切换引擎只需指定 `MODEL_TYPE`，无需修改计算脚本或 INCAR 结构。

### 引擎能力对比

| 引擎 | `MODEL_TYPE` | 后端包 | 支持能量/力 | 支持应力 | task 取值 | 推理模式 |
|------|--------------|--------|-------------|----------|-----------|----------|
| UMA (FAIRChem) | `uma`（默认，别名 `fairchem`） | `fairchem-core` | ✓ | ✓ | `omat`/`omol`/`oc20`/`oc25`/`odac`/`omc` | `default`/`turbo` |
| MACE | `mace` | `mace-torch` | ✓ | ✓ | `bulk`/`molecule` | - |
| DPA (DeepMD-kit) | `dpa` | `deepmd-kit` | ✓ | ✓ | `bulk`/`molecule` | - |
| GRACE | `grace` | `tensorpotential` | ✓ | ✓ | `bulk`/`molecule` | - |

> `INFERENCE_MODE = turbo` 仅 UMA 支持（需 `VALID_INFERENCE_MODES` 属性）。其他引擎会忽略该设置。

### 切换引擎的四种方式

**方式一：CLI 参数 `--model-type`**（适用于 sp/opt/md/batch）

```bash
.venv-mace/bin/mlipx sp structure.cif --model mace.model --model-type mace --task bulk
```

**方式二：INCAR 文件 `MODEL_TYPE` 键**

```ini
CALC_TYPE   = OPT
MODEL_TYPE  = MACE
MODEL_PATH  = mace.model
TASK        = bulk
FMAX        = 0.05
```

**方式三：TUI 下拉选择**

TUI 进程使用启动它的 Python 环境。因此 UMA 必须运行
`uv run mlipx tui`，MACE 必须运行 `.venv-mace/bin/mlipx tui`。界面中的
Model Engine 下拉框不会自动切换虚拟环境；不要在 UMA TUI 中选择 MACE。

**方式四：Python API `model_type` 参数**

```python
from mlipx.api import run_single_point

result = run_single_point(
    structure="structure.cif",
    model_path="mace.model",
    model_type="mace",     # 默认 'uma'
    task="bulk",
    device="cpu",
)
```

### 环境、模型 task/head 与运行前验证

安装命令只在 §2 维护；本节不再复制第二套容易过期的安装教程。运行时必须使用
对应环境的命令前缀。若曾把 MACE 装进 UMA 的 `.venv`，执行
`uv sync --frozen` 可恢复锁文件规定的 UMA 环境，再按 §2.1 重建
`.venv-mace`。

`task` 与模型 `head` 是两件不同的事：非 UMA 的 `bulk|molecule` 明确声明
周期性语义，MACE head / DPA branch 则选择势能面。mlipx 不会再自动改写输入
结构的 PBC；`bulk` 必须是完整三维 PBC，`molecule` 必须完全非周期。多 head
的 MACE/DPA 模型若没有显式 `--head` 会直接报错。DPA 可用
`dp show MODEL model-branch` 查看规范分支名和 alias。

正式 MD 前先用 doctor 做不落盘的单结构探针，例如：

```bash
.venv-mace/bin/mlipx doctor --engine mace --device cuda:0 \
  --model mace.model --task bulk --head default --structure test.vasp

.venv-dpa/bin/mlipx doctor --engine dpa --device cuda:0 \
  --model models/dpa/DPA-3.2-5M.pt --task bulk \
  --head Domains_SSE_PBE --structure test.vasp
```

**GRACE 邻居缓存（默认开启）**使用 `cutoff + NEIGHBOR_SKIN` 的 Verlet 候选表，
每一步仍按模型精确 cutoff 过滤。原子数、元素、晶胞或 PBC 改变，或者任一原子的
一般最小像位移超过 `skin/2` 时立即重建；倾斜晶胞和重复周期像均保留。缓存与
无缓存路径使用同一规范化邻居顺序，已用单元测试和 200 帧短 GPU 随机轨迹交叉
检查。若当前 TensorPotential 导出格式不支持安全安装缓存，mlipx 会报错而不是
静默退回。可用 INCAR `NEIGHBOR_CACHE=False` 或 CLI
`--no-neighbor-cache` 显式关闭。历史 V100/400 原子 LGPS 测试约为 51 ms/步，
无缓存约 66 ms/步；这是特定硬件/模型的参考值，不是通用性能保证。

### 任务映射与周期性边界 (PBC)

UMA 有明确的任务体系（对应不同训练数据集）；其他引擎不感知 task，`task` 仅用于控制 PBC 策略：

| `task` | PBC | 等价 UMA task | 适用体系 |
|--------|------|---------------|----------|
| `bulk` | True | `omat` | 周期性晶体、表面 slab、MOF |
| `molecule` | False | `omol` | 孤立分子（自动设 charge=0） |

> **电荷/自旋默认值（任务相关）：**
> - UMA `omol`：`atoms.info` 未设置时自动填 `charge=0`、`spin=1`（自旋**多重度**，1 = 单重态）。
> - 非 UMA `molecule`（MACE/DPA/GRACE）：仅自动填 `charge=0`；**不**注入 `spin`。MACE 将 `atoms.info["spin"]` 读作**总自旋 S**（0 = 单重态），与 UMA 的多重度语义不同，贸然注入 `spin=1` 会把自旋敏感的 MACE 模型强制为双自由基。如需自旋，请显式设置。
> - 分子任务可在 TUI 直接填写，也可用 CLI 的 `--charge` / `--spin` 或 INCAR 的 `CHARGE` / `SPIN` 设置；它们会覆盖输入结构已有的 `atoms.info`。仍可用 Python 直接编辑结构元数据。
> - 分子体系（`pbc=False`）不计算应力张量。

### 各引擎完整示例

```bash
# ── UMA（默认）── 块体材料优化
mlipx opt Li2O.cif --model uma-s-1.pt --model-type uma --task omat --cell-opt --fmax 0.02

# ── MACE ── 周期性体系单点能
.venv-mace/bin/mlipx sp bulk.cif --model mace.model --model-type mace --task bulk --device cuda

# ── MACE head：只能选择模型实际列出的 head
.venv-mace/bin/mlipx sp bulk.cif --model mace-mpa-0-medium.model --model-type mace --task bulk --head default

# ── MACE dtype（所有计算类型均默认 float64；追求速度时显式选择 float32）
.venv-mace/bin/mlipx opt bulk.cif --model mace.model --model-type mace --task bulk --dtype float32

# ── DPA (DeepMD) ── 在隔离环境中进行周期结构优化
.venv-dpa/bin/mlipx opt bulk.vasp --model dpa.pt --model-type dpa --task bulk --fmax 0.01 --optimizer LBFGS

# ── GRACE ── 高通量筛选（批量）
mlipx batch structures/ --model grace_model/ --model-type grace --task bulk --calc-type sp --output results/
```

### MACE 专属选项

| 选项 | 说明 |
|------|------|
| `--head HEAD` | 选择模型中实际存在的 head；单 head 模型应省略。head 名称取决于具体模型及版本。 |
| `--dtype float32\|float64` | 计算精度。**SP、优化和 MD 均默认 `float64`**；追求速度时可显式选择 `float32`。 |

> ⚠️ `--head` 接受单个字符串。虽然 mace-torch 对未知名称可能只告警并
> 回退，但 mlipx 会直接拒绝，因为静默换 head 等于更换势能面。当前随仓库
> 提供的 `models/mace/mace-mpa-0-medium.model` 只有 `default`。
> 默认 `float64` 保持准确性优先。只有明确接受精度/性能权衡时才传入
> `--dtype float32`。

### 引擎故障排除

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| `Backend mace_torch not installed` | 当前解释器不是 MACE 环境 | 按隔离步骤安装，并用 `.venv-mace/bin/mlipx tui` 启动 |
| `ImportError: e3nn` 版本冲突 | mace 与 fairchem 同环境 | 为 MACE 创建独立 venv |
| `too many values to unpack (expected 2)` | MACE 模型由 e3nn 0.4.4 保存，却被 e3nn 0.5/0.6 加载 | 使用 `.venv-mace`，确认 `e3nn.__version__ == 0.4.4` |
| `Object of type Tensor is not JSON serializable` | 旧版 mlipx 将 MACE Tensor 元数据直接写入 JSON | 更新 mlipx；当前版本已将 Tensor 转成标量/列表 |
| `Model file not found` | 路径错误 | 用绝对路径指定 `--model` |
| UMA 能用但 MACE 报错 | 同环境 e3nn 冲突 | 见上文环境隔离 |
| task `bulk` 在 UMA 下无效 | UMA 不识别 bulk | UMA 用 omat；bulk 仅用于非 UMA 引擎 |

---

## 5. 计算类型

### 5.1 单点能计算 (SP)

单点计算针对固定的原子构型计算势能、原子力和（如支持）应力张量。这是最简单、最快的计算类型。

**产生的数据：**
- 总能量 (eV)
- 每个原子的能量 (eV/atom)
- 每个原子上的力 (eV/Å)，包括最大力和均方根力
- 应力张量（Voigt 表示，eV/Å³）—— 如果模型/任务支持
- 压力 (GPa) —— 从应力迹导出

**CLI 用法：**

```bash
mlipx sp <structure> --model <model.pt> [选项]

# 基本用法
mlipx sp POSCAR --model uma-s-1.pt --task omat

# 指定输出目录和任务名称
mlipx sp structure.cif \
    --model uma-s-1.pt \
    --task omat \
    --device cuda \
    --output ./results \
    --name my_calculation
```

**输出文件：** `OUTCAR`、`CONTCAR`、`mlipx_results.json`

### 5.2 几何优化 (OPT)

优化原子位置（以及可选的晶胞参数）以找到局部能量极小值。当任意原子上的最大力低于收敛阈值（`fmax`）或达到最大步数时计算停止。

**算法：**

| 优化器 | 描述 | 最适用于 |
|--------|------|----------|
| `FIRE` | 快速惯性弛豫引擎（默认） | 大多数体系，稳健 |
| `BFGS` | Broyden-Fletcher-Goldfarb-Shanno | 小体系，快速收敛 |
| `LBFGS` | 有限内存 BFGS | 大体系 |

**CLI 用法：**

```bash
mlipx opt <structure> --model <model.pt> [选项]

# 基本优化
mlipx opt POSCAR --model uma-s-1.pt

# 严格收敛 + 晶胞弛豫
mlipx opt POSCAR \
    --model uma-s-1.pt \
    --fmax 0.02 \
    --max-steps 1000 \
    --cell-opt \
    --optimizer BFGS

# 保持晶体对称性
mlipx opt structure.cif \
    --model uma-s-1.pt \
    --fix-symmetry
```

**参数：**

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--fmax` | 0.05 | 力收敛阈值 (eV/Å) |
| `--max-steps` | 500 | 最大优化步数 |
| `--optimizer` | FIRE | FIRE / BFGS / LBFGS |
| `--cell-opt` | 关 | 启用晶胞参数优化 |
| `--fix-symmetry` | 关 | 保持晶体对称性 |

**输出文件：** `OUTCAR`、`CONTCAR`（优化后的结构）、`OSZICAR`（逐步进度）、`mlipx_results.json`

### 5.3 分子动力学 (MD)

模拟原子在给定温度下的时间演化。支持以下方法：

| 系综 | 积分器 | 描述 |
|------|--------|------|
| NVT | Langevin | 随机 Langevin 恒温器 |
| NVT | Bussi / CSVR | 随机速度重标度 |
| NVT | Nosé-Hoover Chain | 确定性链式恒温器 |
| NVE | Velocity Verlet | 恒粒子数、体积、能量（微正则系综） |

**预弛豫：** 在开始 MD 前，mlipx 可执行短程、仅优化原子位置的 FIRE
优化（默认 50 步，fmax=0.1 eV/Å），用于降低过大的初始原子力。它不优化
晶胞，也不保证达到局部极小值，因此通常不能称为“消除晶胞应力”。默认开关
与系综相关：
- **NVT**：默认**启用**（避免初始高力导致爆炸）。
- **NVE**：默认**关闭**——预先改变原子位置会改变守恒能量基准，使轨迹
  不再对应输入结构；只有明确接受这种初态变化时才显式开启。

可用 `--pre-relax` / `--no-pre-relax`（CLI）或 `PRE_RELAX = .TRUE./.FALSE.`（INCAR）显式覆盖默认值。
若重启结构已经保存动量，请使用 `--no-pre-relax` 并选择
`--velocity-policy auto` 或 `preserve`；一边改变位置、一边保留旧速度并不是
严格的相空间重启，程序会明确拒绝这种组合。

**CLI 用法：**

```bash
mlipx md <structure> --model <model.pt> [选项]

# 300K 下运行 10 ps NVT
mlipx md POSCAR \
    --model uma-s-1.pt \
    --ensemble NVT \
    --temp 300 \
    --timestep 1.0 \
    --steps 10000 \
    --save-interval 10

# NVE 系综
mlipx md CONTCAR \
    --model uma-s-1.pt \
    --ensemble NVE \
    --temp 300 \
    --steps 5000
```

**参数：**

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--ensemble` | NVT | NVT 或 NVE |
| `--temp` | 300 | 温度 (K) |
| `--timestep` | 1.0 | 时间步长 (fs) |
| `--steps` | 1000 | 生产段 MD 步数 |
| `--equilibration-steps` | 0 | 同一系综、恒温器和时间步下的平衡段步数 |
| `--thermostat` | LANGEVIN | NVT 使用 `LANGEVIN`、`BUSSI` 或 `NHC` |
| `--friction` | 0.001 | Langevin 摩擦系数（fs⁻¹） |
| `--bussi-tau` | 1000.0 | Bussi/CSVR 耦合时间（fs） |
| `--nhc-tdamp` | 100.0 | NHC 阻尼时间（fs） |
| `--nhc-tchain` | 3 | NHC 链长度 |
| `--nhc-tloop` | 1 | NHC 恒温器子步数 |
| `--save-interval` | 10 | 每隔 N 步保存轨迹 |
| `--seed` | 自动生成并记录 | 复现初始速度与 NVT 随机力的随机种子 |
| `--velocity-policy` | auto | `auto`、`initialize` 或 `preserve` 已有动量 |
| `--pre-relax` | NVT 开 / NVE 关 | MD 前预弛豫（`--no-pre-relax` 关闭） |
| `--pre-relax-steps` | 50 | 预弛豫最大步数 |
| `--pre-relax-fmax` | 0.1 | 预弛豫力收敛阈值 (eV/Å) |

恒温器选择和耦合强度可能影响动力学与输运性质。面向输运的计算应检查恒温器
敏感性。`--equilibration-steps` 可在同一任务中先运行同系综平衡段；如果需要
NVT 平衡后切换到 NVE 生产段，仍应拆成两个任务并明确处理重启速度。

真实后端接口 smoke test 默认使用 CPU、4 个原子、每种方法 5 步。请分别用
对应隔离环境运行 `mlipx/examples/smoke_md_backends.py`。

**MD 输出目录：**

```text
<任务目录>/
├── raw/
│   ├── trajectory.traj       # 不丢信息的 ASE 轨迹，内部事实源
│   ├── md.csv                # 时间、能量、温度、体积、应力和压力
│   └── mlipx_results.json    # 机器可读结果摘要
├── vasp/
│   ├── XDATCAR               # VASP 语法兼容、未折回分数坐标
│   ├── CONTCAR               # VASP POSCAR/CONTCAR 语法
│   └── OUTCAR                # 明确标注的 VASP-like MD 子集
├── artifacts.json            # 输出契约、单位、帧信息和文件索引
├── resolved_config.json
└── run.log
```

`raw/trajectory.traj` 是轨迹事实源；`vasp/XDATCAR` 主要用于 OVITO、ASE
和其他 VASP 生态工具。做后处理时，最好传入整个任务目录，而不是只传轨迹
文件。这样程序还能读取 `raw/md.csv`、`artifacts.json` 和
`resolved_config.json` 中的时间、阶段、单位与坐标约定。

### 5.4 内置轨迹后处理（Analysis v2）

当前版本可以直接计算热力学统计、RDF、RMSD/RMSF、方向 MSD、扩散系数、
Nernst–Einstein 电导率、三维占据密度、Arrhenius 拟合、离子位点与跳跃、
VACF 和速度谱。分析层不加载模型，所以 DPA、MACE、GRACE 或 UMA 生成的
轨迹都可以放到同一个分析环境中处理。

从仓库根目录安装所需依赖：

```bash
# MSD、RDF、密度和作图
python -m pip install -e './mlipx[analysis]'

# 一次装齐 kinisi 和 GEMDAT
python -m pip install -e './mlipx[analysis-all]'
```

`RUN` 可以是 mlipx 的 MD 任务目录、ASE `.traj` 或 XDATCAR。外部轨迹如果
缺少时间或坐标约定，必须在确实知道这些信息时显式补上，例如：

```bash
mlipx analyze trajectory.traj validate \
    --positions-convention wrapped \
    --frame-interval-fs 10
```

不要凭文件外观猜 wrapped/unwrapped，也不要把 MD 时间步误当成保存帧间隔。

#### 先验证，再计算

```bash
mlipx analyze RUN validate
```

`validate` 会报告时间轴、MD 时间步与保存间隔、PBC、晶胞是否变化、坐标约定、
速度、equilibration/production 阶段、Nyquist 频率、任务状态，以及每种分析
是否满足条件。标记为 failed、aborted 或 cancelled 的 mlipx 任务会被拒绝；
外部轨迹没有 mlipx 状态，是否完整仍需用户自己确认。

MD 可以用 `--equilibration-steps N` 和 `--steps M` 在同一系综、积分器、恒温器、
时间步和温度下分别记录平衡段与生产段。分析默认只使用 production 帧；
`thermo`、RDF、RMSD、MSD、密度、VACF 和速度谱可用
`--include-equilibration` 把平衡段纳入诊断，`transport` 和 `electrolyte` 则固定
使用 production 帧。程序不会自动决定热化何时结束，也不会自动替你挑扩散
线性区间。没有 phase 元数据的旧轨迹会把全部帧视为 production，这只是兼容
规则，不表示整段已经平衡。应先看 `thermo`，再用支持帧范围的任务所提供的
`--start-frame`/`--stop-frame` 选取范围；这里的 frame 是保存下来的帧，不是
MD 步数。

#### 常用分析

```bash
# 温度、能量、压力和 NVE 能量漂移诊断
mlipx analyze RUN thermo

# Li–S 部分 RDF；配位数截断必须显式给出
mlipx analyze RUN rdf --center Li --neighbor S --rmax 6 --cn-cutoff 3

# 晶体周期位移的 RMSD/RMSF
mlipx analyze RUN rmsd --species Li

# 多时间原点平均的方向 MSD
mlipx analyze RUN msd --mobile Li --axes x,y,z,xy,xyz \
    --drift-reference nonmobile

# 三维占据密度、VACF 和 VACF 导出的速度谱
mlipx analyze RUN density --mobile Li --spacing 0.25
mlipx analyze RUN vacf --species Li
mlipx analyze RUN spectrum --species Li --taper one-sided-cosine
```

RDF 的 `center` 和 `neighbor` 有顺序；两者相同时会排除自身。`rmax` 不能超过
三斜晶胞最小面高的一半，配位数只有在显式设置 `--cn-cutoff` 时才计算。密度
结果同时给出总和为 1 的占据概率和单位为 Å⁻³ 的数密度，后者对晶胞积分应
等于所选离子数。

MSD 使用多时间原点平均，默认 FFT 实现，也可用 `--method direct` 交叉检查。
计算使用连续坐标：unwrapped 轨迹直接使用；固定晶胞的 wrapped 轨迹按相邻帧
最小镜像重建，并报告可能跨越半个晶胞的歧义。变胞轨迹会被拒绝，因为当前
实现不能可靠地区分晶胞仿射形变和真实迁移。漂移修正从不暗中启用，可选
`none`、`nonmobile` 或明确的 `indices`，选择会写入 provenance。只有同时给出
`--fit-start-ps` 和 `--fit-stop-ps` 时才计算普通最小二乘诊断斜率；
没有显式窗口时只输出 MSD 和局部 log–log 指数，不猜测扩散系数。OLS 仍只是
诊断值，不代替带协方差和不确定度的输运估计。

#### 扩散系数和电导率

协方差感知的输运拟合使用 kinisi v2：

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

上面的 20 ps 只是命令示例，不能照搬到所有体系。应结合 `thermo`、MSD 曲线、
采样长度和体系物理选择已经平衡且进入扩散区间的起点。输运分析要求固定晶胞、
三维周期、至少 4 个 production 帧、均匀采样和明确的坐标约定。对 wrapped
轨迹，如果相邻帧位移已接近最小镜像失效范围，程序会直接拒绝，不会给出一个
看似正常但无法验证的扩散系数。

对于长时间且高频保存的轨迹，可用 `--lag-step-ps` 和
`--lag-stop-ps` 显式指定 kinisi 的 lag-time 网格；两个参数必须同时给出。
它们只减少 kinisi 计算的时间间隔数量，不会对轨迹降采样，完整保存帧和时间
原点仍会保留。`--fit-start-ps` 是另一个概念：它决定从哪个 lag 开始进行
扩散回归，而 lag 网格决定 kinisi 可使用哪些 lag。应先根据 MSD 找到扩散区间，
再比较例如 0.5、1、2 ps 的间距做 convergence check。kinisi 文档指出，在完整
协方差分析中计算每一个可能的时间间隔通常过度；未指定网格时，小轨迹仍保留
kinisi 原生行为，但过密的默认网格会被 mlipx 拒绝并要求显式指定参数。

kinisi 2.x 的三斜晶胞解析器会创建多组 `帧数 × 原子数 × 8` 数组。mlipx 先按
kinisi 的非加权框架平均位移定义做一次漂移修正，再只把移动原子交给解析器，
并在分配内存前做保守估算。默认上限为 4 GiB，可用
`--parser-memory-limit-gib` 调整；仅在确认可用 RAM 后提高。结果会记录原始/
解析器原子数、三斜路径、估算字节数和上限。这是峰值分配保护，不是进程总 RSS
的精确预测。

kinisi 会输出后验均值、标准差和 95% 区间，并同时记录 m²/s 与 cm²/s。方向
扩散遵循 `D = slope / (2d)`，其中 `d` 是所选方向数。`--charge` 必须显式给出，
软件不会根据元素猜价态；温度来自 production 帧平均值，缺失时必须用
`--temperature-K` 指定，不会回退到 300 K。Nernst–Einstein 示踪电导率采用
`σ = n(ze)²D/(kBT)`，它不等同于考虑相关运动后的实验电导率。只有明确加上
`--collective-conductivity` 才会额外计算集体量，软件也不会自动给出 Haven 比。

Nernst–Einstein 输出现在还包含 S/m、S/cm 和 mS/cm 三种单位的后验摘要。这只是
在固定数密度、电荷、体积和温度条件下，对 kinisi 示踪扩散后验的线性传播，
不是总物理不确定度；不包含模型、有限尺寸、重复轨迹、温度、体积、
Nernst–Einstein 近似和离子相关运动的不确定度。旧的 `sigma_NE_tracer_*`
标量字段仍保留，并等于相应后验均值。

kinisi 的 ASE 后端会从周期/缩放坐标重新构造位移。对于 exact unwrapped 源，
mlipx 在调用 kinisi 前逐个比较保存帧位移和 ASE 最小镜像重建结果；如果会丢失
image history，就直接拒绝分析。通过检查时结果也会明确记录：重建等价，但
kinisi 并没有直接消费 exact image counters。

成功的 transport 还会在 `kinisi_arrays.npz` 旁写出
`transport_summary.csv`、`transport_msd.png` 和 `transport_msd.svg`；CLI 会直接
打印扩散后验、95% credible interval、拟合区间、lag 网格和
Nernst–Einstein 电导率。优先使用 `mlipx analyze RUN transport ...`，这样程序可
从任务目录读取时间、保存步长、坐标约定、production 阶段和温度。外部轨迹只有
在确实知道时才填写 `--positions-convention`、`--frame-interval-fs` 和
`--temperature-K`。当前 LGPS 使用的 `2 ps` 是经过灵敏度比较后的工作参数，
不是所有材料的通用默认值。

多个独立温度的扩散结果可用 `arrhenius` 拟合
`ln D = ln D₀ - Eₐ/(kBT)`。建议至少使用 3 个温度点；2 点只会给出警告，无法
检验线性。若提供 `--diffusivity-std`，拟合会用近似
`σ(ln D) = σ(D)/D` 加权；外推值会被明确标记。完整参数可用：

```bash
mlipx analyze RUN arrhenius --help
```

#### 电解质位点、跳跃和渗流

这部分由 GEMDAT 提供。必须明确给出参考位点，或明确要求从密度中发现位点：

```bash
mlipx analyze RUN electrolyte \
    --mobile Li \
    --sites Li_sites.cif \
    --jump-dimensions 3 \
    --percolation-axes xyz

# 没有参考位点时才使用，并检查发现结果
mlipx analyze RUN electrolyte --mobile Li --discover-sites-from-density
```

两种位点来源互斥，软件不会静默选择。`--jump-dimensions` 决定跳跃扩散公式中的
维数；`--percolation-axes` 只限制路径连通方向，不会改变跳跃维数、kinisi
示踪扩散或电导率定义。输出包括参考/发现位点、跃迁和跳跃表及矩阵、密度与
自由能数组、路径坐标和势垒。GEMDAT 的端点示踪估计不会覆盖 kinisi 的输运
结果，两者的方法和用途不同。

VACF 和 `spectrum` 只使用轨迹中实际保存且均匀采样的速度，不会对位置做数值
微分。默认单边余弦 taper 在零时刻权重为 1，负谱值会保留并报告；输出应称为
“VACF 导出的速度谱”，不能直接当作谐振子声子 DOS。

#### 输出与复现

结果写入 `RUN/analysis/<任务>/<请求哈希>/`，其中包含 `request.json`、
`provenance.json`、`results.json`、任务对应的 CSV/NPZ、PNG/SVG 和诊断信息。
请求哈希包含输入指纹、选区、帧范围、方向、漂移规则、科学参数和后端版本；
相同请求会复用已有结果，`--force` 才会重算。失败信息写入 `error.json`，且不
生成 `results.json`，不会把未完成任务标成成功。

只看本 README 已经可以完成上述流程。更细的算法定义、单位表和参考文献保留在
[Analysis v2 专题说明](ANALYSIS.md)、[输运定义](TRANSPORT.md) 和
[电解质机制分析](ELECTROLYTE.md) 中。

### 5.5 批量处理

对目录中的多个结构运行相同类型的计算，支持 `sp` 和 `opt`。当前批量
入口仅在 CLI 中提供，并按顺序处理文件；模型在整批任务中只加载一次。

**CLI 用法：**

```bash
uv run mlipx batch <input_dir> --model <model.pt> [选项]

# 对所有 CIF 文件进行 SP 计算
uv run mlipx batch structures/ \
    --model uma-s-1.pt \
    --calc-type sp \
    --pattern "*.cif" \
    --output batch_results

# 使用独立 MACE 环境批量计算
.venv-mace/bin/mlipx batch structures/ \
    --model mace.model \
    --model-type mace \
    --task bulk \
    --device cuda \
    --calc-type sp \
    --pattern "*.cif" \
    --output mace_batch_results
```

**输出：** 每个结构在输出目录下获得自己的子目录，`batch_summary.json`
列出所有结果。当前 CLI 不接受 `--parallel` 或 `--workers`，TUI 也没有
Batch 菜单。

---

## 6. 用户界面

### 6.1 CLI — 命令行界面

CLI 通过 `mlipx <command> [选项]` 调用。不带参数运行 `mlipx` 默认启动 TUI。

#### 完整命令参考

##### `mlipx sp` — 单点能

```
mlipx sp STRUCTURE --model MODEL [--model-type TYPE] [--task TASK] [--device DEVICE]
                       [--charge INTEGER] [--spin INTEGER]
                       [--cpu-threads N] [--inference-mode MODE]
                       [--activation-checkpointing | --no-activation-checkpointing]
                       [--dtype DTYPE] [--head HEAD] [--output DIR] [--name NAME]

  STRUCTURE             输入结构文件（CIF, XYZ, POSCAR, VASP 等）
  --model MODEL         模型检查点路径 [必需]
  --model-type TYPE     MLIP 引擎：uma|mace|dpa|grace [默认: uma]
  --task TASK           UMA: omat|omol|oc20|oc25|odac|omc；其他: bulk|molecule [默认: omat]
  --device DEVICE       cpu|gpu|cuda|cuda:N [默认: cpu]
  --charge INTEGER      分子总电荷（UMA omol 默认: 0）
  --spin INTEGER        分子自旋元数据；UMA omol 中为自旋多重度（默认: 1）
  --cpu-threads N       CPU intra-op 线程数（全部后端；省略则由后端决定）
  --inference-mode MODE UMA 推理预设：default|turbo
  --[no-]activation-checkpointing
                        UMA 显存/速度覆盖；省略则跟随推理预设
  --dtype DTYPE         MACE 精度 float32|float64 [默认: float64]
  --head HEAD           MACE head 或 DeepMD/DPA 多任务分支
  --output DIR, -o DIR  输出目录 [默认: .]
  --name NAME, -n NAME  任务名称（输出至 DIR/NAME）
```

> `--model-type` 适用于所有计算命令（sp/opt/md/batch），默认仍为 `uma`。

##### `mlipx opt` — 几何优化

```
mlipx opt STRUCTURE --model MODEL [选项]

  --fmax FMAX           力收敛阈值 eV/Å [默认: 0.05]
  --max-steps N         最大优化步数 [默认: 500]
  --optimizer ALGO      优化算法：FIRE|BFGS|LBFGS [默认: FIRE]
  --cell-opt            启用晶胞参数优化
  --fix-symmetry        保持晶体对称性
```

##### `mlipx md` — 分子动力学

```
mlipx md STRUCTURE --model MODEL [选项]

  --ensemble ENSEMBLE   系综：NVT|NVE [默认: NVT]
  --temp TEMP           温度 (K) [默认: 300]
  --timestep DT         时间步长 (fs) [默认: 1.0]
  --steps N             生产段 MD 步数 [默认: 1000]
  --equilibration-steps N
                        同系综平衡段步数 [默认: 0]
  --thermostat TYPE     LANGEVIN|BUSSI|NHC [默认: LANGEVIN]
  --friction VALUE      Langevin 摩擦系数 fs^-1 [默认: 0.001]
  --bussi-tau FS        Bussi/CSVR 耦合时间 [默认: 1000.0]
  --nhc-tdamp FS        NHC 阻尼时间 [默认: 100.0]
  --nhc-tchain N        NHC 链长度 [默认: 3]
  --nhc-tloop N         NHC 恒温器子步数 [默认: 1]
  --save-interval N     轨迹保存间隔 [默认: 10]
  --seed N              可复现 MD 随机种子 [默认: 自动生成并记录]
  --velocity-policy P   auto|initialize|preserve [默认: auto]
  --pre-relax           MD 前预弛豫 [默认: NVT 开 / NVE 关]
  --pre-relax-steps N   预弛豫最大步数 [默认: 50]
  --pre-relax-fmax F    预弛豫力收敛阈值 eV/Å [默认: 0.1]
  --fmax-abort F        力安全中止阈值 eV/Å [默认: 20.0]
```

##### `mlipx analyze` — 轨迹后处理

```text
mlipx analyze RUN validate
mlipx analyze RUN thermo
mlipx analyze RUN msd --mobile Li --axes x,y,z,xyz \
    --drift-reference nonmobile
```

`RUN` 可以是 mlipx 任务目录、ASE `.traj` 或 XDATCAR；传入任务目录
能保留最完整的时间、PBC、阶段和单位信息。可用任务包括
`validate`、`thermo`、`rdf`、`rmsd`、`msd`、`transport`、
`density`、`arrhenius`、`electrolyte`、`vacf` 和 `spectrum`。
先运行 `validate`，再根据报告决定后续分析。

依赖安装、完整示例和科学限制都在第 5.4 节；各子命令的当前参数以
`mlipx analyze RUN <任务> --help` 为准。

##### `mlipx batch` — 批量处理

```
mlipx batch INPUT_DIR --model MODEL [选项]

  --calc-type TYPE      计算类型：sp|opt [默认: sp]
  --pattern PATTERN     显式文件通配符；省略时发现 CIF/XYZ/VASP/POSCAR*
  --output DIR          输出目录 [默认: batch_results]
```

运行时仍须选择正确解释器：UMA 使用 `uv run mlipx batch ...`，MACE 使用
`.venv-mace/bin/mlipx batch ...`。

##### `mlipx run` — 从 INCAR 文件运行

```
mlipx run [-i INCAR] [-s STRUCTURE] [-o OUTPUT]

  -i, --incar INCAR     INCAR 文件路径 [默认: INCAR.mlipx]
  -s, --structure FILE  结构文件（自动检测：POSCAR, CONTCAR, *.cif, *.xyz）
  -o, --output DIR      输出目录 [默认: .]
```

##### `mlipx template` — 生成 INCAR 模板

```
mlipx template TYPE [-o OUTPUT]

  TYPE                  sp|opt|md
  -o, --output FILE     输出文件名 [默认: INCAR.<type>]
```

##### `mlipx jobs` — 列出后台任务

```
mlipx jobs
```

显示所有后台任务及其 ID、状态、类型、化学式和设备。

##### `mlipx kill` — 终止后台任务

```
mlipx kill JOB_ID
```

终止指定任务（跨平台：Windows 使用 `taskkill`，Unix 使用 `SIGTERM`）。

##### `mlipx clean` — 清理已完成/失败的任务

```
mlipx clean
```

删除已完成、失败或取消的任务的状态文件。保留正在运行的任务。

##### `mlipx queue pause/resume` — 暂停/恢复单个待执行任务

```bash
mlipx queue pause <job-id>     # 只暂停指定的 PENDING 任务
mlipx queue status
mlipx queue resume <job-id>    # 恢复指定任务
```

单个任务暂停后会变为 `paused`，调度器会跳过它并继续寻找其他 `pending` 任务；
当前 RUNNING 任务和其他 pending 任务不受影响。恢复后该任务重新进入队列，
按原提交时间参与 FIFO 调度。省略 `<job-id>` 仍可暂停/恢复整个 pending 队列；
这不同于 `mlipx queue stop`，后者会停止调度器进程。

##### `mlipx tui` — 启动 TUI

```
mlipx tui
```

启动交互式终端用户界面。

### 6.2 TUI — 终端交互界面

TUI 基于 [Textual](https://textual.textualize.io/) 构建，提供交互式、键盘驱动的体验。

#### 导航

| 按键 | 操作 |
|------|------|
| `↑` / `↓` | 导航菜单项 / 滚动 |
| `Tab` | 移至下一个输入字段 |
| `Shift+Tab` | 移至上一个输入字段 |
| `Enter` | 选择 / 确认 |
| `Esc` | 返回上一屏幕 |
| `Q` | 退出程序 |
| `PgUp` / `PgDn` | 可滚动区域的上下翻页 |
| `C` | 取消选中的任务（Jobs 屏幕） |
| `D` | 删除任务记录（Jobs 屏幕） |
| `P` | 暂停选中的 pending 任务（Jobs 屏幕） |
| `R` | 刷新任务列表（Jobs 屏幕） |
| `U` | 恢复选中的 paused 任务（Jobs 屏幕） |

#### 屏幕说明

**主菜单**（Main Menu）— 选择 SP、OPT、MD、Jobs、Template 或 Exit。
Batch 当前仅能通过 CLI 运行。

**配置界面**（Configuration）— 填写路径、引擎/任务，以及设备字符串
（`cpu`、`cuda` 或 `cuda:N`）。其中 **Backend & Resource Options**
（后端与资源选项）直接提供：

- 全部引擎的 CPU 线程数（UMA/MACE 与 DPA `.pt` 模型为 PyTorch，GRACE 为
  TensorFlow；DPA `.pb` 使用 DeepMD TensorFlow 后端设置）
- UMA 的推理模式和激活检查点
- MACE 精度，以及 MACE/DPA 的模型 head 或分支
- 在 `omol` / `molecule` 任务下设置总电荷和自旋；UMA 会明确显示自旋多重度语义

选择某个引擎后，只显示该引擎实际使用的后端控件；例如选择 UMA 时不会再显示
DPA branch。电荷/自旋仅在分子任务下出现，周期任务不会携带上一次的分子设置。
OPT 还可设置晶胞优化和保持晶体对称性；MD 可设置系综、温度、时间步、
平衡/生产步数、保存间隔、NVT 恒温器及其对应的耦合参数、预弛豫、
随机种子、速度策略和力安全中止阈值。
路径支持实时验证和视觉反馈：

```
📁 Structure File: [structure.cif                ]
   ✅ Found: /home/user/mlipx/structure.cif
   💡 提示：支持相对路径（例如：./data/structure.cif）
```

**运行界面**（Run）— 计算会自动作为独立后台进程启动：
- 可立即按 Back 返回，计算不会被取消
- 退出 TUI 或 SSH 断开后，计算仍会继续
- Cancel 会终止后台计算，而 Back 只关闭当前监视页面
- 启动时显示实时日志的绝对路径和可直接复制的 `tail -f` 命令

**任务管理**（Jobs）— 使用 DataTable 列出所有后台任务及状态图标：
- ● 运行中 | ✓ 已完成 | ✗ 失败 | ⊘ 已取消
- 按 Enter 查看任务日志输出
- 每 2 秒自动刷新
- 在 Jobs 表中选中任务后使用 **Pause Job** / **Resume Job**，或按 `P` / `U`；只影响单个 pending/paused 任务
- **Pause Queue** / **Resume Queue** 仍可用于整体暂停/恢复 pending 队列；正在运行的任务不受影响

### 6.3 Python API

用于脚本编写和工作流集成，导入 `mlipx.api`：

```python
from mlipx.api import run_single_point, run_optimization, run_md
from mlipx.api import calculate_energy

# 单点能
results = run_single_point(
    structure="structure.cif",
    model_path="uma-s-1.pt",
    task="omat",
    device="cuda",
    job_name="my_calc",
)
print(f"能量: {results['energy']:.4f} eV")
print(f"力: {results['forces']}")

# 几何优化
results = run_optimization(
    structure="POSCAR",
    model_path="uma-s-1.pt",
    fmax=0.02,
    cell_opt=True,
)
print(f"收敛: {results['converged']}，步数: {results['nsteps']}")

# 分子动力学
results = run_md(
    structure="CONTCAR",
    model_path="uma-s-1.pt",
    ensemble="NVT",
    temperature=300,
    steps=10000,
    save_interval=10,
)
print(f"最终温度: {results['temperature']:.1f} K")

# 快速能量计算
energy = calculate_energy("structure.cif", "uma-s-1.pt")
print(f"能量: {energy:.4f} eV")

```

完整 API 参考：

| 函数 | 返回值 | 描述 |
|------|--------|------|
| `run_single_point(structure, model_path, ...)` | `dict` | SP 能量、力、应力 |
| `run_optimization(structure, model_path, ...)` | `dict` | OPT 含收敛信息 |
| `run_md(structure, model_path, ...)` | `dict` | MD 含轨迹和温度 |
| `calculate_energy(structure, model_path, ...)` | `float` | 快速获取能量值 |

---

## 7. INCAR 配置文件参考

INCAR 文件采用 VASP 风格的 `KEY = VALUE` 格式。以 `#` 或 `!` 开头的行是注释。

### 7.1 所有 INCAR 关键字

#### 计算控制

| 键 | 类型 | 默认值 | 描述 |
|-----|------|--------|------|
| `CALC_TYPE` | 字符串 | `SP` | 计算类型：`SP`、`OPT`、`MD`、`BATCH` |

#### 模型设置

| 键 | 类型 | 默认值 | 描述 |
|-----|------|--------|------|
| `MODEL_PATH` | 字符串 | `uma-s-1p2p1.pt`（模板占位） | 模型检查点路径（.pt 文件） |
| `MODEL_TYPE` | 字符串 | `uma` | MLIP 引擎：`uma`、`mace`、`dpa`、`grace`（`uma` 也可写 `fairchem`） |
| `TASK` | 字符串 | `omat` | 任务类型。UMA：`omat`/`omol`/`oc20`/`oc25`/`odac`/`omc`；其他引擎：`bulk`/`molecule` |
| `DEVICE` | 字符串 | `cpu` | 计算设备：`cpu`、`cuda`、`gpu` 或 `cuda:N` |
| `CHARGE` | 整数 | 未设置（UMA omol 使用 `0`） | 分子总电荷；覆盖结构的 `atoms.info["charge"]` |
| `SPIN` | 整数 | 未设置（UMA omol 使用 `1`） | 分子自旋元数据；UMA omol 中为自旋多重度 |
| `INFERENCE_MODE` | 字符串 | `default` | 推理模式：`default`、`turbo` |
| `DEFAULT_DTYPE` | 字符串 | `float64` | 所有计算类型的 MACE 精度默认值。仅 MACE。 |
| `HEAD` | 字符串 | - | MACE head 或 DeepMD/DPA 多任务分支。 |
| `NEIGHBOR_CACHE` | 布尔 | `.TRUE.` | GRACE Verlet 邻居缓存；保留精确 cutoff 与完整周期像语义。浮点 bond 向量可有约 1e-14 Å 舍入差。仅 GRACE。 |
| `NEIGHBOR_SKIN` | 浮点数 | `1.5` | GRACE 邻居缓存的皮肤距离（Å）：任意原子位移超过 `NEIGHBOR_SKIN/2` 时重建邻居表。仅 GRACE。 |

#### 输出控制

| 键 | 类型 | 默认值 | 描述 |
|-----|------|--------|------|
| `WRITE_FORCES` | 布尔 | `.TRUE.` | 写入力到 OUTCAR |
| `WRITE_STRESS` | 布尔 | `.TRUE.` | 写入应力到 OUTCAR |
| `WRITE_TRAJECTORY` | 布尔 | `.TRUE.` | MD 写入轨迹 |
| `OUTPUT_FORMAT` | 字符串 | `VASP` | 输出格式：`VASP` |

#### 优化设置

| 键 | 类型 | 默认值 | 描述 |
|-----|------|--------|------|
| `OPT_ALGO` | 字符串 | `FIRE` | 优化器：`FIRE`、`BFGS`、`LBFGS` |
| `FMAX` | 浮点数 | `0.05` | 力收敛阈值 (eV/Å) |
| `MAX_STEPS` | 整数 | `500` | 最大优化步数 |
| `CELL_OPT` | 布尔 | `.FALSE.` | 优化晶胞参数 |
| `FIX_SYMMETRY` | 布尔 | `.FALSE.` | 保持对称性 |

#### MD 设置

| 键 | 类型 | 默认值 | 描述 |
|-----|------|--------|------|
| `MD_ENSEMBLE` | 字符串 | `NVT` | 系综：`NVT`、`NVE` |
| `TEMPERATURE` | 浮点数 | `300.0` | 温度 (K) |
| `TIMESTEP` | 浮点数 | `1.0` | 时间步长 (fs) |
| `STEPS` | 整数 | `10000` | 生产段 MD 步数 |
| `EQUILIBRATION_STEPS` | 整数 | `0` | 生产段前的同系综平衡步数；`EQUIL_STEPS` 是别名 |
| `THERMOSTAT` | 字符串 | `LANGEVIN` | NVT 恒温器：`LANGEVIN`、`BUSSI`、`NHC` |
| `FRICTION` | 浮点数 | `0.001` | Langevin 摩擦系数（fs⁻¹） |
| `BUSSI_TAU` | 浮点数 | `1000.0` | Bussi/CSVR 耦合时间（fs） |
| `NHC_TDAMP` | 浮点数 | `100.0` | NHC 阻尼时间（fs） |
| `NHC_TCHAIN` | 整数 | `3` | NHC 链长度 |
| `NHC_TLOOP` | 整数 | `1` | NHC 恒温器子步数 |
| `SAVE_INTERVAL` | 整数 | `10` | 轨迹保存间隔 |
| `PRE_RELAX` | 布尔 | NVT=`.TRUE.`/NVE=`.FALSE.` | MD 前预弛豫（系综相关默认值） |
| `PRE_RELAX_STEPS` | 整数 | `50` | 预弛豫最大步数 |
| `PRE_RELAX_FMAX` | 浮点数 | `0.1` | 预弛豫力收敛阈值 (eV/Å) |

`SAVE_INTERVAL` 只控制包含位置、速度、力等完整构型的轨迹保存间隔。
`run.log` 中的能量和温度是轻量热力学记录，当前每个 MD step 输出一次；例如
400000 步会产生 400001 条 step 日志，但 `SAVE_INTERVAL=200` 时仍只有
2001 个轨迹构型。任务开头会分别打印这两个间隔。

### 7.2 模板示例

**INCAR.sp**（单点能）：
```bash
CALC_TYPE = SP
TASK = omat
MODEL_PATH = uma-s-1.pt
DEVICE = cpu
INFERENCE_MODE = default
WRITE_FORCES = .TRUE.
WRITE_STRESS = .TRUE.
```

**INCAR.opt**（几何优化）：
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

**INCAR.md**（分子动力学）：
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

### 7.3 布尔值

以下均被识别为 `TRUE` 和 `FALSE`（不区分大小写）：

- **TRUE**：`.TRUE.`、`.T.`、`TRUE`、`T`、`YES`、`Y`、`1`
- **FALSE**：`.FALSE.`、`.F.`、`FALSE`、`F`、`NO`、`N`、`0`

---

## 8. 输出文件参考

### 8.1 文件清单

| 文件 | 生成者 | 格式 | 描述 |
|------|--------|------|------|
| `OUTCAR` | SP, OPT | 文本 | VASP 风格详细输出，含能量、力、应力、耗时 |
| `vasp/OUTCAR` | MD | 文本 | 明确标注的 VASP-like 子集；逐帧位置、力、能量、温度、晶胞和应力 |
| `CONTCAR` | SP, OPT | 文本 | 当前/最终原子结构（VASP POSCAR 格式） |
| `vasp/CONTCAR` | MD | 文本 | MD 最终结构，VASP POSCAR/CONTCAR 语法 |
| `OSZICAR` | OPT | 文本 | 逐步优化进度，含能量和力 |
| `vasp/XDATCAR` | MD | 文本 | VASP 语法兼容轨迹，使用未折回分数坐标 |
| `mlipx_results.json` | SP, OPT | JSON | 机器可读结果，含所有计算量 |
| `raw/mlipx_results.json` | MD | JSON | MD 机器可读结果摘要和规范化数据路径 |
| `raw/trajectory.traj` | MD | 二进制 | 不丢失坐标、速度、力等信息的 ASE 轨迹 |
| `raw/md.csv` | MD | CSV | 逐帧热力学标量、构型/总应力及对应压力；仅完整 3D PBC 报告应力/压力 |
| `artifacts.json` | MD | JSON | 版本化输出契约、单位、帧间隔和文件索引 |
| `optimization.log` | OPT | 文本 | ASE 优化器日志 |
| `run.log` | 所有 | 文本 | 实时刷新日志；CLI 与 TUI 启动时显示其路径 |
| `batch_summary.json` | BATCH | JSON | 批量处理所有结果的汇总 |

每次计算启动后，终端或 TUI 日志区都会显示类似提示：

```text
Live log: /absolute/path/to/results/run.log
Follow live output: tail -f /absolute/path/to/results/run.log
```

`run.log` 逐条刷新，可以在另一个终端执行所显示的命令实时查看。计算完成后，终端日志、`run.log`、对应的 `OUTCAR` 末尾及 JSON 输出均记录：

- **总耗时**：用户发起运行到所有标准输出文件生成完成。
- **实际计算耗时**：模型加载就绪后的首个计算阶段到开始写输出文件；不包含模型加载和最终输出写入。

### 8.2 OUTCAR 格式

OUTCAR 文件包含以下部分：

SP/OPT 继续使用原有摘要格式。MD 的 `vasp/OUTCAR` 使用版本化的
`mlipx.vasp-like-outcar.md/1` 子集，文件头明确声明它不是原生 VASP
OUTCAR；每个保存帧包含晶格、`POSITION / TOTAL-FORCE`、势能、动能、
总能、温度、体积、ASE 应力和压力，不虚构 SCF、POTCAR 或电子结构数据。

```
================================================================================
                         UMA CALCULATION RESULTS
================================================================================

--------------------------------------------------------------------------------
 SYSTEM INFORMATION          ← 体系信息（化学式、原子数、元素种类）
--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
 MODEL INFORMATION           ← 模型信息（路径、设备、推理模式、支持属性）
--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
 INPUT STRUCTURE             ← 输入结构（晶格矢量、原子位置）
--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
 ENERGY                      ← 能量（总能量、每原子能量）
--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
 FORCES (eV/Å)              ← 力（每个原子的力分量、最大力、RMS 力）
--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
 STRESS TENSOR               ← 应力张量（Voigt 表示、GPa、压力）
--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
 TIMING                      ← 耗时
--------------------------------------------------------------------------------

================================================================================
 END OF UMA CALCULATION
================================================================================
```

---

## 9. 任务类型参考

### 9.1 UMA 任务（`MODEL_TYPE = UMA`）

UMA 模型在不同数据集上训练；每个任务对应特定领域：

| 任务 | 领域 | 体系 | 电荷/自旋 | 应力 | 典型用途 |
|------|------|------|-----------|------|----------|
| `omat` | 无机材料 | 块体晶体 | 可选 | ✓ | 电池材料、固态电解质、氧化物 |
| `omol` | 分子 | 孤立分子 | 默认 0/1 | ✗ | 有机化学、药物分子 |
| `oc20` | 催化 (OC20) | 表面 slab | 可选 | ✓ | 多相催化、吸附 |
| `oc25` | 催化 (OC25) | 表面 slab | 可选 | ✓ | 扩展催化基准 |
| `odac` | MOFs | 金属有机框架 | 可选 | ✓ | 气体存储、分离 |
| `omc` | 分子晶体 | 有机晶体 | 可选 | ✓ | 药物、有机电子学 |

当前安装的 `fairchem-core 0.1.dev1316` 以其 `UMATask` API 为准，其中不包含
OC22；因此 mlipx 不会在 TUI 暴露随后会被计算器拒绝的 `oc22`。加载 checkpoint
时还会再次按该 checkpoint 的实际数据集检查 task。

**分子体系（omol）重要提示：** UMA 的 `omol` 任务需要总电荷与自旋多重度。若未
设置，mlipx 自动填 `charge=0`、`spin=1`（单重态）。TUI 会在选择 `omol` 后显示
这两个字段；CLI/INCAR 可直接设置：

```bash
mlipx sp molecule.xyz --model uma.pt --task omol --charge -1 --spin 2
# INCAR.mlipx: CHARGE = -1, SPIN = 2
```

也可在 Python 中写入结构元数据：

```python
from ase.io import read, write

atoms = read("molecule.xyz")
atoms.info["charge"] = 0    # 净电荷
atoms.info["spin"] = 1      # 自旋多重度 = 2S+1
write("molecule.xyz", atoms)
```

mlipx 只验证、不改写 PBC：周期任务（omat、oc20、oc25、odac、omc）要求输入为完整三维 PBC 和非退化晶胞；omol 要求输入完全非周期。

### 9.2 通用任务（MACE / DPA / GRACE）

非 UMA 引擎本身不感知 task，`task` 仅控制周期性边界（PBC）策略：

| 任务 | PBC | 等价 UMA task | 适用 |
|------|------|---------------|------|
| `bulk` | True | `omat` | 周期性晶体、表面、MOF |
| `molecule` | False | `omol` | 孤立分子 |

```bash
# MACE 周期性材料
.venv-mace/bin/mlipx sp bulk.cif --model mace.model --model-type mace --task bulk

# DPA 孤立分子
mlipx sp molecule.xyz --model dpa2.pth --model-type dpa --task molecule
```
---

## 10. 后台任务管理

从 TUI 启动的 SP、OPT 和 MD 任务默认作为独立后台进程运行。运行页面中的
**Back (Keep Running)** 可安全返回主界面；也可以直接退出 TUI，随后从
Background Jobs 页面或以下 CLI 命令查看和管理任务。

```bash
# 列出所有任务
mlipx jobs

# 终止正在运行的任务
mlipx kill <job_id>

# 清理已完成/失败的任务记录
mlipx clean
```

任务状态文件存储在 `~/.mlipx/jobs/`。

---

## 11. 资源控制

常用资源选项已经直接出现在 TUI 和 CLI 中；环境变量只是可选替代方式，
不是使用 TUI 的前置要求。

| 控制项 | TUI | CLI | 适用后端 |
|---|---|---|---|
| 设备 / GPU 编号 | **Device** | `--device cpu`、`cuda` 或 `cuda:N` | 全部 |
| CPU 线程数 | **CPU Threads** | `--cpu-threads N` | 全部 |
| 推理预设 | **UMA Inference Mode** | `--inference-mode default|turbo` | 仅 UMA |
| 激活检查点 | **UMA Activation Checkpointing** | `--activation-checkpointing` / `--no-activation-checkpointing` | 仅 UMA |

### 11.1 CPU 线程

```bash
mlipx sp structure.cif --model uma-s-1.pt --cpu-threads 4
```

TUI 中填写 **CPU Threads** 即可；留空表示让后端/系统自行决定。mlipx 会把
该值映射到 UMA/MACE 和 DPA PyTorch 模型的 PyTorch intra-op 线程，或
GRACE 的 TensorFlow intra-op 线程。DPA 的旧 TensorFlow `.pb` 模型使用
DeepMD TensorFlow 后端自身的线程设置。旧写法 `--torch-num-threads` 继续作为兼容别名；
`OMP_NUM_THREADS=4` 仍可作为 PyTorch 后端的环境级替代。

### 11.2 GPU 显存

UMA 激活检查点会以部分速度换取更低的显存占用：

```bash
mlipx sp structure.cif --model uma-s-1.pt --device cuda \
    --activation-checkpointing
```

省略该选项（TUI 选择 **Auto**）时跟随推理模式预设；若结构能够放入显存且
更重视速度，可用 `--no-activation-checkpointing`。

### 11.3 GPU 选择

CLI 使用 `--device cuda:N`，TUI 在 **Device** 中填写相同值：

```bash
mlipx sp structure.cif --model uma-s-1.pt --device cuda:0
```

DPA 和 GRACE 会在创建计算器前把这个统一选择映射为 DeepMD/TensorFlow
使用的环境机制。`CUDA_VISIBLE_DEVICES` 仍适合做进程级 GPU 隔离，但普通
TUI/CLI 使用无需手工设置。

### 11.4 推理模式

| 模式 | 最适用于 |
|------|----------|
| `default` | 一般用途、SP、OPT |
| `turbo` | MD、大体系、生产环境 |

UMA 的 MD 默认使用 `turbo`，SP/OPT 默认使用 `default`；可以在 TUI 或用
`--inference-mode` 显式覆盖。MACE、DPA 和 GRACE 不使用这个 UMA 专属选项。

---

## 12. 故障排除与常见问题

### 12.1 常见错误

#### "No edges found in structure"（未找到边）

**原因：** 模型无法为结构构建邻接图。原子距离太远或 PBC 设置错误。

**解决方案：**
1. 检查输入结构文件
2. 确保原子间距在 ~6 Å 以内
3. 周期性体系用 `omat`，分子用 `omol`

#### "no kernel image is available for execution on the device"

**原因：** 当前 PyTorch 构建没有与你的 GPU 兼容的 CUDA 内核。PyTorch 2.7+ 从预编译 CUDA wheel 中移除了 `sm_50`/`sm_60`，导致 Maxwell/Pascal 架构（compute capability 5.x/6.x）失去匹配内核。

| GPU 系列 | CC | 推荐 torch |
|-----|-----|------|
| Maxwell（GTX 750/9xx，含 GTX 960） | sm_50/52 | `torch==2.6.0+cu124` |
| Pascal（GTX 10xx, P104-100） | sm_60/61 | `torch==2.6.0+cu124` |
| Volta~Hopper（V100…H100, RTX 20/30/40） | sm_70~90 | `torch==2.6.0+cu124` |
| Blackwell（RTX 50） | sm_100/120 | `torch==2.8.0+cu128` |
| Kepler（GTX 700/600） | sm_30/37 | 不支持 |

> **`sm_50`/`sm_60` 内核分别与 `sm_52`/`sm_61` 设备二进制兼容**（同一 major 内 minor 向下兼容）。因此仍含 `sm_50`/`sm_60` 的 PyTorch 构建——例如 `torch 2.6.0+cu124` wheel——可以正常驱动 GTX 960（sm_52）和 P104-100（sm_61）。问题出在 torch 2.7+，而非 GPU 本身。

**诊断（装 torch 前即可用）：**
```bash
uv run mlipx setup      # 检测 GPU 并打印精确的 torch 安装命令
uv run python -c "import torch; print(torch.__version__, torch.cuda.get_arch_list())"
uv run mlipx doctor --engine uma --device cuda:0  # 检查 CC 与 PyTorch 架构
```

**解决方案（推荐顺序）：**
1. 安装该 GPU 推荐的 PyTorch 构建（见上表）：
   ```bash
   clashctl on  # 如需代理
   uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124   # Maxwell~Hopper
   # uv pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128  # Blackwell
   ```
2. 从源码编译含所需内核的 PyTorch：`TORCH_CUDA_ARCH_LIST="6.0;6.1" python setup.py develop`
3. 回退到 CPU：`--device cpu`


#### CUDA 显存不足

**解决方案：**
1. 切换到 CPU：`--device cpu`
2. 使用较小的模型

#### MD 原子爆炸

**解决方案：**
1. NVT 默认启用仅位置预弛豫；NVE 默认关闭，只有接受改变输入初始位置时才用 `--pre-relax` 开启
2. 先运行几何优化：`mlipx opt ...` 再 `mlipx md CONTCAR ...`
3. 降低初始温度
4. 若出现 NaN/inf 能量或力，运行会自动中止（不会写出 NaN 结果）--检查输入结构是否有原子过近或超出训练分布

---

## 13. 性能指南

| 体系 | 原子数 | CPU（8 核） | GPU（RTX 3080） |
|------|--------|-------------|-----------------|
| Li₃PS₄（块体） | 28 | 2-3 s | 0.5-1 s |
| Cu slab（3×3×4） | 144 | 15-20 s | 3-5 s |
| MOF-5 | 424 | 60-90 s | 15-20 s |

---

## 14. 使用示例

### 14.1 电池材料能量

```bash
mlipx sp LLZO.cif --model uma-s-1.pt --task omat --device cuda
```

### 14.2 表面弛豫

```bash
mlipx opt Pt111_slab.cif \
    --model uma-s-1.pt --task oc20 \
    --fmax 0.02 --max-steps 300 --optimizer FIRE --device cuda
```

### 14.3 NVT 分子动力学

```bash
mlipx md CONTCAR --model uma-s-1.pt \
    --ensemble NVT --temp 400 --steps 100000 --device cuda
```

### 14.4 批量筛选

```bash
uv run mlipx batch candidates/ --model uma-s-1.pt \
  --model-type uma --task omat --calc-type sp --pattern "*.cif"
```

### 14.5 状态方程 (EOS)

计算能量随体积变化的曲线：

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
# 随后收集 mlipx_results.json 中的能量拟合 EOS
```

### 14.7 NEB 过渡态准备

生成 NEB 中间镜像并对每个镜像优化：

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

### 14.8 声子计算 (配合 phonopy)

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

### 14.9 形成能计算

```python
from ase.io import read
from collections import Counter
import subprocess, json

compound = read("Li2O.cif")
subprocess.run(["mlipx","sp","Li2O.cif","--model","uma-s-1.pt",
                "--task","omat","--output","Li2O_results"])
with open("Li2O_results/mlipx_results.json") as f:
    e_total = json.load(f)["calculation"]["results"]["energy"]

e_ref = {"Li": -1.9, "O": -4.9}  # 元素参考能（需单独计算）
e_form = e_total - sum(c * e_ref[el] for el, c in
                       Counter(compound.get_chemical_symbols()).items())
print(f"形成能: {e_form/len(compound):.4f} eV/atom")
```

### 14.10 推荐设置速查

| 体系类型 | 推荐设置 |
|----------|----------|
| 小分子 (1-50 原子) | `--optimizer LBFGS --fmax 0.01` |
| 块体材料 | `--optimizer FIRE --fmax 0.05` |
| 表面 | `--optimizer FIRE --fmax 0.03` |
| MD 平衡 | `--ensemble NVT --timestep 1.0` |
| MD 生产 | `--ensemble NVE --timestep 1.0` |
| 高温 MD | `--timestep 0.5 --friction 0.002` |

---

## 15. 许可证

本项目基于 MIT 许可证开源。

版权所有 (c) Meta Platforms, Inc. and affiliates.
