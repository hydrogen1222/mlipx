# mlipx 科学计算核心核查报告(只读核查,未改动)

> **性质**:只读审计。本报告只核查与科学计算正确性相关的核心逻辑(能量/力/应力、单位换算、MD 系综、温度/自由度、电荷/自旋约定、数值稳定性、输出格式约定),**不做任何代码修改**。
> **背景**:按用户要求,算法/理论层面若发现问题仅汇报总结,由用户决定是否处理;本轮代码修复仅针对工程性 bug。
> **核查对象**:`mlipx/mlipx/`(引擎、runners、writers、calculators、config)及其依赖的 `src/fairchem/core`(fairchem 官方后端)与 ASE 库。

---

## 0. 结论摘要

| 结论 | 数量 |
|---|---|
| ✅ 核查通过(符合 ASE / fairchem 官方约定) | 8 项 |
| ⚠️ 发现但**未修改**(不影响当前运行,仅汇报) | 2 项 |

**一句话结论**:核心科学计算链路(能量/力/应力求值、单位换算、MD 积分、温度/DOF、电荷/自旋传递、NaN 守卫)均与 ASE 与 fairchem 官方接口约定一致,未发现需要修改的原理性问题。两处发现均为**死代码或格式约定层面**问题,不影响任何实际运行结果。

---

## 1. 核查范围与方法

- **静态核查**:通读 `mlipx/mlipx/` 中 runner / writer / calculator / engine 全链路代码,并与依赖库(ASE、fairchem-core)的官方接口定义逐条对照。
- **动态验证**:用 ASE 内置解析势 EMT(fcc Al₄)通过完整引擎管线(engine → runner → writer)跑 sp / opt(位置+晶胞)/ MD(NVT、NVE),收集数值证据(见 §4)。
- **对照验证**:fairchem 官方 `FAIRChemCalculator` 源码(`src/fairchem/core/calculate/ase_calculator.py`)确认电荷/自旋读取约定。

---

## 2. 核查结论总表

| # | 主题 | 结论 | 关键位置 |
|---|------|------|----------|
| 1 | 能量/力/应力求值链路 | ✅ 通过 | `runners/singlepoint.py`、`runners/base.py` |
| 2 | 单位换算(fs、kB、GPa) | ✅ 通过 | `runners/md.py`、`utils.py`、`writers/outcar.py` |
| 3 | 温度与自由度(DOF)计算 | ✅ 通过 | `runners/md.py::_calculate_temperature` |
| 4 | 速度初始化与 NVT/NVE 积分 | ✅ 通过 | `runners/md.py`(Langevin/FixCom/VelocityVerlet) |
| 5 | 电荷/自旋(charge/spin)约定 | ✅ 通过 | `runners/base.py::_prepare_atoms` ↔ fairchem 官方 |
| 6 | 预松弛与几何优化 | ✅ 通过 | `runners/md.py::_pre_relax_structure`、`runners/optimization.py` |
| 7 | 数值稳定性守卫(NaN/Inf 中止) | ✅ 通过 | `runners/base.py::_check_finite` |
| 8 | GPU 架构二进制兼容判断 | ✅ 通过 | `gpu_compat.py::arch_supports_device` |
| 9 | VASP 输出格式约定 | ✅ 已修复 | XDATCAR 使用 VASP 固定列宽和未折回分数坐标(见 §3.9) |
| 10 | 死代码中的潜在判断缺陷 | ⚠️ 汇报 | `utils.py::check_structure_valid` 用 `np.diag` 代替行列式(见 §3.10) |

---

## 3. 逐项核查详情

### 3.1 能量/力/应力求值链路 ✅

- 所有 runner 通过 `BaseMLIPCalculator.get_calculator()` 拿到标准 ASE `Calculator`,只调用 `atoms.get_potential_energy() / get_forces() / get_stress()`,不直接触碰任何后端张量,接口契约干净。
- 应力仅在 `calculator.has_stress and atoms.pbc.any()` 时求值(`singlepoint.py`、`optimization.py`)——对非周期分子不强行计算应力,避免 MACE 等对非周期体系返回无意义应力。✅
- 优化回调中能量/力在 `opt.nsteps` 后重新求值,与 ASE 优化器状态一致。✅

### 3.2 单位换算 ✅

| 换算 | 代码 | 数值验证 |
|------|------|----------|
| MD 时间步 fs → ASE 时间单位 | `timestep * units.fs`(`md.py`) | `units.fs = 9.8227e-02`(ASE 定义),与 ASE 官方 MD 教程一致 |
| Langevin 摩擦 1/fs → ASE 单位 | `friction / units.fs` | 同上 |
| 温度公式 `2·KE / (n_dof·k_B)` | `_calculate_temperature` 回退路径 | `units.kB = 8.6173e-05 eV/K`,公式与 ASE `get_temperature()` 同源 |
| eV/Å³ → GPa | `× 160.2177`(`utils.py`、`outcar.py`、`json_writer.py`) | 实测:单轴 1 eV/Å³ → P = −53.4059 GPa ✅;CODATA 精确因子 160.21766208,代码精度(7 位有效数字)在输出格式精度内 |

### 3.3 温度与自由度(DOF) ✅

- `_calculate_temperature` 委托 `atoms.get_temperature()`,与 `MaxwellBoltzmannDistribution(force_temp=True)`、Langevin 恒温器使用**同一套约束 DOF 约定**(`3N − Σ removed_dof`),因此报告温度与目标温度自洽,不因 COM 修正偏移。✅
- `_ensure_com_constraint` 用显式 `FixCom` + `Langevin(fixcm=False)`,是 ASE 官方推荐的规范采样形式(替代被弃用的 `fixcm=True`),且使 COM 的 3 个自由度对 `get_temperature()` 可见。✅
- FixSymmetry 等不报告 removed_dof 的约束:回退 `3N−3` 并**显式警告**;不做硬编码 `−3` 猜测(后者对多原子晶胞是错误的)。✅

### 3.4 速度初始化与 MD 系综 ✅

- `MaxwellBoltzmannDistribution(force_temp=True, rng=seeded_rng)` + `Stationary(preserve_temperature=True)`:标准做法;`rng` 同时驱动初始速度与 Langevin 随机力,因此**种子能复现整个 NVT 轨迹**(不只是初速度)。✅
- `velocity_policy`(auto/initialize/preserve)以"是否已有 momenta 数组"判定重启,`preserve` 对无速度结构显式报错,`auto` 语义与文档一致。✅
- NVE 用 `VelocityVerlet`,默认关闭 pre-relax(避免改变守恒能量基线),NVT 默认开启(防爆炸),显式用户设置优先。✅
- **数值证据**:EMT Al₄ NVE 40 步(1 fs/步),总能量漂移仅 `3.9e-5 eV`(0.035%),能量守恒正常(见 §4)。✅

### 3.5 电荷/自旋(charge/spin)约定 ✅(与 fairchem 官方对照)

- fairchem 官方(`src/fairchem/core/calculate/ase_calculator.py` L49–54、L190–193):
  - `omol` 模式从 `atoms.info["charge"]`(总电荷,整数)与 `atoms.info["spin"]`(**自旋多重度**,整数)读取;
  - 未设置时默认 `charge=0, spin=1`。
- mlipx `_prepare_atoms`(`runners/base.py`):对 UMA `omol` 任务 `setdefault("charge", 0)`、`setdefault("spin", 1)` —— **与官方默认值完全一致**;且显式设置使运行确定。✅
- 对通用引擎(MACE/DPA/GRACE)的 `molecule` 任务:只默认 `charge=0`,**不注入 spin**——因为 MACE 的 `spin` 语义是未配对电子数(0=单重态),若照搬"多重度 1"会错误地把分子变成双自由基。该区分正确。✅

### 3.6 预松弛与几何优化 ✅

- `_pre_relax_structure` 用 FIRE 仅优化原子位置(`fmax`/`steps` 上限),不改晶胞,符合"降低初始大力、不保证极小点"的文档语义;失败时**fail-closed**(不带着未收敛/出错结构进 MD)。✅
- 优化器 `FIRE/BFGS/LBFGS` 均为 ASE 标准实现;`cell_opt` 用 `FrechetCellFilter`(恒定压力下正确的晶胞自由度处理);`fix_symmetry` 用 ASE `FixSymmetry`。✅
- `cell_opt` 对 `omol/molecule` 任务或 `has_stress=False` 时自动降级并警告——物理上正确。✅

### 3.7 数值稳定性守卫 ✅

- `_check_finite`:能量/力任一 NaN/Inf 即抛错中止,**永不写入"成功"的 NaN 结果**;MD 每步、优化每步、最终结构均检查。✅
- MD 爆炸预警:`max|F| > fmax_abort`(默认 20 eV/Å)每 100 步告警(不中止),阈值来自 `[safety]` 配置而非魔法数字。✅

### 3.8 GPU 架构二进制兼容判断 ✅

- `arch_supports_device`:`sm_XY` 内核可在同 major、`minor ≤ 设备minor` 的 GPU 上运行(如 sm_60 内核 → sm_61 设备),这是 CUDA SASS 二进制兼容规则的正确建模。
- 实测:`('sm_61', ['sm_60','sm_80']) → True`、`('sm_50', ['sm_60']) → False`、`('sm_100', ['sm_90']) → False`,均符合预期。✅

### 3.9 ✅ XDATCAR 使用未折回分数坐标

- **旧实现**:`writers/xdatcar.py::append_frame` 调用 ASE 默认的 `wrap=True`,原子跨晶胞边界后会被折叠回 `[0,1)`。
- **VASP 惯例**:XDATCAR 写 **unwrapped** 分数坐标,长轨迹中原子平滑穿越边界;折叠写法在 ASE/常见可视化工具读取上**完全兼容**,仅在超长轨迹中观察扩散/位移时坐标会"跳回"。
- **当前实现**:原生 writer 使用 `get_scaled_positions(wrap=False)`,采用 VASP 固定列宽并保留跨周期连续坐标。历史结果应优先从 `trajectory.traj` 重建,因为已经折回的旧 XDATCAR 无法恢复 image 信息。

### 3.10 ⚠️ 发现 2:`check_structure_valid` 用 `np.diag(cell)` 代替行列式(死代码,未修改)

- **位置**:`utils.py::check_structure_valid`。
- **缺陷**:`np.diag(atoms.cell)` 取的是"每个晶格矢量在自身轴上的投影",**不是体积**。对非正交/退化晶胞,`np.diag` 全正但行列式(真实体积)可以为 0。
  - 实测构造:cell = `[[4,0,0],[0,2,4],[4,2,4]]`(三矢量共面)→ `np.diag = [4,2,4]` 全正 → 该函数返回"有效";而 `det = 0`(零体积)。
- **为何不影响运行**:全仓检索确认该函数**无任何调用方**(引擎实际用的是 `atoms.cell.volume` 行列式判断,`_prepare_atoms` / `SinglePointRunner` 均正确)。属于死代码中的潜在缺陷。
- **未修改原因**:按用户要求仅汇报;若未来启用该函数,应改为 `atoms.cell.volume <= 0`(与引擎路径一致)。

---

## 4. 数值验证证据(EMT Al₄,秒级)

| 验证 | 结果 |
|------|------|
| sp:能量有限性 + 6 分量应力 | ✅ 能量 −0.0060 eV,应力 `[0.0097, 0.0097, 0.0097, ~0, 0, 0]`(Voigt `[xx,yy,zz,yz,xz,xy]`) |
| 应力方向性:拉伸 x 后 xx 分量最大 | ✅ `[0.0222, 0.0162, 0.0162, ...]` |
| FIRE 位置优化收敛 | ✅ 18 步收敛于 fmax=0.01 eV/Å,OSZICAR 收敛标记正确 |
| FrechetCellFilter 晶胞优化收敛 | ✅ 18 步收敛 |
| NVE 40 步能量守恒 | ✅ 总能量漂移 `3.9e-5 eV`(0.035%) |
| NVT 60 步(含预松弛)+ Langevin | ✅ XDATCAR 7 帧可被 ASE 读回,md.csv 标量流式写入正确 |
| 协作取消 | ✅ 干净抛 `CancellationRequested`,不产生半成品"成功"结果 |
| GPU 兼容判断 | ✅ 三组用例符合 CUDA SASS 二进制兼容规则 |

复现方式:`tests/mlipx/mlipx/test_writers.py` 等单测 + 引擎级 EMT 冒烟脚本(临时文件,未入库)。

---

## 5. 发现与处置汇总

| # | 位置 | 问题 | 影响 | 当前处置 |
|---|------|------|------|------------------|
| 1 | `writers/xdatcar.py::append_frame` | 旧实现用 `wrap=True`，不符合 VASP 连续轨迹惯例 | 旧长轨迹跨胞时出现坐标“跳回” | ✅ 已改为 `get_scaled_positions(wrap=False)`，并增加标准格式测试与转换命令 |
| 2 | `utils.py::check_structure_valid` | `np.diag(cell)` 不能代表体积,共面零体积晶胞误判为有效 | 无:死代码,无调用方 | 若启用,改用 `cell.volume <= 0` |

**核查结论：核心理论计算原理未发现硬错误；已修复影响轨迹交换语义的
XDATCAR 输出问题。剩余一项位于无调用方的死代码中，若将来启用该函数再改用
晶胞体积/行列式判断。**
