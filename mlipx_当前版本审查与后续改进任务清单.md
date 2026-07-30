# mlipx 后续审查与改进任务清单

> 目标仓库：`https://github.com/hydrogen1222/mlipx`
> 适用范围：当前仓库最新版本，而不是任何固定历史提交。
> 文档用途：交给代码 Agent 进行全面调查、验证和必要修改。
> 核心原则：**先调查当前源码和官方后端接口，再判断问题是否仍然存在；不得仅根据本清单直接假定代码有错。**

---

## 0. 使用说明与边界

本清单根据此前对仓库的有限静态阅读整理，但仓库已经经过多轮更新，且远程源码读取可能存在遗漏、缓存、分支差异或上下文不完整。因此：

1. 本文中的每一项都应视为**待验证假设、审查方向或潜在改进点**，不是已经确认的 bug。
2. Agent 必须以当前工作区源码、当前依赖版本、实际测试结果和官方文档为准。
3. 在修改代码前，应先给出证据：
   - 涉及文件和函数；
   - 当前控制流或参数流；
   - 可复现条件；
   - 实际错误后果；
   - 对应官方 API 或源码依据；
   - 当前测试是否已经覆盖。
4. 如果某项在当前版本中已经正确实现，应记录为“已验证，无需修改”，不要重复重构。
5. 如果实现方式与本文建议不同但行为正确，也不应为了形式一致而修改。
6. 避免一次性大范围重写。优先做小范围、可回退、带测试的修改。
7. 不要先修改 TUI、WebUI 或外观层，除非底层接口已经稳定且相关问题确实来自前端。

---

## 1. 明确不纳入本轮修改的内容

### 1.1 不要求将有限大力警告改成强制终止

当前 MD 中若已经对 NaN/Inf 做了硬错误处理，而对有限但较大的力只进行警告，这一设计可以保留。

此前“框框炸飞”指的是：

> 显存被大量占用并发生 OOM，即“嘎嘎炸显存”，不是原子飞出周期性边界盒子。

因此本轮不应仅因为最大力超过某个固定阈值，就强制终止所有 MD。不同体系、初始结构和高温条件下，固定阈值可能产生误杀。

可以审查但不强制修改的内容：

- 警告是否清晰；
- 是否记录最大力、步数和原子编号；
- NaN/Inf 是否会可靠终止；
- 是否允许用户自行配置警告阈值；
- 是否避免每一步重复刷屏。

除非 Agent 通过真实测试证明当前处理会造成明确错误，否则不要把有限大力警告改成默认 abort。

---

### 1.2 不要求在 mlipx 中实现扩散和电导率分析

mlipx 的主要职责是：

- 统一调用不同 MLIP 后端；
- 完成 SP、OPT、MD 等任务；
- 稳定生成标准轨迹和结构输出；
- 支持批量任务、恢复和结果追踪。

Li-MSD、扩散系数、Nernst–Einstein 电导率、Haven ratio 和阿伦尼乌斯拟合可以由独立分析脚本或其他工具完成。

因此本轮不新增：

```text
mlipx analyze diffusion
MSD 分析器
扩散系数拟合
离子电导率计算
阿伦尼乌斯拟合
```

但应保证 mlipx 输出的轨迹具有足够信息，可供外部程序正确分析：

- 原子顺序稳定；
- 晶胞和 PBC 正确；
- 时间步和保存间隔明确；
- 轨迹中保留坐标；
- 重启后时间和步数连续或有清晰元数据；
- 不因内存优化而破坏轨迹完整性。

---

# 2. 审查工作流要求

Agent 开始工作后，应按以下顺序执行。

## 2.1 建立当前仓库事实基线

首先运行：

```bash
git status --short
git branch --show-current
git rev-parse HEAD
git log -1 --oneline
git diff --stat
```

记录：

- 当前分支；
- 当前提交；
- 是否有未提交修改；
- Python 版本；
- mlipx 版本；
- 当前测试状态。

然后查看仓库结构，但不要无目的地一次性读取所有文件：

```bash
find . -maxdepth 3 -type f | sort
```

重点定位：

```text
Engine / EngineConfig
CalculatorFactory
UMA wrapper
MACE wrapper
DPA wrapper
GRACE wrapper
MDRunner
OptimizationRunner
Batch / Queue
配置解析
settings.ini
测试目录
文档
```

---

## 2.2 先运行测试，再审查源码

至少运行当前仓库已有的测试：

```bash
pytest -q
```

若测试规模较大，可先列出：

```bash
pytest --collect-only -q
```

记录：

- 总测试数；
- 失败和跳过项；
- 哪些测试真正实例化了 Calculator；
- 哪些测试只验证 CLI 或配置解析；
- 是否存在需要真实模型文件的集成测试。

不得因为 `pytest` 全绿就直接判定后端正确。需要区分：

```text
配置测试
接口测试
Mock 测试
真实后端集成测试
数值一致性测试
物理行为测试
```

---

## 2.3 每项问题的报告格式

每一项审查结果使用：

```text
状态：
- CONFIRMED
- ALREADY_FIXED
- NOT_REPRODUCED
- UNCERTAIN
- DESIGN_CHOICE
- OUT_OF_SCOPE

证据：
- 文件和函数
- 当前行为
- 最小复现
- 官方依据
- 测试结果

影响：
- correctness
- reliability
- performance
- usability
- documentation only

处理：
- 修改
- 增加测试
- 更新文档
- 无需处理
```

不要只输出“看起来有问题”。

---

# 3. UMA 后端审查

UMA 经过多轮修正，预计其核心 Calculator 接入已经相对成熟。本轮目标不是重写 UMA，而是确认参数链、设备选择、任务设置和数值结果是否可靠。

## 3.1 参数传递

调查以下参数是否从所有入口完整传递：

```text
CLI
INCAR 或任务配置
settings.ini
Python API
EngineConfig
CalculatorFactory
UMA wrapper
官方 Predictor / Calculator
```

重点核对：

```text
model path
task
device
inference_mode
precision 或 dtype（若适用）
compile / turbo 相关选项
```

检查是否存在：

- 配置被解析但未使用；
- fallback 默认值覆盖用户值；
- CLI、INCAR 和 API 默认值不一致；
- 非 UMA 参数被错误传给 UMA；
- 允许的值与官方版本不匹配。

---

## 3.2 设备索引

如果支持：

```text
cuda
cuda:0
cuda:1
```

确认所有 GPU 检查都使用用户实际指定的设备，而不是硬编码设备 0。

需要调查：

- capability 检查；
- GPU 名称；
- 显存信息；
- turbo/compile 兼容性判断；
- 实际模型所在设备；
- `CUDA_VISIBLE_DEVICES` 后的逻辑索引。

单卡环境下不易暴露此问题，应增加 mock 或多设备索引测试。

---

## 3.3 官方直调一致性

准备一个很小的周期体系，使用相同：

```text
模型
task
device
inference_mode
结构
```

分别执行：

```text
A. 官方 UMA Calculator 直接调用
B. mlipx UMA wrapper 调用
```

比较：

```text
总能量
每原子力
应力
单位
原子顺序
PBC
```

要求误差在合理数值容差内。

若无法在 CI 中下载模型，可将此测试设为：

```text
integration
requires_model
```

并提供本地运行说明。

---

## 3.4 UMA 验收结论

只有满足下列条件，才能在文档中标为稳定支持：

- 单点结果与官方直调一致；
- 固定晶胞优化可运行；
- 应力接口行为明确；
- 短程 MD 可运行；
- 参数不会静默失效；
- 设备信息不误导；
- 至少有一个真实模型集成测试。

---

# 4. MACE 后端审查

MACE 的 `default_dtype`、模型 head 和参数传递据称已经经过修正。本轮应验证这些改动确实全链路生效。

## 4.1 dtype 全链路

调查：

```text
settings.ini
profile
CLI
INCAR
Python API
EngineConfig.calculator_options
CalculatorFactory
MACECalculatorWrapper
官方 MACECalculator
```

确认：

- `float32` 确实进入底层；
- `float64` 确实进入底层；
- `auto` 的语义清晰；
- SP、OPT、MD 的默认选择与文档一致；
- 用户显式设置不会被任务默认覆盖；
- 输出 metadata 记录的是实际 dtype，而不是请求值。

建议增加测试：

```text
用户不指定 dtype
用户指定 float32
用户指定 float64
profile 和 CLI 冲突时 CLI 优先
```

---

## 4.2 模型 head

调查当前 MACE 版本中正确参数名和允许值。

确认：

- 单头模型不需要 head 时可正常运行；
- 多头模型能选择指定 head；
- 无效 head 给出清晰错误；
- metadata 记录实际 head；
- 不将 `head` 错写成旧接口或不存在的复数形式。

必须以当前安装的 MACE 官方源码/API 为准，不应仅依赖旧文档。

---

## 4.3 模型信息与元素覆盖

检查 wrapper 是否能够读取或合理报告：

```text
支持元素
cutoff
dtype
head
模型数量
device
```

加载结构后，若模型不支持某元素，应尽早报错。

但不要为了读取 metadata 而依赖 MACE 内部不稳定私有属性。优先使用公开接口；若只能使用私有属性，应：

- 加版本兼容处理；
- 加测试；
- 在文档中说明。

---

## 4.4 性能选项

以下功能可以调查，但不是本轮阻断项：

```text
torch compile
fullgraph
cuEquivariance
OpenEquivariance
warmup
委员会模型
```

如果当前代码尚未实现，不必一次性补齐。

需要确保：

- 文档不宣称尚未实现的功能；
- 未实现参数不会被静默接受；
- 不会因默认启用实验性加速导致 V100 不兼容。

---

## 4.5 显存与长轨迹

针对 V100 16 GB，分别记录：

```text
50 原子 LGPS
400 原子 LGPS
float32
float64
```

至少测量：

- 模型加载后显存；
- 第一次力计算峰值；
- 稳态单步显存；
- 单步时间；
- 是否存在随 MD 步数持续上涨的 GPU 显存。

“框框炸显存”应按 OOM、缓存和张量生命周期调查，不要与原子穿越 PBC 混淆。

需要区分：

```text
一次性 PyTorch 缓存
正常 allocator reserved memory
真实泄漏
批量任务同时加载模型
轨迹保存在 GPU
Autograd graph 未释放
```

---

# 5. 公共 MDRunner 审查

本节只审查 mlipx 作为 MD 驱动器的可靠性，不要求加入扩散分析，也不要求把有限大力警告改成强制终止。

## 5.1 预弛豫语义

调查当前 MD 是否自动进行预弛豫，以及：

- 是只优化原子位置，还是同时优化晶胞；
- 日志是否准确描述；
- 未收敛后会发生什么；
- 异常后使用的是原始结构还是部分优化后的结构；
- 是否复制了初始 `Atoms`；
- 是否可关闭；
- 配置项是否真正生效。

推荐语义：

```text
none
positions
cell
```

但只有在当前设计确实需要时再引入。

关键要求：

- 不要声称 `FIRE(atoms)` 能消除晶胞应力；
- 如果预弛豫失败后继续，应明确说明使用哪个结构；
- 默认行为应稳定、可追踪；
- 正式 MD 可以从单独 OPT 的结果开始。

---

## 5.2 配置项是否静默失效

重点调查 Schema 中存在、但 Runner 可能未使用的参数，例如：

```text
SEED
VELOCITY_POLICY
EQUIL_STEPS
PRE_RELAX_MODE
PRE_RELAX_FAILURE
CHECKPOINT_INTERVAL
RESTART_FROM
APPEND
```

以当前源码为准确定实际清单。

每个配置项必须属于三种状态之一：

```text
已实现并测试
明确标记为未实现并拒绝
从 Schema 和文档中移除
```

不允许：

> 解析成功、校验成功、运行成功，但参数实际上没有任何作用。

---

## 5.3 随机种子和速度初始化

调查：

- Maxwell-Boltzmann 初始化是否接受 RNG；
- Langevin 是否使用同一或可记录 RNG；
- seed 是否可配置；
- 未指定 seed 时是否生成并记录；
- restart 时是否保留速度；
- restart 时是否错误地重新初始化速度；
- 是否移除质心平动；
- 是否处理转动；
- 约束原子的自由度是否正确。

建议行为：

```text
新结构：
  auto → 没有速度时初始化

重启结构：
  auto → 保留已有速度

initialize：
  强制重新初始化

preserve：
  没有速度则报错
```

具体接口以 ASE 当前版本为准。

---

## 5.4 温度和自由度

确认温度计算是否使用 ASE 官方的约束感知接口。

检查：

- `FixAtoms`；
- 其他约束；
- 质心平动处理；
- 零自由度；
- 单原子或小体系；
- 动能和温度单位；
- 日志温度是否与 ASE 一致。

避免维护一套容易漏约束类型的自定义自由度公式。

---

## 5.5 轨迹内存占用

调查当前长程 MD 是否同时：

```text
写入 trajectory.traj
又在 Python list 中保存每一帧的 atoms.copy()
```

如果仍然如此，应改为流式输出。

推荐：

```text
trajectory.traj  保存完整帧
md.csv           保存标量
内存             只保存当前帧和有限摘要
```

需要确认：

- XDATCAR 是否依赖内存轨迹；
- 是否可从 `.traj` 流式生成；
- Runner 返回值是否包含完整轨迹对象；
- 100 ps 时 RAM 是否随帧数线性增长。

增加一个不依赖真实 MLIP 的长轨迹测试，确认内存不会明显线性增长。

---

## 5.6 checkpoint 与 restart

调查当前是否已经实现：

```text
坐标
晶胞
PBC
动量
累计步数
物理时间
阶段
随机数状态
```

不要仅靠 CONTCAR 充当 MD restart。

至少应支持：

- 中断后从最后 checkpoint 恢复；
- 不重复初始化速度；
- 输出步数不从零造成歧义；
- append 不破坏旧轨迹；
- checkpoint 原子写入；
- 损坏 checkpoint 能被识别。

对于确定性 NVE，增加：

```text
连续 100 步
vs
50 步 + restart + 50 步
```

结果一致性测试。

对于 Langevin，调查是否需要保存 RNG 状态；若未保存，文档必须说明恢复后是统计连续而非逐步完全一致。

---

## 5.7 NaN/Inf 和有限大力

要求：

- 能量、力、位置或速度出现 NaN/Inf 时可靠终止；
- 保存有助于排查的最后状态；
- 错误返回非零状态；
- Batch 能识别失败。

有限大力可继续使用 warning 设计，不要求默认 abort。

可以改进：

```text
警告节流
记录最大力原子
记录力值和步数
用户可配置 warning threshold
```

但避免把固定阈值包装成普适物理判据。

---

# 6. DPA 后端审查

DPA 当前可能仍是较薄的 DeepMD Calculator 包装。Agent 应先识别仓库所说的 “DPA” 实际覆盖哪些模型和 DeepMD 版本。

## 6.1 模型格式

核对当前 DeepMD 官方版本对以下格式的定义：

```text
.pb
.pth
.pt
其他格式
```

调查：

- 哪些是冻结推理模型；
- 哪些是训练 checkpoint；
- 当前 wrapper 是否错误接受不可推理格式；
- 错误信息是否指导用户先 freeze/export；
- README、docstring 和 CLI help 是否一致。

不要根据旧版本经验直接修改，必须查询当前安装版本的官方文档或源码。

---

## 6.2 设备选择

调查 `--device` 是否真正控制 DPA 所在设备。

DeepMD 可能通过：

```text
CUDA_VISIBLE_DEVICES
环境变量
框架配置
模型后端
```

而不是 Calculator 的 `device=` 参数。

需要确认：

- wrapper 是否只是保存 requested device；
- 实际计算设备如何确定；
- Queue 子进程是否设置环境变量；
- CPU 模式是否可靠；
- `cuda:1` 是否映射正确；
- metadata 是否区分 requested 和 actual。

禁止报告未经验证的：

```text
device = cuda:0
```

建议：

```text
requested_device
actual_backend
actual_device
```

如果 actual device 无法可靠获得，应明确标为 unknown，而不是猜测。

---

## 6.3 多头模型和专属参数

检查当前 DeepMD `DP` ASE Calculator 是否支持：

```text
head
type_dict
nlist_backend
其他推理参数
```

确认 mlipx 是否需要暴露。

原则：

- 只暴露有实际使用场景且能够测试的参数；
- 不要一次性复制官方 API 全部参数；
- 未实现参数不应静默忽略。

---

## 6.4 官方直调一致性

使用一个真实、合法的 DPA/DeepMD 冻结模型：

```text
官方 DP Calculator
vs
mlipx DPA wrapper
```

比较：

```text
能量
力
应力
数据类型
设备
原子类型映射
```

尤其检查：

- `type_dict=None` 是否在目标模型中可靠推断；
- 元素顺序变化；
- 模型 type map 与 ASE chemical symbols；
- virial/stress 符号和单位。

---

## 6.5 DPA 最低验收

在文档中标为稳定支持前，至少完成：

- 一个真实冻结模型的 CPU SP；
- 一个真实冻结模型的 GPU SP；
- 官方直调一致性；
- 固定晶胞 OPT；
- 10～100 步短 MD；
- 模型格式错误测试；
- 设备元数据测试。

---

# 7. GRACE 后端审查

GRACE 可能依赖 TensorFlow、TensorPotential 或 GraceMaker。必须以仓库当前锁定或安装版本为准。

## 7.1 模型加载来源

调查当前支持的是：

```text
本地 SavedModel
foundation model 名称
本地多模型委员会
其他导出格式
```

确认：

- `model_path` 是否必须存在；
- 是否支持官方 foundation model helper；
- 路径与模型名如何区分；
- 错误提示是否清晰；
- 下载和缓存是否属于 mlipx 职责。

不必强制实现 foundation model alias，但文档必须准确反映当前能力。

---

## 7.2 mode 选择

核对当前 GRACE/TensorPotential Calculator 中：

```text
uniform
diverse
```

等 mode 的真实含义、默认值和适用场景。

调查：

- 单结构 SP；
- OPT；
- 固定原子数 MD；
- 异构结构 Batch；

是否应使用不同 mode。

不要直接假定所有 MD 都必须改成某个值。应通过：

- 官方文档；
- Calculator 源码；
- 编译次数；
- 运行时间；
- 数值一致性；

确定合理默认值。

如果 mode 只影响性能而不影响结果，应标为 performance，而非 correctness。

---

## 7.3 最小距离参数

调查 GRACE Calculator 是否提供：

```text
min_dist
```

或类似短键保护参数。

需要判断：

- 当前版本是否支持；
- 默认值；
- 触发行为；
- 是否适合在 wrapper 中暴露；
- 是否与公共 NaN/Inf 检查重复；
- 是否可能误杀高压或特殊结构。

这属于可选后端安全功能，不要求强制默认开启。

---

## 7.4 Padding 和 XLA

调查：

```text
pad_neighbors_fraction
pad_atoms_number
XLA 编译缓存
输入 shape
```

对：

```text
单结构 MD
同尺寸 Sweep
异构 Batch
```

的影响。

记录：

- 首步编译时间；
- 后续单步时间；
- 显存；
- 是否重复编译；
- 是否随结构变化重新 tracing；
- 结果是否一致。

只在有基准数据支持时调整默认值。

---

## 7.5 TensorFlow 设备与显存

GRACE 需要特别检查 TensorFlow：

- 是否默认占满全部 GPU 显存；
- 是否开启 memory growth；
- `CUDA_VISIBLE_DEVICES` 是否在 import TensorFlow 前设置；
- `--device cpu` 是否真的禁用 GPU；
- Queue 切换后端时是否残留进程；
- 实际设备是否可报告；
- PyTorch 与 TensorFlow 同环境时是否争抢显存。

即使技术上可以与 UMA/MACE 共存，也应评估独立虚拟环境是否更可靠。

---

## 7.6 委员会模型

如果官方 Calculator 支持多个模型并输出不确定度：

```text
energy_std
forces_std
stress_std
```

可以作为后续增强，但不是本轮必须项。

当前文档不应宣称尚未接入的委员会功能。

---

## 7.7 GRACE 最低验收

至少完成：

- 本地真实模型加载；
- 官方直调一致性；
- CPU/GPU 设备确认；
- SP；
- 固定晶胞 OPT；
- 10～100 步 MD；
- uniform/diverse 基准；
- TensorFlow 显存行为记录。

---

# 8. Batch 与 Job Queue 审查

仓库规模已经包含 UMA、MACE、DPA、GRACE 等多个可能依赖冲突的后端，因此需要区分两类批量任务。

## 8.1 Sweep Batch

定义：

```text
同一后端
同一模型
同一任务类型
相同参数
多个结构
```

调查当前 BatchRunner 是否：

- 只支持 SP/OPT；
- 声明支持 MD 但执行路径未实现；
- 模型是否只加载一次；
- 是否共享同一个 Calculator 给多个线程；
- GPU 并行是否可能 OOM；
- 失败后是否继续；
- 是否可恢复；
- 同名文件是否覆盖；
- 结果摘要是否稳定。

建议默认：

```text
单 GPU 串行
```

如果 CPU 并行，应优先使用独立进程和独立 Calculator，而不是多个线程共享一个带缓存状态的 ASE Calculator。

---

## 8.2 异构 Job Queue

定义：

```text
不同结构
不同任务
不同模型
不同后端
不同 Python 环境
任务依赖
夜间连续接力
```

调查当前是否已实现：

```text
manifest
validate
plan
run
status
resume
retry
cancel
depends_on
structure_from
restart_from
```

若尚未实现完整，不必一次性补完全部高级功能。

第一版最低要求：

- 文本文件定义任务顺序；
- 每个任务可指定后端、模型、结构和任务类型；
- 串行执行；
- 失败策略；
- 状态持久化；
- 中断后跳过已成功任务；
- 子任务使用对应虚拟环境中的 `mlipx`；
- 不使用 `shell=True`；
- 输出日志清晰。

---

## 8.3 多环境执行

调查 settings 是否能够定义：

```ini
[engine:uma]
executable = /path/to/.venv-uma/bin/mlipx

[engine:mace]
executable = /path/to/.venv-mace/bin/mlipx
```

Queue Controller 不应在单一进程中同时 import 所有重型后端。

推荐：

```text
轻量 controller
    ↓
subprocess 参数数组
    ↓
对应虚拟环境中的 mlipx
```

启动前检查：

- executable；
- 模型文件；
- 输入结构；
- GPU；
- 输出目录；
- 磁盘空间；
- 后端 import；
- 任务配置。

---

## 8.4 GPU 并发

单张 V100 16 GB 默认：

```text
max_gpu_jobs = 1
```

调查：

- 是否存在 GPU 排他锁；
- 两个 Queue 是否会同时启动；
- stale lock；
- 子进程崩溃后锁是否释放；
- CPU 任务是否被不必要阻塞；
- `CUDA_VISIBLE_DEVICES` 是否正确传给子进程。

不要求实现复杂的显存动态调度。

---

# 9. settings.ini 与配置系统审查

## 9.1 唯一默认值来源

调查默认值是否仍散落在：

```text
argparse
Engine
Config
Runner
API
TUI
```

目标是建立一个逻辑上的唯一默认值来源。

CLI 参数可使用：

```python
default=None
```

再由 ConfigResolver 决定最终值。

但不应为了形式统一而破坏现有向后兼容。

---

## 9.2 配置优先级

核对并测试：

```text
内置默认值
用户 settings
项目 settings
模型别名
profile
任务文件 / INCAR
CLI
```

建议优先级：

```text
CLI > Job/INCAR > Profile > Model alias
    > Project settings > User settings > Built-in defaults
```

如果当前项目采用不同但合理的规则，应：

- 文档化；
- 增加测试；
- 输出最终 resolved config。

---

## 9.3 配置路径

调查：

```text
--settings
MLIPX_SETTINGS
./settings.ini
~/.config/mlipx/settings.ini
内置默认
```

是否符合当前实现。

不建议唯一配置文件位于 Python 包安装目录，因为：

- 多虚拟环境产生多个副本；
- 重装可能覆盖；
- 后端环境读取结果不同。

---

## 9.4 严格校验

调查未知键处理。

推荐：

```text
拼写错误 → 报错并提示候选
未实现参数 → 明确报错
后端不支持参数 → 报错
```

避免：

```text
DEFAULT_DTPE 被忽略
随后静默使用 float64
```

允许提供：

```text
strict_config = false
```

作为兼容模式，但正式计算建议严格模式。

---

## 9.5 resolved config

每个任务输出：

```text
resolved_config.ini 或 JSON
```

至少记录：

- 最终值；
- 模型；
- 后端；
- device；
- dtype；
- task；
- seed；
- MD 参数；
- 输出间隔；
- 参数来源；
- 软件版本。

---

# 10. 输出和可追踪性

每次任务建议写出：

```text
run_manifest.json
run.log
resolved_config.ini
result.json
```

manifest 至少包含：

```text
mlipx 版本
git commit
Python
ASE
后端版本
PyTorch/TensorFlow/DeepMD 版本
CUDA
GPU
模型路径
模型哈希
输入结构哈希
最终配置
开始和结束时间
退出状态
```

对于动态下载的模型，可记录：

```text
模型 ID
缓存路径
实际文件哈希
```

---

## 10.1 轨迹输出要求

虽然 mlipx 不负责扩散分析，但轨迹必须适合外部分析：

- 原子顺序固定；
- 晶胞正确；
- PBC 正确；
- 保存间隔明确；
- 时间步明确；
- restart 后时间信息清晰；
- 不丢帧；
- 不因异常退出损坏整个轨迹；
- XDATCAR 元素顺序和坐标顺序一致。

`.traj` 可以作为内部真源，XDATCAR 作为兼容输出。

---

# 11. 测试补充清单

## 11.1 配置测试

- 优先级；
- profile；
- model alias；
- CLI 覆盖；
- 未知键；
- 未实现参数；
- 相对路径；
- 多 settings 文件；
- 环境变量；
- resolved config。

---

## 11.2 Calculator 参数链测试

对每个后端使用 mock Calculator，捕获最终构造参数。

验证：

```text
CLI → Resolver → Engine → Factory → Wrapper
```

避免只测试 argparse。

---

## 11.3 官方直调集成测试

分别为：

```text
UMA
MACE
DPA
GRACE
```

增加可选真实模型测试：

```text
mlipx wrapper
vs
官方 Calculator
```

比较能量、力和应力。

测试可通过环境变量提供模型路径，CI 无模型时跳过。

---

## 11.4 MD 测试

使用轻量假 Calculator：

- 正常势；
- NaN；
- Inf；
- 高但有限的力；
- 有约束；
- 零速度；
- 已有速度；
- checkpoint；
- restart；
- 中途终止；
- 长轨迹内存；
- NVE 连续与重启一致性。

有限大力测试应验证：

```text
产生受控警告，但不默认强制终止
```

---

## 11.5 Queue 测试

使用假命令或轻量子进程：

- 成功；
- 失败；
- retry；
- stop；
- continue；
- resume；
- stale RUNNING；
- GPU lock；
- 路径含空格；
- 输出已存在；
- 依赖失败；
- 循环依赖；
- manifest 修改。

---

# 12. 建议优先级

## P0：可能影响正确性或造成静默失效

1. 调查 Schema 中已接受但 Runner 未使用的参数。
2. 验证 MACE dtype/head 全链路。
3. 验证 DPA 和 GRACE 的设备选择是否真实生效。
4. 修复错误或误导性的模型格式文档。
5. 官方 Calculator 直调一致性测试。
6. 预弛豫失败后的实际结构语义。
7. restart 时速度是否被错误重置。
8. 多后端参数是否被静默忽略。

---

## P1：长程运行可靠性

1. 流式轨迹，避免 RAM 随帧数增长。
2. checkpoint/restart。
3. seed 和 RNG。
4. Queue 状态恢复。
5. 单 GPU 串行和设备锁。
6. TensorFlow/PyTorch 显存行为。
7. 任务和模型元数据追踪。

---

## P2：性能与易用性

1. GRACE uniform/diverse 基准。
2. MACE compile/cueq 等可选加速。
3. DPA 专属推理参数。
4. foundation model alias。
5. 委员会模型。
6. CPU 多进程 Sweep。
7. TUI 和文档同步。

---

# 13. 建议执行阶段

## 阶段 A：只读审查

禁止修改源码。

输出：

```text
AUDIT_CURRENT.md
```

内容：

- 当前提交；
- 测试现状；
- 每项状态；
- 确认问题；
- 已修复项目；
- 不确定项；
- 建议修改顺序。

---

## 阶段 B：P0 修复

只处理已经确认的 P0。

要求：

- 一个问题一个小提交；
- 增加回归测试；
- 不顺手重构无关模块；
- 不修改 TUI；
- 不新增扩散分析；
- 不把有限大力 warning 改成默认 abort。

---

## 阶段 C：真实后端验收

分别在对应环境运行：

```text
UMA
MACE
DPA
GRACE
```

输出：

```text
BACKEND_VALIDATION.md
```

记录：

- 环境；
- 模型；
- 设备；
- 结构；
- 官方直调；
- mlipx 结果；
- 差值；
- 显存；
- 时间；
- 结论。

---

## 阶段 D：长程与 Queue 验收

使用小体系先验证：

- 轨迹内存；
- restart；
- seed；
- Queue 接力；
- 失败恢复；
- GPU 锁。

再对 LGPS 做：

```text
SP
OPT
短程 MD
```

此阶段不要求在 mlipx 内部计算扩散或电导率。

---

# 14. Agent 执行提示词

可将以下内容和本文一起交给代码 Agent：

```text
请对当前 mlipx 工作区进行审查和必要改进。

重要要求：

1. 本文是审查方向，不是已确认 bug 清单。
2. 必须先检查当前提交和源码，不能直接按文档机械修改。
3. 对每项给出 CONFIRMED、ALREADY_FIXED、NOT_REPRODUCED、
   UNCERTAIN、DESIGN_CHOICE 或 OUT_OF_SCOPE。
4. 修改前必须给出证据、触发条件和测试。
5. 必须核对当前依赖版本的官方文档或源码。
6. 不要因为 pytest 全绿就认为功能正确。
7. 第一阶段只读，输出 AUDIT_CURRENT.md，不修改代码。
8. 不要新增 MSD、扩散系数、电导率或阿伦尼乌斯分析功能。
9. 不要把有限但较大的力默认改成强制终止；保留 warning 设计，
   除非能够证明当前实现存在另一个明确 bug。
10. “框框炸显存”指 OOM，不是原子飞出 PBC 晶胞。
11. 不要一次性重写项目。
12. 不要先修改 TUI。
13. 不要使用 shell=True。
14. 保持现有 CLI 向后兼容。
15. 每个确认修改都必须增加测试。
16. 真实后端结论必须来自官方 Calculator 直调对照或明确标记未验证。

第一阶段重点调查：
- UMA 和 MACE 是否已经稳定；
- MACE dtype/head 是否全链路生效；
- DPA 模型格式、设备和多头参数；
- GRACE mode、设备、TensorFlow 显存和模型加载；
- MD 中配置静默失效、速度初始化、restart 和轨迹内存；
- Sweep Batch 与异构 Job Queue；
- settings.ini 的唯一默认值和优先级；
- 当前测试是否真正覆盖数值行为。

最终报告不要复述大段源码。每个问题必须包含文件、函数、证据、
影响、复现和建议测试。
```

---

# 15. 最终验收标准

## UMA

- 官方直调一致；
- task、device、inference mode 生效；
- GPU 索引正确；
- SP/OPT/短 MD 可运行；
- 无静默参数失效。

## MACE

- dtype/head 生效；
- 官方直调一致；
- 真实 dtype 可追踪；
- 400 原子 LGPS 不发生异常显存持续增长；
- SP/OPT/短 MD 可运行。

## DPA

- 模型格式说明正确；
- 合法冻结模型可加载；
- 设备行为明确；
- 官方直调一致；
- SP/OPT/短 MD 通过。

## GRACE

- 模型来源明确；
- mode 行为经过调查；
- 设备与 TensorFlow 显存行为明确；
- 官方直调一致；
- SP/OPT/短 MD 通过。

## 公共 MD

- NaN/Inf 可靠终止；
- 有限大力按设计警告；
- seed/速度策略不静默失效；
- 长轨迹不持续积累完整帧到 RAM；
- checkpoint/restart 行为明确；
- 预弛豫语义准确；
- 输出轨迹可供外部分析。

## Batch/Queue

- 单 GPU 默认串行；
- 任务可连续接力；
- 失败状态明确；
- 中断后可恢复；
- 不重复成功任务；
- 跨环境通过子进程隔离；
- 配置和输出可追踪。

---

# 16. 本轮工作的核心目标

本轮不是继续扩大 mlipx 的功能范围，而是确认：

```text
用户设置的参数真正生效
不同后端得到与官方接口一致的结果
长时间计算不会因工程缺陷浪费资源
中断后可以合理恢复
批量任务能够在夜间连续接力
结果具有足够的可追踪性
```

mlipx 不需要承担所有后处理工作。它应首先成为一个：

> **可靠、可验证、可恢复、支持多后端和批量调度的 MLIP 计算驱动器。**
