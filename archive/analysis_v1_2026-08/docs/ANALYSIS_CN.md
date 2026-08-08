# MD 后处理：统一轨迹、基础分析与固态电解质

本模块从同一份规范化轨迹出发，分为三个层次。它不按 UMA、MACE、DPA、
GRACE 分叉；模型只负责产生轨迹，后处理的数据契约和算法与模型无关。

```text
MD 任务目录
├── raw/trajectory.traj       # mlipx 原始、无损、未折回轨迹（新任务）
├── raw/md.csv                # 每个存帧对应的热力学标量
├── vasp/XDATCAR              # VASP 语法兼容的交换格式
└── analysis/
    ├── validate/<hash>/      # 第 1 层：统一轨迹与质量检查
    ├── msd|rdf|.../<hash>/   # 第 2 层：轻依赖基础分析
    ├── transport_kinisi/...  # 第 3 层：kinisi 输运统计
    └── electrolyte_gemdat/...# 第 3 层：GEMDAT 位点/跃迁/贯通
```

旧版平铺目录中的 `trajectory.traj`、`md.csv`、`XDATCAR` 也能读取。读取优先级
是 `raw/trajectory.traj`、根目录 `trajectory.traj`、`vasp/XDATCAR`、根目录
`XDATCAR`。分析不会改写原始轨迹。

## 1. 安装

基础数值分析只需要 mlipx 已有的 NumPy 和 ASE。图片、kinisi 和 GEMDAT 是
可选依赖：

```bash
# 只增加 PNG 图片
uv sync --extra analysis

# kinisi：带相关误差的扩散与电导率
uv sync --extra transport

# GEMDAT：固态电解质位点、跃迁、密度和贯通网络
uv sync --extra electrolyte

# 全部后处理依赖
uv sync --extra analysis-all
```

后处理不需要加载 MLIP 模型，一般只需在一个安装了这些 extras 的环境中运行，
无需分别在四个模型环境中重复安装。

## 2. 最常用命令

默认运行轨迹验证、热力学汇总、RMSD/RMSF、RDF/CN、MSD/普通线性扩散拟合
和三维概率密度：

```bash
uv run mlipx analyze results/my-md --mobile Li --framework Ge,P,S
```

针对 LGPS 一类固态电解质：

```bash
uv run mlipx analyze results/my-md \
  --tasks validate msd rdf density \
  --mobile Li --framework Ge,P,S \
  --rdf-pair Li-Li --rdf-pair Li-S --rdf-pair Li-P \
  --fit-start-ps 40 --fit-stop-ps 200
```

kinisi 输运分析（`--fit-start-ps` 应来自 MSD 进入线性扩散区的判断）：

```bash
uv run mlipx analyze results/my-md \
  --tasks transport --mobile Li --temperature 800 --charge 1 \
  --fit-start-ps 40
```

GEMDAT 固态电解质分析：

```bash
uv run mlipx analyze results/my-md \
  --tasks electrolyte --mobile Li --temperature 800 \
  --gemdat-resolution 0.5 --percolation xyz
```

如果已有人工或文献确定的迁移位点，优先传入它们，避免自动分割密度峰带来的
歧义：

```bash
uv run mlipx analyze results/my-md \
  --tasks electrolyte --mobile Li --temperature 800 \
  --sites migration_sites.cif --site-radius 1.2 --minimal-residence 2
```

旧任务没有 `artifacts.json` / `resolved_config.json` 时，mlipx 会读取旧版
`run.log` 中明确打印的 `Time step` 与 `Save interval`。如果只有一个独立的
XDATCAR 或日志也不存在，必须显式提供相邻存帧的时间，而不是 MD 积分步长：

```bash
uv run mlipx analyze old-run --mobile Li --frame-interval-fs 100
```

注意，`run.log` 中 `Step ...` 行的数量不是轨迹帧数：热力学日志可以每一步
记录一次，而 `trajectory.traj` / XDATCAR 仍按 `Save interval` 保存完整构型。
后处理的时间轴以实际轨迹帧和轨迹保存间隔为准。

## 3. 三层分别做什么

### 第 1 层：规范化轨迹

`TrajectoryDataset` 统一保存：

- 原子顺序与元素；
- 每帧笛卡尔坐标、晶胞和 PBC；
- 轨迹中可用时的每帧能量、力和应力；
- 模拟步、时间轴和存帧间隔；
- 可用时的速度与 `md.csv` 热力学量；
- 输入解释、警告和来源元数据。

新 `trajectory.traj` 被视为未折回坐标。普通 XDATCAR 默认按连续最小镜像位移
解包；其前提是任一原子在两个存帧之间没有移动超过半个晶胞。可用
`--wrapped` 或 `--no-wrapped` 明确覆盖。被旧程序折回且存帧过稀的轨迹无法
唯一恢复 image 信息，这不是格式转换能修复的。

`validate` 检查帧数、原子顺序、有限数值、晶胞体积、时间轴和采样间隔。
大于 10 fs 的存帧间隔会提示 VACF/VDOS 与跃迁时间可能欠采样，但不代表 MSD
一定不可用。

旧任务缺少 `md.csv` 时，只要 `trajectory.traj` 的 SinglePointCalculator 数据
仍在，能量、力、应力、动能和温度会从轨迹帧恢复成统一热力学表。

### 第 2 层：mlipx 基础分析

| 任务 | 主要输出 | 实现和注意事项 |
|---|---|---|
| `thermo` | `thermodynamics.csv`, `summary.json` | 温度、能量、体积、应力统计；NVT 总能量不要求守恒 |
| `rmsd` | `rmsd.csv`, `rmsf.csv` | PBC 连续坐标；可用框架质心漂移校正 |
| `rdf` | `rdf_A-B.csv` | 部分 RDF、直接累计配位数；`r_max` 不得超过最小镜像半径 |
| `msd` | `msd.csv`, `per_particle_msd.npz` | FFT 时间原点平均，给出总量、xyz 分量和逐离子 MSD |
| `density` | `density.npz` | 周期分数坐标中的三维占据概率 |
| `vacf` | `vacf.csv`, `vdos.csv` | 需要轨迹中的速度；高频谱要求足够密的存帧 |

基础 `msd` 同时给普通最小二乘的 Einstein 斜率和扩散系数，适合快速检查，
但它没有正确处理不同 lag-time MSD 点之间的相关性。用于论文定量误差时应运行
`transport`，由 kinisi 处理协方差和后验分布。

框架漂移校正的含义是从所有原子坐标中减去指定框架原子的平均位移。例如
LGPS 可用 `--framework Ge,P,S`。它不会做局部非刚性配准；使用前应确认框架
在所选时间段内保持固态。

### 第 3 层：成熟库适配器

mlipx 没有复制 kinisi 或 GEMDAT 的算法。适配器负责把规范化轨迹、单位、物种
选择和结果文件契约接到它们的原生 API，并记录实际包版本。

`transport` 调用 kinisi：

- tracer MSD 与扩散系数的协方差感知贝叶斯拟合；
- 扩散系数后验样本与 95% credible interval；
- 给定温度和离子电荷后的 collective charge displacement 与电导率后验。

`electrolyte` 调用 GEMDAT：

- 移动离子三维密度和自由能体数据；
- 自动密度峰与迁移位点，或读取已知位点结构；
- 位点占据、停留时间、跃迁事件、jump 矩阵与 jump diffusivity；
- collective/solo jump 分类；
- x/y/z 方向的最优贯通路径和自由能势垒。

这里的 **percolation（贯通）** 是低自由能迁移网络是否能跨越周期晶胞的某个
方向；它回答通道是否在宏观上连通，而不仅是相邻两个位点之间能否跳跃。
自动位点的数目和连接关系对网格分辨率、轨迹长度、温度和
`background_level` 敏感，必须做参数稳定性检查。

## 4. 输出、缓存与 provenance

每项分析写到 `analysis/<task>/<12位参数哈希>/`。常见格式是 CSV（表格）、
NPZ（高维数组）、PNG（可选图）、CIF（位点）和 VASP volumetric data
（`density.vasp`、`free_energy.vasp`）。

每个目录都有 `metadata.json`，其中的 **provenance（来源记录）** 包括：

- 输入轨迹绝对路径与 SHA-256；
- 完整分析参数和对轨迹的时间/PBC 解释；
- mlipx、ASE、NumPy 及 kinisi/GEMDAT 等实际版本；
- 输出清单、摘要与所有告警。

输入和参数完全相同时直接复用缓存；`--force` 强制重算。不同拟合区间、RDF
pair、网格或轨迹时间解释会进入不同的哈希目录，不覆盖旧结果。

## 5. Python API

```python
from mlipx.analysis import TrajectoryDataset, analyze_run

dataset = TrajectoryDataset.load("results/my-md")
print(dataset.validation_report())

result = analyze_run(
    "results/my-md",
    tasks=["msd", "rdf", "density"],
    mobile="Li",
    framework="Ge,P,S",
    rdf_pairs=[("Li", "Li"), ("Li", "S")],
    plots=False,
)
```

确定性多温度 Arrhenius 拟合可直接使用
`mlipx.analysis.core.arrhenius_fit(temperatures_K, diffusivities_m2_s)`；需要后验
传播时可使用 `mlipx.analysis.transport.kinisi_arrhenius(...)`。至少需要两个
温度，实际研究通常应使用更多温度和独立重复轨迹。

## 6. 科学使用检查表

1. 先看 `validate/report.json` 和 `metadata.json` 的警告。
2. 明确积分步长与存帧间隔不是同一概念。
3. 对 MSD 同时查看 xyz 分量、逐离子分布和拟合区间，不只报告一个斜率。
4. 比较有无框架漂移校正，并报告选择的框架物种。
5. RDF 配位数的截断半径应由第一谷等物理依据确定。
6. 自动迁移位点要对分辨率和阈值做稳定性测试，最好与晶体学已知位点对照。
7. 电导率必须报告使用的离子价态、温度和是 tracer/Nernst–Einstein 还是
   collective charge-displacement 定义。
8. 单条短轨迹不能替代独立重复；贝叶斯后验也不能消除采样不足。

上游接口参考：[kinisi Analyze](https://kinisi.readthedocs.io/en/latest/analyze.html)、
[GEMDAT Trajectory API](https://gemdat.readthedocs.io/en/latest/api/gemdat_trajectory/)、
[GEMDAT simulation metrics](https://gemdat.readthedocs.io/en/latest/api/gemdat_simulation_metrics/)。
