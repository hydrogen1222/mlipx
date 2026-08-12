# mlipx Analysis v2 最后一轮收尾修改计划
## 给 Pi Agent / GLM 5.2 的执行说明

> **任务性质：小范围 Analysis v2 hardening，不是重构。**
>
> 这轮修改完成后，目标是把 `transport` 这一条分析链真正收尾，然后停止继续扩展源码，进入 LGPS 8 条轨迹的正式批处理。
>
> 请先完整阅读本文件，再开始修改。不要边读边大改。

---

## 0. 总目标

当前 `mlipx` 的 MSD → kinisi transport 主线已经基本可用了，而且实际 400 ps / 40001 帧 LGPS 数据已经验证：

- 原先 kinisi dense lag grid 会极慢；
- 新增 `--lag-step-ps / --lag-stop-ps` 后，`1 ps / 200 ps` 约 3 分钟完成；
- `2 ps / 200 ps` 约 3 分钟完成；
- UMA 700 K 的两个 lag grid 给出的 posterior 几乎一致：
  - 1 ps：`D_mean = 3.442900243e-9 m²/s`
  - 2 ps：`D_mean = 3.440443641e-9 m²/s`
- 95% posterior credible interval 也基本重合。

因此 **不要再重写 kinisi adapter 的统计方法，也不要增加自动拟合、自动 Arrhenius、GPU covariance 等大功能。**

这轮只解决四类剩余问题：

1. **kinisi 对 wrapped / unwrapped 坐标的后端语义要严谨，不能丢失 exact image information 后还假装是 exact。**
2. **TUI 要补齐 transport 现在已经必需的 lag-grid / temperature / source-semantics 参数。**
3. **Nernst–Einstein 电导率要传播 kinisi 的 posterior uncertainty，而不是只输出一个 mean。**
4. **transport 跑完要产出人能直接看的 summary + plot + CLI 摘要，不应该再要求用户手工 `jq`。**

修改完成后：

- bump `transport` output revision；
- 补测试；
- 补文档；
- 全部检查通过；
- commit；
- push；
- **STOP，不要继续替用户跑 LGPS 数据。**

---

# 1. 当前仓库快照中已经确认的事实

以下内容基于用户提供的当前 `mlipx-main_2.zip`。

## 1.1 sparse kinisi lag grid 已经存在，不要重做

当前：

```text
mlipx/mlipx/analysis/transport.py
```

已经有：

```python
DEFAULT_MAX_NATIVE_KINISI_LAG_POINTS = 1000
```

以及：

```python
_resolve_kinisi_lag_grid(...)
```

支持：

```text
--lag-step-ps
--lag-stop-ps
```

且 custom grid 会通过：

```python
common["dt"] = sc.array(...)
```

真正传给：

```python
DiffusionAnalyzer.from_ase(...)
```

`ConductivityAnalyzer.from_ase(...)` 也复用同一个 `common`，因此 tracer / collective 使用相同 `dt`。

这些设计已经经过测试，**不要推翻。**

---

## 1.2 dense-grid fail-fast 已经存在

如果用户不提供 custom lag grid，且 kinisi 原生 lag 数超过 1000，目前会在真正进入 kinisi 之前 fail fast。

这个行为是正确的。

不要改成：

- 自动 trajectory downsampling；
- 自动偷偷选 1 ps；
- 自动偷偷选 200 ps；
- 自动绕过 guard。

科学参数仍然必须显式。

---

## 1.3 transport output revision 当前是 1

当前：

```python
_TASK_OUTPUT_REVISIONS = {"msd": 5, "transport": 1}
```

本次输出 schema 会发生变化，因此最终必须更新为：

```python
_TASK_OUTPUT_REVISIONS = {"msd": 5, "transport": 2}
```

并更新对应测试。

---

## 1.4 TUI 目前没有 lag-grid 控件

当前：

```text
mlipx/mlipx/tui/analysis_screen.py
```

Transport 页面只有：

- Mobile Species
- Drift Correction
- Axes
- Ionic Charge
- Fit Start
- kinisi Random Seed

但是 CLI 已经要求长高频轨迹显式提供：

```text
--lag-step-ps
--lag-stop-ps
```

否则 dense-grid guard 会直接拒绝。

也就是说，**TUI 的 transport 路径目前对真实 400 ps / 10 fs 轨迹实际上是不完整的。**

---

## 1.5 Nernst–Einstein 目前只使用 posterior mean

当前：

```python
diffusion_m2_s = _sample_summary(analyzer.D, target_unit="m^2/s")
```

已经有：

- mean
- std
- median
- 95% credible interval
- posterior sample count

但随后：

```python
nernst_einstein = nernst_einstein_tracer_conductivity(
    ...,
    tracer_diffusion_m2_s=diffusion_m2_s["mean"],
    ...
)
```

只拿了 `D` posterior 的 mean。

因此 `sigma_NE_tracer` 目前只有点估计，丢掉了已经花大量计算得到的 posterior uncertainty。

这是本次必须补上的科学输出缺口。

---

# 2. 第一优先级：修正 kinisi 的 coordinate semantics

这是本次最重要的 correctness 修改。

---

## 2.1 当前潜在问题

当前 mlipx adapter：

```python
def _kinisi_frames_and_indices(...):
    ...
    atoms = Atoms(...)
    atoms.wrap()
    frames.append(atoms)
```

会把送给 kinisi 的 ASE frames 显式 wrap。

而 kinisi 当前官方 `ASEParser` 中：

```python
scaled_positions = struct.get_scaled_positions()
```

ASE `get_scaled_positions()` 默认也是 wrapped fractional coordinates。

随后 kinisi `Parser` 会根据这些周期坐标自己重建 displacement。

上游官方源码：

```text
https://raw.githubusercontent.com/kinisi-dev/kinisi/main/kinisi/ase.py
https://raw.githubusercontent.com/kinisi-dev/kinisi/main/kinisi/parser.py
```

其中 `Parser` 对正交 / 非正交晶胞均使用自己的 periodic displacement reconstruction。

因此：

> **即使 mlipx 的输入 trajectory 本来是 exact unwrapped Cartesian coordinates，经过当前 ASE adapter 后，kinisi 并没有直接消费这些 exact unwrapped image histories，而是重新从周期坐标重建 displacement。**

这本身未必导致错误——如果保存帧足够密，重建结果可能与 exact trajectory 完全一致。

真正的问题是：

> **mlipx 目前对 `positions_convention="unwrapped"` 直接认为 exact images 已知，从而跳过 wrapped-source safety refusal；但后端其实又重新做了 periodic reconstruction。**

这个语义必须补严谨。

---

# 3. 推荐的 coordinate-semantics 修复方案

## 3.1 不要简单禁止所有 unwrapped trajectory

注意：

当前 mlipx 自己的 canonical：

```text
raw/trajectory.traj
```

在：

```text
mlipx/mlipx/runners/md.py
```

的 artifact contract 中明确标记：

```json
"positions_convention": "unwrapped"
```

而且：

```text
raw/trajectory.traj
```

正是 RUN directory 自动优先选择的 canonical trajectory。

因此绝对不要做：

```python
if positions_convention == "unwrapped":
    raise UnsupportedAnalysisError(...)
```

否则会把 mlipx 自己最标准的 RUN-directory transport 路径直接废掉。

---

## 3.2 正确思路：对 exact unwrapped source 做“backend reconstructability check”

对于：

```text
positions_convention == "unwrapped"
```

我们已经知道真实的连续 Cartesian trajectory。

因此可以在真正调用 kinisi 之前检查：

> 如果把这条轨迹交给周期最小镜像重建，能不能逐帧得到和 exact unwrapped trajectory 相同的 displacement increment？

如果能：

```text
exact unwrapped trajectory
        ↓
periodic reconstruction
        ↓
same consecutive displacement
```

则允许 transport。

如果不能：

```text
exact displacement = +6 Å
periodic MIC       = -4 Å
```

说明保存间隔中发生了 image-information loss：

> kinisi ASE backend 会重建错误 displacement。

此时必须 fail closed。

---

## 3.3 建议新增 helper

建议在：

```text
mlipx/mlipx/analysis/transport.py
```

增加类似：

```python
def _kinisi_position_semantics(
    dataset: TrajectoryDataset,
) -> dict[str, Any]:
    ...
```

或者拆成：

```python
def _validate_kinisi_periodic_reconstruction(
    dataset: TrajectoryDataset,
) -> dict[str, Any]:
    ...
```

要求：

### wrapped source

保持当前逻辑：

```python
_, unwrap_diagnostics = unwrap_positions(view)
```

如果：

```text
unwrap_safety_ratio > 0.8
```

继续拒绝 publication transport。

wrapped source 没有 exact image counter，因此这里仍只能是 heuristic safety check。

### unwrapped source

不要因为：

```text
exact_image_information_available=True
```

就直接认为 kinisi 后端也 exact。

应该：

1. 取 exact consecutive Cartesian displacement：

```python
exact_steps = np.diff(dataset.positions, axis=0)
```

2. 使用固定晶胞 + PBC 得到 periodic minimum-image displacement。

推荐优先使用 ASE 自己经过验证的 periodic geometry helper，例如：

```python
from ase.geometry import find_mic
```

而不是自己再写一套 triclinic MIC。

3. 比较：

```python
exact_steps
```

和：

```python
mic_steps
```

是否在合理数值容差内一致。

4. 如果不一致：

```python
raise UnsupportedAnalysisError(...)
```

错误信息必须明确：

```text
The source contains exact unwrapped image information, but kinisi's ASE
backend reconstructs periodic displacements from wrapped/scaled coordinates.
At least one saved-frame displacement is not equal to its minimum-image
reconstruction, so exact image history would be lost.

Use a denser saved trajectory or a future exact-displacement transport backend.
Native mlipx MSD can still use the exact unwrapped coordinates.
```

5. 如果一致，允许 transport，并记录：

```json
"kinisi_position_semantics": {
  "source_positions_convention": "unwrapped",
  "backend_input": "periodic ASE frames",
  "backend_reconstruction": "kinisi periodic displacement reconstruction",
  "exact_unwrapped_preserved_directly": false,
  "exact_unwrapped_reconstruction_equivalent": true,
  "checked_saved_intervals": ...,
  "maximum_exact_vs_mic_difference_A": ...
}
```

---

## 3.4 对 wrapped source 也记录 backend semantics

输出不要只写：

```text
unwrap_diagnostics
```

建议额外加一个明确块：

```json
"kinisi_position_semantics": {
  "source_positions_convention": "wrapped",
  "backend_input": "wrapped/scaled periodic coordinates",
  "backend_reconstruction": "kinisi periodic displacement reconstruction",
  "exact_unwrapped_preserved_directly": false,
  "exact_unwrapped_reconstruction_equivalent": null,
  "wrapped_source_safety": "comfortably_safe"
}
```

这样以后查看 `results.json`，不用读源码也知道真正发生了什么。

---

## 3.5 `_kinisi_frames_and_indices()` 中的 `atoms.wrap()`

请先验证 kinisi 2.1.0 / 当前官方源码真实行为。

由于 kinisi `ASEParser` 自己调用：

```python
struct.get_scaled_positions()
```

并默认 wrap，因此即使去掉：

```python
atoms.wrap()
```

也不会让 exact unwrapped coordinates 穿透进去。

所以：

> **不要把“删掉 atoms.wrap()”误认为解决了 exact-unwrapped 问题。**

可以：

- 保留 `atoms.wrap()`，但把语义写清楚；
- 或删除这个显式 wrap 作为冗余清理；

两种都可以。

关键是：

> **必须有 backend reconstructability check + 明确 provenance。**

---

# 4. coordinate-semantics 必须增加的测试

至少增加：

## 4.1 safe unwrapped source 被接受

构造：

- fixed 10 Å cell；
- exact unwrapped positions；
- 每帧 Li 只移动 0.1 Å；
- periodic MIC 与 exact displacement 完全一致。

预期：

```text
transport accepted
exact_unwrapped_reconstruction_equivalent == true
```

---

## 4.2 exact unwrapped source 中存在 hidden image crossing 时拒绝

例如一维简化：

```text
cell = 10 Å

frame 0: x = 1 Å
frame 1: x = 7 Å
```

exact displacement：

```text
+6 Å
```

periodic minimum image：

```text
-4 Å
```

这两者明显不同。

预期：

```text
UnsupportedAnalysisError
```

而且：

```text
DiffusionAnalyzer.from_ase
```

根本不应该被调用。

---

## 4.3 wrapped source 原有 safety guard 继续工作

不要破坏：

```text
unwrap_safety_ratio > 0.8
```

的现有 fail-closed 行为。

---

## 4.4 safe unwrapped canonical RUN 仍然能跑 transport

现有 synthetic / smoke tests 中很多 trajectory 使用：

```text
positions_convention="unwrapped"
```

不要简单改成 wrapped 来逃避问题。

应该让它们通过新的 reconstructability check。

---

# 5. 第二优先级：传播 Nernst–Einstein posterior uncertainty

当前公式：

$$
\sigma_{\mathrm{NE}}
=
\frac{n(z e)^2}{k_B T}D_{\mathrm{tracer}}.
$$

对于一次固定分析：

- $n$ 固定；
- $z$ 固定；
- $T$ 当前视为固定输入；
- 因此 $\sigma_{\rm NE}$ 对 $D$ 是严格线性变换。

所以没必要只用：

```text
D posterior mean
```

去计算一个 conductivity mean。

---

## 5.1 正确实现

从：

```python
analyzer.D
```

获取全部 posterior samples。

假设：

```python
D_samples_m2_s
```

则：

```python
factor = n * (z * e)**2 / (k_B * T)
sigma_samples_S_m = factor * D_samples_m2_s
```

再分别生成：

```text
S/m
S/cm
mS/cm
```

的 posterior summary：

- mean
- std
- median
- credible_interval_95
- posterior_samples

---

## 5.2 建议重构 sample-summary helper

当前：

```python
_sample_summary(variable, *, target_unit)
```

只接受 scipp variable。

可以保留它，同时新增一个纯 NumPy helper：

```python
def _numeric_sample_summary(
    values: np.ndarray,
    *,
    unit: str,
) -> dict[str, Any]:
    ...
```

然后 `_sample_summary()` 内部 convert 后调用它。

这样：

```text
D posterior
sigma posterior
collective posterior
```

都可以复用同一份 summary logic。

不要复制三遍 quantile/std/median 代码。

---

## 5.3 output schema 建议

为了 backward compatibility，保留已有：

```json
"nernst_einstein": {
  "sigma_NE_tracer_S_m": ...,
  "sigma_NE_tracer_S_cm": ...,
  "sigma_NE_tracer_mS_cm": ...
}
```

这些 scalar 字段继续表示 posterior mean。

新增：

```json
"nernst_einstein": {
  "definition": "...",
  "ionic_charge_e": 1.0,

  "sigma_NE_tracer_S_m": 12.34,
  "sigma_NE_tracer_S_cm": 0.1234,
  "sigma_NE_tracer_mS_cm": 123.4,

  "sigma_NE_tracer_posterior_S_m": {
    "unit": "S/m",
    "mean": ...,
    "std": ...,
    "median": ...,
    "credible_interval_95": [..., ...],
    "posterior_samples": 3200
  },

  "sigma_NE_tracer_posterior_S_cm": {...},
  "sigma_NE_tracer_posterior_mS_cm": {...},

  "uncertainty_semantics":
    "Linear propagation of the kinisi tracer-D posterior; n, z, V and T are treated as fixed."
}
```

注意：

> 不要暗示这是整个物理模型的总 uncertainty。

它只传播：

```text
kinisi tracer-D posterior
```

没有包含：

- model uncertainty；
- finite-size uncertainty；
- run-to-run replica uncertainty；
- temperature uncertainty；
- volume uncertainty；
- Nernst–Einstein approximation error；
- ion-ion correlation correction。

文档里必须说清楚。

---

# 6. Nernst–Einstein posterior 测试

必须增加：

## 6.1 线性缩放精确测试

人工设置：

```text
D samples = [1, 2, 3, 4] × 10^-10 m²/s
```

验证：

```text
sigma samples = factor × D samples
```

因此：

- mean scaling；
- std scaling；
- median scaling；
- CI scaling；

都必须严格符合线性关系。

---

## 6.2 charge-squared 逻辑不变

现有：

```text
z = 2
```

应仍然得到：

```text
sigma = 4 × monovalent
```

不要破坏已有测试。

---

## 6.3 backward-compatible scalar 字段等于 posterior mean

断言：

```python
result["nernst_einstein"]["sigma_NE_tracer_mS_cm"]
==
result["nernst_einstein"]["sigma_NE_tracer_posterior_mS_cm"]["mean"]
```

---

# 7. 第三优先级：把 TUI transport 补完整

当前 TUI 已经无法完整表达 CLI transport 参数。

这不是新增高级功能，而是修复 parity。

---

## 7.1 必须新增的 Transport 控件

在：

```text
mlipx/mlipx/tui/analysis_screen.py
```

Transport 模式新增：

### Lag Step (ps)

```text
analysis-lag-step-input
```

默认建议：

```text
blank
```

不要写死 1 或 2 ps。

### Lag Stop (ps)

```text
analysis-lag-stop-input
```

默认：

```text
blank
```

同样不要写死。

### Temperature (K, optional)

```text
analysis-temperature-input
```

空值表示：

> 从 RUN metadata / production temperature 获取。

### Collective Conductivity

```text
analysis-collective-switch
```

默认：

```text
False
```

---

# 8. TUI 的 imported trajectory source overrides

当前 TUI 标题写：

```text
Run Directory or Trajectory
```

但 external trajectory 如果缺：

- positions convention；
- frame interval；

用户没有地方填写。

因此建议增加一个很小的“Trajectory Overrides”区域。

---

## 8.1 Positions Convention

Select：

```text
Auto / metadata
wrapped
unwrapped
unknown
```

推荐内部值：

```text
auto
wrapped
unwrapped
unknown
```

如果：

```text
auto
```

则 **不要把 `positions_convention` 放进 parameters**。

这样 RUN metadata 可正常生效。

---

## 8.2 Frame Interval (fs)

Input：

```text
analysis-frame-interval-input
```

blank：

```text
不传参数
```

有值：

```python
parameters["frame_interval_fs"] = float(...)
```

---

## 8.3 这些 source overrides 是否只在 transport 显示？

两种设计都可接受：

### 推荐

作为 analysis 的通用 “Imported trajectory overrides” 小区域。

因为 CLI `_analysis_source_options()` 对多个 Analysis v2 task 都存在。

但不要因此大改整个 TUI layout。

如果为了最小改动，只在：

```text
msd
transport
validate
```

等需要明确 time/PBC semantics 的任务中显示，也可以。

---

# 9. TUI 参数校验

Transport 中必须：

### charge

仍然 required。

### fit_start

仍然 required。

### lag step / lag stop

两者：

```text
要么同时为空
要么同时有值
```

如果只填一个：

```text
Transport lag step and lag stop must be provided together
```

### temperature

optional。

### collective

boolean。

### frame interval

如填写，必须：

```text
> 0
```

### positions convention

auto 时 omit。

---

# 10. TUI 测试必须增加

扩展：

```text
tests/mlipx/mlipx/test_tui.py
```

至少验证：

## 10.1 transport progressive disclosure

选择 transport 后：

```text
lag step visible
lag stop visible
temperature visible
collective visible
```

切换到 RDF / electrolyte 后隐藏。

---

## 10.2 `_parameters("transport")`

填：

```text
charge = 1
fit = 40
lag step = 2
lag stop = 200
temperature = 700
collective = true
positions convention = wrapped
frame interval = 10
```

最终参数必须正确得到：

```python
{
    "mobile_species": "Li",
    "drift_reference": ...,
    "dimensions": "xyz",
    "ionic_charge_e": 1.0,
    "fit_start_ps": 40.0,
    "lag_step_ps": 2.0,
    "lag_stop_ps": 200.0,
    "temperature_K": 700.0,
    "collective_conductivity": True,
    "positions_convention": "wrapped",
    "frame_interval_fs": 10.0,
    "random_seed": 0,
}
```

---

## 10.3 auto / blank 不应该制造参数

如果：

```text
positions convention = auto
frame interval blank
temperature blank
```

则这几个 key 不应该出现在 request parameters 里。

---

# 11. 第四优先级：transport 结果必须直接可读

用户目前跑完只能看到：

```text
Analysis success: ...
Output: ...
```

然后还得自己：

```bash
jq ...
```

这对一个科研分析 CLI 来说太难用了。

本次应增加：

1. `transport_summary.csv`
2. `transport_msd.png`
3. `transport_msd.svg`
4. CLI compact summary

---

# 12. `transport_summary.csv`

建议一行一个分析结果。

字段至少：

```text
mobile_species
dimensions
temperature_K

fit_start_ps
fit_stop_ps

lag_grid_mode
lag_step_ps
lag_stop_ps
n_lag_points_total
n_lag_points_in_fit

D_mean_m2_s
D_std_m2_s
D_median_m2_s
D_ci95_low_m2_s
D_ci95_high_m2_s

sigma_NE_mean_mS_cm
sigma_NE_std_mS_cm
sigma_NE_median_mS_cm
sigma_NE_ci95_low_mS_cm
sigma_NE_ci95_high_mS_cm

kinisi_version
random_seed

positions_convention
kinisi_backend_reconstruction
```

如果：

```text
collective conductivity
```

存在，可以追加 collective 字段。

不要为“统一列数”编出 fake 0。

没有 collective 时留空即可。

---

# 13. transport plot

在：

```text
mlipx/mlipx/analysis/plots.py
```

新增：

```python
plot_transport(...)
```

图只做一件事：

> **让用户一眼看清 kinisi MSD 数据和真正用于 diffusion regression 的 fit window。**

---

## 13.1 图的内容

横轴：

```text
Lag time (ps)
```

纵轴：

```text
MSD (Å²)
```

数据：

```text
kinisi_msd_A2
```

误差：

```text
sqrt(kinisi_msd_variance_A4)
```

可使用：

```python
errorbar(...)
```

或者轻量 error band。

---

## 13.2 明确画出 fit window

从：

```python
result["tracer_diffusion"]["fit_start_ps"]
result["tracer_diffusion"]["fit_stop_ps"]
```

画：

```text
40 ps -------------------------------- 200 ps
       <------ regression window ------>
```

推荐：

```python
axis.axvspan(...)
```

或者两条竖线。

图例中必须清楚写：

```text
kinisi MSD
diffusion fit window
```

不要画一个假的 OLS fit line。

因为 transport 用的是 covariance-aware Bayesian regression，不是普通 OLS。

除非能够从 kinisi 正确拿到 posterior linear-model prediction，否则本轮不画 fit line。

---

# 14. runner 产物

修改：

```text
mlipx/mlipx/analysis/runner.py
```

当前 transport 只写：

```text
kinisi_arrays.npz
```

改为同时：

```text
kinisi_arrays.npz
transport_summary.csv
transport_msd.png
transport_msd.svg
```

并全部加入：

```json
"artifacts"
```

---

# 15. CLI transport summary

在：

```text
mlipx/mlipx/cli.py
```

`cmd_analyze()` 中，在通用：

```text
Analysis success
Output
```

之后，如果：

```python
args.analysis_task == "transport"
```

打印紧凑摘要。

建议类似：

```text
Tracer diffusion:
  D = 3.44044e-09 m^2/s
  posterior SD = 2.08790e-10 m^2/s
  95% credible interval = [3.02726e-09, 3.84527e-09] m^2/s

Fit window:
  40 – 200 ps

Kinisi lag grid:
  custom, nominal step 2 ps, 100 total points

Nernst-Einstein tracer conductivity:
  sigma_NE = ... mS/cm
  95% credible interval = [..., ...] mS/cm
```

如果 reused：

```text
Existing completed result reused...
```

仍然照常显示 summary。

---

## 15.1 术语要求

必须打印：

```text
95% credible interval
```

不要打印：

```text
95% confidence interval
```

这是 Bayesian posterior。

---

# 16. 顺手修一个 metadata 小问题：`actual_step_ps`

当前 custom grid 允许：

```text
lag-step = 3 ps
fit-start = 40 ps
```

regular grid 本来是：

```text
3, 6, 9, ..., 198
```

程序会把：

```text
40
```

额外插进去。

因此真正 grid 变成：

```text
..., 39, 40, 42, ...
```

这时已经不再严格等间距。

但当前 metadata 仍然：

```json
"actual_step_ps": 3
```

这个名字不严谨。

---

## 16.1 推荐处理

保持 backward compatibility，但语义改清楚：

```json
"requested_step_ps": 3.0,
"nominal_step_ps": 3.0,
"actual_step_ps": null,
"fit_start_inserted": true,
"is_uniform_grid": false
```

对于正常：

```text
step=2
fit-start=40
```

则：

```json
"requested_step_ps": 2.0,
"nominal_step_ps": 2.0,
"actual_step_ps": 2.0,
"fit_start_inserted": false,
"is_uniform_grid": true
```

如果不想新增太多字段，最低限度也要：

```text
actual_step_ps=None
fit_start_inserted=True
```

避免误导。

---

# 17. 不要改变的东西

本轮严禁扩大 scope。

不要改：

- MSD FFT 算法；
- direct MSD；
- alpha 算法；
- OLS 公式；
- Analysis v2 dataset contract；
- phase/equilibration semantics；
- NPT/NVT/NVE engine；
- thermostat；
- calculator backends；
- UMA；
- MACE；
- DPA；
- GRACE；
- GEMDAT；
- VACF；
- spectrum；
- Arrhenius；
- queue；
- MD writer 的核心逻辑；
- kinisi 源码；
- MCMC 数学；
- posterior model；
- conductivity theory。

---

# 18. 本轮明确不要增加的新功能

不要增加：

```text
auto-fit-window
auto-lag-step
auto-lag-stop
log-spaced lag grid
adaptive lag grid
automatic replicas
automatic Haven ratio
automatic Arrhenius
automatic model ranking
GPU covariance
CuPy
JAX
Dask
multiprocessing rewrite
native Bayesian estimator
kinisi replacement
GEMDAT fallback
GUI plot viewer
```

如果你发现这些方向“很诱人”，记录为 future idea 即可，不要实现。

---

# 19. 文档更新

至少检查并更新：

```text
mlipx/docs/TRANSPORT.md
mlipx/docs/README_CN.md
```

以及英文对应 Analysis 文档（如果当前仓库有对应章节）。

---

## 19.1 文档必须解释 RUN directory 优先

对 mlipx 自己产生的结果，推荐用户优先：

```bash
mlipx analyze RUN transport ...
```

而不是直接：

```bash
mlipx analyze RUN/vasp/XDATCAR transport ...
```

原因不是“XDATCAR 不能用”，而是 RUN directory 自动携带：

- trajectory metadata；
- timestep；
- save stride；
- frame interval；
- positions convention；
- production phase；
- temperature 等。

这样用户不用重复手工声明 source semantics。

---

## 19.2 当前 LGPS 推荐示例

文档可以使用：

```bash
mlipx analyze RUN transport \
  --mobile Li \
  --charge 1 \
  --drift-reference nonmobile \
  --fit-start-ps 40 \
  --lag-step-ps 2 \
  --lag-stop-ps 200 \
  --random-seed 0
```

并明确：

> `2 ps` 不是全局科学默认值，只是当前 LGPS 数据经过 1 ps vs 2 ps sensitivity check 后采用的工作参数。

不要把：

```text
2 ps
40–200 ps
```

写成所有材料的默认。

---

## 19.3 direct external trajectory 示例

如果使用无完整 metadata 的 external XDATCAR：

```bash
mlipx analyze XDATCAR transport \
  --mobile Li \
  --charge 1 \
  --drift-reference nonmobile \
  --positions-convention ... \
  --frame-interval-fs ... \
  --temperature-K ... \
  --fit-start-ps ... \
  --lag-step-ps ... \
  --lag-stop-ps ...
```

必须强调：

> `positions-convention` 应反映文件真实语义，不能为了“通过 validation”随便填 wrapped/unwrapped。

---

# 20. 现有测试基础

当前仓库已经有较完整 transport tests：

```text
tests/mlipx/mlipx/analysis/test_runner_kinisi.py
tests/mlipx/mlipx/analysis/test_transport_arrhenius_spectral.py
tests/mlipx/mlipx/test_cli.py
tests/mlipx/mlipx/test_tui.py
```

请在现有测试体系上增量修改，不要另外发明一套测试框架。

---

# 21. 必须更新的现有断言

当前：

```python
assert request["task_output_revision"] == 1
```

改为：

```python
assert request["task_output_revision"] == 2
```

---

# 22. runner artifact tests

新增/更新测试，确保 transport 成功后 artifacts 至少包含：

```text
kinisi_arrays.npz
transport_summary.csv
transport_msd.png
transport_msd.svg
```

并确认：

```text
results.json
provenance.json
request.json
```

仍然正常。

---

# 23. CLI summary test

用 fake dispatch / monkeypatch 避免真实跑 kinisi。

调用：

```text
cmd_analyze / main
```

捕获 stdout。

至少 assert：

```text
Tracer diffusion
95% credible interval
Fit window
Kinisi lag grid
Nernst-Einstein
```

---

# 24. transport plot test

不要做像素级 regression test。

只需要：

- synthetic result；
- 调 `plot_transport()`；
- 检查 PNG / SVG 文件存在且非空。

不要引入 image snapshot testing。

---

# 25. kinisi 2.1.0 compatibility smoke test

用户实际环境是：

```text
kinisi 2.1.0
```

修改完成后必须在包含 kinisi 2.1.0 的环境中运行至少一次真实 smoke test。

重点确认：

```text
DiffusionAnalyzer.from_ase
ConductivityAnalyzer.from_ase
dt
```

仍然兼容。

不要只依赖 fake analyzer tests。

---

# 26. 科学 regression test

修改后，至少确保 existing real/synthetic kinisi adapter test 仍能得到：

- 合法 lag grid；
- 合法 posterior；
- fit stop 正确；
- tracer / collective share dt。

本轮代码修改不应改变：

```text
D posterior mathematical method
```

所以对同一个固定 random seed 的 synthetic test，结果不应出现无原因的大漂移。

---

# 27. 代码执行顺序

Pi Agent 建议严格按以下顺序执行。

---

## Phase 0 — 状态检查

先：

```bash
git status --short
git rev-parse --abbrev-ref HEAD
git log -1 --oneline
```

如果存在用户未提交修改：

> **不要覆盖。**

先汇报，再决定是否能继续。

不要：

```bash
git reset --hard
git clean -fd
```

---

## Phase 1 — 阅读当前实现

至少阅读：

```text
mlipx/mlipx/analysis/transport.py
mlipx/mlipx/analysis/msd.py
mlipx/mlipx/analysis/runner.py
mlipx/mlipx/analysis/plots.py
mlipx/mlipx/analysis/dataset.py
mlipx/mlipx/analysis/validation.py
mlipx/mlipx/cli.py
mlipx/mlipx/tui/analysis_screen.py

tests/mlipx/mlipx/analysis/test_runner_kinisi.py
tests/mlipx/mlipx/analysis/test_transport_arrhenius_spectral.py
tests/mlipx/mlipx/test_cli.py
tests/mlipx/mlipx/test_tui.py
```

然后再开始改。

---

## Phase 2 — coordinate semantics

先只做：

```text
backend reconstructability check
results/provenance semantics
tests
```

跑相关测试通过后再继续。

---

## Phase 3 — conductivity posterior

实现 posterior propagation + tests。

不要和 TUI 一起改。

---

## Phase 4 — human-readable artifacts

实现：

```text
summary CSV
plot
CLI summary
```

跑测试。

---

## Phase 5 — TUI parity

最后加 TUI fields 和 tests。

这样如果 Textual 部分出现问题，不会把科学核心修改混在一起。

---

## Phase 6 — docs + revision bump

最后：

```text
transport revision 1 → 2
docs
```

---

# 28. 推荐的测试命令

先 targeted：

```bash
pytest -q tests/mlipx/mlipx/analysis/test_runner_kinisi.py
pytest -q tests/mlipx/mlipx/analysis/test_transport_arrhenius_spectral.py
pytest -q tests/mlipx/mlipx/test_cli.py
pytest -q tests/mlipx/mlipx/test_tui.py
```

再运行仓库现有 analysis 测试：

```bash
pytest -q tests/mlipx/mlipx/analysis
```

最后全量：

```bash
pytest -q
```

如果全量测试包含需要不存在外部模型/GPU/网络的项目，请严格按仓库现有 CI 约定处理，不要为了“全绿”乱跳过测试。

---

# 29. 静态检查

至少：

```bash
ruff check .
```

如果仓库当前使用 formatter：

```bash
ruff format --check .
```

如果已有 mypy / pyright / pre-commit 配置，按仓库约定执行。

不要引入新的 formatter 配置。

---

# 30. 手工 CLI help 检查

确认：

```bash
mlipx analyze RUN transport --help
```

仍有：

```text
--lag-step-ps
--lag-stop-ps
--temperature-K
--collective-conductivity
--positions-convention
--frame-interval-fs
```

---

# 31. TUI 手工 smoke test

启动 TUI：

```bash
mlipx
```

进入：

```text
Analyze Existing Run
→ Transport estimate (kinisi)
```

确认：

- lag step/stop 可输入；
- temperature optional；
- collective switch；
- source overrides；
- 切换其他 task 时不会留下乱七八糟 transport fields；
- 页面高度较小时仍能通过 VerticalScroll 使用。

---

# 32. 修改后不要跑用户的 8 条 LGPS 正式数据

这非常重要。

源码阶段只做：

- synthetic tests；
- short smoke tests；
- kinisi API compatibility。

不要擅自：

```text
UMA 700
UMA 800
MACE 700
...
```

跑正式数据。

用户会在代码推送 GitHub 后自己进行 regression + batch processing。

---

# 33. Git diff 审查

完成后必须：

```bash
git diff --stat
git diff
```

检查 diff 应主要集中在：

```text
analysis/transport.py
analysis/runner.py
analysis/plots.py
cli.py
tui/analysis_screen.py
相关 tests
transport docs
```

如果出现：

- calculator；
- MD engine；
- model backend；
- unrelated formatting；

请撤回无关修改。

---

# 34. Commit

建议 commit message：

```text
fix(analysis): finalize transport uncertainty and semantics
```

如果修改较多需要两个 commit，也可拆为：

```text
fix(analysis): harden kinisi coordinate semantics
feat(analysis): expose transport uncertainty summaries
```

但不要拆成十几个碎 commit。

---

# 35. Push

如果 git remote / auth 正常：

```bash
git push
```

推送当前工作 branch。

不要：

- 强推；
- 修改远程历史；
- 自动创建 release；
- 自动打 tag。

如果 push 因 auth 失败：

> 报告错误即可。

不要擅自改用户 git credential。

---

# 36. 最终给用户的报告格式

Pi Agent 完成后，必须给用户一个简洁但完整的总结。

至少包含：

### 代码改动

1. coordinate semantics 怎么修的；
2. exact unwrapped 如何检查 backend reconstructability；
3. Nernst–Einstein posterior uncertainty 如何传播；
4. TUI 新增哪些 fields；
5. 新增哪些 artifacts；
6. CLI 现在会打印什么；
7. `transport` output revision 改成多少。

### 测试

列：

```text
targeted pytest
analysis pytest
full pytest
ruff
kinisi 2.1.0 smoke test
```

每项给最终结果。

### Git

给：

```text
branch
commit hash
push result
```

### STOP

最后停止。

不要继续说：

> “我顺便帮你跑了 8 条轨迹。”

---

# 37. 修改完成后的用户回归测试（只写进报告，不代跑）

代码推送 GitHub 后，用户首先会重新跑：

```bash
mlipx analyze RUN transport \
  --mobile Li \
  --charge 1 \
  --drift-reference nonmobile \
  --fit-start-ps 40 \
  --lag-step-ps 2 \
  --lag-stop-ps 200 \
  --random-seed 0
```

对于 UMA 700 K，预期 tracer posterior 应仍接近此前：

```text
D mean:
3.440443641e-9 m²/s

D std:
2.087898119e-10 m²/s

95% credible interval:
[3.027256045e-9, 3.845268093e-9] m²/s
```

不要求 bitwise identical，但如果偏差明显超出 MCMC sampling / code-path 合理范围，就需要停下来排查。

---

# 38. 下一阶段科研工作（本次不要实现）

这轮代码完成后，用户真正要做的是：

```text
8 条 trajectory
→ 每条 D_self posterior
→ 每条 sigma_NE posterior
→ 汇总比较
```

当前 8 条：

```text
UMA 700 / 800 K
MACE 700 / 800 K
DPA 700 / 800 K
GRACE 700 / 800 K
```

统一先使用：

```text
fit window: 40–200 ps
lag step: 2 ps
lag stop: 200 ps
```

其中：

```text
DPA 700 K
```

此前 fitting-window sensitivity 较差，需要后续单独 review。

本次源码修改 **不要实现这一批处理逻辑。**

---

# 39. 设计原则总结

这轮代码修改最终应满足下面几句话。

### 轨迹语义

> mlipx 不会因为源文件标记为 exact unwrapped，就错误声称 kinisi 后端也直接保留 exact image counters；在 backend periodic reconstruction 可能改变真实 displacement 时必须 fail closed。

### MSD / transport 职责

> `msd` 继续负责透明、快速的 multiple-time-origin MSD 和 OLS diagnostic。

> `transport` 继续负责 kinisi covariance-aware Bayesian tracer-D posterior。

### conductivity

> Nernst–Einstein conductivity 不再丢弃已经存在的 D posterior uncertainty，但明确说明这只是 D posterior 的线性传播，不是总物理 uncertainty。

### UI

> CLI 和 TUI 至少要能表达同一套核心 transport 参数。

### 输出

> 用户跑完 transport 不需要再手写 `jq` 才知道 D、credible interval 和真正 fit window。

### scope

> 本轮之后冻结 Analysis v2 的这一条主线，开始处理真实科研数据，而不是继续无限开发软件。

---

# 40. 最终验收清单

在宣布完成之前逐项确认：

- [ ] wrapped-source transport safety guard 仍存在
- [ ] exact-unwrapped source 不再被错误认为“kinisi 直接保留 exact images”
- [ ] safe exact-unwrapped source 可以通过 reconstructability check
- [ ] image-history 会丢失的 exact-unwrapped source fail closed
- [ ] backend position semantics 写入 results / provenance
- [ ] tracer D posterior 不变
- [ ] NE conductivity posterior mean/std/median/95% CrI 存在
- [ ] legacy NE scalar 字段仍存在并等于 posterior mean
- [ ] TUI 有 lag step
- [ ] TUI 有 lag stop
- [ ] TUI 有 optional temperature
- [ ] TUI 有 collective switch
- [ ] TUI 有 imported trajectory positions convention override
- [ ] TUI 有 imported trajectory frame interval override
- [ ] lag pair validation 正确
- [ ] `transport_summary.csv` 存在
- [ ] `transport_msd.png` 存在
- [ ] `transport_msd.svg` 存在
- [ ] plot 明确显示 fit window
- [ ] CLI 打印 D posterior summary
- [ ] CLI 使用 “credible interval” 术语
- [ ] `actual_step_ps` 对插入 fit-start 的非均匀 grid 不再误导
- [ ] `transport` output revision = 2
- [ ] targeted tests pass
- [ ] analysis tests pass
- [ ] full applicable tests pass
- [ ] ruff pass
- [ ] kinisi 2.1.0 smoke test pass
- [ ] diff 无无关大改
- [ ] commit 完成
- [ ] push 完成或明确报告 push failure
- [ ] STOP
