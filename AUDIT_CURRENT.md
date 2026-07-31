# mlipx 审查报告（阶段 A：只读审查）

> 本报告依据 `mlipx_当前版本审查与后续改进任务清单.md` 的要求，对当前工作区源码进行只读审查。
> 核心原则已遵守：先调查当前源码与官方后端接口，再判断问题是否存在；每项给出状态与证据。
> 修改代码在阶段 B 进行，本阶段不修改源码。

## 0. 事实基线

| 项 | 值 |
|---|---|
| 分支 | `main` |
| HEAD | `ea340d5 fix bugs` |
| 未提交修改 | 无（仅新增本清单 md） |
| Python | 3.12.13 |
| mlipx 包位置 | `uma/mlipx/`（workspace member） |
| 测试结果 | `pytest tests/uma/mlipx tests/core/mlipx` → **209 passed, 1 failed, 1 skipped** |
| 失败测试 | `tests/uma/mlipx/config/test_defaults.py::test_mace_default_dtype_is_float32` |
| 已安装后端 | mace-torch 0.3.16（`.venv-mace`）；fairchem-core/UMA（`.venv`） |
| 未安装后端 | `deepmd-kit`、`tensorpotential`（DPA/GRACE 无法本地实跑） |

测试区分（按 §2.2）：
- 配置/接口/Mock 测试：充分（`test_resolver`、`test_schema`、`test_factory`、`test_engine`、`test_cli`）。
- 真实后端集成测试：仅有 MACE 环境校验（`test_mace_calculator`）；UMA/MACE/DPA/GRACE 的“官方直调一致性”测试**不存在**。
- 数值一致性/物理行为测试：缺少。

---

## 1. UMA 后端（清单 §3）

### 3.1 参数传递 — 状态：ALREADY_FIXED / 已验证
- 证据：`uma/mlipx/calculator.py` `UMACalculator.__init__` 接收 `task/device/inference_mode/torch_num_threads/activation_checkpointing`；`load_predictor()` 把 `device`、`inference_settings`（由 `guess_inference_settings(inference_mode)` 派生，并覆盖 `activation_checkpointing`/`torch_num_threads`/`compile`）传给 `load_predict_unit(path=, device=, inference_settings=)`。
- 引擎（`engine.py` `_create_calculator`）仅对 UMA 引擎把顶层 `inference_mode/torch_num_threads/activation_checkpointing` setdefault 进 calc_opts；非 UMA 引擎不转发（避免跨引擎告警）。链路完整。
- 影响：无。无需修改。

### 3.2 设备索引 — 状态：CONFIRMED（次要）
- 证据：`calculator.py` `_check_gpu_compatibility()` 与 `_compile_supported()` 均硬编码 `torch.cuda.get_device_capability(0)` / `get_device_name(0)`，未使用 `self.device` 指定的 `cuda:N`。
- 当前行为：GPU 能力/turbo 兼容性检查永远探测物理 GPU 0。用户指定 `cuda:1` 时，检查的是 0 号卡的架构。
- 实际模型设备：`load_predict_unit(device=self.device)` 正确转发用户指定设备，故**计算本身不受影响**，仅“兼容性预检”可能误判。
- 影响：usability（多卡时预检可能误导）。
- 处理：阶段 B 小修（用 `self.device` 解析出的索引做探测；单卡环境无影响）。

### 3.3 / 3.4 官方直调一致性 / 验收 — 状态：OUT_OF_SCOPE（本轮）
- 本地具备 UMA 模型文件（`uma-s-1.pt` 等），但本轮聚焦源码缺陷；“官方直调对照”数值测试需要专门基准，列为后续阶段 C。

---

## 2. MACE 后端（清单 §4）

### 4.1 dtype 全链路 — 状态：CONFIRMED（P0）
- 证据：
  - `uma/mlipx/config/defaults.py` `BUILTIN_DEFAULTS["calculator.mace"]["default_dtype"] = "float64"`（代码注释解释为“calc-type aware”）。
  - `engine.py` `_create_calculator`：`calc_opts.setdefault("default_dtype", "float32" if calc_type=="md" else "float64")` —— 对 sp/opt **静默**把默认 dtype 改成 float64。
  - 文档 `docs/core/mlipx_configuration.md` 第 74 行（及 CN 版第 49/56 行）明确写 `calculator.mace | default_dtype=float32`，且“`--dtype float64` 覆盖别名默认的 float32”。
  - 测试 `test_defaults.py::test_mace_default_dtype_is_float32` 断言默认为 float32 —— **当前失败**。
- 当前行为：用户不指定 dtype 时，sp/opt 实际跑 float64（与文档承诺的 float32 不符）；MD 跑 float32。这是一个“任务默认值静默覆盖文档默认值”的违反（§4.1：SP/OPT/MD 默认选择须与文档一致；用户显式设置不被任务默认覆盖）。
- 影响：correctness（与文档不一致）、reliability（用户误以为 sp/opt 是 float32）。
- 处理：阶段 B 统一为 float32 默认（匹配文档+测试），移除 sp/opt 的静默 float64 覆盖；`--dtype float64` 仍可显式提精度。factory 直构默认也改为 float32 保持一致。

### 4.1 head 全链路 — 状态：已验证
- 证据：`mace_calc.py` 以单数 `head=` 传给 `MACECalculator`（已对照 mace 0.3.16 `inspect.signature`：`head` 经 `**kwargs` 接收）。注释明确说明不用复数 `heads`。`info()` 记录 `active_head`（实际选中）与 `available_heads`。
- 影响：无。无需修改。

### 4.2 / 4.3 模型 head / 信息与元素覆盖 — 状态：已验证
- `info()` 对 `models[0].r_max`/`atomic_numbers`、`calc.available_heads`/`head`/`z_table`/`num_models` 做了 best-effort 读取，每个访问都有 guard；未依赖不存在的 `elements`/`heads` 属性。
- 影响：无。

### 4.4 性能选项 — 状态：DESIGN_CHOICE
- `compile_mode/fullgraph/enable_cueq/enable_oeq/warmup` 未实现且未在 schema 暴露；`_CALC_KEYS["mace"]={"default_dtype","head"}`，传入未实现键会被 factory `_check_unknown_kwargs` 拒绝/告警（非静默忽略）。符合“未实现参数不被静默接受”。
- 影响：无。无需修改。

### 4.5 显存与长轨迹 — 状态：OUT_OF_SCOPE（本轮）
- 需 V100 实测，列为阶段 D。

---

## 3. 公共 MDRunner（清单 §5）

### 5.1 预弛豫语义 — 状态：CONFIRMED（P0，次要）
- 证据：`runners/md.py` `_pre_relax_structure` 用 `FIRE(atoms)`（仅优化位置，不动晶胞）。异常分支日志写 `! Continuing with original structure...`，但 `atoms` 已被 `optimizer.run()` 部分修改，此时并非“原始结构”。
- 当前行为：预弛豫失败时，日志声称用原始结构，实际用的是部分弛豫后的结构。
- 影响：usability/correctness（误导排查）。
- 处理：阶段 B 修正日志为“部分弛豫后的结构”并记录步数；不改变“继续运行”的设计。

### 5.2 配置项是否静默失效 — 状态：CONFIRMED（P0）
- 证据（Schema 接受但 Runner 未使用）：
  - `seed`：`resolver.py` 第 296-299 行在 MD 时**自动生成** seed 并写入 `run_options["seed"]` 与 `sources["seed"]`（出现在 resolved_config.json 里），但 `MDRunner` 从不读取 `seed`，`_initialize_velocities` 调 `MaxwellBoltzmannDistribution` 时不传 `rng`。→ seed 被解析、记录、却完全不生效，且给用户“可复现”的假象。
  - `velocity_policy`（默认 auto）：在 schema/settings 中，Runner 不读取；始终强制初始化速度。
  - `equil_steps`（默认 0）：在 schema/settings 中，Runner 不读取。
  - `pre_relax_mode`（默认 none）：在 schema/settings 中，Runner 用旧的 `pre_relax`(bool)，不读取 mode。
- 这正是 §5.2 禁止的“解析成功、校验成功、运行成功，但参数无任何作用”。
- 影响：correctness/reliability（seed 假复现；velocity_policy/equil_steps/pre_relax_mode 假生效）。
- 处理：阶段 B —— 实现 seed（经 `rng`）与 velocity_policy（auto/initialize/preserve）；对 equil_steps(>0) 与 pre_relax_mode(!=none) 改为“显式拒绝 NotImplementedError”，从静默失效转为明确未实现。

### 5.3 随机种子和速度初始化 — 状态：CONFIRMED（P0）
- 证据：`_initialize_velocities` 无 RNG/seed；始终 `MaxwellBoltzmannDistribution(..., force_temp=True)` + `Stationary`。无“重启保留速度”逻辑（也无重启机制）。`Stationary` 移除质心平动（preserve_temperature=True）。
- ASE 3.28.0 的 `MaxwellBoltzmannDistribution` 支持 `rng=`（已确认签名），可接 `np.random.default_rng(seed)`。
- 影响：correctness（不可复现）。
- 处理：阶段 B 与 5.2 一并实现。

### 5.4 温度和自由度 — 状态：已验证
- 证据：`_calculate_temperature` 优先 `atoms.get_temperature()`（ASE 约束感知，`3N - sum(removed_dof)`），仅对不报告 DOF 的约束（如 FixSymmetry）回退 `3N-3` 并告警。未自维护易漏约束的公式。
- 影响：无。

### 5.5 轨迹内存占用 — 状态：CONFIRMED（P1）
- 证据：`runners/md.py` `run()` 维护 `trajectory = []`，每个 `save_interval` 步 `trajectory.append({"atoms": atoms.copy(), ...})`；同时 `traj_writer.write(atoms)` 写 `.traj`。XDATCAR 由 `XdatcarWriter.write_from_md(path, trajectory, ...)` 从**内存列表**生成。`results["trajectory"]` 返回完整帧列表。
- 当前行为：RAM 随帧数线性增长；100 ps 长程 MD 会持续积累完整帧。
- 影响：reliability/performance（长程 MD 内存膨胀；“框框炸显存”语境下属工程缺陷）。
- 处理：阶段 B 改为流式——内存只保留标量摘要；XDATCAR 用 `append_frame` 增量写入；`.traj` 仍为完整帧真源；`results["trajectory"]` 只含标量。

### 5.6 checkpoint 与 restart — 状态：CONFIRMED（P1，未实现）
- 证据：MDRunner 无任何 checkpoint/restart 逻辑；仅在结束时写 CONTCAR。无 `checkpoint_interval`/`restart_from`/`append`。`BUILTIN_DEFAULTS["output"]["checkpoint_interval"]=1000` 但 Runner 不读取。
- 影响：reliability（中断不可恢复）。属 P1，本轮不强制补全；阶段 B 至少把 `checkpoint_interval` 等“声明但未实现”项显式标注，避免静默。
- 处理：阶段 B 仅做“显式拒绝/文档标注”；完整 restart 列入后续。

### 5.7 NaN/Inf 和有限大力 — 状态：已验证（符合设计）
- 证据：`base.py` `_check_finite` 对能量/力 NaN/inf **无条件 raise**（MD 每步检查能量与力）。有限大力：`md.py` 在 `max_force > 20.0` 时每 100 步告警（不 abort）。符合 §1.1/§5.7“保留 warning、不默认 abort”。
- 小缺口：告警阈值 `20.0` 硬编码，不可配置（`BUILTIN_DEFAULTS["safety"]["fmax_abort"]=20.0` 存在但 MDRunner 不读取）。
- 影响：usability（阈值不可配）。
- 处理：阶段 B 把告警阈值接上 `safety.fmax_warn`/`fmax_abort`（小改），保持 warning 不 abort。

---

## 4. DPA 后端（清单 §6）

### 6.1 模型格式 — 状态：UNCERTAIN
- 证据：`dpa_calc.py` docstring 称 `.pth/.pt`(PyTorch) 或 `.pb`(TF)。对照 deepmd-kit master：ASE `DP` calculator 接收 `model` 路径并构造 `DeepPot`；PT 后端 `deep_eval_pt` 对 `.pt` 以 `torch.load(map_location=env.DEVICE)` 加载冻结模型。
- 未本地安装 deepmd-kit，无法验证 `.pt` 训练 checkpoint 与冻结模型的区分报错。README 示例用 `dpa2.pth`。
- 影响：documentation only（潜在）。
- 处理：阶段 B 保持现状；在 docstring 补一句“仅支持冻结/导出模型，训练 checkpoint 需先 freeze/export”并指引。

### 6.2 设备选择 — 状态：CONFIRMED（P0）
- 证据：`dpa_calc.py` `get_calculator()` 调 `DP(model=str(self.model_path), type_dict=None)` —— **不传 device**。`self._device` 仅存于 `info()` 里以 `"device": self._device` 上报。
- 官方依据：deepmd-kit master `DP.__init__(model, label, type_dict, neighbor_list, head, nlist_backend, **kwargs)` **无 device 参数**；`**kwargs` 进 ASE `Calculator.__init__`，不进 `DeepPot`。DeepMD 设备由 `CUDA_VISIBLE_DEVICES`/`deepmd.env.DEVICE` 决定。
- 当前行为：用户 `--device cuda:1` 被保存但完全不影响计算设备；`info()` 把“请求设备”当“实际设备”上报（§6.2 禁止猜测）。
- 影响：correctness/reliability（设备不生效；元数据误导）。
- 处理：阶段 B —— 在构造 DP 前按 `cuda:N` 设置 `CUDA_VISIBLE_DEVICES`（使设备真正生效）；`info()` 区分 `requested_device` 与 `actual_device`（无法可靠获取时标 `unknown`，不猜测）。

### 6.3 多头模型和专属参数 — 状态：DESIGN_CHOICE
- `DP` 支持 `head`/`nlist_backend`，mlipx 未暴露；但这些键不在 `_CALC_KEYS["dpa"]`(空集)，传入会被 factory 拒绝/告警（非静默）。符合“只暴露可测参数；未实现不静默忽略”。
- 影响：无。

### 6.4 / 6.5 官方直调 / 最低验收 — 状态：OUT_OF_SCOPE（本轮）
- 需 deepmd-kit 与冻结模型，列阶段 C。

---

## 5. GRACE 后端（清单 §7）

### 7.1 模型加载来源 — 状态：已验证（本地 SavedModel）/ 部分 CONFIRMED
- 证据：对照 tensorpotential 0.6.0（`ICAMS/grace-tensorpotential`）：`TPCalculator.__init__(model, pad_neighbors_fraction=0.05, pad_atoms_number=1, min_dist=None, ..., mode="diverse", ...)`，第一参数 `model` 可为 SavedModel 路径或模型列表。`grace_calc.py` 调 `TPCalculator(model=str(self.model_path))` **构造方式正确**。
- 未支持 foundation model 名称（需 `grace_fm(model=name)`）。§7.1 允许不实现 alias，但文档须准确。
- 处理：阶段 B 核对文档不宣称 foundation alias（若宣称则修正）。

### 7.2 mode 选择 — 状态：DESIGN_CHOICE（P2）
- `TPCalculator` `mode` 默认 `"diverse"`（segment_sum，异构结构），可选 `"uniform"`（dense，单结构/同尺寸）。mlipx 未暴露，使用默认 diverse。mode 只影响性能不影响结果（两者都算能量/力），属 performance 非 correctness。
- 影响：无（默认 diverse 安全）。
- 处理：本轮不强制；可选 P2 暴露。

### 7.3 最小距离参数 — 状态：DESIGN_CHOICE
- `TPCalculator` 有 `min_dist=None`。mlipx 未暴露；`min_dist` 不在 schema/`_CALC_KEYS["grace"]`，传入会被拒绝（非静默）。§7.3 不要求强制开启。
- 影响：无。

### 7.4 Padding 和 XLA — 状态：DESIGN_CHOICE
- `pad_neighbors_fraction`/`pad_atoms_number` 存在，未暴露（同上，非静默）。
- 影响：无。

### 7.5 TensorFlow 设备与显存 — 状态：CONFIRMED（P0）
- 证据：`grace_calc.py` `get_calculator()` 调 `TPCalculator(model=str(self.model_path))` —— **不传 device**；`TPCalculator` 亦无 device 参数（TF 设备由 `CUDA_VISIBLE_DEVICES` 决定，须在 import TF 前设置）。`self._device` 仅在 `info()` 当 `"device"` 上报。
- 当前行为：`--device cuda:1`/`--device cpu` 不生效；`info()` 误导。
- 影响：correctness/reliability。
- 处理：阶段 B 与 DPA 同型修复（env + 诚实 info()）。未实现 TF memory growth 控制——本轮标注文档，不强制。

### 7.6 委员会模型 — 状态：DESIGN_CHOICE
- `TPCalculator` 支持模型列表（ensemble）与 `enable_uq_if_available`；mlipx 仅传单模型。需核对文档不宣称已接入委员会/UQ。
- 影响：无。

### 7.7 最低验收 — 状态：OUT_OF_SCOPE（本轮）

---

## 6. Batch 与 Job Queue（清单 §8）

### 8.1 Sweep Batch — 状态：部分 CONFIRMED（P1）
- 证据：`runners/batch.py` `BatchRunner` 支持 sp/opt；`md` 显式 `NotImplementedError`（非静默，好）。模型在 `engine.run_batch` 中只加载一次（`calculator.get_calculator()` 一次）。并行用 `ThreadPoolExecutor` **共享同一个 `self.calculator`** 给多线程。
- 问题：§8.1 警告“多线程共享带缓存状态的 ASE Calculator”不可靠；建议独立进程+独立 Calculator。
- `BUILTIN_DEFAULTS["batch"]` 的 `continue_on_error/resume/retry_failed/mode/...` 被 BatchRunner **部分忽略**（continue_on_error 实际由 try/except 默认开；resume/retry 未实现）。属“声明但未完全使用”。
- 影响：reliability（并行共享 Calculator 状态）。
- 处理：P1，本轮不强制改并行模型；默认 `max_workers=1` 已是单 GPU 串行（安全）。

### 8.2 异构 Job Queue — 状态：CONFIRMED（未实现，P1）
- 证据：`jobs.py` 仅为“后台子进程”管理（PENDING/RUNNING/DONE/FAILED/CANCELLED + kill/clean），无 manifest/depends_on/restart_from/resume-skip-success。
- 子进程用 `subprocess.Popen([...])` 参数数组，**无 `shell=True`**（符合 §14 #13）。
- 影响：reliability（无任务接力/恢复）。§8.2 允许不一次补全。
- 处理：本轮不实现；列后续。

### 8.3 多环境执行 — 状态：部分实现
- `jobs.py` `submit` 用 `sys.executable` + `job_worker`；未发现 `settings.ini [engine:*] executable=` 的解析使用。多 venv 隔离靠用户手动 `.venv-mace/bin/mlipx`。
- 处理：本轮不动。

### 8.4 GPU 并发 — 状态：DESIGN_CHOICE
- 默认 `max_gpu_jobs=1`/`max_workers=1`（单 GPU 串行）；无 GPU 排他锁实现。
- 处理：本轮不动。

---

## 7. settings.ini 与配置系统（清单 §9）

### 9.1 唯一默认值来源 — 状态：已验证
- `defaults.py` 为单一来源；CLI argparse 默认均为 `None`（由 resolver 决定）；INCAR 模板由 `build_incar_default` 从 defaults 生成。
- 影响：无。

### 9.2 配置优先级 — 状态：已验证
- `resolver.py` 层次：built-in < settings.ini < model alias < profile < INCAR < CLI。每键带 `source` trace；`mlipx config explain` 可查。有 `test_resolver` 覆盖。
- 影响：无。

### 9.3 配置路径 — 状态：已验证
- `--settings`/`MLIPX_SETTINGS`/`./settings.ini`/`~/.config/mlipx/settings.ini` 四级搜索（`settings.py`）。非包安装目录。
- 影响：无。

### 9.4 严格校验 — 状态：已验证
- `schema.py` + `factory._check_unknown_kwargs` + `engine._validate`：strict 模式未知键 raise（带 typo 建议）；非 strict 警告。`DEFAULT_DTPE` 这类 typo 会被建议 `default_dtype`。
- 影响：无。

### 9.5 resolved config — 状态：已验证
- `cli._emit_resolved_config` 写 `resolved_config.json`（含 sources trace）。
- 影响：无。

---

## 8. 输出和可追踪性（清单 §10）
- 现状：写 OUTCAR/XDATCAR/CONTCAR/mlipx_results.json/resolved_config.json/run.log。无 `run_manifest.json`（git commit/模型哈希/版本等）。§10 期望的 manifest 未实现。
- 处理：P1，本轮不强制。

---

## 9. 测试补充（清单 §11）
- 缺口：无官方直调集成测试；无真实 RNG/seed 测试；无 velocity_policy 测试；无 DPA/GRACE 设备元数据测试；无长轨迹内存测试。
- 阶段 B 随各修复补回归测试。

---

## 10. P0 确认问题汇总与修改顺序

| # | 问题 | 清单条目 | 文件 | 处理 |
|---|---|---|---|---|
| P0-1 | MACE 默认 dtype 代码(float64)与文档/测试(float32)不一致；sp/opt 被静默改 float64 | §4.1 | `config/defaults.py`, `calculators/factory.py`, `engine.py` | 统一 float32 默认 |
| P0-2 | `seed` 自动生成并记录但 MDRunner 完全不使用 | §5.2/§5.3 | `runners/md.py`, `engine.py` | 接 `rng` |
| P0-3 | `velocity_policy` 被接受但始终强制初始化速度 | §5.2/§5.3 | `runners/md.py` | 实现 auto/initialize/preserve |
| P0-4 | `equil_steps`/`pre_relax_mode` 被接受但静默忽略 | §5.2 | `runners/md.py` | 非默认值显式拒绝 |
| P0-5 | DPA `device` 保存但不生效；info() 误导 | §6.2 | `calculators/dpa_calc.py` | env + 诚实 info() |
| P0-6 | GRACE `device` 保存但不生效；info() 误导 | §7.5 | `calculators/grace_calc.py` | env + 诚实 info() |
| P0-7 | 预弛豫失败日志谎称“原始结构” | §5.1 | `runners/md.py` | 修正日志 |
| P0-8 | UMA GPU 兼容预检硬编码 device 0 | §3.2 | `calculator.py` | 用请求设备索引 |

附带 P1（本轮一并做，因属明确工程缺陷且有文档指引）：
| # | 问题 | 处理 |
|---|---|---|
| P1-1 | MD 轨迹内存随帧线性增长；XDATCAR 依赖内存列表 | §5.5 流式化 |
| P1-2 | 有限大力告警阈值硬编码 20，不可配 | 接 `safety.fmax_warn/abort` |

不在本轮：扩散分析（§1.2）、把有限大力改默认 abort（§1.1）、TUI（§0.7）、完整 Job Queue/manifest（§8.2）、run_manifest（§10）、真实后端直调数值基准（§3.3/6.4/7.7，需模型与环境）。

---

## 11. 结论
UMA/MACE 的参数链主干已较成熟；核心缺陷集中在：
1. MACE dtype 默认值与文档/测试不一致（且 sp/opt 被静默覆盖）；
2. MD 侧 `seed`/`velocity_policy` 等“假生效”参数；
3. DPA/GRACE 的 `device` 完全不生效且元数据误导。
阶段 B 将按上表顺序做小范围、带测试、可回退的修改。

---

## 12. 阶段 B 修复完成（Phase B Resolution）

所有 P0 与附带 P1 项已修复并附回归测试。全量测试：`pytest tests/core/mlipx tests/uma/mlipx` →
**241 passed, 1 skipped**（基线 209 passed / 1 failed / 1 skipped；原失败 `test_mace_default_dtype_is_float32` 现通过；新增 ~31 项回归测试）。

| # | 问题 | 修复 | 关键文件 | 测试证据 |
|---|---|---|---|---|
| P0-1 | MACE 默认 dtype 代码(float64)与文档/测试(float32)不一致 | 统一为 float32 默认（所有 calc_type），移除 sp/opt 静默 float64 覆盖；显式 `--dtype float64` 仍可提精度 | `config/defaults.py`, `calculators/factory.py`, `engine.py` | `test_engine.py::TestMaceDtypeDefaults`, `test_factory.py::test_factory_mace_default_dtype_is_float32`, `test_fixes.py::test_mace_dtype_default_is_float32_all_calc_types`, `test_defaults.py::test_mace_default_dtype_is_float32` |
| P0-2 | `seed` 自动生成但 MDRunner 不使用 | `_initialize_velocities` 用 `np.random.default_rng(seed)` 经 ASE `rng=` 注入；engine/CLI 透传 `seed` | `runners/md.py`, `engine.py`, `cli.py` | `test_md_runner.py::test_md_seed_makes_velocities_reproducible` |
| P0-3 | `velocity_policy` 被接受但始终强制初始化 | 实现 auto（保留现有速度）/initialize（强制重init）/preserve（无速度则报错）；校验非法值 | `runners/md.py` | `test_md_runner.py::test_md_velocity_policy_*` |
| P0-4 | `equil_steps`/`pre_relax_mode` 静默忽略 | 非默认值显式抛 `NotImplementedError`（从静默失效转为明确未实现） | `runners/md.py` | `test_md_runner.py::test_md_rejects_equil_steps`, `test_md_rejects_pre_relax_mode` |
| P0-5 | DPA `device` 保存但不生效；info() 误导 | 构造 DP 前按 `cuda:N`/`cpu` 设置 `CUDA_VISIBLE_DEVICES`（`setdefault` 不覆盖用户值）；`info()` 区分 `requested_device`/`actual_device`（不可靠时标 `unknown`，不猜测） | `calculators/dpa_calc.py` | `test_factory.py::TestDpaGraceDevice::test_dpa_*` |
| P0-6 | GRACE `device` 保存但不生效；info() 误导 | 同 DPA：构造 TPCalculator 前设 `CUDA_VISIBLE_DEVICES`；诚实 `info()` | `calculators/grace_calc.py` | `test_factory.py::TestDpaGraceDevice::test_grace_*` |
| P0-7 | 预弛豫失败日志谎称“原始结构” | 异常分支日志改为“部分弛豫后的结构 (step N)”并记录步数 | `runners/md.py` | （行为日志修正，随 P0-2..4 测试覆盖路径） |
| P0-8 | UMA GPU 兼容预检硬编码 device 0 | 新增 `_gpu_index()` 解析 `cuda:N`；`_check_gpu_compatibility`/`_compile_supported` 用请求逻辑索引探测 | `calculator.py` | `test_calculator.py::TestGpuIndex` |
| P1-1 | MD 轨迹内存随帧线性增长；XDATCAR 依赖内存列表 | 流式化：完整帧→`trajectory.traj`+XDATCAR（逐帧 append）；标量→`md.csv`（逐行）；内存只留标量摘要；`results["trajectory"]` 只含标量 | `runners/md.py` | `test_md_runner.py::test_md_streams_trajectory_to_disk`, `test_md_long_trajectory_keeps_only_scalars_in_memory` |
| P1-2 | 有限大力告警阈值硬编码 20，不可配 | `MDRunner.fmax_abort` 默认取自 `BUILTIN_DEFAULTS["safety"]["fmax_abort"]`（单一来源，非魔法字面量）；engine 透传 | `runners/md.py`, `engine.py` | `test_md_runner.py::test_md_fmax_abort_defaults_to_safety`, `test_md_fmax_abort_warning_uses_configured_threshold` |

### 设计不变量（保持）
- 预弛豫失败仍“继续运行”（仅修正日志措辞，不改变语义）。
- 有限大力仍仅 `warning`，不默认 `abort`（符合 §1.1/§5.7）。
- `velocity_policy=preserve` 无速度时**显式报错**而非静默初始化（避免假生效）。
- DPA/GRACE 设备 env 用 `setdefault`，**永不覆盖**用户已设的 `CUDA_VISIBLE_DEVICES`。
- `info()` 的 `device` 键保留（向后兼容输出 writer），新增 `requested_device`/`actual_device`。

### 未在本轮（列后续）
扩散分析(§1.2)、有限大力改默认 abort(§1.1)、TUI(§0.7)、完整 Job Queue/manifest(§8.2)、run_manifest(§10)、真实后端直调数值基准(§3.3/6.4/7.7，需模型与环境)、完整 checkpoint/restart(§5.6)。

---

## 13. 阶段 C 复核、勘误与真实模型验收（2026-07-30）

本节是对前述阶段 A/B 的再次核验；若与历史段落冲突，以本节和当前代码为准。
阶段 B 的 241-pass 记录是当时事实，不是当前最终测试数。

### 13.1 对阶段 B 结论的必要勘误

- `seed` 过去只控制 Maxwell–Boltzmann 初速度，不能复现 Langevin 随机力；
  现在同一个 `numpy.random.Generator` 同时驱动初速度与全部 NVT 随机 kick，
  整段随机轨迹可复现。
- 阶段 B 写“预弛豫失败继续运行”，但这会从半修改、可能不安全的结构启动
  MD。现在失败即中止；短程 FIRE 只降低原子力，不宣称消除晶胞应力或保证
  局部极小值。
- 阶段 B 写 DPA/GRACE 的设备选择使用 `setdefault`、保留旧环境变量。这样
  `--device cpu` 或 `cuda:N` 可能不生效。现在显式 mlipx 参数覆盖继承的
  `CUDA_VISIBLE_DEVICES`，而 `info()` 仍诚实区分 requested/actual。
- 阶段 B 写“DPA 多头参数不暴露无影响”。本地
  `DPA-3.2-5M.pt` 实际是 24 分支多任务模型；默认分支是 OMat24，用于
  LGPS 固态电解质会选错势能面。现已为 DPA 打通 `--head`，LGPS 验收使用
  `Domains_SSE_PBE`，分子验收使用 `OMol25`。
- `safety.fmax_abort` 虽在阶段 B 被称为已接通，实际仍停留在
  `ResolvedConfig.settings`，没有进入 `MDRunner`。现已补上 resolver →
  EngineConfig → MDRunner 的桥接。

### 13.2 本轮新增 correctness / reliability 修复

1. NVT 使用显式 `FixCom` 约束并关闭 Langevin 的旧 `fixcm` 修正，使温度
   自由度和质心处理使用同一约束语义；参数的温度、步长、步数、保存间隔和
   NVT friction 均做物理范围校验。
2. 带已存 momenta 的相空间重启若同时要求 positions-only 预弛豫，程序明确
   拒绝；否则“保留旧速度但改变位置”并不是严格重启。
3. 全零 momenta 仍被视为用户明确提供的合法初态，不再误判为“没有速度”。
4. 分子任务禁止晶胞优化；`FixSymmetry` 不再覆盖用户已有约束；SP/OPT/MD
   最终能量、力、应力做有限值检查。
5. MACE 遇到不存在的 head 时拒绝 mace-torch 的静默 fallback，防止暗中更换
   势能面。本地 MACE 文件只有 `default` head。
6. XDATCAR 改为标准 `Direct configuration=` 标记；连续元素块和原子顺序
   一致，已由 ASE 反向读取验证。JSON 的零能量不再被当作假值丢掉。
7. CLI batch 未给 `--pattern` 时会发现 CIF/XYZ/VASP/POSCAR；显式 pattern
   只匹配用户指定格式。同 stem 的不同后缀不再互相覆盖。
8. 多线程共享一个带缓存 ASE calculator 不安全，现阶段显式拒绝，而不是保留
   一个看似可并行的竞态路径；串行批处理仍只加载一次模型。
9. `mlipx config init` 过去生成的文件会被自己的 `config validate` 判出三十多
   项非法，而且含 manifest/resume/checkpoint 等未实现伪开关。模板和内建默认
   现只列真正消费的选项；MD/OPT section 不再交叉泄漏；CLI/alias/profile 选择
   MACE 后能够正确读取 `[engine:mace]`。
10. GRACE 输入现在明确要求导出的 SavedModel 目录，文档不再声称 YAML 可直接
    交给 `TPCalculator`；MACE/DPA/GRACE 都校验设备字符串。
11. INCAR 的 `WRITE_FORCES`、`WRITE_STRESS`、`WRITE_TRAJECTORY`、
    `WRITE_JSON` 过去只被解析而未执行；现已贯通到各 runner/batch。
    非 VASP `OUTPUT_FORMAT` 会提前明确拒绝，不再静默仍写 VASP。

### 13.3 真实后端小任务矩阵

统一使用从 `LGPS222.vasp` 经 spglib 得到的 50 原子 primitive cell；每次 MD
仅 1 步、优化最多 1 步，避免大型长测。Batch 使用单结构。训练域的
`omat/omol/oc20/...` 不是 SP/OPT/MD 的同义“计算任务”，不能把同一个 LGPS
结构无物理依据地塞进全部训练域；LGPS 统一选择材料/固态电解质对应分支。

| 后端 / checkpoint | 设备 / 域 | SP | OPT≤1 | NVE MD 1 | Batch SP |
|---|---|---:|---:|---:|---:|
| UMA `uma-s-1.pt` | V100 / omat | PASS | PASS | PASS | PASS |
| UMA `uma-s-1p1.pt` | V100 / omat | PASS | PASS | PASS | PASS |
| UMA `uma-s-1p2.pt` | V100 / omat | PASS | PASS | PASS | PASS |
| UMA `uma-s-1p2p1.pt` | V100 / omat | PASS | PASS | PASS | PASS |
| UMA `uma-m-1p1.pt`（约 11 GB） | V100 / omat | PASS | PASS（初态已收敛） | PASS | PASS |
| MACE `mace-mpa-0-medium.model` | V100 / bulk, head=default | PASS | PASS | PASS | PASS |
| DPA `DPA-3.2-5M.pt` | CPU / bulk, head=Domains_SSE_PBE | PASS | PASS | PASS | PASS |
| DPA `DPA-3.2-5M.pt` | V100 / bulk, head=Domains_SSE_PBE | PASS | PASS | PASS | PASS |
| GRACE SavedModel | CPU / bulk | PASS | PASS | PASS | PASS |
| GRACE SavedModel | V100 / bulk（独立 `.venv-grace`） | PASS | PASS | PASS | PASS |

另用 water.xyz 对 DPA `task=molecule, head=OMol25` 做 SP，PASS。Batch 的 OPT
共享路径使用零势回归测试覆盖。典型 LGPS SP 能量（不同模型不可直接横比）：
UMA s-1 `-216.334019 eV`、MACE `-216.184937 eV`、DPA
`-216.233999 eV`、GRACE `-216.129217 eV`。

### 13.4 环境结论

- UMA：项目 `.venv`，PyTorch 2.6.0+cu124，V100 实测通过。根项目为保留
  Pascal `sm_60` 内核而显式覆盖了当前 fairchem-core 的 `torch~=2.8.0`
  元数据约束，因此 `uv pip check --python .venv/bin/python` 会报告这一项；
  这是有注释的兼容性取舍，不是解析器遗漏。本报告只确认 mlipx 使用的 UMA
  推理路径，不能据此声称 fairchem-core 的其他功能也受该非上游组合支持。
- MACE：独立 `.venv-mace`，mace-torch 0.3.16，V100 实测通过；本地模型
  权重以 float64 保存，mlipx 的性能默认 float32 会由 mace-torch 明示转换，
  需要保持存储精度时显式 `--dtype float64`。
- DPA：项目环境的 PyTorch ABI 与 DeepMD wheel 不匹配，因此使用独立
  `.venv-dpa`。最初为节省磁盘使用 PyTorch 2.10 CPU ABI1；空间释放后替换为
  PyTorch 2.10.0+cu126 ABI1。DeepMD 3.1.3 确认为 CUDA variant，V100 上
  Torch CUDA 算子及 DPA 的 SP/OPT≤1/NVE MD 1/Batch SP 全部通过；GPU SP
  能量 `-216.234000 eV`，与 CPU 结果一致到约 1 µeV。
- GRACE：共享环境 CPU 全部通过；UMA/PyTorch 固定的 cuDNN 9.1 不满足
  TensorFlow 2.20 的 cuDNN ≥9.3 要求，因此新增独立 `.venv-grace`。
  V100 上自动解析到的 cuDNN 9.24 虽能运行 GRACE 实际计算图，但通用
  TensorFlow Conv2D 算法探测失败；固定为 cuDNN 9.3.0.75 后，GPU
  MatMul、Conv2D 和 GRACE 的 SP/OPT≤1/NVE MD 1/Batch SP 全部通过。
  GRACE GPU SP 能量为 `-216.129213 eV`，与 CPU 结果一致到约 5 µeV。
  中英文 README 已给出可复现的 `.venv-grace` 安装与固定命令。
- 安装入口已重排：根 `README.md`、`uma/README.md` 及中英文完整手册现在都在
  前部先给出 UMA/MACE/DPA/GRACE 四环境映射、完整安装命令和各自的运行前缀，
  高级 ABI/cuDNN 原理留在后文。`doctor` 也不再向独立 DPA 环境错误推荐 UMA
  的 Torch 2.6；未安装的可选后端显示为 SKIP，而不是容易误解的 WARN。

### 13.5 当前边界

仍未实现且不再伪装成配置开关：完整 checkpoint/restart、run manifest、依赖
队列/resume、GPU 排他锁和安全的多进程 batch。有限但较大的力仍按任务清单要求
只告警；NaN/Inf 始终硬中止。扩散/电导率分析仍由外部分析工具负责。

最终回归命令：

```text
uv run pytest -q tests/core/mlipx tests/uma/mlipx
```

最终结果：**273 passed, 1 skipped**，耗时 381.85 s；唯一 skip 是原有的
“需要真实 checkpoint”的 progress 测试，而真实 checkpoint 已由上面的独立
小任务矩阵覆盖。源代码及相关测试同时通过 `ruff check` 与
`git diff --check`。

---

## 14. TUI / CLI 资源控制复核（2026-07-30）

用户复核文档时发现：手册列出了 CPU 线程、GPU 选择、UMA 推理模式和激活
检查点等控制项，但旧 TUI 没有对应控件，CPU 线程在 CLI 中也只能借助
`OMP_NUM_THREADS`。这属于界面能力与文档承诺不完整，而不是用户漏看。

现已补齐以下链路：

- TUI 的 Device 接受 `cpu`、`cuda`、`cuda:N`，不再只有 CPU/CUDA 二选一；
- TUI 与 CLI 均可设置 CPU 线程数：UMA/MACE/DPA 映射到 PyTorch，
  GRACE 映射到 TensorFlow intra-op 线程；
- UMA 推理模式和激活检查点同时提供 TUI/CLI 入口；
- MACE dtype、MACE/DPA head，以及 OPT 的保持对称性均进入 TUI；
- MD 的 friction、预弛豫步数/阈值、seed、velocity policy 和大力告警阈值
  均进入 TUI；
- TUI 会按引擎禁用不适用控件；公开入口统一为 `--cpu-threads`，旧的
  `--torch-num-threads` 仅作为兼容别名保留。

中英文手册的命令参考、TUI 配置说明和资源控制章节已同步，明确环境变量只是
可选替代方式，不再要求 TUI 用户猜测或手工设置。
