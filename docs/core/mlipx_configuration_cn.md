# mlipx 配置系统（第一阶段）

> **状态：** 第一阶段 · **起于：** mlipx v1.x

本文档介绍 mlipx 第一阶段配置重构中引入的配置系统。涵盖统一默认值模型、
`settings.ini` 文件、分层配置解析器、模型别名、可复用配置模板、MACE 后端支持，
以及 `mlipx config` 命令行工具。

---

## 1. 快速入门

### 1.1 查看当前配置

```bash
mlipx config show
```

输出示例：
```
Resolved configuration (calc_type=md, no CLI overrides):
  settings.ini: (built-in defaults)
  model_type  : uma
  task        : omat
  device      : cuda
  inference_mode: turbo
  calculator_options: {}
  run_options: {ensemble: NVT, temperature: 300.0, steps: 1000, ...}
```

### 1.2 创建 settings.ini

```bash
mlipx config init --project    # 写入 ./settings.ini（项目级）
mlipx config init --user       # 写入 ~/.config/mlipx/settings.ini（用户级）
```

### 1.3 使用模型别名

```ini
; settings.ini
[model:mace_mpa0]
engine = mace
path = ./mace-mpa-0.model
task = bulk
dtype = float32
```

```bash
mlipx sp structure.cif --model-alias mace_mpa0 --dtype float64
```

这里 `--dtype float64` 覆盖了别名中的 `float32` 默认值。
使用 `mlipx config explain default_dtype` 可以追踪最终使用的值来源。

---

## 2. 默认值的唯一来源

所有内置默认值定义在 **唯一位置**（`mlipx/config/defaults.py`）。
每个界面——CLI、Python API、INCAR 模板——都从同一处读取默认值。

| 作用域 (Scope)  | 示例                                  |
|-----------------|---------------------------------------|
| `general`       | `output_root`, `strict_config`        |
| `sp`            | `device=cpu`, `inference_mode=default`|
| `opt`           | `fmax=0.05`, `optimizer=FIRE`         |
| `md`            | `temperature=300`, `inference_mode=turbo` |
| `calculator.mace` | `default_dtype=float32`, `head=None` |

运行 `mlipx config schema` 可查看所有已注册的选项。

---

## 3. 分层解析顺序

解析器按优先级从低到高合并七个层级：

```
1. 内置默认值           （始终存在，优先级最低）
2. settings.ini          （用户级）
3. settings.ini          （项目级，覆盖用户级）
4. 模型别名              （[model:NAME] 节）
5. 配置模板              （[profile:NAME] 节）
6. INCAR / 任务配置      （CALC_TYPE, FMAX … 等键）
7. CLI / API 参数        （--fmax 0.02, --dtype float64 等）
```

每层*追加或覆盖*较早的值——绝不会删除。每个解析后的键都带有**来源追踪**，
记录它来自哪一层。

### 3.1 示例：追踪值的来源

```bash
mlipx config explain temperature
# temperature = 300.0  (source: built-in defaults)

# 加上 --temp 500 后：
mlipx --settings prod.ini sp s.xyz --model-alias mace_mpa0 --temp 500 \
  config explain temperature
# temperature = 500.0  (source: CLI)
```

### 3.2 代码中的来源追踪

```python
from mlipx.config import resolve_config

rc = resolve_config(calc_type="md", cli={"temperature": 500})
print(rc.sources["temperature"])
# ResolvedValue(value=500.0, source='CLI')
```

---

## 4. settings.ini 格式

### 4.1 查找顺序

```
1. --settings PATH        （CLI 参数，优先级最高）
2. MLIPX_SETTINGS 环境变量
3. ./settings.ini         （项目级）
4. ~/.config/mlipx/settings.ini  （用户级）
```

多个文件会被*合并*——项目文件可以覆盖用户文件。
使用 `mlipx config paths` 可查看所有候选路径。

### 4.2 各节参考

```ini
; ── 全局设置 ─────────────────────────────────────────────
[general]
output_root = ./results
strict_config = false
write_resolved_config = true

; ── 各计算类型的默认值 ─────────────────────────────────
[sp]
device = cpu

[opt]
fmax = 0.05
optimizer = BFGS

[md]
temperature = 400
steps = 5000

; ── 模型别名 ────────────────────────────────────────────
[model:mace_mpa0]
engine = mace
path = ./models/mace-mpa-0.model
task = bulk
dtype = float64

[model:uma_prod]
engine = uma
path = /opt/models/uma-s-1.pt
task = omat

; ── 可复用配置模板 ──────────────────────────────────────
[profile:gpu_2x]
device = cuda:1
inference_mode = turbo

[profile:high_precision]
fmax = 0.01
max_steps = 2000
```

---

## 5. 模型别名

**模型别名**是一个命名的 `[model:NAME]` 节，将以下内容打包：

| 键       | 必填 | 说明                                       |
|----------|------|--------------------------------------------|
| `engine` | 是   | `uma` / `mace` / `dpa` / `grace`           |
| `path`   | 是   | 模型文件路径（相对于 settings.ini）         |
| `task`   | 是   | 模型头（`omat`, `bulk`, …）                |
| `dtype`  | 否   | MACE 数据类型（`float32` 或 `float64`）    |
| `head`   | 否   | MACE 基础模型的 head 名称                  |

**使用方式：**
```bash
# 通过 --model-alias 参数
mlipx sp s.xyz --model-alias mace_mpa0

# 通过 --model（当名称匹配别名时的简写形式）
mlipx sp s.xyz --model mace_mpa0

# 通过 Python API
from mlipx.api import run_single_point
results = run_single_point("s.xyz", model_alias="mace_mpa0")
```

---

## 6. 配置模板

**配置模板**是一个 `[profile:NAME]` 节，提供可复用的参数覆盖。
模板在模型别名层*之后*、CLI 参数层*之前*应用。

```ini
[profile:gpu_production]
device = cuda:1
inference_mode = turbo
max_steps = 5000
```

```bash
mlipx opt s.xyz --model-alias uma_prod --profile gpu_production
```

可以对一个模型别名叠加使用模板；当两者定义相同键时，模板优先。

---

## 7. MACE dtype 和 head

### 7.1 完整传播链路

```
settings.ini [model:NAME] dtype
    ↓
resolve_config()  内置默认值 → 别名层 → CLI 层
    ↓
EngineConfig.calculator_options  {"default_dtype": "float64"}
    ↓
factory.build_calculator(..., default_dtype="float64")
    ↓
MACECalculatorWrapper(default_dtype="float64")
    ↓
mace-torch Calculator
```

### 7.2 覆盖 dtype

```bash
# 覆盖别名的默认值
mlipx sp s.xyz --model-alias mace_mpa0 --dtype float64

# 在配置模板中设置
[profile:mace_f64]
dtype = float64
```

### 7.3 Head（基础模型选择器）

```bash
mlipx sp s.xyz --model-alias mace_fm --head "some_head_name"
```

`--head` 参数会传递给 MACE 的基础模型加载过程。

---

## 8. CLI 参考

### 8.1 新增全局参数

| 参数                | 说明                                   |
|---------------------|----------------------------------------|
| `--settings PATH`   | 指定 settings.ini（必须在子命令之前）   |

### 8.2 新增子命令参数（sp / opt / md / batch）

| 参数                  | 说明                        |
|-----------------------|-----------------------------|
| `--dtype {float32,float64}` | MACE 数据类型覆盖           |
| `--head HEAD`         | MACE 基础模型 head 名称          |
| `--model-alias NAME`  | 来自 settings.ini 的模型别名      |
| `--profile NAME`      | 来自 settings.ini 的可复用模板 |

### 8.3 config 子命令

| 命令                              | 说明                              |
|-----------------------------------|-----------------------------------|
| `mlipx config show`               | 显示解析后的配置摘要              |
| `mlipx config paths`              | 列出 settings.ini 搜索路径        |
| `mlipx config init [--project,--user]` | 创建 settings.ini 模板文件   |
| `mlipx config validate [PATH]`    | 验证 settings.ini 文件            |
| `mlipx config explain KEY`        | 追踪某个参数的来源                |
| `mlipx config schema`             | 列出所有已注册的选项键名          |

### 8.4 向后兼容性

所有旧有参数保持不变：
`--model-type`, `--task`, `--device`, `--fmax`, `--max-steps`,
`--optimizer`, `--cell-opt`, `--fix-symmetry`, `--ensemble`, `--temp`,
`--timestep`, `--steps`, `--friction`, `--save-interval`, `--pre-relax`,
`--continue-on-error`, `--pattern`, `--job-name`, `--output`。

INCAR 的 `run` 流程同样保持可用。

---

## 9. Python API

所有 API 函数现在都支持新的配置参数：

```python
from mlipx.api import run_single_point, run_optimization, run_md

# 使用模型别名和 dtype 覆盖
results = run_single_point(
    "structure.cif",
    model_alias="mace_mpa0",
    default_dtype="float64",
    job_name="sp_test",
)

# 使用 settings.ini 和配置模板
results = run_optimization(
    "structure.cif",
    model_alias="uma_prod",
    profile="gpu_production",
    settings_path="./prod-settings.ini",
    fmax=0.02,
)
```

解析器会合并 API 参数、指定模板、模型别名以及所有 settings.ini 文件，
因此你只需要指定要覆盖的内容。

---

## 10. 严格模式

在 settings.ini 中设置 `strict_config = true`（或在支持严格模式的 CLI 命令中
使用 `--strict-config`），可将未知键的警告变为硬错误。在非严格模式下，
每个未识别的键都会通过 difflib 模糊匹配给出「你是否想输入 …？」的警告提示。

---

## 11. 解析后的配置产物

当 `write_resolved_config` 为 true（默认值）时，每次计算都会在输出目录中写入
一个 `resolved_config.json` 文件。可用于审计和调试：

```json
{
  "calc_type": "md",
  "model_type": "mace",
  "device": "cuda",
  "inference_mode": "turbo",
  "calculator_options": {"default_dtype": "float64"},
  "run_options": {"temperature": 300.0, "steps": 1000, "seed": 312904314},
  "sources": {
    "default_dtype": {"value": "float64", "source": "CLI"},
    "temperature": {"value": 300.0, "source": "built-in defaults"},
    "seed": {"value": 312904314, "source": "auto-generated"}
  }
}
```

---

*由第一阶段配置重构生成。详细说明请参阅 `mlipx_稳定性_默认配置_批量队列_MD改进方案.md` 方案文档。*
