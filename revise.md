---

# mlipx 本轮完整修改方案

## 0. 本轮总体目标

这一轮完成后，mlipx 应满足下面这个架构：

```text
mlipx repository
│
├── mlipx/                     # 真正的 Python package project
│   ├── pyproject.toml
│   ├── mlipx/
│   │   ├── install/
│   │   │   ├── compatibility.py
│   │   │   ├── hardware.py
│   │   │   ├── sources.py
│   │   │   └── plan.py
│   │   └── ...
│   ├── templates/
│   └── examples/
│
├── scripts/
│   └── install_mlipx.sh       # 极薄 bootstrap wrapper
│
├── tests/                     # 只保留 mlipx 自己的测试
├── archive/                   # 历史归档，只读、不导入
├── README.md
├── README_CN.md
├── LICENSE.md
├── CITATION.cff
├── CONTRIBUTING.md
└── GitHub CI / tooling
```

最终不应该再存在：

```text
src/fairchem/
packages/fairchem-*
FairChem training configs
FairChem upstream docs
FairChem upstream tests
FairChem release workflows
FairChem devcontainer
FairChem workspace uv.lock
```

核心原则是：

> **mlipx 依赖官方 `fairchem-core`，而不是继续携带一份 FairChem monorepo。**

同时安装系统必须真正做到：

```text
compatibility.py
       ↓
sources.py
       ↓
plan.py
       ↓
installer / mlipx setup / doctor / README
```

不能再出现五份互相漂移的安装逻辑。

---

# 1. P0：彻底修复 UMA 安装链

这是本轮优先级最高的问题。

当前：

```text
compatibility.py:
UMA = fairchem-core 2.21.0
torch = 2.8.0
V100/Pascal/Volta = cu126
```

但是 `plan.py` 实际执行：

```bash
uv sync --frozen
```

而根 `uv.lock` 又锁着：

```text
torch 2.6.0 + cu124
```

所以当前 installer 的 compatibility matrix **并没有真正控制 UMA 安装结果**。

必须彻底取消这种行为。

## 1.1 UMA 必须和其它 backend 一样显式安装

GPU UMA 安装流程应改成：

```text
创建 .venv
↓
根据 compatibility.py 安装指定 torch + CUDA channel
↓
安装 fairchem-core==指定版本
↓
editable 安装 mlipx
↓
doctor
```

例如 V100：

```bash
uv venv --python 3.12 .venv

uv pip install \
    --python .venv/bin/python \
    "torch==2.8.0" \
    --index-url <cu126 index>

uv pip install \
    --python .venv/bin/python \
    "fairchem-core==2.21.0" \
    -e ./mlipx

.venv/bin/mlipx doctor --engine uma --device auto
```

CPU UMA 同理：

```text
torch CPU wheel
+
fairchem-core
+
editable mlipx
```

禁止再出现：

```bash
uv sync
uv sync --frozen
```

作为 UMA runtime 安装手段。

---

# 2. P0：删除 root workspace 对 runtime 的控制

当前根目录 `pyproject.toml` 仍然定义整个 FairChem workspace：

```toml
[tool.uv.workspace]
members = [
    "packages/*",
    "mlipx",
]
```

并且：

```toml
fairchem-core = { workspace = true }
...
torch = { index = "pytorch-cu124" }
```

这些在 FairChem 脱钩之后全部失去意义。

## 推荐做法

最终根 `pyproject.toml` 不要再作为 Python runtime project。

可以保留一个**纯 tooling pyproject**，例如只放：

```text
pytest
ruff
coverage 等配置
```

但删除：

```text
[project]
[tool.uv.workspace]
[tool.uv.sources]
torch override
FairChem dependencies
```

同时删除根：

```text
uv.lock
```

因为这个 lockfile 是 FairChem workspace 的历史产物，不应该再影响 mlipx runtime。

### 很重要

不要重新生成一个新的巨大 root `uv.lock` 来继续控制四套 backend。

四套 backend 本来就是互相冲突的：

```text
UMA
MACE
DPA
GRACE
```

所以继续企图用一个 workspace lock 管四套 runtime，本身就是错误抽象。

---

# 3. P0：真正实现 SourceProfile

现在 `sources.py` 的设计不错，但实际上只有很少一部分被使用。

目前定义了：

```python
pypi_index
pypi_extra_index
pytorch_index
pytorch_find_links
env
```

但真正被 plan 使用的基本只有：

```python
pytorch_index
pytorch_find_links
```

所以当前：

```bash
--source china
```

只影响 torch。

而：

```bash
mace-torch
deepmd-kit
tensorflow
tensorpotential
fairchem-core
mlipx extras
```

仍然走默认 PyPI。

这和 README 写的：

```text
China:
PyPI → TUNA
PyTorch → Aliyun
```

不一致。

---

## 3.1 建立统一 source argument builder

不要在 MACE/DPA/GRACE/UMA 四个分支里面各自拼源参数。

建议做统一 helper，概念上类似：

```python
build_package_source_args(profile)
build_torch_source_args(profile, cuda_channel)
build_offline_args(profile)
build_source_env(profile)
```

所有安装 command 都必须经过这一层。

---

# 4. P0：修复 `--source china`

China 模式要求：

普通 Python package：

```text
fairchem-core
mace-torch
e3nn
deepmd-kit
tensorflow
tensorpotential
mlipx optional dependencies
```

全部真正使用 China PyPI mirror。

torch CUDA wheel 则使用配置的 PyTorch mirror。

不能出现：

```text
torch → Aliyun
MACE → pypi.org
DPA → pypi.org
GRACE → pypi.org
```

这种半切换状态。

---

# 5. P0：`offline` 必须真的 offline

当前：

```python
"offline": SourceProfile(
    env={"UV_NO_BUILD": "1"},
)
```

但这些 env 根本没传入 InstallStep。

而实际生成的 offline plan 里还存在：

```text
https://download.pytorch.org/...
```

所以当前：

```bash
--source offline
```

完全不是 offline。

必须定义严格语义：

```text
offline =
不访问任何网络
只允许本地 uv/pip cache / wheel cache
```

验收标准非常简单：

> 对 `generate_plan(source="offline")` 产生的所有安装 step 做字符串检查，不能包含任何 `http://` 或 `https://`。

并且应该使用 uv 的 offline/cache-only 机制，而不是 `UV_NO_BUILD` 这种完全不同含义的变量。

`SourceProfile.env` 必须真正传播到每个对应 InstallStep。

---

# 6. P0：`custom` 必须真正继承用户环境变量

README 现在说：

```text
custom → your env vars
```

那么就应该允许：

```text
UV_INDEX_URL
UV_EXTRA_INDEX_URL
UV_FIND_LINKS
```

等用户配置进入安装进程。

不要重新覆盖。

同时保留：

```text
不修改 ~/.config/uv/uv.toml
```

这一设计，这是好的。

---

# 7. P0：修复 `--clean`

当前 shell：

```bash
--clean)
    CLEAN=1
```

之后 `CLEAN` 再没被用到。

属于确定 bug。

建议把 clean 变成 plan 参数：

```python
generate_plan(
    ...,
    clean=True,
)
```

如果 clean：

每个目标 engine 在创建 venv 前增加：

```text
remove environment
```

例如：

```text
.venv
.venv-mace
.venv-dpa
.venv-grace
```

但必须有安全限制：

只能删除 compatibility matrix 中声明的已知 venv 路径。

不要写成任意：

```bash
rm -rf "$SOME_USER_STRING"
```

避免以后参数失误删目录。

---

# 8. P0：修复 `--skip-doctor`

当前：

```bash
--skip-doctor
```

也只是设置变量：

```bash
RUN_DOCTOR=0
```

但 plan 仍然无条件加入：

```text
verify
```

step。

应该改成：

```python
generate_plan(
    ...,
    verify=False,
)
```

当：

```text
verify=False
```

时：

```text
InstallPlan.steps
```

里面根本不能出现：

```text
stage == "verify"
```

这样 dry-run 显示的计划和真正执行行为才能完全一致。

---

# 9. P0：`--device cuda` 必须 fail closed

我实际调用了：

```python
generate_plan(
    gpus=None,
    engines=["mace"],
    device="cuda",
)
```

当前结果居然是：

```text
CPU installation
```

而且没有 warning。

这非常危险。

用户明确要求：

```bash
--device cuda
```

意味着：

> 我要 CUDA 环境。

如果没有 GPU，却悄悄变成 CPU 环境，这是 installer 最不应该做的事情。

应该定义：

```text
device=auto + 无GPU
→ CPU，正常

device=cpu
→ CPU，正常

device=cuda + 无GPU
→ ERROR

device=cuda + unsupported GPU
→ ERROR

device=auto + unsupported GPU
→ 可以 CPU fallback，但必须明确 warning
```

这和 mlipx 一贯的：

> fail closed

理念也是一致的。

---

# 10. P0：修复多 GPU 检测

Python：

```python
detect_gpus()
```

本身支持多 GPU。

而且 compatibility 逻辑正确地：

```python
_pick_oldest()
```

选择最低 compute capability 的 GPU。

这是合理的保守策略。

但 `install_mlipx.sh` 自己又重新实现了一套检测，而且：

```bash
... | head -n1
```

只读取第一张 GPU。

于是：

```text
GPU0 = RTX 4090
GPU1 = P100
```

Python planner 理论上应该选择 Pascal → cu126。

shell installer 实际却只看到 4090 → cu128。

这会直接破坏 multi-GPU compatibility。

## 正确做法

shell 不应该自己解析 `nvidia-smi`。

直接让 Python：

```python
detect_gpus()
generate_plan()
```

完成。

也就是说删掉 shell 里的：

```text
CC_RAW
GPU_NAME
GPU_DRIVER
GPU_VRAM
head -n1
GPU_JSON 手动拼接
```

---

# 11. 强烈建议：进一步把 installer shell 变成真正的 thin wrapper

现在文件头自己写：

> All logic lives in Python.

但实际上 shell 仍然负责：

```text
GPU detection
JSON construction
执行器
环境变量处理
Python inline code
```

这已经不算 thin wrapper。

建议最终：

```text
scripts/install_mlipx.sh
```

只负责：

```text
1. 确认 uv 存在
2. 确定 repo root
3. 调用 Python installer entry point
```

例如设计一个：

```text
mlipx.install.cli
```

或者：

```text
python -m mlipx.install
```

负责：

```text
arg parsing
GPU detection
plan generation
dry-run rendering
execution
```

这样 shell 可以缩到几十行。

---

# 12. 一个对 Rocky Linux 非常-run rendering

execution

````

这样 shell 可以缩到几十行。

---

# 12. 一个对 Rocky Linux 非常重要的问题：不要依赖系统 `python3`

现在 installer：

```bash
need_cmd python3
````

然后直接：

```bash
python3 -c ...
```

导入：

```text
mlipx.install.*
```

但是 mlipx 要求：

```text
Python >=3.10
```

而 Rocky Linux 9 系机器非常可能系统 `/usr/bin/python3` 仍然较老。

更关键的是，install 模块本身用了：

```python
str | None
tuple[int, int]
```

等现代 Python 语法。

于是完全可能出现：

```text
用户有 uv
用户要求 --python 3.12
uv 本来能自动提供 3.12
↓
installer 却先拿系统旧 python3 import install module
↓
SyntaxError
↓
还没创建 3.12 venv 就死了
```

这是 bootstrap 逻辑上的问题。

### 建议

既然 prerequisite 已经是 uv，就让 uv 提供 planner 所需 Python。

也就是说 installer bootstrap 最好只要求：

```text
bash
uv
```

而不是要求宿主机 Python 本身已经满足 mlipx runtime。

---

# 13. 删除没用的 `git` prerequisite

installer 现在：

```bash
need_cmd git
```

但 clone 完以后安装过程中没有看到实际需要 git 的地方。

如果 installer 不执行 git 操作，就删除这个要求。

减少不必要依赖。

---

# 14. P1：不要继续用 `shell=True`

当前：

```python
subprocess.run(
    step["command"],
    shell=True,
)
```

同时 InstallStep：

```python
command: str
```

这让：

```text
路径 quoting
用户参数
URL
Python 路径
shell escaping
```

全部变复杂。

推荐改成：

```python
@dataclass
class InstallStep:
    argv: list[str]
```

例如：

```python
[
    "uv",
    "pip",
    "install",
    "--python",
    ".venv-mace/bin/python",
    "torch==2.8.0",
    "--index-url",
    "...",
]
```

执行：

```python
subprocess.run(
    step.argv,
    shell=False,
    ...
)
```

dry-run 展示时再：

```python
shlex.join(step.argv)
```

这会显著提升 installer 的可靠性。

---

# 15. shell 当前还有字符串注入/escaping 脆弱点

现在 shell 把：

```text
GPU_JSON
ENGINES
SOURCE
PYVER
DEVICE
```

直接插入 Python `-c` 字符串。

例如：

```python
json.loads('''$GPU_JSON''')
```

虽然正常 NVIDIA GPU 名一般不会乱来，但这种设计没有必要。

如果按上一条做成 Python-native installer，这些问题会自然消失。

---

# 16. InstallPlan 应区分 warnings 和 errors

现在未知 engine：

```python
plan.warnings.append(...)
continue
```

这太宽松。

对于 installer 这种工具：

```text
未知 engine
空 engine list
明确 cuda 但无 GPU
非法 Python
非法 source
无法解析架构
```

很多情况应该是 error，而不是继续生成一个部分 plan。

可以：

```python
class InstallPlanError(Exception):
    ...
```

或者：

```python
InstallPlan.errors
```

我更倾向直接 raise 明确异常，因为安装前这些都是配置错误。

---

# 17. `--engines` 应做 normalize + 去重

例如：

```bash
--engines uma,mace,uma
```

现在会重复产生 UMA install steps。

应该：

```text
strip
lower
fairchem → uma
deduplicate preserving order
```

并拒绝：

```text
空列表
```

---

# 18. 校验 Python 版本

现在：

```bash
--python 3.12
```

实际上可以传任意字符串。

至少明确支持：

```text
3.10
3.11
3.12
```

因为项目 metadata 明确：

```text
>=3.10,<3.13
```

传：

```bash
--python 3.13
```

应该在 plan generation 阶段立即报错，而不是安装到中途才发现 package metadata 不允许。

---

# 19. 定义已有 venv 的行为

现在没有明确说明：

```bash
./install_mlipx.sh
```

遇到已经存在的：

```text
.venv
.venv-mace
...
```

到底是什么语义。

建议正式定义：

默认：

```text
reuse/update existing venv
```

`--clean`：

```text
delete + recreate
```

如果已有 venv Python 版本与：

```bash
--python
```

不一致：

默认 fail，并提示：

```bash
--clean
```

不要偷偷换解释器。

---

# 20. Compatibility matrix：当前“单一事实来源”还可以继续完善

`compatibility.py` 的方向很好，我建议保留。

但现在存在一个小的设计矛盾。

你定义了：

```python
_torch_modern_cuda()
select_cuda_channel()
```

理论上：

```text
architecture
+
torch version
↓
CUDA channel
```

可是 `BACKENDS` 每一个 arch entry 又手写：

```python
cuda_channel="cu126"
cuda_channel="cu128"
```

所以实际上还是重复数据。

### 建议

让 torch backend 默认根据：

```text
architecture + framework version
```

自动 derive CUDA channel。

只有特殊情况才 override。

例如概念上：

```python
cuda_channel: str | None = None
```

如果没写：

```python
select_cuda_channel(...)
```

自动算。

这样以后：

```text
torch 2.13
torch 2.14
cu129
cu130
cu13x
```

变化时只需要改中央 mapping。

---

# 21. `CudaChannel.sm_min/sm_max` 目前几乎只是装饰数据

既然已经定义：

```python
sm_min
sm_max
```

就应该加 invariant tests。

例如：

```text
backend arch profile 指向 cuXXX
↓
该 architecture 必须落在 channel 支持 SM 范围
```

否则这些字段没有实际保护作用。

---

# 22. Blackwell classification 有一个逻辑漏洞

现在：

```python
blackwell:
    cc_min=(10,0)
    cc_max=(12,0)
```

而判断：

```python
cc_min <= cc <= cc_max
```

意味着：

```text
11.0
11.5
10.8
```

都会被自动分类为 Blackwell。

但 `cc_arch_name()` 又只认：

```python
major in (10, 12)
```

于是两套 classification 已经不完全一致。

建议不要用一个连续 range 跨过整个 major 11。

可以把 architecture profile 改成：

```text
一个或多个 CC range
```

或者显式 known CC set。

原则：

> 对未知未来 compute capability，宁可显示 unknown / needs update，也不要自动猜成 Blackwell。

---

# 23. `cc_arch_name()` 和 `classify_gpu()` 不应该维护两套 GPU 分类逻辑

当前：

```text
hardware.py::cc_arch_name
compatibility.py::ARCH_PROFILES
```

都在定义 architecture。

应该让：

```python
cc_arch_name()
```

优先调用：

```python
classify_gpu()
```

然后返回 profile.label。

Kepler 等 unsupported GPU 可以单独识别。

减少未来新增架构时漏改一处。

---

# 24. Backend package version 不要重复写

例如 MACE：

```python
distribution="mace-torch"
version="0.3.16"
```

同时：

```python
install_extra=(
    "e3nn==0.4.4",
    "mace-torch==0.3.16",
)
```

版本写了两遍。

DPA：

```text
version=3.1.3
deepmd-kit==3.1.3
```

也是两遍。

GRACE 同样。

建议：

```python
BackendSpec.distribution
BackendSpec.version
```

负责自动生成：

```text
distribution==version
```

而：

```python
install_extra
```

只保存真正额外依赖，例如：

```text
e3nn==0.4.4
```

这样不会发生：

```text
version 改成 0.3.17
install_extra 忘了改仍然 0.3.16
```

---

# 25. compatibility matrix 要增加内部一致性测试

至少自动验证：

```text
BACKENDS 的 engine key 与 backend.engine 一致

每个 backend 的 version 非空

torch backend framework version 非空

torch backend CUDA channel 可解析

每个 arch_profile key 都存在于 ARCH_PROFILES

distribution/version 与安装 requirement 一致

status derivation 正确

mlipx_verified=True 必须 upstream_supported=True

experimental GPU 不应误显示 verified

CUDA channel 支持对应 SM
```

---

# 26. `gpu_setup.py` 不应该继续拥有第二套 command generator

文件头已经写：

> legacy public API
> delegates to mlipx.install

但现在：

```python
_cpu_commands()
_gpu_commands()
engine_install_commands()
```

仍然自己拼安装命令。

所以当前实际上是：

```text
compatibility.py
plan.py
gpu_setup.py
root pyproject
uv.lock
README
```

六套安装事实。

### 修改

保留：

```text
gpu_setup.py
```

作为 backward compatibility facade 没问题。

但：

```python
engine_install_commands()
```

应该内部：

```python
generate_plan(
    engines=[engine],
    ...
)
```

然后把 plan 渲染成人类可复制 command。

不能自己再实现一次安装逻辑。

---

# 27. `mlipx setup` 应与新 installer 完全一致

现在：

```text
mlipx setup
```

依赖 `gpu_setup.py` 的第二套逻辑。

改完以后应该：

```text
compatibility.py
↓
generate_plan()
↓
setup report
```

这样：

```bash
mlipx setup
```

和：

```bash
./scripts/install_mlipx.sh --dry-run
```

对于同一个 GPU 给出的版本必须完全一致。

可以直接加 regression test 比较二者。

---

# 28. CPU-only 下 `mlipx setup` 不应该返回 exit code 1

现在：

```python
return 0 if gpus else 1
```

但：

```text
无 NVIDIA GPU
```

并不是 mlipx setup 失败。

因为 CPU 是正式支持的。

建议：

```text
CPU-only detection/report → exit 0
```

只有真正的 command/configuration failure 才非零。

这样在无 GPU CI 或 CPU 节点运行：

```bash
mlipx setup
```

不会被错误判断为失败。

---

# 29. doctor 中清理所有旧 workspace 假设

`doctor.py` 目前还有很多历史文字：

```text
Run `uv sync --frozen`
workspace override
workspace recommendation
Torch 2.6
```

这些在新架构下全部不应该存在。

例如：

```python
_ENGINE_SPECS["uma"]["install"]
```

应变成：

```text
Run the mlipx installer for UMA
```

或者提供当前 matrix 对应命令。

所有：

```text
workspace recommendation
```

改成：

```text
mlipx compatibility matrix recommendation
```

---

# 30. doctor 应直接依赖 `mlipx.install`

长期建议：

```text
doctor.py
```

不要通过 legacy：

```text
gpu_setup.py
```

取 compatibility 信息。

应该直接：

```text
mlipx.install.compatibility
mlipx.install.hardware
```

然后：

```text
gpu_setup.py
```

仅用于兼容旧 API。

这样依赖方向会更干净：

```text
install/
↑
doctor
↑
gpu_setup legacy facade
```

而不是循环引用旧层。

---

# 31. doctor 可以增加 recommended version 对比

已经安装某 engine 时，可以显示：

```text
Installed:
fairchem-core 2.xx
torch 2.xx

mlipx recommended:
fairchem-core 2.21.0
torch 2.8.0 + cu126

Status:
OK / warn
```

但不要因为用户用了一个同样可工作的 patch version 就直接 fatal。

版本偏离 matrix：

```text
warn
```

runtime/kernel 不兼容：

```text
fail
```

这个区分比较合理。

---

# 32. 顶层 `src/`：最终删除

完成 UMA 外部依赖迁移以后：

```text
src/
```

整目录删除。

我搜索了当前 mlipx 代码，真正的：

```python
import fairchem
```

发生在：

```text
mlipx/mlipx/calculator.py
```

它依赖的是 Python package namespace：

```text
fairchem.core
```

并不要求这些源码必须来自 repo 顶层 `src/`。

因此外部 `fairchem-core` 正确安装后，顶层源码就没有存在意义。

---

# 33. `packages/` 同时删除

不要只删：

```text
src/
```

却留下：

```text
packages/
```

两者是同一份 FairChem monorepo 遗产。

全部删除：

```text
packages/fairchem-core
packages/fairchem-data-*
packages/fairchem-applications-*
packages/fairchem-lammps
packages/fairchem-demo-*
...
```

---

# 34. 顶层 `docs/` 基本可以整目录删除

当前约：

```text
8.5 MB
93 files
```

主要是：

```text
OC20
ODAC
OMat
OMol
UMA training
FairChem tutorials
FairChem API docs
```

而 `mlipx/README.md` 自己甚至写着：

```text
standalone docs removed
content merged into root README
```

说明它们明显属于遗留文件。

建议整个删除。

将来 mlipx 真需要 documentation site，再从零建立：

```text
docs/
```

不要继续背着 FairChem 的 docs tree。

---

# 35. 顶层 `configs/` 整目录删除

现在是：

```text
FairChem / UMA training
benchmark
finetune
EScAIP
evaluation
```

这些和 mlipx 的目标：

```text
MLIP inference / MD
```

无关。

而且你之前就已经明确不想让 mlipx 变成模型训练大合集。

删除。

---

# 36. 根 `tests/` 清掉所有 FairChem tests

当前 tests 大量是：

```text
tests/core
tests/data
tests/demo
tests/lammps
tests/perf
```

这些是 FairChem upstream tests。

真正 mlipx 是：

```text
tests/mlipx/mlipx/
```

应该删除其它 upstream tests。

---

# 37. 可以顺便把 tests 目录扁平化

现在：

```text
tests/mlipx/mlipx/test_xxx.py
```

有点历史包袱。

如果 Codex 可以安全完成，我建议整理成：

```text
tests/
├── test_cli.py
├── test_doctor.py
├── test_install_plan.py
├── test_install_sources.py
├── test_gpu_setup.py
├── ...
```

没有必要再：

```text
tests/mlipx/mlipx/
```

套两层。

不过这属于 P2。

如果担心一次移动太多文件，也可以暂时保持。

---

# 38. 删除 `tests/requirements.txt` 等 FairChem 测试遗物

清理 FairChem tests 时一起检查：

```text
tests/requirements.txt
tests/conftest.py
```

尤其当前根：

```text
tests/conftest.py
```

会 import：

```python
ray
```

我之前执行 mlipx tests 时就是被它先拦住的。

这说明它已经实际污染 mlipx test discovery。

---

# 39. `.github` 应该基本重建，而不是逐个修 upstream workflow

当前有大量：

```text
release-drafter-core
release-drafter-data-omat
release-drafter-omol
release-drafter-lammps
integration-test
FairChem build docs
FairChem perf
...
```

我的建议不是慢慢筛。

而是：

> 删除现有 FairChem workflow，只重新建立 mlipx 真正需要的几个。

---

# 40. 推荐只保留 3 类 CI

### CI 1：Python tests

支持：

```text
Python 3.10
Python 3.11
Python 3.12
```

执行：

```text
install mlipx dev dependencies
pytest
```

不要默认安装四个大型 ML backend。

backend 相关测试尽量 mock。

---

### CI 2：lint

运行：

```text
ruff check mlipx/mlipx tests
ruff format --check mlipx/mlipx tests
```

---

### CI 3：package build smoke

执行：

```text
build wheel
build sdist
安装 wheel 到干净环境
mlipx --help
mlipx setup --json
```

CPU CI 下：

```text
mlipx setup
```

必须 exit 0。

---

# 41. 当前 ruff 配置实际上还在 lint FairChem，而不是 mlipx

根：

```text
ruff.toml
```

现在：

```toml
include = [
    "src/fairchem/core/**/*.py",
    "src/fairchem/data/oc/**/*.py",
    "tests/**/*.py",
]
```

这意味着核心：

```text
mlipx/mlipx/
```

根本不在主要 lint target 里。

必须修。

建议直接重写 root ruff config：

```text
mlipx/mlipx/**/*.py
tests/**/*.py
scripts 的 Python 文件
```

删除所有：

```text
src/fairchem
fairchem.core known-first-party
FairChem per-file-ignore
```

---

# 42. `target-version = "py39"` 也应该修

`mlipx/pyproject.toml` 当前：

```toml
target-version = "py39"
```

而 package 自己：

```text
requires-python >=3.10
```

不一致。

建议 Ruff target：

```text
py310
```

因为 3.10 是最低正式支持版本。

---

# 43. `.devcontainer` 是纯 FairChem 遗留

当前里面会：

```text
install packages/fairchem-core
install fairchem-data
install cattsunami
convert FairChem docs
start Jupyter
```

如果你目前不使用 devcontainer：

**直接删除。**

不要为了“看起来项目功能多”保留没维护的东西。

---

# 44. `postBuild` 同样删除

当前是：

```text
FairChem hosted docs build
```

已经与 mlipx 无关。

删除。

---

# 45. Dependabot 配置必须修或者删除

现在 `.github/dependabot.yml` 里面的 assignees 仍然是 FairChem upstream 开发者。

这个非常明显属于 fork 遗留。

如果暂时不维护 dependabot：

直接删。

如果保留：

```text
只追踪 GitHub Actions
mlipx Python project
```

并删除 upstream assignee。

---

# 46. GitHub Issue template 也需要 mlipx 化

现在 bug report：

```text
description: FAIR-Chem bug report
```

而且强制：

```text
fairchem-core version
torch version
```

这对于：

```text
MACE
DPA
GRACE
```

都不合理。

改成：

```text
mlipx version
engine
Python version
GPU
driver
framework version
model type/task
minimal reproduction
doctor output
```

尤其推荐用户直接附：

```bash
mlipx doctor --engine ...
```

输出。

---

# 47. `setup.py` 可以删除

`mlipx/setup.py` 只是：

```python
from setuptools import setup
setup()
```

现代：

```toml
[build-system]
build-backend = "setuptools.build_meta"
```

已经不需要它。

删掉可以继续减少 FairChem fork 痕迹。

---

# 48. package metadata 应完善

`mlipx/pyproject.toml` 目前 metadata 很少。

可以补：

```text
readme
license
project.urls
repository
issues
Python classifiers
```

不要乱填作者姓名。

实际作者信息由项目维护者自己确定。

---

# 49. 版本号应该尽量只有一个 source of truth

现在：

```text
mlipx/pyproject.toml → 2.0.0
mlipx/__init__.py → 2.0.0
CITATION.cff → 2.0.0
```

至少 Python runtime 可以改成：

```python
importlib.metadata.version("mlipx")
```

这样：

```text
pyproject.toml
```

成为正式 package version source。

避免 release 时忘改：

```python
__version__
```

CITATION release version 可以在发布流程中同步。

---

# 50. LICENSE 两份现在不完全一致

根：

```text
LICENSE.md
```

目前基本只有：

```text
Copyright Meta
```

但：

```text
mlipx/LICENSE
```

已经写成：

```text
mlipx contributors
Portions Copyright Meta
```

更符合当前项目实际情况。

建议最终建立一份 canonical license notice。

保留：

```text
Meta 对衍生 FairChem 代码的原版权声明
```

同时明确：

```text
mlipx 新代码的版权归 mlipx contributors
```

不要批量删除衍生代码文件中的 Meta notice。

但对于这一轮新写的：

```text
install/
```

如果确实是 mlipx 新代码而不是 FairChem 原文件修改，也没有必要假装所有代码都是 Meta 写的。

这一块要谨慎做版权归属整理，不要简单全局 search/replace。

---

# 51. CITATION.cff 需要重新审视

现在：

```text
mlipx authors
```

列的是一大批 FairChem/UMA upstream contributors。

这会造成学术引用语义混乱：

> 他们是 FairChem 作者，不等于 mlipx 软件作者。

建议：

```text
authors:
    实际 mlipx 作者/维护者
```

FairChem 则在：

```text
README
LICENSE attribution
references
```

里明确注明基础来源。

另外当前 DOI 也应由维护者确认确实属于 mlipx release，不要让 Codex凭空修改 DOI。

---

# 52. `.gitignore` 可以大幅瘦身

当前还保留很多：

```text
FairChem docs
training
LMDB
wandb
checkpoint
各种旧 tutorial
```

其中不少已经与 mlipx 不相关。

这一轮删掉 FairChem tree 后可以顺便整理。

但保留与你实际工作相关的：

```text
.venv*
models/
*.pt
*.pth
results/
OUTPUT/
raw trajectory
LGPS*
私人科研目录
```

尤其你那些 private/research ignore 规则不要误删。

---

# 53. README：UMA 安装命令必须重写

删掉：

```bash
uv sync --frozen
```

变成和 installer 相同的显式方式。

而且手动命令最好明确写：

> 以下只是某个 architecture 的示例，真正推荐使用 `--dry-run` 获取本机对应命令。

因为 README 当前写：

```text
MACE torch 2.8/cu126
DPA torch 2.10/cu126
```

4090 用户照抄就会被误导。

---

# 54. README 不再推荐 `uv run mlipx` 作为 UMA runtime

如果 root workspace 被移除后：

```bash
uv run mlipx
```

的行为会变得不再可靠/直观。

建议统一：

```bash
.venv/bin/mlipx
```

或者先：

```bash
source .venv/bin/activate
```

然后：

```bash
mlipx
```

四套环境统一：

```text
UMA   → .venv/bin/mlipx
MACE  → .venv-mace/bin/mlipx
DPA   → .venv-dpa/bin/mlipx
GRACE → .venv-grace/bin/mlipx
```

非常清晰。

---

# 55. README GPU Status 表当前明显过度承诺

当前 README：

```text
Pascal    Verified
Volta     Verified
Turing    Verified
Ampere    Verified
Ada       Verified
Hopper    Verified
Blackwell Verified
```

但 compatibility matrix 里真正：

```python
mlipx_verified=True
```

主要是 Volta/V100。

其它很多只是：

```text
upstream_supported
+
needs_smoke_test
```

因此 README 应与 compatibility matrix 对齐。

最好不要给“GPU family”一个单一 Verified 状态，因为：

```text
UMA × Pascal
MACE × Pascal
DPA × Pascal
GRACE × Pascal
```

是四个不同组合。

推荐：

硬件表只显示：

```text
Architecture
CC
legacy/modern route
```

另外给 engine compatibility/status 表。

例如：

```text
Volta/V100:
UMA   verified
MACE  verified
DPA   verified
GRACE verified

Pascal:
UMA   needs smoke test
...
```

---

# 56. README Arrhenius 示例是确定错误

现在：

```bash
--temperature 600,700,800
```

但 argparse：

```python
type=float
action="append"
```

必须写：

```bash
--temperature 600 \
--temperature 700 \
--temperature 800
```

同理：

```bash
--diffusivity 1e-10 \
--diffusivity 2e-10 \
--diffusivity 5e-10
```

以及：

```bash
--diffusivity-std ...
```

英文、中文 README 同时修。

---

# 57. README Background Queue 英文版有整段重复

英文：

```text
Background Jobs & Queue
```

后面的 JSON/submit 示例基本出现两遍。

删掉重复部分。

中文虽然没有整段重复，但：

```markdown
## 后台任务与队列
### 后台任务与队列
```

标题重复。

删一个。

---

# 58. `mlipx/README.md` installer 路径错误

该 README 位于：

```text
repo/mlipx/README.md
```

却写：

```bash
./scripts/install_mlipx.sh
```

从这个目录执行并不存在。

最好不要假设 cwd。

直接写：

```text
From repository root:
./scripts/install_mlipx.sh
```

比写：

```bash
../scripts/...
```

更清楚。

---

# 59. README Development 完全需要更新

现在：

```bash
uv pip install -e './mlipx[dev]'
uv run pytest tests/mlipx -q
```

在新架构下建议专门开发环境：

```bash
uv venv --python 3.12 .venv-dev

uv pip install \
    --python .venv-dev/bin/python \
    -e './mlipx[dev]'

.venv-dev/bin/python -m pytest tests -q
```

这样开发环境不会与：

```text
.venv = UMA runtime
```

冲突。

---

# 60. README Development 中这句必须删

现在：

> the fairchem fork is under `packages/` and `src/`

完成脱钩后自然不存在。

应该改成：

```text
UMA is consumed through the external fairchem-core dependency.
```

---

# 61. Analysis optional dependencies 需要写清楚

当前 base package：

```text
ase
numpy
packaging
tqdm
textual
```

没有：

```text
scipy
matplotlib
kinisi
gemdat
```

但是 README 直接大量使用：

```text
mlipx analyze
transport
plots
electrolyte
```

用户很容易安装四个 engine 后发现分析功能缺包。

建议明确：

基础分析：

```bash
uv pip install \
    --python .venv/bin/python \
    -e './mlipx[analysis]'
```

transport：

```bash
-e './mlipx[transport]'
```

或者一次：

```bash
-e './mlipx[analysis-all]'
```

如果 GEMDAT 不在 `analysis-all` 的最终定义中，则额外：

```text
electrolyte
```

README 必须和 pyproject extras 完全一致。

不建议默认给 installer 安一大堆分析依赖。

installer 的职责保持：

> backend runtime

分析包明确 opt-in 即可。

---

# 62. `mlipx[uma]` extra 要处理版本漂移问题

现在：

```toml
uma = ["fairchem-core"]
```

这意味着用户：

```bash
pip install mlipx[uma]
```

可能装到未来最新 fairchem-core，而 compatibility matrix 却仍写：

```text
2.21.0
```

这里有两种合理选择。

我更倾向：

```text
installer = 正式 GPU-aware 安装方式
```

而 `uma extra` 要么：

```text
pin 到 matrix 推荐版本
```

要么干脆删除这个容易误导的 extra。

如果保留 pin：

```text
fairchem-core==...
```

那就写单元测试确保：

```text
pyproject UMA extra version
==
BACKENDS["uma"].version
```

避免两处漂移。

---

# 63. `pre-commit` 应确保真的检查 mlipx source

当前 pre-commit 可以保留：

```text
ruff
ruff-format
trailing whitespace
EOF
large files
```

但调整完 ruff include 后，需要确保：

```text
mlipx/mlipx/
tests/
```

确实进入检查范围。

---

# 64. 安装模块必须新增真正的 unit tests

这是这一轮非常重要的一部分。

至少新增：

```text
test_install_plan.py
test_install_sources.py
test_install_compatibility.py
```

---

# 65. Install plan 必测：V100

模拟：

```text
V100
CC 7.0
```

UMA 应包含：

```text
torch 2.8.0
cu126
fairchem-core 2.21.0
editable mlipx
```

且：

```text
不能包含 uv sync
```

MACE：

```text
torch 2.8
cu126
```

DPA：

```text
torch 2.10
cu126
```

GRACE：

```text
TF 对应版本
tensorpotential
```

---

# 66. 必测：4090

Ada：

```text
UMA → modern
MACE → modern
DPA → modern
```

且状态：

```text
needs_smoke_test
```

除非以后真的完成真实硬件 smoke test。

---

# 67. 必测：mixed GPU

模拟：

```text
4090
+
P100
```

最终 plan 必须按：

```text
Pascal
```

选择保守兼容路线。

还要增加 installer integration test，防止 shell 又只读 GPU0。

---

# 68. 必测：无 GPU

```text
device=auto
```

必须：

```text
CPU plan
success
```

---

# 69. 必测：明确 cuda + 无 GPU

必须：

```text
raise / fail
```

绝不能 CPU fallback。

---

# 70. 必测：Kepler

`device=auto`：

```text
warning
CPU fallback
```

`device=cuda`：

```text
error
```

---

# 71. 必测：China source

对每个 engine 检查。

普通包安装 command 必须体现 PyPI China mirror。

torch 必须体现对应 PyTorch mirror。

特别检查：

```text
GRACE
fairchem-core
mace-torch
deepmd-kit
```

不能漏掉。

---

# 72. 必测：offline

所有 command：

```python
assert "http://" not in command
assert "https://" not in command
```

并确保 offline mode 真正传递。

---

# 73. 必测：custom

模拟：

```text
UV_INDEX_URL
UV_EXTRA_INDEX_URL
UV_FIND_LINKS
```

确保 plan execution environment 中被保留。

---

# 74. 必测：clean

```python
clean=True
```

必须在每个目标 engine 前产生 clean action。

如果只：

```text
engines=["mace"]
```

只能清：

```text
.venv-mace
```

不能清：

```text
.venv
.venv-dpa
.venv-grace
```

---

# 75. 必测：skip doctor

```python
verify=False
```

最终：

```python
assert all(step.stage != "verify")
```

---

# 76. 必测：未知 engine

例如：

```text
foo
```

必须 error。

不要：

```text
warning + silently skipped
```

---

# 77. 必测：duplicate engine

```text
uma,mace,uma
```

最终只产生一次 UMA plan。

---

# 78. 必测：非法 Python

```text
3.9
3.13
```

应该提前拒绝。

---

# 79. 必测：InstallStep 不允许 shell string 执行

如果采用：

```python
argv: list[str]
```

就加测试确保：

```text
shell=False
```

执行器不会重新拼 shell command。

---

# 80. 必测：compatibility matrix invariants

前面提到的：

```text
framework
version
architecture
CUDA channel
status
SM range
```

全部做纯 Python unit tests。

这些测试基本毫秒级，不需要真实 GPU。

---

# 81. README 最好加入 command consistency regression

不需要做复杂 README parser。

但是可以至少保证：

```text
README 里不再出现：
uv sync --frozen
```

以及脱 FairChem 后：

```text
非 archive 目录不得出现：
packages/fairchem
src/fairchem
```

可以作为 repository hygiene test。

---

# 82. CI 不需要真的下载四套巨大 GPU backend

千万不要为了“测试完整”让 GitHub Actions：

```text
安装 TensorFlow + Torch + FairChem + MACE + DeepMD
```

既慢又不稳定。

安装 planner 本身就是 pure Python。

大部分 test 应 mock runtime。

真正硬件验证单独做 manual smoke。

---

# 83. 建立真实硬件 smoke-test 定义

既然 matrix 有：

```python
mlipx_verified
```

就应该明确什么叫 verified。

建议定义最少：

```text
1. backend import
2. model load
3. 一个真实 single-point
4. 一个极短 MD
5. energy/forces finite
6. 无 kernel error
7. 无 obvious dependency conflict
```

只有实际通过上述流程，才能：

```python
mlipx_verified=True
```

不能仅因为：

```text
upstream theoretically supports
```

就标 Verified。

---

# 84. 你现在 V100 可以作为 reference platform

当前 matrix 把 Volta 标：

```text
verified
```

这个可以继续。

以后你如果真的在：

```text
P100
3080Ti
4090
```

等卡上跑过相同 smoke script，再改对应状态。

这样 compatibility table 会非常可信。

---

# 85. Maxwell 保持 experimental 是合理的

这一轮不需要为了“支持更多老卡”去 source-build PyTorch/TF。

保持：

```text
best effort
experimental
```

就很好。

不要让 installer 自动编译 PyTorch 或 TensorFlow。

那会让安装系统膨胀得离谱。

---

# 86. 不要在这一轮添加未来 CUDA 自动联网探测

`auto source` 当前：

```text
auto → official
```

很简单，也没什么问题。

暂时不需要增加：

```text
自动测速 TUNA
自动判断中国网络
自动探测 PyPI latency
自动下载测速
```

这些不是 mlipx 的核心价值。

README 写清：

```text
auto currently resolves to official
```

即可。

---

# 87. 不要在这一轮实现自动系统 CUDA Toolkit 管理

mlipx 应安装：

```text
Python wheels/runtime libraries
```

不要开始：

```text
apt install CUDA
yum install driver
安装 NVIDIA driver
编译 toolkit
```

用户系统级 CUDA 和驱动是另一个层次。

doctor 可以诊断。

installer 不负责系统管理员工作。

---

# 88. 不要重新把四套 engine 合并进一个环境

当前：

```text
UMA
MACE
DPA
GRACE
```

隔离环境这个设计我非常赞成。

尤其：

```text
UMA e3nn
MACE e3nn
DPA torch
GRACE tensorflow
```

冲突明显。

不要为了“安装简单”重新尝试：

```text
一个 .venv 安所有 backend
```

---

# 89. `archive/` 保留

当前：

```text
archive/
```

只有约 180 KB。

而且 AGENTS.md 已经明确：

> historical reference only
> must never be imported

很好。

不需要删。

继续确保：

```text
production import graph
```

永远不能引用 archive。

---

# 90. README/代码里的安装命令最终应统一成一个清晰模型

我推荐用户最终看到的是：

```text
第一次安装：

./scripts/install_mlipx.sh
```

然后：

```text
UMA:
.venv/bin/mlipx

MACE:
.venv-mace/bin/mlipx

DPA:
.venv-dpa/bin/mlipx

GRACE:
.venv-grace/bin/mlipx
```

想知道本机 plan：

```bash
./scripts/install_mlipx.sh --dry-run
```

已经进入某个环境：

```bash
mlipx doctor
mlipx setup
```

不要再混合：

```text
uv run
裸 python -m pip
venv/bin/mlipx
workspace sync
```

四种使用哲学。

---

# 91. 最终建议的仓库清理结果

这轮完成后，我希望根目录大致只剩：

```text
.github/
archive/
mlipx/
scripts/
tests/

.gitignore
.pre-commit-config.yaml
.python-version
AGENTS.md
CITATION.cff
CODE_OF_CONDUCT.md
CONTRIBUTING.md
LICENSE.md
README.md
README_CN.md
pyproject.toml      # 可选，只放 tooling config
ruff.toml           # 或合并进 pyproject
codecov.yml         # 如果仍使用
```

删除：

```text
src/
packages/
configs/
docs/
.devcontainer/
postBuild
uv.lock
FairChem tests
FairChem workflows
FairChem release drafter configs
FairChem dependabot assignees
```

---

# 92. 修改顺序非常重要

不要让 Codex 一上来：

```bash
rm -rf src packages
```

正确顺序：

### Phase 1：installer 脱 workspace

先实现：

```text
UMA external fairchem-core
SourceProfile
clean
skip-doctor
explicit cuda validation
multi-GPU
plan tests
```

并确认：

```text
不再存在 runtime 对 src/packages 的依赖
```

---

### Phase 2：setup / doctor 收敛

完成：

```text
gpu_setup → generate_plan
doctor → compatibility matrix
删除 workspace wording
```

---

### Phase 3：删除 FairChem tree

再删除：

```text
src/
packages/
docs/
configs/
FairChem tests
FairChem CI
devcontainer
postBuild
root workspace lock
```

---

### Phase 4：tooling/metadata

修：

```text
ruff
pytest
CI
LICENSE
CITATION
CONTRIBUTING
pyproject metadata
```

---

### Phase 5：README 最后对齐

最后再改：

```text
README
README_CN
mlipx/README
```

因为 README 应描述**最终真实行为**，而不是让代码反过来追 README。

---

# 93. 最终验收清单

Codex 完成后至少要求：

```bash
python -m compileall mlipx/mlipx
```

通过。

在 Python 3.10 / 3.11 / 3.12：

```bash
pytest tests -q
```

通过。

```bash
ruff check mlipx/mlipx tests
ruff format --check mlipx/mlipx tests
```

通过。

package wheel/sdist 能 build。

下面 dry-run 全部正常：

```bash
./scripts/install_mlipx.sh --dry-run

./scripts/install_mlipx.sh \
    --engines uma \
    --dry-run

./scripts/install_mlipx.sh \
    --engines mace,dpa \
    --source china \
    --dry-run

./scripts/install_mlipx.sh \
    --device cpu \
    --dry-run

./scripts/install_mlipx.sh \
    --clean \
    --skip-doctor \
    --dry-run
```

另外 pure Python tests 模拟：

```text
Maxwell
Pascal
Volta
Turing
Ampere
Ada
Hopper
Blackwell
Kepler
CPU
mixed GPU
```

全部得到预期 plan。

---

# 94. Repository hygiene 验收

非 archive 区域：

```bash
grep -R "uv sync --frozen" .
```

runtime/install/README 中应不存在。

```bash
grep -R "packages/fairchem" .
grep -R "src/fairchem" .
```

除历史文档/归档明确保留引用外，应没有 production dependency。

根目录不再包含：

```text
src
packages
configs
FairChem docs
```

---

# 95. 一条特别重要的边界：这轮不要顺便重构科研功能

Codex 很容易看到项目以后开始：

```text
重写 calculator
重写 engine
优化 MD
增加 thermostats
修改 transport
改 analysis schema
增加模型训练
增加 LAMMPS
增加 benchmark framework
```

全部不要。

这轮 scope 应明确写死：

> 只修改安装系统、GPU compatibility plumbing、repository packaging/cleanup、README、CI/tooling、项目 metadata，以及这些修改直接需要的测试。
> 不改变现有 SP/OPT/MD 的科学算法、calculator 行为、trajectory semantics、analysis numerical methods 和用户已有计算输出格式。

---

## 最终我希望 Codex 达到的状态

现在的 mlipx 已经不是“FairChem 上面随手改了几处”的小 fork 了，它已经形成了：

```text
统一 CalculatorFactory
四 backend
CLI/TUI/API
VASP-style IO
MD
安装 compatibility matrix
doctor
GPU diagnostics
独立 analysis
```

所以这次清理真正重要的不是省掉那二十几 MB。

而是完成这个身份变化：

```text
之前：
FairChem monorepo
└── 里面长出了 mlipx

这轮之后：
mlipx
├── depends on fairchem-core
├── depends on MACE
├── depends on DeepMD
└── depends on GRACE
```

**FAIRChem 应该变成 mlipx 的一个 backend dependency，而不是 mlipx 仓库本身的一半。**
