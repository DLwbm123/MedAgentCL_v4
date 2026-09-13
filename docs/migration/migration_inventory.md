# MedAgentCL v4 Phase 0 迁移清单

审计日期：2026-07-13  
旧项目：`/root/MedAgentCL`（只读）  
目标项目：`/root/MedAgentCL_v4`（Phase 0 未创建）  
目标框架：ms-swift `v4.4.1`，commit `98a09c18cdf95ff07051324b9b8cc90f5184b24b`  
目标模型：`Qwen/Qwen3-8B` 与 `Qwen/Qwen3-VL-8B-Instruct`

## 1. Phase 0 边界与结论

本阶段只读检查了旧服务器项目，并以固定 tag 的官方源码核对新版 API。没有修改 `/root/MedAgentCL`，没有创建 `/root/MedAgentCL_v4` 或 Conda 环境，没有下载模型，也没有启动训练。

主要结论：

1. ms-swift v4.4.1 已原生注册 `Qwen/Qwen3-8B` 和 `Qwen/Qwen3-VL-8B-Instruct`；Qwen3-VL 对应 `model_type=qwen3_vl`、`template=qwen3_vl`。
2. 旧工程不能整体复制 `swift/`。旧 fork 中真正需要保留的算法语义是 Med-PRISM 的 rank-1、正交损失、shared/private、checkpoint composition，以及 Octopus 的 task-wise adapter/gradient 约束。
3. 旧数据主体 `messages` + `images` 与 v4 标准格式兼容，可继续按原路径只读引用；Grounding 应在后续阶段补充/规范 `objects` 与 bbox 格式。
4. 最主要的重构点是自定义 tuner 注册、Trainer loss 注入、Qwen3-VL language-only target filter，以及 adapter 保存/加载语义。

## 2. 已核验基线

### 2.1 服务器与旧环境

| 项目 | 只读检查结果 |
|---|---|
| 旧项目 | `/root/MedAgentCL`，不是 Git repository |
| Python | `/root/anaconda3/bin/python`，3.12.7 |
| 非登录 shell 默认 Python | `/usr/bin/python`，2.7.17，禁止依赖 |
| GPU | 2 x NVIDIA A100-PCIE-40GB |
| torch | 2.5.1+cu124 |
| CUDA runtime | 12.4 |
| transformers/tokenizers | 4.53.1 / 0.21.4 |
| accelerate/peft/datasets | 1.7.0 / 0.15.2 / 3.5.0 |
| safetensors/vLLM | 0.5.3 / 0.7.3 |
| ms-swift | 3.7.0.dev0，editable 指向 `/root/MedAgentCL` |
| 缺失项 | 新 Qwen3-VL Transformers 类、`qwen_vl_utils`、`trl`、`decord`、`flash_attn`、`deepspeed` |

### 2.2 v4.4.1 官方源码事实

- 固定 tag/commit：`v4.4.1` / `98a09c18cdf95ff07051324b9b8cc90f5184b24b`。
- 官方 framework 范围：`transformers>=4.33,<5.13.0`、`datasets>=3.0,<4.8.5`、`peft>=0.11,<0.20`、`trl>=0.15,<1.0`。
- Qwen3-VL registry 额外要求：`transformers>=4.57`、`qwen_vl_utils>=0.0.14`、`decord`。
- Qwen3-VL 架构声明：language model 为 `model.language_model` 和 `lm_head`；vision tower 为 `model.visual`；aligner 为 `model.visual.merger` 与 `model.visual.deepstack_merger_list`。
- v4 标准多模态数据接受 `messages`、`images`、`videos`、`audios`、`objects` 等字段；`messages` 必需。

## 3. 旧项目差异审计

旧 `swift/` 与旧备份 `swift.bak.20260520_045231/` 的源码差异只集中在以下文件：

| 文件 | 旧改动语义 | v4 处理 |
|---|---|---|
| `swift/trainers/trainers.py` | 注入 orth loss、shared drift、日志、DDP static graph | 禁止复制；重写为项目侧自定义 Trainer 子类 |
| `swift/llm/model/register.py` | Mistral `ReasoningEffort` 兼容 | 废弃；与目标 Qwen3/Qwen3-VL 无关 |
| `swift/llm/train/tuner.py` | `saved_folder=None` 兼容 | 废弃；v4 pipeline 不再沿用此返回契约 |
| `swift/trainers/__init__.py` | `EvaluationStrategy` 移除兼容 | 废弃；使用 v4 原生实现 |
| `swift/llm/argument/base_args/template_args.py` | 旧参数兼容 | 不复制 |
| `swift/llm/export/cached_dataset.py` | 旧缓存导出补丁 | 不复制，若后续确有需求单独重写 |
| `swift/plugin/agent_template/mistral.py` | Mistral agent 模板 | 废弃 |

## 4. 迁移文件分类

### 4.1 可直接复用算法主体

这些模块原则上保持数学语义，只允许做 import、类型与模型命名适配：

- `med_prism/adapters/rank1_lora.py`
- `med_prism/adapters/routed_rank1_lora.py`
- `med_prism/projection/orth_losses.py`
- `med_prism/projection/peft_orth.py`
- `med_prism/projection/shared_private.py`
- `med_prism/experts/expert_bank.py`
- `med_prism/continual/eval_matrix.py`
- `med_prism/routing/`
- `med_prism/utils/loss_log.py`
- `med_prism/utils/tensor_checks.py`

必须保留的数值语义：

- 每个 task 的 private expert 为 rank 1。
- 单 expert 的 LoRA scaling 只由该 expert 的 `alpha/rank` 决定，不随累计 expert 数量变化。
- orth loss 的 squared/RMS 分支和 `eps=1e-8` 行为不变。
- shared 跨 task 连续更新；private 每 task 独立保存。

### 4.2 需要修改后迁移

- `med_prism/config.py`：增加 Qwen3/Qwen3-VL、模型 revision、target filter、运行 manifest 字段。
- `med_prism/adapters/injection.py`：改用 Qwen3-VL 实际 `named_modules()`/model architecture，拒绝视觉与 aligner 注入。
- `med_prism/utils/ddp.py`：按 v4 Trainer/Accelerate 生命周期复核，不能直接假定旧 hook 时机。
- `med_prism/scripts/medimeta/`
- `med_prism/scripts/omnimedvqa/`
- `med_prism/scripts/MedTrinity/`
- `med_prism/scripts/skill_incremental/`
- 顶层 `scripts/` 中的数据构建与评估脚本。
- `evaluate.py`、`m4c_evaluator.py` 及任务指标代码：保留指标核心，替换 LLaVA/vLLM/merge 加载路径。
- 现有 JSONL sanitizer：保留 `messages/images`，增加 v4 schema 校验、任务名和绝对图像路径检查。

脚本统一要求：

```bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/anaconda3/envs/medagentcl_v4/bin/python}"
```

### 4.3 必须重新实现

- `med_prism/swift_plugins/med_prism_rank1_plugin.py`
  - 注册到 v4 `tuners_map`。
  - 使用 v4 `SftArguments` 与 `swift.pipelines.train.tuner` helper。
  - 实现 language-only target module 统计与强制断言。
  - 明确实现 `prepare_model`、`save_pretrained`、`from_pretrained`。
- `med_prism/swift_plugins/med_prism_shared_private_plugin.py`
  - 按 v4 adapter API 重写 shared/private prepare、save、load 与 manifest。
- Med-PRISM Trainer 集成
  - 新建项目侧 `MedPrismSeq2SeqTrainer`，继承 v4 `swift.trainers.Seq2SeqTrainer`。
  - 在 `compute_loss` 返回的 task loss 上添加 orth/shared loss。
  - 通过外部插件在 v4 参数初始化后、TrainerFactory 选类前登记 trainer class。
  - 不再修改 `swift/trainers/*.py`。
- Qwen3-VL target filter
  - 第一版只接受 language backbone 下的 `q_proj`、`v_proj`。
  - 启动时打印匹配总数、语言层数、视觉层数和前若干完整名称。
  - `vision_target_count != 0` 时立即失败。
- shared/private checkpoint composer
  - 显式记录 shared/private 名称、来源 task、加载次序与是否 merge。
  - 防止 shared 重复加载、private 重复 merge 或仅加载最后一个 private。
- Octopus v4 实现
  - 保留 per-task independent LoRA、旧梯度子空间约束和 cumulative evaluation。
  - 不复制旧 `swift/tuners/base.py` 中按日期目录猜 checkpoint 的逻辑。
  - checkpoint 和 gradient 文件均由 manifest 明确索引。

### 4.4 仅作参考、不得复制

- 整个旧 `/root/MedAgentCL/swift/`。
- `swift.bak.*`、`__pycache__`、editable 安装元数据。
- 旧 LLaVA 模型注册、processor/template 私有补丁。
- 旧 `output/`、checkpoint、日志、模型权重、缓存。
- `data/` 中的大规模图像与已有 JSONL；新项目只引用旧路径，不复制。
- Octopus 脚本中的 `rm -rf saved_gradients_*` 和依赖当前工作目录的相对路径。
- `generate_gradients_*.py` 中硬编码 `llava-hf/llava-1.5-7b-hf`、`all-linear`、按目录名猜 checkpoint 的部分。

## 5. 方法/基线语义边界

| 方法 | 训练语义 | 评估语义 |
|---|---|---|
| Sequential LoRA | Task t 从 Task t-1 adapter 继续更新同一 LoRA | current adapter only |
| Independent LoRA | 每 task 独立从 base 训练 | 对应 task adapter |
| Octopus | 每 task 独立 adapter，旧 adapter/gradient 子空间用于约束 | cumulative adapter composition |
| Med-PRISM rank-1 | 每 task 新增 private rank-1 experts，加新旧正交约束 | 按 manifest 组合 |
| Med-PRISM shared/private | shared 连续更新，private 每 task 保存 | cumulative 或 skill-aware |

这四类实现和 output 命名必须分开，不能把 Sequential LoRA 或普通 LoRA 标记为 Med-PRISM/Octopus。

## 6. 数据与任务顺序

目标 5-skill 顺序固定为：

1. OmniMedVQA / Medical VQA / Accuracy x 100
2. MedIMeta / Diagnosis Classification / Accuracy x 100
3. MedTrinity Concept / Core Concept Micro-F1 x 100
4. Grounding / Acc@IoU>=0.5 x 100
5. MedTrinity Caption / ROUGE-L x 100

已发现旧 `MedSkill_CL_4Skill` manifest 的顺序/任务集合与上述最终顺序并不完全相同，因此只能复用数据文件，不能直接把旧 manifest 当作新 5-skill 运行清单。

数据路径策略：

- `/root/MedAgentCL/data/MedSkill_CL_4Skill`：只读引用。
- `/root/MedAgentCL/data/MedTrinity_ConceptCaption_CL`：存在时只读引用。
- 新运行产物统一写 `/root/MedAgentCL_v4/output`。
- Phase 2 前建立 `data_manifest.json`，记录每个 task 的 train/test JSONL、样本数、图像根目录和 schema hash。
- CL 指标矩阵按要求直接在 test set 评估，不依赖 val set。

## 7. 精确环境方案

### 7.1 原则

- 新建干净环境 `medagentcl_v4`，不 clone 旧 base 环境，防止旧 editable ms-swift 污染。
- 第一轮保留 torch/CUDA 组合，先只升级 Qwen3-VL 必需依赖。
- Phase 1 不安装 `flash-attn`、vLLM、DeepSpeed；训练 smoke test 使用 Transformers + SDPA/eager。
- 所有命令使用环境绝对 Python 路径。

### 7.2 Phase 1 顶层锁定版本

| 依赖 | 固定版本/来源 | 理由 |
|---|---|---|
| Python | 3.12.7 | 与服务器现状和 v4 推荐一致 |
| torch | 2.5.1+cu124 | 先保持现状；满足 v4 `torch>=2.0` |
| torchvision | 0.20.1+cu124 | 与 torch 2.5.1 配套 |
| torchaudio | 2.5.1+cu124 | 与 torch 2.5.1 配套 |
| ms-swift | v4.4.1 / `98a09c18...` editable | 固定 tag 和 commit |
| transformers | 4.57.6 | 满足 Qwen3-VL `>=4.57`，也是 v4 推荐 4.x |
| tokenizers | 0.22.1 | 与 transformers 4.57.x 配套 |
| qwen-vl-utils | 0.0.14 | registry 最低要求，先锁最低可用版本 |
| decord | 0.6.0 | registry 明确要求；Linux wheel 为 `py3-none` |
| peft | 0.15.2 | 保留现有版本，位于 v4 支持范围 |
| accelerate | 1.7.0 | 保留现有版本，减少变量 |
| datasets | 3.5.0 | 保留现有版本，位于 v4 支持范围 |
| safetensors | 0.5.3 | 保留现有版本 |
| trl | 0.29.1 | v4.4.1 README 推荐，满足 `>=0.15,<1.0` |
| huggingface-hub | 0.36.0 | 与 transformers 4.57.x 兼容的 0.x 固定版本 |

说明：上述为 Phase 1 bootstrap lock。安装后必须运行 `pip check` 和 import smoke；成功后用 `pip freeze --all` 生成包含所有传递依赖的 `requirements-lock.txt`，不得用手写表代替最终 lock。如果 resolver 证明某一保留版本不兼容，只改动该最小依赖组并记录证据，不顺带升级 torch/CUDA。

### 7.3 Phase 1 应生成的环境文件

- `environment.yml`：Conda Python 与 CUDA/PyTorch channel/source。
- `requirements-bootstrap.txt`：上表的精确顶层 pin。
- `requirements-lock.txt`：安装后 `pip freeze --all` 的完整锁。
- `environment_report.json`：Python、包版本、CUDA、GPU、ms-swift commit、pip check 结果。

## 8. `/root/MedAgentCL_v4` 初始化方案

Phase 1 获得确认后才执行：

1. 检查目标目录不存在或为空、磁盘空间充足。
2. 从官方仓库获取 tag `v4.4.1` 到 `/root/MedAgentCL_v4`。
3. 校验 `HEAD == 98a09c18cdf95ff07051324b9b8cc90f5184b24b`，tag 必须精确匹配。
4. 创建工作分支 `medagentcl-v4`，保留 `origin`/`upstream` 来源信息。
5. 扩充 `.gitignore`：`data/`、`output/`、`checkpoints/`、`models/`、`logs/`、cache、模型权重和环境报告中的敏感路径。
6. 添加 `UPSTREAM_MS_SWIFT.md`、环境 bootstrap 文件和迁移文档；提交 clean baseline。
7. 新建 `medagentcl_v4` 环境，使用目标环境绝对 Python 安装该目录 editable。
8. 只运行 `swift --help`、import、registry/template 可见性检查，不加载模型。
9. 生成完整 lock/environment report，并提交 Phase 1 环境配置；环境本体不纳入 Git。

建议提交序列：

1. `chore: pin clean ms-swift v4.4.1 baseline`
2. `chore: add reproducible medagentcl_v4 environment`
3. `feat: migrate project data and evaluation utilities`
4. `test: add qwen3 and qwen3-vl smoke coverage`
5. `feat: migrate med-prism rank1 tuner`
6. `feat: migrate med-prism shared-private adapters`
7. `test: validate two-task continual learning matrix`
8. `feat: migrate five-skill continual learning pipeline`
9. `feat: reimplement octopus baseline on swift v4`

## 9. Phase 1 验收门槛

Phase 1 只在全部满足时通过：

- `which python` 和脚本 `PYTHON_BIN` 都指向 `/root/anaconda3/envs/medagentcl_v4/bin/python`。
- `pip show ms-swift` 的 editable location 为 `/root/MedAgentCL_v4`，不是旧项目。
- `git describe --tags --exact-match` 为 `v4.4.1`，commit 完全匹配。
- `pip check` 无错误。
- 可 import `Qwen3ForCausalLM`、`Qwen3VLForConditionalGeneration`、`AutoProcessor`。
- ms-swift model registry 可见 `qwen3`、`qwen3_vl`，template registry 可见 `qwen3`、`qwen3_vl`。
- `swift --help` 与 `swift sft --help` 可运行。
- 尚未加载/下载模型，尚未迁移 Med-PRISM，尚未启动训练。

## 10. 官方依据

- ms-swift v4.4.1：<https://github.com/modelscope/ms-swift/tree/v4.4.1>
- Qwen3-VL-8B-Instruct model card：<https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct>
- decord 0.6.0 wheel：<https://pypi.org/project/decord/>

