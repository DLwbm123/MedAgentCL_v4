# ms-swift 3.x 到 v4.4.1 API 映射

固定新版源码：ms-swift `v4.4.1`，commit `98a09c18cdf95ff07051324b9b8cc90f5184b24b`。

## 1. 自定义 tuner/plugin

| 旧工程 | v4.4.1 | 迁移动作 |
|---|---|---|
| `from swift.plugin import PeftTuner, extra_tuners` | `from swift.tuner_plugin import PeftTuner, tuners_map` | 修改 import |
| `extra_tuners['med_prism_rank1'] = ...` | `tuners_map['med_prism_rank1'] = ...` | 修改注册表 |
| `extra_tuners['med_prism_shared_private'] = ...` | `tuners_map['med_prism_shared_private'] = ...` | 修改注册表 |
| `from swift.llm import TrainArguments` | `from swift.arguments import SftArguments` | 修改类型与字段读取 |
| `swift.llm.train.tuner.get_target_modules` | `swift.pipelines.train.tuner.get_target_modules` | 修改 import；再加项目侧过滤/断言 |
| `swift.llm.train.tuner.get_modules_to_save` | `swift.pipelines.train.tuner.get_modules_to_save` | 修改 import |
| `PeftTuner.prepare_model(...)` | 自定义 `Tuner.prepare_model(args, model)` | 按 v4 接口重写 |
| 隐式依赖 PEFT 默认保存 | `Tuner.save_pretrained(...)` | 明确保存 adapter 与项目 manifest |
| 旧插件缺少可靠 load contract | `Tuner.from_pretrained(model, model_id, **kwargs)` | rank-1/shared-private 分别实现并测试 |
| `--custom_register_path` | `--external_plugins` | v4 保留前者兼容，但新脚本只用后者 |

v4 的 `TunerMixin.prepare_model` 会在 `tuner_type in tuners_map` 时调用自定义 `prepare_model/from_pretrained`，Trainer 保存时也会调用对应 `save_pretrained`。因此不需要修改 v4 核心 tuner pipeline。

## 2. CLI 与参数

| 旧参数/行为 | v4.4.1 | 迁移规则 |
|---|---|---|
| `--train_type lora` | `--tuner_type lora` | 全部替换 |
| `--train_type med_prism_rank1` | `--tuner_type med_prism_rank1` | 插件注册后使用 |
| `--train_type med_prism_shared_private` | `--tuner_type med_prism_shared_private` | Phase 5 才启用 |
| `--target_modules all-linear` | 仍支持 | Med-PRISM 首版禁止使用；标准 LoRA smoke 可使用并核验范围 |
| `--target_modules q_proj v_proj` | 仍支持 | Med-PRISM 首版默认值 |
| `--freeze_vit true` | 仍支持，默认 true | 脚本仍显式传递并记录 |
| `--freeze_aligner true` | 仍支持，默认 true | 脚本仍显式传递并记录 |
| `--freeze_llm false` | v4 支持，默认 false | 脚本显式传递 |
| `--template ...` | 仍支持，默认按模型自动选择 | Qwen3-VL smoke 显式 `qwen3_vl` 便于审计 |
| `--use_hf true` | 仍支持 | Hugging Face 模型 ID 时保留 |
| `--external_plugins file.py` | 仍支持 | 使用绝对路径，日志检查导入成功 |
| 隐式系统 `python` | 不允许 | 使用 `/root/anaconda3/envs/medagentcl_v4/bin/python` |

`dataset_format` 不是 v4 标准 JSONL 的必要参数。项目本地 JSONL 直接通过 `--dataset /abs/path/file.jsonl` 使用，并在训练前独立运行 schema validator。

## 3. 模型、processor 与 template

| LLaVA-1.5 旧实现 | Qwen3/Qwen3-VL v4.4.1 |
|---|---|
| `llava-hf/llava-1.5-7b-hf` | `Qwen/Qwen3-VL-8B-Instruct` |
| LLaVA model/processor class | `Qwen3VLForConditionalGeneration` + `AutoProcessor` |
| 旧 LLaVA template | `qwen3_vl` |
| 视觉/语言模块命名依赖 `language_model` 等猜测 | 使用 v4 `model_arch`：`model.language_model`、`model.visual`、merger list |
| `AutoModelForVision2Seq` 泛型评估加载 | 优先使用 ms-swift TransformersEngine/registry；必要时显式 Qwen3VL class |
| 旧 `<image>` + `images` JSONL | v4 仍接受；模板负责视觉 token 展开 |
| 硬编码 4096 长度 | 按任务显式设定并记录，先用保守 smoke 长度 |

Qwen3-VL 官方 registry 在 v4.4.1 中声明：

```text
model_type: qwen3_vl
template: qwen3_vl
model_arch: qwen3_vl
requires: transformers>=4.57, qwen_vl_utils>=0.0.14, decord
```

## 4. Target module 选择

v4 内置 `get_target_modules` 在多模态模型且 `all-linear` 时会调用 `get_multimodal_target_regex`，结合 `freeze_llm/freeze_vit/freeze_aligner` 和 model architecture 形成正则。Qwen3-VL 的官方 architecture 为：

```text
language_model = ['model.language_model', 'lm_head']
vision_tower = ['model.visual']
aligner = ['model.visual.merger', 'model.visual.deepstack_merger_list']
```

Med-PRISM 首版不依赖 `all-linear` 的宽匹配，采用双重门控：

1. full module name 必须位于 `model.language_model` 子树。
2. leaf name 必须为 `q_proj` 或 `v_proj`。
3. 再拒绝包含 `visual`、`vision`、`merger`、`projector`、`aligner`、`patch_embed`、`image` 的名称。
4. 打印所有统计并断言视觉匹配数为 0。

稳定后扩展目标的候选集合为 `q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj`，每次扩展都必须重新跑 filter 和数值回归测试。

## 5. Trainer 与损失注入

| 旧实现 | v4.4.1 迁移 |
|---|---|
| 修改 `swift/trainers/trainers.py` 的 training step | 不复制 |
| 插件 monkey-patch 多个 Trainer 的 `compute_loss` | 替换为显式 `MedPrismSeq2SeqTrainer` 子类 |
| 从环境变量读取大部分 Med-PRISM 配置 | 迁为项目 dataclass/CLI 字段；环境变量仅保留兼容层 |
| 在 training step 后临时拼 loss | 在 `super().compute_loss(...)` 得到 task loss 后添加可微 orth/shared loss |
| 依赖旧日志分支 | 使用 v4 `self.log`/`custom_metrics`，同时写 `run_config.json` |
| 修改旧 DDP static graph | 在 v4/Accelerate 上先复现必要性，再以 callback/helper 最小接入 |

外部插件加载发生在 `BaseArguments.__post_init__`，Trainer 创建发生在 `SftPipeline` 后续阶段。可采用：

```python
from swift.trainers import Seq2SeqTrainer
from swift.trainers.trainer_factory import TrainerFactory

class MedPrismSeq2SeqTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 调用 super，再叠加 Med-PRISM 可微损失。
        ...

TrainerFactory.TRAINER_MAPPING['causal_lm'] = (
    'med_prism.swift_plugins.med_prism_trainer.MedPrismSeq2SeqTrainer'
)
```

Phase 4 必须验证实际 trainer class、wrapper 数量和非零 orth 梯度，防止插件导入成功但算法未启用。

## 6. 保存、恢复与 adapter composition

| 旧行为 | v4 目标 |
|---|---|
| 按日期目录和 `checkpoint-*` 最大值猜路径 | 每个 stage 写 `adapter_manifest.json`，只按 manifest 加载 |
| `PeftModel.from_pretrained` 后反复 merge | 默认非破坏性 composition；仅导出时 merge |
| shared/private 路径由 shell 隐式拼接 | manifest 记录 adapter role、task、source、base revision、顺序 |
| 可能重复加载 shared | composer 维护已加载 adapter name 集合并拒绝重复 |
| cumulative 只加载最后 private 的风险 | manifest 列出 `private_1...private_t`，逐项核验 |
| 保存前后没有统一数值比较 | 固定输入、eval mode、同 dtype，比较 reload 前后 logits/output |

建议 manifest 最小字段：

```json
{
  "schema_version": 1,
  "backbone": "Qwen/Qwen3-VL-8B-Instruct",
  "backbone_revision": "<immutable revision>",
  "task_id": 2,
  "task_name": "diagnosis_classification",
  "method": "med_prism_shared_private",
  "shared": "/.../shared/after_task_2",
  "private": ["/.../private/Task1", "/.../private/Task2"],
  "source_checkpoint": "/.../shared/after_task_1",
  "merge_state": "unmerged"
}
```

## 7. 数据与评估

| 旧流程 | v4 迁移 |
|---|---|
| JSONL `messages/images` | 可复用，训练前做 schema/path 校验 |
| 旧 LLaVA processor | Qwen3-VL template/processor |
| VQA/分类短文本 parser | 保留指标逻辑，增加 Qwen3 输出规范化 |
| Concept 训练 target | 使用 `normalized_concepts` |
| Concept 核心词表 | 只用于 Core Concept Micro-F1 映射，不改训练 target |
| Caption | 保留 ROUGE-L，单独配置长输出生成参数 |
| Grounding 自定义 bbox 文本 | 优先转换为 v4 `objects={ref,bbox}` 标准格式 |
| val/test 混用 | CL matrix 直接在 test set；val 不参与最终矩阵 |
| vLLM 作为默认评估后端 | Phase 2-6 先用 Transformers；vLLM 后置 |

## 8. Octopus 映射

旧 Octopus 包含 Stage 1 独立 LoRA、保存历史 adapter 梯度、Stage 2 子空间约束与 cumulative evaluation。需要迁移的是语义，不是旧 fork：

- `generate_gradients_*.py` 中的数据编码改用 v4 dataset/template pipeline。
- adapter key 解析改为明确的 PEFT adapter state API，不通过字符串裁剪猜 base key。
- 历史 gradient 文件使用 safetensors + manifest，包含来源 adapter hash、task 和 target modules。
- Stage 2 约束通过独立 tuner/trainer 实现，不修改 v4 `swift/tuners/base.py`。
- 删除所有 `rm -rf saved_gradients_*` 工作流，改为 run-scoped 输出目录。
- 累计评估必须记录实际加载的 adapter 列表。

## 9. 废弃映射

以下旧补丁在 v4 项目中没有对应迁移动作：

- `EvaluationStrategy` fallback。
- Mistral `ReasoningEffort` patch 和 agent template。
- `saved_folder` 的旧 tuner 返回值。
- 旧 `swift/llm/*` package layout。
- LLaVA 专属 model/template patch。
- 旧 vLLM 0.7.3 路径和 merge helper。
- 整个旧 `swift/` 目录覆盖方案。

