# MedAgentCL v4 迁移风险登记

状态说明：`Open` 表示尚未在新环境/新模型上验证；`Controlled` 表示已有明确隔离措施，但仍需执行验收。

## 1. 风险表

| ID | 风险 | 概率/影响 | 当前证据 | 控制与验收 | 阶段 | 状态 |
|---|---|---|---|---|---|---|
| R01 | 新环境误用旧 editable ms-swift | 高/致命 | base 的 ms-swift 指向 `/root/MedAgentCL` | 新建干净 env；检查 `pip show`、`swift.__file__`、`sys.path` 必须指向 v4 | 1 | Open |
| R02 | 非登录 shell 调用 Python 2.7 | 高/高 | 默认 `python=/usr/bin/python` | 所有脚本使用绝对 `PYTHON_BIN`；启动时打印版本并要求 >=3.10 | 1 | Controlled |
| R03 | v4 tag/commit 漂移到 main/dev | 中/高 | 用户要求固定 v4.4.1 | 校验 exact tag 和完整 commit；记录到 environment/run config | 1 | Controlled |
| R04 | Qwen3-VL class/registry 依赖不完整 | 高/高 | 旧 transformers 4.53.1 无类；旧环境缺 qwen_vl_utils/decord | 锁 transformers 4.57.6、qwen-vl-utils 0.0.14、decord 0.6.0；registry/import smoke | 1 | Open |
| R05 | torch 2.5.1 与新 Transformers/Qwen3-VL 运行时不兼容 | 中/高 | v4 只要求 torch>=2.0，但推荐版本更高 | 先保留 torch；Phase 1 import、Phase 2 processor/model smoke；有明确错误再最小升级 torch 组 | 1-2 | Open |
| R06 | 一次升级 torch/CUDA/flash-attn/vLLM 导致无法定位问题 | 中/高 | 旧栈版本较老 | Phase 1 禁装 flash-attn/vLLM/DeepSpeed，保留 CUDA/torch | 1 | Controlled |
| R07 | LoRA 注入到 vision/merger/projector | 高/致命 | 旧脚本普遍 `all-linear`；Qwen3-VL 结构与 LLaVA 不同 | 首版只允许 `model.language_model.*.(q_proj|v_proj)`；打印并断言 vision count=0 | 3-4 | Open |
| R08 | 只靠名称排除导致漏/误匹配 | 中/高 | 用户明确禁止纯字符串猜测 | 结合 v4 model_arch prefix、模块类型和 `named_modules()`；单元测试真实名称样例 | 3-4 | Open |
| R09 | 插件成功 import 但 Med-PRISM 实际未启用 | 高/致命 | 旧实现依赖 monkey patch/环境变量，容易静默退化为普通 LoRA | 检查 tuner key、trainer class、wrapper count、trainable names、orth pair count；任一为 0 时失败 | 4 | Open |
| R10 | rank-1 scaling 随 expert 数量变化 | 中/致命 | 旧项目曾修复该问题 | 迁移旧数值测试；1/2/3 experts 下固定单 expert 输出，比较 scaling 与 logits | 4 | Open |
| R11 | orth loss 有数值但不可反向传播/无 pair | 中/致命 | 旧 Trainer 内部拼接，迁移接口变化 | 测试 `loss.requires_grad`、backward、目标参数非零 grad、pair/layer count | 4 | Open |
| R12 | v4 `compute_loss` 签名变化破坏自定义 Trainer | 高/高 | v4 包含 `num_items_in_batch`、template compute loss 等逻辑 | 子类透传完整签名；先调用 `super()`；覆盖 return_outputs true/false 测试 | 4 | Open |
| R13 | DDP unused parameter/static graph 问题 | 中/高 | 旧 fork 有专用 DDP hook | 先单卡；再 2 卡 5-10 step smoke，记录 unused 参数；只在证据需要时迁移 helper | 4-5 | Open |
| R14 | checkpoint 保存成功但 rank-1 wrapper/旧 expert 丢失 | 中/致命 | 自定义 wrapper 不一定被 PEFT 默认 state dict 覆盖 | 明确 `save_pretrained/from_pretrained`；比较 keys、manifest 和 reload 前后输出 | 4 | Open |
| R15 | shared 被重复加载或 private composition 错误 | 高/致命 | 旧评估存在多 adapter merge 路径 | adapter name 去重、manifest 驱动、打印组合；分别测试 shared2+private1/private2/private1+2 | 5 | Open |
| R16 | shared 没有跨 task 更新或 private 被覆盖 | 中/致命 | 语义依赖 stage 间路径 | 每 stage 比较 hash/norm；manifest 记录 source shared；private 目录不可复用 | 5 | Open |
| R17 | merge 改变模型并污染后续评估 | 中/高 | Octopus gradient 代码反复 `merge_and_unload` | 默认使用新进程或非破坏性 adapter composition；merge 只写临时导出目录 | 5-9 | Open |
| R18 | 旧 4-skill manifest 与最终 5-skill 顺序不一致 | 高/高 | 已观察旧顺序为 classification/VQA/caption/reportgen 等变体 | 新建唯一 5-skill task manifest；训练前校验 task ID、数据集、指标 | 6-8 | Open |
| R19 | 数据 schema 看似兼容但 image 路径/占位符失配 | 中/高 | v4 接受 `messages/images`，但 processor 不同 | 每文件统计空/不存在路径；检查 `<image>` 数量与 images 数量；Phase 2 单样本 encode | 2 | Open |
| R20 | Grounding bbox 坐标在 resize 后不一致 | 高/致命 | Grounding 尚未最终接入，Qwen3-VL 有专用格式 | 优先 v4 `objects` 标准格式；固定原图尺寸；做坐标往返与人工样例测试 | 8 | Open |
| R21 | Qwen3 thinking 输出污染短答案/指标 parser | 中/高 | Qwen3 有 thinking/non-thinking 模式 | Instruct/任务模板明确 non-thinking；按 skill 独立 generation config；保留 raw output | 2-8 | Open |
| R22 | 生成 collapse 被后处理掩盖 | 中/高 | 旧工程曾出现 prediction collapse 排查需求 | 同时保存 raw model response、抽取结果、unique/top-ratio 统计；raw 与 processed 分层诊断 | 2-8 | Open |
| R23 | 40GB A100 单卡 OOM | 中/高 | 8B VLM + 图像动态分辨率 + optimizer 有压力 | BF16、batch 1、gradient checkpointing、冻结 vision/aligner、限制像素/长度；先 5-step | 2-4 | Open |
| R24 | vLLM 0.7.3 不支持目标模型或 LoRA composition | 高/高 | 旧 vLLM 明显过旧 | Phase 1-6 不使用 vLLM；Transformers 验证正确性后单独升级/验证 | 7+ | Controlled |
| R25 | Octopus 旧实现依赖 fork 内部字段与路径猜测 | 高/高 | `swift/tuners/base.py`、gradient scripts 有硬编码与目录猜测 | 以算法规格重写；gradient/adapter manifest；先两 task 数值验证 | 9 | Open |
| R26 | Octopus、Sequential LoRA、Independent LoRA 语义混淆 | 中/高 | 多套旧脚本命名接近 | method 字段、输出根目录和 eval composition 分离；测试每种 adapter 数量 | 6-9 | Controlled |
| R27 | 指标实现因输出规范变化产生虚假遗忘 | 中/高 | VQA/分类/Concept/Caption 输出长度与 parser 不同 | 保存 raw/normalized/prediction；按 skill 固定 parser 和 generation config；人工 golden cases | 6-8 | Open |
| R28 | 旧项目被误写或旧结果被覆盖 | 低/致命 | 旧项目非 Git，回滚困难 | 所有新输出限制在 v4；旧数据绝对路径只读；Phase 1 前后记录旧根目录状态 | 全程 | Controlled |
| R29 | 模型 Hub revision 漂移导致不可复现 | 中/高 | model ID 默认跟随仓库最新 revision | 首次授权下载前解析并固定 immutable revision；写入所有 manifest | 2 | Open |
| R30 | 依赖 lock 只锁顶层，传递依赖后来漂移 | 中/高 | v4 requirements 多为范围约束 | bootstrap 安装后 `pip freeze --all`；保存 `pip check` 与包 hash/来源 | 1 | Open |

## 2. 阻断式验收门槛

### Phase 1

- 新环境与旧 editable 路径完全隔离。
- 固定 tag/commit、`pip check`、Qwen3/Qwen3-VL class 和 registry/template 可见性全部通过。
- 不满足时不得下载/加载模型。

### Phase 2

- 单条文本和单图 processor/inference 成功。
- JSONL 的 `messages/images` 被正确编码，输入包含像素/网格相关 tensor。
- raw output 与处理后 output 都保存；未出现 collapse。

### Phase 3

- 标准 Qwen3-VL LoRA 运行 5-10 step。
- 视觉与 aligner 注入数都为 0。
- adapter 保存、重载和固定样本输出一致。

### Phase 4

- 实际 Trainer class 为 Med-PRISM 自定义类。
- rank-1 wrapper 数量大于 0，target 全部位于 language backbone。
- scaling、orth backward、save/load 四个自动测试全部通过。

### Phase 5

- `shared/after_task_1`、`shared/after_task_2`、`private/Task1`、`private/Task2` 均存在且 manifest 完整。
- 三种评估 composition 的实际加载列表准确且无重复。

### Phase 6

- 两 task lower-triangle matrix 可在新进程重现。
- current/shared/private checkpoint 语义与指标行列定义一致。
- Phase 6 通过前不得扩展到完整 skill CL。

## 3. 当前最高优先级风险

1. **R09 插件静默失效**：这是最危险的“能训练但方法没有运行”情况，必须以 wrapper、Trainer class 和 orth gradient 三重证据验收。
2. **R07/R08 视觉层误注入**：Qwen3-VL 结构变化使旧 `all-linear` 策略不适合 Med-PRISM 首版，必须用官方 model architecture 做正向选择。
3. **R14-R17 checkpoint composition**：shared/private 与 Octopus 的科学结论都依赖精确组合，必须从目录猜测迁为 manifest 驱动。
4. **R18 任务顺序不一致**：旧 4-skill 数据可以复用，但新 5-skill CL 的唯一 task manifest 必须重建。

## 4. Phase 0 未解决项

- 未在新环境实际安装依赖，因此 bootstrap pin 仍需 Phase 1 的 resolver、`pip check` 和 imports 证实。
- 未下载/加载 Qwen3 或 Qwen3-VL，因此 target module 实际名称、显存和 processor 输出留待 Phase 2/3。
- 未修改或运行 Med-PRISM，Trainer subclass 注册方案需 Phase 4 自动测试确认。
- Grounding 数据集尚未最终确定，bbox 规范和指标实现留待 Phase 8。
- vLLM、flash-attn、DeepSpeed 均有意后置，不构成 Phase 1 阻塞项。

